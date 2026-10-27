# BENCH_RESULTS_v011 — flash-tile GQA attention (prefill + split-KV decode)

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, 12 GB RAM, CUDA 13.0
Model: Sage-A1.5-Q4_K_M (GGUF 21.7 GB), expert pool 18.1 GB
Engine: `sager_engine` v0.11, pool reused warm, ctx 8192, vram_mb 9984, threads 2
Protocol: 512/2048 random-token prompts (seed 2026), tg128 at temp 1.0 / top_k 20 /
top_p 0.95, median of 3, per-round session reset, equal-warmth expert profile.
Raw JSON + logs ship with the release tarball.

## Headline (T4, protocol-matched, same-VM binary A/B)

| metric | v0.10 binary (this VM) | v0.11 (this VM) | speedup | v0.8 published |
|---|---|---|---|---|
| prefill pp512 | 727.4 tok/s | **823.8 tok/s** | 1.13x | 245.6 tok/s (3.35x) |
| prefill pp2048 | 518.3 tok/s | **731.9 tok/s** | **1.41x** | 218.8 tok/s (**3.35x**) |
| decode tg128 | 26.00 tok/s | **33.30 tok/s** | **1.28x** | ~27.3 tok/s (1.22x) |
| parity gate (greedy 64 tok) | | IDENTICAL | | |

Cumulative vs the v0.8 published baseline: **pp2048 3.35x, pp512 3.35x,
tg128 +22%**.

## Kernel-level attribution (nsys, 2 x 2048-token prefills, warm)

| kernel | v0.10 | v0.11 | delta |
|---|---|---|---|
| prefill attention (80 launches) | 2836.3 ms, 39.1% GPU | **1055.7 ms, 20.3%** | **2.69x** |
| decode attention per step | ~ms-scale, 16 warps total | **16 us (partial+merge)** | ~1000x parallelism |

Same-VM interleaved binary A/B (2 rounds each, alternating engine processes,
prof driver): reset+2048 prefill 516.8 -> 724.6 tok/s (+40%), warm 2048
423.2 -> 551.9 (+30%). tg128 in that ad-hoc driver is not protocol-comparable;
the bench-protocol tg128 numbers above are authoritative.

## What v0.11 changes

1. **Prefill flash-tile attention** (`attn_prefill_v2<CH>`, SAGER_ATTN_NEW=1
   default): one block = (kv group, query token) with 8 warps = the group's 8
   q heads. The block stages 16 K rows + 16 V rows in shared memory and
   dequantizes Q8 KV exactly once per block (the v0.10 kernel re-dequantized
   the same K row once per (token, head) warp - 8x redundant per token, and
   paid the 6-bit scale unpack twice per element). Each lane keeps its
   head_dim/32 q dims in registers; scores and V accumulate from shared
   memory; expf runs once per (row, warp) instead of once per (row, chunk).
   Instruction work per (warp, context row) drops ~6x.
2. **Split-KV flash decode** (`attn_decode_partial` + `attn_decode_merge`):
   the v0.10 decode kernel launched n_head=16 warps TOTAL - 16 blocks on 40
   SMs, each serially scanning the whole context with element-wise Q8
   dequant; per-token attention latency dominated. v0.11 launches a fixed
   (n_head, 64) grid (graph-replay safe: blocks self-range on device via
   d_params->pos, stale splits are never read), each active split covers 256
   rows with 4 warp-interleaved independent online-softmax states merged in
   block, then a 32-thread merge kernel combines per-split (m, l, acc)
   partials. Decode attention collapses to microseconds.
3. Both kernels are templated on CH = head_dim/32; the real Sage-A1.5
   attention dims are **h=16, kvh=2, head_dim=256** (the common.hpp defaults
   say 32/4/128 - they do NOT match the gguf). A one-shot stderr diagnostic
   prints the dispatch decision at first use.
4. `SAGER_ATTN_NEW=0` restores the v0.10 kernels (numerics of the fallback
   path differ from the released v0.10 binary only where that binary's
   v_acc[4] overflowed for head_dim=256; see numerics below).
5. Zero new VRAM outside two 64 KB static device arrays (~1 MB total decode
   partial buffer); prefill shared memory 18 KB/block, 3 blocks/SM.

## Numerics

NOT bit-identical to v0.10 by design, for two reasons:

1. The new prefill stages dequantized K/V as fp16 (one rounding of
   `d_q8 * q`), where v0.10 dequantized element-wise to fp32 at use. Scores
   and outputs carry ~2^-11 relative rounding - the same class of change the
   TC GEMM paths introduced in v0.9/v0.10.
2. **Fixed a latent v0.10 bug**: the v0.10 kernel declared `float v_acc[4]`
   but looped `c < head_dim/32` - for head_dim=256 (this model) it wrote
   v_acc[4..7] out of bounds. The overflow was deterministic and benign in
   practice (stable outputs across v0.8-v0.10, parity gates compare
   same-binary arms), but dims 128..255 of every attention output carried
   undefined values after any online-softmax rescale. v0.11 sizes the
   accumulator correctly (template CH). After the fix the model's greedy
   streams change - this is a *quality fix*, the pre-fix streams were
   computed on partially-corrupted attention output.

The parity gate pins `SAGER_PF_TC=0 MOE_TC=0` on both arms and verifies the
rest of the tree under the NEW attention. Greedy 64-token streams: IDENTICAL
(64 vs 64). ctest 3/3.

## Remaining profile (v0.11, nsys kern-sum, 2 x 2048 prefill)

attention 20.3% | gdn_scan 10.9% | dequant_experts 9.2% | gemv_mtok trio
18.2% (dp4a fallback rows) | TC GEMMs (turing_fp16_s1688gemm) 11.7% |
everything else < 2% each. Next levers, in expected-value order: GDN scan
chunk kernel (30 layers), the remaining dp4a gemv rows (cpu_rows=256 fallback
path), MoE dequant bandwidth.
