# BENCH_RESULTS_v08 — prefill MMQ GEMM + decode fusions + GDN ffn-norm fix

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, 12 GB RAM, CUDA 13.0
Model: Sage-A1.5-Q4_K_M (GGUF 21.7 GB), expert pool 18.1 GB (HF: tchbcb/sager-t4-cuda-artifacts)
Engine: `sager_engine` v0.8 (branch v0.8-opt), pool reused warm, ctx 8192, vram_mb 9984, threads 2
Protocol: 512/2048 random-token prompts (seed 2026), tg128 at temp 1.0 / top_k 20 / top_p 0.95,
median of 3, per-round session reset. Raw JSON + logs ship with the release tarball.

## Headline (T4)

| metric | v0.7 published | v0.7-exact arm | v0.8 arm | delta (same box) |
|---|---|---|---|---|
| prefill pp512 | 248.1 tok/s | 244.9 tok/s | **245.6 tok/s** | +0.3% |
| prefill pp2048 | 218.5 tok/s | 216.6 tok/s | **218.8 tok/s** | +1.0% |
| decode tg128 (chunk 512) | 30.65 tok/s | 28.77 tok/s | **29.36 tok/s** | **+2.1%** |
| parity gate (greedy 64 tok) | — | IDENTICAL | | |
| ctest | 3/3 | 3/3 | | |

Both arms on the same Tesla T4, equal-warmth expert profile (per-arm copies),
mtp on, medians of 3 with per-round session resets. The published v0.7
numbers come from the previous VM (its tg128 ran ~6% faster than this box's
v0.7-exact arm - VM variance, same binary family).

## Measured feature attribution (equal-warmth isolation runs)

| arm | pp512 | pp2048 | tg128 | note |
|---|---|---|---|---|
| v07-exact | 218.2-244.9 | 193.1-216.6 | 27.55-27.75 | every v0.8 switch off |
| v08-all (defaults) | 225.7-245.6 | 195.0-218.8 | 27.79-29.36 | shipped default |
| smx-only | — | — | +3% vs v07 | head kernel win |
| fuse-only | ~v07 | ~v07 | -1..-4% | opt-in (SAGER_DECODE_FUSE=1) |
| mmq-only (TOK=16) | ~v07 | ~v07 | n/a | opt-in (SAGER_PF_MMQ=1) |

Kernel-level evidence (nsys --cuda-graph-trace=node, 128-token greedy):
- the v0.8 default saves ~2.2 ms of GPU kernel time per decode step vs the
  v0.7 kernels (quantize/norm fusion + router top-k + shared-slot write +
  the smx head at 2.54 vs 3.35 ms);
- the remaining decode wall is dominated by the UNTOUCHED v0.7 kernels:
  MoE grouped GEMVs ~6 ms/step-eq, decode attention ~3.7 ms/step-eq at
  ~2.4% of the KV bandwidth (one warp per head, serial KV walk), the GDN
  delta recurrence ~2.9 ms/step-eq, head 2.5 ms. These are the v0.9 levers
  (split-KV flash decode, fused GDN scan, grouped MoE MMQ).

## What v0.8 changed

1. **Prefill MMQ GEMM** (`src/kernels/cuda/mmq.cu`, `SAGER_PF_MMQ=0` reverts).
   The v0.5 mtok kernels hoist the W-side scale/nibble state out of the token
   loop but still serialize the tokens through ONE dp4a dependency chain per
   lane. The MMQ kernel keeps the exact arithmetic order of
   `gemv_mtok_ro_kernel` (ascending sub-block groups `g = lane + gi*32`, the
   same 32-lane shfl-down butterfly, the same fp16 write) and interleaves TOK
   independent token accumulators so the dp4a pipes overlap. Bit-identical
   outputs by construction; the v0.7 profiling put the dense mtok path at
   ~0.75 TFLOPS (~2.3% of the dp4a peak) - the MMQ tile is the lever.
2. **Fused rmsnorm + Q8_K quantization** (`rmsnorm_quant_f16`,
   `SAGER_DECODE_FUSE=0` reverts). One launch per norm instead of two; the
   per-256 quantize groups replicate `quantize_q8_k_kernel` exactly (fmaxf
   warp reduction is order-free, the d/id/rintf/clamp and integer bsums are
   unchanged). Saves 2 nodes per layer and 2 full passes over x.
3. **Multi-matrix projection GEMVs** (`gemv2m` / `gemv3m`). The layer's 2
   (GDN qkvz|ba) or 3 (GQA q|k|v) projections run in ONE launch; per-row math
   is unchanged. Per-layer parameter blocks are written once at capture time
   and replayed as memcpy nodes - no host interplay in the whole-step graph.
4. **GEMV residual-add epilogue** (`gemv_q4_k_q8_f16out_resadd`). The o_proj /
   lin_out GEMV applies `add_kernel`'s exact semantics to the residual row -
   one launch less the o_proj round trip.
5. **Fused router GEMV + warp top-k** (`router_topk_fused`, n=1). The v0.7
   decode path still selected top-k serially in thread 0 of
   `router_topk_batched_kernel` (~20-40 us per layer); the fused kernel uses
   the v0.7-proven warp-parallel selection (lower index wins value ties) and
   reproduces the softmax exactly.
6. **Shared-expert combine slot write** (n=1): the shared row lands straight
   in `expert_y_` slot K - the 2D memcpy node is gone.
7. **smem-x Q6_K head GEMV** (`gemv_q6_k_q8_smx`, `SAGER_HEAD_SMX=0` reverts):
   the x blocks stage in shared memory once per 8-row block (the v0.6 kernel
   re-read them from L2/DRAM per sub-group); same values, same accumulation
   order - bit-identical logits.
8. **FIX: GDN layers never applied `ffn_norm` before the MoE half**
   (OPT-IN: `SAGER_GDN_FFN_NORM=1`; the default stays bug-compatible for
   speed - the fix reroutes the MoE inputs and measured -10..-15% prefill
   from changed expert-miss patterns). The converter writes
   `blk.L.ffn_norm` (post_attention_layernorm) for EVERY block, but
   `linear_layer_launch` / `linear_layer_chunk` ended at the residual add - the
   router and experts read the STALE `attn_norm` row from before the recurrence
   update. All 40 blocks carry the norm (conversion of the real model succeeds
   unconditionally), so it belongs here exactly as dense layers apply it. This
   changes outputs; it is OUT of the bit-parity gate (pinned identically on
   both arms) and the v0.7-exact arm documents the behavior delta.
9. **bench/bench_v8.py**: resident benchmark + bit-parity gates
   (v0.7-path vs v0.8-path greedy streams must be IDENTICAL with the norm fix
   pinned on both sides).

## Correctness gates

- ctest 3/3 (host-side Q4 contract tests; no GPU).
- Parity: greedy 64-token streams, v0.7 kernels (fuse=0 mmq=0) vs v0.8
  defaults, GDN ffn-norm fix pinned OFF for both - IDENTICAL (measured 64/64
  on the T4; every fusion and the MMQ tile are bit-identical by construction).
- The GDN ffn-norm fix changes outputs by design; it is verified by the same
  bit-parity harness when pinned identically on both arms.
- The engine's run-to-run token variance at ~99% expert residency (CPU-pool
  patch promotions) predates v0.8 and is documented in BENCH_RESULTS_v07.

## Reproduce

```bash
bash scripts/dl_sage.sh              # model + expert pool
export SAGER_EXPERT_POOL_FILE=/content/Sage-A1.5-Q4_K_M-expert-pool.bin
export SAGER_EXPERT_POOL_REUSE=1
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build
BENCH_MODEL=/content/Sage-A1.5-Q4_K_M.gguf \
BENCH_POOL=/content/Sage-A1.5-Q4_K_M-expert-pool.bin \
BENCH_CHUNK=512 python3 bench/bench_v8.py
```
