# BENCH_RESULTS_v012 — Q4_K correctness fix + GDN R4 / shared-expert & router TensorCore

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, 12 GB RAM, Colab VM
Model: Sage-A1.5-Q4_K_M (GGUF 21.7 GB), expert pool 18.1 GB (file-backed mmap,
`SAGER_EXPERT_POOL_FILE` + `SAGER_EXPERT_POOL_REUSE=1`)
Engine: `sager_engine` v0.12-wip @ 40d86c0, ctx 8192, vram_mb 9728-9984, threads 2
Protocol: bench_v8.py, 512/2048 random-token prompts (seed 2026), tg128 at
temp 1.0 / top_k 20 / top_p 0.95, median of 3, per-round session reset.
Four arms per chunk config (same binary, feature flags toggled via env):

| arm | meaning |
|---|---|
| `v12-default` | all v0.12 features on (GDN_R4 + SH_TC + ROUTER_TC + v0.11 paths) |
| `v07-exact` | every fast path off (DECODE_FUSE/PF_MMQ/HEAD_SMX/GDN_FFN_NORM/PF_TC/MOE_TC/SH_TC/ROUTER_TC/GDN_R4 = 0) |
| `v12-off` | GDN_R4/SH_TC/ROUTER_TC off (isolates the three new v0.12 kernels) |
| `moe-tc-only` | MoE TC on, PF TC + GDN ffn-norm off |

## Headline (T4, medians of 3)

| metric | v0.11 | v0.12 c512 run | v0.12 c2048 run | vs v0.11 |
|---|---|---|---|---|
| prefill pp512 | 823.8 tok/s | **1097.2 tok/s** | **995.2 tok/s** | **1.21-1.33x** |
| prefill pp2048 | 731.9 tok/s | **915.0 tok/s** | **1320.5 tok/s** | **1.25-1.80x** |
| decode tg128 | 33.30 tok/s | **40.1 tok/s** | **39.4 tok/s** | **1.18-1.20x** |
| parity gate | IDENTICAL | see quality note | see quality note | |

Cumulative vs the v0.8 published baseline (pp512 245.6 / pp2048 218.8 /
tg128 ~27.3): **pp512 4.1-4.5x, pp2048 4.2-6.0x, tg128 +44-47%**.

Caveat: the c512 run was the first load after the pool file was (re)built on
this fresh VM - its fill ran inside the first bench load; the c2048 run
reused the warm pool. Both are reported; the c2048 run is the cleaner A/B.

## Arm-level detail (medians, tok/s)

### chunk 512 (`v12_c512.json`)

| arm | pp512 | pp2048 | tg128 |
|---|---|---|---|
| v12-default | 1097.2 | 915.0 | 40.1 |
| v07-exact | 227.2 | 212.8 | 38.0 |
| v12-off | 794.3 | 786.7 | 39.1 |
| moe-tc-only | 287.7 | 302.1 | 38.2 |

### chunk 2048 (`v12_c2048.json`)

| arm | pp512 | pp2048 | tg128 |
|---|---|---|---|
| v12-default | 995.2 | 1320.5 | 39.4 |
| v07-exact | 224.5 | 219.4 | 38.5 |
| v12-off | 801.6 | 994.5 | 39.1 |
| moe-tc-only | 289.0 | 331.9 | 38.2 |

Same-binary A/B (`v12-default` vs `v07-exact`, chunk 2048): **pp512 4.4x,
pp2048 6.0x, tg128 ~1.02x** - the prefill gain comes almost entirely from the
tensor-core + MMQ fast paths; decode is unchanged by design (it was already
fused in v0.11) and sits at the H2D streaming ceiling for a 21.7 GB model on
a 12 GB-RAM host (pool-stream pump: 31.6 GB H2D per 2048-token prefill pass,
0 drops / 0 skipped / 0 arm timeouts).

## What v0.12 changes

### Correctness (the word-soup fix, critical)

1. **Q4_K nibble order adopted ggml official order across the whole stack**
   (40d86c0). sager had been reading/writing Q4_K in Q4_0-style adjacent
   nibble order (byte j: low = w[2j], high = w[2j+1]); the llama.cpp-converted
   Sage-A1.5 GGUF stores the official ggml Q4_K order (per 32-byte block:
   low nibbles -> w[0..31], high nibbles -> w[32..63]). Every raw-Q4_K tensor
   (embedding, FFN gate/up, shared expert) was permuted garbage at runtime.
   Fixed in: host `q4_k.cpp` (dequant_row / quantize_row / vec_dot_ref),
   CUDA `q4_k.cu` (dp4a sub-blocks + mtok load), `mmq.cu`, `tc_gemm.cu`,
   CPU `q4_avx2.cpp` (scalar + AVX2), and the toolchain
   (`convert_hf_to_sager.py`, `make_smoke_gguf.py`, `validate_quant.py`).
   Verified bit-exact against an independent ggml reference implementation
   (quantize->dequant RMSE 8.4%, typical Q4_K; vec_dot error 8e-9), and the
   GGUF embedding row-0 cosine vs the bf16 safetensors ground truth went
   0.1093 (broken) -> 0.9957 (fixed).
2. **Q6_K scale byte offset fix** (2e96815): `dequant_row_q6` read the block
   scales from bytes 208-209 instead of 192-193 (`ffn_down_exps` is Q6_K in
   this GGUF), so every Q6_K row was silently wrong.
3. **`gdn_ffn_norm` defaults ON** (158c411): qwen35moe GDN blocks have a real
   post-attention norm; the default-off skipped it, compounding the drift.

### Performance

4. **GDN scan R4 warp kernel** (`SAGER_GDN_R4`, 1b318f6): 4-token warp-level
   scan for the gated-delta-net recurrence in prefill.
5. **Shared-expert TensorCore GEMM** (`SAGER_SH_TC`): the always-on shared
   expert moves to the tc_gemm path instead of the dp4a MMQ path.
6. **Router TensorCore GEMM** (`SAGER_ROUTER_TC`): router logits via
   TensorCore tile GEMM.

## Quality note

The bench above is speed-only (`--skip-parity`). Text-quality validation with
the fixed stack (3-prompt greedy generation) runs as `v12_quality.py`; the
Q4_K ground-truth cosine check (0.9957) and the bit-exact quantizer
cross-check already gate the release, and the generation smoke test is the
final gate - see the release notes for its output.
