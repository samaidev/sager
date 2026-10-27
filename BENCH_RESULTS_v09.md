# BENCH_RESULTS_v09 — prefill tensor-core dense path (SAGER_PF_TC) + v8 dequant

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, 12 GB RAM, CUDA 13.0
Model: Sage-A1.5-Q4_K_M (GGUF 21.7 GB), expert pool 18.1 GB (HF: tchbcb/sager-t4-cuda-artifacts)
Engine: `sager_engine` v0.9, pool reused warm, ctx 8192, vram_mb 9984, threads 2
Protocol: 512/2048 random-token prompts (seed 2026), tg128 at temp 1.0 / top_k 20 / top_p 0.95,
median of 3, per-round session reset, equal-warmth expert profile (per-arm copies).
Raw JSON + logs ship with the release tarball.

## Headline (T4)

| metric | v0.7 published | v0.7-exact arm | v0.9 arm | delta (same box) |
|---|---|---|---|---|
| prefill pp512 | 248.1 tok/s | 225.6 tok/s | **387.0 tok/s** | **+71.6%** |
| prefill pp2048 | 218.5 tok/s | 199.7 tok/s | **316.6 tok/s** | **+58.6%** |
| decode tg128 (chunk 512) | 30.65 tok/s | 28.24 tok/s | **27.70 tok/s** | -1.9% (noise) |
| parity gate (greedy 64 tok) | — | IDENTICAL | | |

Same Tesla T4, both arms fresh engine processes with per-arm expert-usage
profiles, mtp on, medians of 3 with per-round session resets. Decode is
untouched by v0.9 (the TC path is prefill-only, n_tokens >= stream_min);
the -1.9% tg128 delta is VM run-to-run noise (v0.7 published numbers come
from a faster box).

Against the v0.8 release numbers measured on the previous VM
(pp512 245.6 / pp2048 218.8) the v0.9 arm is +57.6% / +44.7%.

## What v0.9 changes

1. `SAGER_PF_TC=1` (default): the prefill dense projections (attention
   q/k/v/attn_gate/o and GDN-linear qkvz/ba/lin_out) dequantize Q4_K
   weights to fp16 once per chunk and run cuBLAS GemmEx
   (CUDA_R_16F in/out, CUBLAS_COMPUTE_32F -> HMMA tensor cores, fp32
   accumulate). Failure of any call falls back to the dp4a GEMV path
   per-projection. Decode (n=1) never takes this path.
2. The dequant kernel is v8 "linear-store": one thread writes one 16 B
   output slab (8 weights), output index == thread index. Probes 6-10 on
   sm_75 showed the write pattern is everything: 512 B-per-thread chunked
   stores run at 18-50 GB/s while linear 16 B-per-thread stores hit
   166 GB/s (reads were never the issue: 218 GB/s; memcpy ceiling 242).
   Bit-identical output to the sub-block kernel
   (0 / 25165824 half mismatches), 3.4x faster (1.06 -> 0.31 ms for the
   12288x2048 qkvz projection).
3. `SAGER_PF_TC=0` restores the exact v0.7/v0.8 dense path.

## Numerics

The TC path is NOT bit-identical to the dp4a path by design (weights round
to fp16, activations stay fp16 instead of being requantized to Q8_K, and
the accumulation order changes). The parity gate therefore pins
`SAGER_PF_TC=0` on both arms and verifies the rest of the v0.9 tree (fuse,
mmq, scheduler, v8 dequant is bit-exact by construction and exercised on
the TC path only). Greedy 64-token streams: IDENTICAL (64 vs 64).

## Same-VM paired A/B (SAGER_PF_TC 0 vs 1, prof driver)

| gen | dp4a wall | TC wall | dp4a tps | TC tps | speedup |
|---|---|---|---|---|---|
| fresh 2048 | 9.81 s | 6.90 s | 210.3 | 300.2 | 1.43x |
| warm 2048 | 12.48 s | 9.68 s | 165.6 | 214.2 | 1.29x |
| after reset 2048 | 9.93 s | 6.72 s | 207.7 | 307.9 | 1.48x |
| fresh 512 | 3.00 s | 2.23 s | 175.6 | 238.3 | 1.36x |

(The TC arm ran at systematically lower SM clocks - 585-930 MHz min vs
1095-1155 for dp4a - and still won; clock parity would look better.)

## Where the remaining prefill wall is (nsys, one 2048-token window, per gen)

| kernel group | time/gen | share |
|---|---|---|
| MoE grouped GEMVs (down + gate/up, mtok-ro) | 3.38 s | ~45% |
| attention_kernel (warp per (head, token), fp32 SIMT) | 2.67 s | ~36% |
| TC GemmEx (all dense projections) | 0.14 s | ~2% |
| dequant v8 (all dense projections) | 0.065 s | ~1% |
| everything else (norms, scans, gathers, router) | ~1.2 s | ~16% |

Notes:
- The expert pump is NOT the bottleneck at this operating point: it needs
  ~7.65 GB/gen at ~1.2-1.3 GB/s effective and hides under ~6 s of compute
  (0 steals / 0 drops / 0 arm timeouts throughout). It becomes the wall
  again only if the kernels above get ~2x faster.
- attention_kernel is one warp per (head, token) scanning the KV serially
  in fp32 - ~15x off the fp32 SIMT roofline for this shape and the single
  largest remaining kernel. A tensor-core flash-prefill rewrite is the
  next structural lever.
- The MoE grouped GEMVs run at ~195 GFLOPS (0.6% of dp4a peak), 5-6x above
  their DRAM-traffic floor (weights are read exactly once per expert per
  window at tok_per=16). Register-blocking the token loop is the cheap
  first step; a real dp4a tile GEMM is the structural one.
