# BENCH_RESULTS_v010 — MoE tensor-core grouped GEMM (SAGER_MOE_TC)

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, 12 GB RAM, CUDA 13.0
Model: Sage-A1.5-Q4_K_M (GGUF 21.7 GB), expert pool 18.1 GB
Engine: `sager_engine` v0.10, pool reused warm, ctx 8192, vram_mb 9984, threads 2
Protocol: 512/2048 random-token prompts (seed 2026), tg128 at temp 1.0 / top_k 20 /
top_p 0.95, median of 3, per-round session reset, equal-warmth expert profile.
Raw JSON + logs ship with the release tarball.

## Headline (T4)

| metric | v0.8 published | v07-exact arm | v0.9 arm | v10 arm | v10 vs v0.9 | v10 vs v0.8 |
|---|---|---|---|---|---|---|
| prefill pp512 | 245.6 tok/s | 223.8 tok/s | 387.0 tok/s | **785.5 tok/s** | **+103%** | **3.20x** |
| prefill pp2048 | 218.8 tok/s | 200.3 tok/s | 316.6 tok/s | **559.5 tok/s** | **+76.7%** | **2.56x** |
| decode tg128 | — | 28.31 tok/s | 27.70 tok/s | **27.31 tok/s** | -1.4% (noise) | — |
| parity gate (greedy 64 tok) | — | IDENTICAL | | | | |

v0.10 arm = v0.9 dense TC + the new MoE TC path (both default-on).
The MoE TC contribution alone (dense TC pinned off, `SAGER_PF_TC=0`),
measured with the same load protocol on the same box:

| arm | pp2048 (3 rounds) | median |
|---|---|---|
| v07-exact (all TC off) | 198.2 / 200.0 / 203.6 | 200.3 tok/s |
| MoE-TC only (`SAGER_PF_TC=0`) | 270.7 / 288.4 / 285.9 | **285.9 tok/s (+42.7%)** |

Same-VM paired A/B with the prof driver (equal warmth, all-default env,
`SAGER_MOE_TC` 0 vs 1):

| gen | v0.9 tps | v0.10 tps | speedup |
|---|---|---|---|
| fresh 2048 | 294.0 | 475.4 | 1.62x |
| warm 2048 | 218.9 | 323.1 | 1.48x |
| after reset 2048 | 315.5 | 584.9 | 1.85x |
| fresh 512 | 243.5 | 384.9 | 1.58x |

Engine profiler attribution over 6656 chunk tokens (fw: dense= / moe=):

| path | v0.9 | v0.10 | delta |
|---|---|---|---|
| dense projections | 9715 ms | 9684 ms | (parity check - unchanged) |
| MoE expert math | 13956 ms | **4471 ms** | **3.12x** |

## What v0.10 changes

1. `SAGER_MOE_TC=1` (default): the routed experts of every MoE layer are
   served by a grouped tensor-core GEMM instead of the v0.4 grouped dp4a
   GEMV family (`gemv_grouped2_tab` / `gemv_grouped_tab`). Per layer:
   - routed rows are compacted expert-major straight from the fp16 input
     (`gather_rows_f16_n` - the Q8_K detour and the down-proj requant are
     gone; the TC GEMM consumes fp16 activations),
   - the chunk pointer table is mirrored to the host once per layer (one
     stream-ordered D2H after the pump publishes the layer's pairs), and
   - the experts walk in groups of 8 through a two-phase pipeline:
     dequantize gate+up into a small fp16 ring, immediately run the group's
     `cublasGemmEx` pairs (m = per-expert routed rows, n = 512, k = 2048),
     then SiLU-mul over the slab, then the same pattern for the down matrix
     (n = 2048, k = 512). In-order-stream slot reuse keeps the ring coherent
     without extra memory.
2. **Zero new VRAM**: the fp16 routed-row slab reuses `expert_y_` (their
   live ranges are disjoint) and the fp16 expert ring reuses `d_wtc_` (the
   dense TC dequant scratch is free between the layer body and the next
   layer). Scratch was the binding constraint: only ~580 MiB of VRAM are
   free after the 10 GB hot tier + 1.5 GB pump arena + weights.
3. Absent experts (pump miss) keep the exact v0.4 semantics: their
   down-output rows are zeroed, the mailbox fires and the AVX2 CPU-pool
   patch re-adds the rows at the next layer boundary
   (`cpu_pool_patch_rows`, shared with `resolve_cpu_mailbox`).
4. Chunks < 256 tokens (verify windows, decode) and CUDA-graph capture
   never take the TC path - decode tg128 is unchanged.
5. `SAGER_MOE_TC=0` restores the exact v0.9 grouped GEMV path; with
   `SAGER_PF_TC=0` as well, the tree is bit-faithful v0.7 again.

## Why the old path was slow

The v0.3/v0.4 grouped GEMV processes Q4_K weights with scalar dp4a +
per-32-weight sub-block scale unpacking at ~1.2 TFLOPS effective
(~3.7% of the dp4a peak) - not a bandwidth problem: the 84 GB/generation
weight re-read traffic models to ~260 ms, but the kernels take 3.4 s. The
warp-per-row structure serializes on dependent fp32 accumulation, re-reads
the full weight row per token tile (tok_per ~ 12), and pays two
`__half2float` + 6-bit scale unpacks per 32 MACs. The TC path trades a
1.77 MB Q4_K read + 6.3 MB fp16 write per expert (166 GB/s linear-store
dequant, 3.12x total MoE speedup end-to-end) for 25 TFLOPS-class HMMA
GEMMs on the same weights.

## Numerics

Like the dense TC path, NOT bit-identical to the dp4a path by design
(fp16 weight rounding, fp16 activations instead of the Q8_K requant for
the down projection, different accumulation order). The parity gate pins
`SAGER_PF_TC=0` **and** `SAGER_MOE_TC=0` on both arms and verifies the
rest of the v0.10 tree. Greedy 64-token streams: IDENTICAL (64 vs 64).

Determinism note: greedy generation is not run-to-run deterministic
within the SAME config across session resets (penalty/recent-window state
survives `reset`); this is pre-existing engine behavior, identical with
`SAGER_MOE_TC=0` - verified on both arms. Fresh-session A/B streams (the
parity gate) remain bit-stable.

## Remaining walls (nsys, v0.10, single warm 2048-token window)

1. **GQA attention prefill** (`attention_kernel`, 10 dense layers,
   warp-per-(head,token), 32-thread blocks, Q8 KV dequantized twice per
   element with no smem staging): ~2.6 s/generation, untouched by v0.9/v0.10
   - now the single largest prefill wall. A flash-style tiled rewrite
     (Q/K tiles in smem, dequant-once, multiple warps per block) is the
     next big lever.
2. GDN chunk scan kernels (conv/l2norm/out_norm/gate) are each < 0.7% of
   GPU time - healthy.
3. Expert pump streams 5.6-15 GB/generation behind the compute shadow
   (0 drops / 0 timeouts in all measured runs); the loader hunt stall
   appears briefly when the ring saturates but does not drop pairs.
4. dequant kernels (dense 12288x2048 + expert ring): ~130 ms/window,
   bandwidth-bound at the v8 linear-store rate.

## Reproduce

```bash
BENCH_MODEL=/content/Sage-A1.5-Q4_K_M.gguf BENCH_CHUNK=512 \
BENCH_POOL=/content/Sage-A1.5-Q4_K_M-expert-pool.bin \
  python3 bench/bench_v8.py --json bench_v10_summary.json
# arms: v10-default | v07-exact | moe-tc-only (SAGER_PF_TC=0)
# parity gate: SAGER_PF_TC=0 SAGER_MOE_TC=0 on both arms
```
