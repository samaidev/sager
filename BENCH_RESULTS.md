# SAGER v0.3.0 — mtok tiled kernels + real streamed prefill (Tesla T4)

## Result (same GPU, same model, Sage-A1.5-Q4_K_M, commit db6f2bf base)

| Metric | v0.2.2 (per-row) | v0.2.5b (streamed, broken) | **v0.3.0** | vs v0.2.2 |
|---|---|---|---|---|
| prefill pp512  | 48.96 tok/s | 36.0 tok/s | **103–109 tok/s** | **2.1–2.2x** |
| prefill pp2048 | -           | 65.0 tok/s | **108–120 tok/s** | **1.7–1.8x** |
| decode tg128   | 18.59 tok/s | (unmeasured) | **21.0–21.7 tok/s** | **+13–17%** |

Correctness gates: ctest 3/3; token-stream equality VERIFIED at the token
level — A (mtok kernels) == A (SAGER_MTOK=0) == B (per-row walk), 128/128
ids identical on the shared random-512 prompt; 0 "exceed cap" row drops.

## What was fixed / changed

### 1. The streamed prefill never streamed (upstream bug, found + fixed)
`build_stream_plan` received the MISS bitmap (1 = needs streaming) but
filtered with `!resident[p]` - the plan held exactly the HOT set (5309
pairs, never the ~4931 misses). Masked under the old layer-major residency
(lookups fast-failed past the plan), fatal under uniform residency
(60 s stalls per lookup). Fixed the condition and the not-in-plan check
(`lower_bound()==end()` only rejected pairs above the plan max - now
`binary_search`). Also bounded the loader slot-hunt (was unbounded: ring
full + walk blocked in lookup = deadlock).

### 2. Cold-start residency was layer-major (fixed)
`ExpertCache::rebalance` with an empty profile ranked all-zero scores and
`partial_sort` kept the FIRST 5309 pairs = layers 0-20 fully resident,
layers 21-39 empty. Seeded EXPERT-MAJOR round-robin instead: every layer
gets the same hit rate, the walk paces against the loaders, and decode
sees a uniform miss pattern.

### 3. mtok tiled kernels (the prefill speed)
The v0.2.3 "batched" kernels put n_tokens in grid.y: every token's warp
re-read the whole weight row - a 512-token chunk routed 4096 rows/layer
over 256 experts (~16 rows/expert) and paid ~16x memory amplification
(~290 GB of weight traffic per chunk on T4). New mtok kernels tile tokens
through shared memory (weights staged once per tile, 8-32x less traffic),
keeping the exact per-(row, token) accumulation order - bit-identical
outputs:

- `gemv_q4_k_q8_f16out_mtok` - single matrix
- `gemv2_q4_k_q8_f16out_mtok` - fused gate+up (one launch, x read once)
- `gemv_grouped2_...` / `gemv_grouped_...` - ALL routed experts of a layer
  in ONE launch (flattened tile map), MoE goes from ~770 launches/layer
  to 2
- `gemv_f32_out_f16x_mtok` - fp32 router, k-chunked, order-preserving

Used by the batched MoE walk, the dense attention projections, and the
shared expert. SAGER_MTOK=0 falls back to the v0.2.3 kernels (A/B).

### 4. GDN chunk kernels (the launch storm)
The per-token GDN walk issued ~12 tiny launches per token - 30 layers x
512 tokens = ~184k launches per chunk on a 2-vCPU host. New chunk kernels
batch conv/gates/l2-norm/out-norm over the window and fuse the sequential
rank-1 recurrence into ONE kernel whose blocks keep their state slice in
shared memory (loaded once, stored once per chunk). Bit-identical per-row
math (thread-per-state-row, serial i order). A GDN layer is now ~11
launches per CHUNK.

### 5. Decode shaves
- the per-layer miss-path activation D2H (h_x16_) is only pulled when a
  CPU-pool miss actually happens (was unconditional every layer/step)
- grouped MoE + mtok also serve verify windows (MTP path benefits too)

## Reproduce

```bash
git clone https://github.com/samaidev/sager && cd sager && ./setup.sh
hf download InternScience/Sage-A1.5-Q4_K_M-GGUF Sage-A1.5-Q4_K_M.gguf
export SAGER_EXPERT_POOL_FILE=$PWD/pool.bin SAGER_EXPERT_POOL_REUSE=0   # first run builds 16.9 GB pool (~35 min)
python3 bench/bench_v023.py all      # arm A = mtok streamed, arm B = per-row
python3 bench/bench_v023.py A        # warm pass - the headline numbers
```

Run A twice and report the second pass (first pass pays cold page-cache
reads of the pool). Measured on Colab T4 (2 vCPU, 12.7 GB RAM), engine
env: vram_mb=8960, pump_arena_mb=1024, ctx 8192, chunk 2048, kv q8,
graphs on, mtp off.

## Notes

- Decode at 52% residency is PCIe-miss bound (~150 misses/step x 1.69 MB
  over a pageable-mmap pool on a 12.7 GB RAM box): tg128 improves only
  ~15%. Bigger cards / agentic workloads with expert locality gain more.
- Strata's own sm_75 experiment (bench/results/2026-09-30-tc-gemv-sm75)
  measured tensor-core GEMV as a regression for decode; the warp dp4a
  decode path is unchanged, consistent with that finding.

## Tuning note: vram_mb 9984

The Colab T4 has ~1.8 GB of spare VRAM after the 8960 MB pool + 1024 MB
arena + weights + scratch. vram_mb=9984 fits (capacity 5916/10240 = 58%)
and measured pp512 = 116 tok/s (vs 103-109 at 8960). 10240 does NOT fit
(scratch OOM at lin_on_b_, 14 MiB short).

## Decode analysis (v0.4 direction)

At the v0.3 residency the router is heavily skewed on real inputs
(cache_hit_rate ~0.985-0.99 on the bench prompt): PCIe miss traffic is no
longer the decode bottleneck. GPU utilization during decode is 91-100%,
but the roofline math says the same work needs only ~8-9 ms of the
measured ~47 ms step: decode is bound by small-kernel launch/drain
overhead (~600 kernels/step; one-row GEMVs at ~12% per-launch occupancy).
The v0.4 lever is whole-step graph capture with device-side expert
dispatch (static (layer,expert)->VRAM pointer table + arena query for the
rare miss), which removes the per-layer host round-trips and the eager
MoE launches entirely.
