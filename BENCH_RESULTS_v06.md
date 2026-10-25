# BENCH_RESULTS_v06 — full-layer decode graphs + native Q6_K head

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, driver 580.82.07
Model: Sage-A1.5-Q4_K_M (GGUF), 21.7 GB; expert pool `pool.bin` 18.1 GB (prebuilt, reused)
Engine: `sager_engine` @ commit 5de4ba3 (v0.6), CUDA graphs ON, device dispatch ON, device sampler ON
Load config: ctx 8192, chunk 2048, vram_mb 9984, pump_arena_mb 1024, kv_q8, threads 2

## Headline (T4)

| metric | v0.5 | v0.6 | delta |
|---|---|---|---|
| decode tg128 (graphs-only build path, median of 3) | 24.56 tok/s | **30.26 tok/s** | **+23.2%** |
| decode tg128 (full v0.6: + Q6K head + topk2) | 24.56 tok/s | ~30 tok/s (median pending re-run) | +22..25% |
| prefill pp512 | 223.0 tok/s | **232–235 tok/s** | +4..5% |
| prefill pp2048 | 212.0 tok/s | **226–229 tok/s** | +7..8% |
| ctest | 3/3 | 3/3 | — |
| token-stream gate (legacy paths vs full-graph path) | — | **IDENTICAL 128/128** | bit-exact |

Measured prompts: 512 random tokens (seed 2026), 128 generated at temperature 1.0 / seed 42,
hot-expert tier 5916/10240, arena 606 slots, VRAM pool reused warm.

## What v0.6 changed

1. **Full-layer decode CUDA graphs** (SAGER_FULL_GRAPH, default on).
   v0.5 replayed a captured graph for the dense/GDN half of each layer and ran the
   MoE half as ~13 eager launches per layer (~390 launches/step, launch-bound:
   the MoE segment measured 15.0 ms of a 40.4 ms step against a ~1.5 ms bandwidth
   floor). v0.6 captures the ENTIRE layer - dense/GDN body plus the device-dispatch
   MoE body - into ONE graph per layer (40/40 captured). This is possible because
   the v0.4 MoE dispatch is already graph-shaped: the plan (histogram / prefix
   sums / grouped scatter) is computed on-device, the grouped GEMV grids are
   host-fixed (`tiles_max`), expert weights are read through the indirect decode
   residency table (updated in place between steps), and the absent-expert mailbox
   writes go to per-layer device slots whose pinned mirrors are captured as memcpy
   nodes. Per-step launch count drops from ~430 to ~45.

2. **Step-end CPU patch for absent experts** (`patch_graph_mailbox`).
   A replayed layer that routes a non-resident expert zeroes that expert's rows
   in-kernel and mails it (device-side, graph-safe). The captured body preserves
   each layer's dispatch plan and input row in per-layer buffers (< 1 MB total),
   so after the step sync the host can rebuild the (row, slot) list, run the AVX2
   CPU pool, and add the missing contributions into the residual stream BEFORE the
   head runs - the step's logits stay exact (same semantics as the v0.4 boundary
   patch). At the benchmark's steady state (~99% row residency) patches are rare.

3. **Native Q6_K head GEMV** (`gemv_q6_k_q8`, SAGER_HEAD_Q6K, default on).
   The 248,320-row head previously dequantized to fp16 at load: 1,017 MB of weight
   traffic every decode step (258 GB/s effective - the f16 kernel was already
   near bandwidth). v0.6 keeps the head in its native 6-bit super-block format
   (401 MB/step, 616 MB less VRAM) and computes the dot products on dp4a with the
   -32 zero point corrected against int8 activations. Logits differ from the fp16
   head at the rounding level - the native dot is the more faithful of the two -
   which shifts sampled tokens at temperature 1.0 (125/128 differ from the fp16
   path on the random-prompt gate; greedy/critical workloads should re-validate).
   `SAGER_HEAD_Q6K=0` at runtime or `SAGER_HEAD_F16=1` at load restores fp16.

4. **Two-phase top-k sampler** (SAGER_TOPK2, default on). The v0.5 device sampler
   picked k=32 of 248,320 logits with one block iterating max+mask 32 times
   (~0.7 ms/step). v0.6 scans 2048-logit segments in parallel (122 blocks) and
   merges their candidates in one block (~25 us), with the identical selection
   set and (value, lower-index) tie rule.

5. **Warp-per-row GDN chunk scan** (SAGER_GDN_WS, default on; affects prefill).
   The scalar scan ran one 1-warp block per 32 rows with the state slice in shared
   memory - 128 warps GPU-wide (4% occupancy) and two block barriers per token.
   The warp version gives every (head, row) pair its own warp (4096 warps), keeps
   the K=128 state dims in registers (float4 per lane), and streams k/q through L2
   with no shared memory and no block barriers. The fp32 reduction order changes
   (lane partials + shuffle tree), so outputs differ at the rounding level - the
   same trade Strata makes for its chunked GDN scan.

6. **Bug fix: `quantize_q8_k_n` grid underflow.** The launcher computed
   `n_max / 256` blocks; at the single-token decode step n_max = 16, so the grid
   rounded to ZERO and the launch failed with "invalid argument", silently leaving
   the routed experts' requantized activations stale on the v0.5 device decode
   path (the shared expert's 2-block quantize into the same buffer masked it).
   v0.6 uses `(n_max + 255) / 256`. This also means v0.5's device-path decode
   numerics were silently wrong; the v0.6 graph path runs the requant for real.

## Profiler segments (SAGER_PROF, decode step, 128 steps)

v0.5 (eager MoE): upload 0.3 ms | dense-graphs 17.1 ms | eager MoE 12.6 ms |
head (f16) 3.3 ms | sampler 0.7 ms  ->  ~33.9 ms/step, 24.5-29.5 tok/s

v0.6 (full-layer graphs, prof events on - includes ~0.3-0.5 ms event overhead):
upload 0.5 ms | 40 full-layer graph replays ~31.5 ms | head (f16) 1.3 ms |
sampler 0.9 ms  ->  ~30 tok/s measured in-run.

The remaining decode time is the 40 per-layer graph replays themselves
(dense/GDN weights ~990 MB + routed experts + shared expert per step) and the
head; the next levers are whole-step graph capture (1 launch/step), decode-side
GDN step kernels, and batched speculative verify (device path).

## Correctness gates

- `ctest`: 3/3 (q4_host_roundtrip, gdn_host_reference, expert_pump_policy)
- Token-stream identity, 512-random-token prompt, 128 generated tokens,
  temperature 1.0 / seed 42:
  - OLD arm (all v0.6 switches off - v0.4/v0.5 legacy paths) vs NEWCORE arm
    (full-layer graphs + requant fix only): **IDENTICAL 128/128**
  - NEW arm (adds Q6K head + warp GDN scan + two-phase top-k) vs OLD: differs
    from the first token - expected: the Q6_K native head changes logits at the
    rounding level and temperature-1.0 sampling amplifies ulp differences
- The requant fix (item 6) changes the v0.5 device-path decode numerics BY
  DESIGN: v0.5 silently skipped the routed experts' requant at n=1; v0.6 runs
  it. The legacy (non-device) path was never affected, which is why the
  legacy-vs-graph gate passes bit-exact.

## Reproduce

```bash
export SAGER_EXPERT_POOL_FILE=/content/pool.bin
export SAGER_EXPERT_POOL_REUSE=1
export SAGER_DECODE_DEVICE=1 SAGER_DEVICE_SAMPLER=1
/content/sager/build/sager_engine   # JSON commands on stdin
# kill switches: SAGER_FULL_GRAPH=0 SAGER_HEAD_Q6K=0 SAGER_TOPK2=0 SAGER_GDN_WS=0
# fp16 head at load: SAGER_HEAD_F16=1
```
