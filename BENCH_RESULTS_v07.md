# BENCH_RESULTS_v07 — whole-step decode graph + warp router + fast sampler + fixes

Hardware: NVIDIA Tesla T4 (sm_75, 15 GB), 2 vCPU, driver 580.82.07, CUDA 13.0
Model: Sage-A1.5-Q4_K_M (GGUF 21.7 GB), expert pool 18.1 GB (HF: tchbcb/sager-t4-cuda-artifacts)
Engine: `sager_engine` v0.7 (this commit), pool reused warm, ctx 8192, vram_mb 9984, threads 2
Protocol: 512/2048 random-token prompts (seed 2026), tg128 at temp 1.0 / top_k 20 / top_p 0.95,
median of 3, per-round session reset, SAGER_PROFILE_PATH warm.

## Headline (T4)

| metric | v0.6 (published) | v0.7 chunk 2048 | v0.7 chunk 512 |
|---|---|---|---|
| prefill pp512 | 232–235 tok/s | **248.0 tok/s (+6–7%)** | **248.1 tok/s** |
| prefill pp2048 | 226–229 tok/s | 223.8 tok/s | 218.5 tok/s |
| decode tg128 | 30.26 tok/s | 29.92 tok/s | **30.65 tok/s** |
| ctest | 3/3 | 3/3 | 3/3 |

## What v0.7 changed

1. **Whole-step decode graph** (`SAGER_STEP_GRAPH`, default on with `SAGER_DECODE_DEVICE=1`).
   The decode step was measured at **643 kernel launches/step with the GPU only 32% busy**
   (nsys: ~23 ms/step of launch starvation on the 2-vCPU host). v0.6's per-layer graphs
   still cost one `cudaGraphLaunch` per layer plus eager sampler calls. v0.7 captures the
   ENTIRE step - embedding + all 40 layers (dense/GDN body + graph-safe device-dispatch
   MoE) - into ONE graph, and the head + penalties + two-phase top-k + pinned mirrors into
   a second one: **2 `cudaGraphLaunch` calls per step**, one sync, rare absent-expert
   patch re-runs the head graph. On T4 the wall becomes the GPU front-end's ~30 µs/node
   × 643 nodes - parity speed today, but the structure is now launch-cost-free for
   future kernel fusion (v0.8).
2. **Warp-parallel router top-k** (`router_topk_kernel`). The serial single-thread
   selection (8 × 256 scans) measured 40 µs/call × 40 calls/step = 1.6 ms/step. The warp
   version keeps the exact (value, lower-index) tie semantics of the serial walk
   (token-stream parity verified within sampling routes) and runs in ~5 µs.
3. **Fast device top-k on the legacy decode path** (`SAGER_FAST_SAMPLE=1`, default on).
   The host sampler pays a softmax/argmax over the 248,320-token vocab in scalar code
   every step. The device route (penalties + two-phase top-k on GPU, ~25 µs) replaces it;
   the full logits still land on the host as the `finish_device_sample` fallback.
4. **Deterministic penalties kernel** (`apply_penalties_f32_devn`). The original
   penalties kernel loses updates when an id repeats inside the 64-entry recent window
   (unsynchronized read-modify-write; outcome depended on the launch configuration). The
   new kernel folds all m occurrences into the last-occurrence thread - race-free and
   bit-equal to the sequential semantics (rep m times, then pres m times, matching
   `sample_host`'s order).
5. **Bug fix: the device sampler never applied request penalties.**
   `Session::set_sampler_penalties` existed since v0.5 but was NEVER called, so the
   device-route samples ignored `repeat_penalty`/`presence_penalty`. generate.cpp now
   sets them per request.
6. **Bug fix: `serve/engine_proc.py` crashed on startup** with
   `Popen() got an unexpected keyword argument 'buffering'` (invalid kwarg; `bufsize`
   intended). The engine could not start under Python 3.13 without this fix.
7. **`scripts/dl_sage.sh`** - one command fetches the model GGUF + the prebuilt expert
   pool (+ optional binaries tarball) from HuggingFace (primary) or ModelScope
   (`--source ms`), skip-if-present and resumable.
8. **v0.7 measurement infrastructure**: `bench_v6.py`-style resident benchmark with
   per-round resets (the accumulated-context bug invalidated any benchmark that reuses
   the session: decode_tps decays 41.9 → 5.3 tok/s as pos grows).

## Profiling record (v0.7 investigation, T4)

- Legacy decode step (v0.6 path): 643 launches, GPU busy 32% (10.9 ms kernels,
  33.8 ms wall). Launch starvation, not bandwidth.
- Whole-step graph: 2 launches, GPU span 35.0 ms/step - T4's per-node front-end
  (~30 µs × 643 nodes) becomes the wall. Kernel fusion is the v0.8 lever.
- Prefill (3072 tokens): dense 7.66 s + MoE 5.15 s + upload 0.21 s. The MoE segment
  includes expert streaming: **4,324 non-resident (layer, expert) pairs = 7.1 GB H2D per
  2048-token prompt at ~1.7 GB/s effective** (2-vCPU staging memcpy bound). The dense
  mtok GEMV kernels run at ~0.75 TFLOPS (~2.3% of T4's dp4a peak) - an MMQ-style
  dp4a GEMM or dequant+cuBLAS fp16 path (Strata's `STRATA_DENSE_MMQ` / `STRATA_BF16_TC`
  approach) is the v0.8 prefill lever.
- `BENCH_VRAM=11500` grows the hot tier to 6814/10240 experts but leaves no VRAM for
  the pump arena (`pump=off`) - prefill collapses to 93 tok/s. 9984 MB is the sweet spot.
- Router top-k at 40 µs/call was pure serial-scan waste; head Q6K GEMV runs at
  120 GB/s (3.35 ms/step) - a tuned Q6K row kernel is a further ~2 ms/step on the table.

## Reproduce

```bash
bash scripts/dl_sage.sh              # model + expert pool (HF primary, MS fallback)
export SAGER_EXPERT_POOL_FILE=/content/Sage-A1.5-Q4_K_M-expert-pool.bin
export SAGER_EXPERT_POOL_REUSE=1
python3 bench_v6.py                  # prefill/decode medians (chunk 2048)
BENCH_CHUNK=512 python3 bench_v6.py  # decode-heavy config
```

Correctness gates: ctest 3/3; greedy token-stream A/B between the step-graph and
per-layer paths is equality-gated per sampling route; the engine's run-to-run
token variance at 99% expert residency (CPU-pool patch promotions) predates v0.7
and is documented here for future gating work.
