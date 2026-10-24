# SAGER v0.2.2 vs llama.cpp — Sage-A1.5-Q4_K_M on Tesla T4

## Result (same GPU, same model, 2026-10-24)

| Metric | sager v0.2.2 (expert streaming, all-GPU) | llama.cpp ngl30 (10 CPU layers) | speedup |
|---|---|---|---|
| prefill pp512  | **48.96 tok/s** | 20.51 tok/s | **2.4x** |
| decode  tg128  | **18.59 tok/s** | 3.45 tok/s  | **5.4x** |
| VRAM model fit | 20.2 GB weights on 15.4 GB card (52% hot experts resident, rest streamed per-token over PCIe) | needs 19.9 GB -> OOM at ngl99, must offload 10 layers to 2 weak CPU cores | - |

sager config: vram_mb=8960 hot pool, pump_arena 1024 MB, ctx 8192, chunk 2048,
kv q8, 40 per-layer decode CUDA graphs captured, suffix drafter on (drafted=125
over 128 tokens, acceptance 0 at temperature 1.0 on random prompts - expected),
mtp off. Load 2min07s with SAGER_EXPERT_POOL_REUSE=1 (30 min cold build).

Decode cost model at 52% residency: ~152 hot misses/step x 1.69 MB PCIe H2D
~= 257 MB/step ~= 43 ms -> matches the measured 54 ms/step. Higher residency
(bigger card) or workload expert locality (agentic/code) pushes decode up.

## Correctness gates passed before this benchmark

- ctest 3/3 (q4 roundtrip, GDN golden vectors, expert pump policy)
- compute-sanitizer memcheck: ERROR SUMMARY: 0 errors over full
  load -> warmup -> 40-layer graph capture -> generation pipeline
  (after fixing the GDN shared-memory reduction scratch overflow,
  commit 7b070c7)
- greedy MTP/suffix bit-exactness covered by tests/smoke_engine.py phases

## What was fixed to get here (v0.2.2)

1. c00fe48 - native qwen35moe GGUFs suffix every tensor with ".weight";
   expert stacks hard-failed, norms+MoE router silently skipped
2. 68c4c16 - CpuPool::stop() lost-wakeup deadlock (join forever)
3. 9448831 - C4 submit check refused every CPU-pool miss job
4. 7b070c7 - GDN shared-memory reduction scratch overflow (sanitizer-found)
5. d93002b - SAGER_EXPERT_POOL_REUSE=1: 30 min cold pool build -> 2 min
6. 94e5378 - alloc_scratch per-buffer OOM reporting (root-caused the
   vocab-248320 budget drift vs the t4-guide)

## Reproduce

```bash
# 1. build (any CUDA GPU, sm_75 verified)
git clone https://github.com/samaidev/sager && cd sager && ./setup.sh

# 2. model
hf download InternScience/Sage-A1.5-Q4_K_M-GGUF Sage-A1.5-Q4_K_M.gguf --local-dir .

# 3. run (echo JSON lines into the engine)
export SAGER_EXPERT_POOL_FILE=$PWD/pool.bin SAGER_EXPERT_POOL_REUSE=0
# first run builds the 16.9 GB pool (~30 min); subsequent runs set REUSE=1
echo '{"cmd":"load","model":"Sage-A1.5-Q4_K_M.gguf","ctx":8192,"vram_mb":8960,
"pump_arena_mb":1024,"threads":2,"graphs":true,"mtp":false,"chunk":2048,"kv_q8":true}
{"cmd":"generate","ids":[72,3820,913],"max_tokens":64,"temperature":1.0,"seed":42}
{"cmd":"shutdown"}' | ./build/sager_engine
```

Artifacts in the release tarball are sm_75 CUDA binaries built by nvcc 13.0 on
commit 7b070c7 (see BUILD_INFO.txt).

---

# v0.2.3 - Strata-style streamed prefill (status: code landed, re-bench pending)

## What changed (commits 78b2f88..c754ed2)

v0.2.2's prefill walked the prompt PER ROW: every (token, layer) dispatched
its own top-k, synced the whole device, and submitted its own pageable
copies - a 512-token chunk paid 20,480 device syncs (measured 48.96 tok/s).
v0.2.3 rewrites the prefill MoE half the way Strata streams experts:

- per-LAYER batched MoE: one route sync per layer, routed rows compacted
  expert-major, batched per-expert GEMVs, batched combine (new kernels:
  router_topk_batched, gather_rows_q8, scatter_routed, moe_combine_batched,
  sigmoid_gate_scalar_batched, scatter_rows_generic);
- stream-all ring in ExpertPump: for chunks >= SAGER_PF_STREAM_MIN (384
  tokens, top-8 of 256 -> nearly every expert is routed) the routing is not
  needed to schedule copies - hot-tier AND arena misses stream in fixed
  (layer, id) order through the arena ring while compute consumes that same
  order; copies overlap attention compute;
- pinned-RAM loader threads (SAGER_PF_READERS x SAGER_PF_SLABS): pool file
  -> pread -> pinned slab -> cudaMemcpyAsync -> arena slot. No more pageable
  mmap as the copy source;
- sessions are per CHUNK; resident slots are parked (reused without a copy)
  and a fence event orders the copy stream behind compute across the chunk
  boundary;
- SAGER_PF_BATCHED=0 switches back to the exact v0.2.2 per-row walk (A/B).
  Both walks share the same kernels per row - outputs are bit-identical.

## Failure mode found and fixed (first streamed run, 2 tok/s)

The first run stalled: zero-count pairs (experts no token routes to, ~15 of
256 per layer on random-id prompts) that the loaders landed AFTER their
layer had passed clogged ring slots forever -> 112 x 2s arm timeouts = the
whole 238 s prefill. Fixed in c754ed2: stream_progress(L) releases every
slot owned by a pair of an earlier layer (parked: immediately; consumed:
once its down-projection event completes), loaders skip past-layer pairs at
take AND publish time, sessions became per-chunk, timeout pairs are dropped.

## Measured so far (pre-fix run + disk physics, Tesla T4 Colab)

- non-hot plan per 512-token chunk: 4,931-5,309 pairs = 8.1-8.7 GB to stream
  (52% hot residency is profile-driven and uncorrelated with random-id
  routing);
- Colab T4 box: 12.7 GB RAM < 16.9 GB pool -> the cold fraction of every
  first pass reads from disk at ~261 MB/s sequential (dd measured). A fully
  cold pass is therefore disk-bound: ~33 s/chunk floor, warm page-cache
  passes are PCIe-bound (~1 s/chunk ceiling);
- first (buggy) run: 5.74 GB H2D in 316 s - timeout-dominated, not
  representative; post-fix numbers pending a fresh GPU session.

## Reproduce the A/B

```bash
python3 bench_v023.py all    # arm A = streamed (default), arm B = per-row
                             # (SAGER_PF_BATCHED=0); token streams must be
                             # IDENTICAL - that is the correctness gate
```

Run A twice and report the second pass: the first pass pays the cold disk
reads, every later pass is the steady-state number (matching how the v0.2.2
48.96 tok/s baseline was itself measured on a warm cache).
