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
