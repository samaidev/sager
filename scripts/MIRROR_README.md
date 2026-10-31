# sager-t4-cuda-artifacts

Prebuilt SAGER engine binaries and (T4-built) large model artifacts for
[Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) (35B MoE, ~3B
active) on one NVIDIA Tesla T4 (16 GB). Engine source:
https://github.com/samaidev/sager

## Files

| file | what | status |
|---|---|---|
| `sager_v0128_t4_cuda.tar.gz` | engine v0.12.8 (sm_75), host tests, libs, runbook, publisher | **current** (roots 5-16 fixed) |
| `sager_v011_t4_artifacts.tar.gz` and older | previous releases | superseded |
| `Sage-A1.5-Q4_K_M-expert-pool.bin` | expert pool built from the **InternScience GGUF** | **DO NOT USE** - weights corrupt (see WORKLOG) |
| `Sage-A1.5-Q4_K_M-expert-pool-v0128.bin` | pool rebuilt from the reconverted GGUF | pending T4 rebuild (scripts/colab_v128_run.sh) |
| `Sage-A1.5-Q4_K_M.gguf` | reconverted GGUF from HF safetensors | pending T4 rebuild (scripts/colab_v128_run.sh) |

## IMPORTANT - the corrupt-weights warning

`InternScience/Sage-A1.5-Q4_K_M-GGUF` (HF + ModelScope) is corrupt: `ssm_conv1d`
garbage past a fixed 66 KB boundary per layer, `ssm_a` stride-2 scramble
(`-exp(A_log)` mis-shuffled), `blk0 ssm_out` corr 0.05 vs HF safetensors.
Every engine version since v0.1 produced word soup on it. The engine-side
root causes (1-16) are fixed as of v0.12.8; the weights must be reconverted
from HF safetensors with the bundled converter before quality runs:

```bash
bash scripts/colab_v128_run.sh   # on a Colab T4: reconvert -> pool -> gate -> bench -> publish
```

Reproduce benchmarks with `bench/bench_v8.py` (pp512 / pp2048 / tg128,
protocol in BENCH_RESULTS_v012.md). v0.12 reference numbers on one Tesla T4:
pp512 ~995-1097 tok/s, pp2048 ~915-1320 tok/s, tg128 ~39-40 tok/s
(chunk-config dependent, medians of 3).
