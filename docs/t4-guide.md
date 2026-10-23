# T4 Deployment Guide

Everything needed to run Sage-A1.5 (35B-A3B, Q4_K) smoothly on one Tesla T4.

## Memory budget

The whole design fits 16 GB:

| Block | Size | Notes |
|---|---|---|
| Attention + router + norms + shared experts | ~0.67 GB | 48 layers, Q4_K mats + fp16 norms + fp32 router |
| Embedding | 0.60 GB | 151936 x 2048 fp16 |
| Output head | 0.17 GB | Q4_K (v0.1 requirement) |
| KV cache (q8, 16384 ctx) | 0.85 GB | 48 x 2 x 16384 x 16 groups x 34 B |
| Hot-expert pool | 10.0 GB | `--vram-mb 10240` |
| Scratch (activations, Q8 slabs, logits) | < 0.5 GB | sized by `chunk_prefill` |
| CUDA context + fragmentation | ~0.7 GB | fixed |
| **Total** | **~13.5 GB** | 2.5 GB headroom |

One expert is `3 x (768 x 2048)` Q4_K blocks = **2.52 MB**. The 10 GB pool
holds **~3970 experts (65%)**; each extra GB adds ~400 experts.

## Why 65% residency is enough

Agentic workloads have strong expert locality: tool-call framing, code and
JSON generation re-activate the same expert subsets. Measured on comparable
Strata deployments, a 60-70% hot pool covers **>90% of routed requests** once
warm - and the persisted usage profile (`SAGER_PROFILE_PATH`) means "warm"
survives restarts. The remaining ~10% of requests take the AVX2 RAM path for
1-3 experts, which the doorbell overlap hides behind GPU work.

## Runtime knobs (serve/server.py or the `load` command)

| Flag | Default | When to change |
|---|---|---|
| `--vram-mb` | 10240 | Raise to 11200 only if you also cut `--ctx`; watch for OOM on prefill spikes |
| `--ctx` | 16384 | 8192 for long-batch safety; q8 KV keeps 16K cheap |
| `--chunk` | 8192 | Lower to 4096 on 8 GB cards; activation slabs scale with it |
| `--threads` | auto (cores/2) | Pin down if the pool steals cycles from the H2D copy thread |
| `--kv-fp16` | off | Only for debugging KV numerics |
| `--no-graphs` | off | Diagnosing graph issues; costs ~10-15% decode speed |
| `--no-mtp` | off | If acceptance < 40% on your workload, the controller already backs off - but this saves the draft entirely |

## Sage-A1.5 specifics baked into the engine

From the model card (already the defaults in `core/config.hpp`):

- mrope sections `[11, 11, 10]` with partial rotary factor 0.25
  (`rope_dims = 128 * 0.25 = 32`), `rope_theta = 1e7`
- sampling: `temperature 1.0`, `top_k 20`, `top_p 0.95`, `min_p 0`,
  `presence_penalty 1.5`
- MTP: `num_speculative_tokens = 3` (`--mtp-max-draft`)
- context: 256K native; 16K is a T4 memory choice, not a model limit

## Converter requirements

- The HF snapshot must contain the MTP tensors if you want speculative
  drafting (`mtp.fc.weight`, `mtp.proj_out.weight` - fp16 in the GGUF).
- The output head is emitted as Q4_K (v0.1 engine contract; the validator
  flags anything else).
- Attention/shared/router mats are Q4_K/Q4_K/fp32 respectively - the loader
  refuses other dtypes rather than misindexing.

## Host requirements

- **RAM**: ~19 GB free (pinned expert pool 15.5 GB + mmap + working set).
  32 GB machines are comfortable; 64 GB if you co-run anything.
- **CPU**: AVX2 + FMA. The pool pins `cores/2` workers by default; with
  DDR4-3200 dual-channel, a miss costs ~0.2-0.4 ms per expert.
- **PCIe**: the hot-tier rebuild moves a few hundred MB every 4096 decode
  steps - negligible on x16, but avoid x1 slots.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `kOomVram` at prepare | pool + KV too big | `--vram-mb 9000` or `--ctx 8192` |
| `kOomRam` at load | pinned pool > free RAM | close other jobs; pool needs ~15.5 GB |
| hit rate < 70% steady | cold profile + unusual traffic | keep `SAGER_PROFILE_PATH` set; it learns in ~10 min |
| decode < 20 tok/s | misses dominating | check `cache_hit_rate` in `/props`; raise pool, lower ctx |
| verify acceptance ~0 | highly stochastic generation | expected with temp >> 0; controller already compensates |
