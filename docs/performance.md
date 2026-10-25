# Performance

What SAGER optimizes for, what to measure, and the honest v0.1 state.

## The three regimes

1. **Decode (1 token, 48 layers)** - bandwidth-bound on the hot experts:
   ~3B active params * 0.5 B/param ≈ 1.5 GB of weight traffic per token;
   at the T4's ~300 GB/s effective, the floor is ~5 ms/token. The kernels
   are built to sit near that floor: fused dequant-GEMVs, one launch per
   matrix group, per-layer CUDA graphs to strip launch overhead
   (~48 x 10 us saved per step), q8 KV to halve attention traffic.
2. **CPU-miss decode** - 1-3 experts from RAM, ~0.2-0.4 ms each, overlapped
   with GPU hits (C6 doorbell). A 20% miss rate costs well under 10%.
3. **Prefill** - compute-bound in theory, but v0.1 dispatches MoE per token
   (the router decision is data-dependent). Dense parts run batched; the
   per-token MoE loop is the prefill bottleneck. Chunked at 8192 with
   client-abort honored between chunks.

## What bench/bench.py reports

| Metric | Meaning | Healthy on T4 |
|---|---|---|
| `prefill tok/s` | prompt ingestion | hundreds (see limitation below) |
| `decode tok/s` | steady-state generation | 30-60 plain; 1.5-1.8x with healthy speculation |
| `cache hit rate` | routed requests served from VRAM | >90% after warm-up |
| `spec accepted/drafted` | verify-window acceptance | 1.8-2.6 tokens/pass on agentic text |

## Speculative decoding economics

The controller (`spec/controller.hpp`) drafts only when

```
E[gain] = A(d) * d * t_decode - (d * t_mtp + E[t_miss] + t_sync) > 0
```

With `t_decode ~ 30 ms`, `t_mtp ~ 4 ms` per draft token and Strata-prior
acceptance `A(3) ~ 0.50`, a depth-3 window costs ~12 ms + sync against an
expected ~45 ms of decode - a clear win. When acceptance sags (highly
creative sampling, cold cache), the learned prior crosses the threshold and
drafting turns itself off. `observe()` feeds every window back into `A(d)`.

## Honest v0.1 limitations

- **Per-token MoE prefill.** Dense prefill is batched; the MoE half walks
  tokens one at a time (router sync per token). Long prompts pay for it.
  v0.2 plan: batch the router GEMV across the chunk, D2H all top-k at once,
  group tokens by hit-set, and run the fused ptr-GEMV per group.
- **Single-stream serving.** One request owns the session (graphs bake
  addresses). Concurrency comes from multiple engine processes; a
  multi-session scheduler is roadmap.
- **Head is Q4_K-only.** Q6_K heads are refused at load (carve-size safety).
- **Greedy verification.** Sampling-aware verification (rejection-sampling
  on the draft distribution) is the standard upgrade path and is roadmap.

## Measurement methodology

Run `bench/bench.py` after a 10-minute warm-up (or with a persisted
`SAGER_PROFILE_PATH`), 5 rounds of 4K-prompt / 256-generated. Report the
median, and always pair decode tok/s with the cache hit rate - a decode
number without residency context is meaningless for a tiered engine.
