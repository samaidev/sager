# SAGER WORKLOG — v0.12 word-soup investigation (Sage-A1.5-Q4_K_M, Tesla T4)

Chronological root-cause chain for the v0.11/v0.12 word-soup regression, kept so
future sessions don't re-derive it. Each fix is verified by the hook named in
brackets; the quality gate is `bench/v12_quality.py` (MANUAL text review — the
letter/CJK heuristics can mis-score word soup).

## Root causes found & fixed

| # | commit | bug | verification |
|---|--------|-----|--------------|
| 1-4 | v0.12.0/v0.12.1 | Q4_K nibble order in pool requant; fused per-head q\|gate deinterleave; GDN conv stride; MTP stash | unit tests + S1 corr |
| 5 | a3b5735 | deinterleave per-head q\|gate of the fused q_proj | S1 |
| 6 | d999de7 | q8 KV scale bytes stored via `__half`→`uint16` implicit truncation → scale always 0 → attention output exactly zero (both attention paths) | attn.out sum 0 → 2.9052 |
| 7 | 5120e0c | `d_wtc_`: dense TC dequant scratch overlapped the MoE TC fp16 expert ring; pump stream-all queueing clobbered the first 8192 rows of every qkvz dequant | S1 q/k/v corr 0.003 → 0.9996 |
| 8 | v0.12.4 (this commit) | **shared-expert combine weight (`route_w_[row][K]`) never written on the device paths.** prepare() seeded only row 0; the v0.6 graph-warmup memset zeroed row 0 again; `router_topk*` wrote only kk<K; device prefill never uploads `route_w_`. Net effect: decode steps AND device-prefill rows >= 1 combined the shared expert with weight 0/garbage — silently dropping it from the residual stream. The host-batched path (moe_dispatch_batched) was the only correct one (wts[K]=1.0 per row), which is why SAGER_EXPERT_PUMP=0 / SAGER_PF_BATCHED=0 changed behavior and why every earlier fix "did nothing" for the gate. | router kernels now own the slot (out_w[K]=shared_w every launch); prepare() seeds all RT rows; moe increment at L0 should jump from ~0.4 to healthy +10..40 |

## Diagnostic notes (do not re-litigate)

- Pool file (`Sage-A1.5-Q4_K_M-expert-pool-v012-fixed.bin`) is byte-correct vs
  GGUF (corr=1.0); hot slab == pool; pointer table slot=e, off=e*peb correct.
- "Frozen x_row across binaries/configs" observations were partly artifacts of
  reading stale `/content/dmp/*` dumps after OOM-crashed runs; treat any
  cross-run byte equality with suspicion unless the run provably completed.
- GPU-vs-CPU whole-engine bisection is impossible (no non-CUDA dense path);
  use numpy per-stage references (GGUFReader + deq) instead.
- Remote exec quirk: functions inside payloads cannot see module-level names —
  every function must take everything via parameters / re-import internally.

## Pipeline

pool build (engine cold-fill, ~37 min on 2-vCPU Colab) -> manual quality gate
-> bench (bench_v8.py, c512/c2048) -> package (scripts/publish_v12.py)
-> GitHub release (binaries) + ModelScope/HF (pool + GGUF-scale artifacts).
