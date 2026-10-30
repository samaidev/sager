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

## Root causes 13-14 (v0.12.7 / v0.12.8) and the ACTUAL soup cause

| # | commit | bug | verification |
|---|--------|-----|--------------|
| 13a | 2258280 | `moe_dispatch_device` passed `d_ptr_table_ + t*lstride` WITHOUT `+ layer*E` to the grouped GU/down GEMV + TC dequant launches. Table layout is `[t][Lm][E]`; every decode layer >= 1 resolved residency and expert weights from LAYER 0's slice. L0-only numpy checks passed, everything else was garbage. | shape audit + per-layer sums |
| 13b | 2258280 | eager mailbox mirror `h_mail_n_ <- d_mail_n_` copied right after the scan (which zeroes `*mail_n`) - the grouped GU kernel adds the mails AFTER that, so the host always read 0 and `resolve_cpu_mailbox` never patched on eager (46/46 steps mails=0). Moved the handshake after the down GEMM. | diag10b rerun: mails 34-70/step |
| 12b | 2258280 | `absent_cnt_` still indexed the flat table `[L][3][E]` (diag9 fixed `L*E+e` to the wrong layout); the table is `[t][Lm][E]`. | code audit |
| - | 2258280 | prev-session uncommitted: `moe_combine_add_slots` grid.y=n_list RMW race (lost adds) - serialized per thread. | code audit |
| - | 2258280 | the L3 dense-layer diag block does SYNCHRONOUS memcpys - illegal during stream capture; with SAGER_DUMP_LOGITS=2 it killed the L3 full-layer capture and silently downgraded to eager. Guarded with cudaStreamIsCapturing. | diag10b: "captured 40/40" |
| 14 | f042558 | GDN q/k RMSNorm weights (C-GDN-1) never applied - kernels did bare L2; lin_q_norm/lin_k_norm loaded by nobody. Sage-A1.5 has NO linear q/k norms (bare L2 correct); kernels now branch on null weights for checkpoints that have them. | checkpoint key audit |
| - | 092bd0c | converter: raw bf16 safetensors reader (numpy cannot materialize BF16), header-shape reshape. | end-to-end |

### THE WORD SOUP WAS BAKED INTO THE WEIGHTS (the real root cause)

The GGUF we ran since v0.1 (downloaded from
InternScience/Sage-A1.5-Q4_K_M-GGUF, NOT produced by our converter) is
corrupt, verified against the HF safetensors with HTTP range reads:

- `ssm_conv1d` (f32, no dequant ambiguity): elements [0..16896) match HF
  verbatim, everything after is garbage - the SAME 66 KB boundary in
  blk0/1/2.
- `ssm_a` (f32): `gg[i] = -exp(hf_A_log[2i])` - a stride-2 scramble,
  reproducible across layers (wrong decay rates).
- `blk0 ssm_out` (q4_K): corr 0.05 vs HF under direct AND transposed
  layouts.

The engine and every numpy cross-check read the same corrupt file, which
is why L0 "verified" (corr 0.9999) while every version since v0.1 produced
soup (BENCH_RESULTS_v04 already said "the real-model pipeline has never
been coherent"). Fixes 1-14 were real bugs but none of them was THE soup.

Fix path: reconvert from HF with our own converter (5853ebb + follow-ups:
streaming write + in-flight HF shard release - the 67 GB bf16 + 21 GB
GGUF + 18 GB pool do not fit a 113 GB Colab disk otherwise). The real
checkpoint layout differs from what the converter assumed:
`linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, norm}` -
rebuilt into the engine's fused `[q|k|v|z]` / `[b|a]` carves at convert
time; A is stored pre-negated (`-exp(A_log)`).
