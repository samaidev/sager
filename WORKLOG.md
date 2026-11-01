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

## Session 2026-10-31 (VM reset -> full re-derive): roots 15-20

Colab VM was recycled mid-hunt. Rebuilt from scratch (repo@diag-kv, T4 build,
model re-download) and re-ran the whole ladder. Three NEW fatal bugs found,
each masked by the next one down the stack:

| # | commit | bug | verification |
|---|--------|-----|--------------|
| 15 | 768e503 | load blockers for converter GGUF: (a) `#include "sager/core/log.hpp"` typo (file is `sager/log.hpp`), (b) `GPE()` macro without trailing `;` x3, (c) `lin_k_norm` required by an else-branch although Sage-A1.5 has no GDN q/k norms (root 14 said kernels branch on null, the loader didn't). | engine loads converter GGUF |
| 16 | 768e503 | shared-expert lookups used hardcoded `ffn_*_shexp.weight` while the converter writes `ffn_*_shexp`; switched to `find_tensor_w` (both spellings accepted). | load passes L0..L39 |
| 17 | 95b3e10 | **THE converter wrote the data section without aligning it to `general.alignment` (32)** while every tensor offset in the directory assumes an aligned data_start. Every weight read was shifted by (32 - header_len % 32) = 4 bytes. corr vs HF was -0.05; after pad corr=1.000000 on every probed tensor (A_log -exp relation exact, dt_bias bitwise, conv1d 1.000000, lm_head rows 11/12 >0.99). **This is the actual soup cause in the v0.1 GGUF lineage too**: any engine read of a converter-produced file was garbage-but-self-consistent (the pool copies what it reads, so numpy cross-checks agreed with the engine - corr 1.0 against the same wrong bytes). | tools/conv_verify-style numpy audit vs HF shard 00012 |
| 18 | 5d8c3ca | attention per-head q/k norms: engine looked up `attn_q_norm.weight`, converter writes `attn_q_norm` -> never loaded -> rmsnorm fed uninitialised VRAM -> q/k activations exactly 0 on every full-attention layer. | attn.q/k sum 0 -> 22.3/21.4 after fix |
| 20 | e398f6e | GDN q/k norm POINTERS were unconditionally dmalloc'd in alloc() (comment even says "full-attention layers"), so `lw.lin_q_norm/lin_k_norm` were non-null with never-written VRAM for a checkpoint that has no such tensors. The L2-norm kernel's `if (qw)` RMSNorm branch multiplied by zeros: qn/kn exactly 0 -> scan output exactly 0 -> **every GDN layer contributed nothing since v0.1**. Alloc now leaves them null (kernels branch on null per root 14); load dmallocs on demand when a checkpoint does carry them. | qn/kn 0 -> 69.6/94.0, scan_y 0 -> 0.715 (prefill) and 1.06-5.78 per decode step |
| - | 5de501f | dbg: f32 row writer overread 4x (16B/item on 4B arrays) -> segfault with SAGER_DBG_DUMP_ROW. | fixed, but a new segfault appeared after it (see open items) |

### Verified-correct chain (numpy vs engine, clean weights)
embed->rmsnorm->qkvz GEMV corr=1.000000; causal conv (ch-major [ck x conv_dim]
taps) t0 corr=0.998 fp16-accurate; gate ab = {exp(A*dt), sigmoid(b)} sane;
L2 qn/kn healthy; scan (gated delta rule) nonzero and token-varying; MoE
patch path (gpatch) corr=1.000000/0.999999/1.000000; L3 attention q/k/v/out
all live after root 18. Prefill: all 40 layers post/moe nan=0.

### Open items (next session)
1. **Segfault in eager prefill right after the L3 dbg block** (attn.oproj dbg
   prints, then engine EOF, no stderr). Repro: graphs=false,
   SAGER_DUMP_LOGITS=2, any generate. Started after roots 18/20 changed
   values; may be a latent race exposed by new timing. Note: graphs=true runs
   DO NOT crash (diag10b completes 48 tokens).
2. **Residual growth**: decode x_after (L0) grows 118 -> 1149 over 48 steps
   (~+20/step). Either a real scale bug left (root 21?) or downstream of
   something still off. Prefill L39 post is ~700 with per-layer increments
   that look large but finite.
3. **step_logits.bin dump reads NaN** while the live pipeline shows nan=0
   everywhere - dump path suspicious, not necessarily the model. Greedy still
   picks tokens, so the NaN is probably a dump artifact; needs a proper
   logits cross-check (dump x_final row + numpy head GEMV vs sampled token).
4. Text is still soup-like (paired repeats + multilingual). All layer dbg is
   healthy; next differential is decode-path numpy cross-check (conv state
   roll, scan n=1) and the L3+ MoE/attention decode path.


## Session 2026-10-31 II (5th VM rebuild): root 21 + full numpy verification chain

VM recycled again (5th tunnel). Full recovery: clone@diag-kv, T4 build, HF
shards 0-12 re-download (69.8 GB), converter rerun -> Sage-fixed.gguf
(18.91 GiB, 633 tensors, CONV_EXIT=0), pool cold-build. CUDA toolchain had
to be restored from pypi.nvidia.com wheels + NVIDIA apt repo (nvcc 13.0.88,
libnvvm for cicc, cudart/cublas/culibos-dev) after the disk-crunch deletions.

### Verification chain (numpy vs engine, converter GGUF, all PASS)
embed row sums bit-exact; attn_norm/ffn_norm/lin_o_norm corr=1.000000 vs HF
shard 00012 (the small norms ARE the trained values); L0 GDN full-prefill
31-token numpy reference matches every stage (qkvz 0.26%, conv 0.66%, scan
0.7%, o_proj 2.7%); L3 attention full numpy (neox rope, GQA h//8, gated)
corr=1.0000 all 31 rows, norms 0.99998; KV slab == fresh k/v (corr 1.0);
patch path (pool AVX2) corr=0.998; head GEMV corr=1.000000 top-5 identical.
Prefill top-1 token is 'alise' -> soup starts in prefill (not decode-only).

### Root 21 (fixed, 0d733e6)
rmsnorm_quant_f16_kernel had NO row offset: grid = n_rows blocks but every
block reduced/wrote x[0..n). Decode (n_rows=1) was fine, so it hid through
every decode-focused test. The MoE ffn-norm calls it with n_rows=n tokens ->
rows 1..N-1 of xn_/d_xq_ stale -> MoE input garbage -> every layer's MoE
collapsed. Pre-fix direct delta |moe-post| at L0 row 30 = 0.117 (expected
~5.8); post-fix = 3.58. Text STILL soup after the fix.

### Where the hunt stands (next session)
1. y-slab probe (c418f1d) says all 31 rows x 9 slots are populated with
   healthy-magnitude values (shared 3.5-4.6, experts 0.5-1.1) but a few
   slots show ~7.0 outliers in varying slots/rows - verify per-slot y against
   numpy with SELF-CONSISTENT pairing (probe #1 dumps y for the prefill but
   w_routew/routeidx get overwritten by probe #2, the n=2 window; gate the
   dbg_write_row_f32 hooks on the same counter as the probe).
2. If y content verifies, the remaining suspects are the combine route_w_
   row mapping and the decode-path conv state roll (open item from roots
   15-20 list).
3. SAGER_STEP_GRAPH=0 does NOT de-graph the MoE dumps (layer graphs still
   capture); the batched-path probe is the working window into real MoE.

## Session 2026-10-31 III (6th VM, tunnel UA43ZAGZ): attention chain fully verified; no new engine root cause; IMA = pump/warmup interaction

VM died with the previous tunnel mid-MoE-verify. Full rebuild (clone@diag-kv,
T4, HF shards 69.8 GB re-download, converter with **--fast** [the slow path
per-sub-block grid search never finishes - ~2.5 h wasted without the flag],
Sage-fixed.gguf 20.32 GB = 18.91 GiB CONV_RC=0, pool 16.9 GB fresh fill).

### Verified clean (numpy vs engine dumps, this VM, clean weights)
- q4k.py numpy Q4_K mirror == gguf-py official dequantize (corr 1.0, bitwise).
- L3 attention FULL chain, all 31 prefill rows, corr **1.000000** per row:
  xin(residual) -> attn_norm -> fused q|gate GEMV with **per-head interleave**
  (head h owns q rows [h*2*hd, h*2*hd+hd), gate rows [h*2*hd+hd, (h+1)*2*hd))
  -> per-head RMSNorm (f16 store) -> NeoX partial rope (theta 1e7, 64 dims,
  pairs (j, j+32), positions 0..30) -> GQA h//8 -> causal softmax (1/16)
  -> q8 KV slab (kv_append matches fresh k/v, corr 0.99996; slab layout:
  16 groups x [2B f16 scale | 32 int8], scale FIRST) -> sigmoid gate mult
  -> o_proj -> residual. [ATT] with kernel-exact slab dequant = 1.000000.
- v0.11 flash-tile prefill attention is **CORRECT** (an earlier "progressive
  per-row corruption" was the numpy reference feeding pre-quantization k/v;
  with the slab dequant the flash-tile path is bit-exact too).
- LM head + decode logits: corr 0.99999641, top-12 overlap 12/12. The
  step_logits NaN from earlier sessions was an IMA-poisoned-run artifact
  (clean runs: nan=0).
- Engine ctest 4/4 on the T4 (incl. cuda_q4_stack) after the diag11 patch.

### NEW BUG (not the soup): IMA at capture warmup with pump ON
`capture_graphs()` warmup launches dense/GDN layers + the MoE device warmup
(moe_dispatch_device(0,1) with zeroed routes, stream ring not armed) ->
"capture warmup left error: an illegal memory access" -> sticky error poisons
TC prefill (dp4a fallback, all-zero tokens in one run) and kills some runs.
Workaround that keeps the engine usable: `SAGER_EXPERT_PUMP=0` (pump off ->
graphs capture fine, everything eager-clean). Did NOT reproduce on the 5th VM
(reused pool) - fresh-pool fill vs reuse is the suspected delta. Fix idea:
guard the warmup MoE device call on stream_active()/loader_count, or arm the
ring before warmup.

### Consequence for the MoE hunt (IMPORTANT, cost us an hour)
`moe_dispatch_batched` is gated on `pump_ != nullptr` - with the pump off the
prefill MoE silently runs the **per-row walk** (moe_dispatch_row), so the
batched probe never fires on pump-off runs, and the earlier "pump-off clean"
runs were exercising the ROW path, not batched/device. The batched/device MoE
paths remain numpy-UNVERIFIED; the per-row path is now the best-covered one
(it produced the same soup, so the soup lives in SHARED code: router,
expert GEMVs, combine weights, or upstream GDN/residual).

### Tooling (pushed to branch diag11-probe = 7562972)
- diag11 engine patch: self-consistent batched-MoE probe - per call dumps
  batched_{ridx,rw,rlogits,xpre,y,xn}_<probe>.bin locked to the same static
  counter (fixes the probe #1/#2 pairing ambiguity from c418f1d).
- q4k.py (numpy Q4_K/Q6_K/Q8_K mirrors, validated vs gguf-py), l3_verify.py,
  head_verify.py, moe_verify.py (metadata-driven, probe-id arg), rexec.py
  (Samai envelope executor), payloads/ with probe/setup/chain scripts.
- Colab quirks re-learned: pypi nvidia-cuda-nvcc wheel 13.0.88 has a broken
  cicc/ptxas pairing - use nvidia-cuda-nvcc==13.4.92; converter REQUIRES
  --fast; payload functions cannot see module-level imports; python -u for
  all drivers; driver stdin JSONL must use chr(10) not escaped \\n literals.

### Next session
1. conv_verify: MY GGUF vs HF shard 00012 (script ready; q/k/v/o/router/
   shared/expert0-1 row corr) - the --fast converter output has NOT yet been
   checked against HF directly.
2. Re-run with SAGER_PF_DEVICE=0 + pump ON (no EXPERT_PUMP=0): batched probe
   fires, run moe_verify 1/2 -> the combine/route_w verdict. If the IMA
   bites, fix the warmup first (small guard patch).
3. Then GDN L1+ / residual-growth / prefill-logits 4x-scale dump timing.

## Session 2026-11-01 (7th VM, tunnel 2TZVJCB2): ROOT 22 = zero-centered RMSNorm

VM was fresh again (6th tunnel's VM gone; /content empty). Full rebuild
launched (diag-kv @ c437ab7, build+ctest 4/4 clean, shards re-downloaded,
converter --fast rerun). While it ran, the hunt went STATIC against the
ground truth - the transformers reference implementation
(models/qwen3_5_moe/modeling_qwen3_5_moe.py, fetched from HF transformers
main) plus HTTP-range reads of the actual checkpoint values.

### Reference audit - everything else verified CORRECT
- RoPE: text path = NeoX half-split pairs (j, j+32), partial 64/256, theta
  1e7 (recomposition_frequencies -> cat(freqs,freqs) + rotate_half; the
  mrope_interleaved section mix only affects vision grids). Engine matches.
- Full attention: per-head q|gate interleave, sigmoid output gate, GQA
  h//8, softmax 1/16 fp32, q_norm/k_norm (per-head 256) BEFORE rope.
- GDN: conv over fused q|k|v (8192 ch) + SiLU; q/k weightless l2norm;
  v-head j uses k-head j//2 (engine scan: vh = h*2 + j/V - identical to
  repeat_interleave); g = -exp(A_log)*softplus(a+dt_bias), beta =
  sigmoid(b); delta rule s <- a*s - a*b*(s.k)k + b*v*k, y = s.q (FLA-exact);
  o = RMSNormGated_w(o) * silu(z) (STANDARD w*x - engine gdn_out_norm
  matches).
- MoE: router softmax -> topk8 -> renormalize-to-1 (engine has it);
  shared expert gated per-token by sigmoid(shared_expert_gate.x) (engine
  layer.cpp matches); silu(gate)*up experts.

### ROOT 22 (commit 9e7b9df remote / aa3c298 local)
Qwen3_5MoeRMSNorm is ZERO-CENTERED (Gemma-style): gain = (1 + w), weights
init to zeros. Checkpoint values prove it (HTTP range reads, shard 00012):
  layers.0.input_layernorm   mean +0.031 std 0.049  (all |w| < 0.35)
  layers.3 q_norm / k_norm   mean +0.33 / +0.31
  final model.norm           mean +1.63 std 0.23
  linear_attn.norm (o_norm)  mean +0.88  <- RMSNormGated, STANDARD w*x
The engine's rmsnorm kernels multiply w directly, so EVERY normalized
activation ran at ~0.03-0.33x gain since v0.1: attention logits crushed
~1000x (uniform attention), MoE inputs crushed, final logits ~1.6x short.
This is THE word soup. It survived every numpy cross-check because the
numpy references encoded the same w*x mistake (engine-vs-my-numpy = 1.0
corr against the same wrong formula).
Fix: copynorm() in weights.cpp bakes +1.0 into the uploaded fp16 weights
(attn_norm / ffn_norm / attn_q_norm / attn_k_norm / output_norm all load
through it). lin_o_norm (RMSNormGated) is NOT touched. Kernels unchanged
(graph-safe); GGUF stays faithful; pool unaffected (norms are not experts).

### Ops notes this session
- tunnel 502 EOF: a payload that raises SystemExit kills the Flask handler
  thread -> relay reports 502. Use RuntimeError in payloads. Related trap:
  `git add` succeeded on a broken patch before `git commit` failed (no
  identity), so `git checkout -- file` restored from the INDEX, not HEAD;
  use `git reset --hard HEAD` when re-patching.
- Tooling (q4k.py etc.) from session III was NEVER in git (only the
  session.cpp engine patch was pushed on diag11-probe). Rewrite as needed;
  gguf reader + Q4_K dequant are being rebuilt locally from the converter
  contract.

## Session 2026-11-01 II (7th VM cont.): root 23 + full verification sweep

Root-22 fix deployed (remote commit 9e7b9df) and the quality gate STILL
produces soup (heuristic PASS, manual read = soup; raw completions: 2/3
prompts emit EOS immediately). The hunt then went exhaustive:

### conv_verify (GGUF vs HF, HTTP ranges) - WEIGHTS FULLY CLEAN
All f16/f32 anchors bit-exact (norms, dt_bias, A_log->-exp, conv_w,
router, embedding rows incl. token alignment); all Q4_K surfaces
cos 0.989-0.996 (lm_head, fused [q|k|v|z] all four carves, [b|a],
out_proj, shexp, attn q/k/v/o, experts of L0/L20 x4 experts x3 mats).
The --fast encoder is EXONERATED on real data.

### moe_verify (diag11 batched probe vs GGUF reference) - MoE CHAIN CLEAN
Router: top-8 sets + renormalized weights EXACT (5/5 tokens).
ffn_norm with the root-22 (1+w) fix: cos=1.000000 live.
Routed expert GEMV chain (Q8 quant -> grouped gemv -> silu -> requant ->
down -> scatter): cos 0.9996-0.9998 across all 8 slots.
Combine: cos=1.000000, maxdiff 1e-4.
EXCEPT the shared expert: implied gate = 0.5031 = sigmoid(0).

### ROOT 23 (commit 80b624a): converter never wrote
blk.L.ffn_gate_inp_shexp.weight (HF mlp.shared_expert_gate.weight).
The engine dmallocs lw.shared_gate_w UNCONDITIONALLY and silently skips
the load when the tensor is absent -> uninitialised VRAM (fresh pages
read 0) -> every layer's shared expert ran with a CONSTANT 0.5 gate.
Fix: converter writes it (f32 [2048]); the existing GGUF can be patched
by append-surgery (scripts/gguf_shgate_surgery.py - offsets are
data-section-relative, the data blob moves as a whole, so old offsets
stay UNCHANGED; first attempt wrongly added the dir-growth delta).
CAVEAT: the true gate vectors are tiny (L0 std 0.0063 ->
sigmoid ~ 0.5 +/- 0.07), so the constant-0.5 may be nearly right -
root 24 may still be at large. The surgically-fixed GGUF must be
quality-gated to decide.

### Ops
- Tunnel 2TZVJCB2 died ~02:52 mid-surgery (7th tunnel death). VM state
  unknown; the corrected surgery may or may not have completed
  (/content/Sage-fixed.gguf.fixed).
- repo state: local sandbox has root-22 + worklog + 4 validate_quant/
  make_synthetic_hf fixes + root-23 (unpushed - no GitHub PAT in this
  session); the dead VM held 9e7b9df + d4d208f likewise unpushed.
- Engine-side tooling this session (in the sandbox scripts/ dir):
  conv_verify.py, moe_verify.py, attn_verify.py (L3 layer differential,
  ready to fire), gguf_shgate_surgery.py, norm_stats.py, cmdbus.py.
- NEXT: (1) complete surgery + quality gate (root-23 verdict);
  (2) attn_verify on the existing attn_L3_* dumps;
  (3) GDN layer differential with real inputs; (4) full 40-layer numpy
  forward vs step_logits if the per-layer diffs stay clean.

## Session 2026-11-01 III (8th VM, tunnel 3RRLBA3M): root 24 - the word-soup master cause

Fresh-VM full rebuild with root-22+23 (Sage-fixed.gguf regenerated, shgate
tensor verified present, engine rebuilt) STILL produced soup (heuristic
gate 3/3 "PASS" is meaningless - manual read = soup).

### L3 full-attention differential (attn_verify_v2, file-layout fixed)
q/k/v/gate projections ALL cos=1.000000 vs GGUF reference (post norm+rope);
KV q8 cache clean (0.99998); attention OUTPUT wrong for tok1..4
(cos 0.13-0.42) while tok0 is perfect - single-position softmax masks any
score corruption. Hypothesis grid (temperature/masking/GQA/shifts/uniform)
matched nothing -> kernel-level probe.

### Isolated kernel replay (attn_probe.cu)
Replaying the dumped q/kv-cache through attention_prefill_f16 reproduces
the engine output EXACTLY (per-token cos 1.000000) -> the engine faithfully
runs attn_prefill_v2 and the KERNEL is wrong on the same inputs numpy gets
right. Instrumented copy (attn_probe2/3) printed the smoking gun:

  kernel qr[0..7] = [16848, 48256, 13440, ...]
  raw uint16 q[4,0,c*32] = [16852, 48269, 13441, ...]  (= f16 bits of 2.91, -1.14, ...)

### ROOT 24: __half2float(q[i]) on uint16_t STORAGE
attention.cu:232 (attn_prefill_v2) and :346 (attn_decode_partial) passed the
uint16_t ELEMENT to __half2float, which takes __half - the implicit
uint16_t->__half constructor converts the INTEGER VALUE (16852 -> half
16852.0), not the bit pattern. Every q register became a 5-digit integer;
scores q.k collapsed to ~-1e5 garbage; softmax degenerated to a
single-position pick; every full-attention layer emitted near-random
mixtures -> word soup since v0.11. tok0 (and any single-token softmax)
looked correct, which is why decode-only spot checks and the earlier
"flash-tile correct" verification missed it. Staging used
reinterpret_cast (correct); the v0.10 fallback kernel used ldh() (correct);
a whole-codebase audit found NO other __half2float(<raw uint16>) sites.

Fix: qr[c] = ldh(q + q_off + c*32 + lane) in both kernels.
