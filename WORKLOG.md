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

## Session 2026-11-03 IV (9th VM, tunnel E5MC8O94): root 24 verified fixed but soup persists -> root 25/26 (artifact persistence)

Fresh-VM full rebuild (v3 driver: diag-kv + session patch r22+r23+r24, ctest
4/4, SHGATE_TENSOR_PRESENT=True, pool cold-fill) STILL produced soup in the
quality gate (all three prompts incoherent CJK/EN token salad). Root 24 was
real but not the whole story.

### Layer-1 differential ladder (full_ref.py + l0_fine*.py, 5-tok prefill)
- embed row: cos=1.000000; L0 GDN internals (r_proj/r_conv/r_scan): cos=1.000000
  (all 5 tokens) - delta-rule scan, causal conv, qkvz projection exact.
- L0 residual r_post: eng-cos=0.91770 (FIRST divergence); L1 0.59, L2 0.53.
- l0_fine2 per-(token,head) cos matrix: on and y are 5x32 ALL 1.000000 ->
  gdn_out_norm_chunk_f16 kernel EXACT; w_on device bits == GGUF bits.
- Clean-on o recompute: o_eng vs on_ref@Wout.T cos=0.9168, rel-err 62%,
  |eng o| 39% too large, diff argmax 1269, ALL 16 dim-segments polluted ->
  the lin_out GEMM consumes corrupted weights. PF_TC=0 (dp4a) reproduced the
  SAME statistics -> GEMM kernels innocent; device weight bytes innocent for
  rows 0-7 (blind box: device == GGUF byte-exact, TC dequant scratch cos=1.0).

### ROOT 25: the GGUF on disk is not what the engine loaded
Full NaN-scale scan (every Q4_K block's d/dmin f16 bits): ~2% of blocks carry
NaN-pattern scales spread over EVERY big tensor (lin_out, attn_*, ffn_*_exps,
output.weight; 371 tensors) - yet the engine (loaded from the same file 40
minutes earlier) produced NaN-free outputs and cos=1.0 internals. md5 of the
region was stable across reads AND matched the v8-era extract, i.e. the disk
content had been garbage since long before. First bad block = row 20; raw
bytes are high-entropy junk, not Q4_K. quantize_q4_k_fast cannot emit
mantissa-nonzero sNaN d bits -> the NaN blocks are garbage bytes on disk that
the converter never wrote.

### ROOT 26: hf_transfer downloads are not fsynced; evicted pages resurface as garbage
- converter write() had no fsync either; on Colab's overlay disk the written
  pages lived in the page cache only. Every read 07:55-10:26 hit cache and
  looked perfect (hence all cos=1.0 differentials); after eviction the disk
  showed junk. pool.bin tail showed the same 2.26% NaN fingerprint.
- Control: single shard downloaded with HF_HUB_ENABLE_HF_TRANSFER=0 +
  os.sync() -> md5 stable, disk good.
- fsync-only rebuild (v4) still failed the NaN gate (371 tensors): the shards
  themselves were downloaded with hf_transfer (no fsync) and the converter
  quantized garbage bytes into NaN scales deterministically.

### Fixes
- tools/convert_hf_to_sager.py: f.flush() + os.fsync(f.fileno()) before
  write() returns.
- Rebuild driver v5: HF_HUB_ENABLE_HF_TRANSFER=0 for the shard download,
  os.sync() at every persistence boundary (after download, after convert,
  after pool fill), and a hard NaN-scale gate (nan-ratio must be 0.0 over
  every Q4_K tensor / pool sample) before the quality gate can run.

## 2026-11-04 session (post tunnel-loss): v6 driver staged, awaiting VM

### VM loss confirmed
Both tunnels offline (`E5MC8O94` and `PXY314IH` -> "Tunnel offline" from
aitun.cc /health). The Colab VM behind them was recycled: the v4 rebuild
(converter shard 8/13 in flight) and all /content assets (tree, GGUF,
pool, shards) are gone. Memory reads at 10.8% via the new tunnel had
already suggested a fresh VM.

### Loss is smaller than it looks
WORKLOG post-mortem (recorded just before the loss): the fsync-only v4 was
DOOMED anyway - it still failed the NaN gate with 371 damaged tensors
because the SHARDS themselves were hf_transfer downloads (never fsynced),
and the converter quantized garbage bytes into NaN scales deterministically.
v5's remedy (HF_HUB_ENABLE_HF_TRANSFER=0 + os.sync() at every persistence
boundary) was designed but the VM died before it could run end-to-end.

### Remote base pinned
`git ls-remote origin diag-kv` -> c437ab7 (local origin/diag-kv ref was
stale at 86a5ddc; c437ab7 is its descendant and exactly our patch base).

### v6 driver (launch_rebuild_v6.py, local scripts/) - READY, not yet run
Merge of v3 + v4 + v5, one-shot for a FRESH VM:
  S1  clone -> checkout c437ab7 -> apply patch (roots 22+23+24+26 in one
      diff; fsync is COMMITTED locally in eda63d7, so NO runtime anchor
      patch - v4's would assert-fail) -> grep-verify all four root markers
  S2  shard download with HF_HUB_ENABLE_HF_TRANSFER=0, os.sync() inside the
      downloader before SNAP_OK
  S3  cmake build + ctest (recorded)
  S4  wait for shards + os.sync() + NEW shard health gate (f16 nan-ratio
      < 1e-4 over 3x4MB samples per shard) - catches root-25 BEFORE the
      40-60min converter run instead of after
  S5  os.sync() -> converter --fast (fsync baked in) -> sync -> size gate
      + SHGATE_TENSOR_PRESENT + full Q4_K NaN scan (DAMAGED_TENSORS=0)
  S6  disk cleanup (keep shard 00012) -> pool cold-fill PUMP=0 -> sync ->
      pool size gate -> pool NaN gate (POOL_GATE_OK) -> v12_quality output
      (+ v3's direct-load fallback fill if PUMP=0 leaves no pool)
Local verification: bash -n OK; 14/14 strategy checks; patch 15262 bytes
with all four ROOT markers. Default tunnel PXY314IH (AITUN_CODE overrides).

### Next
1. User restarts the Colab VM + tunnel client (tunnel code may change).
2. `AITUN_CODE=<code> python3 scripts/launch_rebuild_v6.py`, then poll
   `/content/rebuild6.log` via fetch_log.py.
3. Gates green -> root-26 case closed -> re-run bench (pp512/tg128) ->
   clean tree (`git checkout`) for the release build -> publish.
4. Quality gate still soup -> re-run the o-vs-ref differential on the
   clean data (o cos=0.917 / 62% err was measured on poisoned data).

## 2026-11-04 (later): ROOT 27 - the NaN gates were scanner false positives

### The v6 run
Fresh VM (tunnel IQD7BK9A), v6 driver: s1 patch(4 roots)+build+ctest 4/4 clean,
s4 shard gate 14/14 clean (hf_transfer OFF), s5 converter RC=0 (18.91 GiB,
673 tensors) -> NaN gate FAILED with DAMAGED_TENSORS=371 - the SAME count
as the v4 run built from (allegedly garbage) hf_transfer shards.

### The contradiction that cracked it
Clean shards in, fsync'd write, sync before scan - yet identical 371. Both
quantizers write d == dmin (same f16 bits twice: `block[:,0:2]=d16;
block[:,2:4]=d16`), so ANY correctly-aligned scan must report bad_d == bad_m.
v4/v6 reported 2241 vs 2251, 733 vs 792, 39748 vs 38572 -> the scan was
reading bytes the converter never wrote as scales.

### Root cause
gguf_scan_nan.py computed `DATA = (f.tell()+31) & ~31` AFTER the KV loop but
BEFORE the tensor-directory loop - i.e. data_start = align32(end-of-KV),
~len(td) bytes short of the real data section. Every tensor read landed
inside the directory/f16/f32 region: "NaN scales" were names/dims/weight
bytes at a fixed wrong offset. Deterministic per model -> identical 371
across runs/VMs/shard sources. validate_quant.py (fixed in bf70353) always
parsed in the correct order - which is why it validated cos ~0.995 while
the NaN scanner screamed.

Local proof (scripts/scan_check.py + /tmp/synth.gguf synthetic chain):
- broken order: d_ne_dmin=5,728,029 of 5,728,256 blocks, first "bad block"
  hexdump spells "ffn_down_exps" (directory bytes!), VERDICT A
- fixed order (DATA after td loop): d_ne_dmin=0, nan_d=0, VERDICT C_CLEAN

### Consequences
- The GGUF conversions were NEVER damaged. The root-25/26 "disk garbage"
  narrative was an artifact of this scanner bug (the engine's own C++ reader
  was always right - hence cos=1.0 internals "despite" disk NaNs).
- fsync discipline (r26) retained as hygiene, but it was never the soup cause.
- The S5 NaN gate blocked every rebuild since v4 for nothing; the S6 quality
  gate has NEVER actually run on an r22+r23+r24 build. Launched it directly
  (scripts/launch_s6_v2.py -> /content/s6run.log): fixed scan first, then
  pool cold-fill + v12_quality max-tokens 96 - the first real word-soup
  verdict on a fully-patched engine.

Fixes applied locally: scripts/gguf_scan_nan.py + scripts/scan_check.py now
compute data_start after the tensor directory.

## 2026-11-04 (final): v6 VERDICT on fresh VM (tunnel LAQUTFHI) - soup survives all four roots; S6 pool gate is another false positive

### Run timeline (all UTC)
13:59 start (T4 16GB, 113G free) -> 13:59:15 s1 done (patch e9f27a7 on
c437ab7, four markers) -> 14:03:37 s3 done (BUILD_RC=0, ctest 4/4) ->
14:11:44 s4 done (14/14 shards worst_f16_nan=0.000000, SHARD_GATE_OK,
hf_transfer OFF) -> 14:39:07 s5 done (CONV_RC=0, GGUF_SIZE=20301730496,
SHGATE_TENSOR_PRESENT=True, DAMAGED_TENSORS=0 with the root-27-fixed
scanner - first conversion EVER past the NaN gate) -> s6: quality ran,
QUALITY_RC=0, POOL_SIZE=18119393280 = exactly 10240 x 1769472
(3 mats x 589824B Q4_K per expert), then POOL_GATE_FAILED.

### ROOT 28 (candidate): pool gate sampled at unaligned offsets
The s6 pool gate seeks int(sz*frac) WITHOUT 144-byte block alignment.
Misaligned u16 pairs from Q4_K quant payload hit f16 NaN exponent bits at
~1/32 = 3.1% (observed 2.1-2.5% at frac 0.955-0.999). Re-probed at
144-ALIGNED offsets: ratio = 0.0 at every fraction 0.90-0.999.
The pool is CLEAN. Same bug class as root 27 (readers sampling at wrong
offsets). Fix when reused: off = (int(sz*frac)//144)*144.

### THE VERDICT: word soup SURVIVES roots 22+23+24+26
v12_quality 3 prompts (96 tok each), statistical gates pass (letter/CJK
0.86-0.89, 8-gram max run 1) but every output is soup, e.g. prompt
"What is the capital of France?":
  !历史股东RWRW"_tuples襲 single h0feefee#比普通,1emersawnacusò口的olntag伪...
CRITICAL PATTERN: the SAME token cloud recurs across all three prompts
(fee / 襲 / _tuples / 历史股东 / RWRW / single / domestic / prezi / casin /
awn / eder / 不見 / 国家一级 / second / listed / ó / ò) - generation is
dominated by a prompt-INDEPENDENT token distribution: hidden state barely
depends on input, or logits are drowned by a constant term.

### Where to hunt next (in order)
1. o-vs-ref differential on CLEAN data (planned in root-27 entry; previous
   cos=0.917 / 62% err was measured on "poisoned" data = untrustworthy).
   Layer ladder: full_ref.py / l0_fine*.py already exist locally.
2. GDN / linear-attention path: 30 of 40 layers are linear (gated delta
   net, conv_kernel=4, k_heads=16, v_heads=32, kdim=128, vdim=128,
   RMSNormGated o_norm). Root 24 only fixed the 10 FULL-attention layers
   (attn_prefill_v2 / attn_decode_partial). The GDN kernels have never
   been implicated or audited - and 30/40 layers is plenty to soup output.
3. Re-audit any remaining raw-storage -> __half conversions in the GDN
   kernels (root-24's audit covered attention.cu only).
4. Final logits path: output_norm (copynorm +1.0), output head Q4_K.

Artifacts on VM (tunnel LAQUTFHI): /content/Sage-fixed.gguf (clean),
/content/pool.bin (clean, aligned-scan verified), /content/rebuild6.log,
stage markers s1-s5 (s6 marker not set - gate bug, not a real failure).

## 2026-11-04 (audit round): full source audit vs the TRUE Qwen3.5 reference - all layer math VERIFIED, soup cause narrowed to runtime glue

Context: v6 verdict (soup survives r22-24/26) + tunnel LAQUTFHI died mid-audit.
Reference obtained: transformers main modeling_qwen3_5.py + modeling_qwen3_5_moe.py
(fetched raw to /tmp; Sage-A1.5 = Qwen3_5MoeForConditionalGeneration, hidden_act=silu,
attn_output_gate=true, full_attention_interval=4, tie_word_embeddings=false).

### Verified CORRECT (engine vs reference, line by line)
- GDN conv: applies to [q|k|v] only (conv_dim = 2*key_dim + value_dim), silu,
  z/b/a bypass. Engine carve [q|k|v|z] and [b|a] matches converter vstack.
- GDN gate: beta=sigmoid(b); alpha=exp(-exp(A_log)*softplus(a+dt_bias));
  converter pre-negates A_log correctly.
- GDN l2 norm: bare L2 (no weights) - correct, Sage-A1.5 has no linear q/k norm.
- GDN delta recurrence: engine s_new = a*s - a*b*(S k)k + b*v*k is ALGEBRAICALLY
  IDENTICAL to reference (decay-first + correction-on-decayed-state + un-decayed
  v-term). v-head sharing (2 consecutive v-heads per k-head) matches
  repeat_interleave. (Ref q *= 1/sqrt(K) is normalized away by o_norm.)
- GDN out: RMSNorm(y)*w_on*silu(z) per v-head = Qwen3_5RMSNormGated. Standard
  (non-zero-centered) w - root-22 correctly left it alone.
- Attention: q_proj is heads*hd*2 with PER-HEAD [q|gate] interleave (view
  (..., -1, hd*2).chunk(2)); engine v0.12.1 deinterleave + sigmoid_mul_rows
  matches exactly. Buffer layout [wq|attn_gate] contiguous carve verified.
- Attention norms: Qwen3_5RMSNorm zero-centered (1+w) on q/k per head = root-22.
- MoE router: softmax->topk->renorm == topk->softmax (Z cancels). SwiGLU silu.
- Shared expert: sigmoid(shared_expert_gate(h)) * shared(h) + routed. r23 tensor
  now written by converter.
- Decoder layer: standard pre-norm residual x + mixer(norm(x)); x + mlp(norm(x)).
- conv weight layout [D,ck] channel-major matches kernel w[d*ck+j]; converter
  reshapes [D,1,ck]->[D,ck].

### Remaining suspects (ordered)
1. RUNTIME GLUE, not math: decode CUDA graphs ("captured 40 per-layer decode
   graphs", graphs=on) - stale-state/pointer bugs on replay; dev_sampler=1.
   Test: graphs:false eager rerun (launch_eager96.py infra exists) + greedy
   first-token (check_logits.py vs numpy ref "Paris 18.572").
2. Kernels not yet line-audited: norm_rotary.cu (rope partial 0.25 theta 1e7),
   moe.cu routing execution, q4_k.cu dequant paths used by mmq/tc_gemm,
   sampling.cu.
3. Embedding / output head load path (fixed token cloud across prompts is
   consistent with constant-ish final hidden state OR scrambled head rows).

Note for next session: prior numpy references may encode the same engine bug
(root-22 lesson) - validate reference code against the fetched HF source, not
against the engine.

## 2026-11-07 (offline audit, tunnel UYJ23XUZ down): engine kernels ALL clean; two converter quantizer bugs found (root 30 + 31)

Context: tunnel offline the whole session -> full static audit of every kernel
the v6 prefill/decode path touches, plus the converter, done locally against
the verified HF reference semantics (1555d90).

### Verified CLEAN (line-by-line, this session)
- Q4_K whole stack: converter quantizers <-> q4_k.cpp dequant <-> CUDA gemv
  (`dp4a_subblock`, `(sb>>1)*32` group, even-LOW/odd-HIGH nibble) <-> mtok/mmqa
  family <-> TC dequant (`dequant_q4k_u4`) <-> cuBLAS row/col-major trick.
  nibble order differs from llama.cpp (LOW-first) but is self-consistent -
  irrelevant for self-converted GGUFs.
- Q8_K activation quant (CUDA): amax/127, int8 clamp, int16 bsums. Correct.
- attention.cu KV q8_0 append/read: v0.12.2 bit-cast scale fix in, symmetric.
- sampling.cu: penalties/softmax/top-k/top-p/LCG all correct.
- norm_rotary.cu: rope NeoX partial (style=0, theta 1e7, partial 0.25 from
  the v6 log), rmsnorm, sigmoid gates, embedding gather. Correct.
- gdn.cu ALL kernels (gate/conv/l2/delta/out_norm; chunk + warp + r4):
  recurrence, v-head sharing (vh = h*ratio + j/V), z-at-conv_dim offset,
  state layout. Matches the verified reference math.
- moe.cu: router softmax+greedy topk (tie rule), renorm, shared slot
  (root-8 fix in), combine, dispatch scan, scatter/gather. Correct.
- weights.cpp: copynorm +1.0 (root-22), q-gate per-head deinterleave
  (v0.12.1), GDN loads (qkvz/ba single-tensor copies), A_log pre-negation.
- converter carves: [q|k|v|z], [b|a], conv [D,ck], router, shexp, expert
  stacks, dims order (fastest-first), root-17 data-start pad. Correct.
- IMPORTANT deduction: the FIRST generated token is already soup (P3 starts
  走吧 vs P1/P2 "!") -> the bug is in prefill/head/weights, NOT decode graphs.

### ROOT 30 (converter, REAL, v6 NOT affected): non-fast quantizer wrote zero scales
quantize_q4_k_row called _set_scale(block[4:], ...) - a bytearray SLICE COPY:
the 12 scale bytes never reached the block. Any non---fast conversion
dequantizes to ALL-ZERO weights (local round-trip cos 0.00000). v6 used
--fast so this did not fire, but it is a landmine: FIXED by building `scales`
in its own buffer and copying into block[4:16]. Fixed-quantizer round-trip:
cos 0.9968 (gauss).

### ROOT 31 (converter, quality): --fast anchored quantizer is above the format floor
Local round-trip on weight-like synthetics (engine dequant formula):
  gauss:        fast relRMSE 0.0932 cos 0.99566
  outliers 1%:  fast relRMSE 0.1301 cos 0.99154   <- worst
  validate_quant's own operative gate is relRMSE<0.08 ("iid-gaussian floor");
  fast sits ABOVE the floor, and the v6 pipeline NEVER ran validate_quant
  (it was shipped in the patch set but not wired into any stage).
Added quantize_q4_k_qsearch: vectorized two-stage 5x5+5x5 (is,im) grid
search, chunked (1<<15 sub-blocks), selectable via --qsearch. Hits the floor
(gauss relRMSE 0.079 cos 0.9969; outliers 0.105/0.9945) at ~30x the speed of
the scalar search (512x2048 matrix: 613 ms -> ~157 s per expert stack).
NOTE: cos 0.9957 alone is NOT proven to word-soup a model - real-weight
validation on the actual v6 GGUF is still required (next VM session).

### VM plan when tunnel returns (ordered)
1. ls /content - did Sage-fixed.gguf / pool.bin / hf_sage survive?
2. python tools/validate_quant.py --gguf /content/Sage-fixed.gguf --hf
   /content/hf_sage --tensors all   (5 min) -> REAL cosine of the v6 artifact.
3. If real cos < ~0.993 on major classes: re-convert with --qsearch (~6 h)
   then rerun bench/v12_quality.py.
4. Eager isolation run: SAGER_STEP_GRAPH=0 SAGER_FULL_GRAPH=0
   SAGER_PF_DEVICE=0 SAGER_DECODE_DEVICE=0 graphs:false quality rerun (10 min)
   - discriminates runtime glue vs weights.
5. Still soup -> o-vs-ref layer ladder on clean data (full_ref.py family)
   against the fetched HF reference (beware root-22 lesson: references must
   be re-derived from modeling_qwen3_5.py, not from old engine mirrors).

## Session 2026-11-07 (tunnel 1LCGODPP): E-battery re-queued behind running v6 rebuild
Cron check found rebuild7.log/ebat2.log absent: UYJ23XUZ died; the VM now on
1LCGODPP is mid-rebuild6 (S5 converter running, GGUF 7.25/44.5 GB at 02:06 UTC).
Crucially this VM's rebuild6.sh still has the OLD unaligned S6 pool gate, so it
will false-FAIL (POOL_GATE_FAILED) and never write stage_s6_done (root-28).
Queued /content/ebat3_waiter.sh instead (launch_ebat3_waiter.py): waits for
pool.bin >= 17 GB with no writers, runs the root-28 ALIGNED gate itself, then
E1/E2/E3 (ebat.py verbatim) and E4/E5 (ebat2.py verbatim), marker
stage_ebat_done. Details + timeline: diag/2026-11-07-1LCGODPP-ebat3-queue.md.
Follow-up cron 510772 at 15:30 +08:00 pulls ebat3_waiter/ebat/ebat2 logs and
applies the verdict matrix.

## 2026-11-07 (v6-v2 on fresh VM, tunnel 1LCGODPP): root-31 CONFIRMED LIVE; runtime glue EXONERATED; qsearch chain launched

Reproduction: S1-S5 all green on the fresh T4 (patch commit eaeebff = c437ab7
+ roots 22/23/24/26/30/31). GGUF_SIZE=20301730496 == v6 byte-exact ->
deterministic artifact. Soup reproduced token-identically (3 prompts, same
token families, P3 starts 走吧 again). S6 pool-gate tail NaN (0.0244 @0.99) =
root-28 unaligned-sample FP again, everything else green.

### Discriminator 1 - validate_quant (extended this session)
_map_back previously validated ONLY expert stacks (attn_*/lin_*/output
returned None "skip quietly"). Extended: identity mapping for
attn_{q,k,v,output}/lin_out/shexp/output head, two-segment vstack mapping for
lin_in_qkvz (in_proj_qkv | in_proj_z) and lin_in_ba (in_proj_b | in_proj_a),
--include substring filter; bf16-safe HF row reader (numpy framework cannot
materialize bf16 -> torch fallback). Local unit test 12/12 (caught a missing
layer-id in the key builder pre-push).
REAL v6 numbers (32 rows/tensor, all classes):
  expert stacks   relRMSE 0.093-0.096  cos ~0.9956
  GDN qkvz/ba/out 0.099-0.118          0.9931-0.9952
  attn q/k/v/out  0.100-0.135          0.9909-0.9950 (worst blk.39.attn_v)
  output head     0.0983               0.9952
EVERY tensor above the 0.08 iid-gaussian floor; a healthy Q4_K_M should sit
at 2-5% (peaked distributions beat the floor). The --fast anchored quantizer
underperforms the format floor by 2-4x on real weights, exactly as the
offline synthetic prediction (0.093-0.130). ROOT 31 IS LIVE IN THE ARTIFACT.

### Discriminator 2 - eager isolation (graphs:false + STEP/FULL_GRAPH/PF/DECODE_DEVICE=0)
Output is TOKEN-IDENTICAL to the graphs-on S6 run on all three prompts
(P1 byte-equal: `!历史股东RWRW"_tuples襲 single h0feefee#...`), decode
1.2-2.0 tok/s vs 3.4-7.7 with graphs (expected slowdown, same stream).
-> runtime glue (decode graphs / device paths) EXONERATED. GDN decode
kernels personally re-verified clean the same session (gate/conv/l2/delta/
out_norm: alpha=exp(-exp(A_log)*softplus(a+dt)), beta=sigmoid(b),
S<-aS-ab(Sk)k+bvk, vh=h*ratio+j/V, bare-L2 branch; ldh() everywhere).

### Verdict + action
Only remaining suspect shared by both paths: WEIGHT quantization noise
(root-31). Launched the unattended qsearch chain 03:17:30 UTC
(/content/qchain.sh, log /content/qconv.log): wait shard topup ->
convert_hf_to_sager --workers 4 --qsearch -> Sage-qsearch.gguf ->
engine load fills pool_qsearch.bin -> v12_quality graphs-on.
Markers: /content/qconv_done, /content/qchain_done.
Local commits this session: 3eda4d3 (v12_quality --no-graphs), 0561509
(extended mapping+include), f410384 (layer-id fix), 56f3917 (bf16 reader).

### Tunnel/exec traps learned this session
- /execute is SERIAL (is_executing lock): a long payload keeps the lock and
  outlives local HTTP timeouts - check /health before assuming tunnel dead.
- Envelope bodies >~12KB arrive truncated (binascii padding error) - gzip
  files before embedding (validator 14KB -> 4.2KB gz).
- exec globals/locals separation (HANDOVER trap #1) still bites: functions
  must bind outer names via parameter defaults.
- The converter DELETES hf shards as it processes them (disk management) -
  post-pipeline validate needs a shard re-download (subset via weight_map).
