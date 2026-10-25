# SAGER v0.4.0-dev — device-side expert dispatch + the coherence bug hunt (Tesla T4)

## What landed (all on top of v0.3.0, commit 95f8588)

### 1. P1: device-side MoE dispatch for the streamed prefill walk (SAGER_PF_DEVICE=1, default on)

The v0.2.3 batched walk synced the whole stream once per layer (40
`stream_.sync()` per chunk), did all routing bookkeeping on the host, and
uploaded 7-8 job-table arrays per layer. v0.4 moves the plan onto the device:

- one `moe_dispatch_scan` kernel per layer: histogram (256 bins) + row/tile
  prefix sums + grouped scatter + bounds validation + an absent-expert
  mailbox - all on device;
- the chunk pointer table `[3][n_moe_layer][E]` is built at chunk start from
  RESIDENCY ONLY (no routing needed); hot-slab and parked-arena rows hold
  real pointers, streaming rows start null;
- each pump loader PATCHES its pair's table row right after the H2D copy
  (same stream) and records a per-plan-pair publish event - failure or skip
  records the event too, so `stream_wait_layer` can never hang;
- the compute stream waits device-side on exactly the current layer's pair
  events (Strata's "next layer's experts arrive while the current layer's
  attention runs" - zero host blocking);
- `stream_consume_layer` frees the ring slots deterministically after each
  layer's down GEMM; the per-pair `stream_lookup` blocking wait is GONE from
  the batched path;
- absent experts (loader failure) contribute 0 and are patched by the AVX2
  CPU pool after the layer via `moe_combine_add_slots_f16` (off the hot
  path; healthy runs never pay for it);
- the residency decay is fed by a chunk-level touch drain (one async D2H per
  layer, consumed at the chunk boundary) instead of per-route host touches.

Correctness: token-stream equality vs the v0.3 host dispatch (SAGER_PF_DEVICE=0)
holds - 4/4 interleaved A/B runs on Sage-A1.5-Q4_K_M produced identical ids
(48 seeded-sampled tokens each). ctest 3/3 (incl. the stale
`build_stream_plan` policy test updated to the v0.3 MISS-bitmap semantics).

Measured (Colab T4, 2 vCPU, 12.7 GB RAM, vram_mb 9984, chunk 2048, pool warm):
pp512 / pp2048 / tg128 are UNCHANGED in the warm state (~122-124 / ~118-119 /
~21.5 tok/s both arms) - the profile shows why: the Q4_K mtok GEMV kernels
are ~89% of GPU time and run 20-50x below their bandwidth roofline, so the
removed host overhead is hidden behind the kernel wall. The P1 win shows on
cold/contended states (first-run pp512: 125 vs 99.6 = +25%) and it is the
prerequisite for the whole-step decode graph (D2).

### 2. THE mtok kernel x-block indexing bug (real bug, found + fixed)

`gemv_q4_k_q8_f16out_mtok` (and all five mtok variants) staged each token's
`nb` activation blocks at a ONE-block stride in shared memory
(`stage_bytes(x_sh + t, ..., x_row_bytes)`), so the copies overlapped, and
the inner loop read `xb = x_sh + t` - activation block 0 only - while the W
side iterated all `nb` blocks. For nb > 1 every W block was multiplied
against activation block 0. A host-side C++ simulation of the exact kernel
logic confirms the divergence (128/128 mismatches, rel err ~300x vs the
reference W@x).

Fixed: non-overlapping staging (`x_sh + t*nb`) and per-(t, blk) activation
blocks (`xb_row[blk]`) in all six mtok kernels. NOTE: this changes model
outputs vs the v0.2.2/v0.3.0 binaries (which were internally consistent but
mathematically wrong for the dense/GDN/MoE k=2048 and k=512 projections).

### 3. The deeper finding: the real-model pipeline has never been coherent

With the kernel fix in place, generation on Sage-A1.5-Q4_K_M is STILL
degenerate (repeated single tokens, both SAGER_MTOK=0 and =1, with and
without ChatML). The engine's own GGUF KV dump (added in this release)
points at the root causes, in order:

a. **Tokenizer/vocab mismatch.** The GGUF carries a 248320-token gpt2-BPE
   tokenizer (`tokenizer.ggml.pre = qwen35`), while the serving layer
   (serve/tokenizer.py) and every bench so far used a 151936-token
   tokenizer.json - a DIFFERENT id space. Prompts encoded with the wrong
   tokenizer are noise to the model; all text-path "correctness" evidence
   to date (including v0.3.0's token-equality gates) compares arms sharing
   the same wrong input.
b. **mrope sections.** The GGUF has `rope.dimension_sections` with FOUR
   entries (i32 x4); the engine defaults to three ({11,11,10}) and style=2
   (mrope) is active for this model.
c. **GDN time_step_rank.** `ssm.time_step_rank = 32` has no counterpart in
   EngineDims/GdnParams (ba projection shape is derived from head counts);
   needs an audit against `linear_*` tensor shapes.

None of these can be confirmed/denied by the token-equality gates used so
far - they need a text path that uses the GGUF's OWN tokenizer.

### 4. Profiling facts for the next optimization round (nsys, 512-token prefill)

- Q4_K mtok GEMV kernels: 88.8% of GPU time (dense 50.8%, MoE grouped
  35.5%); the inner loop unpacks 6-bit scales per (row, token, sub-block) -
  ~86 instructions per 32 weights of which dp4a is 8. Loop-reorder (hoist
  the scale unpack out of the token loop, accumulate per token) is
  bit-exact-preserving and the next big prefill lever.
- gdn_scan_chunk: 8 ms/layer x 30 layers = 6% - second target.
- The streamed plan is ~65 pairs per chunk at 99% router hit rate - expert
  streaming is NOT the prefill wall on real routing; the kernel wall is.

## Reproduce / gates

```bash
python3 bench/bench_v023.py A            # v0.3 numbers (SAGER_PF_DEVICE=0 semantics)
SAGER_PF_DEVICE=1 python3 bench/bench_v023.py A   # device dispatch
ctest --test-dir build                    # 3/3
# coherence probe (requires the GGUF-native tokenizer - see (3a) above)
```
