# Architecture

SAGER is a single-model inference engine built around one observation: a
35B-parameter MoE with 3B active parameters is not a compute problem on a
T4 - it is a **storage problem**. The card has 16 GB; the model in Q4_K has
~18.5 GB of experts alone. But each token touches only 8 of 128 experts per
layer. Strata's answer (which we adopt and re-implement) is to treat expert
weights as a three-tier storage hierarchy and to make the residency decision
per-expert, not per-model.

## 1. Three-tier expert storage

```
tier 0  VRAM   hot experts (budget-driven, ~3900 of 6144 on a T4)
               + all attention/router/norm/shared/embedding/head weights
tier 1  RAM    every routed expert, pinned, mmap-copied at load
tier 2  SSD    n-gram cached experts (roadmap, v0.2)
```

- **Residency table** (`core/expert_cache.hpp`): per `(layer, expert)` an
  exponentially-decayed frequency counter (`tick` once per decode step,
  `tau ~ 2 min`). `rebalance()` fills the VRAM slab with the currently
  hottest experts; the table survives restarts via a JSON profile written
  every 10 minutes and at exit, so a long-lived server never re-pays the
  cold ramp.
- **Pointer tables** (`layer_internal.hpp`): after each rebalance the
  session rebuilds three device pointer arrays per layer (gate/up/out,
  `n_expert` entries each, null for non-resident). The MoE GEMVs consume the
  tables through `gemv_ptrs_q4_k_q8_f16out`, which maps `row -> (expert,
  subrow)` in hardware - all resident experts of a layer fuse into ONE
  launch per matrix.
- **Why rebuild copies everything**: a full recopy costs a few hundred MB
  over PCIe and happens off the decode path (every 4096 steps at most).
  Idempotence beats incremental bookkeeping.

## 2. The session loop and the graph contract

One decode token walks 48 layers, each of which is two halves:

```
DENSE (graph-captured):  rmsnorm -> QKV(q4) -> mRoPE -> KV append -> GQA
                         -> o_proj -> residual -> rmsnorm
MOE   (eager):           router top-k -> [hits: fused ptr GEMV | misses: CPU
                         pool] -> shared expert -> combine -> residual
```

The dense half is captured into a per-layer CUDA graph. The capture contract
(`kernels/cuda_ops.hpp` header comment) is the load-bearing trick: **every
value that changes between steps is read from fixed-address device blocks**
(`SeqParams{pos, n_tokens}` for positions/past-length, `d_pos_ids_` for
RoPE), never from baked launch arguments. Replay is therefore correct after
writing 8 bytes of params - no graph re-capture, no address patching.

The MoE half stays on the host because it is data-dependent: the router's
top-k changes per token, and the hit/miss split decides where each expert
runs. Hiding it inside the graph would freeze the wrong thing.

## 3. The CPU-pool contract (C1-C6)

`core/expert_source.hpp` documents the session/pool interface clause by
clause. These are not suggestions; the engine refuses at runtime rather than
misindex:

- **C1** fixed pinned staging - the activation buffer handed to the pool is
  the SAME allocation every call; no per-call allocation on the hot path.
- **C2** the CPU computes `y = W_e @ x` and MUST NOT apply router weights -
  `moe_combine_f16` on the device is the only weighted sum, making a
  CPU/GPU weight mismatch structurally impossible.
- **C3** results land in fixed pinned output slots; one async H2D moves them
  into the combine buffer.
- **C4** geometry is validated at bind time; a wrong expert geometry raises
  `kGeometryMismatch` instead of misindexing.
- **C5** the submit path is single-producer and lock-light: jobs append to a
  ring under one mutex, completions count atomically.
- **C6** misses for layer N are submitted the moment the router decides,
  while the GPU still works on hits + shared expert - the doorbell overlap.

## 4. Speculative decoding

Two drafters feed one verify window (`core/verify.hpp`):

- **MTP drafter** (`core/mtp.cpp`): Sage-A1.5 ships an MTP head
  (`num_speculative_tokens: 3` on the model card). The recurrence
  `cat = [emb(t); h] -> fc -> rmsnorm -> proj -> rmsnorm -> lm_head` proposes
  up to 3 tokens for the cost of two small GEMVs plus one head stream
  (~0.6 ms each on T4).
- **Suffix drafter** (`spec/suffix_drafter.hpp`): a 3-gram hash over recent
  context proposes up to 5 tokens for FREE. Agentic traces re-emit tool-call
  frames and JSON scaffolding verbatim, where this beats MTP on acceptance.

The verify window runs `[t, c1..cd]` through ONE batched eager forward;
candidates are accepted iff the model's own greedy choice equals them, so
output is **bit-identical** to plain decoding. A cost-model controller
(`spec/controller.hpp`) gates drafting with
`E[gain] = A(d) * d * t_decode - (d * t_mtp + E[t_miss] + t_sync)`, where
`A(d)` starts from Strata's published priors and is learned per session
(`observe()`). Speculation never slows the session down on this model.

## 5. Defensive systems design

The engine prefers loud, early, greppable failures over silent corruption:

- GGUF geometry is sanity-checked against the training envelope
  (`validate_dims`); absurd files raise `kGeometryMismatch`.
- The loader refuses Q6_K attention tensors (the VRAM carve is Q4_K-sized);
  the head must be Q4_K in v0.1.
- The quantizer clamps codes to `[0,15]` BEFORE nibble packing (negative
  codes wrap - a bug class the host tests guard explicitly).
- `BlockQ8K::bsums` is int16 because a 32-value int8 sum reaches +-4064;
  the tests catch the wrap.
- Scale pack/unpack round-trips are property-tested over 1000 random
  (d, m) pairs.

## 6. What we deliberately did NOT build (v0.1)

- No tensor-core int4 path by default (`SAGER_T4_INT4_TC=1` flag reserved).
- No SSD tier (roadmap v0.2 - the RAM pool already fits 35B Q4_K).
- No batched MoE prefill (per-token dispatch; see performance.md for the
  honest prefill numbers and the v0.2 plan).
- No multi-GPU, no paged attention, no continuous batching - the T4 target
  is single-stream latency first.
