# The Expert Pump: per-token RAM -> VRAM streaming (v0.2)

## The problem it solves

v0.1's miss path (hot-cache miss -> `CpuPool` computes `W_e @ x` on the
host) is fine for a couple of misses, but on a weak host (2 vCPU, e.g.
Colab T4) the CPU pool IS the decode bottleneck - the same wall llama.cpp
hits with whole-layer offload (measured: 3.45 t/s decode with 10/40 layers
on CPU, 93% of per-token time in those layers). The pump treats an expert
as DATA instead: one async DMA of ~1.7-2.7 MB from the pinned RAM pool into
a VRAM arena slot, computed on the GPU, and KEPT for later reuse.

## Design

```
                 submit(layer,e)                     arm_wait(compute)
  router top-k ────────────────► ExpertPump ───────────────────────► GEMV
                                   │  ▲
                    cudaMemcpyAsync│  │event
                                   ▼  │
  pinned RAM pool (tier 1) ──► copy stream ──► VRAM arena (tier 0b)
  (Weights::host_expert_base,     (dedicated cudaStream_t)   (LRU slots)
   page-locked by the loader)
```

- **Arena**: `cp_.pump_arena_mb` (default 1536 MB) of VRAM carved into
  `per_expert_bytes` slots (~880 at 1.7 MB, ~580 at 2.7 MB). Expert layout
  inside a slot matches the hot tier: `[gate | up | out]` Q4_K.
- **Residency is hierarchical** (C7c): hot tier first (rebalance-driven
  tables), then pump arena, then CPU pool. The pump never talks to the hot
  tier; `moe_dispatch_row` ORs the two.
- **Reuse**: submitting an expert that is already in the arena returns the
  same slot with NO copy (temporal locality across tokens: agentic
  workloads re-route to the same experts constantly - this is the same
  observation Strata's hot-cache exploits, but at token granularity).
- **Eviction**: clock hand for free slots, then LRU among slots whose copy
  has COMPLETED (`cudaEventQuery`). An in-flight slot is never torn (C7b);
  if nothing is evictable the caller falls back to the CPU pool for that
  expert - same generation, same result.
- **Pipelining** (C7a): `submit()` only enqueues on the pump's own copy
  stream and returns; `arm_wait()` drops event-waits into the compute
  stream right before the pumped group's GEMV launches. The copies
  therefore race ahead while the GPU computes the hot hits + shared expert.
- **Single producer** (C7d): all pump state is touched from the session
  thread, like the rest of the MoE dispatch. No locks on the hot path.

## Dispatch flow (per token, per MoE layer)

```
router top-k (device) -> D2H ids/weights (pinned, existing)
for each routed expert k:
    hot tier hit   -> compact ptr table (unchanged v0.1 path)
    pump resident  -> reuse slot, compact pump ptr table
    else           -> pump.submit() -> slot; on -1 -> CPU miss job (v0.1 path)
hits:  fused ptr-GEMVs (v0.1, unchanged)
pump:  arm_wait -> fused ptr-GEMVs against arena pointers (same kernels)
miss:  CpuPool doorbell (v0.1, unchanged)
combine: moe_combine_f16 over all routed slots + shared expert
```

The activation snapshot (`d_xq_moe_`) exists because the hits' out-GEMV
overwrites the shared `d_hq_` scratch before the pumped group runs.

## Why GPU-compute-the-miss beats CPU-compute-the-miss here

| | CPU pool (v0.1) | pump (v0.2) |
|---|---|---|
| per-miss latency | ~0.3-1 ms on 2 weak cores | ~0.2-0.3 ms PCIe (1.7 MB @ ~8 GB/s) |
| warms future tokens | no | **yes** (stays in arena) |
| scales with misses | linear in cores | bandwidth-bound, overlaps |
| host CPU load | full GEMV+dequant | zero |

The CPU pool stays as the correctness fallback and for pathological
routing patterns (all-miss storms before the arena warms).

## Toggles

- `expert_pump` (engine request key) / `SAGER_EXPERT_PUMP=0|1` - default on.
- `pump_arena_mb` / `SAGER_PUMP_ARENA_MB` - default 1536.
- `Session::reset()` calls `pump_->clear()`: residency is conversation-
  tied; stats are lifetime counters.

## Unit tests (no GPU needed)

`tests/test_expert_pump.cpp` exercises the policy half on a CPU-only build
(copies no-op, events presumed complete): reuse identity, LRU eviction
order, stale-residency invalidation, clear() semantics, tiny-arena
fallback, and a churn stress (16 experts through 4 slots).

## v0.3 hooks

The SSD tier plugs in exactly here: `find_victim_locked()` is where an
n-gram-prefetch source (roadmap: "the profile already knows what will be
needed") would replace the RAM pool as the copy source.
