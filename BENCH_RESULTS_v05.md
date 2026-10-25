# SAGER v0.5.0 — mtok loop-reorder + whole-step device decode (Tesla T4)

## Headline numbers (Sage-A1.5-Q4_K_M, Colab T4, 2 vCPU, 12.7 GB RAM, vram_mb 9984, chunk 2048, pool warm)

| metric | v0.4.0-dev | v0.5.0 | gain |
|---|---|---|---|
| prefill pp512 | 136.0 tok/s | **211.0 tok/s** | **+55%** |
| prefill pp2048 | 130.0 tok/s | **201.0 tok/s** | **+55%** |
| decode tg128 | 22.43 tok/s | **24.22 tok/s** | +8% |
| token-stream gate | — | **OLD == NEW, 128/128 ids IDENTICAL** | bit-exact |

Baseline re-measured on the same box/binary with
`SAGER_MTOK_REORDER=0 SAGER_DECODE_DEVICE=0 SAGER_DEVICE_SAMPLER=0`
(v0.4 paths); v0.5 numbers are the same binary with everything on (defaults).
ctest 3/3 before both arms.

## What landed

### 1. P2: mtok GEMV loop-reorder — the W-side state hoisted out of the token loop (bit-exact)

The v0.4 profile showed the Q4_K mtok kernels at ~86 instructions per 32
weights of which only 8 are dp4a: every `(row, token, sub-block)` iterated
`unpack_scales6` (~13 ALU) and re-extracted/re-packed the nibbles into dp4a
operand words (~64 ALU) — both depend on W only, but were recomputed per
token. v0.5 precomputes per `(row, lane-group g)` state ONCE into registers
(`MtokGState{sd, sm, wp[8]}`) and the token loop keeps only
`8×(LDS + dp4a) + 2 FMA` per sub-block. The t-outer/g-inner structure is
unchanged, so every `(row, token)` still accumulates its sub-block terms in
ascending-g order and the same warp-shuffle reduction runs per token —
bit-exact by construction, confirmed by the 128/128 token-stream equality
gate above (and by SAGER_MTOK_REORDER=0 A/B).

New kernels: `gemv{,2,grouped,grouped2,grouped2_tab,grouped_tab}_mtok_ro_kernel<NG>`
(templates over NG = ceil(nb/4) ≤ 4, i.e. k ≤ 4096; larger k falls back to
the original kernels). `SAGER_MTOK_REORDER=0` selects the originals.

### 2. P3/D2-lite: device-dispatch decode — the per-layer host sync is gone

v0.4's decode walked 48 × `[graphLaunch + moe_dispatch_row]` where the row
dispatch pulled the route to the host and `stream_.sync()`-ed EVERY layer
(pipeline drain ~48×/step, ~600 launches, per-layer host planning).
v0.5 routes decode through the v0.4 device dispatch (`moe_dispatch_device(L,1)`):
one route sync per step instead of 48; the residency pointer table
(`build_decode_table()`: hot slab ∪ parked pump-arena slots, rebuilt lazily)
serves the `_tab` GEMVs; absent experts zero + mail and the CPU pool patches
them at the next boundary (stream-ordered, unchanged v0.4 protocol), then the
experts are promoted into the arena at the step end (GPU idle → race-free
slot steals) so the next step hits them in VRAM. Also fixed the n=1 grid
blow-up: `tiles_max` is now bounded by the routed-row count (the old
`Rmax/tile + E + 16` bound launched 94% empty blocks at 1 row), and the three
per-step synchronous H2D copies go through pinned staging instead.

`SAGER_DECODE_DEVICE=0` keeps the v0.4 per-row walk.

### 3. Device top-k sampler (decode path)

`sample_host` costs ~2.7 ms/token at n_vocab 248320 (full-vector copy +
nth_element on 2 vCPUs) plus a ~1 MB logits D2H per step. The decode step now
runs penalties + iterative max+mask top-k on device (`topk_select_f32`, one
block, k ≤ 32 head-room) and `Session::finish_device_sample` finishes
softmax/temp/top-p/min_p/multinomial on the host over the 32 candidates with
the same rng discipline. `SAGER_DEVICE_SAMPLER=0` restores the host sampler.

## Decode status / next lever

tg128 moved 22.4 → 24.2 tok/s. The step is now ~41 ms of almost-pure GPU
small-kernel time: the one-row dense/GDN GEMVs captured in the per-layer
graphs run at low occupancy (the v0.4 note's ~12%) and the eager MoE half
adds ~14 launches/layer. The next decode lever is folding the MoE half into
the capture (whole-step graph) on top of the P1 device dispatch, and/or
multi-token (batch-2/4) decode. Prefill-class mtok kernels already serve the
MoE half, so batched decode is a natural follow-up.

## Reproduce

```bash
python3 bench_v05.py all      # OLD arm + NEW arm + token gate
ctest --test-dir build        # 3/3
```

Env switches: `SAGER_MTOK_REORDER`, `SAGER_DECODE_DEVICE`, `SAGER_DEVICE_SAMPLER`
(all default ON on sm_75).
