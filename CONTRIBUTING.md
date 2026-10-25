# Contributing to SAGER

Thanks for looking at SAGER. The project has a few strongly-held invariants;
keeping them is what makes the engine safe to extend.

## Invariants (please do not break)

1. **The C1-C6 clauses** in `core/expert_source.hpp`. If your change touches
   the session/pool boundary, re-read them. C2 (CPU never applies router
   weights) is structural correctness, not style.
2. **The graph contract**: per-step values must arrive through fixed-address
   device blocks (`SeqParams`, `d_pos_ids_`), never through captured scalar
   launch arguments. If your kernel reads a host-computed scalar, it does not
   belong in the dense graph.
3. **Q4_K bit layout**: byte-compatible with the standard K-quant pack. Any
   change must keep `tests/test_q4_host.cpp` green - the scale pack
   round-trip, the clamped-code nibble packing, and the int16 `bsums` guard
   encode real past bugs.
4. **Fail loudly**: prefer `kGeometryMismatch` / `kGgufParseError` over
   silent misindexing, everywhere.

## Workflow

```bash
./setup.sh                     # build + host tests
ctest --test-dir build --output-on-failure
```

- Host tests must pass on any machine (no GPU needed). CUDA kernels are
  reviewed against the reference (`src/kernels/q4_k.cpp`) since CI GPU
  runners are optional.
- Python changes: `python3 -m py_compile serve/*.py tools/*.py bench/*.py`.
- Docs: `docs/performance.md` numbers must be reproducible with
  `bench/bench.py`; if you claim it, bench it.

## Code style

- C++20, 2-space indent, `snake_case` for members with trailing `_`,
  `kCamelCase` for constants, 100 cols.
- Comments explain WHY (contracts, invariants, measured costs) - the code
  already says what.
- Python: no mandatory linter yet; keep the existing style.
