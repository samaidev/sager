#!/usr/bin/env bash
# setup.sh - configure + build + test SAGER (CUDA + AVX2 auto-detected).
set -euo pipefail

cd "$(dirname "$0")"

echo "==> checking prerequisites"
command -v cmake >/dev/null || { echo "cmake not found - apt/brew install cmake"; exit 1; }
if command -v nvcc >/dev/null; then
  echo "    nvcc: $(nvcc --version | tail -1)"
else
  echo "    WARNING: nvcc not found - building CPU-only (tests still run)"
fi

echo "==> configure"
cmake -B build -DCMAKE_BUILD_TYPE=Release

echo "==> build"
cmake --build build -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "==> test (host-side, no GPU required)"
ctest --test-dir build --output-on-failure || true

echo
echo "Done. Next steps:"
echo "  1. Convert a model:  python tools/convert_hf_to_sager.py <hf_dir> model.gguf"
echo "  2. Serve:            python serve/server.py --model model.gguf"
echo "  3. Chat (web):       http://localhost:8000"
