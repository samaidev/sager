#!/bin/bash
# colab_v8_run.sh - v0.8 build + parity gates + T4 benchmark, one shot.
# Designed to run on the Colab T4 box via the bridge (nohup + log polling).
#
#   GHTOKEN=github_pat_xxx bash colab_v8_run.sh 2>&1 | tee /content/v8_run.log
set -uo pipefail

GHTOKEN=${GHTOKEN:?need GHTOKEN}
BRANCH=${BRANCH:-v0.8-opt}
REPO=/content/sager
POOL=/content/Sage-A1.5-Q4_K_M-expert-pool.bin
MODEL=/content/Sage-A1.5-Q4_K_M.gguf

echo "== $(date -u +%F_%T) v0.8 remote run start =="
nvidia-smi | head -4 || true

if [ ! -d "$REPO/.git" ]; then
  git clone "https://$GHTOKEN@github.com/samaidev/sager.git" "$REPO"
fi
cd "$REPO"
git config user.email "agent@localhost"
git config user.name "agent"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git rev-parse HEAD | tee /content/v8_commit.txt

# --- model + expert pool (skip-if-present, resumable) ---
if [ ! -f "$MODEL" ]; then bash scripts/dl_sage.sh; fi
if [ ! -f "$POOL" ]; then bash scripts/dl_sage.sh --pool-only || bash scripts/dl_sage.sh; fi
ls -la "$MODEL" "$POOL" || true

export SAGER_EXPERT_POOL_FILE="$POOL"
export SAGER_EXPERT_POOL_REUSE=1
export BENCH_MODEL="$MODEL"
export BENCH_POOL="$POOL"

# --- build v0.8 ---
echo "== build v0.8 =="
cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build build -j "$(nproc)" 2>&1 | tail -8 || {
  echo "BUILD FAILED"; exit 1; }
ctest --test-dir build --output-on-failure 2>&1 | tail -6

# --- parity gates + bench (chunk 512: decode-heavy, matches v0.7 table) ---
echo "== bench chunk 512 =="
BENCH_CHUNK=512 python3 bench/bench_v8.py --json /content/v8_c512.json \
  2>&1 | tee /content/bench_c512.log

echo "== bench chunk 2048 =="
BENCH_CHUNK=2048 BENCH_ARMS=0 python3 bench/bench_v8.py --skip-parity \
  --json /content/v8_c2048.json 2>&1 | tee /content/bench_c2048.log

# --- optional: readers=3 arm (prefill streaming probe) ---
echo "== readers=3 probe =="
BENCH_CHUNK=2048 BENCH_ARMS=0 SAGER_PF_READERS=3 python3 bench/bench_v8.py \
  --skip-parity --json /content/v8_c2048_r3.json 2>&1 | tee /content/bench_c2048_r3.log

# --- package artifacts ---
echo "== package =="
cp BUILD_INFO.txt /content/BUILD_INFO_v8.txt 2>/dev/null || true
cat > /content/BUILD_INFO_v8.txt <<EOF
SAGER v0.8.0 T4 CUDA build info
==================================
repo:        samaidev/sager
branch:      $BRANCH
commit:      $(git rev-parse HEAD)
built on:    Colab Tesla T4 (sm_75, 15360 MiB)
toolchain:   $(nvcc --version 2>/dev/null | tail -1) / cmake $(cmake --version | head -1) / $(g++ --version | head -1)
flags:       cmake -B build -DCMAKE_BUILD_TYPE=Release
model:       Sage-A1.5-Q4_K_M.gguf
env:         SAGER_EXPERT_POOL_FILE=$POOL SAGER_EXPERT_POOL_REUSE=1
EOF
tar czf /content/sager_v08_t4_cuda.tar.gz \
  -C "$REPO/build" sager_engine sager_test_q4 sager_test_gdn sager_test_pump \
    libsager_core.a libsager_cuda.a \
  -C /content BUILD_INFO_v8.txt v8_c512.json v8_c2048.json v8_c2048_r3.json \
    bench_c512.log bench_c2048.log bench_c2048_r3.log \
  -C "$REPO" BENCH_RESULTS_v08.md 2>/dev/null || \
tar czf /content/sager_v08_t4_cuda.tar.gz \
  -C "$REPO/build" sager_engine sager_test_q4 sager_test_gdn sager_test_pump \
    libsager_core.a libsager_cuda.a \
  -C /content BUILD_INFO_v8.txt v8_c512.json v8_c2048.json v8_c2048_r3.json \
    bench_c512.log bench_c2048.log bench_c2048_r3.log
ls -la /content/sager_v08_t4_cuda.tar.gz
sha256sum /content/sager_v08_t4_cuda.tar.gz | tee /content/v8_tarball_sha256.txt
echo "== $(date -u +%F_%T) v0.8 remote run done =="
