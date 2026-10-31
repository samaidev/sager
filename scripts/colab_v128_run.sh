#!/usr/bin/env bash
# colab_v128_run.sh - ONE-SHOT T4 (Colab) pipeline for the v0.12.8 release.
#
# Why: the InternScience GGUF (and every pool built from it) is corrupt
# (WORKLOG.md - word soup was baked into the weights). This script rebuilds
# the full artifact chain from the HF safetensors on a 113 GB Colab T4 VM:
#
#   1. build engine @ rel-v0.12.8 (+ ctest 4/4 on GPU, incl. cuda_q4_stack)
#   2. download HF safetensors (14 shards; MTP shard skippable for headroom)
#   3. reconvert  -> /content/Sage-A1.5-Q4_K_M.gguf   (~21.7 GB, streaming)
#   4. cold-fill  -> /content/pool.bin                (~18.1 GB, first load)
#   5. quality gate (bench/v12_quality.py - REVIEW THE TEXT MANUALLY)
#   6. bench_v8 c512 + c2048 (prefill pp512/pp2048, decode tg128, medians of 3)
#   7. package + publish: GitHub release + HF/MS mirrors (engine tarball,
#      pool via PUBLISH_POOL, reconverted GGUF via PUBLISH_GGUF)
#
# Usage (Colab T4 VM):
#   GHTOKEN=github_pat_... HFTOKEN=hf_... MSTOKEN=ms-... \
#     bash scripts/colab_v128_run.sh 2>&1 | tee /content/v128_run.log
set -uo pipefail

GHTOKEN=${GHTOKEN:?need GHTOKEN}
HFTOKEN=${HFTOKEN:-}
MSTOKEN=${MSTOKEN:-}
BRANCH=${BRANCH:-rel-v0.12.8}
REPO=${REPO:-/content/sager}
HF_BASE_REPO=${HF_BASE_REPO:-InternScience/Sage-A1.5}
MODEL=${MODEL:-/content/Sage-A1.5-Q4_K_M.gguf}
POOL=${POOL:-/content/pool.bin}
HF_DIR=${HF_DIR:-/content/hf_sage}
SKIP_MTP=${SKIP_MTP:-0}          # 1 = skip model-mtp.safetensors (disk headroom)
CONVERT_WORKERS=${CONVERT_WORKERS:-8}

echo "== $(date -u +%F_%T) v0.12.8 reconvert run start =="

# --- deps -------------------------------------------------------------------
pip install -q huggingface_hub hf_transfer modelscope safetensors numpy 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi | head -4 || { echo "NO GPU - abort"; exit 1; }

# --- repo @ rel-v0.12.8 -----------------------------------------------------
if [ ! -d "$REPO/.git" ]; then
  git clone "https://$GHTOKEN@github.com/samaidev/sager.git" "$REPO"
fi
cd "$REPO"
git config user.email "agent@localhost"
git config user.name "agent"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git rev-parse HEAD | tee /content/v128_commit.txt

# --- build + full ctest (GPU present: cuda_q4_stack must pass) --------------
echo "== build =="
cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -2
cmake --build build -j "$(nproc)" 2>&1 | tail -4 || { echo "BUILD FAILED"; exit 1; }
ctest --test-dir build --output-on-failure 2>&1 | tail -4

# --- HF safetensors download (resumable) ------------------------------------
echo "== download $HF_BASE_REPO =="
mkdir -p "$HF_DIR"
python3 - "$HF_BASE_REPO" "$HF_DIR" "$SKIP_MTP" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest, skip_mtp = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
allow = ["*.safetensors", "*.json", "tokenizer*"]
ignore = ["model-mtp.safetensors"] if skip_mtp else []
snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=allow,
                  ignore_patterns=ignore)
print("done")
PY

# --- reconvert GGUF from safetensors (the fix for roots 15-16 weights) ------
if [ ! -f "$MODEL" ]; then
  echo "== reconvert GGUF =="
  python3 tools/convert_hf_to_sager.py "$HF_DIR" "$MODEL" --workers "$CONVERT_WORKERS" \
    || { echo "CONVERT FAILED"; exit 1; }
fi
ls -la "$MODEL"
sha256sum "$MODEL" | tee /content/v128_gguf_sha256.txt

# --- pool cold-fill + quality gate + bench ----------------------------------
export SAGER_EXPERT_POOL_FILE="$POOL"
export SAGER_EXPERT_POOL_REUSE=1
export BENCH_MODEL="$MODEL"
export BENCH_POOL="$POOL"

if [ ! -f "$POOL" ]; then
  echo "== pool cold-fill (first load, ~37 min on 2 vCPU) =="
  python3 bench/v12_quality.py --model "$MODEL" --pool "$POOL" \
    2>&1 | tee /content/v128_quality.log || { echo "POOL BUILD/QUALITY FAILED"; exit 1; }
  ls -la "$POOL"
  sha256sum "$POOL" | tee /content/v128_pool_sha256.txt
  echo "!! MANUAL GATE: read /content/v128_quality.log - the text must be
  !! coherent English/Chinese (no word soup). Continue only if it reads well."
fi

echo "== bench chunk 512 =="
BENCH_CHUNK=512 python3 bench/bench_v8.py --json /content/v128_c512.json \
  2>&1 | tee /content/bench_c512.log
echo "== bench chunk 2048 =="
BENCH_CHUNK=2048 python3 bench/bench_v8.py --json /content/v128_c2048.json \
  2>&1 | tee /content/bench_c2048.log

# --- package ----------------------------------------------------------------
echo "== package =="
TARBALL=/content/sager_v0128_t4_cuda.tar.gz
tar -czf "$TARBALL" \
  build/sager_engine build/sager_test_q4 build/sager_test_gdn \
  build/sager_test_pump build/sager_test_cuda_q4 \
  build/libsager_core.a build/libsager_cuda.a \
  BENCH_RESULTS_v012.md WORKLOG.md BUILD_INFO.txt \
  scripts/dl_sage.sh scripts/colab_v128_run.sh scripts/publish_v128.py \
  /content/v128_c512.json /content/v128_c2048.json 2>/dev/null
ls -la "$TARBALL"

# --- publish (GH release + HF/MS mirrors; big files via env) ----------------
echo "== publish =="
GHTOKEN="$GHTOKEN" HFTOKEN="$HFTOKEN" MSTOKEN="$MSTOKEN" \
PUBLISH_POOL="$POOL" PUBLISH_GGUF="$MODEL" \
  python3 scripts/publish_v128.py "$TARBALL" \
  2>&1 | tee /content/v128_publish.log

echo "== done =="
echo "quality log: /content/v128_quality.log (REVIEW MANUALLY)"
echo "bench json : /content/v128_c512.json /content/v128_c2048.json"
echo "publish log: /content/v128_publish.log"
