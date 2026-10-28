#!/usr/bin/env bash
# dl_sage.sh - fetch Sage-A1.5 model weights + SAGER T4 artifacts in one go.
#
# Pulls (resumable, skip-if-present):
#   1. Sage-A1.5-Q4_K_M.gguf            21,713,463,520 B  (model weights)
#   2. Sage-A1.5-Q4_K_M-expert-pool-v012-fixed.bin 18,119,393,280 B
#      (expert pool rebuilt with the FIXED requant pipeline @ sager-v0.12.0;
#      the earlier Sage-A1.5-Q4_K_M-expert-pool.bin artifact contains garbage
#      down segments from the pre-fix Q4_K nibble order - do NOT use it with
#      SAGER_EXPERT_POOL_REUSE=1)
#   3. sager_t4_cuda_artifacts.tar.gz   prebuilt sager_engine binaries (opt-in,
#      ART_TARBALL=1)
#
# Sources:  SOURCE=hf (default)  -> huggingface.co  (HF_TOKEN)
#           SOURCE=ms            -> modelscope.cn   (MS_TOKEN / MODELSCOPE_API_TOKEN)
#
# Usage: bash scripts/dl_sage.sh [--source hf|ms]
set -uo pipefail

SOURCE=hf
if [ "${1:-}" = "--source" ]; then SOURCE="${2:-hf}"; fi
SOURCE="${SAGER_DL_SOURCE:-$SOURCE}"
CONTENT_DIR="${CONTENT_DIR:-/content}"
MODEL_REPO="${MODEL_REPO:-InternScience/Sage-A1.5-Q4_K_M-GGUF}"
MODEL_FILE="${MODEL_FILE:-Sage-A1.5-Q4_K_M.gguf}"
ART_REPO="${ART_REPO:-tchbcb/sager-t4-cuda-artifacts}"
MS_ART_REPO="${MS_ART_REPO:-luoyunlan168/sager-t4-cuda-artifacts}"
ART_FILE="${ART_FILE:-Sage-A1.5-Q4_K_M-expert-pool-v012-fixed.bin}"
ART_TARBALL="${ART_TARBALL:-0}"
mkdir -p "$CONTENT_DIR"

have() { command -v "$1" >/dev/null 2>&1; }

fetch_hf() {  # fetch_hf <repo> <file>
  local repo="$1" file="$2"
  local tok="${HF_TOKEN:-}"
  if [ -n "$tok" ]; then
    hf download "$repo" "$file" --repo-type model --local-dir "$CONTENT_DIR" --token "$tok"
  else
    hf download "$repo" "$file" --repo-type model --local-dir "$CONTENT_DIR"
  fi
}

fetch_ms() {  # fetch_ms <repo> <file>
  local repo="$1" file="$2"
  if ! have modelscope; then pip install -q modelscope; fi
  MODELSCOPE_API_TOKEN="${MS_TOKEN:-${MODELSCOPE_API_TOKEN:-}}" \
    modelscope download --model "$repo" "$file" --local_dir "$CONTENT_DIR"
}

fetch() {  # fetch <repo> <file> <fallback_ms_repo>
  local repo="$1" file="$2"
  local dest="$CONTENT_DIR/$file"
  if [ -f "$dest" ]; then
    echo "[dl] $file already present - skip ($(stat -c%s "$dest") bytes)"
    return 0
  fi
  echo "[dl] fetching $file from $SOURCE:$repo"
  local rc=0
  case "$SOURCE" in
    hf) fetch_hf "$repo" "$file" || rc=$? ;;
    ms) fetch_ms "${MS_ART_REPO:-$repo}" "$file" || rc=$? ;;
    *) echo "[dl] unknown SOURCE=$SOURCE (hf|ms)"; return 1 ;;
  esac
  if [ $rc -ne 0 ] && [ "$SOURCE" = "hf" ] && [ -n "${MS_TOKEN:-${MODELSCOPE_API_TOKEN:-}}" ]; then
    echo "[dl] hf failed (rc=$rc) - falling back to modelscope"
    fetch_ms "$repo" "$file" || return 1
  fi
  [ -f "$dest" ] || { echo "[dl] $file missing after fetch"; return 1; }
  echo "[dl] $file OK ($(stat -c%s "$dest") bytes)"
}

fetch "$MODEL_REPO" "$MODEL_FILE"
fetch "$ART_REPO" "$ART_FILE"
if [ "$ART_TARBALL" = "1" ]; then fetch "$ART_REPO" "sager_t4_cuda_artifacts.tar.gz"; fi

echo "[dl] export SAGER_EXPERT_POOL_FILE=$CONTENT_DIR/$ART_FILE"
echo "[dl] export SAGER_EXPERT_POOL_REUSE=1"
echo "[dl] done."
