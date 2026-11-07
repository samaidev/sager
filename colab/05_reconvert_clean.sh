#!/bin/bash
# v128_serve_chain.sh - reconvert clean weights from HF safetensors (root-15/16 fix),
# quantize with --qsearch (root-31 fix, same speed as --fast), cold-fill pool,
# quality-gate, then bring the 200k OpenAI API back up on clean weights.
# 2026-11-07 FIX: converter and engine MUST come from the same tree. The first
# version ran the converter from a rel-v0.12.8 worktree, whose converter predates
# --qsearch -> CONVERT_FAILED (unrecognized arguments). Build the matched
# converter+engine pair from $REPO (the clone that carries the qsearch line).
set -u
LOG=/content/logs
MODEL=/content/Sage-v128.gguf
POOL=/content/pool_v128.bin
HF_DIR=/content/hf_sage
REPO=/content/sager

echo "== v128 chain start $(date -u +%F_%T) =="
pkill -f 'serve/server.py' 2>/dev/null; pkill -f sager_engine 2>/dev/null; sleep 2
rm -f /content/sage_pool.bin /content/sage_pool_v012.bin /content/Sage-A1.5-Q4_K_M.gguf
df -h / | tail -1

cd "$REPO" || { echo REPO_MISSING; touch /content/V128_FAILED; exit 1; }
# guard: this repo's converter must actually carry the qsearch quantizer
grep -q "qsearch" tools/convert_hf_to_sager.py \
  || { echo NO_QSEARCH_IN_CONVERTER; touch /content/V128_FAILED; exit 1; }

echo "== build HEAD engine (matched pair with converter) =="
cmake -B build -DCMAKE_BUILD_TYPE=Release > "$LOG/build_head.log" 2>&1 \
  && cmake --build build -j4 >> "$LOG/build_head.log" 2>&1 \
  || { echo BUILD_FAILED; tail -5 "$LOG/build_head.log"; touch /content/V128_FAILED; exit 1; }
ctest --test-dir build --output-on-failure 2>&1 | tail -3

echo "== download safetensors (skip MTP) =="
mkdir -p "$HF_DIR"
python3 - InternScience/Sage-A1.5 "$HF_DIR" <<'PY' || { echo DL_FAILED; touch /content/V128_FAILED; exit 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
                  ignore_patterns=["model-mtp.safetensors"])
print("download done")
PY
df -h / | tail -1

echo "== reconvert GGUF (qsearch) -> $MODEL =="
python3 tools/convert_hf_to_sager.py "$HF_DIR" "$MODEL" --workers 8 --qsearch \
  > "$LOG/convert_v128.log" 2>&1 \
  || { echo CONVERT_FAILED; tail -5 "$LOG/convert_v128.log"; touch /content/V128_FAILED; exit 1; }
ls -la "$MODEL"
sha256sum "$MODEL" | tee /content/v128_gguf_sha256.txt
rm -rf "$HF_DIR"
df -h / | tail -1

echo "== pool cold-fill + quality gate (~40 min) =="
export SAGER_EXPERT_POOL_FILE="$POOL" SAGER_EXPERT_POOL_REUSE=1
python3 bench/v12_quality.py --model "$MODEL" --pool "$POOL" \
  > /content/v128_quality.log 2>&1 \
  || { echo QUALITY_FAILED; tail -10 /content/v128_quality.log; touch /content/V128_FAILED; exit 1; }
tail -20 /content/v128_quality.log

echo "== serve clean weights (200k) =="
SAGER_EXPERT_POOL_FILE="$POOL" SAGER_EXPERT_POOL_REUSE=1 SAGER_LOG=info \
  nohup python3 serve/server.py --model "$MODEL" --binary "$REPO/build/sager_engine" \
  --tokenizer-dir /content/sage_tokenizer --ctx 204800 --chunk 8192 \
  --vram-mb 8192 --threads 2 --port 8000 > "$LOG/serve_v128.log" 2>&1 &
echo $! > /content/serve.pid
touch /content/V128_SERVE_DONE
echo "== chain done $(date -u +%F_%T) =="
