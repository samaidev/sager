#!/bin/bash
# 02_serve.sh - start the OpenAI-compatible API with 200k context.
# First load also builds the 18 GB expert pool into SAGER_EXPERT_POOL_FILE
# (file-backed mmap; later loads reuse it via SAGER_EXPERT_POOL_REUSE=1).
# Marker: /content/SERVE_DONE  Log: /content/logs/serve.log
set -euo pipefail
CTX="${SAGER_CTX:-204800}"
VRAM="${SAGER_VRAM_MB:-8192}"
PORT="${SAGER_PORT:-8000}"
export SAGER_EXPERT_POOL_FILE="${SAGER_EXPERT_POOL_FILE:-/content/sage_pool.bin}"
export SAGER_EXPERT_POOL_REUSE="${SAGER_EXPERT_POOL_REUSE:-1}"
export SAGER_LOG="${SAGER_LOG:-info}"

if curl -s "http://127.0.0.1:$PORT/props" > /dev/null 2>&1; then
  echo "[02] API already up on :$PORT - skip"
  touch /content/SERVE_DONE
  exit 0
fi

pkill -f 'serve/server.py' 2>/dev/null || true
cd /content/sager
nohup python3 serve/server.py \
  --model /content/Sage-A1.5-Q4_K_M.gguf \
  --binary /content/sager/build/sager_engine \
  --tokenizer-dir /content/sage_tokenizer \
  --ctx "$CTX" --chunk 8192 --vram-mb "$VRAM" --threads 2 \
  --port "$PORT" > /content/logs/serve.log 2>&1 &
echo $! > /content/serve.pid
echo "[02] server starting pid $(cat /content/serve.pid) ctx=$CTX vram_mb=$VRAM"
echo "[02] first load builds the expert pool - watch: tail -f /content/logs/serve.log"
