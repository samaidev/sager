#!/bin/bash
# 01_bootstrap.sh - fresh VM: clone repo, download GGUF+tokenizer, build engine.
# Idempotent: re-run skips whatever is already done. Marker: /content/BOOT1_DONE
set -u
LOG=/content/logs
mkdir -p "$LOG"
echo "[01] start $(date -u +%H:%M:%S)"

if [ ! -d /content/sager/.git ]; then
  git clone --depth 1 https://github.com/samaidev/sager.git /content/sager > "$LOG/clone.log" 2>&1
  echo "[01] clone rc=$?"
else
  echo "[01] repo present - skip clone"
fi

if [ ! -f /content/sage_tokenizer/tokenizer.json ]; then
  pip install -q -U huggingface_hub tokenizers > "$LOG/pip.log" 2>&1
  python3 -c "from huggingface_hub import hf_hub_download; p=hf_hub_download(repo_id='InternScience/Sage-A1.5', filename='tokenizer.json', local_dir='/content/sage_tokenizer'); print('tokenizer at', p)" > "$LOG/tok.log" 2>&1
  echo "[01] tokenizer rc=$?"
else
  echo "[01] tokenizer present - skip"
fi

if [ ! -f /content/Sage-A1.5-Q4_K_M.gguf ]; then
  python3 -c "from huggingface_hub import hf_hub_download; p=hf_hub_download(repo_id='InternScience/Sage-A1.5-Q4_K_M-GGUF', filename='Sage-A1.5-Q4_K_M.gguf', local_dir='/content'); print('gguf at', p)" > "$LOG/dl.log" 2>&1
  echo "[01] gguf dl rc=$?"
else
  echo "[01] gguf present - skip download"
fi

if [ ! -x /content/sager/build/sager_engine ]; then
  (cd /content/sager && bash setup.sh > "$LOG/build.log" 2>&1)
  echo "[01] build rc=$?"
else
  echo "[01] engine binary present - skip build"
fi

touch /content/BOOT1_DONE
echo "[01] done $(date -u +%H:%M:%S)"
