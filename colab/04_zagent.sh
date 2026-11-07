#!/bin/bash
# 04_zagent.sh - install ZAgent (samaidev/zagent_r) and point it at the local API.
# The pip wheel's downloader looks for the wrong filename inside the release
# tarball (wrapper expects 'zagent', archive ships 'zagent-linux-amd64'), so we
# place the binary at the path the wrapper probes: <pkg>/bin/zagent-linux-amd64.
# Marker: /content/ZAGENT_DONE  Log: /content/logs/zagent.log
set -euo pipefail
ZPORT="${ZAGENT_PORT:-8384}"

pip install -q samai-zagent
PKG=$(python3 -c "import zagent, os; print(os.path.dirname(zagent.__file__))")
BIN="$PKG/bin"
mkdir -p "$BIN"
if [ ! -x "$BIN/zagent-linux-amd64" ]; then
  curl -sL --max-time 300 -o /tmp/zagent.tar.gz 'https://zagent.samai.cc/downloads/zagent-linux-amd64.tar.gz'
  tar -xzf /tmp/zagent.tar.gz -C "$BIN" ./zagent-linux-amd64 ./web
  chmod +x "$BIN/zagent-linux-amd64"
fi
"$BIN/zagent-linux-amd64" -h > /dev/null 2>&1 || { echo "[04] binary broken"; exit 1; }

if curl -s "http://127.0.0.1:$ZPORT/api/agent/status" > /dev/null 2>&1; then
  echo "[04] ZAgent already up on :$ZPORT - skip"
  touch /content/ZAGENT_DONE
  exit 0
fi

nohup "$BIN/zagent-linux-amd64" \
  -llm-url  "${SAGER_LLM_URL:-http://127.0.0.1:${SAGER_PORT:-8000}/v1}" \
  -llm-key  "${SAGER_LLM_KEY:-sk-sager-local}" \
  -llm-model "${SAGER_LLM_MODEL:-sage-a1.5}" \
  -web-addr ":$ZPORT" \
  -data /root/.zagent \
  -log-file /content/logs/zagent.log \
  -daemon
sleep 3
curl -s "http://127.0.0.1:$ZPORT/api/agent/status" | head -c 300 || { echo "[04] zagent not responding - check /content/logs/zagent.log"; exit 1; }
echo
touch /content/ZAGENT_DONE
echo "[04] ZAgent web UI: http://<host>:$ZPORT  (log: /content/logs/zagent.log)"
