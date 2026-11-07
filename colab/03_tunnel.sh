#!/bin/bash
# 03_tunnel.sh - map the OpenAI API port to the public internet with a
# REGISTERED aitun tunnel (stable subdomain). Requires AITUN_API_KEY (ait_...).
set -euo pipefail
: "${AITUN_API_KEY:?export AITUN_API_KEY=ait_... first (never commit the key)}"
PORT="${SAGER_PORT:-8000}"
LOG=/content/logs/tunnel_api.log

pkill -f "aitun.*-k .*-p $PORT" 2>/dev/null || true
: > "$LOG"
nohup aitun -k "$AITUN_API_KEY" -p "$PORT" > "$LOG" 2>&1 &
echo $! > /content/tunnel_api.pid
for i in $(seq 1 15); do
  sleep 2
  # keyed tunnels print a subdomain (https://test.aitun.cc); free ones print
  # https://aitun.cc/<CODE> - match both forms
  URL=$(grep -Eo 'https://[A-Za-z0-9.-]*aitun\.cc(/[A-Za-z0-9]+)?' "$LOG" | grep -v aitun.cc: | head -1)
  [ -n "$URL" ] && { echo "[03] public API URL: $URL"; exit 0; }
done
echo "[03] tunnel did not report a URL - check $LOG" >&2
tail -20 "$LOG" >&2 || true
exit 1
