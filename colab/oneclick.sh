#!/bin/bash
# oneclick.sh - whole stack on a fresh Colab VM, in order, all idempotent.
#   bash colab/oneclick.sh
# Public API tunnel additionally needs:  export AITUN_API_KEY=ait_...
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
export AITUN_API_KEY="${AITUN_API_KEY:-}"

echo "==== stage 1/4 bootstrap (clone + download + build) ===="
bash "$DIR/01_bootstrap.sh"

echo "==== stage 2/4 serve (200k ctx OpenAI API) ===="
bash "$DIR/02_serve.sh"
echo "waiting for engine load (first run builds the 18 GB expert pool)..."
for i in $(seq 1 240); do
  sleep 10
  if curl -s "http://127.0.0.1:${SAGER_PORT:-8000}/props" | grep -q '"engine":{'; then
    echo "engine ready after ~$((i * 10))s"; break
  fi
done

echo "==== stage 3/4 public tunnel ===="
if [ -n "$AITUN_API_KEY" ]; then
  bash "$DIR/03_tunnel.sh"
else
  echo "AITUN_API_KEY not set - skipping public URL (API stays VM-local)"
fi

echo "==== stage 4/4 ZAgent ===="
bash "$DIR/04_zagent.sh"

echo "==== stack up ===="
echo "API:     http://127.0.0.1:${SAGER_PORT:-8000}/v1  (+ public URL above)"
echo "ZAgent:  http://127.0.0.1:${ZAGENT_PORT:-8384}"
