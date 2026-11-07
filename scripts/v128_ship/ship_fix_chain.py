#!/usr/bin/env python3
"""ship_fix_chain.py - write v128_fix_chain.sh on the VM, clear V128_FAILED,
launch it nohup, verify it started. Usage: python3 ship_fix_chain.py [URL]
"""
import base64, json, sys, time, urllib.request, zlib

URL = sys.argv[1] if len(sys.argv) > 1 else "https://aitun.cc/UKHF0RIR"

SCRIPT = r"""#!/bin/bash
# v128_fix_chain.sh - matched HEAD converter+engine (colab-oneclick lineage).
# Fixes 05_reconvert_clean.sh CONVERT_FAILED: rel-v0.12.8 converter predates --qsearch.
# Clean safetensors (already downloaded) -> --qsearch reconvert (root-31) -> cold pool
# -> v12_quality gate -> 200k ctx OpenAI API on :8000. Markers unchanged.
set -u
LOG=/content/logs
MODEL=/content/Sage-v128.gguf
POOL=/content/pool_v128.bin
HF_DIR=/content/hf_sage
SRC=/content/sager_head
exec >> $LOG/v128_chain.log 2>&1

echo "== v128 fix chain start $(date -u +%F_%T) =="
rm -f /content/V128_FAILED
pkill -f 'serve/server.py' 2>/dev/null
cd "$SRC" || { echo SRC_MISSING; touch /content/V128_FAILED; exit 1; }

echo "== build HEAD engine =="
cmake -B build -DCMAKE_BUILD_TYPE=Release > $LOG/build_head.log 2>&1 \
  && cmake --build build -j4 >> $LOG/build_head.log 2>&1 \
  || { echo BUILD_FAILED; tail -5 $LOG/build_head.log; touch /content/V128_FAILED; exit 1; }
ctest --test-dir build --output-on-failure 2>&1 | tail -3

echo "== reconvert GGUF (qsearch) -> $MODEL =="
python3 tools/convert_hf_to_sager.py $HF_DIR $MODEL --workers 8 --qsearch \
  > $LOG/convert_v128.log 2>&1 \
  || { echo CONVERT_FAILED; tail -5 $LOG/convert_v128.log; touch /content/V128_FAILED; exit 1; }
ls -la $MODEL
sha256sum $MODEL | tee /content/v128_gguf_sha256.txt
rm -rf $HF_DIR
df -h / | tail -1

echo "== pool cold-fill + quality gate (~40 min) =="
export SAGER_EXPERT_POOL_FILE=$POOL SAGER_EXPERT_POOL_REUSE=1
python3 bench/v12_quality.py --model $MODEL --pool $POOL \
  > /content/v128_quality.log 2>&1 \
  || { echo QUALITY_FAILED; tail -10 /content/v128_quality.log; touch /content/V128_FAILED; exit 1; }
tail -20 /content/v128_quality.log

echo "== serve clean weights (200k) =="
SAGER_EXPERT_POOL_FILE=$POOL SAGER_EXPERT_POOL_REUSE=1 SAGER_LOG=info \
  nohup python3 serve/server.py --model $MODEL --binary $SRC/build/sager_engine \
  --tokenizer-dir /content/sage_tokenizer --ctx 204800 --chunk 8192 \
  --vram-mb 8192 --threads 2 --port 8000 > $LOG/serve_v128.log 2>&1 &
echo $! > /content/serve.pid
touch /content/V128_SERVE_DONE
echo "== chain done $(date -u +%F_%T) =="
"""

LAUNCH = """
import subprocess, time, os
open('/content/v128_fix_chain.sh','w').write({script!r})
os.chmod('/content/v128_fix_chain.sh', 0o755)
for f in ('/content/V128_FAILED', '/content/V128_SERVE_DONE'):
    try: os.unlink(f)
    except OSError: pass
try: os.unlink('/content/sager_head_src.b64')
except OSError: pass
subprocess.Popen(['nohup', 'bash', '/content/v128_fix_chain.sh'],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(5)
r = subprocess.run('ps aux | grep v128_fix_chain | grep -v grep; '
                   'echo ---; tail -c 600 /content/logs/v128_chain.log',
                   shell=True, capture_output=True, text=True)
print(r.stdout or r.stderr)
""".format(script=SCRIPT)

def envelope(code: str) -> str:
    p = code.encode()
    eb = base64.urlsafe_b64encode(p).decode().rstrip("=")
    lines = "\n".join(eb[i:i + 76] for i in range(0, len(eb), 76))
    return "samaicmdbegin\nv=1\nenc=b64url\ncrc=%08x\n%s\nsamaicmdend" % (
        zlib.crc32(p) & 0xFFFFFFFF, lines)

body = envelope(LAUNCH)
req = urllib.request.Request(
    URL + "/execute?respenc=b64url&timeout=60", data=body.encode(),
    method="POST", headers={"Content-Type": "text/plain"})
with urllib.request.urlopen(req, timeout=120) as r:
    data = json.loads(r.read().decode())

def dec(v):
    s = str(v or "")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "replace")

print("RAW:", json.dumps(data)[:400])
print(dec(data.get("stdout_b64")))
for k in ("stderr_b64", "error_b64"):
    err = dec(data.get(k))
    if err.strip():
        print(k.upper() + ":", err[:500], file=sys.stderr)
if not data.get("success", True):
    print("--- success=false ---", file=sys.stderr)
    sys.exit(1)
