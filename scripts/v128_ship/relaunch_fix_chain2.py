#!/usr/bin/env python3
"""relaunch_fix_chain2.py - relaunch v128_fix_chain.sh exactly once, with a
self-match-immune guard (match bash by comm, not the payload cmdline)."""
import base64, json, sys, urllib.request, zlib

URL = sys.argv[1] if len(sys.argv) > 1 else "https://aitun.cc/UKHF0RIR"

# $2==comm in ps -eo pid,comm,args ; payload runs as python so it can't match
CODE = r"""
import subprocess, time

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)

chk = sh("ps -eo pid,comm,args | awk '$2==\"bash\" && index($0,\"v128_fix_chain.sh\")'").stdout.strip()
print('existing bash instances:', repr(chk))
if chk:
    print('GUARD: already running, not relaunching')
else:
    subprocess.Popen(['nohup', 'bash', '/content/v128_fix_chain.sh'],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(6)
    print(sh("ps -eo pid,comm,args | awk '$2==\"bash\" && index($0,\"v128_fix_chain.sh\")'").stdout)
    print(sh('tail -c 300 /content/logs/v128_chain.log').stdout)
"""

def envelope(code: str) -> str:
    p = code.encode()
    eb = base64.urlsafe_b64encode(p).decode().rstrip("=")
    lines = "\n".join(eb[i:i + 76] for i in range(0, len(eb), 76))
    return "samaicmdbegin\nv=1\nenc=b64url\ncrc=%08x\n%s\nsamaicmdend" % (
        zlib.crc32(p) & 0xFFFFFFFF, lines)

req = urllib.request.Request(
    URL + "/execute?respenc=b64url&timeout=90", data=envelope(CODE).encode(),
    method="POST", headers={"Content-Type": "text/plain"})
with urllib.request.urlopen(req, timeout=150) as r:
    data = json.loads(r.read().decode())

def dec(v):
    s = str(v or "")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "replace")

print(dec(data.get("stdout_b64")))
for k in ("stderr_b64", "error_b64"):
    e = dec(data.get(k))
    if e.strip():
        print(k.upper() + ":", e[:400], file=sys.stderr)
print("success:", data.get("success"))
