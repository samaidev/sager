#!/usr/bin/env python3
"""ship_head_src.py - chunked-upload the local HEAD source tarball to the VM
via the Samai envelope channel, then reassemble + crc-verify + extract.
Usage: python3 ship_head_src.py [TGZ] [URL]
"""
import base64, json, sys, time, urllib.request, zlib

TGZ = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sager_head_src.tgz"
URL = sys.argv[2] if len(sys.argv) > 2 else "https://aitun.cc/UKHF0RIR"
DEST_B64 = "/content/sager_head_src.b64"
CHUNK = 7000
ATTEMPTS = 3

raw = open(TGZ, "rb").read()
b64 = base64.urlsafe_b64encode(raw).decode().rstrip("=")
crc = "%08x" % (zlib.crc32(raw) & 0xFFFFFFFF)
chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
print("tgz=%d b64=%d chunks=%d crc=%s" % (len(raw), len(b64), len(chunks), crc))

def envelope(code: str) -> str:
    p = code.encode()
    eb = base64.urlsafe_b64encode(p).decode().rstrip("=")
    lines = "\n".join(eb[i:i + 76] for i in range(0, len(eb), 76))
    return "samaicmdbegin\nv=1\nenc=b64url\ncrc=%08x\n%s\nsamaicmdend" % (
        zlib.crc32(p) & 0xFFFFFFFF, lines)

def run(code: str, timeout=90):
    body = envelope(code)
    req = urllib.request.Request(
        URL + "/execute?respenc=b64url&timeout=%d" % timeout,
        data=body.encode(), method="POST",
        headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout + 30) as r:
        data = json.loads(r.read().decode())
    def dec(v):
        s = str(v or "")
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "replace")
    return data.get("success", True), dec(data.get("stdout_b64")), dec(data.get("stderr_b64"))

sent = 0
for i, ch in enumerate(chunks):
    open_cmd = "w" if i == 0 else "a"
    code = "CH=%s\nopen(%r,%r).write(CH)\nimport os\nprint(%d, len(CH), os.path.getsize(%r))" % (
        repr(ch), DEST_B64, open_cmd, i, DEST_B64)
    ok = False
    for att in range(ATTEMPTS):
        try:
            success, out, err = run(code)
            if success and str(i) in out.split():
                ok = True
                break
            print("chunk %d attempt %d bad resp: %r %r" % (i, att, out[:80], err[:120]))
        except Exception as e:
            print("chunk %d attempt %d error: %s" % (i, att, e))
            time.sleep(3)
    if not ok:
        print("FATAL: chunk %d failed after %d attempts" % (i, ATTEMPTS))
        sys.exit(1)
    sent += 1
    if sent % 10 == 0 or sent == len(chunks):
        print("  %d/%d chunks sent" % (sent, len(chunks)))

final = """
import zlib, base64, subprocess
raw = open({dest!r}).read().strip()
pad = '=' * (-len(raw) % 4)
data = base64.urlsafe_b64decode(raw + pad)
crc = '%08x' % (zlib.crc32(data) & 0xffffffff)
print('bytes', len(data), 'crc', crc)
assert len(data) == {size} and crc == {crc!r}, 'MISMATCH ' + crc
open('/content/sager_head_src.tgz','wb').write(data)
r = subprocess.run('rm -rf /content/sager_head && mkdir -p /content/sager_head && '
                   'tar xzf /content/sager_head_src.tgz -C /content/sager_head && '
                   'ls /content/sager_head && du -sm /content/sager_head',
                   shell=True, capture_output=True, text=True)
print(r.stdout or r.stderr)
""".format(dest=DEST_B64, size=len(raw), crc=crc)

for att in range(ATTEMPTS):
    try:
        success, out, err = run(final, timeout=120)
        print(out)
        if err.strip():
            print("STDERR:", err[:400])
        if success and "MISMATCH" not in out:
            print("SHIP_OK: %d chunks, %d bytes, crc %s verified remotely" % (sent, len(raw), crc))
            break
    except Exception as e:
        print("final attempt %d error: %s" % (att, e))
else:
    sys.exit(1)
