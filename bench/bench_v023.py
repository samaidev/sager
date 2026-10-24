#!/usr/bin/env python3
"""sager v0.2.3 A/B bench: Strata-style streamed prefill vs per-row walk.

Runs on the Colab VM. Same model/pool/config for both arms; the only
difference is SAGER_PF_BATCHED. Both arms must emit identical token ids
(same kernels per row, same fp32 accumulation order) - the diff is the
correctness gate.

Env overrides for the ENGINE can be dropped in /content/pf_env as
space-separated K=V pairs (read at bench start, exported to the engine).

Usage: python3 bench_v023.py <arm>     # arm in {poolbuild, A, B, all}
"""
import base64
import json
import os
import random
import re
import subprocess
import sys
import time

ENG = "/content/sager/build/sager_engine"
MODEL = "/content/sage_gguf/Sage-A1.5-Q4_K_M.gguf"
POOL = "/content/pool.bin"
OUT = "/content/bench_v023"
VOCAB = 248320

os.makedirs(OUT, exist_ok=True)

# engine env overrides (pf_env), applied to every arm equally
if os.path.exists("/content/pf_env"):
    for kv in open("/content/pf_env").read().split():
        k, v = kv.split("=", 1)
        os.environ[k] = v
    print("pf_env:", dict(kv.split("=", 1) for kv in
                          open("/content/pf_env").read().split()), flush=True)

def prompt(n, seed):
    rnd = random.Random(seed)
    return [rnd.randrange(1000, VOCAB) for _ in range(n)]

P512 = prompt(512, 2026)
P2048 = prompt(2048, 2027)
P64 = P512[:64]

LOAD = {"cmd": "load", "model": MODEL, "ctx": 8192, "vram_mb": 8960,
        "pump_arena_mb": 1024, "threads": 2, "graphs": True, "mtp": False,
        "chunk": 2048, "kv_q8": True}

def run_engine(cmds, env_extra, log_name):
    env = dict(os.environ)
    env["SAGER_EXPERT_POOL_FILE"] = POOL
    env["SAGER_EXPERT_POOL_REUSE"] = env_extra.pop("REUSE", "1")
    env.update(env_extra)
    for suffix in ("", ".err"):
        p = os.path.join(OUT, log_name + suffix)
        if os.path.exists(p):
            os.unlink(p)
    stdin = "".join(json.dumps(c) + "\n" for c in cmds)
    t0 = time.time()
    proc = subprocess.run([ENG], input=stdin, capture_output=True, text=True,
                          env=env, timeout=5400)
    dt = time.time() - t0
    with open(os.path.join(OUT, log_name), "w") as f:
        f.write(proc.stdout)
    with open(os.path.join(OUT, log_name + ".err"), "w") as f:
        f.write(proc.stderr)
    # v0.2.5b: decode_tps lives in the stdout done-line JSON, not stderr
    pp = tg = None
    for line in proc.stdout.splitlines():
        s2 = line.strip()
        if s2.startswith('{"') and '"type":"done"' in s2:
            try:
                dj = json.loads(s2)
                if dj.get("decode_tps"):
                    tg = float(dj["decode_tps"])
            except Exception:
                pass
    m = re.search(r"prefill\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s",
                  proc.stderr)
    if m:
        pp = {"tokens": int(m.group(1)), "ms": int(m.group(2)),
              "tps": float(m.group(3))}
    if tg is None:
        m = re.search(r"decode_tps[\": ]+([\d.]+)", proc.stderr)
        if m:
            tg = float(m.group(1))
    m = re.search(r"decode\] .*?([\d.]+) tok/s", proc.stderr)
    if tg is None and m:
        tg = float(m.group(1))
    stream = {}
    m = re.search(r"stream-all chunk done: (\d+) pairs, ([\d.]+) GB H2D total, "
                  r"(\d+) steals, (\d+) drops, (\d+) skipped, "
                  r"(\d+) arm timeouts, (\d+) read errors", proc.stderr)
    if m:
        stream = {"pairs": int(m.group(1)), "gb": float(m.group(2)),
                  "steals": int(m.group(3)), "drops": int(m.group(4)),
                  "skipped": int(m.group(5)), "timeouts": int(m.group(6)),
                  "read_errors": int(m.group(7))}
    return {"wall_s": round(dt, 1), "prefill": pp, "decode_tps": tg,
            "stream": stream,
            "err_tail": proc.stderr.strip().splitlines()[-6:]}

def gen_cmds(ids, max_tokens):
    return [LOAD,
            {"cmd": "generate", "ids": ids, "max_tokens": max_tokens,
             "temperature": 1.0, "seed": 42},
            {"cmd": "shutdown"}]

def arm(env_extra, tag):
    print(f"=== arm {tag} (env {env_extra or 'defaults'}) ===", flush=True)
    res = {}
    res["pp512_tg128"] = run_engine(gen_cmds(P512, 128), dict(env_extra),
                                    f"{tag}_pp512_tg128")
    print(f"  pp512: {res['pp512_tg128']['prefill']}", flush=True)
    print(f"  tg128: decode_tps={res['pp512_tg128']['decode_tps']}", flush=True)
    print(f"  stream: {res['pp512_tg128']['stream']}", flush=True)
    res["pp2048"] = run_engine(
        [LOAD, {"cmd": "generate", "ids": P2048, "max_tokens": 1,
                "temperature": 1.0, "seed": 42}, {"cmd": "shutdown"}],
        dict(env_extra), f"{tag}_pp2048")
    print(f"  pp2048: {res['pp2048']['prefill']}", flush=True)
    print(f"  stream: {res['pp2048']['stream']}", flush=True)
    with open(os.path.join(OUT, f"{tag}_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res

def poolbuild():
    print("=== pool build (REUSE=0) ===", flush=True)
    r = run_engine([LOAD, {"cmd": "shutdown"}], {"REUSE": "0"}, "poolbuild")
    print(f"pool build done in {r['wall_s']/60:.1f} min", flush=True)

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "poolbuild":
        poolbuild(); return
    if which in ("A", "all"):
        arm({}, "A_stream")           # defaults: batched + stream-all on
    if which in ("B", "all"):
        arm({"SAGER_PF_BATCHED": "0"}, "B_perrow")   # v0.2.2 behavior
    a = os.path.join(OUT, "A_stream_pp512_tg128")
    b = os.path.join(OUT, "B_perrow_pp512_tg128")
    if os.path.exists(a) and os.path.exists(b):
        def toks(path):
            out = []
            for line in open(path):
                s2 = line.strip()
                if s2.startswith('{"') and '"type":"token"' in s2:
                    try:
                        out.append(json.loads(s2).get("id"))
                    except Exception:
                        pass
            return out
        ta_list, tb_list = toks(a), toks(b)
        ta, tb = len(ta_list), len(tb_list)
        same = ta_list == tb_list
        print(f"TOKEN_STREAMS: A={ta} B={tb} "
              + ("IDENTICAL" if same else "DIFFER"), flush=True)

if __name__ == "__main__":
    main()

