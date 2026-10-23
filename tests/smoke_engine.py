#!/usr/bin/env python3
"""smoke_engine.py - drive the sager_engine JSON-lines protocol end to end.

Run against a synthetic model built by tools/make_smoke_gguf.py:
  python tools/make_smoke_gguf.py smoke.gguf
  python tests/smoke_engine.py hot     # big VRAM budget: pure GPU path
  python tests/smoke_engine.py cold    # tiny VRAM budget: CPU-pool miss path
  python tests/smoke_engine.py long    # 9000-token prompt crosses the 8192
                                       # chunked-prefill boundary
Env overrides: SAGER_ENGINE (default build/sager_engine),
SAGER_SMOKE_MODEL (default smoke.gguf).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ENG = os.environ.get("SAGER_ENGINE", "build/sager_engine")
MODEL = os.environ.get("SAGER_SMOKE_MODEL", "smoke.gguf")
LOG = "/tmp/sager_engine_stderr.log"


class Engine:
    def __init__(self):
        self.errf = open(LOG, "ab")
        self.p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errf, text=True, bufsize=1)

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def resp(self, what=""):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"engine died during {what}; tail:")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                print(f"  [non-json] {line[:160]}")


def generate(e, ids, n, tag, **kw):
    t0 = time.time()
    e.send(dict(cmd="generate", ids=ids, max_tokens=n,
                temperature=1.0, top_k=20, top_p=0.95, seed=42, **kw))
    toks = []
    while True:
        r = e.resp(tag)
        if r.get("type") == "token":
            toks.append(r["id"])
        elif r.get("type") == "done":
            dt = time.time() - t0
            print(f"  [{tag}] {len(toks)} tokens in {dt:.1f}s | "
                  f"decode_tps={r.get('decode_tps', 0):.2f} "
                  f"prefill_tps={r.get('prefill_tps', 0):.1f} "
                  f"drafted={r.get('drafted', 0)} accepted={r.get('accepted', 0)} "
                  f"hit={r.get('cache_hit_rate', 0):.2f} pos={r.get('pos')}")
            print(f"  [{tag}] ids[:12]={toks[:12]}")
            return r
        else:
            print(f"  [{tag}] msg: {json.dumps(r)[:200]}")


def phase_hot():
    e = Engine()
    t0 = time.time()
    e.send({"cmd": "load", "model": MODEL, "ctx": 4096,
            "vram_mb": 10240, "threads": 2, "graphs": True, "mtp": False,
            "chunk": 8192, "kv_q8": True})
    r = e.resp("load")
    print(f"LOAD({time.time()-t0:.1f}s): {json.dumps(r)[:300]}")
    assert r.get("type") == "loaded", r
    rng = np.random.default_rng(7)
    ids = rng.integers(1000, 31000, size=64).astype(int).tolist()
    generate(e, ids, 24, "hot")
    e.send({"cmd": "stats"})
    print("STATS:", json.dumps(e.resp()))
    e.send({"cmd": "reset"})
    print("RESET:", json.dumps(e.resp()))
    generate(e, ids[:32], 8, "hot-reset")
    e.send({"cmd": "shutdown"})
    print("SHUTDOWN:", json.dumps(e.resp()))


def phase_cold():
    e = Engine()
    e.send({"cmd": "load", "model": MODEL, "ctx": 4096,
            "vram_mb": 300, "threads": 2, "graphs": True, "mtp": False,
            "chunk": 8192, "kv_q8": True})
    r = e.resp("load")
    print("LOAD(cold):", json.dumps(r)[:300])
    assert r.get("type") == "loaded", r
    rng = np.random.default_rng(11)
    ids = rng.integers(1000, 31000, size=48).astype(int).tolist()
    generate(e, ids, 16, "cold")
    e.send({"cmd": "stats"})
    print("STATS:", json.dumps(e.resp()))
    e.send({"cmd": "shutdown"})
    print("SHUTDOWN:", json.dumps(e.resp()))


def phase_long():
    e = Engine()
    e.send({"cmd": "load", "model": MODEL, "ctx": 12288,
            "vram_mb": 10240, "threads": 2, "graphs": True, "mtp": False,
            "chunk": 8192, "kv_q8": True})
    r = e.resp("load")
    print("LOAD(long):", json.dumps(r)[:300])
    assert r.get("type") == "loaded", r
    rng = np.random.default_rng(23)
    ids = rng.integers(1000, 31000, size=9000).astype(int).tolist()
    generate(e, ids, 12, "long")
    e.send({"cmd": "shutdown"})
    print("SHUTDOWN:", json.dumps(e.resp()))


def run_gen(mtp_flag, n, **gen_kw):
    """One generation on a fresh engine; gen params via kwargs."""
    e = Engine()
    e.send({"cmd": "load", "model": MODEL, "ctx": 4096, "vram_mb": 10240,
            "threads": 2, "graphs": True, "mtp": mtp_flag, "chunk": 8192,
            "kv_q8": True})
    r = e.resp("load")
    assert r.get("type") == "loaded", r
    rng = np.random.default_rng(101)
    ids = rng.integers(1000, 31000, size=32).astype(int).tolist()
    e.send({"cmd": "generate", "ids": ids, "max_tokens": n, "seed": 1, **gen_kw})
    toks, res = [], None
    while True:
        r = e.resp("gen")
        if r.get("type") == "token":
            toks.append(r["id"])
        elif r.get("type") == "done":
            res = r
            break
        else:
            print("  msg:", json.dumps(r)[:200])
    e.send({"cmd": "shutdown"})
    e.resp("shutdown")
    return toks, res


def run_greedy(mtp_flag, n=40):
    """One greedy generation on a fresh engine, MTP drafting on or off.

    Greedy verification must keep the token stream bit-identical to plain
    decoding, so the mtp=True and mtp=False runs have to agree token for
    token - that is the correctness check; drafted/accepted check that the
    speculative machinery actually runs.
    """
    return run_gen(mtp_flag, n, greedy=True, presence_penalty=0.0,
                   repeat_penalty=1.0)


def phase_mtp():
    ta, ra = run_greedy(True)
    tb, rb = run_greedy(False)
    print(f"  greedy mtp:on  drafted={ra.get('drafted')} "
          f"accepted={ra.get('accepted')} decode_tps={ra.get('decode_tps', 0):.1f}")
    print(f"  greedy mtp:off drafted={rb.get('drafted')} "
          f"accepted={rb.get('accepted')} decode_tps={rb.get('decode_tps', 0):.1f}")
    if ta != tb:
        for i, (a, b) in enumerate(zip(ta, tb)):
            if a != b:
                print(f"  MISMATCH at {i}: mtp={a} plain={b}")
                break
        raise SystemExit("FAIL: speculative stream diverged from plain decode")
    print(f"  STREAMS IDENTICAL ({len(ta)} tokens) - bit-exact greedy chain")

    # Isolate the MTP drafter from the suffix drafter: at temp=1.0 the token
    # stream is non-repetitive, so the 3-gram suffix table goes quiet and any
    # drafted candidates must come from the MTP head.
    sa, ra2 = run_gen(True, 40, temperature=1.0, top_k=20, top_p=0.95,
                      presence_penalty=0.0, repeat_penalty=1.0)
    sb, rb2 = run_gen(False, 40, temperature=1.0, top_k=20, top_p=0.95,
                      presence_penalty=0.0, repeat_penalty=1.0)
    print(f"  temp   mtp:on  drafted={ra2.get('drafted')} "
          f"accepted={ra2.get('accepted')}")
    print(f"  temp   mtp:off drafted={rb2.get('drafted')} "
          f"accepted={rb2.get('accepted')}")
    assert ra2.get("drafted", 0) > 0, "MTP head drafted nothing at temp"
    if rb2.get("drafted", 0) == 0 and sa == sb:
        print("  MTP DRAFTER CONFIRMED (all temp-run drafts from the MTP head, "
              "streams still identical)")
    else:
        print("  note: suffix drafter also active at temp, or one speculative "
              "acceptance shifted the rng stream (benign)")
    print("PHASE mtp: PASS")


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "hot"
    t0 = time.time()
    {"hot": phase_hot, "cold": phase_cold, "long": phase_long,
     "mtp": phase_mtp}[ph]()
    print(f"PHASE {ph}: PASS ({time.time()-t0:.1f}s)")
