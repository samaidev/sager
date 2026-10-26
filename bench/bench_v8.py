#!/usr/bin/env python3
# bench_v8.py - resident-engine benchmark + parity gates for the v0.8 patchset.
#
# Protocol (matches BENCH_RESULTS_v07):
#   - random-token prompts, seed 2026, per-round session reset, medians of 3
#   - pp512/pp2048: max_tokens=1, report prefill_tps
#   - tg128: 128 tokens at temp 1.0 / top_k 20 / top_p 0.95, report decode_tps
#   - parity arm: greedy (temp 0) token streams must be IDENTICAL between the
#     v0.7 path (SAGER_DECODE_FUSE=0 SAGER_PF_MMQ=0) and the v0.8 path with
#     the same SAGER_GDN_FFN_NORM setting (MMQ/fusions are bit-identical by
#     construction; the GDN norm fix changes outputs and is excluded from the
#     equality gate, verified separately).
#
#   BENCH_MODEL=/path/Sage-A1.5-Q4_K_M.gguf \
#   BENCH_POOL=/path/Sage-A1.5-Q4_K_M-expert-pool.bin \
#   BENCH_CHUNK=512 python3 bench/bench_v8.py [--skip-parity] [--json out.json]
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time


def base_env(extra: dict) -> dict:
    env = dict(os.environ)
    # benchmark knobs only; feature flags come from `extra`
    for k in ("SAGER_DECODE_FUSE", "SAGER_PF_MMQ", "SAGER_GDN_FFN_NORM",
              "SAGER_HEAD_SMX"):
        env.pop(k, None)
    env.setdefault("SAGER_EXPERT_POOL_REUSE", "1")
    # v0.7 benchmark protocol: device decode + device sampler enabled
    env.setdefault("SAGER_DECODE_DEVICE", "1")
    env.setdefault("SAGER_DEVICE_SAMPLER", "1")
    env.setdefault("SAGER_PROFILE_PATH", "/content/sager_profile.json")
    if os.environ.get("BENCH_POOL"):
        env["SAGER_EXPERT_POOL_FILE"] = os.environ["BENCH_POOL"]
    env.update(extra)
    return env


class Engine:
    def __init__(self, model: str, chunk: int, extra_env: dict | None = None):
        self.proc = subprocess.Popen(
            [os.environ.get("BENCH_ENGINE", "build/sager_engine")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1,
            env=base_env(extra_env or {}))
        self.model = model
        self.chunk = chunk

    def cmd(self, req: dict, timeout: float = 1800) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"engine timeout on {req.get('cmd')}")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died: EOF")
            v = json.loads(line)
            if v.get("type") == "error":
                raise RuntimeError("engine error: " + v.get("message", "?"))
            if v.get("type") in ("done", "loaded", "stats", "reset", "ok",
                                 "bye"):
                return v
            # token lines stream past; ignored in the timing arms

    def load(self) -> dict:
        return self.cmd({"cmd": "load", "model": self.model, "ctx": 8192,
                         "vram_mb": int(os.environ.get("BENCH_VRAM", "9984")),
                         "threads": int(os.environ.get("BENCH_THREADS", "2")),
                         "graphs": True,
                         "mtp": os.environ.get("BENCH_MTP", "1") != "0",
                         "chunk": self.chunk,
                         "kv_q8": True}, timeout=3600)

    def reset(self) -> None:
        self.cmd({"cmd": "reset"})

    def generate(self, ids: list, max_tokens: int, greedy: bool) -> dict:
        return self.cmd({"cmd": "generate", "ids": ids,
                         "max_tokens": max_tokens,
                         "temperature": 0.0 if greedy else 1.0,
                         "top_k": 20, "top_p": 0.95,
                         "greedy": greedy, "seed": 7}, timeout=1800)

    def close(self) -> None:
        try:
            self.cmd({"cmd": "shutdown"}, timeout=30)
        except Exception:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def rand_ids(n: int, seed: int = 2026) -> list:
    rng = random.Random(seed)
    return [rng.randrange(1000, 200000) for _ in range(n)]


def bench_arm(label: str, model: str, chunk: int, rounds: int,
              extra_env: dict) -> dict:
    print(f"=== arm {label} (chunk {chunk}, env {extra_env}) ===", flush=True)
    eng = Engine(model, chunk, extra_env)
    ld = eng.load()
    keys = ("hot_experts", "hot_capacity", "pump", "graphs", "ctx")
    print("load:", json.dumps({k: ld.get(k) for k in keys if k in ld}),
          flush=True)
    res = {"label": label, "chunk": chunk, "env": extra_env,
           "pp512": [], "pp2048": [], "tg128": [], "load": ld}
    for r in range(rounds):
        eng.reset()
        d = eng.generate(rand_ids(512, 2026), 1, greedy=False)
        res["pp512"].append(d.get("prefill_tps", 0.0))
        print(f"  pp512 r{r + 1}: {res['pp512'][-1]:7.1f} tok/s", flush=True)
    for r in range(rounds):
        eng.reset()
        d = eng.generate(rand_ids(2048, 2026), 1, greedy=False)
        res["pp2048"].append(d.get("prefill_tps", 0.0))
        print(f"  pp2048 r{r + 1}: {res['pp2048'][-1]:7.1f} tok/s", flush=True)
    for r in range(rounds):
        eng.reset()
        d = eng.generate(rand_ids(512, 2026), 128, greedy=False)
        res["tg128"].append(d.get("decode_tps", 0.0))
        print(f"  tg128 r{r + 1}: {res['tg128'][-1]:7.1f} tok/s", flush=True)
    eng.close()
    print(f"  MEDIANS  pp512 {statistics.median(res['pp512']):7.1f} | "
          f"pp2048 {statistics.median(res['pp2048']):7.1f} | "
          f"tg128 {statistics.median(res['tg128']):6.2f} tok/s", flush=True)
    return res


def parity_arm(extra_env: dict, model: str, chunk: int, prompt: list) -> list:
    """Greedy 64-token stream with per-token capture (own engine process)."""
    proc = subprocess.Popen(
        [os.environ.get("BENCH_ENGINE", "build/sager_engine")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=sys.stderr, text=True, bufsize=1,
        env=base_env(extra_env))
    toks = []

    def send(req: dict, want_done: str, timeout: float = 1800) -> dict:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError("parity timeout")
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died in parity arm")
            v = json.loads(line)
            if v.get("type") == "token":
                toks.append(v["id"])
            elif v.get("type") == "error":
                raise RuntimeError("parity error: " + v.get("message", "?"))
            elif v.get("type") == want_done:
                return v
            elif v.get("type") in ("ok", "bye") and want_done == "reset":
                return v

    send({"cmd": "load", "model": model, "ctx": 8192,
          "vram_mb": int(os.environ.get("BENCH_VRAM", "9984")),
          "threads": 2, "graphs": True,
          "mtp": os.environ.get("BENCH_MTP", "1") != "0", "chunk": chunk,
          "kv_q8": True}, "loaded")
    send({"cmd": "reset"}, "ok")
    send({"cmd": "generate", "ids": prompt, "max_tokens": 64,
          "temperature": 0.0, "top_k": 20, "top_p": 0.95, "greedy": True,
          "seed": 7}, "done")
    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--json", default="bench_v8_summary.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--skip-bench", action="store_true",
                    help="parity gates only")
    a = ap.parse_args()
    model = os.environ["BENCH_MODEL"]
    chunk = int(os.environ.get("BENCH_CHUNK", "512"))

    prompt = rand_ids(96, 4242)
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
           "chunk": chunk, "parity": None, "arms": []}

    if not a.skip_parity:
        print("=== parity: v0.7 path (fuse=0 mmq=0) vs v0.8 defaults ===",
              flush=True)
        # the GDN ffn-norm fix changes outputs; keep it OUT of the equality
        # gate by fixing it identically on both sides
        base = parity_arm({"SAGER_DECODE_FUSE": "0", "SAGER_PF_MMQ": "0",
                           "SAGER_GDN_FFN_NORM": "0"}, model, chunk, prompt)
        new = parity_arm({"SAGER_GDN_FFN_NORM": "0"}, model, chunk, prompt)
        ok = base == new and len(base) == 64
        out["parity"] = {"gdn_ffn_norm": "off", "v07_len": len(base), "v08_len": len(new),
                         "identical": ok,
                         "v07_head": base[:16], "v08_head": new[:16]}
        print(f"  streams: {'IDENTICAL' if ok else 'DIVERGED'} "
              f"({len(base)} vs {len(new)} tokens)", flush=True)
        if not ok:
            print("  v07:", base[:32], flush=True)
            print("  v08:", new[:32], flush=True)
            out_path = a.json
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            sys.exit(1)

    if not a.skip_bench:
        arms = [("v08-default", {})]
        if os.environ.get("BENCH_ARMS", "1") != "0":
            # exact v0.7 behavior: every v0.8 switch off, GDN norm fix off
            arms.append(("v07-exact", {"SAGER_DECODE_FUSE": "0",
                                       "SAGER_PF_MMQ": "0",
                                       "SAGER_HEAD_SMX": "0",
                                       "SAGER_GDN_FFN_NORM": "0"}))
        if os.environ.get("BENCH_ISOLATE", "0") != "0":
            arms = [("v07-exact", {"SAGER_DECODE_FUSE": "0",
                                   "SAGER_PF_MMQ": "0",
                                   "SAGER_HEAD_SMX": "0",
                                   "SAGER_GDN_FFN_NORM": "0"}),
                    ("v08-all", {"SAGER_GDN_FFN_NORM": "0"}),
                    ("mmq-only", {"SAGER_DECODE_FUSE": "0",
                                  "SAGER_GDN_FFN_NORM": "0"}),
                    ("fuse-only", {"SAGER_PF_MMQ": "0",
                                   "SAGER_HEAD_SMX": "0",
                                   "SAGER_GDN_FFN_NORM": "0"})]
        for label, env in arms:
            r = bench_arm(label, model, chunk, a.rounds, env)
            out["arms"].append(r)

    with open(a.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"summary -> {a.json}", flush=True)


if __name__ == "__main__":
    main()
