#!/usr/bin/env python3
# v12_quality.py - final text-quality gate for the v0.12 fixed stack.
#
#   python3 bench/v12_quality.py [--model /content/Sage-A1.5-Q4_K_M.gguf]
#                                [--pool /content/Sage-A1.5-Q4_K_M-expert-pool.bin]
#                                [--max-tokens 160] [--engine build/sager_engine]
#
# Three real-text ChatML prompts, greedy decode, exact tokenizer (pulled from
# the InternScience/Sage-A1.5 base repo on first use). Prints full generations
# plus crude coherence stats (letter/CJK ratio, 8-gram loop detection) and a
# per-prompt PASS/FAIL verdict; exit 0 only if all prompts pass. This is the
# release gate referenced in BENCH_RESULTS_v012.md.
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE_REPO = "https://huggingface.co/InternScience/Sage-A1.5/resolve/main/"
TOK_DIR = os.environ.get("SAGER_TOK_DIR", "/content/sage_tokenizer")
IM_END = 151645

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that reverses a string.",
    "用两句话介绍量子计算。",
]

SYSTEM = "You are a helpful assistant."


def ensure_tokenizer(dirpath: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    tok = os.environ.get("HF_TOKEN", "")
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(dirpath, fn)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        req = urllib.request.Request(BASE_REPO + fn)
        if tok:
            req.add_header("Authorization", "Bearer " + tok)
        print("downloading", fn, flush=True)
        with urllib.request.urlopen(req, timeout=300) as r, open(p, "wb") as f:
            f.write(r.read())
    return dirpath


def load_tokenizer(dirpath: str):
    from tokenizers import Tokenizer as HFTokenizer
    return HFTokenizer.from_file(os.path.join(dirpath, "tokenizer.json"))


def chatml(user: str, system: str = SYSTEM) -> str:
    return ("<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")


class Engine:
    def __init__(self, engine: str, model: str, chunk: int, pool: str | None):
        env = dict(os.environ)
        env.setdefault("SAGER_EXPERT_POOL_REUSE", "1")
        env.setdefault("SAGER_DECODE_DEVICE", "1")
        env.setdefault("SAGER_DEVICE_SAMPLER", "1")
        if pool:
            env["SAGER_EXPERT_POOL_FILE"] = pool
        env.pop("SAGER_DUMP_LOGITS", None)
        self.proc = subprocess.Popen(
            [engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, env=env, cwd=os.getcwd())

    def cmd(self, req: dict, timeout: float = 1800) -> tuple[dict, list]:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        toks: list[int] = []
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError("engine timeout on %s" % req.get("cmd"))
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            v = json.loads(line)
            if v.get("type") == "token":
                toks.append(v["id"])
            elif v.get("type") == "error":
                raise RuntimeError("engine error: " + v.get("message", "?"))
            elif v.get("type") in ("done", "loaded", "ok", "bye"):
                return v, toks

    def close(self):
        try:
            self.cmd({"cmd": "shutdown"}, timeout=30)
        except Exception:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass


def coherence(text: str) -> tuple[float, int]:
    """(letter/CJK ratio, longest 8-gram repeat count)."""
    if not text:
        return 0.0, 0
    good = sum(1 for c in text if c.isalnum() or '\u3400' <= c <= '\u9fff')
    ratio = good / len(text)
    words = text.split()
    runs, best = {}, 1
    for i in range(len(words) - 8):
        g = " ".join(words[i:i + 8])
        runs[g] = runs.get(g, 0) + 1
        best = max(best, runs[g])
    return ratio, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "BENCH_MODEL", "/content/Sage-A1.5-Q4_K_M.gguf"))
    ap.add_argument("--pool", default=os.environ.get("BENCH_POOL"))
    ap.add_argument("--engine", default=os.environ.get(
        "BENCH_ENGINE", "build/sager_engine"))
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--vram-mb", type=int, default=9984)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    tok_dir = ensure_tokenizer(TOK_DIR)
    tk = load_tokenizer(tok_dir)

    eng = Engine(args.engine, args.model, args.chunk, args.pool)
    ld, _ = eng.cmd({"cmd": "load", "model": args.model, "ctx": 8192,
                     "vram_mb": args.vram_mb, "threads": 2, "graphs": True,
                     "mtp": False, "chunk": args.chunk, "kv_q8": True},
                    timeout=1800)
    print("load:", json.dumps(ld)[:220], flush=True)

    n_pass = 0
    for pi, p in enumerate(PROMPTS):
        eng.cmd({"cmd": "reset"}, timeout=120)
        ids = tk.encode(chatml(p)).ids
        v, toks = eng.cmd({"cmd": "generate", "ids": ids,
                           "max_tokens": args.max_tokens,
                           "temperature": 0.0, "top_k": 20, "top_p": 0.95,
                           "greedy": True, "seed": 7,
                           "stop_ids": [IM_END]}, timeout=1800)
        if toks and toks[-1] == IM_END:
            toks = toks[:-1]
        text = tk.decode(toks)
        ratio, loop = coherence(text)
        ok = ratio >= 0.55 and loop <= 2
        n_pass += ok
        print("\n" + "=" * 72, flush=True)
        print("PROMPT %d: %s" % (pi + 1, p))
        print("-" * 72)
        print(text)
        print("-" * 72)
        print("gen %d tok | %.1f tok/s decode | letter/CJK %.2f | "
              "8-gram max run %d | %s"
              % (len(toks), v.get("decode_tps", 0.0), ratio, loop,
                 "PASS" if ok else "FAIL"), flush=True)
    eng.close()
    print("\nQUALITY GATE: %d/%d prompts passed" % (n_pass, len(PROMPTS)))
    sys.exit(0 if n_pass == len(PROMPTS) else 1)


if __name__ == "__main__":
    main()
