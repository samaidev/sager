#!/usr/bin/env python3
# bench.py - SAGER benchmark: prefill/decode throughput + expert cache
# behavior on the target card.
#
#   python bench/bench.py --host http://localhost:8000 \
#       [--prompt-len 4096] [--gen 256] [--rounds 5]
#
# Reports the four numbers that matter for "35B MoE on one T4":
#   prefill tok/s, decode tok/s, hot-expert hit rate, spec-decode acceptance.
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request


def post(host: str, path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        host.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def make_prompt(tokenizer_words: int, seed: int) -> str:
    rng = random.Random(seed)
    # Synthetic agentic-style prompt: JSON scaffolding + natural language -
    # exercises both suffix drafting and the MTP drafter.
    words = ["def", "return", "config", "tool", "call", "result", "status",
             "user", "system", "http", "POST", "json", "value", "list",
             "模型", "推理", "专家", "缓存", "the", "of", "and", "run"]
    toks = [rng.choice(words) for _ in range(tokenizer_words)]
    lines = []
    for i in range(0, len(toks), 12):
        lines.append(" ".join(toks[i:i + 12]))
    return ("Process the following trace and summarize:\n```json\n"
            + "\n".join(lines) + "\n```")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()

    props = json.loads(urllib.request.urlopen(a.host.rstrip("/") + "/props",
                                              timeout=10).read())
    print(f"engine: {props['engine']['model']}")
    print(f"hot experts: {props['engine']['hot_experts']}/"
          f"{props['engine']['hot_capacity']} "
          f"({props['stats'].get('cache_hit_rate', 1):.1%} lifetime hit rate)")

    pre, dec = [], []
    for r in range(a.rounds):
        prompt = make_prompt(a.prompt_len, seed=1000 + r)
        t0 = time.time()
        res = post(a.host, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.gen, "sager_session": f"bench-{r}",
            "temperature": 1.0, "top_k": 20, "top_p": 0.95,
        })
        wall = time.time() - t0
        u = res["usage"]
        ptoks, ctoks = u["prompt_tokens"], u["completion_tokens"]
        pre.append(res["sager"]["prefill_tps"])
        dec.append(res["sager"]["decode_tps"])
        s = res["sager"]
        print(f"round {r + 1}: prefill {s['prefill_tps']:7.1f} tok/s | "
              f"decode {s['decode_tps']:6.1f} tok/s | hit "
              f"{s['cache_hit_rate']:6.1%} | spec {s['accepted']}/"
              f"{max(s['drafted'], 1)} | wall {wall:.1f}s")

    print(f"\nmedian: prefill {statistics.median(pre):.1f} tok/s | "
          f"decode {statistics.median(dec):.1f} tok/s "
          f"({a.prompt_len}-token prompts, {a.gen} generated)")


if __name__ == "__main__":
    main()
