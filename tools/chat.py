#!/usr/bin/env python3
# chat.py - terminal chat against a running serve/server.py or the engine.
#
#   python tools/chat.py --host http://localhost:8000
#   python tools/chat.py --engine build/sager_engine --model model.gguf
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BLUE, GREEN, DIM, RESET = "\033[1;34m", "\033[1;32m", "\033[2m", "\033[0m"


def stream_chat(host: str, history: list) -> tuple:
    body = json.dumps({
        "messages": history, "stream": True, "sager_session": "cli",
        "temperature": 1.0, "top_k": 20, "top_p": 0.95,
        "presence_penalty": 1.5, "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req) as r:
        buf = ""
        while True:
            chunk = r.read(256).decode()
            if not chunk:
                break
            buf += chunk
            while "\n\n" in buf:
                line, buf = buf.split("\n\n", 1)
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return out
                j = json.loads(payload)
                d = j.get("choices", [{}])[0].get("delta", {}).get("content")
                if d:
                    out_text = d
                    out.append(out_text)
                    print(out_text, end="", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8000")
    a = ap.parse_args()

    print(f"{DIM}SAGER CLI chat - ctrl-C or 'exit' to quit{RESET}")
    history = []
    while True:
        try:
            user = input(f"{BLUE}you › {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.strip() in ("exit", "quit"):
            break
        if not user.strip():
            continue
        history.append({"role": "user", "content": user})
        print(f"{GREEN}sage › {RESET}", end="")
        try:
            reply = "".join(stream_chat(a.host, history))
        except Exception as e:  # noqa: BLE001
            print(f"{DIM}[error: {e}]{RESET}")
            history.pop()
            continue
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
