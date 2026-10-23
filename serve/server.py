# server.py - OpenAI-compatible HTTP server for SAGER.
#
#   python serve/server.py --model /models/sage-a15-q4_k_m.gguf \
#                          [--binary build/sager_engine] [--port 8000]
#
# Endpoints:
#   GET  /                      web chat UI (serve/web/chat.html)
#   GET  /props                 engine + expert-cache stats
#   POST /v1/chat/completions   OpenAI-compatible (stream and JSON modes)
#   POST /v1/completions        raw completions (prompt = text)
#
# Conversation retention: per-session token history with checkpointed
# prefixes (core/conversation.hpp semantics mirrored here) - follow-ups on
# the same thread re-prefill only the new suffix.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_proc import EngineError, EngineOptions, EngineProcess  # noqa: E402
from tokenizer import Tokenizer, load_tokenizer_for_model  # noqa: E402

DEFAULT_SYSTEM = (
    "You are Sage-A1.5, a helpful and precise assistant. "
    "Answer in the language of the user's message."
)

app = FastAPI(title="SAGER", version="0.1.0")
engine: Optional[EngineProcess] = None
tokenizer: Optional[Tokenizer] = None


# ---------------------------------------------------------------------------
# Session state: conversation history per client (cookie-less: client supplies
# a session_id; OpenAI clients get a fresh thread per request unless they pass
# the sager_session field).
# ---------------------------------------------------------------------------
class SessionState:
    __slots__ = ("ids", "last_ckpt", "ts")

    def __init__(self):
        self.ids: List[int] = []          # full committed token history
        self.last_ckpt: int = 0           # prefix length known resident in KV
        self.ts: float = time.time()


SESSIONS: Dict[str, SessionState] = {}
MAX_SESSIONS = 128


def get_session(sid: str) -> SessionState:
    s = SESSIONS.get(sid)
    if s is None:
        if len(SESSIONS) >= MAX_SESSIONS:
            oldest = min(SESSIONS, key=lambda k: SESSIONS[k].ts)
            del SESSIONS[oldest]
        s = SessionState()
        SESSIONS[sid] = s
    s.ts = time.time()
    return s


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def start_engine(args) -> None:
    global engine, tokenizer
    tok_dir = args.tokenizer_dir or os.path.dirname(os.path.abspath(args.model))
    tokenizer = load_tokenizer_for_model(args.model)
    if not tokenizer.exact:
        print("[sager] WARNING: no tokenizer.json found - tokenization is "
              "approximate. Point --tokenizer-dir at the HF snapshot.",
              file=sys.stderr)
    opts = EngineOptions(ctx=args.ctx, chunk=args.chunk, vram_mb=args.vram_mb,
                         threads=args.threads, kv_q8=not args.kv_fp16,
                         graphs=not args.no_graphs, mtp=not args.no_mtp)
    engine = EngineProcess(args.binary, args.model, opts,
                           log_level=os.environ.get("SAGER_LOG", "info"))
    print(f"[sager] engine ready: {engine.info.experts} experts, "
          f"{engine.info.hot_experts}/{engine.info.hot_capacity} hot resident, "
          f"graphs={'on' if engine.info.graphs else 'off'}, "
          f"mtp={'on' if engine.info.mtp else 'off'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trim_to_session(state: SessionState, prompt_ids: List[int]) -> List[int]:
    """Reuse the engine's KV when the request continues a known thread."""
    n = len(state.ids)
    if n and n <= len(prompt_ids) and prompt_ids[:n] == state.ids:
        return prompt_ids[n:]           # only the suffix is new
    state.ids = []
    state.last_ckpt = 0
    return prompt_ids


def build_generation_kwargs(body: dict) -> dict:
    msg = body.get("temperature", 1.0)
    return {
        "max_tokens": int(body.get("max_tokens", 512)),
        "temperature": float(msg) if msg is not None else 1.0,
        "top_k": int(body.get("top_k", 20)),
        "top_p": float(body.get("top_p", 0.95)),
        "min_p": float(body.get("min_p", 0.0)),
        "presence_penalty": float(body.get("presence_penalty", 1.5)),
        "repeat_penalty": float(body.get("repeat_penalty", 1.0)),
        "greedy": bool(body.get("greedy", False)),
        "seed": int(body.get("seed", 0)),
        "stop_ids": tokenizer.stop_ids(),
    }


def sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "web", "chat.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "web UI missing"})


@app.get("/props")
def props():
    st = engine.stats() if engine else {}
    return {
        "engine": engine.info.__dict__ if engine else None,
        "stats": st,
        "tokenizer_exact": tokenizer.exact if tokenizer else False,
        "sessions": len(SESSIONS),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages required")

    system = DEFAULT_SYSTEM
    turns = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", system)
        elif m.get("role") in ("user", "assistant"):
            turns.append({"role": m["role"], "content": m.get("content", "")})
    prompt_ids = tokenizer.apply_chat(system, turns)

    sid = body.get("sager_session") or f"req-{uuid.uuid4().hex[:12]}"
    state = get_session(sid)
    suffix_ids = trim_to_session(state, prompt_ids)

    kw = build_generation_kwargs(body)
    created = int(time.time())
    gen_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model = engine.info.model if engine else "sage-a1.5"

    if body.get("stream", False):
        async def event_stream():
            yield sse({"id": gen_id, "object": "chat.completion.chunk",
                       "created": created, "model": model,
                       "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]})
            try:
                # Streaming path: run in the engine thread, pump via polling.
                import queue as _q
                q: _q.Queue = _q.Queue()

                def worker():
                    try:
                        engine.generate(suffix_ids,
                                        on_token=lambda t: (q.put(t) or True), **kw)
                    finally:
                        q.put(None)

                import threading
                th = threading.Thread(target=worker, daemon=True)
                th.start()
                # Incremental decode: multi-byte UTF-8 spans tokens, so
                # decode the accumulated ids and emit only the stable delta.
                all_ids: List[int] = []
                sent = ""
                while True:
                    item = await run_in_threadpool(q.get)
                    if item is None:
                        break
                    all_ids.append(item)
                    text = tokenizer.decode(all_ids)
                    delta = text[len(sent):] if text.startswith(sent) else text
                    sent = text
                    if not delta:
                        continue
                    if request.is_disconnected:
                        engine.abort()
                        break
                    yield sse({"id": gen_id, "object": "chat.completion.chunk",
                               "created": created, "model": model,
                               "choices": [{"index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None}]})
            except EngineError as e:
                yield sse({"error": str(e)})
            yield sse({"id": gen_id, "object": "chat.completion.chunk",
                       "created": created, "model": model,
                       "choices": [{"index": 0, "delta": {},
                                    "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"
            state.ids = list(prompt_ids) + all_ids
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---- non-streaming ----
    from starlette.concurrency import run_in_threadpool
    gen_ids: List[int] = []

    def collect(t: int) -> bool:
        gen_ids.append(t)
        return True

    result = await run_in_threadpool(
        lambda: engine.generate(suffix_ids, on_token=collect, **kw))
    text = tokenizer.decode(gen_ids)
    state.ids = list(prompt_ids) + gen_ids
    return {
        "id": gen_id, "object": "chat.completion", "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": result.tokens,
            "total_tokens": len(prompt_ids) + result.tokens,
        },
        "sager": {
            "session": sid, "prefill_tps": result.prefill_tps,
            "decode_tps": result.decode_tps,
            "cache_hit_rate": result.cache_hit_rate,
            "drafted": result.drafted, "accepted": result.accepted,
        },
    }


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    kw = build_generation_kwargs(body)
    gen_ids: List[int] = []

    def collect(t: int) -> bool:
        gen_ids.append(t)
        return True

    result = await run_in_threadpool(
        lambda: engine.generate(ids, on_token=collect, **kw))
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.info.model,
        "choices": [{"index": 0, "text": tokenizer.decode(gen_ids),
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(ids),
                  "completion_tokens": result.tokens,
                  "total_tokens": len(ids) + result.tokens},
    }


# ---------------------------------------------------------------------------
from starlette.concurrency import run_in_threadpool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SAGER OpenAI-compatible server")
    ap.add_argument("--model", required=True, help="path to the Q4_K GGUF")
    ap.add_argument("--binary", default=None,
                    help="sager_engine binary (default: build/sager_engine)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--vram-mb", type=int, default=10240)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--kv-fp16", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--tokenizer-dir", default=None)
    args = ap.parse_args()
    if args.binary is None:
        args.binary = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "build", "sager_engine")

    start_engine(args)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
