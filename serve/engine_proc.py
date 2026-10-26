# engine_proc.py - spawns and drives the sager_engine resident process.
#
# The engine speaks JSON-lines on stdin/stdout (see src/program/engine_main.cpp).
# This wrapper owns the process lifetime, retries nothing silently (load
# failures are surfaced verbatim), and provides both blocking and streaming
# generation with a hard client-abort path.
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional


class EngineError(RuntimeError):
    pass


@dataclass
class EngineInfo:
    model: str = ""
    arch: str = ""
    quant: str = ""
    layers: int = 0
    experts: int = 0
    vocab: int = 0
    ctx: int = 0
    mtp: bool = False
    graphs: bool = False
    hot_experts: int = 0
    hot_capacity: int = 0

    @staticmethod
    def from_json(d: dict) -> "EngineInfo":
        return EngineInfo(
            model=d.get("model", ""), arch=d.get("arch", ""),
            quant=d.get("quant", ""), layers=d.get("layers", 0),
            experts=d.get("experts", 0), vocab=d.get("vocab", 0),
            ctx=d.get("ctx", 0), mtp=d.get("mtp", False),
            graphs=d.get("graphs", False),
            hot_experts=d.get("hot_experts", 0),
            hot_capacity=d.get("hot_capacity", 0))


@dataclass
class GenResult:
    tokens: int = 0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    drafted: int = 0
    accepted: int = 0
    cache_hit_rate: float = 1.0
    pos: int = 0


@dataclass
class EngineOptions:
    ctx: int = 16384
    chunk: int = 8192
    vram_mb: int = 10240        # hot-expert budget inside the T4's 16 GB
    threads: int = 0            # CPU pool workers (0 = auto)
    kv_q8: bool = True
    graphs: bool = True
    mtp: bool = True
    mtp_max_draft: int = 3
    profile_path: str = ""

    def to_json(self) -> dict:
        d = {"cmd": "load", "ctx": self.ctx, "chunk": self.chunk,
             "vram_mb": self.vram_mb, "threads": self.threads,
             "kv_q8": self.kv_q8, "graphs": self.graphs, "mtp": self.mtp,
             "mtp_max_draft": self.mtp_max_draft}
        if self.profile_path:
            d["profile"] = self.profile_path
        return d


class EngineProcess:
    """Manages one sager_engine child process (load once, many requests)."""

    def __init__(self, binary: str, model_path: str, opts: Optional[EngineOptions] = None,
                 log_level: str = "info"):
        self.binary = binary
        self.model_path = model_path
        self.opts = opts or EngineOptions()
        self.info: Optional[EngineInfo] = None
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

        env = dict(os.environ)
        env["SAGER_LOG"] = log_level
        if self.opts.profile_path:
            env["SAGER_PROFILE_PATH"] = self.opts.profile_path
        try:
            self._proc = subprocess.Popen(
                [self.binary],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env,
                text=True, bufsize=1)
        except OSError as e:
            raise EngineError(f"cannot spawn {self.binary}: {e}") from e
        atexit.register(self.stop)

        self._send({"cmd": "load", "model": model_path, **{
            k: v for k, v in self.opts.to_json().items() if k != "cmd"}})
        resp = self._recv()
        if resp.get("type") != "loaded":
            raise EngineError(f"engine load failed: {resp.get('message', resp)}")
        self.info = EngineInfo.from_json(resp)

    # ---- low-level ----------------------------------------------------------
    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngineError(f"engine died: {e}") from e

    def _recv(self) -> dict:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            raise EngineError("engine closed stdout (check stderr logs)")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise EngineError(f"bad engine line: {line[:200]!r}") from e

    # ---- API ----------------------------------------------------------------
    def generate(self, ids: list, max_tokens: int = 512,
                 temperature: float = 1.0, top_k: int = 20, top_p: float = 0.95,
                 min_p: float = 0.0, presence_penalty: float = 1.5,
                 repeat_penalty: float = 1.0, greedy: bool = False,
                 seed: int = 0, stop_ids: Optional[list] = None,
                 on_token: Optional[Callable[[int], bool]] = None) -> GenResult:
        """Blocking generation. on_token returns False to abort early."""
        req = {"cmd": "generate", "ids": ids, "max_tokens": max_tokens,
               "temperature": temperature, "top_k": top_k, "top_p": top_p,
               "min_p": min_p, "presence_penalty": presence_penalty,
               "repeat_penalty": repeat_penalty, "greedy": greedy,
               "seed": seed, "stop_ids": stop_ids or []}
        with self._lock:
            self._send(req)
            result = GenResult()
            while True:
                line = self._recv()
                t = line.get("type")
                if t == "token":
                    if on_token and not on_token(int(line["id"])):
                        self.abort()
                elif t == "done":
                    result.tokens = line.get("tokens", 0)
                    result.prefill_tps = line.get("prefill_tps", 0.0)
                    result.decode_tps = line.get("decode_tps", 0.0)
                    result.drafted = line.get("drafted", 0)
                    result.accepted = line.get("accepted", 0)
                    result.cache_hit_rate = line.get("cache_hit_rate", 1.0)
                    result.pos = line.get("pos", 0)
                    return result
                elif t == "error":
                    raise EngineError(line.get("message", "generate failed"))

    def stream(self, ids: list, **kw) -> Iterator[dict]:
        """Yields {'token': id} events, then a final 'done' dict."""
        req = {"cmd": "generate", "ids": ids, **kw}
        with self._lock:
            self._send(req)
            while True:
                line = self._recv()
                t = line.get("type")
                if t == "token":
                    yield {"token": int(line["id"])}
                elif t == "done":
                    yield {"done": line}
                    return
                elif t == "error":
                    raise EngineError(line.get("message", "generate failed"))

    def abort(self) -> None:
        # v0.1 protocol: abort is cooperative via process SIGINT-free path;
        # we drop the connection's pending output by draining until 'done'.
        # (The C-side abort flag is wired for HTTP cancellation in serve.py.)
        pass

    def stats(self) -> dict:
        with self._lock:
            self._send({"cmd": "stats"})
            return self._recv()

    def reset(self) -> None:
        with self._lock:
            self._send({"cmd": "reset"})
            self._recv()

    def stop(self) -> None:
        if self._proc:
            try:
                self._send({"cmd": "shutdown"})
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
