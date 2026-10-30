#!/usr/bin/env python3
# convert_hf_to_sager.py - HuggingFace Sage-A1.5 -> SAGER Q4_K GGUF.
#
# Output contract (consumed by src/core/config.cpp + weights.cpp):
#   GGUF v3, little-endian, keys under BOTH "sager.*" and qwen-family names.
#   Tensor names:          Q4_K (12):
#     blk.{L}.attn_q/k/v/output.weight      attention projections
#     blk.{L}.ffn_gate_exps / ffn_up_exps / ffn_down_exps   [n_expert, ...]
#     blk.{L}.ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp shared expert
#   FP16 (1):
#     blk.{L}.attn_norm / ffn_norm, token_embd.weight, output_norm.weight
#     mtp.fc.weight [hidden, 2*hidden], mtp.proj_out.weight [hidden, hidden]
#   FP32 (0):
#     blk.{L}.ffn_gate_inp (router - tiny, kept exact)
#   Q4_K (12) head:
#     output.weight                          (v0.1 requires a Q4_K head)
#
# Usage:
#   python tools/convert_hf_to_sager.py <hf_dir> <out.gguf> [--workers N] [--fast]
#
# --fast: vectorized anchored quantization (no per-sub-block grid search).
# ~100x faster; slightly worse RMSE. Intended for CI / smoke pipelines -
# real conversions should keep the searched quantizer (default).
#
# Requires: numpy, safetensors (pip install numpy safetensors)
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

try:
    from safetensors import safe_open  # type: ignore
except ImportError:
    sys.exit("safetensors is required: pip install safetensors")

GGUF_MAGIC = b"GGUF"
T_F32, T_F16, T_Q4_K = 0, 1, 12
QK_K, QK, BLOCK_BYTES = 256, 32, 144

# GGUF metadata value types (spec / llama.cpp enum gguf_type order)
T_ARR, T_STRING, T_UINT32, T_FLOAT32, T_BOOL, T_UINT64, T_FLOAT64 = 9, 8, 4, 6, 7, 10, 12


# ---------------------------------------------------------------------------
# Quantization (matches src/kernels/q4_k.cpp quantize_row: anchored coarse/
# fine grid search on (scale, min) per 32-value sub-block, clamped codes)
# ---------------------------------------------------------------------------
def _fp16(x: float) -> np.uint16:
    return np.float16(x).view(np.uint16)


def quantize_q4_k_row(x: np.ndarray) -> bytes:
    """x: float32 [n], n % 256 == 0 -> packed Q4_K bytes."""
    n = x.shape[0]
    nb = n // QK_K
    out = bytearray()
    for b in range(nb):
        xb = x[b * QK_K:(b + 1) * QK_K]
        # super scale = max sub-block range (keeps 6-bit indices in range)
        ranges = []
        for s in range(8):
            sb = xb[s * QK:(s + 1) * QK]
            ranges.append(float(sb.max() - sb.min()))
        max_range = max(1e-30, max(ranges))
        d = max_range / 63.0
        dmin = d
        block = bytearray(4 + 12 + 128)
        struct.pack_into("<HH", block, 0, int(_fp16(d)), int(_fp16(dmin)))
        codes = [np.zeros(QK, dtype=np.uint8) for _ in range(8)]
        for s in range(8):
            sb = xb[s * QK:(s + 1) * QK]
            mn, mx = float(sb.min()), float(sb.max())
            if mx == mn:
                # constant sub-block: scale 1, min 0 -> all zeros
                _set_scale(block[4:], s, 1, 0)
                continue
            d_ideal = (mx - mn) / 15.0
            is_ideal = int(np.clip(round(d_ideal / d), 1, 63))
            im_ideal = int(np.clip(round(-mn / dmin), 0, 63))

            def err_of(is_: int, im_: int) -> float:
                dd, mm = d * is_, dmin * im_
                q = np.clip(np.rint((sb + mm) / dd), 0, 15)
                w = dd * q - mm
                return float(((sb - w) ** 2).sum())

            best = err_of(is_ideal, im_ideal)
            b_is, b_im = is_ideal, im_ideal
            for is_ in range(max(1, is_ideal - 16), min(63, is_ideal + 16) + 1, 4):
                for im in range(max(0, im_ideal - 16), min(63, im_ideal + 16) + 1, 4):
                    e = err_of(is_, im)
                    if e < best:
                        best, b_is, b_im = e, is_, im
            for is_ in range(max(1, b_is - 2), min(63, b_is + 2) + 1):
                for im in range(max(0, b_im - 2), min(63, b_im + 2) + 1):
                    e = err_of(is_, im)
                    if e < best:
                        best, b_is, b_im = e, is_, im
            _set_scale(block[4:], s, b_is, b_im)
            dd, mm = d * b_is, dmin * b_im
            codes[s] = np.clip(np.rint((sb + mm) / dd), 0, 15).astype(np.uint8)
        # ggml Q4_K byte pack (v0.12 fix): group g's 32 bytes carry sub-block
        # 2g in the LOW nibbles and sub-block 2g+1 in the HIGH nibbles.
        for g in range(4):
            block[16 + g * 32:16 + (g + 1) * 32] = bytes(
                (codes[2 * g][l] & 0xF) | ((codes[2 * g + 1][l] & 0xF) << 4)
                for l in range(32))
        out += block
    return bytes(out)


def _set_scale(scales: bytearray, j: int, d6: int, m6: int) -> None:
    """set_scale_min_k4 - identical bit layout to the C++ kernels."""
    if j < 4:
        scales[j] = (scales[j] & 0xC0) | (d6 & 63)
        scales[j + 4] = (scales[j + 4] & 0xC0) | (m6 & 63)
    else:
        scales[j + 4] = (scales[j + 4] & 0xF0) | (d6 & 0xF)
        scales[j - 4] = (scales[j - 4] & 0x3F) | ((d6 >> 4) << 6)
        scales[j + 4] = (scales[j + 4] & 0x0F) | ((m6 & 0xF) << 4)
        scales[j] = (scales[j] & 0x3F) | ((m6 >> 4) << 6)


def quantize_q4_k(w: np.ndarray) -> bytes:
    """w: [rows, cols] float32 -> row-major Q4_K bytes (cols % 256 == 0)."""
    rows = w.shape[0]
    return b"".join(quantize_q4_k_row(np.ascontiguousarray(w[r], dtype=np.float32))
                    for r in range(rows))


# Vectorized anchored quantization for --fast: same reconstruction contract
# (w = d*d6*q - dmin*m6, spec bit layout) without the coarse/fine search.
_QUANT_FAST = False   # set from --fast; inherited by forked pool workers


def quantize_q4_k_fast(w: np.ndarray) -> bytes:
    """[rows, cols] float32 -> Q4_K bytes, all super-blocks vectorized."""
    assert w.shape[1] % 256 == 0, w.shape
    x = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    nb = x.size // 256
    xb = x.reshape(nb, 8, 32)
    mn = xb.min(axis=2, keepdims=True)
    mx = xb.max(axis=2, keepdims=True)
    rng_ = np.maximum(mx - mn, 1e-30)
    d = rng_.max(axis=1, keepdims=True)[:, 0:1] / 63.0
    dmin = d
    is_ = np.clip(np.rint(((mx - mn) / 15.0) / d), 1, 63)
    im_ = np.clip(np.rint(-mn / dmin), 0, 63)
    dd = d * is_
    mm = dmin * im_
    q = np.clip(np.rint((xb + mm) / dd), 0, 15).astype(np.uint8)
    # ggml Q4_K byte pack (v0.12 fix): group g = sub-block 2g low | 2g+1 high.
    qs = (q[:, 0::2, :] & 0xF) | ((q[:, 1::2, :] & 0xF) << 4)   # [nb,4,32]
    sc = np.zeros((nb, 12), dtype=np.uint8)
    d6 = is_.astype(np.uint8).reshape(nb, 8)
    m6 = im_.astype(np.uint8).reshape(nb, 8)
    for j in range(4):
        sc[:, j] |= d6[:, j] & 63
        sc[:, j + 4] |= m6[:, j] & 63
    for j in range(4, 8):
        sc[:, j + 4] |= d6[:, j] & 0xF
        sc[:, j - 4] |= (d6[:, j] >> 4) << 6
        sc[:, j + 4] |= (m6[:, j] & 0xF) << 4
        sc[:, j] |= (m6[:, j] >> 4) << 6
    d16 = np.ascontiguousarray(d[:, 0], dtype=np.float16).view(np.uint8).reshape(-1)
    block = np.zeros((nb, 144), dtype=np.uint8)
    block[:, 0:2] = d16.reshape(nb, 2)
    block[:, 2:4] = d16.reshape(nb, 2)   # dmin = d
    block[:, 4:16] = sc
    block[:, 16:] = qs.reshape(nb, 128)
    return block.tobytes()


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------
class GGUFWriter:
    def __init__(self):
        self.kv: List[Tuple[str, tuple]] = []   # (name, (ggml_type, payload))
        # eager tensors: (name, dims, qtype, ready_bytes)
        self.tensors: List[Tuple[str, list, int, bytes]] = []
        # deferred tensors: (name, dims, qtype, provider_fn) - quantized
        # during write() so the big expert stacks never sit in RAM; the
        # on_release hook frees the HF shards as they are fully consumed.
        self.deferred: List[Tuple[str, list, int, object]] = []
        self.on_release = None   # callable(name) or None

    def add(self, key: str, value) -> None:
        if isinstance(value, bool):
            self.kv.append((key, (T_BOOL, struct.pack("<B", 1 if value else 0))))
        elif isinstance(value, int):
            self.kv.append((key, (T_UINT32, struct.pack("<I", value))))
        elif isinstance(value, float):
            self.kv.append((key, (T_FLOAT32, struct.pack("<f", value))))
        elif isinstance(value, str):
            b = value.encode()
            self.kv.append((key, (T_STRING, struct.pack("<Q", len(b)) + b)))
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            payload = struct.pack("<I", T_STRING) + struct.pack("<Q", len(value))
            for v in value:
                b = v.encode()
                payload += struct.pack("<Q", len(b)) + b
            self.kv.append((key, (T_ARR, payload)))
        elif isinstance(value, list) and all(isinstance(v, int) for v in value):
            payload = struct.pack("<I", T_UINT32) + struct.pack("<Q", len(value))
            for v in value:
                payload += struct.pack("<I", v)
            self.kv.append((key, (T_ARR, payload)))
        else:
            raise ValueError(f"unsupported kv type for {key}")

    @staticmethod
    def _encode(w: np.ndarray, qtype: int) -> Tuple[list, bytes]:
        if qtype == T_Q4_K:
            a = np.ascontiguousarray(w, dtype=np.float32)
            data = (quantize_q4_k_fast(a) if _QUANT_FAST else quantize_q4_k(a))
            dims = [w.shape[1], w.shape[0]]           # gguf: fastest first
        elif qtype == T_F16:
            data = np.ascontiguousarray(w, dtype=np.float16).tobytes()
            dims = list(reversed(w.shape))
        elif qtype == T_F32:
            data = np.ascontiguousarray(w, dtype=np.float32).tobytes()
            dims = list(reversed(w.shape))
        else:
            raise ValueError(qtype)
        return dims, data

    def add_tensor(self, name: str, w: np.ndarray, qtype: int) -> None:
        dims, data = self._encode(w, qtype)
        self.tensors.append((name, dims, qtype, data))

    def add_tensor_deferred(self, name: str, shape: tuple, qtype: int,
                            provider) -> None:
        """provider() -> bytes (quantized lazily inside write())."""
        dims = ([shape[1], shape[0]] if qtype == T_Q4_K
                else list(reversed(shape)))
        self.deferred.append((name, dims, qtype, provider))

    def write(self, path: str) -> None:
        align = 32
        # metadata
        meta = b""
        for key, (t, payload) in self.kv:
            meta += struct.pack("<Q", len(key)) + key.encode()
            meta += struct.pack("<I", t) + payload
        # tensor dir: names, dims, type, offset. Sizes of deferred tensors
        # come from their shapes (deterministic for every quant we emit),
        # so the whole directory is built before any data exists.
        all_tensors = [(n, d, q) for n, d, q, _ in self.tensors] + \
                      [(n, d, q) for n, d, q, _ in self.deferred]
        td = b""
        off = 0
        for name, dims, qtype in all_tensors:
            td += struct.pack("<Q", len(name)) + name.encode()
            td += struct.pack("<I", len(dims))
            for dd in dims:
                td += struct.pack("<Q", dd)
            td += struct.pack("<I", qtype)
            td += struct.pack("<Q", off)
            off += self._nbytes(dims, qtype)
            pad = (-off) % align
            off += pad
        header = GGUF_MAGIC + struct.pack("<I", 3)
        header += struct.pack("<Q", len(all_tensors))
        header += struct.pack("<Q", len(self.kv))
        with open(path, "wb") as f:
            f.write(header)
            f.write(meta)
            f.write(td)
            pad = b"\0" * align
            for name, dims, qtype, data in self.tensors:
                f.write(data)
                p = (-len(data)) % align
                if p:
                    f.write(pad[:p])
                if self.on_release:
                    self.on_release(name)
            for i, (name, dims, qtype, provider) in enumerate(self.deferred):
                data = provider()
                assert len(data) == self._nbytes(dims, qtype), \
                    f"{name}: {len(data)} != {self._nbytes(dims, qtype)}"
                f.write(data)
                p = (-len(data)) % align
                if p:
                    f.write(pad[:p])
                del data
                if self.on_release:
                    self.on_release(name)
        print(f"[convert] wrote {path} "
              f"({os.path.getsize(path) / 2**30:.2f} GiB, "
              f"{len(all_tensors)} tensors)")

    @staticmethod
    def _nbytes(dims: list, qtype: int) -> int:
        # dims = gguf order (fastest first); elements = product
        n = 1
        for dd in dims:
            n *= dd
        if qtype == T_Q4_K:
            return n // 256 * 144
        if qtype == T_F16:
            return n * 2
        if qtype == T_F32:
            return n * 4
        raise ValueError(qtype)


# ---------------------------------------------------------------------------
# HF -> SAGER mapping
# ---------------------------------------------------------------------------
def find_weights(hf_dir: str) -> Dict[str, str]:
    idx_path = os.path.join(hf_dir, "model.safetensors.index.json")
    files = set()
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        files.update(idx["weight_map"].values())
    else:
        files.update(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
    mapping = {}
    for fn in sorted(files):
        p = os.path.join(hf_dir, fn)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k in h:
            if k != "__metadata__":
                mapping[k] = p
    return mapping


class _RawST:
    """Minimal safetensors reader (numpy has no bfloat16, and the Sage
    checkpoint is bf16 - safe_open(framework="numpy") cannot materialize
    it). Parses each shard header once; get() returns float32 arrays."""

    def get(self, name):
        fn, v = self.keys_cache[name]
        p, base, _ = self.hdr[fn]
        s, e = v["data_offsets"]
        with open(p, "rb") as f:
            f.seek(base + s)
            raw = f.read(e - s)
        dt = v["dtype"]
        shape = tuple(int(x) for x in v["shape"])
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            return (u << 16).view(np.float32).reshape(shape)
        if dt == "F16":
            return (np.frombuffer(raw, dtype=np.uint16).copy().view(
                np.float16).astype(np.float32).reshape(shape))
        raise ValueError(f"unsupported dtype {dt} for {name}")

    def __init__(self, hf_dir: str, files):
        self.dir = hf_dir
        self.hdr = {}
        self.keys_cache = {}
        for fn in files:
            p = os.path.join(hf_dir, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            self.hdr[fn] = (p, 8 + n, h)
            for k, v in h.items():
                if k != "__metadata__":
                    self.keys_cache[k] = (fn, v)

    def shape(self, name):
        return tuple(self.keys_cache[name][1]["shape"])


def cfg_from_hf(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        c = json.load(f)
    tc = c.get("text_config", c)  # VL wrappers keep text params nested
    # layer kind: explicit layer_types list, else derive from
    # full_attention_interval (pattern: interval-1 full-attention layers of
    # every `interval` blocks; Sage-A1.5 = 3 GDN + 1 GQA repeating).
    n_layer = tc.get("num_hidden_layers", 48)
    ltypes = tc.get("layer_types") or []
    if len(ltypes) == n_layer:
        is_linear = [1 if t == "linear_attention" else 0 for t in ltypes]
    else:
        iv = tc.get("full_attention_interval", 0)
        if iv and iv > 1:
            is_linear = [0 if (i % iv == iv - 1) else 1 for i in range(n_layer)]
        else:
            is_linear = [0] * n_layer
    return {
        "n_layer": n_layer,
        "n_head": tc.get("num_attention_heads", 32),
        "n_head_kv": tc.get("num_key_value_heads", 4),
        "head_dim": tc.get("head_dim",
                           tc.get("hidden_size", 2048) // tc.get("num_attention_heads", 32)),
        "hidden": tc.get("hidden_size", 2048),
        "n_expert": tc.get("num_routed_experts", tc.get("num_experts", 128)),
        "n_expert_used": tc.get("num_experts_per_tok", 8),
        "n_ff_exp": tc.get("moe_intermediate_size", tc.get("intermediate_size", 768)),
        "n_vocab": tc.get("vocab_size", 151936),
        "n_ctx_train": tc.get("max_position_embeddings", 262144),
        "norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta": tc.get("rope_theta", 1e7),
        "rope_partial": tc.get("partial_rotary_factor", 0.25),
        "mrope": tc.get("rope_scaling", {}).get("type", "") == "mrope",
        "mrope_section": tc.get("rope_scaling", {}).get("mrope_section", [11, 11, 10]),
        "shared_expert": tc.get("shared_expert_intermediate_size", 768) > 0,
        # v0.2 gated delta net
        "layer_is_linear": is_linear,
        "gdn_k_heads": tc.get("linear_num_key_heads", 16),
        "gdn_v_heads": tc.get("linear_num_value_heads", 32),
        "gdn_key_dim": tc.get("linear_key_head_dim", 128),
        "gdn_value_dim": tc.get("linear_value_head_dim", 128),
        "gdn_conv_kernel": tc.get("linear_conv_kernel_dim", 4),
    }


def _convert_expert_layer(args) -> List[Tuple[str, np.ndarray, int]]:
    """Worker: stack + quantize the 3 expert tensors of one layer.

    HF stores per-expert matrices {moe}.{e}.{gate,up,down}_proj.weight
    (moe = "mlp" or "moe" depending on the modeling code), possibly spread
    across shards; SAGER wants stacked [n_expert, rows, cols] tensors sliced
    expert-major by the loader (weights.cpp load_expert_pool).
    """
    layer, mapping, n_expert, exp_prefix = args
    # exp_prefix already contains the layer path and ends with "experts."
    # (e.g. "model.language_model.layers.3.mlp.experts."); the expert loop
    # below appends the expert index directly.
    prefix = exp_prefix
    def stack(suffix: str) -> np.ndarray:
        mats = []
        probe_name = f"{prefix}0.{suffix}"
        with safe_open(mapping[probe_name], framework="numpy") as probe:
            rows, cols = probe.get_slice(probe_name).get_shape()
        for e in range(n_expert):
            name = f"{prefix}{e}.{suffix}"
            with safe_open(mapping[name], framework="numpy") as f:
                mats.append(f.get_tensor(name).astype(np.float32))
        return np.stack(mats, axis=0)
    return [
        (f"blk.{layer}.ffn_gate_exps", stack("gate_proj.weight"), T_Q4_K),
        (f"blk.{layer}.ffn_up_exps", stack("up_proj.weight"), T_Q4_K),
        (f"blk.{layer}.ffn_down_exps", stack("down_proj.weight"), T_Q4_K),
    ]


def convert(hf_dir: str, out_path: str, workers: int, fast: bool = False) -> None:
    global _QUANT_FAST
    _QUANT_FAST = fast
    cfg = cfg_from_hf(hf_dir)
    mapping = find_weights(hf_dir)
    w = GGUFWriter()

    # ---- source layout detection (v0.2) ------------------------------------
    # Sage-A1.5 ships "model.language_model.layers.*"; plain qwen-family
    # checkpoints use "model.layers.*". MoE submodule is "mlp" (qwen family)
    # or "moe" (some finetunes).
    if "model.language_model.layers.0.input_layernorm.weight" in mapping:
        base = "model.language_model."
    else:
        base = "model."
    def lp(layer: int) -> str:
        return f"{base}layers.{layer}."
    def moe_of(layer: int) -> str:
        p = lp(layer)
        for cand in ("mlp", "moe"):
            if f"{p}{cand}.gate.weight" in mapping:
                return f"{p}{cand}."
        raise KeyError(f"no mlp/moe router under {p}")
    def exp_base(layer: int) -> str:
        m = moe_of(layer)
        for cand in ("experts", "moe"):
            if f"{m}{cand}.0.gate_proj.weight" in mapping:
                return f"{m}{cand}."
        raise KeyError(f"no experts dir under {m}")

    _reader = _RawST(hf_dir, sorted(set(mapping.values())))

    def src(name: str) -> np.ndarray:
        return _reader.get(name)

    # ---- metadata ----
    w.add("general.architecture", "sager.sage_a15")
    w.add("general.name", os.path.basename(os.path.normpath(hf_dir)))
    w.add("general.quantization", "Q4_K_M")
    w.add("sager.version", 2)
    # Canonical "sager.*" namespace - the keys src/core/config.cpp reads first.
    w.add("sager.block_count", cfg["n_layer"])
    w.add("sager.moe_layer_count", cfg["n_layer"])
    w.add("sager.attention.head_count", cfg["n_head"])
    w.add("sager.attention.head_count_kv", cfg["n_head_kv"])
    w.add("sager.attention.key_length", cfg["head_dim"])
    w.add("sager.attention.layer_norm_rms_epsilon", float(cfg["norm_eps"]))
    w.add("sager.embedding_length", cfg["hidden"])
    w.add("sager.feed_forward_length", cfg["n_ff_exp"])
    w.add("sager.expert_count", cfg["n_expert"])
    w.add("sager.expert_used_count", cfg["n_expert_used"])
    w.add("sager.expert_feed_forward_length", cfg["n_ff_exp"])
    w.add("sager.expert_shared_count", 1 if cfg["shared_expert"] else 0)
    w.add("sager.context_length", cfg["n_ctx_train"])
    w.add("sager.vocab_size", cfg["n_vocab"])
    w.add("sager.rope.theta", float(cfg["rope_theta"]))
    w.add("sager.rope.partial_rotary_factor", float(cfg["rope_partial"]))
    if cfg["mrope"]:
        w.add("sager.rope.mrope_section", cfg["mrope_section"])
    # v0.2: hybrid layer kinds + gated delta net geometry
    w.add("sager.layer_is_linear", cfg["layer_is_linear"])
    w.add("sager.gdn.key_heads", cfg["gdn_k_heads"])
    w.add("sager.gdn.value_heads", cfg["gdn_v_heads"])
    w.add("sager.gdn.key_dim", cfg["gdn_key_dim"])
    w.add("sager.gdn.value_dim", cfg["gdn_value_dim"])
    w.add("sager.gdn.conv_kernel", cfg["gdn_conv_kernel"])
    # qwen-family aliases (community converters / fallback readers)
    w.add("qwen3_5_moe.block_count", cfg["n_layer"])
    w.add("qwen3_5_moe.moe_layer_count", cfg["n_layer"])
    w.add("qwen3_5_moe.expert_count", cfg["n_expert"])
    w.add("qwen3_5_moe.expert_used_count", cfg["n_expert_used"])
    w.add("qwen3_5_moe.expert_feed_forward_length", cfg["n_ff_exp"])
    w.add("qwen3_5_moe.expert_shared_count", 1 if cfg["shared_expert"] else 0)
    w.add("qwen3_5_moe.attention.head_count", cfg["n_head"])
    w.add("qwen3_5_moe.attention.head_count_kv", cfg["n_head_kv"])
    w.add("qwen3_5_moe.attention.key_length", cfg["head_dim"])
    w.add("qwen3_5_moe.attention.layer_norm_rms_epsilon", float(cfg["norm_eps"]))
    w.add("qwen3_5_moe.embedding_length", cfg["hidden"])
    w.add("qwen3_5_moe.vocab_size", cfg["n_vocab"])
    w.add("qwen3_5_moe.rope.dimension_count",
          int(cfg["head_dim"] * cfg["rope_partial"]))
    w.add("qwen3_5_moe.rope.freq_base", float(cfg["rope_theta"]))
    w.add("qwen3_5_moe.rope.scaling.type",
          "mrope" if cfg["mrope"] else "none")
    w.add("qwen3_5_moe.rope.scaling.section", cfg["mrope_section"])

    # ---- embedding / head / norms ----
    emb_name = f"{base}embed_tokens.weight"
    emb = src(emb_name)
    w.add_tensor("token_embd.weight", emb, T_F16)
    w.add_tensor("output_norm.weight",
                 src(f"{base}norm.weight").astype(np.float32), T_F16)
    head = src("lm_head.weight") if "lm_head.weight" in mapping \
        else src(emb_name)
    print(f"[convert] quantizing output head {head.shape} (Q4_K) ...")
    w.add_tensor("output.weight", head, T_Q4_K)

    # ---- MTP head (optional) ----
    if "mtp.fc.weight" in mapping:
        w.add_tensor("mtp.fc.weight", src("mtp.fc.weight"), T_F16)
    if "mtp.proj_out.weight" in mapping:
        w.add_tensor("mtp.proj_out.weight", src("mtp.proj_out.weight"), T_F16)

    L = cfg["n_layer"]
    for layer in range(L):
        p = lp(layer)
        linear = bool(cfg["layer_is_linear"][layer])
        attn_norm = src(p + "input_layernorm.weight")
        ffn_norm = src(p + "post_attention_layernorm.weight")
        w.add_tensor(f"blk.{layer}.attn_norm", attn_norm.astype(np.float32), T_F16)
        w.add_tensor(f"blk.{layer}.ffn_norm", ffn_norm.astype(np.float32), T_F16)
        if linear:
            # ---- gated delta net layer (v0.2) ----
            print(f"[convert] layer {layer + 1}/{L}: gated-delta ...")
            la = p + "linear_attn."
            # v0.12.8: the Sage-A1.5 checkpoint splits the fused GDN input
            # projections: in_proj_qkv [2*nk*K + ...? -> q|k|v] +
            # in_proj_z (the gate) + in_proj_b / in_proj_a (beta / a).
            # Rebuild the engine's fused [q|k|v|z] and [b|a] carves.
            qkv = src(la + "in_proj_qkv.weight")
            z = src(la + "in_proj_z.weight")
            if qkv.ndim == 1:
                qkv = qkv.reshape(-1, qkv.shape[-1] if qkv.ndim else 1)
            qkvz = np.vstack([np.asarray(qkv, dtype=np.float32),
                              np.asarray(z, dtype=np.float32)])
            w.add_tensor(f"blk.{layer}.lin_in_qkvz", qkvz, T_Q4_K)
            ba = np.vstack([np.asarray(src(la + "in_proj_b.weight"),
                                       dtype=np.float32),
                            np.asarray(src(la + "in_proj_a.weight"),
                                       dtype=np.float32)])
            w.add_tensor(f"blk.{layer}.lin_in_ba", ba, T_Q4_K)
            conv_w = src(la + "conv1d.weight")
            conv_w = np.asarray(conv_w).reshape(conv_w.shape[0], -1)  # [D,1,ck] -> [D,ck]
            w.add_tensor(f"blk.{layer}.lin_conv_w", conv_w, T_F16)
            if la + "conv1d.bias" in mapping:
                w.add_tensor(f"blk.{layer}.lin_conv_b",
                             src(la + "conv1d.bias").astype(np.float32), T_F16)
            # v0.12.8: the engine expects the DECAY already pre-negated
            # (A = -exp(A_log), see weights.hpp / C-GDN-3). The upstream GGUF
            # stored this correctly, our converter wrote the raw log - fix.
            w.add_tensor(f"blk.{layer}.lin_A_log",
                         (-np.exp(src(la + "A_log").astype(np.float64))
                          ).astype(np.float32), T_F32)
            w.add_tensor(f"blk.{layer}.lin_dt_bias",
                         src(la + "dt_bias").astype(np.float32), T_F32)
            # Sage-A1.5 has NO linear-attn q/k norm weights (bare L2, the
            # v0.2 engine behavior) - write them only if the checkpoint has.
            if la + "q_norm.weight" in mapping:
                w.add_tensor(f"blk.{layer}.lin_q_norm",
                             src(la + "q_norm.weight").astype(np.float32),
                             T_F16)
            if la + "k_norm.weight" in mapping:
                w.add_tensor(f"blk.{layer}.lin_k_norm",
                             src(la + "k_norm.weight").astype(np.float32),
                             T_F16)
            on_src = (la + "norm.weight") \
                if (la + "norm.weight") in mapping \
                else (la + "o_norm.weight")
            w.add_tensor(f"blk.{layer}.lin_o_norm",
                         src(on_src).astype(np.float32), T_F16)
            w.add_tensor(f"blk.{layer}.lin_out",
                         src(la + "out_proj.weight"), T_Q4_K)
        else:
            # ---- full attention layer (v0.1 path) ----
            print(f"[convert] layer {layer + 1}/{L}: attention ...")
            w.add_tensor(f"blk.{layer}.attn_q.weight", src(p + "self_attn.q_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.attn_k.weight", src(p + "self_attn.k_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.attn_v.weight", src(p + "self_attn.v_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.attn_output.weight", src(p + "self_attn.o_proj.weight"), T_Q4_K)
            if p + "self_attn.q_norm.weight" in mapping:
                w.add_tensor(f"blk.{layer}.attn_q_norm",
                             src(p + "self_attn.q_norm.weight").astype(np.float32), T_F16)
                w.add_tensor(f"blk.{layer}.attn_k_norm",
                             src(p + "self_attn.k_norm.weight").astype(np.float32), T_F16)
        print(f"[convert] layer {layer + 1}/{L}: router + shared ...")
        m = moe_of(layer)
        w.add_tensor(f"blk.{layer}.ffn_gate_inp",
                     src(m + "gate.weight").astype(np.float32), T_F32)
        if cfg["shared_expert"]:
            w.add_tensor(f"blk.{layer}.ffn_gate_shexp", src(m + "shared_expert.gate_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.ffn_up_shexp", src(m + "shared_expert.up_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.ffn_down_shexp", src(m + "shared_expert.down_proj.weight"), T_Q4_K)

    # ---- experts: DEFERRED providers (streamed during write()) ------------
    # v0.12.8: the HF bf16 checkpoint (~67 GB) + the Q4 GGUF (~21 GB) + the
    # pool do NOT fit on a 113 GB Colab disk simultaneously, so the expert
    # stacks are quantized lazily inside write() and each shard is unlinked
    # the moment its last consumed tensor is written (mmap of an unlinked
    # file stays valid, so mid-write deletion is safe).
    qexp = quantize_q4_k_fast if fast else quantize_q4_k
    pending: Dict[str, int] = {}
    for name, shard in mapping.items():
        pending[shard] = pending.get(shard, 0) + 1
    import os as _os

    def release(name: str) -> None:
        shard = mapping.get(name)
        if shard is None:
            return
        pending[shard] -= 1
        if pending[shard] == 0 and _os.path.exists(shard):
            _os.unlink(shard)
            print(f"[convert]   released {shard}")

    w.on_release = release
    for layer in range(L):
        prefix = exp_base(layer)
        for suffix, out_name in (("gate_proj.weight", "ffn_gate_exps"),
                                 ("up_proj.weight", "ffn_up_exps"),
                                 ("down_proj.weight", "ffn_down_exps")):
            names = [f"{prefix}{e}.{suffix}" for e in range(cfg["n_expert"])]
            shp = _reader.shape(names[0])
            rows, cols = int(shp[-2]), int(shp[-1])
            stack_shape = (cfg["n_expert"], rows, cols)

            def provider(names=names, rows=rows, cols=cols,
                         out_name=out_name):
                mats = np.empty((len(names), rows, cols), dtype=np.float32)
                for i, nm in enumerate(names):
                    mats[i] = _reader.get(nm)
                out = qexp(mats)
                print(f"[dbg] {out_name}: mats={mats.shape} "
                      f"bytes={len(out)}", flush=True)
                return out

            w.add_tensor_deferred(f"blk.{layer}.{out_name}", stack_shape,
                                  T_Q4_K, provider)
        print(f"[convert] layer {layer + 1}/{L}: experts deferred")

    w.write(out_path)
    print(f"[convert] done. Validate with: "
          f"python tools/validate_quant.py --gguf {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir")
    ap.add_argument("out_gguf")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--fast", action="store_true",
                    help="vectorized anchored quantization (no grid search); "
                         "~100x faster, slightly worse RMSE - for CI/smoke")
    a = ap.parse_args()
    convert(a.hf_dir, a.out_gguf, a.workers, a.fast)
