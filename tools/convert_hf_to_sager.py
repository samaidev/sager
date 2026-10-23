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
#   python tools/convert_hf_to_sager.py <hf_dir> <out.gguf> [--workers N]
#
# Requires: torch, safetensors (pip install torch safetensors)
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
    import torch  # noqa: F401
except ImportError:
    sys.exit("torch is required: pip install torch")

from safetensors import safe_open  # type: ignore

GGUF_MAGIC = b"GGUF"
T_F32, T_F16, T_Q4_K = 0, 1, 12
QK_K, QK, BLOCK_BYTES = 256, 32, 144

# GGUF metadata value types
T_ARR, T_STRING, T_UINT32, T_FLOAT32, T_BOOL, T_UINT64, T_FLOAT64 = 9, 8, 4, 3, 7, 10, 5


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
            q = np.clip(np.rint((sb + mm) / dd), 0, 15).astype(np.uint8)
            qs = bytes((q[2 * j] & 0xF) | ((q[2 * j + 1] & 0xF) << 4)
                       for j in range(16))
            block[16 + s * 16:16 + (s + 1) * 16] = qs
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


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------
class GGUFWriter:
    def __init__(self):
        self.kv: List[Tuple[str, tuple]] = []   # (name, (ggml_type, payload))
        self.tensors: List[Tuple[str, list, int, bytes]] = []

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

    def add_tensor(self, name: str, w: np.ndarray, qtype: int) -> None:
        if qtype == T_Q4_K:
            data = quantize_q4_k(np.ascontiguousarray(w, dtype=np.float32))
            dims = [w.shape[1], w.shape[0]]           # gguf: fastest first
        elif qtype == T_F16:
            data = np.ascontiguousarray(w, dtype=np.float16).tobytes()
            dims = list(reversed(w.shape))
        elif qtype == T_F32:
            data = np.ascontiguousarray(w, dtype=np.float32).tobytes()
            dims = list(reversed(w.shape))
        else:
            raise ValueError(qtype)
        self.tensors.append((name, dims, qtype, data))

    def write(self, path: str) -> None:
        align = 32
        # metadata
        meta = b""
        for key, (t, payload) in self.kv:
            meta += struct.pack("<Q", len(key)) + key.encode()
            meta += struct.pack("<I", t) + payload
        # tensor dir (names, dims, type, offset); offsets aligned
        td = b""
        off = 0
        offsets = []
        for name, dims, qtype, data in self.tensors:
            nb = len(data)
            pad = (-off) % align
            off += pad
            offsets.append(pad)
            td += struct.pack("<Q", len(name)) + name.encode()
            td += struct.pack("<I", len(dims))
            for dd in dims:
                td += struct.pack("<Q", dd)
            td += struct.pack("<I", qtype)
            td += struct.pack("<Q", off)
            off += nb
        # ...but padding must live in the data section; simplest contract:
        # no padding - align=32 respected by construction (sizes are multiples)
        header = GGUF_MAGIC + struct.pack("<I", 3)
        header += struct.pack("<Q", len(self.tensors))
        header += struct.pack("<Q", len(self.kv))
        with open(path, "wb") as f:
            f.write(header)
            f.write(meta)
            f.write(td)
            for i, (name, dims, qtype, data) in enumerate(self.tensors):
                f.write(data)
        print(f"[convert] wrote {path} "
              f"({os.path.getsize(path) / 2**30:.2f} GiB, "
              f"{len(self.tensors)} tensors)")


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
        with safe_open(os.path.join(hf_dir, fn), framework="numpy") as f:
            for k in f.keys():
                mapping[k] = os.path.join(hf_dir, fn)
    return mapping


def cfg_from_hf(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        c = json.load(f)
    tc = c.get("text_config", c)  # VL wrappers keep text params nested
    return {
        "n_layer": tc.get("num_hidden_layers", 48),
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
    }


def _convert_expert_layer(args) -> List[Tuple[str, np.ndarray, int]]:
    """Worker: stack + quantize the 3 expert tensors of one layer.

    HF stores per-expert matrices mlp.experts.{e}.{gate,up,down}_proj.weight,
    possibly spread across shards; SAGER wants stacked [n_expert, rows, cols]
    tensors sliced expert-major by the loader (weights.cpp load_expert_pool).
    """
    layer, mapping, n_expert = args
    prefix = f"model.layers.{layer}.mlp.experts."
    def stack(suffix: str) -> np.ndarray:
        mats = []
        with safe_open(mapping[f"{prefix}0.{suffix}"], framework="numpy") as probe:
            rows, cols = probe.get_slice(f"{prefix}0.{suffix}").get_shape()
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


def convert(hf_dir: str, out_path: str, workers: int) -> None:
    cfg = cfg_from_hf(hf_dir)
    mapping = find_weights(hf_dir)
    w = GGUFWriter()

    def src(name: str) -> np.ndarray:
        return safe_open(mapping[name], framework="numpy").get_tensor(name)

    # ---- metadata ----
    w.add("general.architecture", "sager.sage_a15")
    w.add("general.name", os.path.basename(os.path.normpath(hf_dir)))
    w.add("general.quantization", "Q4_K_M")
    w.add("sager.version", 1)
    w.add("sager.block_count", cfg["n_layer"])
    w.add("sager.expert_count", cfg["n_expert"])
    w.add("sager.expert_used_count", cfg["n_expert_used"])
    w.add("sager.expert_feed_forward_length", cfg["n_ff_exp"])
    w.add("sager.shared_expert", 1 if cfg["shared_expert"] else 0)
    w.add("sager.context_length", cfg["n_ctx_train"])
    # qwen-family aliases (dual namespace, config.cpp reads both)
    w.add("qwen3_5_moe.block_count", cfg["n_layer"])
    w.add("qwen3_5_moe.expert_count", cfg["n_expert"])
    w.add("qwen3_5_moe.expert_used_count", cfg["n_expert_used"])
    w.add("qwen3_5_moe.expert_feed_forward_length", cfg["n_ff_exp"])
    w.add("qwen3_5_moe.attention.head_count", cfg["n_head"])
    w.add("qwen3_5_moe.attention.head_count_kv", cfg["n_head_kv"])
    w.add("qwen3_5_moe.attention.key_length", cfg["head_dim"])
    w.add("qwen3_5_moe.embedding_length", cfg["hidden"])
    w.add("qwen3_5_moe.vocab_size", cfg["n_vocab"])
    w.add("qwen3_5_moe.rope.dimension_count",
          int(cfg["head_dim"] * cfg["rope_partial"]))
    w.add("qwen3_5_moe.rope.freq_base", float(cfg["rope_theta"]))
    w.add("qwen3_5_moe.rope.scaling.type",
          "mrope" if cfg["mrope"] else "none")
    w.add("qwen3_5_moe.rope.scaling.section", cfg["mrope_section"])
    w.add("qwen3_5_moe.attention.rms_norm_eps", float(cfg["norm_eps"]))

    def shp(name: str) -> Tuple[int, ...]:
        with safe_open(mapping[name], framework="numpy") as f:
            return f.get_slice(name).get_shape()

    # ---- embedding / head / norms ----
    emb = src("model.embed_tokens.weight")
    w.add_tensor("token_embd.weight", emb, T_F16)
    w.add_tensor("output_norm.weight",
                 src("model.norm.weight").astype(np.float32), T_F16)
    head = src("lm_head.weight") if "lm_head.weight" in mapping \
        else src("model.embed_tokens.weight")
    print(f"[convert] quantizing output head {head.shape} (Q4_K) ...")
    w.add_tensor("output.weight", head, T_Q4_K)

    # ---- MTP head (optional) ----
    if "mtp.fc.weight" in mapping:
        w.add_tensor("mtp.fc.weight", src("mtp.fc.weight"), T_F16)
    if "mtp.proj_out.weight" in mapping:
        w.add_tensor("mtp.proj_out.weight", src("mtp.proj_out.weight"), T_F16)

    L = cfg["n_layer"]
    for layer in range(L):
        p = f"model.layers.{layer}."
        attn_norm = src(p + "input_layernorm.weight")
        ffn_norm = src(p + "post_attention_layernorm.weight")
        w.add_tensor(f"blk.{layer}.attn_norm", attn_norm.astype(np.float32), T_F16)
        w.add_tensor(f"blk.{layer}.ffn_norm", ffn_norm.astype(np.float32), T_F16)
        print(f"[convert] layer {layer + 1}/{L}: attention ...")
        w.add_tensor(f"blk.{layer}.attn_q.weight", src(p + "self_attn.q_proj.weight"), T_Q4_K)
        w.add_tensor(f"blk.{layer}.attn_k.weight", src(p + "self_attn.k_proj.weight"), T_Q4_K)
        w.add_tensor(f"blk.{layer}.attn_v.weight", src(p + "self_attn.v_proj.weight"), T_Q4_K)
        w.add_tensor(f"blk.{layer}.attn_output.weight", src(p + "self_attn.o_proj.weight"), T_Q4_K)
        print(f"[convert] layer {layer + 1}/{L}: router + shared ...")
        w.add_tensor(f"blk.{layer}.ffn_gate_inp",
                     src(p + "mlp.gate.weight").astype(np.float32), T_F32)
        if cfg["shared_expert"]:
            w.add_tensor(f"blk.{layer}.ffn_gate_shexp", src(p + "mlp.shared_expert.gate_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.ffn_up_shexp", src(p + "mlp.shared_expert.up_proj.weight"), T_Q4_K)
            w.add_tensor(f"blk.{layer}.ffn_down_shexp", src(p + "mlp.shared_expert.down_proj.weight"), T_Q4_K)

    # experts in parallel (the expensive part)
    print(f"[convert] quantizing {L} expert stacks with {workers} workers ...")
    jobs = [(layer, mapping, cfg["n_expert"]) for layer in range(L)]
    if workers <= 1:
        for j in jobs:
            for name, arr, t in _convert_expert_layer(j):
                print(f"[convert]   {name} {arr.shape}")
                w.tensors.append((name, list(reversed(arr.shape)), t,
                                  quantize_q4_k(np.ascontiguousarray(arr, dtype=np.float32))))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_convert_expert_layer, jobs):
                for name, arr, t in res:
                    print(f"[convert]   {name} {arr.shape}")
                    w.tensors.append((name, list(reversed(arr.shape)), t,
                                      quantize_q4_k(np.ascontiguousarray(arr, dtype=np.float32))))

    w.write(out_path)
    print(f"[convert] done. Validate with: "
          f"python tools/validate_quant.py --gguf {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir")
    ap.add_argument("out_gguf")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()
    convert(a.hf_dir, a.out_gguf, a.workers)
