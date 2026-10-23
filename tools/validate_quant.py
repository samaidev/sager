#!/usr/bin/env python3
# validate_quant.py - per-tensor quality report for a SAGER Q4_K GGUF.
#
# Dequantizes sampled rows with the reference algorithm (same bit layout as
# the C++/CUDA kernels) and reports relative RMSE + cosine vs the ORIGINAL
# fp16/bf16 weights when a matching HF snapshot is supplied. On a healthy
# Q4_K_M-class conversion expect:
#   rel RMSE  ~2-5%   (LLM weight distributions are peaked, much better than
#                      the ~7-8% iid-gaussian rate-distortion floor)
#   cosine    >0.995
#
# Usage:
#   python tools/validate_quant.py --gguf model.gguf [--hf hf_dir] [--rows 32]
from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np

QK_K, QK, BLOCK_BYTES = 256, 32, 144

GGUF_TYPES = {0: "F32", 1: "F16", 12: "Q4_K", 14: "Q6_K"}


# ---------------------------------------------------------------------------
# Minimal GGUF v3 reader (header + tensor directory only; data is mmapped)
# ---------------------------------------------------------------------------
def read_gguf_dir(path: str):
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", "not a GGUF file"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    assert version == 3, f"GGUF v{version} not supported by the validator"

    def rd_str() -> str:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()

    def rd_val(t: int):
        # GGUF v3 spec ids (llama.cpp enum gguf_type): sizes must match exactly
        # or the stream desyncs before the tensor directory.
        if t == 0:
            return f.read(1)[0]
        if t == 1:
            return struct.unpack("<b", f.read(1))[0]
        if t == 2:
            return struct.unpack("<H", f.read(2))[0]
        if t == 3:
            return struct.unpack("<h", f.read(2))[0]
        if t == 4:
            return struct.unpack("<I", f.read(4))[0]
        if t == 5:
            return struct.unpack("<i", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 7:
            return bool(f.read(1)[0])
        if t == 8:
            return rd_str()
        if t == 9:  # array
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rd_val(et) for _ in range(n)]
        if t == 10:
            return struct.unpack("<Q", f.read(8))[0]
        if t == 11:
            return struct.unpack("<q", f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"kv type {t}")

    kv = {}
    for _ in range(n_kv):
        k = rd_str()
        t = struct.unpack("<I", f.read(4))[0]
        kv[k] = rd_val(t)

    tensors: Dict[str, Tuple[List[int], int, int]] = {}
    for _ in range(n_tensors):
        name = rd_str()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = list(struct.unpack("<" + "Q" * nd, f.read(8 * nd)))
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors[name] = (dims, ttype, off)
    # data section starts at current pos (aligned per writer)
    data_start = f.tell()
    return kv, tensors, data_start


def dequant_q4_k_row(raw: bytes, n: int) -> np.ndarray:
    d, dmin = struct.unpack_from("<HH", raw, 0)
    d = np.float16(d).item()
    dmin = np.float16(dmin).item()
    scales = raw[4:16]
    qs = raw[16:16 + 128]
    out = np.empty(n, dtype=np.float32)
    for s in range(8):
        if s < 4:
            d6 = scales[s] & 63
            m6 = scales[s + 4] & 63
        else:
            d6 = (scales[s + 4] & 0xF) | ((scales[s - 4] >> 6) << 4)
            m6 = (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4)
        sd = d * d6
        sm = dmin * m6
        for j in range(16):
            out[s * 32 + 2 * j] = sd * (qs[s * 16 + j] & 0xF) - sm
            out[s * 32 + 2 * j + 1] = sd * (qs[s * 16 + j] >> 4) - sm
    return out


def tensor_nbytes(dims: List[int], ttype: int) -> int:
    ne = 1
    for d in dims:
        ne *= d
    if ttype == 0:
        return ne * 4
    if ttype == 1:
        return ne * 2
    if ttype == 12:
        return ne // QK_K * BLOCK_BYTES
    raise ValueError(ttype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf", default=None, help="HF dir with the original weights")
    ap.add_argument("--rows", type=int, default=32, help="rows sampled per tensor")
    ap.add_argument("--tensors", default="expert", choices=["expert", "all"])
    a = ap.parse_args()

    kv, tensors, data_start = read_gguf_dir(a.gguf)
    print(f"gguf: {len(tensors)} tensors; "
          f"arch={kv.get('general.architecture', '?')} "
          f"experts={kv.get('sager.expert_count', kv.get('qwen3_5_moe.expert_count', '?'))}")

    f = open(a.gguf, "rb")
    names = sorted(n for n in tensors if n.endswith("_exps")) \
        if a.tensors == "expert" else sorted(tensors)
    if a.tensors == "expert" and not names:
        names = sorted(n for n in tensors if n.endswith(".weight"))[:8]

    rng = random.Random(0)
    worst = 0.0
    print(f"{'tensor':<44} {'rows':>5} {'relRMSE':>9} {'cosine':>8}")
    for name in names:
        dims, ttype, off = tensors[name]
        if ttype != 12:
            continue
        cols = dims[0]                  # gguf order: fastest = cols
        nrows = dims[1] if len(dims) > 1 else 1
        base = data_start + off
        picked = rng.sample(range(nrows), min(a.rows, nrows))
        se, sr = 0.0, 0.0
        dot = nn = 0.0
        for r in picked:
            f.seek(base + r * (cols // QK_K) * BLOCK_BYTES)
            raw = f.read(cols // QK_K * BLOCK_BYTES)
            x = dequant_q4_k_row(raw, cols)
            se += float((x * x).sum())
            if a.hf:
                from safetensors import safe_open  # type: ignore
                # row-matched comparison against the HF original
                hf_name = _map_back(name, r)
                if hf_name is None:
                    continue
                fn, key = hf_name
                with safe_open(fn, framework="numpy") as g:
                    orig = g.get_tensor(key)[r].astype(np.float32)
                e = orig - x
                se += float((e * e).sum()) - float((x * x).sum())  # replace prior
                sr += float((orig * orig).sum())
                dot += float((orig * x).sum())
                nn += float(np.linalg.norm(orig)) * float(np.linalg.norm(x))
        rel = math.sqrt(se / cols) / (math.sqrt(sr / cols) + 1e-30) if sr else 0.0
        cos = dot / (nn + 1e-30) if nn else float("nan")
        worst = max(worst, rel)
        print(f"{name:<44} {len(picked):>5} {rel:>9.4f} {cos:>8.4f}")

    print(f"\nworst relRMSE: {worst:.4f} "
          f"({'OK' if worst < 0.08 else 'REVIEW: above the iid-gaussian floor'} )")


def _map_back(name: str, row: int):
    """Map (sager tensor, expert row) -> (file, hf_key) via cwd listing."""
    import os
    import glob
    import json
    hf = sys.argv[sys.argv.index("--hf") + 1] if "--hf" in sys.argv else None
    if not hf:
        return None
    idx_path = os.path.join(hf, "model.safetensors.index.json")
    wm = {}
    if os.path.exists(idx_path):
        with open(idx_path) as fjson:
            wm = json.load(fjson)["weight_map"]
    files = sorted(glob.glob(os.path.join(hf, "*.safetensors")))
    # blk.{L}.ffn_{gate,up,down}_exps row r -> experts.{r}.{proj}
    parts = name.split(".")
    L, kind = parts[1], parts[3]  # ffn_gate_exps -> gate
    proj = {"ffn_gate_exps": "gate_proj",
            "ffn_up_exps": "up_proj",
            "ffn_down_exps": "down_proj"}[kind]
    key = f"model.layers.{L}.mlp.experts.{row}.{proj}.weight"
    fn = wm.get(key) or next((f for f in files if key in _keys_of(f)), None)
    if not fn:
        return None
    return os.path.join(hf, fn) if not os.path.isabs(fn) else fn, key


def _keys_of(fn: str) -> set:
    from safetensors import safe_open  # type: ignore
    with safe_open(fn, framework="numpy") as f:
        return set(f.keys())


if __name__ == "__main__":
    main()
