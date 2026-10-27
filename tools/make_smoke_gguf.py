#!/usr/bin/env python3
"""make_smoke_gguf.py - synthetic Sage-like Q4_K GGUF for end-to-end engine smoke tests.

Geometry (passes validate_dims' >=1e9 approx_params envelope):
  hidden 2048, 16 heads / 4 kv (head_dim 128), 8 layers (all MoE),
  16 experts x top-2, expert ffn 1536, vocab 32000, ctx 4096.
Random weights (N(0, 0.02)); numerics are meaningless, the point is to
exercise the full GGUF -> VRAM -> graph capture -> prefill -> MoE dispatch
(hit + CPU-pool miss) -> sampling pipeline without NaNs.
"""
import numpy as np
import struct
import sys

H, HD, NH, NKV = 2048, 128, 16, 4
L, E, EU, FFE = 8, 16, 2, 1536
V, CTX = 32000, 4096
EPS, THETA, PARTIAL = 1e-6, 1e7, 0.25
SEED = 1234

QK_K, QK = 256, 32
# ggml tensor types
T_F32, T_F16, T_Q4_K = 0, 1, 12
# GGUF v3 metadata value types (spec order: f32=6, not the draft's 3)
T_ARR, T_STRING, T_UINT32, T_FLOAT32 = 9, 8, 4, 6

rng = np.random.default_rng(SEED)


def fp16_bytes(x):
    return np.ascontiguousarray(x, dtype=np.float16).tobytes()


def quant_q4k(w):
    """[rows, cols] f32 -> row-major Q4_K bytes. Vectorized anchored quant
    (same reconstruction contract as the kernels: w = d*d6*q - dmin*m6)."""
    assert w.shape[1] % 256 == 0, w.shape
    x = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    assert x.size % QK_K == 0
    nb = x.size // QK_K
    xb = x.reshape(nb, 8, QK)
    mn = xb.min(axis=2, keepdims=True)
    mx = xb.max(axis=2, keepdims=True)
    rng_ = np.maximum(mx - mn, 1e-30)
    d = rng_.max(axis=1, keepdims=True)[:, 0:1] / 63.0        # [nb,1]
    dmin = d
    d_ideal = (mx - mn) / 15.0
    is_ = np.clip(np.rint(d_ideal / d), 1, 63)                 # [nb,8,1]
    im_ = np.clip(np.rint(-mn / dmin), 0, 63)
    dd = d * is_
    mm = dmin * im_
    q = np.clip(np.rint((xb + mm) / dd), 0, 15).astype(np.uint8)
    qs = (q[:, 0::2, :] & 0xF) | ((q[:, 1::2, :] & 0xF) << 4)  # [nb,4,32] ggml order
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
    block[:, 2:4] = d16.reshape(nb, 2)   # dmin = d (sub-block scales carry the range)
    block[:, 4:16] = sc
    block[:, 16:] = qs.reshape(nb, 128)
    return block.tobytes()


class W:
    def __init__(self):
        self.kv = []
        self.tensors = []

    def add(self, k, v):
        if isinstance(v, bool):
            self.kv.append((k, 7, struct.pack("<B", 1 if v else 0)))
        elif isinstance(v, int):
            self.kv.append((k, T_UINT32, struct.pack("<I", v)))
        elif isinstance(v, float):
            self.kv.append((k, T_FLOAT32, struct.pack("<f", v)))
        elif isinstance(v, str):
            b = v.encode()
            self.kv.append((k, T_STRING, struct.pack("<Q", len(b)) + b))
        elif isinstance(v, list) and all(isinstance(x, int) for x in v):
            p = struct.pack("<I", T_UINT32) + struct.pack("<Q", len(v))
            self.kv.append((k, T_ARR, p + b"".join(struct.pack("<I", x) for x in v)))
        else:
            raise ValueError(k)

    def tensor(self, name, arr, qtype):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if qtype == T_Q4_K:
            data = quant_q4k(a)
            dims = [a.shape[1], a.shape[0]] if a.ndim == 2 else list(reversed(a.shape))
        elif qtype == T_F16:
            data = fp16_bytes(a)
            dims = list(reversed(a.shape))
        elif qtype == T_F32:
            data = a.tobytes()
            dims = list(reversed(a.shape))
        else:
            raise ValueError(qtype)
        assert len(data) % 32 == 0, (name, len(data))
        self.tensors.append((name, dims, qtype, data))
        print(f"  + {name} dims={dims} type={qtype} {len(data)/2**20:.1f} MiB", flush=True)

    def write(self, path):
        meta = b""
        for k, t, p in self.kv:
            meta += struct.pack("<Q", len(k)) + k.encode() + struct.pack("<I", t) + p
        td, off = b"", 0
        for name, dims, qt, data in self.tensors:
            td += struct.pack("<Q", len(name)) + name.encode()
            td += struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            td += struct.pack("<I", qt) + struct.pack("<Q", off)
            off += len(data)
        with open(path, "wb") as f:
            f.write(b"GGUF" + struct.pack("<I", 3) +
                    struct.pack("<Q", len(self.tensors)) + struct.pack("<Q", len(self.kv)))
            f.write(meta)
            f.write(td)
            for _, _, _, data in self.tensors:
                f.write(data)
        print(f"wrote {path}: {off/2**20:.1f} MiB, {len(self.tensors)} tensors, "
              f"{len(self.kv)} kv", flush=True)


def rand2(r, c, s=0.02):
    return (rng.standard_normal((r, c), dtype=np.float32) * s)


def main(out):
    w = W()
    w.add("general.architecture", "sager.sage_a15")
    w.add("general.name", "sage-smoke-1.3b")
    w.add("general.quantization", "Q4_K_M")
    w.add("sager.version", 1)
    w.add("sager.block_count", L)
    w.add("sager.moe_layer_count", L)
    w.add("sager.attention.head_count", NH)
    w.add("sager.attention.head_count_kv", NKV)
    w.add("sager.attention.key_length", HD)
    w.add("sager.attention.layer_norm_rms_epsilon", EPS)
    w.add("sager.embedding_length", H)
    w.add("sager.feed_forward_length", FFE)
    w.add("sager.expert_count", E)
    w.add("sager.expert_used_count", EU)
    w.add("sager.expert_feed_forward_length", FFE)
    w.add("sager.expert_shared_count", 1)
    w.add("sager.context_length", CTX)
    w.add("sager.vocab_size", V)
    w.add("sager.rope.theta", THETA)
    w.add("sager.rope.partial_rotary_factor", PARTIAL)
    w.add("sager.rope.mrope_section", [11, 11, 10])
    w.add("qwen3_5_moe.block_count", L)
    w.add("qwen3_5_moe.moe_layer_count", L)
    w.add("qwen3_5_moe.attention.head_count", NH)
    w.add("qwen3_5_moe.attention.head_count_kv", NKV)
    w.add("qwen3_5_moe.attention.key_length", HD)
    w.add("qwen3_5_moe.attention.layer_norm_rms_epsilon", EPS)
    w.add("qwen3_5_moe.embedding_length", H)
    w.add("qwen3_5_moe.vocab_size", V)
    w.add("qwen3_5_moe.rope.dimension_count", int(HD * PARTIAL))
    w.add("qwen3_5_moe.rope.freq_base", THETA)
    w.add("qwen3_5_moe.rope.scaling.type", "mrope")
    w.add("qwen3_5_moe.rope.scaling.section", [11, 11, 10])

    print("[1/3] emb + head + mtp", flush=True)
    w.tensor("token_embd.weight", rand2(V, H), T_F16)
    w.tensor("output_norm.weight", 1.0 + rand2(H, 1).reshape(-1) * 0.01, T_F16)
    w.tensor("output.weight", rand2(V, H), T_Q4_K)
    # MTP head: fc maps cat=[emb(t); h] (2H) -> H, proj_out H -> H. Both fp16;
    # the drafter reuses output_norm for its two rmsnorms (see mtp.cpp).
    w.tensor("mtp.fc.weight", rand2(2 * H, H), T_F16)
    w.tensor("mtp.proj_out.weight", rand2(H, H), T_F16)

    for l in range(L):
        print(f"[2/3] layer {l}", flush=True)
        p = f"blk.{l}."
        w.tensor(p + "attn_norm", 1.0 + rand2(H, 1).reshape(-1) * 0.01, T_F16)
        w.tensor(p + "ffn_norm", 1.0 + rand2(H, 1).reshape(-1) * 0.01, T_F16)
        w.tensor(p + "attn_q.weight", rand2(H, H), T_Q4_K)
        w.tensor(p + "attn_k.weight", rand2(NKV * HD, H), T_Q4_K)
        w.tensor(p + "attn_v.weight", rand2(NKV * HD, H), T_Q4_K)
        w.tensor(p + "attn_output.weight", rand2(H, H), T_Q4_K)
        w.tensor(p + "ffn_gate_inp", rand2(E, H, 0.05), T_F32)
        w.tensor(p + "ffn_gate_shexp", rand2(FFE, H), T_Q4_K)
        w.tensor(p + "ffn_up_shexp", rand2(FFE, H), T_Q4_K)
        w.tensor(p + "ffn_down_shexp", rand2(H, FFE), T_Q4_K)
        print(f"[3/3] layer {l} experts", flush=True)
        w.tensor(p + "ffn_gate_exps", rand2(E * FFE, H), T_Q4_K)
        w.tensor(p + "ffn_up_exps", rand2(E * FFE, H), T_Q4_K)
        w.tensor(p + "ffn_down_exps", rand2(E * H, FFE), T_Q4_K)

    w.write(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/content/smoke.gguf")
