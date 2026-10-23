#!/usr/bin/env python3
"""make_synthetic_hf.py - synthesize a Sage-A1.5-shaped HF checkpoint.

Produces config.json + model.safetensors with every tensor
convert_hf_to_sager.py reads, at a reduced geometry that still passes the
engine's validate_dims parameter envelope. Purpose: exercise the FULL
HF -> GGUF -> validate -> engine chain without the real 35B weights.

  python tests/make_synthetic_hf.py /tmp/hf_fake
  python tools/convert_hf_to_sager.py /tmp/hf_fake fake.gguf --fast
  python tools/validate_quant.py --gguf fake.gguf --hf /tmp/hf_fake --rows 8
"""
import json
import os
import sys

import numpy as np

H, HD, NH, NKV = 2048, 128, 16, 4
L, E, FFE = 8, 16, 1536
V, CTX = 32000, 4096

rng = np.random.default_rng(2024)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "architectures": ["SageA15MoeForCausalLM"],
        "model_type": "sage_a15_moe",
        "num_hidden_layers": L,
        "hidden_size": H,
        "num_attention_heads": NH,
        "num_key_value_heads": NKV,
        "head_dim": HD,
        "num_routed_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": FFE,
        "shared_expert_intermediate_size": FFE,
        "vocab_size": V,
        "max_position_embeddings": CTX,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e7,
        "partial_rotary_factor": 0.25,
        "rope_scaling": {"type": "mrope", "mrope_section": [11, 11, 10]},
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    tensors = {}

    def add(name, shape, scale=0.02, f16=True):
        a = rng.standard_normal(shape, dtype=np.float32) * scale
        if f16:
            a = a.astype(np.float16)
        tensors[name] = a
        return name

    print("embedding + head + mtp")
    add("model.embed_tokens.weight", (V, H))
    add("model.norm.weight", (H,), 0.01)
    add("lm_head.weight", (V, H), 0.08)
    add("mtp.fc.weight", (2 * H, H))
    add("mtp.proj_out.weight", (H, H))

    for l in range(L):
        p = f"model.layers.{l}."
        print(f"layer {l}")
        add(p + "input_layernorm.weight", (H,), 0.01)
        add(p + "post_attention_layernorm.weight", (H,), 0.01)
        add(p + "self_attn.q_proj.weight", (H, H))
        add(p + "self_attn.k_proj.weight", (NKV * HD, H))
        add(p + "self_attn.v_proj.weight", (NKV * HD, H))
        add(p + "self_attn.o_proj.weight", (H, H))
        add(p + "mlp.gate.weight", (E, H), 0.05)
        add(p + "mlp.shared_expert.gate_proj.weight", (FFE, H))
        add(p + "mlp.shared_expert.up_proj.weight", (FFE, H))
        add(p + "mlp.shared_expert.down_proj.weight", (H, FFE))
        for e in range(E):
            q = p + f"mlp.experts.{e}."
            add(q + "gate_proj.weight", (FFE, H))
            add(q + "up_proj.weight", (FFE, H))
            add(q + "down_proj.weight", (H, FFE))

    # safetensors.numpy wants contiguous arrays
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    from safetensors.numpy import save_file
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    total = sum(v.size for v in tensors.values())
    print(f"wrote {out_dir}: {len(tensors)} tensors, {total/1e9:.2f}B params, "
          f"{os.path.getsize(os.path.join(out_dir, 'model.safetensors'))/2**30:.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hf_fake")
