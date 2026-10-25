# GDN: the Gated Delta Net kernel (v0.2)

Sage-A1.5 is a HYBRID: per its config.json, `full_attention_interval=4` over
40 blocks - **30 gated-delta (linear attention) layers + 10 full-attention
layers**. v0.1 assumed every layer is GQA, which is why the engine could not
run the model at all. This document is the normative spec for the v0.2 GDN
path: `include/sager/kernels/gdn.hpp`, `src/kernels/cpu/gdn_cpu.cpp`
(reference), `src/kernels/cuda/gdn.cu` (sm_75).

## Geometry (metadata-driven, from the model card)

| param | value | GGUF key |
|---|---|---|
| k-heads | 16 | `sager.gdn.key_heads` |
| v-heads | 32 (ratio 2) | `sager.gdn.value_heads` |
| key dim K | 128 | `sager.gdn.key_dim` |
| value dim V | 128 | `sager.gdn.value_dim` |
| conv kernel | 4 | `sager.gdn.conv_kernel` |
| per-layer kind | 1=GDN | `sager.layer_is_linear` (u32 array) |

Derived: `conv_dim = 2*nk*K + nv*V` (8192), fused projection rows
`proj = conv_dim + nv*V` (12288), state per layer
`nk * (ratio*V) * K` fp32 = 2 MB (60 MB across 30 layers), conv state
`conv_dim * (ck-1)` fp16 (48 KB/layer).

## Conventions (C-GDN-*)

These pin the math. `scripts/gen_gdn_vectors.py` is the executable form
(numpy); `tests/test_gdn_host.cpp` enforces the same numbers in C++ and
includes golden vectors generated from it. Any change here MUST regenerate
the vectors and update both implementations.

- **C-GDN-1** q,k: per-head RMSNorm with a SHARED weight (`[K]`,
  `lin_q_norm`/`lin_k_norm`), then L2-normalize per head. (Qwen3-Next
  applies RMSNorm(head_dim) with one weight across heads.)
- **C-GDN-2** `beta = sigmoid(b)`, with `b` = first `nk` rows of the
  `in_proj_ba` output.
- **C-GDN-3** `g = -exp(A_log) * softplus(a + dt_bias)`; `alpha = exp(g)`;
  `a` = last `nk` rows of `in_proj_ba`. `A_log`/`dt_bias` are fp32
  per-k-head parameters.
- **C-GDN-4** state per k-head: `S` `[ratio*V x K]` fp32, row-major.
  ```
  u     = S @ k                      # [ratio*V]
  S     <- alpha*S - alpha*beta*u k^T + beta*v k^T
  y     = S @ q                      # INCLUSIVE: retrieve after write
  ```
  A k-head owns `ratio` consecutive v-heads, CONCATENATED along the value
  axis: `v_h = v[h*ratio*V : (h+1)*ratio*V]`.
- **C-GDN-5** causal depthwise conv (kernel `ck`) over the FIRST `conv_dim`
  rows of the fused projection (q|k|v parts; z bypasses), then SiLU:
  `out_d = silu(sum_j w[d,j] * win_d[j] + b_d)`,
  `win_d = [state(d,0..ck-2), x_d]`; state shifts by one per token.
- **C-GDN-6** output gating per v-head:
  `o = rmsnorm(y, lin_o_norm) * silu(z)`.

SiLU is `x * sigmoid(x) = x / (1 + e^-x)` - NOT `x/(1+|x|)` (softsign). A
v0.2-internal bug had the CPU reference using softsign; the golden vectors
caught it. Both must stay `x/(1+exp(-x))`.

## Layer dataflow (one token)

```
rmsnorm(attn_norm)
  -> Q4 GEMV in_qkvz  [proj x H] -> proj  [q | k | v | z]
  -> Q4 GEMV in_ba    [2nk  x H] -> ba    [b | a]
  -> gdn_conv_step    (conv state shift, SiLU)      -> conv [conv_dim]
  -> gdn_gate         (alpha/beta per k-head)       -> ab    [2nk]
  -> gdn_qk_norm      (RMSNorm + L2, per head)      -> qn,kn [nk*K]
  -> gdn_delta_step   (rank-1 update, fp32 state)   -> y     [nv*V]
  -> gdn_out_norm     (rmsnorm * silu(z))           -> on    [nv*V]
  -> Q4 GEMV out_proj [H x nv*V] -> residual add
```

The MoE half after the attention body is identical to full-attention layers
(Sage-A1.5: every block is MoE), so `moe_dispatch` is shared.

## CUDA notes (sm_75 / T4)

- `gdn_delta_step`: one block per k-head, one thread per state row
  (rows = ratio*V = 256). Row `j` computes `u_j = S[j,:].k`, updates the row
  in place and dots it with `q`. State traffic dominates:
  ~4 MB/layer/token read+write -> ~0.5 ms/token across 30 layers on a T4
  (300 GB/s), which is the expected steady-state cost of the recurrence.
- All kernels are fixed-shape, fixed-address: **the GDN body is captured
  into the same per-layer decode CUDA graphs as the dense half** (no
  data-dependent control flow, no KV, no position input). States are zeroed
  after warmup+capture and on `reset()`.
- fp32 state (NOT fp16): a half-precision recurrent state drifts measurably
  over long agentic contexts.
- Prefill walks tokens sequentially through the same kernels (correctness
  first). The chunked-parallel scan (FLA-style block-parallel form) is the
  v0.3 upgrade; at 8192-token chunks the sequential walk is launch-bound but
  correct, and GDN prefill FLOPs are ~64x smaller than full attention at
  this geometry anyway.

## Parity checklist (next GPU session)

1. `python tools/parity_check_gdn.py` vs HF `fla` reference on one real
   layer (conventions C-GDN-1..6, especially l2norm placement and beta
   source).
2. Engine logits vs llama.cpp on the same prompt, greedy, 32 tokens.
3. Confirm `softplus`/`sigmoid` dtypes match (fp32 params end-to-end).
