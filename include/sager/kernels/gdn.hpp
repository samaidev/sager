// gdn.hpp - Gated Delta Net (linear attention) kernels for Sage-A1.5's
// hybrid blocks (30 of 40 layers are gated-delta layers; see config.json
// linear_* fields and full_attention_interval=4).
//
// This is THE piece v0.1 was missing: without it the engine cannot run the
// model at all (v0.1 assumed every layer is GQA and the converter hit
// KeyError on linear_attn.* tensors).
//
// Spec conventions (docs/gdn-kernel.md is the normative text; the numpy
// generator scripts/gen_gdn_vectors.py is the executable form):
//   C-GDN-1  q,k: per-head RMSNorm(w) then L2-normalize
//   C-GDN-2  beta = sigmoid(b)
//   C-GDN-3  g = -exp(A_log) * softplus(a + dt_bias);  alpha = exp(g)
//   C-GDN-4  per k-head state S [ratio*V x K] (row-major, rows = value slots):
//              u     = S @ k
//              S     <- alpha*S - alpha*beta*u k^T + beta*v k^T
//              y     = S @ q        (INCLUSIVE: retrieve from the post-write
//                                    state - matches linear-attn convention
//                                    where token t sees its own write)
//   C-GDN-5  causal depthwise conv (kernel ck, default 4) over the qkv part
//            of the fused projection, then SiLU; z bypasses the conv.
//   C-GDN-6  output gating: o = rmsnorm(y, w_on) * silu(z), per v-head.
//
// Layouts (one token, fp16 unless noted):
//   fused proj out  proj[2*nk*K + 2*nv*V] = [q | k | v | z]   (q,k: nk heads;
//                     v,z: nv heads) - q/k/v parts are conv_dim wide.
//   ba              ba[2*nk] = [b | a]
//   conv weight     w[conv_dim * ck] row-major (row d: ck taps)
//   conv state      st[conv_dim * (ck-1)]  (previous raw conv inputs)
//   delta state     S[nk * (ratio*V) * K] fp32 (per k-head [rows x K])
//   gate out        ab[nk * 2] = {alpha, beta} fp32
//   y               y[nv * V] fp16 (nk blocks of ratio*V rows)
//   v slice         proj + 2*nk*K ... conv applies to the first conv_dim rows,
//                   so v = conv_out + 2*nk*K, z = proj + conv_dim.
#pragma once

#include <cstdint>

#include "sager/common.hpp"

namespace sager {

struct GdnParams {
  int n_k_heads    = 16;
  int n_v_heads    = 32;    // must be a multiple of n_k_heads
  int key_dim      = 128;
  int value_dim    = 128;
  int conv_kernel  = 4;

  int ratio()        const { return n_v_heads / n_k_heads; }
  int conv_dim()     const { return 2 * n_k_heads * key_dim +
                                    n_v_heads * value_dim; }
  int proj_rows()    const { return conv_dim() + n_v_heads * value_dim; }
  int ba_rows()      const { return 2 * n_k_heads; }
  int y_rows()       const { return n_v_heads * value_dim; }
  // fp32 delta-state floats per layer: nk * (ratio*V) * K
  size_t state_elems() const {
    return static_cast<size_t>(n_k_heads) * ratio() * value_dim * key_dim;
  }
  size_t conv_state_elems() const {
    return static_cast<size_t>(conv_dim()) * (conv_kernel - 1);
  }
};

// ---- CPU reference (tests + parity workloads; plain fp32, no SIMD) ---------
//
// All functions operate on ONE token. fp16 inputs are converted internally.
namespace gdn_cpu {

// C-GDN-2/3: ba[2nk] (fp16) + A_log/dt_bias (fp32) -> ab[2nk] = {alpha,beta}.
void gate(const uint16_t* ba, const float* A_log, const float* dt_bias,
          float* ab, int nk);

// C-GDN-5: one causal conv step. proj = fused qkvz projection output
// [proj_rows]; the FIRST conv_dim rows are convolved; z bypasses. Updates
// conv_state in place (shift + append). Returns nothing; conv_out[conv_dim].
void conv_step(uint16_t* conv_out, uint16_t* conv_state, const uint16_t* proj,
               const uint16_t* w, const uint16_t* b, int conv_dim, int ck);

// C-GDN-1: per-head RMSNorm(w) then L2-normalize for q and k parts of the
// conv output. qn/kn get [nk*K] each; conv_out stays untouched.
void qk_norm(uint16_t* qn, uint16_t* kn, const uint16_t* conv_out,
             const uint16_t* wq, const uint16_t* wk, int nk, int K,
             float eps);

// C-GDN-4: one recurrent delta step for ALL k-heads.
//   state  [nk * (ratio*V) * K] fp32, updated in place
//   v      [nv * V] fp16 (the conv output v slice)
//   ab     [nk * 2] fp32 from gate()
//   y      [nv * V] fp16 out
void delta_step(float* state, uint16_t* y, const uint16_t* qn,
                const uint16_t* kn, const uint16_t* v, const float* ab,
                int nk, int K, int ratio, int V);

// C-GDN-6: o[nv*V] = rmsnorm(y_h, w_on) * silu(z_h), per v-head.
// y is the [nk * (ratio*V)] delta output (== [nv*V] rows grouped by k-head);
// z is the [nv*V] gate slice of the fused projection.
void out_norm(uint16_t* out, const uint16_t* y, const uint16_t* z,
              const uint16_t* w_on, int nv, int V, float eps);

}  // namespace gdn_cpu

// ---- CUDA launchers (decode step, one token; prefill loops per token) ------
#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
namespace cuda {

void gdn_gate_f32(const uint16_t* ba, const float* A_log, const float* dt_bias,
                  float* ab, int nk, cudaStream_t stream);

void gdn_conv_step_f16(uint16_t* conv_out, uint16_t* conv_state,
                       const uint16_t* proj, const uint16_t* w,
                       const uint16_t* b, int conv_dim, int ck,
                       cudaStream_t stream);

void gdn_qk_norm_f16(uint16_t* qn, uint16_t* kn, const uint16_t* conv_out,
                     const uint16_t* wq, const uint16_t* wk, int nk, int K,
                     float eps, cudaStream_t stream);

void gdn_delta_step_f16(float* state, uint16_t* y, const uint16_t* qn,
                        const uint16_t* kn, const uint16_t* v,
                        const float* ab, int nk, int K, int ratio, int V,
                        cudaStream_t stream);

void gdn_out_norm_f16(uint16_t* out, const uint16_t* y, const uint16_t* z,
                      const uint16_t* w_on, int nv, int V, float eps,
                      cudaStream_t stream);

}  // namespace cuda
#endif  // SAGER_HAVE_CUDA

}  // namespace sager
