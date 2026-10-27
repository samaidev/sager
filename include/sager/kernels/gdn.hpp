// gdn.hpp - Gated Delta Net (linear attention) kernels for Sage-A1.5's
// hybrid blocks (30 of 40 layers are gated-delta layers; see config.json
// linear_* fields and full_attention_interval=4).
//
// This is THE piece v0.1 was missing: without it the engine cannot run the
// model at all (v0.1 assumed every layer is GQA and the converter hit
// KeyError on linear_attn.* tensors).
//
// Spec conventions (docs/gdn-kernel.md is the normative text):
//   C-GDN-1  q,k: per-head L2-normalize (y = x / sqrt(sum(x^2) + eps));
//            NO learnable weight (real Sage-A1.5 GDN layers have none -
//            the q/k RMSNorm weights belong to the full-attention layers)
//   C-GDN-2  beta = sigmoid(b)                  per v-head (nv)
//   C-GDN-3  g = A[h] * softplus(a + dt_bias[h]) per v-head (nv), where the
//            checkpoint stores A = -exp(A_log) pre-negated (A_NOSCAN);
//            alpha = exp(g)
//   C-GDN-4  per k-head state S [ratio*V x K] (row-major, rows = value slots;
//            value row j of k-head h belongs to v-head h*ratio + j/V):
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
//   ba              ba[2*nv] = [b | a]      (per V-head; the checkpoint's
//                     ssm_beta/ssm_alpha projections concatenated row-wise)
//   conv weight     w[conv_dim * ck] row-major (row d: ck taps)
//   conv state      st[conv_dim * (ck-1)]  (previous raw conv inputs)
//   delta state     S[nk * (ratio*V) * K] fp32 (per k-head [rows x K])
//   gate out        ab[2*nv] = {alpha[nv] | beta[nv]} fp32
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
  int ba_rows()      const { return 2 * n_v_heads; }   // [b(nv) | a(nv)]
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

// C-GDN-2/3: ba[2nv] (fp16: [b | a]) + A (fp32, pre-negated -exp) /
// dt_bias (fp32) -> ab[2nv] = {alpha[nv] | beta[nv]}.
void gate(const uint16_t* ba, const float* A, const float* dt_bias,
          float* ab, int nv);

// C-GDN-5: one causal conv step. proj = fused qkvz projection output
// [proj_rows]; the FIRST conv_dim rows are convolved; z bypasses. Updates
// conv_state in place (shift + append). Returns nothing; conv_out[conv_dim].
void conv_step(uint16_t* conv_out, uint16_t* conv_state, const uint16_t* proj,
               const uint16_t* w, const uint16_t* b, int conv_dim, int ck);

// C-GDN-1: per-head L2 normalization of the q and k parts of the conv
// output (y = x / sqrt(sum(x^2) + eps), no learnable weight). qn/kn get
// [nk*K] each; conv_out stays untouched.
void l2_norm(uint16_t* qn, uint16_t* kn, const uint16_t* conv_out,
             int nk, int K, float eps);

// C-GDN-4: one recurrent delta step for ALL k-heads.
//   state  [nk * (ratio*V) * K] fp32, updated in place
//   v      [nv * V] fp16 (the conv output v slice)
//   ab     [2*nv] fp32 from gate() ({alpha | beta}, per v-head)
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

}  // namespace sager

// ---- CUDA launchers (decode step, one token; prefill loops per token) ------
// NOTE: this include MUST stay at global scope. Putting <cuda_runtime.h>
// inside `namespace sager` makes CUDA 13's cuda_runtime.h (line 148) pull
// libstdc++'s <utility> INSIDE the namespace - std::pair becomes
// sager::std::pair and GCC 13 emits ~79 "is_nothrow_move_constructible was
// not declared in this scope" errors. First caught by the T4 build.
#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
namespace sager {
namespace cuda {

void gdn_gate_f32(const uint16_t* ba, const float* A, const float* dt_bias,
                  float* ab, int nv, cudaStream_t stream);

void gdn_conv_step_f16(uint16_t* conv_out, uint16_t* conv_state,
                       const uint16_t* proj, const uint16_t* w,
                       const uint16_t* b, int conv_dim, int ck,
                       cudaStream_t stream);

void gdn_l2_norm_f16(uint16_t* qn, uint16_t* kn, const uint16_t* conv_out,
                     int nk, int K, float eps, cudaStream_t stream);

void gdn_delta_step_f16(float* state, uint16_t* y, const uint16_t* qn,
                        const uint16_t* kn, const uint16_t* v,
                        const float* ab, int nk, int K, int ratio, int V,
                        cudaStream_t stream);

void gdn_out_norm_f16(uint16_t* out, const uint16_t* y, const uint16_t* z,
                      const uint16_t* w_on, int nv, int V, float eps,
                      cudaStream_t stream);

// ---- v0.3 chunk kernels (prefill / verify windows) -------------------------
// Batched over n tokens; math is bit-identical to the per-token walk above
// (same serial fp32 order per element / per state row).

// Causal depthwise conv over the chunk: conv_out [n x conv_dim], raw conv
// inputs are the first conv_dim rows of proj [n x proj_rows]; state_old is
// read-only here (commit below).
void gdn_conv_chunk_f16(uint16_t* conv_out, const uint16_t* proj,
                        const uint16_t* state_old, const uint16_t* w,
                        const uint16_t* b, int conv_dim, int ck, int n,
                        cudaStream_t stream);

// conv_state <- the last (ck-1) raw conv inputs of the chunk.
void gdn_conv_state_commit_f16(uint16_t* conv_state, const uint16_t* proj,
                               int conv_dim, int ck, int n,
                               cudaStream_t stream);

// ab [n x 2nv] = {alpha | beta} per token from ba [n x 2nv].
void gdn_gate_chunk_f32(const uint16_t* ba, const float* A,
                        const float* dt_bias, float* ab, int nv, int n,
                        cudaStream_t stream);

// Per-head L2 norm of q/k for every token: qn/kn [n x nk*K].
void gdn_l2_norm_chunk_f16(uint16_t* qn, uint16_t* kn,
                           const uint16_t* conv_out, int conv_dim, int nk,
                           int K, float eps, int n, cudaStream_t stream);

// Fused chunk recurrence: one kernel walks all n tokens; each block keeps
// its (k-head, row-chunk) state slice in shared memory. y [n x nv*V].
void gdn_scan_chunk_f16(float* state, uint16_t* y, const uint16_t* qn,
                        const uint16_t* kn, const uint16_t* conv_out,
                        int conv_dim, const float* ab, int nk, int K,
                        int ratio, int V, int n, cudaStream_t stream);

// v0.6 warp-per-row variant: one warp per (k-head, row) pair, state in
// registers, k/q streamed through L2, no shared memory or block barriers.
// Same recurrence, fp32 reduction order differs (lane partials + shuffle
// tree) -> outputs differ at the rounding level, like Strata's chunked GDN
// scan. Falls back to gdn_scan_chunk_f16 for shapes it cannot register-map.
void gdn_scan_chunk_f16_warp(float* state, uint16_t* y, const uint16_t* qn,
                             const uint16_t* kn, const uint16_t* conv_out,
                             int conv_dim, const float* ab, int nk, int K,
                             int ratio, int V, int n, cudaStream_t stream);

// v0.12 4-rows-per-warp variant (SAGER_GDN_R4): single-wave grid + k/q
// register prefetch. Final state bit-identical to gdn_scan_chunk_f16_warp;
// y differs by <= 1 fp16 ulp on ~0.017% of values (FMA contraction).
// Falls back to gdn_scan_chunk_f16_warp for shapes it cannot register-map.
void gdn_scan_chunk_f16_warp4(float* state, uint16_t* y, const uint16_t* qn,
                              const uint16_t* kn, const uint16_t* conv_out,
                              int conv_dim, const float* ab, int nk, int K,
                              int ratio, int V, int n, cudaStream_t stream);

// Output gating for every token: out [n x nv*V] = rmsnorm(y) * silu(z),
// where z is the [nv*V] tail of each proj row.
void gdn_out_norm_chunk_f16(uint16_t* out, const uint16_t* y,
                            const uint16_t* proj, int proj_rows, int conv_dim,
                            const uint16_t* w_on, int nv, int V, float eps,
                            int n, cudaStream_t stream);

}  // namespace cuda

}  // namespace sager
#endif  // SAGER_HAVE_CUDA
