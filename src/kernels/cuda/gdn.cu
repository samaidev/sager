// gdn.cu - Gated Delta Net decode-step kernels for sm_75 (T4).
//
// One token walks a linear-attn layer as:
//   rmsnorm -> [Q4 GEMV: fused qkvz + ba projections (existing kernels)]
//   -> gdn_conv_step  (causal depthwise conv + SiLU, state shift)
//   -> gdn_gate       (alpha/beta from ba, A_log, dt_bias)
//   -> gdn_qk_norm    (per-head RMSNorm + L2 normalize)
//   -> gdn_delta_step (per k-head rank-1 delta update of the fp32 state)
//   -> gdn_out_norm   (per v-head RMSNorm * silu(z))
//   -> [Q4 GEMV: out_proj -> residual (existing kernels)]
//
// Everything here is fixed-shape, fixed-address -> safe to capture into the
// per-layer decode CUDA graphs (same contract as the dense half in
// session.cpp). State buffers are allocated once by the session and zeroed
// at prepare()/reset(); the conv state shift and the delta rank-1 update are
// the only mutations.
//
// Numerics: state in fp32 (a half-precision recurrent state measurably
// drifts over 256K-token contexts); activations fp16; accumulators fp32.
#include "sager/kernels/gdn.hpp"

#include <cuda_fp16.h>

namespace sager::cuda {
namespace {

__device__ __forceinline__ float ldh(const uint16_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ void sth(uint16_t* p, float v) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}
__device__ __forceinline__ float siluf(float x) {
  return x / (1.f + expf(-x));
}
__device__ __forceinline__ float softplusf(float x) {
  if (x > 20.f) return x;
  if (x < -20.f) return __expf(x);
  return log1pf(expf(x));
}

// Block-wide sum reduction over shared memory. blockDim.x MUST be a power of
// two (the launchers snap it); runs a few times per block at most.
__device__ __forceinline__ float block_sum_reduce(float* red, float v) {
  const int tid = threadIdx.x;
  const int n = blockDim.x;
  red[tid] = v;
  __syncthreads();
  for (int off = n / 2; off > 0; off >>= 1) {
    if (tid < off) red[tid] += red[tid + off];
    __syncthreads();
  }
  const float out = red[0];
  __syncthreads();
  return out;
}

// smallest power-of-two block covering n (capped at 1024)
inline int pow2_block(int n) {
  int b = 32;
  while (b < n && b < 1024) b <<= 1;
  return b;
}

// ---------------------------------------------------------------------------
// gate: ab = {alpha[nv] | beta[nv]} from ba (fp16, [b|a] per v-head),
// A (fp32, pre-negated -exp) and dt_bias (fp32), all per v-head.
// ---------------------------------------------------------------------------
__global__ void gdn_gate_kernel(const uint16_t* ba, const float* A,
                                const float* dt_bias, float* ab, int nv) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nv) return;
  const float b = ldh(ba + h);
  const float a = ldh(ba + nv + h);
  const float beta = 1.f / (1.f + expf(-b));
  const float dt = softplusf(a + dt_bias[h]);
  const float g = A[h] * dt;   // A = -exp(A_log) pre-negated
  ab[h] = expf(g);             // alpha
  ab[nv + h] = beta;
}

// ---------------------------------------------------------------------------
// conv step: out[d] = silu(sum_j w[d,j]*win[j] + b[d]); state shifts by one.
// win = [state(d,0..ck-2), proj_d]. One thread per channel.
// ---------------------------------------------------------------------------
__global__ void gdn_conv_step_kernel(uint16_t* conv_out, uint16_t* conv_state,
                                     const uint16_t* proj, const uint16_t* w,
                                     const uint16_t* b, int conv_dim, int ck) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= conv_dim) return;
  const int nm1 = ck - 1;
  uint16_t* st = conv_state + static_cast<size_t>(d) * nm1;
  float acc = ldh(b + d);
  for (int j = 0; j < nm1; ++j) {
    acc += ldh(w + static_cast<size_t>(d) * ck + j) * ldh(st + j);
  }
  const float x = ldh(proj + d);
  acc += ldh(w + static_cast<size_t>(d) * ck + nm1) * x;
  sth(conv_out + d, siluf(acc));
  // shift window left, append the raw current input (verbatim fp16 bits)
  for (int j = 0; j < nm1 - 1; ++j) st[j] = st[j + 1];
  st[nm1 - 1] = proj[d];
}

// ---------------------------------------------------------------------------
// l2 norm: per-head L2 normalization of the q and k parts (y = x /
// sqrt(sum(x^2) + eps), no learnable weight). One block per head; two
// sequential in-block passes (q, then k).
// ---------------------------------------------------------------------------
__global__ void gdn_l2_norm_kernel(uint16_t* qn, uint16_t* kn,
                                   const uint16_t* conv_out,
                                   int K, float eps) {
  extern __shared__ float sm[];            // K (values) + 1 (reduction)
  float* vals = sm;
  float* red = sm + K;

  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int off_k = gridDim.x * K;         // k part starts after all q heads

  // q part
  const uint16_t* src_q = conv_out + static_cast<size_t>(h) * K;
  float ss = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float v = ldh(src_q + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv_q = rsqrtf(block_sum_reduce(red, ss) + eps);
  for (int i = tid; i < K; i += blockDim.x) {
    sth(qn + static_cast<size_t>(h) * K + i, vals[i] * inv_q);
  }
  __syncthreads();

  // k part
  const uint16_t* src_k = conv_out + off_k + static_cast<size_t>(h) * K;
  ss = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float v = ldh(src_k + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv_k = rsqrtf(block_sum_reduce(red, ss) + eps);
  for (int i = tid; i < K; i += blockDim.x) {
    sth(kn + static_cast<size_t>(h) * K + i, vals[i] * inv_k);
  }
}

// ---------------------------------------------------------------------------
// delta step: one block per k-head, one thread per state row j.
//   vh  = h*ratio + j/V   (the v-head owning state row j)
//   u   = S[j,:] . k
//   S[j,:] <- alpha*S[j,:] - alpha*beta*u*k + beta*v_j*k
//   y_j = S_new[j,:] . q
// ab layout: {alpha[nv] | beta[nv]}
// ---------------------------------------------------------------------------
__global__ void gdn_delta_step_kernel(float* state, uint16_t* y,
                                      const uint16_t* qn, const uint16_t* kn,
                                      const uint16_t* v, const float* ab,
                                      int K, int rows, int ratio, int V) {
  extern __shared__ float sm[];            // k[K] + q[K] (+1 pad)
  float* ks = sm;
  float* qs = sm + K;

  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int nv = gridDim.x * ratio;

  for (int i = tid; i < K; i += blockDim.x) {
    ks[i] = ldh(kn + static_cast<size_t>(h) * K + i);
    qs[i] = ldh(qn + static_cast<size_t>(h) * K + i);
  }
  __syncthreads();

  float* S = state + static_cast<size_t>(h) * rows * K;
  const uint16_t* v_h = v + static_cast<size_t>(h) * rows;
  for (int j = tid; j < rows; j += blockDim.x) {
    const int vh = h * ratio + j / V;
    const float alpha = ab[vh];
    const float beta = ab[nv + vh];
    float u = 0.f;
    for (int i = 0; i < K; ++i) u += S[static_cast<size_t>(j) * K + i] * ks[i];
    const float vj = ldh(v_h + j);
    float yj = 0.f;
    for (int i = 0; i < K; ++i) {
      float& s = S[static_cast<size_t>(j) * K + i];
      s = alpha * s - alpha * beta * u * ks[i] + beta * vj * ks[i];
      yj += s * qs[i];
    }
    sth(y + static_cast<size_t>(h) * rows + j, yj);
  }
}

// ---------------------------------------------------------------------------
// out norm: o_w = rmsnorm(y_w, w_on) * silu(z_w), one block per v-head.
// ---------------------------------------------------------------------------
__global__ void gdn_out_norm_kernel(uint16_t* out, const uint16_t* y,
                                    const uint16_t* z, const uint16_t* w_on,
                                    int V, float eps) {
  extern __shared__ float sm[];            // V values + 1 reduction
  float* vals = sm;
  float* red = sm + V;

  const int w = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t base = static_cast<size_t>(w) * V;

  float ss = 0.f;
  for (int i = tid; i < V; i += blockDim.x) {
    const float v = ldh(y + base + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum_reduce(red, ss) /
                                static_cast<float>(V) + eps);
  for (int i = tid; i < V; i += blockDim.x) {
    const float n = vals[i] * inv * ldh(w_on + i);
    const float zg = ldh(z + base + i);
    sth(out + base + i, n * siluf(zg));
  }
}

}  // namespace

// ============================================================================
// v0.3 chunk kernels (prefill / verify windows, n tokens at once).
//
// The decode path walks a GDN layer token-by-token with ~12 tiny kernel
// launches per token - on a 2-vCPU host a 512-token chunk pays 30 layers x
// 512 x 12 ~ 184k launches (launch-bound, GPU mostly idle). The chunk
// kernels batch every per-token-op over the whole window and fuse the
// SEQUENTIAL rank-1 recurrence into ONE kernel whose blocks keep their
// state slice in shared memory for the whole chunk.
//
// Bit-compatibility: every per-element op keeps the original serial
// fp32/fp16 order (thread-per-state-row, i ascending; block_sum_reduce with
// the same block sizes), so chunked output is bit-identical to the per-token
// walk - the A/B token-stream gate stays valid.
// ============================================================================

namespace {

// Causal depthwise conv over the whole chunk. conv_out[t*conv_dim + d] =
// silu(sum_j w[d,j] * raw(t - (ck-1) + j, d) + b[d]); raw(x, d) is
// state_old[d, (ck-1)+x] for x < 0 else proj[x*conv_dim + d].
__global__ void gdn_conv_chunk_kernel(uint16_t* conv_out,
                                      const uint16_t* proj,
                                      const uint16_t* state_old,
                                      const uint16_t* w, const uint16_t* b,
                                      int conv_dim, int ck, int n) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (d >= conv_dim || t >= n) return;
  const int nm1 = ck - 1;
  float acc = ldh(b + d);
  for (int j = 0; j < ck; ++j) {
    const int src_t = t + j - nm1;
    const uint16_t v =
        (src_t < 0)
            ? state_old[static_cast<size_t>(d) * nm1 + (src_t + nm1)]
            : proj[static_cast<size_t>(src_t) * conv_dim + d];
    acc += ldh(w + static_cast<size_t>(d) * ck + j) * v;
  }
  sth(conv_out + static_cast<size_t>(t) * conv_dim + d, siluf(acc));
}

// Commit the conv state at chunk end: state[d, j] = raw(n - (ck-1) + j, d).
__global__ void gdn_conv_state_commit_kernel(uint16_t* conv_state,
                                             const uint16_t* proj,
                                             int conv_dim, int ck, int n) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= conv_dim) return;
  const int nm1 = ck - 1;
  uint16_t* st = conv_state + static_cast<size_t>(d) * nm1;
  uint16_t tmp[16];
  for (int j = 0; j < nm1 && j < 16; ++j) {
    const int src_t = n - nm1 + j;
    tmp[j] = (src_t >= 0)
                 ? proj[static_cast<size_t>(src_t) * conv_dim + d]
                 : st[n + j];   // still inside the OLD state window
  }
  for (int j = 0; j < nm1 && j < 16; ++j) st[j] = tmp[j];
}

// alpha/beta gates for every token of the chunk (elementwise, batched).
__global__ void gdn_gate_chunk_kernel(const uint16_t* ba, const float* A,
                                      const float* dt_bias, float* ab,
                                      int nv, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nv) return;
  const int t = i / nv;
  const int h = i % nv;
  const uint16_t* bar = ba + static_cast<size_t>(t) * 2 * nv;
  const float bb = ldh(bar + h);
  const float aa = ldh(bar + nv + h);
  const float beta = 1.f / (1.f + expf(-bb));
  const float dt = softplusf(aa + dt_bias[h]);
  const float g = A[h] * dt;
  float* abr = ab + static_cast<size_t>(t) * 2 * nv;
  abr[h] = expf(g);
  abr[nv + h] = beta;
}

// Per-head L2 norm of q and k for every token (block = (head), grid.y = t).
__global__ void gdn_l2_norm_chunk_kernel(uint16_t* qn, uint16_t* kn,
                                         const uint16_t* conv_out,
                                         int conv_dim, int nk, int K,
                                         float eps) {
  extern __shared__ float sm[];            // K (values) + reduction scratch
  float* vals = sm;
  float* red = sm + K;

  const int t = blockIdx.y;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int off_k = nk * K;
  const uint16_t* row = conv_out + static_cast<size_t>(t) * conv_dim;

  // q part
  const uint16_t* src_q = row + static_cast<size_t>(h) * K;
  float ss = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float v = ldh(src_q + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv_q = rsqrtf(block_sum_reduce(red, ss) + eps);
  for (int i = tid; i < K; i += blockDim.x) {
    sth(qn + (static_cast<size_t>(t) * nk + h) * K + i, vals[i] * inv_q);
  }
  __syncthreads();

  // k part
  const uint16_t* src_k = row + off_k + static_cast<size_t>(h) * K;
  ss = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float v = ldh(src_k + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv_k = rsqrtf(block_sum_reduce(red, ss) + eps);
  for (int i = tid; i < K; i += blockDim.x) {
    sth(kn + (static_cast<size_t>(t) * nk + h) * K + i, vals[i] * inv_k);
  }
}

// THE chunk recurrence: rank-1 gated delta updates walked token by token
// INSIDE one kernel. Block = (k-head, row-chunk of blockDim.x rows); the
// block's state slice lives in shared memory for the whole chunk (loaded
// once, stored once). Per row the math is IDENTICAL to the per-token
// kernel: u = S[j,.]@k (serial i), S[j,:] <- a*S - a*b*u*k + b*v_j*k,
// y_j = S_new[j,:]@q (serial i).
__global__ void gdn_scan_chunk_kernel(float* state, uint16_t* y,
                                      const uint16_t* qn, const uint16_t* kn,
                                      const uint16_t* conv_out, int conv_dim,
                                      const float* ab, int nk, int K,
                                      int ratio, int V, int n) {
  extern __shared__ float sm[];            // k[K] + q[K] + rows*(K+1)
  float* k_sh = sm;
  float* q_sh = sm + K;
  float* s_sh = sm + 2 * K;                // blockDim.x rows, padded +1

  const int h = blockIdx.x;
  const int rows = ratio * V;
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  const int nv = nk * ratio;
  const bool active = (j < rows);
  const int tid = threadIdx.x;

  float* srow = s_sh + static_cast<size_t>(tid) * (K + 1);
  const size_t state_base = static_cast<size_t>(h) * rows * K;
  if (active) {
    for (int i = 0; i < K; ++i) {
      srow[i] = state[state_base + static_cast<size_t>(j) * K + i];
    }
  }

  for (int t = 0; t < n; ++t) {
    for (int i = tid; i < K; i += blockDim.x) {
      k_sh[i] = ldh(kn + (static_cast<size_t>(t) * nk + h) * K + i);
      q_sh[i] = ldh(qn + (static_cast<size_t>(t) * nk + h) * K + i);
    }
    __syncthreads();
    if (active) {
      const int vh = h * ratio + j / V;
      const float* abr = ab + static_cast<size_t>(t) * 2 * nv;
      const float alpha = abr[vh];
      const float beta = abr[nv + vh];
      const float vj = ldh(conv_out +
                           static_cast<size_t>(t) * conv_dim +
                           2 * nk * K + static_cast<size_t>(h) * rows + j);
      float u = 0.f;
      for (int i = 0; i < K; ++i) u += srow[i] * k_sh[i];
      float yj = 0.f;
      for (int i = 0; i < K; ++i) {
        float& s = srow[i];
        s = alpha * s - alpha * beta * u * k_sh[i] + beta * vj * k_sh[i];
        yj += s * q_sh[i];
      }
      sth(y + static_cast<size_t>(t) * (nv * V) +
              static_cast<size_t>(h) * rows + j, yj);
    }
    __syncthreads();
  }

  if (active) {
    for (int i = 0; i < K; ++i) {
      state[state_base + static_cast<size_t>(j) * K + i] = srow[i];
    }
  }
}

// Output gating for every token: o = rmsnorm(y, w_on) * silu(z).
__global__ void gdn_out_norm_chunk_kernel(uint16_t* out, const uint16_t* y,
                                          const uint16_t* proj,
                                          int proj_rows, int conv_dim,
                                          const uint16_t* w_on, int V,
                                          float eps, int y_rows) {
  extern __shared__ float sm[];            // V values + reduction scratch
  float* vals = sm;
  float* red = sm + V;

  const int t = blockIdx.y;
  const int w = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t base = static_cast<size_t>(w) * V;

  float ss = 0.f;
  for (int i = tid; i < V; i += blockDim.x) {
    const float v = ldh(y + static_cast<size_t>(t) * y_rows + base + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum_reduce(red, ss) /
                                static_cast<float>(V) + eps);
  for (int i = tid; i < V; i += blockDim.x) {
    const float n = vals[i] * inv * ldh(w_on + i);
    const float zg = ldh(proj + static_cast<size_t>(t) * proj_rows +
                         conv_dim + base + i);
    sth(out + static_cast<size_t>(t) * y_rows + base + i, n * siluf(zg));
  }
}

}  // namespace

// ---- chunk launchers -------------------------------------------------------
void gdn_conv_chunk_f16(uint16_t* conv_out, const uint16_t* proj,
                        const uint16_t* state_old, const uint16_t* w,
                        const uint16_t* b, int conv_dim, int ck, int n,
                        cudaStream_t stream) {
  const int block = 256;
  dim3 grid((conv_dim + block - 1) / block, n);
  gdn_conv_chunk_kernel<<<grid, block, 0, stream>>>(conv_out, proj, state_old,
                                                    w, b, conv_dim, ck, n);
}

void gdn_conv_state_commit_f16(uint16_t* conv_state, const uint16_t* proj,
                               int conv_dim, int ck, int n,
                               cudaStream_t stream) {
  const int block = 256;
  gdn_conv_state_commit_kernel<<<(conv_dim + block - 1) / block, block, 0,
                                 stream>>>(conv_state, proj, conv_dim, ck, n);
}

void gdn_gate_chunk_f32(const uint16_t* ba, const float* A,
                        const float* dt_bias, float* ab, int nv, int n,
                        cudaStream_t stream) {
  const int block = 128;
  gdn_gate_chunk_kernel<<<(n * nv + block - 1) / block, block, 0, stream>>>(
      ba, A, dt_bias, ab, nv, n);
}

void gdn_l2_norm_chunk_f16(uint16_t* qn, uint16_t* kn,
                           const uint16_t* conv_out, int conv_dim, int nk,
                           int K, float eps, int n, cudaStream_t stream) {
  const int block = 128;   // MUST match the per-token kernel: same reduce tree
  const size_t smem = (static_cast<size_t>(K) + block) * sizeof(float);
  dim3 grid(nk, n);
  gdn_l2_norm_chunk_kernel<<<grid, block, smem, stream>>>(
      qn, kn, conv_out, conv_dim, nk, K, eps);
}

void gdn_scan_chunk_f16(float* state, uint16_t* y, const uint16_t* qn,
                        const uint16_t* kn, const uint16_t* conv_out,
                        int conv_dim, const float* ab, int nk, int K,
                        int ratio, int V, int n, cudaStream_t stream) {
  const int rows = ratio * V;
  int rrows = 32;
  while (rrows > rows) rrows >>= 1;        // block = rows chunk (pow2, <= 32)
  const size_t smem = static_cast<size_t>(2 * K + rrows * (K + 1)) *
                      sizeof(float);
  dim3 grid(nk, (rows + rrows - 1) / rrows);
  gdn_scan_chunk_kernel<<<grid, rrows, smem, stream>>>(
      state, y, qn, kn, conv_out, conv_dim, ab, nk, K, ratio, V, n);
}

void gdn_out_norm_chunk_f16(uint16_t* out, const uint16_t* y,
                            const uint16_t* proj, int proj_rows, int conv_dim,
                            const uint16_t* w_on, int nv, int V, float eps,
                            int n, cudaStream_t stream) {
  const int block = pow2_block(V);
  const size_t smem = (static_cast<size_t>(V) + block) * sizeof(float);
  dim3 grid(nv, n);
  gdn_out_norm_chunk_kernel<<<grid, block, smem, stream>>>(
      out, y, proj, proj_rows, conv_dim, w_on, V, eps, nv * V);
}

void gdn_gate_f32(const uint16_t* ba, const float* A, const float* dt_bias,
                  float* ab, int nv, cudaStream_t stream) {
  const int block = 128;
  gdn_gate_kernel<<<(nv + block - 1) / block, block, 0, stream>>>(
      ba, A, dt_bias, ab, nv);
}

void gdn_conv_step_f16(uint16_t* conv_out, uint16_t* conv_state,
                       const uint16_t* proj, const uint16_t* w,
                       const uint16_t* b, int conv_dim, int ck,
                       cudaStream_t stream) {
  const int block = 256;
  gdn_conv_step_kernel<<<(conv_dim + block - 1) / block, block, 0, stream>>>(
      conv_out, conv_state, proj, w, b, conv_dim, ck);
}

void gdn_l2_norm_f16(uint16_t* qn, uint16_t* kn, const uint16_t* conv_out,
                     int nk, int K, float eps, cudaStream_t stream) {
  const int block = 128;
  // vals[K] + red[block]: block_sum_reduce writes red[tid] for every thread.
  const size_t smem = (static_cast<size_t>(K) + block) * sizeof(float);
  gdn_l2_norm_kernel<<<nk, block, smem, stream>>>(qn, kn, conv_out, K, eps);
}

void gdn_delta_step_f16(float* state, uint16_t* y, const uint16_t* qn,
                        const uint16_t* kn, const uint16_t* v,
                        const float* ab, int nk, int K, int ratio, int V,
                        cudaStream_t stream) {
  const int rows = ratio * V;
  const int block = pow2_block(rows);
  const size_t smem = (2 * static_cast<size_t>(K) + 1) * sizeof(float);
  gdn_delta_step_kernel<<<nk, block, smem, stream>>>(state, y, qn, kn, v, ab,
                                                     K, rows, ratio, V);
}

void gdn_out_norm_f16(uint16_t* out, const uint16_t* y, const uint16_t* z,
                      const uint16_t* w_on, int nv, int V, float eps,
                      cudaStream_t stream) {
  const int block = pow2_block(V);
  // vals[V] + red[block]: block_sum_reduce writes red[tid] for every thread.
  const size_t smem = (static_cast<size_t>(V) + block) * sizeof(float);
  gdn_out_norm_kernel<<<nv, block, smem, stream>>>(out, y, z, w_on, V, eps);
}

}  // namespace sager::cuda
