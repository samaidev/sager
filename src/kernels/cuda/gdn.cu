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
// gate: ab[2h] = {alpha, beta} from ba (fp16), A_log / dt_bias (fp32).
// ---------------------------------------------------------------------------
__global__ void gdn_gate_kernel(const uint16_t* ba, const float* A_log,
                                const float* dt_bias, float* ab, int nk) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nk) return;
  const float b = ldh(ba + h);
  const float a = ldh(ba + nk + h);
  const float beta = 1.f / (1.f + expf(-b));
  const float dt = softplusf(a + dt_bias[h]);
  const float g = -expf(A_log[h]) * dt;
  ab[2 * h + 0] = expf(g);
  ab[2 * h + 1] = beta;
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
// qk norm: per k-head RMSNorm(w) then L2-normalize, for both q and k parts.
// One block per head; two sequential in-block passes (q, then k).
// ---------------------------------------------------------------------------
__global__ void gdn_qk_norm_kernel(uint16_t* qn, uint16_t* kn,
                                   const uint16_t* conv_out,
                                   const uint16_t* wq, const uint16_t* wk,
                                   int K, float eps) {
  extern __shared__ float sm[];            // K (values) + 1 (reduction)
  float* vals = sm;
  float* red = sm + K;

  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int off_k = gridDim.x * K;         // k part starts after all q heads

  // head-normalize one part: src -> rmsnorm(w) -> L2 normalize -> dst
  // (lambda kept out; two explicit passes keep the smem reuse obvious)
  const uint16_t* src_q = conv_out + static_cast<size_t>(h) * K;
  float ss = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float v = ldh(src_q + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv_rms =
      rsqrtf(block_sum_reduce(red, ss) / static_cast<float>(K) + eps);
  float l2 = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float t = vals[i] * inv_rms * ldh(wq + i);
    vals[i] = t;
    l2 += t * t;
  }
  const float inv_l2 = rsqrtf(block_sum_reduce(red, l2) + 1e-12f);
  for (int i = tid; i < K; i += blockDim.x) {
    sth(qn + static_cast<size_t>(h) * K + i, vals[i] * inv_l2);
  }
  __syncthreads();

  const uint16_t* src_k = conv_out + off_k + static_cast<size_t>(h) * K;
  ss = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float v = ldh(src_k + i);
    vals[i] = v;
    ss += v * v;
  }
  const float inv_rms_k =
      rsqrtf(block_sum_reduce(red, ss) / static_cast<float>(K) + eps);
  l2 = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float t = vals[i] * inv_rms_k * ldh(wk + i);
    vals[i] = t;
    l2 += t * t;
  }
  const float inv_l2_k = rsqrtf(block_sum_reduce(red, l2) + 1e-12f);
  for (int i = tid; i < K; i += blockDim.x) {
    sth(kn + static_cast<size_t>(h) * K + i, vals[i] * inv_l2_k);
  }
}

// ---------------------------------------------------------------------------
// delta step: one block per k-head, one thread per state row j.
//   u   = S[j,:] . k
//   S[j,:] <- alpha*S[j,:] - alpha*beta*u*k + beta*v_j*k
//   y_j = S_new[j,:] . q
// ---------------------------------------------------------------------------
__global__ void gdn_delta_step_kernel(float* state, uint16_t* y,
                                      const uint16_t* qn, const uint16_t* kn,
                                      const uint16_t* v, const float* ab,
                                      int K, int rows) {
  extern __shared__ float sm[];            // k[K] + q[K] (+1 pad)
  float* ks = sm;
  float* qs = sm + K;

  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const float alpha = ab[2 * h + 0];
  const float beta = ab[2 * h + 1];

  for (int i = tid; i < K; i += blockDim.x) {
    ks[i] = ldh(kn + static_cast<size_t>(h) * K + i);
    qs[i] = ldh(qn + static_cast<size_t>(h) * K + i);
  }
  __syncthreads();

  float* S = state + static_cast<size_t>(h) * rows * K;
  const uint16_t* v_h = v + static_cast<size_t>(h) * rows;
  for (int j = tid; j < rows; j += blockDim.x) {
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

// ---- launchers -------------------------------------------------------------
void gdn_gate_f32(const uint16_t* ba, const float* A_log, const float* dt_bias,
                  float* ab, int nk, cudaStream_t stream) {
  const int block = 128;
  gdn_gate_kernel<<<(nk + block - 1) / block, block, 0, stream>>>(
      ba, A_log, dt_bias, ab, nk);
}

void gdn_conv_step_f16(uint16_t* conv_out, uint16_t* conv_state,
                       const uint16_t* proj, const uint16_t* w,
                       const uint16_t* b, int conv_dim, int ck,
                       cudaStream_t stream) {
  const int block = 256;
  gdn_conv_step_kernel<<<(conv_dim + block - 1) / block, block, 0, stream>>>(
      conv_out, conv_state, proj, w, b, conv_dim, ck);
}

void gdn_qk_norm_f16(uint16_t* qn, uint16_t* kn, const uint16_t* conv_out,
                     const uint16_t* wq, const uint16_t* wk, int nk, int K,
                     float eps, cudaStream_t stream) {
  const int block = 128;
  const size_t smem = (static_cast<size_t>(K) + 1) * sizeof(float);
  gdn_qk_norm_kernel<<<nk, block, smem, stream>>>(qn, kn, conv_out, wq, wk,
                                                  K, eps);
}

void gdn_delta_step_f16(float* state, uint16_t* y, const uint16_t* qn,
                        const uint16_t* kn, const uint16_t* v,
                        const float* ab, int nk, int K, int ratio, int V,
                        cudaStream_t stream) {
  const int rows = ratio * V;
  const int block = pow2_block(rows);
  const size_t smem = (2 * static_cast<size_t>(K) + 1) * sizeof(float);
  gdn_delta_step_kernel<<<nk, block, smem, stream>>>(state, y, qn, kn, v, ab,
                                                     K, rows);
}

void gdn_out_norm_f16(uint16_t* out, const uint16_t* y, const uint16_t* z,
                      const uint16_t* w_on, int nv, int V, float eps,
                      cudaStream_t stream) {
  const int block = pow2_block(V);
  const size_t smem = (static_cast<size_t>(V) + 1) * sizeof(float);
  gdn_out_norm_kernel<<<nv, block, smem, stream>>>(out, y, z, w_on, V, eps);
}

}  // namespace sager::cuda
