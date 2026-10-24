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

// ---- launchers -------------------------------------------------------------
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
  const size_t smem = (static_cast<size_t>(K) + 1) * sizeof(float);
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
  const size_t smem = (static_cast<size_t>(V) + 1) * sizeof(float);
  gdn_out_norm_kernel<<<nv, block, smem, stream>>>(out, y, z, w_on, V, eps);
}

}  // namespace sager::cuda
