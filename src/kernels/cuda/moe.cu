// moe.cu - router top-k + MoE combine for the Sage-A1.5 MoE block.
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>

namespace sager::cuda {
namespace {

__device__ __forceinline__ float ldh(const uint16_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ void sth(uint16_t* p, float v) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}

// Router top-k. n_expert <= 1024: single block, probabilities in smem.
// Greedy selection (argmax + mask) k times - n_expert_used is tiny (8).
__global__ void router_topk_kernel(const float* logits, int* idx,
                                   float* weight, int n_expert, int k) {
  extern __shared__ float smem[];   // n_expert probs + scratch
  float* prob = smem;

  const int tid = threadIdx.x;
  // softmax over n_expert (small; single thread computes max/z, threads fill)
  __shared__ float m_sh, z_sh;
  if (tid == 0) {
    float mx = -1e30f;
    for (int i = 0; i < n_expert; ++i) mx = fmaxf(mx, logits[i]);
    m_sh = mx;
    float z = 0.f;
    for (int i = 0; i < n_expert; ++i) z += expf(logits[i] - mx);
    z_sh = z;
  }
  __syncthreads();
  for (int i = tid; i < n_expert; i += blockDim.x) {
    prob[i] = expf(logits[i] - m_sh) / z_sh;
  }
  __syncthreads();

  if (tid == 0) {
    float wsum = 0.f;
    for (int kk = 0; kk < k; ++kk) {
      int best = -1;
      float bp = -1.f;
      for (int i = 0; i < n_expert; ++i) {
        const float p = prob[i];
        // NaN-safe compare: a NaN prob (corrupt upstream logits) must never
        // leave best = -1, which would become a host-side OOB in the cache.
        if (p == p && p > bp) { bp = p; best = i; }
      }
      if (best < 0) best = kk % n_expert;   // degenerate logits: deterministic
      prob[best] = -1.f;
      idx[kk] = best;
      weight[kk] = bp;
      wsum += bp;
    }
    if (wsum > 0.f)
      for (int kk = 0; kk < k; ++kk) weight[kk] /= wsum;  // renormalized top-k
  }
}

// out = residual + sum_k weight[k] * expert_out[idx[k], :]
__global__ void moe_combine_kernel(uint16_t* out, const uint16_t* expert_out,
                                   const int* idx, const float* weight,
                                   int k, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  float acc = ldh(out + i);  // residual already in out
  for (int kk = 0; kk < k; ++kk) {
    const int row = idx[kk];
    if (row < 0) continue;                    // miss not (yet) delivered
    acc += weight[kk] * ldh(expert_out + static_cast<size_t>(kk) * hidden + i);
  }
  sth(out + i, acc);
}

}  // namespace

void router_topk(const float* logits, int* idx, float* weight, int n_expert,
                 int n_expert_used, cudaStream_t stream) {
  router_topk_kernel<<<1, 128, n_expert * sizeof(float), stream>>>(
      logits, idx, weight, n_expert, n_expert_used);
}

void moe_combine_f16(uint16_t* out, const uint16_t* expert_out, const int* idx,
                     const float* weight, int n_expert_used, int hidden,
                     cudaStream_t stream) {
  moe_combine_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
      out, expert_out, idx, weight, n_expert_used, hidden);
}

__global__ void scatter_miss_kernel(uint16_t* expert_out,
                                    const int* slot_of_k, const float* miss_y,
                                    int hidden) {
  const int k = blockIdx.y;
  const int slot = slot_of_k[k];
  if (slot < 0) return;                       // hit row: already filled
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden) {
    const float v = miss_y[static_cast<size_t>(slot) * hidden + i];
    *reinterpret_cast<__half*>(&expert_out[static_cast<size_t>(k) * hidden + i]) =
        __float2half(v);
  }
}

void scatter_miss_f32(uint16_t* expert_out, const int* d_slot_of_k,
                      const float* miss_y, int k_rows, int hidden,
                      cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, k_rows);
  scatter_miss_kernel<<<grid, block, 0, stream>>>(expert_out, d_slot_of_k,
                                                  miss_y, hidden);
}

// Scatter compact hit rows into their routed slots: dst[dst_rows[i]] = src[i].
// Used after the fused pointer-table GEMV produced the n_hit expert outputs
// back-to-back; combine expects one row per routed slot k.
__global__ void scatter_rows_kernel(uint16_t* dst, const uint16_t* src,
                                    const int* dst_rows, int hidden) {
  const int i = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < hidden) {
    dst[static_cast<size_t>(dst_rows[i]) * hidden + j] =
        src[static_cast<size_t>(i) * hidden + j];
  }
}

void scatter_rows_f16(uint16_t* dst, const uint16_t* src, const int* dst_rows,
                      int n_rows, int hidden, cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, n_rows);
  scatter_rows_kernel<<<grid, block, 0, stream>>>(dst, src, dst_rows, hidden);
}

}  // namespace sager::cuda
