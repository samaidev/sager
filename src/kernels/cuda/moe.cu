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

  // v0.7: warp-parallel selection. Same (value, lower-index) tie rule as the
  // serial scan: a lane's local best keeps the lowest index on ties, and the
  // shuffle reduce prefers the lower index on equal values, which reproduces
  // the first-strict-max order of the serial walk bit-exactly.
  if (tid < 32) {
    const int lane = tid;
    float wsum = 0.f;
    for (int kk = 0; kk < k; ++kk) {
      float bv = -1.f;
      int bi = -1;
      for (int i = lane; i < n_expert; i += 32) {
        const float p = prob[i];
        if (p == p && (p > bv || (p == bv && bi >= 0 && i < bi))) {
          bv = p; bi = i;
        }
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
        if (ov > bv || (ov == bv && oi >= 0 && bi >= 0 && oi < bi)) {
          bv = ov; bi = oi;
        } else if (bi < 0 && oi >= 0) { bv = ov; bi = oi; }
      }
      if (lane == 0) {
        if (bi < 0) bi = kk % n_expert;   // degenerate logits: deterministic
        prob[bi] = -1.f;
        idx[kk] = bi;
        weight[kk] = bv;
        wsum += bv;
      }
      __syncwarp();
    }
    if (lane == 0 && wsum > 0.f)
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

// ---------------------------------------------------------------------------
// v0.2.3 batched prefill (Strata-style streamed walk) kernels
// ---------------------------------------------------------------------------

namespace {

// Router top-k for every row of the chunk; one block per row.
__global__ void router_topk_batched_kernel(const float* logits, int* idx,
                                           float* weight, int n_expert, int k,
                                           int stride, int n_rows) {
  extern __shared__ float smem[];
  float* prob = smem;
  const int row = blockIdx.x;
  if (row >= n_rows) return;
  const float* lg = logits + static_cast<size_t>(row) * n_expert;
  const int tid = threadIdx.x;

  __shared__ float m_sh, z_sh;
  if (tid == 0) {
    float mx = -1e30f;
    for (int i = 0; i < n_expert; ++i) mx = fmaxf(mx, lg[i]);
    m_sh = mx;
    float z = 0.f;
    for (int i = 0; i < n_expert; ++i) z += expf(lg[i] - mx);
    z_sh = z;
  }
  __syncthreads();
  for (int i = tid; i < n_expert; i += blockDim.x) {
    prob[i] = expf(lg[i] - m_sh) / z_sh;
  }
  __syncthreads();

  if (tid == 0) {
    float wsum = 0.f;
    int* out_idx = idx + static_cast<size_t>(row) * stride;
    float* out_w = weight + static_cast<size_t>(row) * stride;
    for (int kk = 0; kk < k; ++kk) {
      int best = -1;
      float bp = -1.f;
      for (int i = 0; i < n_expert; ++i) {
        const float p = prob[i];
        if (p == p && p > bp) { bp = p; best = i; }
      }
      if (best < 0) best = kk % n_expert;
      prob[best] = -1.f;
      out_idx[kk] = best;
      out_w[kk] = bp;
      wsum += bp;
    }
    if (wsum > 0.f) {
      for (int kk = 0; kk < k; ++kk) out_w[kk] /= wsum;
    }
  }
}

// dst[r] = src[src_rows[r]] for whole BlockQ8K rows (routed-row compaction).
__global__ void gather_rows_q8_kernel(q4k::BlockQ8K* dst,
                                      const q4k::BlockQ8K* src,
                                      const int* src_rows, int row_blocks) {
  const int r = blockIdx.y;
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= row_blocks) return;
  const int src_r = src_rows[r];
  dst[static_cast<size_t>(r) * row_blocks + b] =
      src[static_cast<size_t>(src_r) * row_blocks + b];
}

// Compact per-expert outputs into the batched combine slab (see header).
__global__ void scatter_routed_kernel(uint16_t* expert_y,
                                      const uint16_t* grouped_out,
                                      const int* gslot, int n_used,
                                      int hidden) {
  const int row = blockIdx.y;
  const int kk = blockIdx.z;
  const int gs = gslot[static_cast<size_t>(row) * n_used + kk];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden || gs < 0) return;
  uint16_t* dst = expert_y +
      (static_cast<size_t>(row) * (n_used + 1) + kk) * hidden + i;
  *dst = grouped_out[static_cast<size_t>(gs) * hidden + i];
}

// Batched combine: out[row] = residual + sum_kk weight[row,kk] * expert_y
// [row,kk]; slots are positional, n_used includes the shared slot.
__global__ void moe_combine_batched_kernel(uint16_t* out,
                                           const uint16_t* expert_y,
                                           const float* weight, int n_used,
                                           int hidden) {
  const int row = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  float acc = ldh(out + static_cast<size_t>(row) * hidden + i);
  const uint16_t* yrow = expert_y +
      static_cast<size_t>(row) * (n_used + 1) * hidden;
  const float* wrow = weight + static_cast<size_t>(row) * (n_used + 1);
  for (int kk = 0; kk < n_used + 1; ++kk) {
    acc += wrow[kk] * ldh(yrow + static_cast<size_t>(kk) * hidden + i);
  }
  sth(out + static_cast<size_t>(row) * hidden + i, acc);
}

// Batched shared-expert scalar gate.
__global__ void sigmoid_gate_scalar_batched_kernel(uint16_t* out,
                                                   const float* g,
                                                   int hidden) {
  const int row = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  const float gate = 1.f / (1.f + expf(-g[row]));
  uint16_t* p = out + static_cast<size_t>(row) * hidden + i;
  sth(p, __half2float(*reinterpret_cast<const __half*>(p)) * gate);
}

// CPU-pool fallback rows: dst[dst_flat[i]] = fp16(src[i]) rows.
__global__ void scatter_rows_generic_kernel(uint16_t* dst,
                                            const float* src,
                                            const int* dst_flat, int hidden) {
  const int i = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < hidden) {
    dst[static_cast<size_t>(dst_flat[i]) + j] =
        __float2half(src[static_cast<size_t>(i) * hidden + j]);
  }
}

}  // namespace

void router_topk_batched(const float* logits, int* idx, float* weight,
                         int n_expert, int n_expert_used, int idx_stride,
                         int n_rows, cudaStream_t stream) {
  router_topk_batched_kernel<<<n_rows, 128, n_expert * sizeof(float), stream>>>(
      logits, idx, weight, n_expert, n_expert_used, idx_stride, n_rows);
}

void gather_rows_q8(q4k::BlockQ8K* dst, const q4k::BlockQ8K* src,
                    const int* src_rows, int n_rows, int row_blocks,
                    cudaStream_t stream) {
  dim3 block(8), grid((row_blocks + 7) / 8, n_rows);
  gather_rows_q8_kernel<<<grid, block, 0, stream>>>(dst, src, src_rows,
                                                    row_blocks);
}

void scatter_routed_f16(uint16_t* expert_y, const uint16_t* grouped_out,
                        const int* gslot, int n_rows, int n_used, int hidden,
                        cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, n_rows, n_used);
  scatter_routed_kernel<<<grid, block, 0, stream>>>(expert_y, grouped_out,
                                                    gslot, n_used, hidden);
}

void moe_combine_batched_f16(uint16_t* out, const uint16_t* expert_y,
                             const float* weight, int n_used, int hidden,
                             int n_rows, cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, n_rows);
  moe_combine_batched_kernel<<<grid, block, 0, stream>>>(
      out, expert_y, weight, n_used, hidden);
}

void sigmoid_gate_scalar_batched_f16(uint16_t* out, const float* g,
                                     int n_rows, int hidden,
                                     cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, n_rows);
  sigmoid_gate_scalar_batched_kernel<<<grid, block, 0, stream>>>(
      out, g, hidden);
}

void scatter_rows_generic_f16(uint16_t* dst, const float* src,
                              const int* dst_flat, int n_list, int hidden,
                              cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, n_list);
  scatter_rows_generic_kernel<<<grid, block, 0, stream>>>(dst, src, dst_flat,
                                                          hidden);
}

// ---------------------------------------------------------------------------
// v0.4 device-side expert dispatch (P1): build the whole grouped plan of a
// layer ON DEVICE - no per-layer D2H sync, no host bookkeeping, no job-table
// uploads. One block: histogram of the routed ids, exclusive prefix sums for
// rows (off) and GEMM tiles (toff), row scatter to expert-major order, and a
// mailbox of experts whose weights are absent (null pointer in the chunk
// pointer table -> CPU-pool patch runs after the layer, off the hot path).
// meta[0] = total valid routed rows R, meta[1] = total GEMM tiles,
// meta[2] = rows killed by bounds validation.
// ---------------------------------------------------------------------------
__global__ void moe_dispatch_scan_kernel(const int* idx, float* weight,
                                         int* gslot, int* grows, int* off,
                                         int* toff, int* meta, int* mail,
                                         int* mail_n, int n_rows, int k,
                                         int n_expert, int tok_per) {
  extern __shared__ int smem[];          // hist[n_expert] + fill[n_expert]
  int* hist = smem;
  int* fill = smem + n_expert;
  const int tid = threadIdx.x;
  const int nth = blockDim.x;
  if (tid == 0) {
    meta[0] = 0;
    meta[1] = 0;
    meta[2] = 0;
    *mail_n = 0;
  }
  for (int e = tid; e < n_expert; e += nth) hist[e] = 0;
  __syncthreads();

  // pass 1: histogram + bounds validation (corrupt ids zero the whole row)
  const int total = n_rows * k;
  for (int i = tid; i < total; i += nth) {
    const int r = i / k;
    const int kk = i % k;
    const int e = idx[static_cast<size_t>(r) * (k + 1) + kk];
    if (e < 0 || e >= n_expert) {
      float* wr = weight + static_cast<size_t>(r) * (k + 1);
      for (int q = 0; q <= k; ++q) wr[q] = 0.f;
      atomicAdd(&meta[2], 1);
      continue;
    }
    atomicAdd(&hist[e], 1);
  }
  __syncthreads();

  // exclusive prefix sums (E <= 1024: a single thread is deterministic)
  if (tid == 0) {
    int acc = 0, tacc = 0;
    for (int e = 0; e < n_expert; ++e) {
      off[e] = acc;
      toff[e] = tacc;
      fill[e] = acc;
      acc += hist[e];
      tacc += (hist[e] + tok_per - 1) / tok_per;
    }
    off[n_expert] = acc;
    toff[n_expert] = tacc;
    meta[0] = acc;
    meta[1] = tacc;
  }
  __syncthreads();

  // pass 2: scatter routes to expert-major grouped slots. Order WITHIN an
  // expert is arbitrary (atomics) - every consumer is position-keyed and the
  // per-(row, token) accumulation order is unchanged, so results are
  // bit-identical to the host-compacted path.
  for (int i = tid; i < total; i += nth) {
    const int r = i / k;
    const int kk = i % k;
    const int e = idx[static_cast<size_t>(r) * (k + 1) + kk];
    if (e < 0 || e >= n_expert) {
      gslot[static_cast<size_t>(r) * k + kk] = -1;
      continue;
    }
    const int pos = atomicAdd(&fill[e], 1);
    gslot[static_cast<size_t>(r) * k + kk] = pos;
    grows[pos] = r;
  }
}

void moe_dispatch_scan(const int* idx, float* weight, int* gslot, int* grows,
                       int* off, int* toff, int* meta, int* mail, int* mail_n,
                       int n_rows, int k, int n_expert, int tok_per,
                       cudaStream_t stream) {
  const int smem = 2 * n_expert * sizeof(int);
  moe_dispatch_scan_kernel<<<1, 1024, smem, stream>>>(
      idx, weight, gslot, grows, off, toff, meta, mail, mail_n, n_rows, k,
      n_expert, tok_per);
}

// gather_rows_q8 variant whose row count is read from the device meta block
// (meta[0]) - the host only bounds the launch grid.
__global__ void gather_rows_q8_n_kernel(q4k::BlockQ8K* dst,
                                        const q4k::BlockQ8K* src,
                                        const int* src_rows, const int* meta,
                                        int row_blocks) {
  const int r = blockIdx.y;
  if (r >= *meta) return;
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= row_blocks) return;
  const int src_r = src_rows[r];
  dst[static_cast<size_t>(r) * row_blocks + b] =
      src[static_cast<size_t>(src_r) * row_blocks + b];
}

void gather_rows_q8_n(q4k::BlockQ8K* dst, const q4k::BlockQ8K* src,
                      const int* src_rows, const int* meta, int r_max,
                      int row_blocks, cudaStream_t stream) {
  dim3 block(8), grid((row_blocks + 7) / 8, r_max);
  gather_rows_q8_n_kernel<<<grid, block, 0, stream>>>(dst, src, src_rows, meta,
                                                      row_blocks);
}

// CPU-pool patch: after the batched combine already ran (absent experts
// contributed 0), add the late CPU results for the patched (row, slot) list.
// rk[i] packs r * (n_used + 1) + kk.
__global__ void moe_combine_add_slots_kernel(uint16_t* out,
                                             const uint16_t* expert_y,
                                             const float* weight,
                                             const int* rk, int n_used,
                                             int hidden, int n_list) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  const int li = blockIdx.y;
  const size_t slot = static_cast<size_t>(rk[li]);
  const size_t row = slot / (n_used + 1);
  const float w = weight[slot];
  const uint16_t* y = expert_y + slot * hidden + i;
  float acc = ldh(out + row * hidden + i) + w * ldh(y);
  sth(out + row * hidden + i, acc);
}

void moe_combine_add_slots_f16(uint16_t* out, const uint16_t* expert_y,
                               const float* weight, const int* rk, int n_used,
                               int hidden, int n_list, cudaStream_t stream) {
  dim3 block(256), grid((hidden + 255) / 256, n_list);
  moe_combine_add_slots_kernel<<<grid, block, 0, stream>>>(
      out, expert_y, weight, rk, n_used, hidden, n_list);
}

}  // namespace sager::cuda
