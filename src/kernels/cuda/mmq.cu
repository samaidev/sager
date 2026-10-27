// mmq.cu - v0.8 prefill MMQ GEMM + decode fusion kernels (sm_75 first).
//
// PREFILL: the v0.3 mtok kernels amortize the W row across a token tile but
// serialize the tokens inside one dp4a dependency chain per lane. The MMQ
// kernel keeps the EXACT arithmetic order of gemv_mtok_ro_kernel (ascending
// sub-block groups g = lane + gi*32 per output, then the same 32-lane
// shfl-down butterfly) but interleaves TOK independent token accumulators so
// the dp4a pipelines overlap: per (row, gi) the scale/nibble state is unpacked
// once and consumed by TOK dp4a chains. Bit-identical outputs by construction
// (same per-output accumulation order, same reduction tree, same fp16 write).
//
// DECODE fusions (all reproduce the exact math of the kernels they replace):
//   rmsnorm_quant_f16   - rmsnorm_kernel + quantize_q8_k_kernel in one launch
//   gemv2m/gemv3m       - the 2/3 projection GEMVs of a layer in one launch
//   gemv.._resadd       - projection GEMV with the residual add as epilogue
//   router_topk_fused   - fp32 router GEMV (n=1) + warp-parallel top-k
//   gemv_q6_k_q8_smx    - Q6_K head GEMV with the x blocks staged in smem
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>

namespace sager::cuda {
namespace {

// ---------------------------------------------------------------------------
// shared helpers (same math as q4_k.cu)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mmq_unpack_scales6(const uint8_t* sc, int s,
                                                   float d, float dmin,
                                                   float* sd, float* sm) {
  uint8_t d6, m6;
  if (s < 4) {
    d6 = static_cast<uint8_t>(sc[s] & 63);
    m6 = static_cast<uint8_t>(sc[s + 4] & 63);
  } else {
    d6 = static_cast<uint8_t>((sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4));
    m6 = static_cast<uint8_t>((sc[s + 4] >> 4) | ((sc[s] >> 6) << 4));
  }
  *sd = d * static_cast<float>(d6);
  *sm = dmin * static_cast<float>(m6);
}

struct MmqGState {
  float sd, sm;
  uint32_t wp[8];
};

__device__ __forceinline__ void mmq_g_load(const q4k::BlockQ4K& b, int sb,
                                           MmqGState& st) {
  const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
  const float dmin = __half2float(*reinterpret_cast<const __half*>(&b.dmin));
  mmq_unpack_scales6(b.scales, sb, d, dmin, &st.sd, &st.sm);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t raw =
        *reinterpret_cast<const uint16_t*>(b.qs + sb * 16 + i * 2);
    st.wp[i] = (raw & 0xFu) | ((raw & 0xF0u) << 4) | ((raw & 0xF00u) << 8) |
               ((raw & 0xF000u) << 12);
  }
}

__device__ __forceinline__ int mmq_g_dot_int(const MmqGState& st,
                                             const q4k::BlockQ8K& xb, int sb) {
  const int8_t* xq32 = xb.qs + sb * 32;
  int dot = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    dot = __dp4a(static_cast<int>(st.wp[i]),
                 *reinterpret_cast<const int*>(xq32 + i * 4), dot);
  }
  return dot;
}

// identical fp32 epilogue term as mtok_g_dot
__device__ __forceinline__ float mmq_g_dot(const MmqGState& st,
                                           const q4k::BlockQ8K& xb, int sb) {
  const int dot = mmq_g_dot_int(st, xb, sb);
  return xb.d * (st.sd * static_cast<float>(dot) -
                 st.sm * static_cast<float>(xb.bsums[sb]));
}

__device__ __forceinline__ void mmq_stage(const unsigned char* src,
                                          unsigned char* dst, int bytes,
                                          int tid, int nthreads) {
  for (int i = tid; i < bytes; i += nthreads) dst[i] = src[i];
}

__device__ __forceinline__ void mmq_shfl_reduce(float& acc) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  }
}

// ---------------------------------------------------------------------------
// MMQ: one warp = one output row x TOK tokens. W state unpacked once per
// (row, gi); TOK dp4a chains interleave. Accumulation order per output
// matches gemv_mtok_ro_kernel exactly.
// ---------------------------------------------------------------------------
template <int NG, int TOK>
__global__ void mmq_q4_k_q8_f16out_kernel(
    const q4k::BlockQ4K* __restrict__ w, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ y, int n_rows, int k, int n_tokens) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = k / 256;
  const int w_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ4K));
  const int x_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  q4k::BlockQ4K* w_sh =
      reinterpret_cast<q4k::BlockQ4K*>(smem + warp * w_row_bytes);
  q4k::BlockQ8K* x_sh = reinterpret_cast<q4k::BlockQ8K*>(
      smem + 8 * w_row_bytes);

  const int row = blockIdx.x * 8 + warp;
  const int tok0 = blockIdx.y * TOK;
  const int nt = min(TOK, n_tokens - tok0);
  if (nt <= 0) return;

  for (int t = 0; t < nt; ++t) {
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  xq + static_cast<size_t>(tok0 + t) * nb),
              reinterpret_cast<unsigned char*>(x_sh + static_cast<size_t>(t) * nb),
              x_row_bytes, threadIdx.x, blockDim.x);
  }
  if (row < n_rows) {
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  w + static_cast<size_t>(row) * nb),
              reinterpret_cast<unsigned char*>(w_sh), w_row_bytes, lane, 32);
  }
  __syncthreads();
  if (row >= n_rows) return;

  const int nsg = nb * 8;
  MmqGState st[NG];
  int blk_of[NG], sb_of[NG];
  #pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int g = lane + gi * 32;
    blk_of[gi] = -1;
    sb_of[gi] = 0;
    if (g < nsg) {
      blk_of[gi] = g >> 3;
      sb_of[gi] = g & 7;
      mmq_g_load(w_sh[blk_of[gi]], sb_of[gi], st[gi]);
    }
  }

  float acc[TOK];
  #pragma unroll
  for (int t = 0; t < TOK; ++t) acc[t] = 0.f;

  // gi outer, token inner: TOK independent int chains overlap the dp4a pipe.
  for (int gi = 0; gi < NG; ++gi) {
    if (blk_of[gi] < 0) continue;
    const int blk = blk_of[gi], sb = sb_of[gi];
    int dot[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) dot[t] = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      #pragma unroll
      for (int t = 0; t < TOK; ++t) {
        const q4k::BlockQ8K& xb = x_sh[static_cast<size_t>(t) * nb + blk];
        dot[t] = __dp4a(
            static_cast<int>(st[gi].wp[i]),
            *reinterpret_cast<const int*>(xb.qs + sb * 32 + i * 4), dot[t]);
      }
    }
    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
      const q4k::BlockQ8K& xb = x_sh[static_cast<size_t>(t) * nb + blk];
      acc[t] += xb.d * (st[gi].sd * static_cast<float>(dot[t]) -
                        st[gi].sm * static_cast<float>(xb.bsums[sb]));
    }
  }

  // same per-token butterfly as the GEMV, tokens sequential
  for (int t = 0; t < nt; ++t) {
    float r = acc[t];
    mmq_shfl_reduce(r);
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &y[static_cast<size_t>(tok0 + t) * n_rows + row]) = __float2half(r);
    }
  }
}

// two weight matrices, same n_rows (fused gate+up of the shared expert)
template <int NG, int TOK>
__global__ void mmq2_q4_k_q8_f16out_kernel(
    const q4k::BlockQ4K* __restrict__ wg, const q4k::BlockQ4K* __restrict__ wu,
    const q4k::BlockQ8K* __restrict__ xq, uint16_t* __restrict__ yg,
    uint16_t* __restrict__ yu, int n_rows, int k, int n_tokens) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = k / 256;
  const int w_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ4K));
  const int x_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  q4k::BlockQ4K* wg_sh =
      reinterpret_cast<q4k::BlockQ4K*>(smem + warp * w_row_bytes);
  q4k::BlockQ4K* wu_sh =
      reinterpret_cast<q4k::BlockQ4K*>(smem + (8 + warp) * w_row_bytes);
  q4k::BlockQ8K* x_sh = reinterpret_cast<q4k::BlockQ8K*>(
      smem + 16 * w_row_bytes);

  const int row = blockIdx.x * 8 + warp;
  const int tok0 = blockIdx.y * TOK;
  const int nt = min(TOK, n_tokens - tok0);
  if (nt <= 0) return;

  for (int t = 0; t < nt; ++t) {
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  xq + static_cast<size_t>(tok0 + t) * nb),
              reinterpret_cast<unsigned char*>(x_sh + static_cast<size_t>(t) * nb),
              x_row_bytes, threadIdx.x, blockDim.x);
  }
  if (row < n_rows) {
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  wg + static_cast<size_t>(row) * nb),
              reinterpret_cast<unsigned char*>(wg_sh), w_row_bytes, lane, 32);
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  wu + static_cast<size_t>(row) * nb),
              reinterpret_cast<unsigned char*>(wu_sh), w_row_bytes, lane, 32);
  }
  __syncthreads();
  if (row >= n_rows) return;

  const int nsg = nb * 8;
  MmqGState stg[NG], stu[NG];
  int blk_of[NG], sb_of[NG];
  #pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int g = lane + gi * 32;
    blk_of[gi] = -1;
    sb_of[gi] = 0;
    if (g < nsg) {
      blk_of[gi] = g >> 3;
      sb_of[gi] = g & 7;
      mmq_g_load(wg_sh[blk_of[gi]], sb_of[gi], stg[gi]);
      mmq_g_load(wu_sh[blk_of[gi]], sb_of[gi], stu[gi]);
    }
  }

  float accg[TOK], accu[TOK];
  #pragma unroll
  for (int t = 0; t < TOK; ++t) { accg[t] = 0.f; accu[t] = 0.f; }

  for (int gi = 0; gi < NG; ++gi) {
    if (blk_of[gi] < 0) continue;
    const int blk = blk_of[gi], sb = sb_of[gi];
    int dotg[TOK], dotu[TOK];
    #pragma unroll
    for (int t = 0; t < TOK; ++t) { dotg[t] = 0; dotu[t] = 0; }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      #pragma unroll
      for (int t = 0; t < TOK; ++t) {
        const q4k::BlockQ8K& xb = x_sh[static_cast<size_t>(t) * nb + blk];
        const int xp = *reinterpret_cast<const int*>(xb.qs + sb * 32 + i * 4);
        dotg[t] = __dp4a(static_cast<int>(stg[gi].wp[i]), xp, dotg[t]);
        dotu[t] = __dp4a(static_cast<int>(stu[gi].wp[i]), xp, dotu[t]);
      }
    }
    #pragma unroll
    for (int t = 0; t < TOK; ++t) {
      const q4k::BlockQ8K& xb = x_sh[static_cast<size_t>(t) * nb + blk];
      accg[t] += xb.d * (stg[gi].sd * static_cast<float>(dotg[t]) -
                         stg[gi].sm * static_cast<float>(xb.bsums[sb]));
      accu[t] += xb.d * (stu[gi].sd * static_cast<float>(dotu[t]) -
                         stu[gi].sm * static_cast<float>(xb.bsums[sb]));
    }
  }

  for (int t = 0; t < nt; ++t) {
    float rg = accg[t], ru = accu[t];
    mmq_shfl_reduce(rg);
    mmq_shfl_reduce(ru);
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &yg[static_cast<size_t>(tok0 + t) * n_rows + row]) = __float2half(rg);
      *reinterpret_cast<__half*>(
          &yu[static_cast<size_t>(tok0 + t) * n_rows + row]) = __float2half(ru);
    }
  }
}

size_t mmq_smem(int nb, int w_rows_staged, int tok) {
  return static_cast<size_t>(w_rows_staged) * nb * sizeof(q4k::BlockQ4K) +
         static_cast<size_t>(tok) * nb * sizeof(q4k::BlockQ8K);
}

}  // namespace

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
namespace {

template <int WROWS>
void mmq_dispatch(const q4k::BlockQ4K* w1, const q4k::BlockQ4K* w2,
                  const q4k::BlockQ8K* xq, uint16_t* y1, uint16_t* y2,
                  int n_rows, int k, int n_tokens, cudaStream_t stream) {
  if (n_tokens <= 0) return;
  const int nb = k / 256;
  const int wrows = w2 ? 2 * WROWS : WROWS;
  // pick the largest token tile whose shared-memory footprint still fits the
  // default 48 KB budget (the sm_75 opt-in maximum is 64 KB)
  int tok = 16;
  while (tok > 1 && mmq_smem(nb, wrows, tok) > 48 * 1024) tok >>= 1;
  dim3 grid((n_rows + 7) / 8, (n_tokens + tok - 1) / tok);
  const size_t smem = mmq_smem(nb, wrows, tok);
  const int nsg = nb * 8;
  const int ng = (nsg + 31) / 32;
  if (ng > 4) {   // k > 4096: keep the v0.3 mtok path (launchers gate on this)
    // never reached through the public wrappers below; safety net
    if (w2) {
      mmq2_q4_k_q8_f16out_kernel<8, 1><<<grid, 256, smem, stream>>>(
          w1, w2, xq, y1, y2, n_rows, k, n_tokens);
    } else {
      mmq_q4_k_q8_f16out_kernel<8, 1><<<grid, 256, smem, stream>>>(
          w1, xq, y1, n_rows, k, n_tokens);
    }
    return;
  }
  #define MMQ_LAUNCH(NGV, TOKV)                                            \
    do {                                                                   \
      if (w2) {                                                            \
        mmq2_q4_k_q8_f16out_kernel<NGV, TOKV>                              \
            <<<grid, 256, smem, stream>>>(w1, w2, xq, y1, y2, n_rows, k,    \
                                          n_tokens);                        \
      } else {                                                             \
        mmq_q4_k_q8_f16out_kernel<NGV, TOKV>                               \
            <<<grid, 256, smem, stream>>>(w1, xq, y1, n_rows, k,           \
                                          n_tokens);                        \
      }                                                                    \
    } while (0)
  if (tok >= 16) {
    if (ng <= 1)      MMQ_LAUNCH(1, 16);
    else if (ng <= 2) MMQ_LAUNCH(2, 16);
    else              MMQ_LAUNCH(4, 16);
  } else if (tok >= 8) {
    if (ng <= 1)      MMQ_LAUNCH(1, 8);
    else if (ng <= 2) MMQ_LAUNCH(2, 8);
    else              MMQ_LAUNCH(4, 8);
  } else if (tok >= 4) {
    if (ng <= 1)      MMQ_LAUNCH(1, 4);
    else if (ng <= 2) MMQ_LAUNCH(2, 4);
    else              MMQ_LAUNCH(4, 4);
  } else if (tok >= 2) {
    if (ng <= 1)      MMQ_LAUNCH(1, 2);
    else if (ng <= 2) MMQ_LAUNCH(2, 2);
    else              MMQ_LAUNCH(4, 2);
  } else {
    if (ng <= 1)      MMQ_LAUNCH(1, 1);
    else if (ng <= 2) MMQ_LAUNCH(2, 1);
    else              MMQ_LAUNCH(4, 1);
  }
  #undef MMQ_LAUNCH
}

}  // namespace

void mmq_q4_k_q8_f16out(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                        uint16_t* y, int n_rows, int k, int n_tokens,
                        cudaStream_t stream) {
  mmq_dispatch<8>(w, nullptr, xq, y, nullptr, n_rows, k, n_tokens, stream);
}

void mmq2_q4_k_q8_f16out(const q4k::BlockQ4K* wg, const q4k::BlockQ4K* wu,
                         const q4k::BlockQ8K* xq, uint16_t* yg, uint16_t* yu,
                         int n_rows, int k, int n_tokens, cudaStream_t stream) {
  mmq_dispatch<8>(wg, wu, xq, yg, yu, n_rows, k, n_tokens, stream);
}

// ---------------------------------------------------------------------------
// v0.8 decode fusions. Every kernel below reproduces the exact arithmetic of
// the kernels it replaces (same values, same order, same rounding) so the
// token stream stays bit-identical; they only merge launches / round trips.
// ---------------------------------------------------------------------------

namespace {

// warp-per-row fp32 GEMV over fp16 x staged in smem; same loop as
// gemv_f32_out_kernel (q4_k.cu) but x comes from shared memory (same values).
__global__ void router_gemv_smem_kernel(const float* __restrict__ w,
                                        const uint16_t* __restrict__ x_f16,
                                        float* __restrict__ y, int n_rows,
                                        int k) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint16_t* x_sh = reinterpret_cast<uint16_t*>(smem);
  const int tid = threadIdx.x;
  for (int i = tid; i < k; i += blockDim.x) x_sh[i] = x_f16[i];
  __syncthreads();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int row = blockIdx.x * (blockDim.x >> 5) + warp; row < n_rows;
       row += gridDim.x * (blockDim.x >> 5)) {
    const float* roww = w + static_cast<size_t>(row) * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) {
      const float xv = __half2float(*reinterpret_cast<const __half*>(&x_sh[i]));
      acc += roww[i] * xv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    if (lane == 0) y[row] = acc;
  }
}

// warp-parallel top-k over smem probs: bit-exact with the serial
// first-strict-max walk of router_topk_batched_kernel (v0.7's proven rule:
// lower index wins value ties at every reduction step).
__device__ __forceinline__ void router_warp_topk(float* prob, int n_expert,
                                                 int k, int* out_idx,
                                                 float* out_w) {
  const int lane = threadIdx.x & 31;
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
      out_idx[kk] = bi;
      out_w[kk] = bv;
      wsum += bv;
    }
    __syncwarp();
  }
  if (lane == 0 && wsum > 0.f) {
    for (int kk = 0; kk < k; ++kk) out_w[kk] /= wsum;
  }
}

// fused router for the single-token decode step: logits + top-k in ONE launch
// (replaces gemv_f32_out_f16x + router_topk_batched for n=1; the softmax and
// selection semantics match router_topk_batched_kernel for row 0).
__global__ void router_topk_fused_kernel(const float* __restrict__ w,
                                         const uint16_t* __restrict__ x_f16,
                                         int* __restrict__ idx,
                                         float* __restrict__ weight,
                                         int n_expert, int k, int hidden) {
  extern __shared__ __align__(16) unsigned char smem[];
  uint16_t* x_sh = reinterpret_cast<uint16_t*>(smem);
  float* logits = reinterpret_cast<float*>(smem + hidden * sizeof(uint16_t));
  const int tid = threadIdx.x;
  for (int i = tid; i < hidden; i += blockDim.x) x_sh[i] = x_f16[i];
  __syncthreads();

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  for (int row = warp; row < n_expert; row += nwarps) {
    const float* roww = w + static_cast<size_t>(row) * hidden;
    float acc = 0.f;
    for (int i = lane; i < hidden; i += 32) {
      const float xv =
          __half2float(*reinterpret_cast<const __half*>(&x_sh[i]));
      acc += roww[i] * xv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    if (lane == 0) logits[row] = acc;
  }
  __syncthreads();

  // identical m/z + prob fill as router_topk_batched_kernel (row 0)
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
    logits[i] = expf(logits[i] - m_sh) / z_sh;   // probs overwrite logits
  }
  __syncthreads();
  if (tid < 32) {
    float* out_w = weight;   // stride 1 row: (K+1) slots, kernel fills K
    int* out_idx = idx;
    router_warp_topk(logits, n_expert, k, out_idx, out_w);
  }
}

}  // namespace

void router_topk_fused(const float* w_f32, const uint16_t* x_f16, int* idx,
                       float* weight, int n_expert, int n_expert_used,
                       int hidden, cudaStream_t stream) {
  const int b = 512;
  const size_t smem = static_cast<size_t>(hidden) * sizeof(uint16_t) +
                      static_cast<size_t>(n_expert) * sizeof(float);
  router_topk_fused_kernel<<<1, b, smem, stream>>>(
      w_f32, x_f16, idx, weight, n_expert, n_expert_used, hidden);
}

// ---------------------------------------------------------------------------
// fused rmsnorm + Q8_K quantization (one launch, bit-identical):
// phase 1 replicates rmsnorm_kernel's reduction and output write; phase 2
// replicates quantize_q8_k_kernel per 256-value block with 256-thread
// groups (warp max-reduce is order-free for fmaxf; the d/id/rintf/clamp and
// bsums integer sums are unchanged). Group barriers are global: every group
// runs the SAME number of loop iterations (rounded up, work skipped past n)
// so __syncthreads stays aligned.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_quant_f16_kernel(uint16_t* out, const uint16_t* x,
                                         const uint16_t* w,
                                         q4k::BlockQ8K* xq, int n, float eps) {
  __shared__ float red[32];
  __shared__ float qred[8][32];
  __shared__ float qd[8], qid[8];
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  // ---- phase 1: rmsnorm reduction (exact rmsnorm_kernel order) ----
  float acc = 0.f;
  for (int i = tid; i < n; i += stride) {
    const float v = __half2float(*reinterpret_cast<const __half*>(&x[i]));
    acc += v * v;
  }
  const int lane = tid & 31, warp = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float s = 0.f;
    const int nw = (blockDim.x + 31) / 32;
    for (int i = 0; i < nw; ++i) s += red[i];
    red[0] = rsqrtf(s / static_cast<float>(n) + eps);
  }
  __syncthreads();
  for (int i = tid; i < n; i += stride) {
    *reinterpret_cast<__half*>(&out[i]) =
        __float2half(__half2float(*reinterpret_cast<const __half*>(&x[i])) *
                     red[0] *
                     __half2float(*reinterpret_cast<const __half*>(&w[i])));
  }

  // ---- phase 2: per-256 quantize groups (exact quantize_q8_k_kernel) ----
  const int nblocks = n / 256;
  if (nblocks <= 0 || blockDim.x < 256) return;   // launcher guarantees both
  const int ngroups = blockDim.x / 256;
  const int g0 = tid >> 8;                        // 256 threads per group
  const int tid_l = tid & 255;
  const int nw = 8;                               // 256/32 warps per group
  const int iters = (nblocks + ngroups - 1) / ngroups;
  for (int it = 0; it < iters; ++it) {
    const int bb = g0 + it * ngroups;
    const int idx = bb * 256 + tid_l;
    const bool live = bb < nblocks;
    float v = 0.f;
    if (live) {
      // recompute the normalized value exactly as phase 1 wrote it
      const float vq =
          __half2float(*reinterpret_cast<const __half*>(&x[idx])) * red[0] *
          __half2float(*reinterpret_cast<const __half*>(&w[idx]));
      v = __half2float(__float2half(vq));          // the fp16 xn value
    }
    float a = live ? fabsf(v) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, off));
    }
    if (lane == 0) qred[g0][tid_l >> 5] = a;
    __syncthreads();
    if (tid_l == 0 && live) {
      float m = 1e-30f;
      for (int i = 0; i < nw; ++i) m = fmaxf(m, qred[g0][i]);
      const float d = m / 127.f;
      qd[g0] = d;
      qid[g0] = d > 0.f ? 1.f / d : 0.f;
    }
    __syncthreads();
    if (live && idx < n) {
      const int q = static_cast<int>(rintf(v * qid[g0]));
      xq[bb].qs[tid_l] =
          static_cast<int8_t>(max(-127, min(127, q)));
      xq[bb].d = qd[g0];
    }
    __syncthreads();
    if (live && tid_l < 8) {
      int sum = 0;
      for (int i = 0; i < 32; ++i) sum += xq[bb].qs[tid_l * 32 + i];
      xq[bb].bsums[tid_l] = static_cast<int16_t>(sum);
    }
    __syncthreads();
  }
}

void rmsnorm_quant_f16(uint16_t* out, const uint16_t* x, const uint16_t* w,
                       q4k::BlockQ8K* xq, int n, float eps, int n_rows,
                       cudaStream_t stream) {
  int b = ((n + 31) / 32) * 32;
  if (b > 1024) b = 1024;
  if (n % 256 != 0 || b < 256) {
    // unsupported geometry: fall back to the two-kernel sequence
    rmsnorm_batched_f16(out, x, w, n, eps, n_rows, stream);
    quantize_q8_k(out, xq, static_cast<size_t>(n_rows) * n, stream);
    return;
  }
  rmsnorm_quant_f16_kernel<<<n_rows, b, 0, stream>>>(out, x, w, xq, n, eps);
}

// ---------------------------------------------------------------------------
// multi-matrix GEMV for the single-token decode step: the layer's 2 (GDN:
// qkvz|ba) or 3 (GQA: q|k|v) projections in ONE launch. Per-row math is
// exactly gemv_mtok_ro_kernel<NG> with n_tokens=1; rows map to their matrix
// by range, outputs are the original per-matrix buffers.
// ---------------------------------------------------------------------------
namespace {

// v0.8 r4: the per-matrix pointers/row-counts ride as DIRECT kernel
// arguments (60 B of the 4 KB param space) - the r3 design copied a param
// block per layer, and 40 tiny H2D memcpy nodes cost ~100 us of
// copy-engine round trip EACH on the T4 whole-step graph (the exact
// failure mode v0.7 documented for per-layer D2H mirrors).
template <int NG, int NM>
__global__ void gemv_nm_q4_k_q8_f16out_kernel(
    const q4k::BlockQ4K* w1, const q4k::BlockQ4K* w2,
    const q4k::BlockQ4K* w3, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* y1, uint16_t* y2, uint16_t* y3, int rows1, int rows2,
    int rows3, int k) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = k / 256;
  const int w_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ4K));
  const int x_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  q4k::BlockQ4K* w_sh =
      reinterpret_cast<q4k::BlockQ4K*>(smem + warp * w_row_bytes);
  q4k::BlockQ8K* x_sh = reinterpret_cast<q4k::BlockQ8K*>(
      smem + 8 * w_row_bytes);

  // scalar select: dynamically-indexed local arrays spill to local memory
  // (measured +80 us/layer on the T4 step graph) - keep everything in scalars
  const int r12 = rows1 + rows2;
  const int total = (NM == 2) ? rows2 : rows3;
  const int row = blockIdx.x * 8 + warp;
  mmq_stage(reinterpret_cast<const unsigned char*>(xq),
            reinterpret_cast<unsigned char*>(x_sh), x_row_bytes,
            threadIdx.x, blockDim.x);
  int m = 0;
  int rlocal = row;
  if (NM == 2) {
    if (row >= rows1) { m = 1; rlocal = row - rows1; }
  } else {
    if (row >= r12) { m = 2; rlocal = row - r12; }
    else if (row >= rows1) { m = 1; rlocal = row - rows1; }
  }
  const q4k::BlockQ4K* wm = (m == 0) ? w1 : ((m == 1) ? w2 : w3);
  uint16_t* ym = (m == 0) ? y1 : ((m == 1) ? y2 : y3);
  if (row < total) {
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  wm + static_cast<size_t>(rlocal) * nb),
              reinterpret_cast<unsigned char*>(w_sh), w_row_bytes, lane, 32);
  }
  __syncthreads();
  if (row >= total) return;

  const int nsg = nb * 8;
  MmqGState st[NG];
  int blk_of[NG], sb_of[NG];
  #pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int g = lane + gi * 32;
    blk_of[gi] = -1;
    sb_of[gi] = 0;
    if (g < nsg) {
      blk_of[gi] = g >> 3;
      sb_of[gi] = g & 7;
      mmq_g_load(w_sh[blk_of[gi]], sb_of[gi], st[gi]);
    }
  }

  const q4k::BlockQ8K* xb_row = x_sh;
  float acc = 0.f;
  #pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    if (blk_of[gi] >= 0)
      acc += mmq_g_dot(st[gi], xb_row[blk_of[gi]], sb_of[gi]);
  }
  mmq_shfl_reduce(acc);
  if (lane == 0) {
    *reinterpret_cast<__half*>(&ym[rlocal]) = __float2half(acc);
  }
}

template <int NM>
void gemv_nm_dispatch(const q4k::BlockQ4K* const* ws, const q4k::BlockQ8K* xq,
                      uint16_t* const* ys, const int* rowss, int k,
                      cudaStream_t stream) {
  const int nb = k / 256;
  const size_t smem = mmq_smem(nb, 8, 1);
  const int nsg = nb * 8;
  const int ng = (nsg + 31) / 32;
  const int rows = rowss[NM - 1];
  #define GNM_LAUNCH(NGV)                                                  \
    do {                                                                   \
      if (NM == 2) {                                                       \
        gemv_nm_q4_k_q8_f16out_kernel<NGV, 2>                              \
            <<<(rows + 7) / 8, 256, smem, stream>>>(                       \
                ws[0], ws[1], nullptr, xq, ys[0], ys[1], nullptr, rowss[0],\
                rowss[1], 0, k);                                           \
      } else {                                                             \
        gemv_nm_q4_k_q8_f16out_kernel<NGV, 3>                              \
            <<<(rows + 7) / 8, 256, smem, stream>>>(                       \
                ws[0], ws[1], ws[2], xq, ys[0], ys[1], ys[2], rowss[0],    \
                rowss[1], rowss[2], k);                                    \
      }                                                                    \
    } while (0)
  if (ng <= 1)      GNM_LAUNCH(1);
  else if (ng <= 2) GNM_LAUNCH(2);
  else if (ng <= 4) GNM_LAUNCH(4);
  else {
    // k > 4096: fall back to separate GEMVs (rare; dense k only)
    for (int m = 0; m < NM; ++m) {
      const int r0 = (m == 0) ? 0 : rowss[m - 1];
      gemv_q4_k_q8_f16out(ws[m], xq, ys[m], rowss[m] - r0, k, stream);
    }
  }
  #undef GNM_LAUNCH
}

}  // namespace

void gemv2m_q4_k_q8_f16out(const q4k::BlockQ4K* w1, const q4k::BlockQ4K* w2,
                           const q4k::BlockQ8K* xq, uint16_t* y1, uint16_t* y2,
                           int rows1, int rows2, int k, int slot,
                           cudaStream_t stream) {
  (void)slot;
  const q4k::BlockQ4K* ws[2] = {w1, w2};
  uint16_t* ys[2] = {y1, y2};
  const int rowss[2] = {rows1, rows1 + rows2};
  gemv_nm_dispatch<2>(ws, xq, ys, rowss, k, stream);
}

void gemv3m_q4_k_q8_f16out(const q4k::BlockQ4K* w1, const q4k::BlockQ4K* w2,
                           const q4k::BlockQ4K* w3, const q4k::BlockQ8K* xq,
                           uint16_t* y1, uint16_t* y2, uint16_t* y3, int rows1,
                           int rows2, int rows3, int k, int slot,
                           cudaStream_t stream) {
  (void)slot;
  const q4k::BlockQ4K* ws[3] = {w1, w2, w3};
  uint16_t* ys[3] = {y1, y2, y3};
  const int rowss[3] = {rows1, rows1 + rows2, rows1 + rows2 + rows3};
  gemv_nm_dispatch<3>(ws, xq, ys, rowss, k, stream);
}

template <int NG>
__device__ __forceinline__ float resadd_row_body(
    const q4k::BlockQ4K* w_sh, const q4k::BlockQ8K* x_sh, int lane, int nb) {
  const int nsg = nb * 8;
  MmqGState st[NG];
  int blk_of[NG], sb_of[NG];
  #pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int g = lane + gi * 32;
    blk_of[gi] = -1;
    sb_of[gi] = 0;
    if (g < nsg) {
      blk_of[gi] = g >> 3;
      sb_of[gi] = g & 7;
      mmq_g_load(w_sh[blk_of[gi]], sb_of[gi], st[gi]);
    }
  }
  float acc = 0.f;
  #pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    if (blk_of[gi] >= 0) acc += mmq_g_dot(st[gi], x_sh[blk_of[gi]], sb_of[gi]);
  }
  return acc;
}

// ---------------------------------------------------------------------------
// GEMV with fused residual-add epilogue (single token): y_row = f16(acc) is
// produced exactly as gemv..f16out does, then the epilogue applies
// add_kernel's semantics to the residual row in place:
//   res[i] = f16( f32(res[i]) + f32(y_row) )
// ---------------------------------------------------------------------------
__global__ void gemv_resadd_q4_k_q8_f16out_kernel(
    const q4k::BlockQ4K* __restrict__ w, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ y, uint16_t* __restrict__ res_inout, int n_rows,
    int k) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = k / 256;
  const int w_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ4K));
  const int x_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  q4k::BlockQ4K* w_sh =
      reinterpret_cast<q4k::BlockQ4K*>(smem + warp * w_row_bytes);
  q4k::BlockQ8K* x_sh = reinterpret_cast<q4k::BlockQ8K*>(
      smem + 8 * w_row_bytes);

  const int row = blockIdx.x * 8 + warp;
  mmq_stage(reinterpret_cast<const unsigned char*>(xq),
            reinterpret_cast<unsigned char*>(x_sh), x_row_bytes,
            threadIdx.x, blockDim.x);
  if (row < n_rows) {
    mmq_stage(reinterpret_cast<const unsigned char*>(
                  w + static_cast<size_t>(row) * nb),
              reinterpret_cast<unsigned char*>(w_sh), w_row_bytes, lane, 32);
  }
  __syncthreads();
  if (row >= n_rows) return;

  const int nsg = nb * 8;
  const int ng = (nsg + 31) / 32;
  float acc = 0.f;
  // ng is a launch-time constant per shape; keep the hot loop in registers
  // (a runtime-bounded st[4] array spills to local memory: 46 vs 23 us/step)
  if (ng <= 1) {
    acc = resadd_row_body<1>(w_sh, x_sh, lane, nb);
  } else if (ng <= 2) {
    acc = resadd_row_body<2>(w_sh, x_sh, lane, nb);
  } else {
    acc = resadd_row_body<4>(w_sh, x_sh, lane, nb);
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) {
    const __half oh = __float2half(acc);
    *reinterpret_cast<__half*>(&y[row]) = oh;
    const float o = __half2float(oh);
    const float rv =
        __half2float(*reinterpret_cast<const __half*>(&res_inout[row]));
    *reinterpret_cast<__half*>(&res_inout[row]) = __float2half(rv + o);
  }
}

void gemv_q4_k_q8_f16out_resadd(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                                uint16_t* y, uint16_t* res_inout, int n_rows,
                                int k, cudaStream_t stream) {
  const int nb = k / 256;
  const size_t smem = mmq_smem(nb, 8, 1);
  const int nsg = nb * 8;
  const int ng = (nsg + 31) / 32;
  if (ng > 4) {
    gemv_q4_k_q8_f16out(w, xq, y, n_rows, k, stream);
    add_f16(res_inout, y, n_rows, stream);
    return;
  }
  gemv_resadd_q4_k_q8_f16out_kernel<<<(n_rows + 7) / 8, 256, smem, stream>>>(
      w, xq, y, res_inout, n_rows, k);
}

// ---------------------------------------------------------------------------
// Q6_K head GEMV, sm_75 tuned: the x blocks are staged in shared memory once
// per 8-row block (the row loop re-read them from L2/DRAM before) and the
// ql/qh nibble sources are loaded as whole uint32 words. Same values, same
// accumulation order as gemv_q6_k_q8_kernel - bit-identical logits.
// ---------------------------------------------------------------------------
__global__ void gemv_q6_k_q8_smx_kernel(const uint8_t* __restrict__ w6,
                                        const q4k::BlockQ8K* __restrict__ xq,
                                        float* __restrict__ y, int n_rows,
                                        int k) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int nb = k / 256;
  q4k::BlockQ8K* x_sh = reinterpret_cast<q4k::BlockQ8K*>(smem);
  const int x_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  mmq_stage(reinterpret_cast<const unsigned char*>(xq),
            reinterpret_cast<unsigned char*>(x_sh), x_row_bytes,
            threadIdx.x, blockDim.x);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const uint8_t* roww =
      w6 + static_cast<size_t>(row) * static_cast<size_t>(nb) * 210;

  float acc = 0.f;
  for (int sg = lane; sg < nb * 16; sg += 32) {
    const int blk = sg >> 4;
    const int t = sg & 15;
    const int qslot = (t & 7) >> 1;
    const int is = t & 1;
    const uint8_t* blk6 = roww + static_cast<size_t>(blk) * 210;
    const uint8_t* ql = blk6 + (qslot & 1) * 32 + is * 16;
    const uint8_t* qh = blk6 + 128 + is * 16;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk6 + 192);
    const float d = __half2float(*reinterpret_cast<const __half*>(blk6 + 208));
    const q4k::BlockQ8K& xb = x_sh[blk];
    const int colbase = (t >> 3) * 128 + qslot * 32 + is * 16;
    const int nib = (qslot & 2) ? 4 : 0;
    const int hshift = qslot * 2;

    int isum = 0, xsum = 0;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {         // 4 weights per dp4a
      uint32_t wpack = 0;
      uint32_t xpack;
      int x0, x1, x2, x3;
      {
        const int8_t* xp = xb.qs + colbase + g * 4;
        x0 = xp[0]; x1 = xp[1]; x2 = xp[2]; x3 = xp[3];
        xpack = static_cast<uint32_t>(static_cast<uint8_t>(x0)) |
                (static_cast<uint32_t>(static_cast<uint8_t>(x1)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(x2)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(x3)) << 24);
      }
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int j = g * 4 + q;
        const int lo = (ql[j] >> nib) & 0xF;
        const int hi = (qh[j] >> hshift) & 3;
        const int qv = lo | (hi << 4);
        if (q == 0) wpack = static_cast<uint32_t>(qv);
        else wpack |= static_cast<uint32_t>(qv) << (q * 8);
      }
      isum = __dp4a(static_cast<int>(wpack), static_cast<int>(xpack), isum);
      xsum = __dp4a(static_cast<int>(xpack), 0x01010101, xsum);
    }
    acc += d * static_cast<float>(sc[t]) *
           static_cast<float>(isum - 32 * xsum);
  }

  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) y[row] = acc;
}

void gemv_q6_k_q8_smx(const void* w6, const q4k::BlockQ8K* xq, float* y,
                      int n_rows, int k, cudaStream_t stream) {
  const int nb = k / 256;
  const size_t smem = static_cast<size_t>(nb) * sizeof(q4k::BlockQ8K);
  gemv_q6_k_q8_smx_kernel<<<(n_rows + 7) / 8, 256, smem, stream>>>(
      static_cast<const uint8_t*>(w6), xq, y, n_rows, k);
}

}  // namespace sager::cuda
