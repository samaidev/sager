// q4_k.cu - Q4_K GEMV + activation quantization kernels (sm_75 first).
//
// Decode on T4 is bandwidth-bound: one token activates ~3B params, so the
// kernel's job is to stream the 4-bit weights at full 320 GB/s and keep the
// dequant overhead off the critical path. We use one warp per output row,
// dp4a int8 dots for the nibble*int8 products (Turing INT32 path, 4x INT8
// throughput), fp32 accumulation with exact per-sub-block scale handling -
// bit-compatible with the AVX2 CPU pool path and the host reference.
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>

namespace sager::cuda {
namespace {

__device__ __forceinline__ void unpack_scales6(const uint8_t* sc, int s,
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

// One sub-block (32 weights = 16 qs bytes) dot via 8 dp4a; returns int dot.
// Each iteration: 2 bytes of nibbles (4 weights, low nibble first) x 4 int8s.
__device__ __forceinline__ int dp4a_subblock(const uint8_t* qs16,
                                             const int8_t* xq32) {
  int acc_i = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t raw = *reinterpret_cast<const uint16_t*>(qs16 + i * 2);
    const uint32_t xpack = *reinterpret_cast<const uint32_t*>(xq32 + i * 4);
    // nibbles: even index = low nibble, odd = high nibble of each byte.
    // Restore natural order w0,w1,w2,w3 within each 4-weight group.
    const int w0 = static_cast<int>(raw & 0xFu);
    const int w1 = static_cast<int>((raw >> 4) & 0xFu);
    const int w2 = static_cast<int>((raw >> 8) & 0xFu);
    const int w3 = static_cast<int>((raw >> 12) & 0xFu);
    const uint32_t wpack = static_cast<uint32_t>(w0) |
                           (static_cast<uint32_t>(w1) << 8) |
                           (static_cast<uint32_t>(w2) << 16) |
                           (static_cast<uint32_t>(w3) << 24);
    // Cast to int: nibble bytes are 0..15 so signed interpretation is exact,
    // while the int8 activation bytes MUST be read as signed. The unsigned
    // overload would corrupt every negative activation.
    acc_i = __dp4a(static_cast<int>(wpack), static_cast<int>(xpack), acc_i);
  }
  return acc_i;
}

// One warp computes one output row: y[r] = W[r, :] @ xq[:].
// Launch: block = 8 warps (256 threads), each warp one row.
__global__ void gemv_q4_k_q8_kernel(const q4k::BlockQ4K* __restrict__ w,
                                    const q4k::BlockQ8K* __restrict__ xq,
                                    float* __restrict__ y, int n_rows, int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;

  const int nb = k / 256;                       // super-blocks per row
  const int nsg = nb * 8;                       // sub-blocks per row
  const q4k::BlockQ4K* roww = w + static_cast<size_t>(row) * nb;

  float acc = 0.f;
  // Each lane strides over the row's sub-blocks (g = lane, lane+32, ...).
  for (int g = lane; g < nsg; g += 32) {
    const int blk = g >> 3;                     // super-block
    const int sb = g & 7;                       // sub-block within it
    const q4k::BlockQ4K& b = roww[blk];
    const q4k::BlockQ8K& xb = xq[blk];
    const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&b.dmin));
    float sd, sm;
    unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
    const int dot = dp4a_subblock(b.qs + sb * 16, xb.qs + sb * 32);
    const float xsum = static_cast<float>(xb.bsums[sb]);
    acc += xb.d * (sd * static_cast<float>(dot) - sm * xsum);
  }

  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  }
  if (lane == 0) y[row] = acc;
}

// fp16 activations -> BlockQ8K on device. One block per 256 values.
__global__ void quantize_q8_k_kernel(const uint16_t* x, q4k::BlockQ8K* out,
                                     int n) {
  __shared__ float red[32];
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int idx = b * 256 + tid;
  const float v = (idx < n)
      ? __half2float(*reinterpret_cast<const __half*>(&x[idx])) : 0.f;
  float a = fabsf(v);

  const int lane = tid & 31, warp = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, off));
  if (lane == 0) red[warp] = a;
  __syncthreads();
  if (tid == 0) {
    float m = 1e-30f;
    const int nw = (blockDim.x + 31) / 32;
    for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
    red[0] = m;
  }
  __syncthreads();
  const float amax = red[0];
  const float d = amax / 127.f;
  const float id = d > 0.f ? 1.f / d : 0.f;
  const int q = static_cast<int>(rintf(v * id));
  if (idx < n) {
    out[b].qs[tid] = static_cast<int8_t>(max(-127, min(127, q)));
    out[b].d = d;
  }
  __syncthreads();
  if (tid < 8) {
    int sum = 0;
    for (int i = 0; i < 32; ++i) sum += out[b].qs[tid * 32 + i];
    out[b].bsums[tid] = static_cast<int16_t>(sum);
  }
}

// Router logits: W fp32 [n_expert x hidden] @ x fp16 -> y fp32.
__global__ void gemv_f32_out_kernel(const float* __restrict__ w,
                                    const uint16_t* __restrict__ x,
                                    float* __restrict__ y, int n_rows, int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const float* roww = w + static_cast<size_t>(row) * k;
  float acc = 0.f;
  for (int i = lane; i < k; i += 32) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(&x[i]));
    acc += roww[i] * xv;
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) y[row] = acc;
}

// fp16-out variant: same warp-per-row structure, y stored as fp16.
__global__ void gemv_q4_k_q8_f16_kernel(const q4k::BlockQ4K* __restrict__ w,
                                        const q4k::BlockQ8K* __restrict__ xq,
                                        uint16_t* __restrict__ y, int n_rows,
                                        int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const int nb = k / 256;
  const int nsg = nb * 8;
  const q4k::BlockQ4K* roww = w + static_cast<size_t>(row) * nb;
  float acc = 0.f;
  for (int g = lane; g < nsg; g += 32) {
    const int blk = g >> 3, sb = g & 7;
    const q4k::BlockQ4K& b = roww[blk];
    const q4k::BlockQ8K& xb = xq[blk];
    const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&b.dmin));
    float sd, sm;
    unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
    const int dot = dp4a_subblock(b.qs + sb * 16, xb.qs + sb * 32);
    acc += xb.d * (sd * static_cast<float>(dot) -
                   sm * static_cast<float>(xb.bsums[sb]));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) *reinterpret_cast<__half*>(&y[row]) = __float2half(acc);
}

// Pointer-table batched variant: row r of output maps to
// (expert e = r / rows_per, sub-row s = r % rows_per); W = ptrs[e] + s*nb.
__global__ void gemv_ptrs_q4_k_q8_f16_kernel(
    const q4k::BlockQ4K* const* __restrict__ ptrs,
    const q4k::BlockQ8K* __restrict__ xq, uint16_t* __restrict__ y,
    int rows_per, int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  const int e = row / rows_per;
  const int s = row % rows_per;
  const q4k::BlockQ4K* wrow = ptrs[e] + static_cast<size_t>(s) * (k / 256);
  const int nb = k / 256;
  const int nsg = nb * 8;
  float acc = 0.f;
  for (int g = lane; g < nsg; g += 32) {
    const int blk = g >> 3, sb = g & 7;
    const q4k::BlockQ4K& b = wrow[blk];
    const q4k::BlockQ8K& xb = xq[blk];
    const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&b.dmin));
    float sd, sm;
    unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
    const int dot = dp4a_subblock(b.qs + sb * 16, xb.qs + sb * 32);
    acc += xb.d * (sd * static_cast<float>(dot) -
                   sm * static_cast<float>(xb.bsums[sb]));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) *reinterpret_cast<__half*>(&y[row]) = __float2half(acc);
}

// ---- batched variants (prefill / verify window): one extra grid dim over
// tokens. Activation rows are contiguous and 256-divisible, so xq slices per
// token are simply xq + tok * (k / 256).
__global__ void gemv_q4_k_q8_f16_batched_kernel(
    const q4k::BlockQ4K* __restrict__ w, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ y, int n_rows, int k, int n_tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const int tok = blockIdx.y;
  const int nb = k / 256;
  const int nsg = nb * 8;
  const q4k::BlockQ4K* roww = w + static_cast<size_t>(row) * nb;
  const q4k::BlockQ8K* xb_row = xq + static_cast<size_t>(tok) * nb;
  float acc = 0.f;
  for (int g = lane; g < nsg; g += 32) {
    const int blk = g >> 3, sb = g & 7;
    const q4k::BlockQ4K& b = roww[blk];
    const q4k::BlockQ8K& xb = xb_row[blk];
    const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&b.dmin));
    float sd, sm;
    unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
    const int dot = dp4a_subblock(b.qs + sb * 16, xb.qs + sb * 32);
    acc += xb.d * (sd * static_cast<float>(dot) -
                   sm * static_cast<float>(xb.bsums[sb]));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) {
    *reinterpret_cast<__half*>(&y[static_cast<size_t>(tok) * n_rows + row]) =
        __float2half(acc);
  }
}

__global__ void gemv_q4_k_q8_batched_kernel(const q4k::BlockQ4K* __restrict__ w,
                                            const q4k::BlockQ8K* __restrict__ xq,
                                            float* __restrict__ y, int n_rows,
                                            int k, int n_tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const int tok = blockIdx.y;
  const int nb = k / 256;
  const int nsg = nb * 8;
  const q4k::BlockQ4K* roww = w + static_cast<size_t>(row) * nb;
  const q4k::BlockQ8K* xb_row = xq + static_cast<size_t>(tok) * nb;
  float acc = 0.f;
  for (int g = lane; g < nsg; g += 32) {
    const int blk = g >> 3, sb = g & 7;
    const q4k::BlockQ4K& b = roww[blk];
    const q4k::BlockQ8K& xb = xb_row[blk];
    const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&b.dmin));
    float sd, sm;
    unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
    const int dot = dp4a_subblock(b.qs + sb * 16, xb.qs + sb * 32);
    acc += xb.d * (sd * static_cast<float>(dot) -
                   sm * static_cast<float>(xb.bsums[sb]));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) y[static_cast<size_t>(tok) * n_rows + row] = acc;
}

// Router logits, batched: W fp32 [n_rows x k] @ x fp16 [T x k] -> y fp32.
__global__ void gemv_f32_out_batched_kernel(const float* __restrict__ w,
                                            const uint16_t* __restrict__ x,
                                            float* __restrict__ y, int n_rows,
                                            int k, int n_tokens) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const int tok = blockIdx.y;
  const float* roww = w + static_cast<size_t>(row) * k;
  const uint16_t* xrow = x + static_cast<size_t>(tok) * k;
  float acc = 0.f;
  for (int i = lane; i < k; i += 32) {
    acc += roww[i] * __half2float(*reinterpret_cast<const __half*>(&xrow[i]));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) y[static_cast<size_t>(tok) * n_rows + row] = acc;
}

// ============================================================================
// v0.3 mtok tiled kernels (multi-token GEMM for prefill / verify windows).
//
// The v0.2.3 "batched" kernels put n_tokens in grid.y, so EVERY token's warp
// re-reads the whole weight row from DRAM: a 512-token chunk with 8 experts
// routes 4096 rows/layer over 256 experts (m ~= 16 rows/expert) and pays a
// ~16x memory amplification (~290 GB of weight traffic per chunk on T4).
//
// The mtok kernels tile tokens: one block stages TILE_M activation rows and
// 8 weight rows in shared memory, so the weight row is read from DRAM once
// per TILE_M tokens (8..32x less traffic at the same arithmetic).
//
// Bit-compatibility: the per-(row, token) accumulation keeps the exact
// lane-strided sub-block order of the original kernels (g = lane, lane+32,
// ... ascending, then shfl_down reduce), so outputs are bit-identical to
// the v0.2.3 batched path - the A/B token-stream gate stays valid.
// ============================================================================

namespace {

constexpr int kMtokTile = 32;   // tokens per block (register/shared budget)

// copy n bytes src -> dst. Falls back to 4B/1B chunks when either pointer
// lacks 16B alignment (x rows at k=512 are 552B-strided: 4B-aligned only).
__device__ __forceinline__ void stage_bytes(void* dst, const void* src,
                                            int n, int tid, int nthr) {
  uint8_t* d = static_cast<uint8_t*>(dst);
  const uint8_t* s = static_cast<const uint8_t*>(src);
  const bool align16 =
      ((((uintptr_t)d) | ((uintptr_t)s)) & 15u) == 0u;
  if (align16) {
    const int n16 = n & ~15;
    uint4* d4 = reinterpret_cast<uint4*>(d);
    const uint4* s4 = reinterpret_cast<const uint4*>(s);
    for (int i = tid; i < n16 / 16; i += nthr) d4[i] = s4[i];
    for (int i = n16 + tid * 4; i < n; i += nthr * 4) {
      if (i + 4 <= n) {
        *reinterpret_cast<uint32_t*>(d + i) =
            *reinterpret_cast<const uint32_t*>(s + i);
      } else {
        for (int j = i; j < n; ++j) d[j] = s[j];
      }
    }
  } else {
    const int n4 = n & ~3;
    uint32_t* d1 = reinterpret_cast<uint32_t*>(d);
    const uint32_t* s1 = reinterpret_cast<const uint32_t*>(s);
    for (int i = tid; i < n4 / 4; i += nthr) d1[i] = s1[i];
    for (int i = n4 + tid; i < n; i += nthr) d[i] = s[i];
  }
}

// One warp computes one output row for TILE_M tokens; W row + X rows staged
// in shared memory. y[tok * n_rows + row] (same layout as the batched kernel).
__global__ void gemv_q4_k_q8_f16out_mtok_kernel(
    const q4k::BlockQ4K* __restrict__ w, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ y, int n_rows, int k, int n_tokens,
    int tok_per) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = k / 256;
  const int w_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ4K));
  const int x_row_bytes = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  // layout: [8 warps × W row] then [tile × X row]
  q4k::BlockQ4K* w_sh =
      reinterpret_cast<q4k::BlockQ4K*>(smem + warp * w_row_bytes);
  q4k::BlockQ8K* x_sh = reinterpret_cast<q4k::BlockQ8K*>(
      smem + 8 * w_row_bytes);

  const int row = blockIdx.x * 8 + warp;
  const int tok0 = blockIdx.y * tok_per;
  const int nt = min(tok_per, n_tokens - tok0);
  if (nt <= 0) return;

  for (int t = 0; t < nt; ++t) {
    stage_bytes(x_sh + static_cast<size_t>(t) * nb, xq + static_cast<size_t>(tok0 + t) * nb,
                x_row_bytes, threadIdx.x, blockDim.x);
  }
  if (row < n_rows) {
    stage_bytes(w_sh, w + static_cast<size_t>(row) * nb, w_row_bytes,
                lane, 32);
  }
  __syncthreads();

  if (row >= n_rows) return;
  const int nsg = nb * 8;
  for (int t = 0; t < nt; ++t) {
    const q4k::BlockQ8K* xb_row = x_sh + static_cast<size_t>(t) * nb;
    float acc = 0.f;
    for (int g = lane; g < nsg; g += 32) {
      const int blk = g >> 3, sb = g & 7;
      const q4k::BlockQ4K& b = w_sh[blk];
      const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const __half*>(&b.dmin));
      float sd, sm;
      unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
      const int dot = dp4a_subblock(b.qs + sb * 16, xb_row[blk].qs + sb * 32);
      acc += xb_row[blk].d * (sd * static_cast<float>(dot) -
                      sm * static_cast<float>(xb_row[blk].bsums[sb]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &y[static_cast<size_t>(tok0 + t) * n_rows + row]) =
          __float2half(acc);
    }
  }
}

// Fused gate+up: two weight matrices share the staged X rows (one launch
// instead of two, x quant slabs read once).
__global__ void gemv2_q4_k_q8_f16out_mtok_kernel(
    const q4k::BlockQ4K* __restrict__ wg, const q4k::BlockQ4K* __restrict__ wu,
    const q4k::BlockQ8K* __restrict__ xq, uint16_t* __restrict__ yg,
    uint16_t* __restrict__ yu, int n_rows, int k, int n_tokens,
    int tok_per) {
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
  const int tok0 = blockIdx.y * tok_per;
  const int nt = min(tok_per, n_tokens - tok0);
  if (nt <= 0) return;

  for (int t = 0; t < nt; ++t) {
    stage_bytes(x_sh + static_cast<size_t>(t) * nb, xq + static_cast<size_t>(tok0 + t) * nb,
                x_row_bytes, threadIdx.x, blockDim.x);
  }
  if (row < n_rows) {
    stage_bytes(wg_sh, wg + static_cast<size_t>(row) * nb, w_row_bytes,
                lane, 32);
    stage_bytes(wu_sh, wu + static_cast<size_t>(row) * nb, w_row_bytes,
                lane, 32);
  }
  __syncthreads();

  if (row >= n_rows) return;
  const int nsg = nb * 8;
  for (int t = 0; t < nt; ++t) {
    const q4k::BlockQ8K* xb_row = x_sh + static_cast<size_t>(t) * nb;
    float accg = 0.f, accu = 0.f;
    for (int g = lane; g < nsg; g += 32) {
      const int blk = g >> 3, sb = g & 7;
      const float d = __half2float(*reinterpret_cast<const __half*>(&wg_sh[blk].d));
      const float dmin =
          __half2float(*reinterpret_cast<const __half*>(&wg_sh[blk].dmin));
      float sd, sm;
      unpack_scales6(wg_sh[blk].scales, sb, d, dmin, &sd, &sm);
      const int dotg =
          dp4a_subblock(wg_sh[blk].qs + sb * 16, xb_row[blk].qs + sb * 32);
      accg += xb_row[blk].d * (sd * static_cast<float>(dotg) -
                          sm * static_cast<float>(xb_row[blk].bsums[sb]));
      const float d2 = __half2float(*reinterpret_cast<const __half*>(&wu_sh[blk].d));
      const float dmin2 =
          __half2float(*reinterpret_cast<const __half*>(&wu_sh[blk].dmin));
      unpack_scales6(wu_sh[blk].scales, sb, d2, dmin2, &sd, &sm);
      const int dotu =
          dp4a_subblock(wu_sh[blk].qs + sb * 16, xb_row[blk].qs + sb * 32);
      accu += xb_row[blk].d * (sd * static_cast<float>(dotu) -
                          sm * static_cast<float>(xb_row[blk].bsums[sb]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      accg += __shfl_down_sync(0xFFFFFFFFu, accg, off);
      accu += __shfl_down_sync(0xFFFFFFFFu, accu, off);
    }
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &yg[static_cast<size_t>(tok0 + t) * n_rows + row]) =
          __float2half(accg);
      *reinterpret_cast<__half*>(
          &yu[static_cast<size_t>(tok0 + t) * n_rows + row]) =
          __float2half(accu);
    }
  }
}

// Grouped fused gate+up over ALL routed experts of a layer in ONE launch.
// Job j: weights wg[j]/wu[j], activations at xq + off[j] rows, m[j] tokens.
// Tiles are flattened: tile_base[j] = prefix of ceil(m[j]/TILE_M);
// job_of_tile[tile] maps the flattened tile back to its job.
__global__ void gemv_grouped2_q4_k_q8_f16out_mtok_kernel(
    const q4k::BlockQ4K* const* __restrict__ wg,
    const q4k::BlockQ4K* const* __restrict__ wu, const int* __restrict__ off,
    const int* __restrict__ m, const int* __restrict__ tile_base,
    const int* __restrict__ job_of_tile, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ yg, uint16_t* __restrict__ yu, int n_rows, int k,
    int tok_per) {
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
  const int job = job_of_tile[blockIdx.y];
  const int joff = off[job];
  const int jm = m[job];
  const int tok0 = (blockIdx.y - tile_base[job]) * tok_per;
  const int nt = min(tok_per, jm - tok0);
  if (nt <= 0 || row >= n_rows) return;

  for (int t = 0; t < nt; ++t) {
    stage_bytes(x_sh + static_cast<size_t>(t) * nb, xq + static_cast<size_t>(joff + tok0 + t) * nb,
                x_row_bytes, threadIdx.x, blockDim.x);
  }
  stage_bytes(wg_sh, wg[job] + static_cast<size_t>(row) * nb, w_row_bytes,
              lane, 32);
  stage_bytes(wu_sh, wu[job] + static_cast<size_t>(row) * nb, w_row_bytes,
              lane, 32);
  __syncthreads();

  const int nsg = nb * 8;
  for (int t = 0; t < nt; ++t) {
    const q4k::BlockQ8K* xb_row = x_sh + static_cast<size_t>(t) * nb;
    float accg = 0.f, accu = 0.f;
    for (int g = lane; g < nsg; g += 32) {
      const int blk = g >> 3, sb = g & 7;
      const float d = __half2float(*reinterpret_cast<const __half*>(&wg_sh[blk].d));
      const float dmin =
          __half2float(*reinterpret_cast<const __half*>(&wg_sh[blk].dmin));
      float sd, sm;
      unpack_scales6(wg_sh[blk].scales, sb, d, dmin, &sd, &sm);
      const int dotg =
          dp4a_subblock(wg_sh[blk].qs + sb * 16, xb_row[blk].qs + sb * 32);
      accg += xb_row[blk].d * (sd * static_cast<float>(dotg) -
                       sm * static_cast<float>(xb_row[blk].bsums[sb]));
      const float d2 = __half2float(*reinterpret_cast<const __half*>(&wu_sh[blk].d));
      const float dmin2 =
          __half2float(*reinterpret_cast<const __half*>(&wu_sh[blk].dmin));
      unpack_scales6(wu_sh[blk].scales, sb, d2, dmin2, &sd, &sm);
      const int dotu =
          dp4a_subblock(wu_sh[blk].qs + sb * 16, xb_row[blk].qs + sb * 32);
      accu += xb_row[blk].d * (sd * static_cast<float>(dotu) -
                       sm * static_cast<float>(xb_row[blk].bsums[sb]));
    }
    #pragma unroll
    for (int offr = 16; offr > 0; offr >>= 1) {
      accg += __shfl_down_sync(0xFFFFFFFFu, accg, offr);
      accu += __shfl_down_sync(0xFFFFFFFFu, accu, offr);
    }
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &yg[static_cast<size_t>(joff + tok0 + t) * n_rows + row]) =
          __float2half(accg);
      *reinterpret_cast<__half*>(
          &yu[static_cast<size_t>(joff + tok0 + t) * n_rows + row]) =
          __float2half(accu);
    }
  }
}

// Grouped single-matrix variant (down projection: n_rows = hidden rows per
// expert, k = n_ff_exp; jobs tile the same way).
__global__ void gemv_grouped_q4_k_q8_f16out_mtok_kernel(
    const q4k::BlockQ4K* const* __restrict__ w, const int* __restrict__ off,
    const int* __restrict__ m, const int* __restrict__ tile_base,
    const int* __restrict__ job_of_tile, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ y, int n_rows, int k, int tok_per) {
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
  const int job = job_of_tile[blockIdx.y];
  const int joff = off[job];
  const int jm = m[job];
  const int tok0 = (blockIdx.y - tile_base[job]) * tok_per;
  const int nt = min(tok_per, jm - tok0);
  if (nt <= 0 || row >= n_rows) return;

  for (int t = 0; t < nt; ++t) {
    stage_bytes(x_sh + static_cast<size_t>(t) * nb, xq + static_cast<size_t>(joff + tok0 + t) * nb,
                x_row_bytes, threadIdx.x, blockDim.x);
  }
  stage_bytes(w_sh, w[job] + static_cast<size_t>(row) * nb, w_row_bytes,
              lane, 32);
  __syncthreads();

  const int nsg = nb * 8;
  for (int t = 0; t < nt; ++t) {
    const q4k::BlockQ8K* xb_row = x_sh + static_cast<size_t>(t) * nb;
    float acc = 0.f;
    for (int g = lane; g < nsg; g += 32) {
      const int blk = g >> 3, sb = g & 7;
      const q4k::BlockQ4K& b = w_sh[blk];
      const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const __half*>(&b.dmin));
      float sd, sm;
      unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
      const int dot = dp4a_subblock(b.qs + sb * 16, xb_row[blk].qs + sb * 32);
      acc += xb_row[blk].d * (sd * static_cast<float>(dot) -
                      sm * static_cast<float>(xb_row[blk].bsums[sb]));
    }
    #pragma unroll
    for (int offr = 16; offr > 0; offr >>= 1) {
      acc += __shfl_down_sync(0xFFFFFFFFu, acc, offr);
    }
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &y[static_cast<size_t>(joff + tok0 + t) * n_rows + row]) =
          __float2half(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// v0.4 device-dispatch grouped kernels (P1): the grouped plan (which expert a
// flattened tile serves) is derived ON DEVICE from off[]/toff[] written by
// moe_dispatch_scan; per-expert weights come from the chunk pointer table
// (null = expert absent -> rows are zeroed so the combine adds 0, and the
// CPU-pool patch fills them after the layer).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int mtok_expert_of_tile(const int* toff,
                                                   int n_expert, int tile) {
  // last e with toff[e] <= tile (binary search over the E+1 prefix)
  int lo = 0, hi = n_expert - 1, res = n_expert - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    if (toff[mid + 1] <= tile) lo = mid + 1;
    else { res = mid; hi = mid - 1; }
  }
  return res;
}

__global__ void gemv_grouped2_tab_q4_k_q8_f16out_mtok_kernel(
    const q4k::BlockQ4K* const* __restrict__ wg_tab,
    const q4k::BlockQ4K* const* __restrict__ wu_tab, const int* __restrict__ off,
    const int* __restrict__ toff, const int* __restrict__ meta,
    int* __restrict__ mail, int* __restrict__ mail_n,
    const q4k::BlockQ8K* __restrict__ xq, uint16_t* __restrict__ yg,
    uint16_t* __restrict__ yu, int n_rows, int k, int n_expert, int tok_per) {
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

  if (blockIdx.y >= *meta) return;          // meta[1] = total tiles
  const int e = mtok_expert_of_tile(toff, n_expert, blockIdx.y);
  const int joff = off[e];
  const int jm = off[e + 1] - off[e];
  const int tok0 = (blockIdx.y - toff[e]) * tok_per;
  const int nt = min(tok_per, jm - tok0);
  const int row = blockIdx.x * 8 + warp;
  if (nt <= 0) return;

  const q4k::BlockQ4K* wg = wg_tab[e];
  const q4k::BlockQ4K* wu = wu_tab[e];
  if (!wg || !wu) {
    // absent expert: zero this tile's outputs (combine adds 0), mailbox once
    if (row < n_rows) {
      for (int t = 0; t < nt; ++t) {
        yg[static_cast<size_t>(joff + tok0 + t) * n_rows + row] = 0;
        yu[static_cast<size_t>(joff + tok0 + t) * n_rows + row] = 0;
      }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      const int mi = atomicAdd(mail_n, 1);
      if (mi < n_expert) mail[mi] = e;
    }
    return;
  }
  if (row >= n_rows) return;

  for (int t = 0; t < nt; ++t) {
    stage_bytes(x_sh + static_cast<size_t>(t) * nb, xq + static_cast<size_t>(joff + tok0 + t) * nb,
                x_row_bytes, threadIdx.x, blockDim.x);
  }
  stage_bytes(wg_sh, wg + static_cast<size_t>(row) * nb, w_row_bytes,
              lane, 32);
  stage_bytes(wu_sh, wu + static_cast<size_t>(row) * nb, w_row_bytes,
              lane, 32);
  __syncthreads();

  const int nsg = nb * 8;
  for (int t = 0; t < nt; ++t) {
    const q4k::BlockQ8K* xb_row = x_sh + static_cast<size_t>(t) * nb;
    float accg = 0.f, accu = 0.f;
    for (int g = lane; g < nsg; g += 32) {
      const int blk = g >> 3, sb = g & 7;
      const float d = __half2float(*reinterpret_cast<const __half*>(&wg_sh[blk].d));
      const float dmin =
          __half2float(*reinterpret_cast<const __half*>(&wg_sh[blk].dmin));
      float sd, sm;
      unpack_scales6(wg_sh[blk].scales, sb, d, dmin, &sd, &sm);
      const int dotg =
          dp4a_subblock(wg_sh[blk].qs + sb * 16, xb_row[blk].qs + sb * 32);
      accg += xb_row[blk].d * (sd * static_cast<float>(dotg) -
                       sm * static_cast<float>(xb_row[blk].bsums[sb]));
      const float d2 = __half2float(*reinterpret_cast<const __half*>(&wu_sh[blk].d));
      const float dmin2 =
          __half2float(*reinterpret_cast<const __half*>(&wu_sh[blk].dmin));
      unpack_scales6(wu_sh[blk].scales, sb, d2, dmin2, &sd, &sm);
      const int dotu =
          dp4a_subblock(wu_sh[blk].qs + sb * 16, xb_row[blk].qs + sb * 32);
      accu += xb_row[blk].d * (sd * static_cast<float>(dotu) -
                       sm * static_cast<float>(xb_row[blk].bsums[sb]));
    }
    #pragma unroll
    for (int offr = 16; offr > 0; offr >>= 1) {
      accg += __shfl_down_sync(0xFFFFFFFFu, accg, offr);
      accu += __shfl_down_sync(0xFFFFFFFFu, accu, offr);
    }
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &yg[static_cast<size_t>(joff + tok0 + t) * n_rows + row]) =
          __float2half(accg);
      *reinterpret_cast<__half*>(
          &yu[static_cast<size_t>(joff + tok0 + t) * n_rows + row]) =
          __float2half(accu);
    }
  }
}

__global__ void gemv_grouped_tab_q4_k_q8_f16out_mtok_kernel(
    const q4k::BlockQ4K* const* __restrict__ w_tab,
    const int* __restrict__ off, const int* __restrict__ toff,
    const int* __restrict__ meta, const q4k::BlockQ8K* __restrict__ xq,
    uint16_t* __restrict__ y, int n_rows, int k, int n_expert, int tok_per) {
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

  if (blockIdx.y >= *meta) return;
  const int e = mtok_expert_of_tile(toff, n_expert, blockIdx.y);
  const int joff = off[e];
  const int jm = off[e + 1] - off[e];
  const int tok0 = (blockIdx.y - toff[e]) * tok_per;
  const int nt = min(tok_per, jm - tok0);
  const int row = blockIdx.x * 8 + warp;
  if (nt <= 0) return;

  const q4k::BlockQ4K* w = w_tab[e];
  if (!w) {
    if (row < n_rows) {
      for (int t = 0; t < nt; ++t) {
        y[static_cast<size_t>(joff + tok0 + t) * n_rows + row] = 0;
      }
    }
    return;
  }
  if (row >= n_rows) return;

  for (int t = 0; t < nt; ++t) {
    stage_bytes(x_sh + static_cast<size_t>(t) * nb, xq + static_cast<size_t>(joff + tok0 + t) * nb,
                x_row_bytes, threadIdx.x, blockDim.x);
  }
  stage_bytes(w_sh, w + static_cast<size_t>(row) * nb, w_row_bytes,
              lane, 32);
  __syncthreads();

  const int nsg = nb * 8;
  for (int t = 0; t < nt; ++t) {
    const q4k::BlockQ8K* xb_row = x_sh + static_cast<size_t>(t) * nb;
    float acc = 0.f;
    for (int g = lane; g < nsg; g += 32) {
      const int blk = g >> 3, sb = g & 7;
      const q4k::BlockQ4K& b = w_sh[blk];
      const float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
      const float dmin =
          __half2float(*reinterpret_cast<const __half*>(&b.dmin));
      float sd, sm;
      unpack_scales6(b.scales, sb, d, dmin, &sd, &sm);
      const int dot = dp4a_subblock(b.qs + sb * 16, xb_row[blk].qs + sb * 32);
      acc += xb_row[blk].d * (sd * static_cast<float>(dot) -
                      sm * static_cast<float>(xb_row[blk].bsums[sb]));
    }
    #pragma unroll
    for (int offr = 16; offr > 0; offr >>= 1) {
      acc += __shfl_down_sync(0xFFFFFFFFu, acc, offr);
    }
    if (lane == 0) {
      *reinterpret_cast<__half*>(
          &y[static_cast<size_t>(joff + tok0 + t) * n_rows + row]) =
          __float2half(acc);
    }
  }
}

// Router logits, tiled multi-token: W fp32 [n_rows x k] @ X fp16 -> fp32.
// k is walked in 256-wide chunks (W chunk + X chunk staged in shared); the
// per-lane global index sequence stays lane, lane+32, ... ascending, so the
// fp32 accumulation order per (row, token) is IDENTICAL to the single-token
// and v0.2.3 batched router kernels - logits stay bit-identical.
__global__ void gemv_f32_out_f16x_mtok_kernel(const float* __restrict__ w,
                                              const uint16_t* __restrict__ x,
                                              float* __restrict__ y,
                                              int n_rows, int k,
                                              int n_tokens) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int KC = 256;                       // k values per chunk
  float* w_sh = reinterpret_cast<float*>(smem + warp * KC * sizeof(float));
  uint16_t* x_sh = reinterpret_cast<uint16_t*>(
      smem + 8 * KC * sizeof(float));

  const int row = blockIdx.x * 8 + warp;
  const int tok0 = blockIdx.y * kMtokTile;
  const int nt = min(kMtokTile, n_tokens - tok0);
  if (nt <= 0) return;

  float acc[kMtokTile];
  for (int t = 0; t < nt; ++t) acc[t] = 0.f;

  const float* roww = (row < n_rows) ? w + static_cast<size_t>(row) * k
                                     : w;
  for (int c = 0; c < k; c += KC) {
    // stage X chunk (all threads): nt x KC fp16
    for (int i = threadIdx.x; i < nt * KC; i += blockDim.x) {
      const int t = i / KC, il = i % KC;
      x_sh[i] = x[static_cast<size_t>(tok0 + t) * k + c + il];
    }
    // stage this warp's W chunk (warp-local, 8 floats per lane)
    if (row < n_rows) {
      for (int il = lane; il < KC; il += 32) {
        w_sh[il] = roww[c + il];
      }
    }
    __syncthreads();
    if (row < n_rows) {
      for (int il = lane; il < KC; il += 32) {
        const float wv = w_sh[il];
        for (int t = 0; t < nt; ++t) {
          acc[t] += wv * __half2float(
              *reinterpret_cast<const __half*>(&x_sh[t * KC + il]));
        }
      }
    }
    __syncthreads();
  }
  if (row >= n_rows) return;
  for (int t = 0; t < nt; ++t) {
    float a = acc[t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      a += __shfl_down_sync(0xFFFFFFFFu, a, off);
    }
    if (lane == 0) y[static_cast<size_t>(tok0 + t) * n_rows + row] = a;
  }
}

}  // namespace

__global__ void gemv_f16_f16out_kernel(const uint16_t* __restrict__ w,
                                       const uint16_t* __restrict__ x,
                                       uint16_t* __restrict__ y, int n_rows,
                                       int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n_rows) return;
  const uint16_t* roww = w + static_cast<size_t>(row) * k;
  float acc = 0.f;
  for (int i = lane; i < k; i += 32) {
    acc += __half2float(*reinterpret_cast<const __half*>(&roww[i])) *
           __half2float(*reinterpret_cast<const __half*>(&x[i]));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) *reinterpret_cast<__half*>(&y[row]) = __float2half(acc);
}

}  // namespace

void gemv_q4_k_q8(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq, float* y,
                  int n_rows, int k, cudaStream_t stream) {
  gemv_q4_k_q8_kernel<<<(n_rows + 7) / 8, 256, 0, stream>>>(w, xq, y, n_rows, k);
}

void gemv_q4_k_q8_f16out(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                         uint16_t* y, int n_rows, int k, cudaStream_t stream) {
  gemv_q4_k_q8_f16_kernel<<<(n_rows + 7) / 8, 256, 0, stream>>>(w, xq, y, n_rows, k);
}

void gemv_ptrs_q4_k_q8_f16out(const q4k::BlockQ4K* const* ptrs,
                              const q4k::BlockQ8K* xq, uint16_t* y,
                              int n_ptrs, int rows_per, int k,
                              cudaStream_t stream) {
  const int rows = n_ptrs * rows_per;
  gemv_ptrs_q4_k_q8_f16_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
      ptrs, xq, y, rows_per, k);
}

void gemv_q4_k_q8_f16out_batched(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                                 uint16_t* y, int n_rows, int k, int n_tokens,
                                 cudaStream_t stream) {
  dim3 grid((n_rows + 7) / 8, n_tokens);
  gemv_q4_k_q8_f16_batched_kernel<<<grid, 256, 0, stream>>>(w, xq, y, n_rows,
                                                            k, n_tokens);
}

void gemv_q4_k_q8_batched(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                          float* y, int n_rows, int k, int n_tokens,
                          cudaStream_t stream) {
  dim3 grid((n_rows + 7) / 8, n_tokens);
  gemv_q4_k_q8_batched_kernel<<<grid, 256, 0, stream>>>(w, xq, y, n_rows, k,
                                                        n_tokens);
}

void gemv_f32_out_f16x_batched(const float* w_f32, const uint16_t* x_f16,
                               float* y, int n_rows, int k, int n_tokens,
                               cudaStream_t stream) {
  dim3 grid((n_rows + 7) / 8, n_tokens);
  gemv_f32_out_batched_kernel<<<grid, 256, 0, stream>>>(w_f32, x_f16, y,
                                                        n_rows, k, n_tokens);
}

void gemv_f16_f16out(const uint16_t* w, const uint16_t* x, uint16_t* y,
                     int n_rows, int k, cudaStream_t stream) {
  gemv_f16_f16out_kernel<<<(n_rows + 7) / 8, 256, 0, stream>>>(w, x, y, n_rows, k);
}

// ---- v0.3 mtok launchers ---------------------------------------------------
namespace {

// tokens per block from the 48 KB dynamic-shared budget (sm_75 default cap;
// keeps every variant opt-in free). w_rows_staged: 8 single / 16 fused.
inline int mtok_tile(int nb, int w_rows_staged) {
  const int w8 = w_rows_staged * nb * static_cast<int>(sizeof(q4k::BlockQ4K));
  const int xrow = nb * static_cast<int>(sizeof(q4k::BlockQ8K));
  int t = (49152 - w8 - 512) / (xrow > 0 ? xrow : 1);
  return t < 4 ? 4 : (t > 16 ? 16 : t);
}

inline size_t mtok_smem(int nb, int w_rows_staged, int tile) {
  return static_cast<size_t>(w_rows_staged) * nb * sizeof(q4k::BlockQ4K) +
         static_cast<size_t>(tile) * nb * sizeof(q4k::BlockQ8K);
}

}  // namespace

void gemv_q4_k_q8_f16out_mtok(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                              uint16_t* y, int n_rows, int k, int n_tokens,
                              cudaStream_t stream) {
  if (n_tokens <= 0) return;
  const int nb = k / 256;
  const int tile = mtok_tile(nb, 8);
  dim3 grid((n_rows + 7) / 8, (n_tokens + tile - 1) / tile);
  gemv_q4_k_q8_f16out_mtok_kernel<<<grid, 256, mtok_smem(nb, 8, tile), stream>>>(
      w, xq, y, n_rows, k, n_tokens, tile);
}

void gemv2_q4_k_q8_f16out_mtok(const q4k::BlockQ4K* wg, const q4k::BlockQ4K* wu,
                               const q4k::BlockQ8K* xq, uint16_t* yg,
                               uint16_t* yu, int n_rows, int k, int n_tokens,
                               cudaStream_t stream) {
  if (n_tokens <= 0) return;
  const int nb = k / 256;
  const int tile = mtok_tile(nb, 16);
  dim3 grid((n_rows + 7) / 8, (n_tokens + tile - 1) / tile);
  gemv2_q4_k_q8_f16out_mtok_kernel<<<grid, 256, mtok_smem(nb, 16, tile), stream>>>(
      wg, wu, xq, yg, yu, n_rows, k, n_tokens, tile);
}

void gemv_grouped2_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* wg, const q4k::BlockQ4K* const* wu,
    const int* off, const int* m, const int* tile_base,
    const int* job_of_tile, int n_tiles, const q4k::BlockQ8K* xq,
    uint16_t* yg, uint16_t* yu, int n_rows, int k, int tok_per,
    cudaStream_t stream) {
  if (n_tiles <= 0) return;
  const int nb = k / 256;
  dim3 grid((n_rows + 7) / 8, n_tiles);
  gemv_grouped2_q4_k_q8_f16out_mtok_kernel<<<grid, 256, mtok_smem(nb, 16, tok_per), stream>>>(
      wg, wu, off, m, tile_base, job_of_tile, xq, yg, yu, n_rows, k, tok_per);
}

void gemv_grouped_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* w, const int* off, const int* m,
    const int* tile_base, const int* job_of_tile, int n_tiles,
    const q4k::BlockQ8K* xq, uint16_t* y, int n_rows, int k, int tok_per,
    cudaStream_t stream) {
  if (n_tiles <= 0) return;
  const int nb = k / 256;
  dim3 grid((n_rows + 7) / 8, n_tiles);
  gemv_grouped_q4_k_q8_f16out_mtok_kernel<<<grid, 256, mtok_smem(nb, 8, tok_per), stream>>>(
      w, off, m, tile_base, job_of_tile, xq, y, n_rows, k, tok_per);
}

void gemv_f32_out_f16x_mtok(const float* w, const uint16_t* x, float* y,
                            int n_rows, int k, int n_tokens,
                            cudaStream_t stream) {
  if (n_tokens <= 0) return;
  dim3 grid((n_rows + 7) / 8, (n_tokens + kMtokTile - 1) / kMtokTile);
  const size_t smem = 8ull * 256 * sizeof(float) +
                      static_cast<size_t>(kMtokTile) * 256 * sizeof(uint16_t);
  gemv_f32_out_f16x_mtok_kernel<<<grid, 256, smem, stream>>>(w, x, y, n_rows,
                                                             k, n_tokens);
}

void quantize_q8_k(const uint16_t* x_f16, q4k::BlockQ8K* out, int n,
                   cudaStream_t stream) {
  quantize_q8_k_kernel<<<n / 256, 256, 0, stream>>>(x_f16, out, n);
}

// device-row-count variant: total = (*d_rows) * row_mult (elements); the
// grid is bounded by the host with the worst case (bit-identical math).
__global__ void quantize_q8_k_n_kernel(const uint16_t* x, q4k::BlockQ8K* out,
                                       const int* d_rows, int row_mult) {
  __shared__ float red[32];
  const int b = blockIdx.x;
  const long long n = static_cast<long long>(*d_rows) * row_mult;
  if (static_cast<long long>(b) * 256 >= n) return;
  const int tid = threadIdx.x;
  const int idx = b * 256 + tid;
  const float v = (idx < n)
      ? __half2float(*reinterpret_cast<const __half*>(&x[idx])) : 0.f;
  float a = fabsf(v);

  const int lane = tid & 31, warp = tid >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, off));
  if (lane == 0) red[warp] = a;
  __syncthreads();
  if (tid == 0) {
    float m = 1e-30f;
    const int nw = (blockDim.x + 31) / 32;
    for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
    red[0] = m;
  }
  __syncthreads();
  const float amax = red[0];
  const float d = amax / 127.f;
  const float id = d > 0.f ? 1.f / d : 0.f;
  const int q = static_cast<int>(rintf(v * id));
  if (idx < n) {
    out[b].qs[tid] = static_cast<int8_t>(max(-127, min(127, q)));
    out[b].d = d;
  }
  __syncthreads();
  if (tid < 8) {
    int sum = 0;
    for (int i = 0; i < 32; ++i) sum += out[b].qs[tid * 32 + i];
    out[b].bsums[tid] = static_cast<int16_t>(sum);
  }
}

void quantize_q8_k_n(const uint16_t* x_f16, q4k::BlockQ8K* out,
                     const int* d_rows, int row_mult, int n_max,
                     cudaStream_t stream) {
  quantize_q8_k_n_kernel<<<n_max / 256, 256, 0, stream>>>(x_f16, out, d_rows,
                                                          row_mult);
}

void gemv_grouped2_tab_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* wg_tab, const q4k::BlockQ4K* const* wu_tab,
    const int* off, const int* toff, const int* meta, int* mail, int* mail_n,
    int tiles_max, const q4k::BlockQ8K* xq, uint16_t* yg, uint16_t* yu,
    int n_rows, int k, int n_expert, int tok_per, cudaStream_t stream) {
  const int nb = k / 256;
  dim3 grid((n_rows + 7) / 8, tiles_max);
  gemv_grouped2_tab_q4_k_q8_f16out_mtok_kernel<<<grid, 256,
      mtok_smem(nb, 16, tok_per), stream>>>(
      wg_tab, wu_tab, off, toff, meta, mail, mail_n, xq, yg, yu, n_rows, k,
      n_expert, tok_per);
}

void gemv_grouped_tab_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* w_tab, const int* off, const int* toff,
    const int* meta, int tiles_max, const q4k::BlockQ8K* xq, uint16_t* y,
    int n_rows, int k, int n_expert, int tok_per, cudaStream_t stream) {
  const int nb = k / 256;
  dim3 grid((n_rows + 7) / 8, tiles_max);
  gemv_grouped_tab_q4_k_q8_f16out_mtok_kernel<<<grid, 256,
      mtok_smem(nb, 8, tok_per), stream>>>(
      w_tab, off, toff, meta, xq, y, n_rows, k, n_expert, tok_per);
}

void gemv_f32_out_f16x(const float* w_f32, const uint16_t* x_f16, float* y,
                       int n_rows, int k, cudaStream_t stream) {
  const int blocks = (n_rows + 7) / 8;
  gemv_f32_out_kernel<<<blocks, 256, 0, stream>>>(w_f32, x_f16, y, n_rows, k);
}

}  // namespace sager::cuda
