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
    acc_i = __dp4a(wpack, xpack, acc_i);
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

}  // namespace

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

// Plain fp16 GEMV (MTP head projections): y[n] = W[n x k] @ x[k], both fp16.
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

void quantize_q8_k(const uint16_t* x_f16, q4k::BlockQ8K* out, int n,
                   cudaStream_t stream) {
  quantize_q8_k_kernel<<<n / 256, 256, 0, stream>>>(x_f16, out, n);
}

void gemv_f32_out_f16x(const float* w_f32, const uint16_t* x_f16, float* y,
                       int n_rows, int k, cudaStream_t stream) {
  const int blocks = (n_rows + 7) / 8;
  gemv_f32_out_kernel<<<blocks, 256, 0, stream>>>(w_f32, x_f16, y, n_rows, k);
}

}  // namespace sager::cuda
