// norm_rotary.cu - RMSNorm, residual add, dtype convert, embedding gather,
// RoPE (partial rotary + optional mrope) and SwiGLU for Sage-A1.5.
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>

namespace sager::cuda {
namespace {

// mrope section boundaries live in device constant memory (copied once at
// session prepare - fixed address, graph-safe).
__device__ int d_mrope_section_sh[3];
int* d_mrope_section_ = nullptr;   // device pointer of the constant above

__device__ __forceinline__ float ldh(const uint16_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ void sth(uint16_t* p, float v) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}

__global__ void rmsnorm_kernel(uint16_t* out, const uint16_t* x,
                               const uint16_t* w, int n, float eps) {
  __shared__ float red[32];
  const int tid = threadIdx.x;
  const int stride = blockDim.x;              // capped at 1024 by the launcher
  float acc = 0.f;
  for (int i = tid; i < n; i += stride) {
    const float v = ldh(x + i);
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
    sth(out + i, ldh(x + i) * red[0] * ldh(w + i));
  }
}

__global__ void add_kernel(uint16_t* in_out, const uint16_t* residual, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sth(in_out + i, ldh(in_out + i) + ldh(residual + i));
}

__global__ void f32_to_f16_kernel(const float* src, uint16_t* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sth(dst + i, src[i]);
}

__global__ void gather_embed_kernel(uint16_t* out, const uint16_t* table,
                                    const int* ids, int hidden) {
  const int tok = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden) {
    const int id = ids[tok];
    out[static_cast<size_t>(tok) * hidden + i] = table[static_cast<size_t>(id) * hidden + i];
  }
}

// RoPE. NeoX convention (rotate_half): for dim j < rope_dims/2 pair (j, j+D/2).
// Interleaved (GPT-J) pairs (2j, 2j+1). mrope assigns each rotary dim to one
// of 3 position streams via mrope_section (t/h/w from the model card).
__global__ void rope_kernel(uint16_t* q, uint16_t* k, const int* pos_ids,
                            int n_head, int n_head_kv, int head_dim,
                            int rope_dims, float theta, int style,
                            const int* mrope_section, int n_tokens) {
  const int half = rope_dims / 2;
  const int total_q = n_tokens * n_head * head_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_k = n_tokens * n_head_kv * head_dim;

  auto rope_one = [&](uint16_t* base, int total, int head_id_off, int n_h) {
    if (idx >= total) return;
    const int tok = idx / (n_h * head_dim);
    const int rem = idx % (n_h * head_dim);
    const int hid = rem / head_dim;
    const int dim = rem % head_dim;
    if (dim >= rope_dims) return;

    float posf;
    if (style == 2) {
      // mrope: pick stream by section boundaries
      int acc = 0;
      int stream = 0;
      for (int s = 0; s < 3; ++s) {
        const int sec = mrope_section[s];
        if (dim % half < acc + sec) { stream = s; break; }
        acc += sec;
      }
      posf = static_cast<float>(pos_ids[stream * n_tokens + tok]);
    } else {
      posf = static_cast<float>(pos_ids[tok]);
    }
    const float inv_freq = powf(theta, -2.f * static_cast<float>((dim % half)) /
                                       static_cast<float>(rope_dims));
    const float ang = posf * inv_freq;
    const float sinv = sinf(ang), cosv = cosf(ang);

    const size_t base_off = static_cast<size_t>(tok) * n_h * head_dim +
                            static_cast<size_t>(hid) * head_dim;
    if (style == 1) {
      // interleaved: pair (2j, 2j+1)
      if ((dim & 1) != 0) return;
      const int j = dim;  // even dim is the first of the pair
      const float a = ldh(base + base_off + j);
      const float b = ldh(base + base_off + j + 1);
      sth(base + base_off + j,     a * cosv - b * sinv);
      sth(base + base_off + j + 1, a * sinv + b * cosv);
    } else {
      // NeoX: pair (j, j + half) within the first rope_dims dims
      if (dim >= half) return;
      const float a = ldh(base + base_off + dim);
      const float b = ldh(base + base_off + dim + half);
      sth(base + base_off + dim,         a * cosv - b * sinv);
      sth(base + base_off + dim + half,  a * sinv + b * cosv);
    }
    (void)head_id_off;
  };

  rope_one(q, total_q, 0, n_head);
  rope_one(k, total_k, 0, n_head_kv);
}

__global__ void silu_mul_kernel(uint16_t* out, const uint16_t* g,
                                const uint16_t* u, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    const float gv = ldh(g + i);
    const float uv = ldh(u + i);
    sth(out + i, gv / (1.f + expf(-gv)) * uv);
  }
}

}  // namespace

void set_mrope_section(const int sec[3]) {
  if (!d_mrope_section_) {
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_mrope_section_), d_mrope_section_sh);
  }
  cudaMemcpy(d_mrope_section_, sec, 3 * sizeof(int), cudaMemcpyHostToDevice);
}

void rmsnorm_f16(uint16_t* out, const uint16_t* x, const uint16_t* w, int n,
                 float eps, cudaStream_t stream) {
  int b = ((n + 31) / 32) * 32;
  if (b > 1024) b = 1024;                     // CUDA block-dim limit
  rmsnorm_kernel<<<1, b, 0, stream>>>(out, x, w, n, eps);
}

// Batched RMSNorm: one block per row (prefill / verify window).
__global__ void rmsnorm_batched_kernel(uint16_t* out, const uint16_t* x,
                                       const uint16_t* w, int n, float eps) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;              // capped at 1024 by the launcher
  const uint16_t* xr = x + static_cast<size_t>(row) * n;
  uint16_t* orow = out + static_cast<size_t>(row) * n;
  float acc = 0.f;
  for (int i = tid; i < n; i += stride) {
    const float v = ldh(xr + i);
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
    sth(orow + i, ldh(xr + i) * red[0] * ldh(w + i));
  }
}

void rmsnorm_batched_f16(uint16_t* out, const uint16_t* x, const uint16_t* w,
                         int n, float eps, int n_rows, cudaStream_t stream) {
  int b = ((n + 31) / 32) * 32;
  if (b > 1024) b = 1024;
  rmsnorm_batched_kernel<<<n_rows, b, 0, stream>>>(out, x, w, n, eps);
}

void add_f16(uint16_t* in_out, const uint16_t* residual, int n,
             cudaStream_t stream) {
  add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in_out, residual, n);
}

void convert_f32_to_f16(const float* src, uint16_t* dst, int n,
                        cudaStream_t stream) {
  f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

void gather_embed(uint16_t* out, const uint16_t* table, const int* ids,
                  int n_tokens, int hidden, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((hidden + 255) / 256, n_tokens);
  gather_embed_kernel<<<grid, block, 0, stream>>>(out, table, ids, hidden);
}

void rope_f16(uint16_t* q, uint16_t* k, const int* pos_ids, int n_head,
              int n_head_kv, int head_dim, int rope_dims, float theta,
              int style, int n_tokens, cudaStream_t stream) {
  // mrope_section lives in device constant memory (copied once at prepare).
  const int total = n_tokens * (n_head + n_head_kv) * head_dim;
  rope_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      q, k, pos_ids, n_head, n_head_kv, head_dim, rope_dims, theta, style,
      d_mrope_section_, n_tokens);
}

void silu_mul_f16(uint16_t* out, const uint16_t* g, const uint16_t* u,
                  int rows, int n_ff, cudaStream_t stream) {
  const int total = rows * n_ff;
  silu_mul_kernel<<<(total + 255) / 256, 256, 0, stream>>>(out, g, u, total);
}

}  // namespace sager::cuda
