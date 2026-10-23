// attention.cu - KV append + streaming GQA attention (decode & prefill).
//
// Memory-bound like everything on T4; the decode kernel streams K/V once and
// keeps a running (max, sum, acc) online-softmax state so nothing is read
// twice. KV can be fp16 or Q8_0-packed (34 B / 32 values, scale folded into
// each block) - q8 KV doubles the context that fits in the T4's 16 GB.
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>

namespace sager::cuda {
namespace {

constexpr int kQ8Group = 32;
constexpr int kQ8RowBytes = 2 + kQ8Group;   // fp16 d + 32 x i8

__device__ __forceinline__ float ldh(const uint16_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ void sth(uint16_t* p, float v) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}

// ---- append -----------------------------------------------------------------
// One thread per element (fp16) / per 32-group (q8). The write position is
// read from the device param block so the launch is graph-replayable.
__global__ void kv_append_fp16_kernel(const uint16_t* k, const uint16_t* v,
                                      uint8_t* kc, uint8_t* vc,
                                      const SeqParams* p, int kv_dim) {
  const int t = blockIdx.y;                        // token in this step
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kv_dim) return;
  const int write_pos = p->pos + t;
  const size_t off = static_cast<size_t>(write_pos) * kv_dim + i;
  *reinterpret_cast<uint16_t*>(kc + off * 2) = k[static_cast<size_t>(t) * kv_dim + i];
  *reinterpret_cast<uint16_t*>(vc + off * 2) = v[static_cast<size_t>(t) * kv_dim + i];
}

__global__ void kv_append_q8_kernel(const uint16_t* k, const uint16_t* v,
                                    uint8_t* kc, uint8_t* vc,
                                    const SeqParams* p, int kv_dim) {
  const int t = blockIdx.y;
  const int nblk = kv_dim / kQ8Group;
  const int blk = blockIdx.x * blockDim.x + threadIdx.x;  // one thread = one 32-group
  if (blk >= nblk) return;
  const int write_pos = p->pos + t;
  const size_t base = static_cast<size_t>(write_pos) * nblk + blk;

  const uint16_t* srcs[2] = {k + static_cast<size_t>(t) * kv_dim + blk * kQ8Group,
                             v + static_cast<size_t>(t) * kv_dim + blk * kQ8Group};
  uint8_t* dsts[2] = {kc + base * kQ8RowBytes, vc + base * kQ8RowBytes};

  #pragma unroll
  for (int which = 0; which < 2; ++which) {
    const uint16_t* src = srcs[which];
    uint8_t* dst = dsts[which];
    float amax = 1e-30f;
    #pragma unroll
    for (int j = 0; j < kQ8Group; ++j) amax = fmaxf(amax, fabsf(ldh(src + j)));
    const float d = amax / 127.f;
    const uint16_t dh = __float2half_rn(d);
    *reinterpret_cast<uint16_t*>(dst) = dh;
    const float id = d > 0.f ? 1.f / d : 0.f;
    #pragma unroll
    for (int j = 0; j < kQ8Group; ++j) {
      const int q = static_cast<int>(rintf(ldh(src + j) * id));
      dst[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(max(-127, min(127, q))));
    }
  }
}

// ---- attention ---------------------------------------------------------------
// One warp per (token, q-head). Streaming online softmax; K/V read exactly
// once. Causal masking for prefill: token i attends positions <= pos + i.
template <bool kIsPrefill>
__global__ void attention_kernel(uint16_t* out, const uint16_t* q,
                                 const uint8_t* kc, const uint8_t* vc,
                                 int kv_dtype, const SeqParams* p,
                                 int n_head, int n_head_kv, int head_dim,
                                 float sm_scale) {
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x % n_head;
  const int tok = blockIdx.x / n_head;             // query token index
  const int kv_head = head / (n_head / n_head_kv);
  const int past = p->pos;
  const int kv_dim = n_head_kv * head_dim;
  const int total_ctx = past + (kIsPrefill ? tok + 1 : 1);
  const int kv_base = kv_head * head_dim;
  const int chunks = head_dim / 32;                // head_dim multiple of 32

  const size_t q_off = static_cast<size_t>(tok) * n_head * head_dim +
                       static_cast<size_t>(head) * head_dim;

  float m = -1e30f, l = 0.f;
  float v_acc[4];
  #pragma unroll
  for (int c = 0; c < 4; ++c) v_acc[c] = 0.f;

  __shared__ float scores_sh[128];
  float* scores = scores_sh;  // 128-token chunk (see loop below)

  for (int t0 = 0; t0 < total_ctx; t0 += 128) {
    const int t1 = min(total_ctx, t0 + 128);
    float chunk_max = -1e30f;
    for (int t = t0; t < t1; ++t) {
      const size_t koff_t = (kv_dtype == 0)
          ? static_cast<size_t>(t) * kv_dim * 2
          : static_cast<size_t>(t) * (kv_dim / kQ8Group) * kQ8RowBytes;
      float s = 0.f;
      for (int d = lane; d < head_dim; d += 32) {
        float kv;
        if (kv_dtype == 0) {
          kv = ldh(reinterpret_cast<const uint16_t*>(kc + koff_t) + kv_base + d);
        } else {
          const uint8_t* row = kc + koff_t;
          const int blk = (kv_base + d) / kQ8Group;
          const int off = (kv_base + d) % kQ8Group;
          const float dk = __half2float(*reinterpret_cast<const __half*>(
              row + static_cast<size_t>(blk) * kQ8RowBytes));
          const int8_t qi = static_cast<int8_t>(
              row[static_cast<size_t>(blk) * kQ8RowBytes + 2 + off]);
          kv = dk * static_cast<float>(qi);
        }
        s += ldh(q + q_off + d) * kv;
      }
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, off);
      s = __shfl_sync(0xFFFFFFFFu, s, 0) * sm_scale;
      if (lane == 0) scores[t - t0] = s;
      chunk_max = fmaxf(chunk_max, s);
    }
    __syncwarp();

    const float m_new = fmaxf(m, chunk_max);
    const float scale_old = expf(m - m_new);
    float l_new = l * scale_old;
    for (int t = t0; t < t1; ++t) l_new += expf(scores[t - t0] - m_new);

    for (int c = 0; c < chunks; ++c) {
      const int d = c * 32 + lane;
      float a = 0.f;
      for (int t = t0; t < t1; ++t) {
        const float w = expf(scores[t - t0] - m_new);
        float vv;
        if (kv_dtype == 0) {
          const size_t voff_t = static_cast<size_t>(t) * kv_dim * 2;
          vv = ldh(reinterpret_cast<const uint16_t*>(vc + voff_t) + kv_base + d);
        } else {
          const uint8_t* row = vc + static_cast<size_t>(t) * (kv_dim / kQ8Group) * kQ8RowBytes;
          const int blk = (kv_base + d) / kQ8Group;
          const int off = (kv_base + d) % kQ8Group;
          const float dv = __half2float(*reinterpret_cast<const __half*>(
              row + static_cast<size_t>(blk) * kQ8RowBytes));
          const int8_t qi = static_cast<int8_t>(
              row[static_cast<size_t>(blk) * kQ8RowBytes + 2 + off]);
          vv = dv * static_cast<float>(qi);
        }
        a += w * vv;
      }
      v_acc[c] = v_acc[c] * scale_old + a;
    }
    m = m_new;
    l = l_new;
  }

  const size_t o_off = static_cast<size_t>(tok) * n_head * head_dim +
                       static_cast<size_t>(head) * head_dim;
  for (int c = 0; c < chunks; ++c) {
    if (lane + c * 32 < head_dim) {
      sth(out + o_off + c * 32 + lane, v_acc[c] / l);
    }
  }
}

}  // namespace

void kv_append(const uint16_t* k, const uint16_t* v, uint8_t* k_cache,
               uint8_t* v_cache, uint16_t*, uint16_t*, int kv_dtype,
               const SeqParams* d_params, int n_tokens_host, int n_head_kv,
               int head_dim, cudaStream_t stream) {
  const int kv_dim = n_head_kv * head_dim;
  if (kv_dtype == 0) {
    dim3 block(256), grid((kv_dim + 255) / 256, n_tokens_host);
    kv_append_fp16_kernel<<<grid, block, 0, stream>>>(k, v, k_cache, v_cache,
                                                      d_params, kv_dim);
  } else {
    dim3 block(64), grid((kv_dim / kQ8Group + 63) / 64, n_tokens_host);
    kv_append_q8_kernel<<<grid, block, 0, stream>>>(k, v, k_cache, v_cache,
                                                    d_params, kv_dim);
  }
}

void attention_decode_f16(uint16_t* out, const uint16_t* q,
                          const uint8_t* k_cache, const uint8_t* v_cache,
                          const uint16_t*, const uint16_t*, int kv_dtype,
                          const SeqParams* d_params, int n_head, int n_head_kv,
                          int head_dim, float sm_scale, cudaStream_t stream) {
  attention_kernel<false><<<dim3(n_head), 32, 0, stream>>>(
      out, q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv,
      head_dim, sm_scale);
}

void attention_prefill_f16(uint16_t* out, const uint16_t* q,
                           const uint8_t* k_cache, const uint8_t* v_cache,
                           const uint16_t*, const uint16_t*, int kv_dtype,
                           const SeqParams* d_params, int n_head, int n_head_kv,
                           int head_dim, float sm_scale, int n_tokens_host,
                           cudaStream_t stream) {
  const dim3 grid(n_head * n_tokens_host);
  attention_kernel<true><<<grid, 32, 0, stream>>>(
      out, q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv,
      head_dim, sm_scale);
}

}  // namespace sager::cuda
