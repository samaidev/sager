// attention.cu - KV append + streaming GQA attention (decode & prefill).
//
// v0.11 flash-tile restructure (SAGER_ATTN_NEW=1, default):
//  - prefill: one block = (kv group, token) with 8 warps = the group's 8 q
//    heads. The block stages 16 K rows + 16 V rows in shared memory,
//    dequantizing Q8 KV exactly once per block instead of once per
//    (token, head) warp, and each lane keeps its head_dim/32 q dims in
//    registers. expf runs once per (row, warp) instead of once per
//    (row, dim-chunk). Templated on CH = head_dim/32 (4 or 8).
//  - decode: split-KV flash decode. The partial kernel always launches a
//    fixed (n_head, kMaxSplits) grid (graph-replay-safe): blocks self-range
//    on device via d_params->pos, warps interleave rows inside their split,
//    and a tiny merge kernel combines the per-split (m, l, acc) partials.
//  - SAGER_ATTN_NEW=0 restores the v0.10 kernels bit-exactly.
//  - The real Sage-A1.5 attention dims are h=16, kvh=2, head_dim=256 (the
//    common.hpp defaults do NOT match the gguf); a one-shot diagnostic line
//    prints the dispatch decision at first use.
//
// Memory-bound like everything on T4; KV can be fp16 or Q8_0-packed
// (34 B / 32 values, scale folded into each block) - q8 KV doubles the
// context that fits in the T4's 16 GB.
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace sager::cuda {
namespace {

constexpr int kQ8Group = 32;
constexpr int kQ8RowBytes = 2 + kQ8Group;   // fp16 d + 32 x i8

// v0.11 tuning constants. The new kernels require a GQA group of exactly 8
// and head_dim/32 <= 8; anything else falls back to the v0.10 kernels.
constexpr int kTileRows = 16;    // staged K/V rows per prefill pass
constexpr int kMaxSplits = 64;   // decode split-KV upper bound (ctx <= 16384)
constexpr int kSplitRows = 256;  // decode rows per split
constexpr int kMaxHeads = 64;    // partial-buffer head stride
constexpr int kMaxDims = 256;    // partial-buffer dim stride (head_dim max)

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

// ---- v0.10 kernels (fallback path, SAGER_ATTN_NEW=0) -------------------------
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
  float v_acc[8];
  #pragma unroll
  for (int c = 0; c < 8; ++c) v_acc[c] = 0.f;

  __shared__ float scores_sh[128];
  float* scores = scores_sh;  // 128-token chunk (see loop below)

  for (int t0 = 0; t0 < total_ctx; t0 += 128) {
    const int t1 = min(total_ctx, t0 + 256);
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

// ---- v0.11 prefill: flash-tile, 8 warps share one staged K/V tile ------------
// Block = (kv group, query token); warp w handles the group's q-head w.
// CH = head_dim / 32 (dims per lane). Host side checks the shape.
template <int CH>
__global__ void __launch_bounds__(256)
attn_prefill_v2(uint16_t* out, const uint16_t* q, const uint8_t* kc,
                const uint8_t* vc, int kv_dtype, const SeqParams* p,
                int n_head, int n_head_kv, float sm_scale) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int kv_head = blockIdx.x;
  const int tok = blockIdx.y;
  const int head = kv_head * 8 + warp;
  const int past = p->pos;
  const int kv_dim = n_head_kv * (CH * 32);
  const int total_ctx = past + tok + 1;            // causal: keys <= own pos

  __shared__ __half k_sh[kTileRows][CH * 32];
  __shared__ __half v_sh[kTileRows][CH * 32];
  __shared__ float s_sh[8][kTileRows];   // raw scores (pre-softmax)
  __shared__ float w_sh[8][kTileRows];   // final weights exp(s - m_new)

  // per-lane q registers: dims {c*32 + lane}
  const size_t q_off = static_cast<size_t>(tok) * n_head * (CH * 32) +
                       static_cast<size_t>(head) * (CH * 32);
  float qr[CH];
  #pragma unroll
  for (int c = 0; c < CH; ++c)
    qr[c] = __half2float(q[q_off + c * 32 + lane]);

  float m = -1e30f, l = 0.f;
  float acc[CH];
  #pragma unroll
  for (int c = 0; c < CH; ++c) acc[c] = 0.f;

  for (int t0 = 0; t0 < total_ctx; t0 += kTileRows) {
    const int nrows = min(kTileRows, total_ctx - t0);
    // stage K then V rows [t0, t0+nrows): dequantize q8 exactly once per block
    for (int e = threadIdx.x; e < nrows * (CH * 32); e += 256) {
      const int r = e / (CH * 32), col = e - r * (CH * 32);
      const int gpos = t0 + r;
      const int gdim = kv_head * (CH * 32) + col;
      if (kv_dtype == 0) {
        k_sh[r][col] = *reinterpret_cast<const __half*>(
            kc + (static_cast<size_t>(gpos) * kv_dim + gdim) * 2);
        v_sh[r][col] = *reinterpret_cast<const __half*>(
            vc + (static_cast<size_t>(gpos) * kv_dim + gdim) * 2);
      } else {
        const uint8_t* krow = kc + static_cast<size_t>(gpos) *
                                   (kv_dim / kQ8Group) * kQ8RowBytes;
        const uint8_t* vrow = vc + static_cast<size_t>(gpos) *
                                   (kv_dim / kQ8Group) * kQ8RowBytes;
        const int blk = gdim >> 5, off = gdim & 31;
        k_sh[r][col] = __hmul(
            *reinterpret_cast<const __half*>(krow + static_cast<size_t>(blk) * kQ8RowBytes),
            __float2half_rn(static_cast<int8_t>(
                krow[static_cast<size_t>(blk) * kQ8RowBytes + 2 + off])));
        v_sh[r][col] = __hmul(
            *reinterpret_cast<const __half*>(vrow + static_cast<size_t>(blk) * kQ8RowBytes),
            __float2half_rn(static_cast<int8_t>(
                vrow[static_cast<size_t>(blk) * kQ8RowBytes + 2 + off])));
      }
    }
    __syncthreads();

    // scores for this tile (once per warp, no per-chunk expf re-evaluation)
    float chunk_max = -1e30f;
    for (int r = 0; r < nrows; ++r) {
      float s = 0.f;
      #pragma unroll
      for (int c = 0; c < CH; ++c)
        s = fmaf(qr[c], __half2float(k_sh[r][c * 32 + lane]), s);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
      s *= sm_scale;
      if (t0 + r > past + tok) s = -1e30f;         // causal mask (belt & braces)
      if (lane == 0) s_sh[warp][r] = s;
      chunk_max = fmaxf(chunk_max, s);
    }
    __syncwarp();

    const float m_new = fmaxf(m, chunk_max);
    const float scale_old = expf(m - m_new);
    float l_new = l * scale_old;
    #pragma unroll
    for (int c = 0; c < CH; ++c) acc[c] *= scale_old;
    for (int r = 0; r < nrows; ++r) {
      const float w = expf(s_sh[warp][r] - m_new);
      if (lane == 0) w_sh[warp][r] = w;
      l_new += w;
    }
    __syncwarp();

    // V accumulation: weights and rows come from shared, broadcast-friendly
    for (int r = 0; r < nrows; ++r) {
      const float w = w_sh[warp][r];
      #pragma unroll
      for (int c = 0; c < CH; ++c)
        acc[c] = fmaf(w, __half2float(v_sh[r][c * 32 + lane]), acc[c]);
    }
    __syncthreads();    // all warps done with this tile before it is restaged
    m = m_new;
    l = l_new;
  }

  #pragma unroll
  for (int c = 0; c < CH; ++c)
    sth(out + q_off + c * 32 + lane, acc[c] / l);
}

// ---- v0.11 decode: split-KV flash decode -------------------------------------
// Partial buffer [split][head] -> (m, l, acc[head_dim]). Fixed strides keep
// the grid constant so graph replay stays valid; stale slots beyond the
// active split count are never read by the merge kernel.
__device__ float g_dec_m[kMaxSplits * kMaxHeads];
__device__ float g_dec_l[kMaxSplits * kMaxHeads];
__device__ float g_dec_acc[kMaxSplits * kMaxHeads * kMaxDims];

template <int CH>
__global__ void __launch_bounds__(128)
attn_decode_partial(const uint16_t* q, const uint8_t* kc, const uint8_t* vc,
                    int kv_dtype, const SeqParams* p, int n_head,
                    int n_head_kv, float sm_scale) {
  const int head = blockIdx.x;
  const int split = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int past = p->pos;
  const int total = past + 1;                      // decode: single query token
  const int start = split * kSplitRows;
  if (start >= total) return;
  const int stop = min(total, start + kSplitRows);

  const int kv_head = head / (n_head / n_head_kv);
  const int kv_dim = n_head_kv * (CH * 32);
  const int kv_base = kv_head * (CH * 32);
  const size_t q_off = static_cast<size_t>(head) * (CH * 32);

  float qr[CH];
  #pragma unroll
  for (int c = 0; c < CH; ++c)
    qr[c] = __half2float(q[q_off + c * 32 + lane]);

  float m = -1e30f, l = 0.f;
  float acc[CH];
  #pragma unroll
  for (int c = 0; c < CH; ++c) acc[c] = 0.f;

  // warps interleave rows inside the split; each warp owns an independent
  // online-softmax state that the block merges at the end
  for (int t = start + warp; t < stop; t += 4) {
    const size_t koff = (kv_dtype == 0)
        ? static_cast<size_t>(t) * kv_dim * 2
        : static_cast<size_t>(t) * (kv_dim / kQ8Group) * kQ8RowBytes;
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < CH; ++c) {
      const int d = kv_base + c * 32 + lane;
      float kv;
      if (kv_dtype == 0) {
        kv = ldh(reinterpret_cast<const uint16_t*>(kc + koff) + d);
      } else {
        const uint8_t* row = kc + koff;
        const int blk = d >> 5, off = d & 31;
        kv = __half2float(*reinterpret_cast<const __half*>(
                  row + static_cast<size_t>(blk) * kQ8RowBytes)) *
             static_cast<float>(static_cast<int8_t>(
                  row[static_cast<size_t>(blk) * kQ8RowBytes + 2 + off]));
      }
      s = fmaf(qr[c], kv, s);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    s *= sm_scale;

    const float m_new = fmaxf(m, s);
    const float scale_old = expf(m - m_new);
    const float w = expf(s - m_new);
    l = l * scale_old + w;
    #pragma unroll
    for (int c = 0; c < CH; ++c) {
      const int d = kv_base + c * 32 + lane;
      float vv;
      if (kv_dtype == 0) {
        vv = ldh(reinterpret_cast<const uint16_t*>(vc + koff) + d);
      } else {
        const uint8_t* row = vc + koff;
        const int blk = d >> 5, off = d & 31;
        vv = __half2float(*reinterpret_cast<const __half*>(
                  row + static_cast<size_t>(blk) * kQ8RowBytes)) *
             static_cast<float>(static_cast<int8_t>(
                  row[static_cast<size_t>(blk) * kQ8RowBytes + 2 + off]));
      }
      acc[c] = acc[c] * scale_old + w * vv;
    }
    m = m_new;
  }

  // merge the 4 warp partials of this block, then publish one split partial
  __shared__ float s_m[4], s_l[4];
  __shared__ float s_acc[4][kMaxDims];
  #pragma unroll
  for (int c = 0; c < CH; ++c) s_acc[warp][c * 32 + lane] = acc[c];
  if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
  __syncthreads();

  if (warp == 0) {
    float mg = -1e30f;
    #pragma unroll
    for (int wI = 0; wI < 4; ++wI) mg = fmaxf(mg, s_m[wI]);
    float lg = 0.f;
    #pragma unroll
    for (int wI = 0; wI < 4; ++wI) lg += s_l[wI] * expf(s_m[wI] - mg);
    float a[CH];
    #pragma unroll
    for (int c = 0; c < CH; ++c) a[c] = 0.f;
    #pragma unroll
    for (int wI = 0; wI < 4; ++wI) {
      const float sc = expf(s_m[wI] - mg);
      #pragma unroll
      for (int c = 0; c < CH; ++c)
        a[c] += s_acc[wI][c * 32 + lane] * sc;
    }
    const size_t idx = static_cast<size_t>(split) * kMaxHeads + head;
    if (lane == 0) { g_dec_m[idx] = mg; g_dec_l[idx] = lg; }
    #pragma unroll
    for (int c = 0; c < CH; ++c)
      g_dec_acc[idx * kMaxDims + c * 32 + lane] = a[c];
  }
}

template <int CH>
__global__ void attn_decode_merge(uint16_t* out, const SeqParams* p) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  const int total = p->pos + 1;
  const int splits = min((total + kSplitRows - 1) / kSplitRows, kMaxSplits);

  float mg = -1e30f;
  for (int sp = 0; sp < splits; ++sp) {
    const size_t idx = static_cast<size_t>(sp) * kMaxHeads + head;
    mg = fmaxf(mg, g_dec_m[idx]);
  }
  float lg = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    const size_t idx = static_cast<size_t>(sp) * kMaxHeads + head;
    lg += g_dec_l[idx] * expf(g_dec_m[idx] - mg);
  }
  float a[CH];
  #pragma unroll
  for (int c = 0; c < CH; ++c) a[c] = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    const size_t idx = static_cast<size_t>(sp) * kMaxHeads + head;
    const float sc = expf(g_dec_m[idx] - mg);
    #pragma unroll
    for (int c = 0; c < CH; ++c)
      a[c] += g_dec_acc[idx * kMaxDims + c * 32 + lane] * sc;
  }
  const size_t o_off = static_cast<size_t>(head) * (CH * 32);
  #pragma unroll
  for (int c = 0; c < CH; ++c)
    sth(out + o_off + c * 32 + lane, a[c] / lg);
}

// ---- env switch + one-shot dispatch diagnostic --------------------------------
bool attn_new_path() {
  static const bool v = [] {
    const char* e = std::getenv("SAGER_ATTN_NEW");
    return !(e && e[0] == '0');
  }();
  return v;
}

int attn_chunks(int head_dim) {
  return (head_dim == 128 || head_dim == 256) ? head_dim / 32 : 0;
}

}  // namespace

namespace {
// one-shot diagnostic: why did the v0.11 path fall back (if it did)?
void attn_fallback_diag(int n_head, int n_head_kv, int head_dim) {
  static bool logged = false;
  if (logged) return;
  logged = true;
  const int ch = attn_chunks(head_dim);
  if (!attn_new_path()) {
    fprintf(stderr, "[sager/attn] v0.11 disabled by SAGER_ATTN_NEW=0 (dims h=%d kvh=%d d=%d)\n",
            n_head, n_head_kv, head_dim);
  } else if (!(ch > 0 && n_head == 8 * n_head_kv && n_head <= kMaxHeads)) {
    fprintf(stderr, "[sager/attn] v0.11 FALLBACK: unsupported dims h=%d kvh=%d d=%d "
            "(need d=128|256, h=8*kvh, h<=%d)\n",
            n_head, n_head_kv, head_dim, kMaxHeads);
  } else {
    fprintf(stderr, "[sager/attn] v0.11 active (dims h=%d kvh=%d d=%d CH=%d)\n",
            n_head, n_head_kv, head_dim, ch);
  }
  fflush(stderr);
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
  attn_fallback_diag(n_head, n_head_kv, head_dim);
  const int ch = attn_chunks(head_dim);
  if (attn_new_path() && ch > 0 && n_head == 8 * n_head_kv &&
      n_head <= kMaxHeads) {
    if (ch == 8) {
      attn_decode_partial<8><<<dim3(n_head, kMaxSplits), 128, 0, stream>>>(
          q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv, sm_scale);
      attn_decode_merge<8><<<dim3(n_head), 32, 0, stream>>>(out, d_params);
    } else {
      attn_decode_partial<4><<<dim3(n_head, kMaxSplits), 128, 0, stream>>>(
          q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv, sm_scale);
      attn_decode_merge<4><<<dim3(n_head), 32, 0, stream>>>(out, d_params);
    }
    return;
  }
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
  attn_fallback_diag(n_head, n_head_kv, head_dim);
  const int ch = attn_chunks(head_dim);
  if (attn_new_path() && ch > 0 && n_head == 8 * n_head_kv &&
      n_head <= kMaxHeads) {
    dim3 grid(n_head_kv, n_tokens_host);
    if (ch == 8) {
      attn_prefill_v2<8><<<grid, 256, 0, stream>>>(
          out, q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv,
          sm_scale);
    } else {
      attn_prefill_v2<4><<<grid, 256, 0, stream>>>(
          out, q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv,
          sm_scale);
    }
    return;
  }
  const dim3 grid(n_head * n_tokens_host);
  attention_kernel<true><<<grid, 32, 0, stream>>>(
      out, q, k_cache, v_cache, kv_dtype, d_params, n_head, n_head_kv,
      head_dim, sm_scale);
}

}  // namespace sager::cuda
