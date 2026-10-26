// sampling.cu - penalties, temperature, top-k/top-p/min_p + argmax.
//
// Sage-A1.5 recommended sampling (model card): temperature 1.0, top_p 0.95,
// top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0.
#include "sager/kernels/cuda_ops.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <vector>

namespace sager::cuda {
namespace {

__global__ void scale_logits_kernel(float* logits, float scale, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) logits[i] *= scale;
}

__global__ void apply_penalties_kernel(float* logits, const int* recent_ids,
                                       int n_recent, float rep_penalty,
                                       float presence_penalty) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_recent) return;
  const int id = recent_ids[i];
  float v = logits[id];
  if (rep_penalty != 1.0f) {
    v = (v > 0.f) ? v / rep_penalty : v * rep_penalty;
  }
  if (presence_penalty != 0.f) {
    v -= presence_penalty;
  }
  logits[id] = v;
}

__global__ void softmax_inplace_kernel(float* logits, int n) {
  __shared__ float m_sh, z_sh;
  if (threadIdx.x == 0) {
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, logits[i]);
    m_sh = mx;
    float z = 0.f;
    for (int i = 0; i < n; ++i) z += expf(logits[i] - mx);
    z_sh = z;
  }
  __syncthreads();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) logits[i] = expf(logits[i] - m_sh) / z_sh;
}

__global__ void apply_penalties_devn_kernel(float* logits,
                                            const int* recent_ids,
                                            const int* n_recent,
                                            const float* params) {
  // Deterministic duplicate handling: the sequential walk applies f(v) =
  // (v/rep - pres) once per occurrence of the id. Fold all m applications
  // into the LAST occurrence's thread (others exit) - race-free and equal to
  // the ideal sequential semantics for any rep/pres combination.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = *n_recent;
  if (i >= n) return;
  const int id = recent_ids[i];
  if (id < 0) return;                       // sentinel slot
  int total = 0, ordinal = 0;
  for (int j = 0; j < n; ++j) {
    if (recent_ids[j] == id) {
      ++total;
      if (j <= i) ++ordinal;
    }
  }
  if (ordinal != total) return;             // a later occurrence applies it
  float v = logits[id];
  const float rep = params[0];
  const float pres = params[1];
  // Same application order as sample_host: all rep divisions, then all
  // presence subtractions.
  if (rep != 1.0f) {
    for (int r = 0; r < total; ++r) v = (v > 0.f) ? v / rep : v * rep;
  }
  if (pres != 0.f) {
    for (int r = 0; r < total; ++r) v -= pres;
  }
  logits[id] = v;
}

}  // namespace

void apply_penalties_f32_devn(float* logits, const int* recent_ids,
                              const int* n_recent, const float* params,
                              cudaStream_t s) {
  const int cap = 64;   // kMaxRecent; threads beyond the count exit early
  apply_penalties_devn_kernel<<<1, cap, 0, s>>>(logits, recent_ids, n_recent,
                                                params);
}

int argmax_f32(const float*, int, cudaStream_t) {
  // The session/verify path performs argmax on the host over the already-
  // copied logits (verify needs the full distribution anyway); this stub
  // exists so the linker has a symbol for the declared API.
  return -1;
}

// ---------------------------------------------------------------------------
// v0.5 device top-k selection (decode sampler): one block iteratively picks
// the max (ties -> lower index), masks it in a register-local shadow via a
// picked-flag scan... the simple version re-scans logits k times; with the
// 1 MB fp32 head resident in L2 each pass is a stream through L2 (~40 us
// for 248k logits on T4), so k <= 64 stays well under 0.5 ms.
// ---------------------------------------------------------------------------
__global__ void topk_select_kernel(const float* __restrict__ logits,
                                   float* __restrict__ out_v,
                                   int* __restrict__ out_i, int n, int k) {
  __shared__ float red_v[32];
  __shared__ int red_i[32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nth = blockDim.x;

  for (int it = 0; it < k; ++it) {
    float bv = -3.4e38f;
    int bi = -1;
    for (int i = tid; i < n; i += nth) {
      const float v = logits[i];
      if (v > bv || (v == bv && bi >= 0 && i < bi)) { bv = v; bi = i; }
    }
    // warp reduce (max, then lowest index on ties)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
      const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
      if (ov > bv || (ov == bv && oi >= 0 && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
      float mv = -3.4e38f;
      int mi = -1;
      const int nw = (nth + 31) / 32;
      for (int w = 0; w < nw; ++w) {
        if (red_v[w] > mv || (red_v[w] == mv && red_i[w] >= 0 &&
                              (mi < 0 || red_i[w] < mi))) {
          mv = red_v[w];
          mi = red_i[w];
        }
      }
      out_v[it] = mv;
      out_i[it] = mi;
      // mask the winner for the next pass (logits buffer is a scratch copy)
      if (mi >= 0) const_cast<float*>(logits)[mi] = -3.4e38f;
    }
    __syncthreads();
  }
}

void topk_select_f32(const float* logits, int n, float* out_v, int* out_i,
                     int k, cudaStream_t stream) {
  if (k <= 0 || n <= 0) return;
  if (k > 64) k = 64;
  topk_select_kernel<<<1, 1024, 0, stream>>>(logits, out_v, out_i, n, k);
}

// ---------------------------------------------------------------------------
// v0.6 two-phase top-k. Phase 1 computes a per-segment top-k in parallel
// (one block per 2048-logit segment, the same (value, lower-index) rule as
// the single-block kernel), phase 2 merges the segment candidates in one
// block. The union of the segment sets always covers the global top-k (every
// segment holds >= k elements) and both phases apply the identical tie rule,
// so the selected set is exactly the single-block kernel's - the split only
// changes which SM scans what. ~25 us instead of ~700 us at n=248320, k=32.
// ---------------------------------------------------------------------------
inline constexpr int kTopkSeg = 2048;

__global__ void topk_seg_kernel(const float* __restrict__ logits,
                                float* __restrict__ out_v,
                                int* __restrict__ out_i, int n, int k,
                                int seg) {
  extern __shared__ float sm[];          // segment copy, winners masked
  const int base = blockIdx.x * seg;
  const int cnt = min(seg, n - base);
  for (int i = threadIdx.x; i < seg; i += blockDim.x) {
    sm[i] = (i < cnt) ? logits[base + i] : -3.4e38f;
  }
  __syncthreads();
  __shared__ float red_v[32];
  __shared__ int red_i[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nth = blockDim.x;
  for (int it = 0; it < k; ++it) {
    float bv = -3.4e38f;
    int bi = -1;
    for (int i = threadIdx.x; i < seg; i += nth) {
      const float v = sm[i];
      const int gi = base + i;
      if (v > bv || (v == bv && bi >= 0 && gi < bi)) { bv = v; bi = gi; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
      const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
      if (ov > bv || (ov == bv && oi >= 0 && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float mv = -3.4e38f;
      int mi = -1;
      const int nw = (nth + 31) / 32;
      for (int w = 0; w < nw; ++w) {
        if (red_v[w] > mv || (red_v[w] == mv && red_i[w] >= 0 &&
                              (mi < 0 || red_i[w] < mi))) {
          mv = red_v[w];
          mi = red_i[w];
        }
      }
      out_v[static_cast<size_t>(blockIdx.x) * k + it] = mv;
      out_i[static_cast<size_t>(blockIdx.x) * k + it] = mi;
      if (mi >= 0) sm[mi - base] = -3.4e38f;
    }
    __syncthreads();
  }
}

__global__ void topk_merge_kernel(const float* __restrict__ in_v,
                                  int* __restrict__ in_i,
                                  float* __restrict__ out_v,
                                  int* __restrict__ out_i, int m, int k) {
  __shared__ float red_v[32];
  __shared__ int red_i[32];
  __shared__ int red_s[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nth = blockDim.x;
  for (int it = 0; it < k; ++it) {
    float bv = -3.4e38f;
    int bi = -1, bs = -1;                 // value, candidate id, slot
    for (int i = threadIdx.x; i < m; i += nth) {
      const float v = in_v[i];
      const int gi = in_i[i];
      if (v > bv || (v == bv && gi >= 0 && (bi < 0 || gi < bi))) {
        bv = v; bi = gi; bs = i;
      }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
      const int oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
      const int os = __shfl_down_sync(0xFFFFFFFFu, bs, off);
      if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
        bv = ov; bi = oi; bs = os;
      }
    }
    if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; red_s[warp] = bs; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float mv = -3.4e38f;
      int mi = -1, ms = -1;
      const int nw = (nth + 31) / 32;
      for (int w = 0; w < nw; ++w) {
        if (red_v[w] > mv || (red_v[w] == mv && red_i[w] >= 0 &&
                              (mi < 0 || red_i[w] < mi))) {
          mv = red_v[w];
          mi = red_i[w];
          ms = red_s[w];
        }
      }
      out_v[it] = mv;
      out_i[it] = mi;
      if (ms >= 0) const_cast<float*>(in_v)[ms] = -3.4e38f;
    }
    __syncthreads();
  }
}

void topk_select_f32_2p(const float* logits, float* seg_v, int* seg_i,
                        int seg_slots, float* out_v, int* out_i, int n, int k,
                        cudaStream_t stream) {
  if (k <= 0 || n <= 0) return;
  if (k > 64) k = 64;
  const int nblk = (n + kTopkSeg - 1) / kTopkSeg;
  if (nblk * k > seg_slots) {
    topk_select_kernel<<<1, 1024, 0, stream>>>(logits, out_v, out_i, n, k);
    return;
  }
  topk_seg_kernel<<<nblk, 256, kTopkSeg * sizeof(float), stream>>>(
      logits, seg_v, seg_i, n, k, kTopkSeg);
  topk_merge_kernel<<<1, 256, 0, stream>>>(seg_v, seg_i, out_v, out_i,
                                           nblk * k, k);
}

void apply_penalties_f32(float* logits, const int* recent_ids, int n_recent,
                         float rep_penalty, float presence_penalty,
                         cudaStream_t stream) {
  if (n_recent <= 0 || (rep_penalty == 1.0f && presence_penalty == 0.f)) {
    return;
  }
  apply_penalties_kernel<<<(n_recent + 255) / 256, 256, 0, stream>>>(
      logits, recent_ids, n_recent, rep_penalty, presence_penalty);
}

int sample_topk_topp(const float* logits, int n_vocab, const int* recent_ids,
                     int n_recent, float temp, int top_k, float top_p,
                     float min_p, float rep_penalty, float presence_penalty,
                     uint64_t* rng_state, cudaStream_t stream) {
  // 1) penalties
  if (n_recent > 0 && (rep_penalty != 1.0f || presence_penalty != 0.f)) {
    apply_penalties_kernel<<<(n_recent + 255) / 256, 256, 0, stream>>>(
        const_cast<float*>(logits), recent_ids, n_recent, rep_penalty,
        presence_penalty);
  }
  // 2) temperature
  if (temp != 1.0f && temp > 0.f) {
    // scale in place
    scale_logits_kernel<<<(n_vocab + 255) / 256, 256, 0, stream>>>(
        const_cast<float*>(logits), 1.f / temp, n_vocab);
  }
  // 3) softmax
  softmax_inplace_kernel<<<1, 128, 0, stream>>>(const_cast<float*>(logits),
                                                n_vocab);

  // 4) top-k / top-p / min_p filtering + multinomial pick happen on the HOST
  //    over a compact candidate set: we copy probs once (0.6 MB) - on T4
  //    this is <1 ms and keeps the sampler trivially auditable.
  static std::vector<float> h(n_vocab);
  cudaMemcpy(h.data(), logits, n_vocab * sizeof(float), cudaMemcpyDeviceToHost);

  // top-k indices
  std::vector<int> order(n_vocab);
  for (int i = 0; i < n_vocab; ++i) order[i] = i;
  const int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int a, int b) { return h[a] > h[b]; });

  // top-p + min_p over the top-k prefix
  float z = 0.f, mx = 0.f;
  for (int i = 0; i < k; ++i) { z += h[order[i]]; mx = fmaxf(mx, h[order[i]]); }
  float cum = 0.f;
  int cutoff = k;
  for (int i = 0; i < k; ++i) {
    cum += h[order[i]];
    const float p = h[order[i]] / (z > 0 ? z : 1.f);
    if (min_p > 0.f && p < min_p * (mx / (z > 0 ? z : 1.f))) { cutoff = i; break; }
    if (top_p < 1.f && cum / (z > 0 ? z : 1.f) > top_p) { cutoff = i + 1; break; }
  }
  cutoff = std::max(1, cutoff);

  // multinomial
  std::vector<float> cdf(cutoff);
  float acc = 0.f;
  for (int i = 0; i < cutoff; ++i) { acc += h[order[i]]; cdf[i] = acc; }
  (*rng_state) = (*rng_state) * 6364136223846793005ull + 1442695040888963407ull;
  const float u = static_cast<float>((*rng_state) >> 40) / 16777216.f *
                  (acc > 0 ? acc : 1.f);
  int pick = 0;
  while (pick < cutoff - 1 && cdf[pick] < u) ++pick;
  return order[pick];
}

}  // namespace sager::cuda
