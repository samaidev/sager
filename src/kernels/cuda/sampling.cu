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

}  // namespace

int argmax_f32(const float*, int, cudaStream_t) {
  // The session/verify path performs argmax on the host over the already-
  // copied logits (verify needs the full distribution anyway); this stub
  // exists so the linker has a symbol for the declared API.
  return -1;
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
