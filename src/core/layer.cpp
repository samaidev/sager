// layer.cpp - hot-tier (VRAM expert cache) rebuild + shared expert forward.
#include "layer_internal.hpp"

#include <cstring>

#include "sager/log.hpp"

#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
#include "sager/kernels/cuda_ops.hpp"
#endif

namespace sager::detail {

// Rebuild the VRAM pointer tables after a residency change. Strategy:
//   1. walk (layer, expert) pairs in residency order, assign VRAM slab slots
//   2. H2D the expert bytes for newly resident experts
//   3. rebuild the 3 pointer tables per layer (null for non-resident)
// The session calls this after cache_.rebalance() reports movement (and once
// after load with the persisted profile to skip the cold-start ramp).
Status HotTier::rebuild(const ExpertCache& cache, const Weights& w,
                        int n_layers, int n_experts, size_t per_expert_bytes) {
#if defined(SAGER_HAVE_CUDA)
  // v0.2 FIX: the parameter shadowed the member (self-assign no-op), so
  // HotTier::per_expert_bytes stayed 0 and the CPU-miss path in
  // session.cpp derived gu = 0 - all three MissJob pointers pointed at the
  // gate matrix. Latent since v0.1 (the model never ran end-to-end).
  this->per_expert_bytes = per_expert_bytes;
  const size_t gu = (static_cast<size_t>(w.dims.n_ff_exp) *
                     static_cast<size_t>(w.dims.hidden) / q4k::kQK_K) *
                    q4k::kBlockBytes;
  const size_t dn = (static_cast<size_t>(w.dims.hidden) *
                     static_cast<size_t>(w.dims.n_ff_exp) / q4k::kQK_K) *
                    q4k::kBlockBytes;
  const uint8_t* host_pool = w.host_expert_base();

  int slot = 0;
  resident_count = 0;
  for (int L = 0; L < n_layers; ++L) {
    q4k::BlockQ4K** tg = tables_host.data() +
                         (static_cast<size_t>(L) * 3 + 0) * n_experts;
    q4k::BlockQ4K** tu = tables_host.data() +
                         (static_cast<size_t>(L) * 3 + 1) * n_experts;
    q4k::BlockQ4K** to = tables_host.data() +
                         (static_cast<size_t>(L) * 3 + 2) * n_experts;
    for (int e = 0; e < n_experts; ++e) {
      if (!cache.resident(L, e)) {
        tg[e] = tu[e] = to[e] = nullptr;
        continue;
      }
      const uint8_t* src = host_pool +
                           (static_cast<size_t>(L) * n_experts + e) * per_expert_bytes;
      uint8_t* dst = pool.get() + static_cast<size_t>(slot) * per_expert_bytes;
      // (Re)copy is idempotent; a full recopy on each rebuild costs at most
      // a few hundred MB over PCIe and happens off the decode path.
      cudaMemcpyAsync(dst, src, per_expert_bytes, cudaMemcpyHostToDevice, 0);
      tg[e] = reinterpret_cast<q4k::BlockQ4K*>(dst);
      tu[e] = reinterpret_cast<q4k::BlockQ4K*>(dst + gu);
      to[e] = reinterpret_cast<q4k::BlockQ4K*>(dst + 2 * gu);
      ++slot;
      ++resident_count;
    }
  }
  (void)dn;
  const size_t table_bytes = tables_host.size() * sizeof(q4k::BlockQ4K*);
  cudaMemcpy(tables.get(), tables_host.data(), table_bytes, cudaMemcpyHostToDevice);
  SLOG_I("cache", "hot tier rebuilt: %d experts resident", resident_count);
  return Status::kOk;
#else
  (void)cache; (void)w; (void)n_layers; (void)n_experts; (void)per_expert_bytes;
  return Status::kUnsupported;
#endif
}

#if defined(SAGER_HAVE_CUDA)
void shared_expert_forward(const Weights& w, int layer,
                           const q4k::BlockQ8K* xq, const uint16_t* x_f16,
                           uint16_t* y, int hidden, int n_ff_exp,
                           cudaStream_t s) {
  const LayerWeights& lw = w.layers()[static_cast<size_t>(layer)];
  if (!lw.shared_gate) return;
  // y_ffn = W_out @ (silu(W_g x) * (W_u x))  - three small GEMVs.
  // Buffers y (hidden) double as scratch: caller passes a [max(hidden,
  // 2*n_ff_exp)] region; we lay g,u after the output row.
  uint16_t* g = y + hidden;
  uint16_t* u = g + n_ff_exp;
  cuda::gemv_q4_k_q8_f16out(reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate), xq, g, n_ff_exp, hidden, s);
  cuda::gemv_q4_k_q8_f16out(reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up), xq, u, n_ff_exp, hidden, s);
  cuda::silu_mul_f16(g, g, u, 1, n_ff_exp, s);
  cuda::gemv_q4_k_q8_f16out(reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out), reinterpret_cast<q4k::BlockQ8K*>(g),
                            y, hidden, n_ff_exp, s);
  // qwen3.5: the shared expert is scaled by sigmoid(router_shexp . x)
  // (a learned scalar gate per token; absent in legacy v0.1 files).
  if (lw.shared_gate_w) {
    float* gate = reinterpret_cast<float*>(u);  // reuse scratch (4 B >= 2 B)
    cuda::gemv_f32_out_f16x(lw.shared_gate_w, x_f16, gate, 1, hidden, s);
    cuda::sigmoid_gate_scalar_f16(y, gate, hidden, s);
  }
}
#endif

}  // namespace sager::detail
