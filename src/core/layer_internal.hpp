// layer_internal.hpp - INTERNAL header (not installed): hot-expert residency
// plumbing shared between session.cpp and layer.cpp.
#pragma once

#include "sager/core/session.hpp"

namespace sager::detail {

// VRAM hot-expert cache layout owned by the session:
//   hot_pool_      : one big device slab (budget bytes)
//   ptr_tables_    : per layer, 3 device pointer arrays (gate/up/out) with
//                    n_expert entries each; entries for NON-resident experts
//                    are null. Rebuilt after every rebalance (host pinned
//                    staging + one H2D).
struct HotTier {
  DeviceBuffer<uint8_t> pool;
  // ptr tables: [layer][mat][expert] (mat: 0=gate,1=up,2=out)
  DeviceBuffer<q4k::BlockQ4K*> tables;         // device side
  std::vector<q4k::BlockQ4K*>  tables_host;    // pinned staging
  size_t per_expert_bytes = 0;
  int    resident_count = 0;

  Status rebuild(const ExpertCache& cache, const Weights& w, int n_layers,
                 int n_experts, size_t per_expert_bytes);
};

// Shared-expert forward for one token/row (device): y = W_out(silu(W_g x) * W_u x)
#if defined(SAGER_HAVE_CUDA)
void shared_expert_forward(const Weights& w, int layer,
                           const q4k::BlockQ8K* xq, const uint16_t* x_f16,
                           uint16_t* y, int hidden, int n_ff_exp,
                           cudaStream_t s);
#endif

}  // namespace sager::detail
