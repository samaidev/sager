// weights.hpp - residency plan + resident weight state.
//
// The loader walks the GGUF tensor directory once and assigns every tensor to
// one of three tiers (Strata's "three-tier" memory model):
//
//   tier 0 (VRAM): token embedding, per-layer QKV/O projections, norms,
//                  router weights, shared experts, output head, MTP head,
//                  plus the hot-expert cache POOL (dynamic, see ExpertCache)
//   tier 1 (RAM) : every routed expert (Q4_K, pinned host pool)
//   tier 2 (SSD) : not used in v0.1 - a Q4 build of a 35B MoE fits in RAM;
//                  see docs/roadmap.md for the n-gram SSD tier
//
// Expert weights are COPIED out of the mmap into the pinned pool so the CPU
// expert pool can compute misses directly against stable addresses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sager/common.hpp"
#include "sager/core/gguf.hpp"
#include "sager/core/tensor.hpp"

#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace sager {

struct LayerWeights {
  // All device pointers (nullptr on CPU-only builds / not-present tensors).
  uint16_t* attn_norm   = nullptr;   // [hidden] fp16
  uint16_t* wq          = nullptr;   // Q4_K? No: attention stays fp16 on T4
  uint16_t* wk          = nullptr;   // (fp16 packed rows, see load())
  uint16_t* wv          = nullptr;
  uint16_t* wo          = nullptr;
  // MoE block
  uint16_t* ffn_norm    = nullptr;   // [hidden]
  float*    router      = nullptr;   // [n_expert * hidden] fp32 (tiny, exact)
  uint16_t* shared_gate = nullptr;
  uint16_t* shared_up   = nullptr;
  uint16_t* shared_out  = nullptr;
};

class Weights {
 public:
  Weights() = default;
  ~Weights() { unload(); }
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  Status load(gguf::File& f, const struct ModelConfig& cfg,
              size_t vram_budget_bytes);

  void unload();

  // Flat accessor for the expert pool (owned by ExpertCache, see below):
  // experts live at host_pinned_experts + (layer*n_expert + e)*expert_bytes
  uint8_t* host_expert_base() const { return expert_pool_host_; }
  size_t   expert_pool_bytes() const { return expert_pool_bytes_; }

  const std::vector<LayerWeights>& layers() const { return layers_; }

  // Embedding / head
  uint16_t* tok_embd    = nullptr;   // [n_vocab x hidden] fp16
  uint16_t* output      = nullptr;   // [hidden x n_vocab] fp16 (tied if absent)
  uint16_t* output_norm = nullptr;   // [hidden]

  // MTP head (optional; used by the drafter)
  uint16_t* mtp_proj_in  = nullptr;  // [2*hidden x hidden]
  uint16_t* mtp_proj_out = nullptr;  // [hidden x vocab_hidden]
  bool mtp_present = false;

  EngineDims dims{};
  uint32_t head_type_ = 12;   // GGUF dtype of output head (12=Q4_K, 14=Q6_K)
  int      head_rows_ = 0;    // n_vocab (rows of the head matrix)

 private:
  Status alloc_vram(const struct ModelConfig& cfg);
  Status copy_attention(gguf::File& f, const struct ModelConfig& cfg);
  Status copy_moe_shared(gguf::File& f, const struct ModelConfig& cfg);
  Status load_expert_pool(gguf::File& f, const struct ModelConfig& cfg);

  std::vector<LayerWeights> layers_;

  // Pinned host pool holding EVERY routed expert in Q4_K (tier 1).
  uint8_t*  expert_pool_host_ = nullptr;
  size_t    expert_pool_bytes_ = 0;
};

// Convert one fp32 row to fp16 packed bytes (used across the loader).
void f32_to_f16_rows(const float* src, uint16_t* dst, size_t n);

}  // namespace sager
