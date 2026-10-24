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
  // Q4_K attention projections. On qwen3.5-style gated attention the q
  // projection carries [q | gate] (2*n_head*head_dim rows); wq points at the
  // q half and attn_gate at the sigmoid-gate half. Legacy (v0.1) files have
  // attn_gate == nullptr and wq covers n_head*head_dim rows only.
  uint16_t* wq          = nullptr;
  uint16_t* attn_gate   = nullptr;
  uint16_t* wk          = nullptr;   // nullptr on gated-delta layers
  uint16_t* wv          = nullptr;
  uint16_t* wo          = nullptr;
  // Per-head q/k RMSNorm weights of the FULL-attention layers (fp16,
  // [head_dim]); GDN layers have none (nullptr).
  uint16_t* attn_q_norm = nullptr;
  uint16_t* attn_k_norm = nullptr;
  // True when the q projection carries [q | gate] (qwen3.5 gated attention);
  // legacy v0.1 files have plain q and the engine skips the gate multiply.
  bool attn_gated = false;
  // MoE block
  uint16_t* ffn_norm    = nullptr;   // [hidden]
  float*    router      = nullptr;   // [n_expert * hidden] fp32 (tiny, exact)
  float*    shared_gate_w = nullptr; // [hidden] fp32 router of the shared expert
  uint16_t* shared_gate = nullptr;
  uint16_t* shared_up   = nullptr;
  uint16_t* shared_out  = nullptr;

  // ---- v0.2: gated delta net (linear attention) layers --------------------
  // Carved from ONE cudaMalloc per layer (freed via lin_in_qkvz in unload).
  //   in_qkvz  Q4_K [proj_rows x hidden]  fused q|k|v|z projection (the
  //            checkpoint's attn_qkv + attn_gate tensors row-concatenated)
  //   in_ba    Q4_K [2*nv x hidden]       beta|a projections (per V-head)
  //   out_proj Q4_K [hidden x nv*V]       output projection
  //   conv_w   fp16 [conv_dim x ck]       depthwise causal conv taps
  //   conv_b   fp16 [conv_dim]            conv bias (optional, may be null)
  //   A        fp32 [nv]                  gate decay (checkpoint ssm_a:
  //                                     pre-negated -exp(A_log))
  //   dt_bias  fp32 [nv]                  softplus offset
  //   o_norm   fp16 [V]                   output-gate normalization
  uint16_t* lin_in_qkvz = nullptr;
  uint16_t* lin_in_ba   = nullptr;
  uint16_t* lin_out     = nullptr;
  uint16_t* lin_conv_w  = nullptr;
  uint16_t* lin_conv_b  = nullptr;
  float*    lin_A_log   = nullptr;   // holds ssm_a (already -exp(A_log))
  float*    lin_dt_bias = nullptr;
  uint16_t* lin_o_norm  = nullptr;
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
  uint16_t* output      = nullptr;   // head: Q4_K blocks or fp16 (head_type_)
  uint16_t* output_norm = nullptr;   // [hidden]

  // MTP head (optional; used by the drafter)
  uint16_t* mtp_proj_in  = nullptr;  // [2*hidden x hidden]
  uint16_t* mtp_proj_out = nullptr;  // [hidden x vocab_hidden]
  bool mtp_present = false;

  EngineDims dims{};

  // head_type_: GGUF dtype of the resident head matrix.
  //   12 = Q4_K (native quant, Q4 GEMV path)
  //    1 = fp16 (Q6_K head dequantized at load; F16 GEMV path)
  uint32_t head_type_ = 12;
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
  int       pool_map_fd_ = -1;    // SAGER_EXPERT_POOL_FILE backing (mmap)
  size_t    pool_map_size_ = 0;
};

// Convert one fp32 row to fp16 packed bytes (used across the loader).
void f32_to_f16_rows(const float* src, uint16_t* dst, size_t n);

}  // namespace sager
