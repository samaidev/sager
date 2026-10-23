// config.hpp - model + runtime configuration, entirely metadata-driven.
//
// Model geometry comes from GGUF KV pairs under both the "sager.*" and
// standard qwen-family keys, so files produced by tools/convert_hf_to_sager.py
// and community GGUFs both load. Runtime knobs come from engine argv / config
// JSON (see serve/server.py and docs/t4-guide.md for T4-specific values).
#pragma once

#include <string>
#include <vector>

#include "sager/common.hpp"
#include "sager/core/gguf.hpp"
#include "sager/kernels/q4_k.hpp"

namespace sager {

struct ModelConfig {
  EngineDims dims;

  // Norm / rope params from metadata (Sage-A1.5 values per model card).
  float norm_eps            = 1e-6f;
  float rope_theta          = 10000000.0f;   // 1e7 per card
  float rope_partial        = 0.25f;         // partial_rotary_factor
  int   rope_style          = 0;             // 0=NeoX, 1=interleaved, 2=mrope
  bool  use_mrope           = false;
  std::vector<int> mrope_section = {11, 11, 10};

  // Architecture identity
  std::string arch          = "sager.sage_a15";
  std::string quant_name    = "Q4_K_M";

  static Status load(const std::string& gguf_path, ModelConfig* out);

  // Bytes of ONE expert (gate+up+out, Q4_K) - the unit of the expert pool.
  size_t expert_bytes() const {
    const size_t blocks = static_cast<size_t>(dims.n_ff_exp * dims.hidden) /
                          q4k::kQK_K; // gate and up have identical geometry
    const size_t per_mat = blocks * q4k::kBlockBytes;
    // out matrix: [hidden x n_ff_exp]
    const size_t out_mat = (static_cast<size_t>(dims.hidden * dims.n_ff_exp) /
                            q4k::kQK_K) * q4k::kBlockBytes;
    return per_mat * 2 + out_mat;
  }
  size_t experts_total_bytes() const {
    return static_cast<size_t>(dims.experts_total()) * expert_bytes();
  }
};

// Runtime context parameters (passed to Session).
struct CtxParams {
  int   n_ctx            = 16384;  // T4 default (16K tokens, KV q8); see docs/t4-guide.md
  int   chunk_prefill    = 8192;   // Strata-style chunked prefill
  bool  kv_q8            = true;   // Q8 KV cache: ~2x more context per GB
  int   batch_threads    = 0;      // CPU pool workers (0 = hw_concurrency/2)
  size_t vram_budget_mb  = 10240;  // hot-expert pool budget inside 16GB T4
  bool  capture_graphs   = true;   // per-layer decode CUDA graphs
  bool  enable_mtp       = true;   // speculative drafting (needs MTP head)
  int   mtp_max_draft    = 3;      // model card: num_speculative_tokens=3
  bool  enable_suffix_draft = true;
  int   suffix_max_draft = 5;
  // Sampling defaults = Sage-A1.5 recommended params (model card).
  float temp             = 1.0f;
  int   top_k            = 20;
  float top_p            = 0.95f;
  float min_p            = 0.0f;
  float presence_penalty = 1.5f;
  float rep_penalty      = 1.0f;
};

// Sanity-check geometry against the training-size envelope; refuse absurd
// files instead of misindexing (kGeometryMismatch).
Status validate_dims(const EngineDims& d);

}  // namespace sager
