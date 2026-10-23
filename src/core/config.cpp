// config.cpp - metadata-driven model config + runtime sanity validation.
#include "sager/core/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sager {

// Accept both the engine's own "sager.*" keys and the qwen-family keys used
// by community converters, so any correct Q4 build of Sage-A1.5 loads.
Status ModelConfig::load(const std::string& gguf_path, ModelConfig* out) {
  gguf::File f;
  Status st = f.open(gguf_path);
  if (st != Status::kOk) return st;

  EngineDims& d = out->dims;
  out->arch = f.kv_str("general.architecture", out->arch);
  out->quant_name = f.kv_str("general.file_type", out->quant_name);

  d.n_layer       = static_cast<int>(f.kv_int("sager.block_count",
                            f.kv_int("qwen3_5_moe.block_count",
                            f.kv_int("qwen3moe.block_count", d.n_layer))));
  d.n_head        = static_cast<int>(f.kv_int("sager.attention.head_count",
                            f.kv_int("qwen3_5_moe.attention.head_count", d.n_head)));
  d.n_head_kv     = static_cast<int>(f.kv_int("sager.attention.head_count_kv",
                            f.kv_int("qwen3_5_moe.attention.head_count_kv", d.n_head_kv)));
  d.head_dim      = static_cast<int>(f.kv_int("sager.attention.key_length",
                            f.kv_int("qwen3_5_moe.attention.key_length",
                            f.kv_int("sager.attention.head_dim", d.head_dim))));
  d.hidden        = static_cast<int>(f.kv_int("sager.embedding_length",
                            f.kv_int("qwen3_5_moe.embedding_length", d.hidden)));
  d.n_ff          = static_cast<int>(f.kv_int("sager.feed_forward_length",
                            f.kv_int("qwen3_5_moe.feed_forward_length", d.n_ff)));
  d.n_expert      = static_cast<int>(f.kv_int("sager.expert_count",
                            f.kv_int("qwen3_5_moe.expert_count", d.n_expert)));
  d.n_expert_used = static_cast<int>(f.kv_int("sager.expert_used_count",
                            f.kv_int("qwen3_5_moe.expert_used_count", d.n_expert_used)));
  d.n_ff_exp      = static_cast<int>(f.kv_int("sager.expert_feed_forward_length",
                            f.kv_int("qwen3_5_moe.expert_feed_forward_length",
                            f.kv_int("qwen3_5_moe.moe_intermediate_size", d.n_ff_exp))));
  d.n_vocab       = static_cast<int>(f.kv_int("sager.vocab_size",
                            f.kv_int("qwen3_5_moe.vocab_size", d.n_vocab)));
  d.n_ctx_train   = static_cast<int>(f.kv_int("sager.context_length",
                            f.kv_int("qwen3_5_moe.context_length", d.n_ctx_train)));
  d.shared_expert = static_cast<int>(f.kv_int("sager.expert_shared_count",
                            f.kv_int("qwen3_5_moe.expert_shared_count", d.shared_expert)));
  // Layers that are MoE (defaults to all blocks; Sage-A1.5 is fully MoE).
  d.n_moe_layer   = static_cast<int>(f.kv_int("sager.moe_layer_count",
                            f.kv_int("qwen3_5_moe.moe_layer_count", d.n_layer)));

  out->norm_eps     = static_cast<float>(f.kv_double("sager.attention.layer_norm_rms_epsilon",
                              f.kv_double("qwen3_5_moe.attention.layer_norm_rms_epsilon", 1e-6)));
  out->rope_theta   = static_cast<float>(f.kv_double("sager.rope.theta",
                              f.kv_double("qwen3_5_moe.rope.theta", 10000000.0)));
  out->rope_partial = static_cast<float>(f.kv_double("sager.rope.partial_rotary_factor",
                              f.kv_double("qwen3_5_moe.rope.partial_rotary_factor", 0.25)));

  // mrope: Sage-A1.5 uses interleaved mrope with sections [11, 11, 10]
  // (from the model card rope_parameters). Detect via presence of the key.
  if (f.kv_find("sager.rope.mrope_section") || f.kv_find("qwen3_5_moe.rope.mrope_section")) {
    out->use_mrope  = true;
    out->rope_style = 2;
    const gguf::Value* sec = f.kv_find("sager.rope.mrope_section");
    sec = sec ? sec : f.kv_find("qwen3_5_moe.rope.mrope_section");
    if (sec && sec->type == gguf::Type::array && sec->raw.size() >= 12) {
      auto rd_i32 = [&](size_t off) {
        int32_t x; std::memcpy(&x, sec->raw.data() + off, 4); return x;
      };
      out->mrope_section = {rd_i32(0), rd_i32(4), rd_i32(8)};
    }
  }

  // MTP head presence is signaled by the MTP tensors themselves.
  d.mtp = f.tensor("mtp.fc.weight") != nullptr || f.tensor("mtp.proj_in.weight") != nullptr;

  out->dims = d;
  return validate_dims(out->dims);
}

Status validate_dims(const EngineDims& d) {
  if (d.n_layer <= 0 || d.n_layer > 512) return Status::kGeometryMismatch;
  if (d.n_head <= 0 || d.n_head_kv <= 0 || d.n_head % d.n_head_kv != 0)
    return Status::kGeometryMismatch;                        // GQA must divide
  if (d.head_dim <= 0 || d.hidden <= 0) return Status::kGeometryMismatch;
  if (d.n_expert <= 0 || d.n_expert_used <= 0 || d.n_expert_used > d.n_expert)
    return Status::kGeometryMismatch;
  if (d.n_ff_exp <= 0 || d.n_vocab <= 0) return Status::kGeometryMismatch;
  if (d.n_ctx_train < 1024) return Status::kGeometryMismatch;
  // Sanity envelope: 3B-500B total params equivalent at any plausible bw.
  const double approx_params = static_cast<double>(d.experts_total()) *
      3.0 * d.n_ff_exp * d.hidden + static_cast<double>(d.n_layer) *
      (2.0 * d.hidden * (d.n_head + 2 * d.n_head_kv) * d.head_dim +
       d.hidden * d.n_head * d.head_dim);
  if (approx_params < 1e9 || approx_params > 5e11) return Status::kGeometryMismatch;
  return Status::kOk;
}

}  // namespace sager
