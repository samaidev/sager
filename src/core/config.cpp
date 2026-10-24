// config.cpp - metadata-driven model config + runtime sanity validation.
#include "sager/core/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "sager/log.hpp"

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
                            f.kv_int("qwen35moe.block_count",
                            f.kv_int("qwen3_5_moe.block_count",
                            f.kv_int("qwen3moe.block_count", d.n_layer)))));
  // llama.cpp-style files count the MTP/nextn draft layer in block_count;
  // the main model walks blocks 0..n_layer-1 (v0.2 ignores the draft head).
  d.n_layer -= static_cast<int>(f.kv_int("qwen35moe.nextn_predict_layers",
                              f.kv_int("qwen3_5_moe.nextn_predict_layers", 0)));
  d.n_head        = static_cast<int>(f.kv_int("sager.attention.head_count",
                            f.kv_int("qwen35moe.attention.head_count",
                            f.kv_int("qwen3_5_moe.attention.head_count", d.n_head))));
  d.n_head_kv     = static_cast<int>(f.kv_int("sager.attention.head_count_kv",
                            f.kv_int("qwen35moe.attention.head_count_kv",
                            f.kv_int("qwen3_5_moe.attention.head_count_kv", d.n_head_kv))));
  d.head_dim      = static_cast<int>(f.kv_int("sager.attention.key_length",
                            f.kv_int("qwen35moe.attention.key_length",
                            f.kv_int("qwen3_5_moe.attention.key_length",
                            f.kv_int("sager.attention.head_dim", d.head_dim)))));
  d.hidden        = static_cast<int>(f.kv_int("sager.embedding_length",
                            f.kv_int("qwen35moe.embedding_length",
                            f.kv_int("qwen3_5_moe.embedding_length", d.hidden))));
  d.n_ff          = static_cast<int>(f.kv_int("sager.feed_forward_length",
                            f.kv_int("qwen3_5_moe.feed_forward_length", d.n_ff)));
  d.n_expert      = static_cast<int>(f.kv_int("sager.expert_count",
                            f.kv_int("qwen35moe.expert_count",
                            f.kv_int("qwen3_5_moe.expert_count", d.n_expert))));
  d.n_expert_used = static_cast<int>(f.kv_int("sager.expert_used_count",
                            f.kv_int("qwen35moe.expert_used_count",
                            f.kv_int("qwen3_5_moe.expert_used_count", d.n_expert_used))));
  d.n_ff_exp      = static_cast<int>(f.kv_int("sager.expert_feed_forward_length",
                            f.kv_int("qwen35moe.expert_feed_forward_length",
                            f.kv_int("qwen3_5_moe.expert_feed_forward_length",
                            f.kv_int("qwen35moe.expert_shared_feed_forward_length",
                            f.kv_int("qwen3_5_moe.moe_intermediate_size", d.n_ff_exp))))));
  d.n_vocab       = static_cast<int>(f.kv_int("sager.vocab_size",
                            f.kv_int("qwen35moe.vocab_size",
                            f.kv_int("qwen3_5_moe.vocab_size", d.n_vocab))));
  // Native llama.cpp exports carry no vocab KV: fall back to the embedding's
  // row count (token_embd.weight = [hidden x n_vocab] in GGUF order).
  if (f.kv_find("sager.vocab_size") == nullptr &&
      f.kv_find("qwen35moe.vocab_size") == nullptr &&
      f.kv_find("qwen3_5_moe.vocab_size") == nullptr) {
    if (const gguf::TensorInfo* te = f.tensor("token_embd.weight");
        te != nullptr && te->dims.size() >= 2)
      d.n_vocab = static_cast<int>(te->dims[1]);
  }
  d.n_ctx_train   = static_cast<int>(f.kv_int("sager.context_length",
                            f.kv_int("qwen35moe.context_length",
                            f.kv_int("qwen3_5_moe.context_length", d.n_ctx_train))));
  d.shared_expert = static_cast<int>(f.kv_int("sager.expert_shared_count",
                            f.kv_int("qwen3_5_moe.expert_shared_count", d.shared_expert)));
  // Layers that are MoE (defaults to all blocks; Sage-A1.5 is fully MoE).
  d.n_moe_layer   = static_cast<int>(f.kv_int("sager.moe_layer_count",
                            f.kv_int("qwen3_5_moe.moe_layer_count", d.n_layer)));

  out->norm_eps     = static_cast<float>(f.kv_double("sager.attention.layer_norm_rms_epsilon",
                              f.kv_double("qwen35moe.attention.layer_norm_rms_epsilon",
                              f.kv_double("qwen3_5_moe.attention.layer_norm_rms_epsilon", 1e-6))));
  out->rope_theta   = static_cast<float>(f.kv_double("sager.rope.theta",
                              f.kv_double("qwen35moe.rope.freq_base",
                              f.kv_double("qwen3_5_moe.rope.theta", 10000000.0))));
  // llama.cpp files express partial rope as rope.dimension_count (rotary dims
  // per head); derive the factor from key_length when the factor is absent.
  {
    const int rope_dims = static_cast<int>(f.kv_int("qwen35moe.rope.dimension_count", 0));
    if (rope_dims > 0 && d.head_dim > 0)
      out->rope_partial = static_cast<float>(rope_dims) /
                          static_cast<float>(d.head_dim);
  }
  out->rope_partial = static_cast<float>(f.kv_double("sager.rope.partial_rotary_factor",
                              f.kv_double("qwen3_5_moe.rope.partial_rotary_factor",
                              static_cast<double>(out->rope_partial))));

  // mrope: Sage-A1.5 uses interleaved mrope with sections [11, 11, 10]
  // (from the model card rope_parameters). Detect via presence of the key.
  if (f.kv_find("sager.rope.mrope_section") ||
      f.kv_find("qwen35moe.rope.dimension_sections") ||
      f.kv_find("qwen3_5_moe.rope.mrope_section")) {
    out->use_mrope  = true;
    out->rope_style = 2;
    const gguf::Value* sec = f.kv_find("sager.rope.mrope_section");
    sec = sec ? sec : f.kv_find("qwen35moe.rope.dimension_sections");
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

  // ---- v0.2: gated-delta (linear attention) layers ------------------------
  // The converter writes sager.layer_is_linear as a u32 array (one entry per
  // block, 1 = GDN). GDN geometry comes from sager.gdn.* keys, defaulting to
  // the Sage-A1.5 model card values.
  if (const gguf::Value* la = f.kv_find("sager.layer_is_linear")) {
    if (la->type == gguf::Type::array && la->elem_type == gguf::Type::u32) {
      const size_t n = la->raw.size() / 4;
      out->layer_is_linear.resize(n);
      for (size_t i = 0; i < n; ++i) {
        uint32_t x;
        std::memcpy(&x, la->raw.data() + i * 4, 4);
        out->layer_is_linear[i] = x != 0;
      }
    }
  } else if (const int fai = static_cast<int>(
                 f.kv_int("qwen35moe.full_attention_interval", 0)); fai > 1) {
    // Native qwen35moe layout: every fai-th block (last of each group) is
    // full GQA attention, the rest are gated-delta linear attention.
    out->layer_is_linear.resize(static_cast<size_t>(d.n_layer));
    for (int L = 0; L < d.n_layer; ++L)
      out->layer_is_linear[static_cast<size_t>(L)] =
          (L % fai) != (fai - 1);
  }
  // llama.cpp-style qwen35moe files carry the GDN geometry under ssm.*:
  //   group_count    -> k heads (each k-head owns state groups)
  //   time_step_rank -> v heads (the a/b/A/dt projections are this wide)
  //   state_size     -> head K dim AND head V dim (square per-head state)
  //   conv_kernel    -> depthwise causal conv taps
  d.n_gdn_k_heads   = static_cast<int>(f.kv_int("qwen35moe.ssm.group_count",
                              f.kv_int("sager.gdn.key_heads", d.n_gdn_k_heads)));
  d.n_gdn_v_heads   = static_cast<int>(f.kv_int("qwen35moe.ssm.time_step_rank",
                              f.kv_int("sager.gdn.value_heads", d.n_gdn_v_heads)));
  d.gdn_key_dim     = static_cast<int>(f.kv_int("qwen35moe.ssm.state_size",
                              f.kv_int("sager.gdn.key_dim", d.gdn_key_dim)));
  d.gdn_value_dim   = static_cast<int>(f.kv_int("qwen35moe.ssm.state_size",
                              f.kv_int("sager.gdn.value_dim", d.gdn_value_dim)));
  d.gdn_conv_kernel = static_cast<int>(f.kv_int("qwen35moe.ssm.conv_kernel",
                              f.kv_int("sager.gdn.conv_kernel", d.gdn_conv_kernel)));
  if (d.n_gdn_v_heads % d.n_gdn_k_heads != 0 || d.gdn_conv_kernel < 2 ||
      d.gdn_key_dim <= 0 || d.gdn_value_dim <= 0) {
    SLOG_E("config", "invalid GDN geometry (k=%d v=%d K=%d V=%d ck=%d)",
           d.n_gdn_k_heads, d.n_gdn_v_heads, d.gdn_key_dim, d.gdn_value_dim,
           d.gdn_conv_kernel);
    return Status::kGeometryMismatch;
  }

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
