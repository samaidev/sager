// weights.cpp - residency planning + the one-pass loader.
//
// Attention/router/norms/embedding go to VRAM in their native GGUF dtype
// (Q4_K stays Q4_K: T4 decode is bandwidth-bound, so shipping fewer bytes is
// a win even with dequant fused into the GEMV). Routed experts are copied
// into the pinned host pool (tier 1). Shared experts go to VRAM (tier 0).
#include "sager/core/weights.hpp"

#include <cstring>
#include <string>

#include "sager/core/config.hpp"
#include "sager/kernels/gdn.hpp"
#include "sager/kernels/q4_k.hpp"
#include "sager/log.hpp"

namespace sager {

void f32_to_f16_rows(const float* src, uint16_t* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = q4k::float_to_half(src[i]);
}

namespace {
// GGUF naming, qwen3-family converters:
//   blk.{i}.attn_norm, attn_q/k/v/o.weight, ffn_norm,
//   ffn_gate_inp (router), ffn_gate_exps / ffn_up_exps / ffn_down_exps,
//   ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp (shared),
//   token_embd.weight, output_norm.weight, output.weight (tied if absent)
std::string cat(const char* a, int i, const char* b) {
  return std::string(a) + std::to_string(i) + b;
}
}  // namespace

Status Weights::alloc_vram(const ModelConfig& cfg) {
  const EngineDims& d = dims;
  layers_.assign(d.n_layer, LayerWeights{});

#if defined(SAGER_HAVE_CUDA)
  auto dmalloc = [](void** p, size_t bytes) -> Status {
    if (cudaMalloc(p, bytes) != cudaSuccess) return Status::kOomVram;
    return Status::kOk;
  };
  const size_t H = static_cast<size_t>(d.hidden);
  const size_t QKV = static_cast<size_t>(d.n_head * d.head_dim);
  const size_t KV = static_cast<size_t>(d.n_embd_gqa());
  // Q4_K bytes per matrix
  auto q4bytes = [](size_t rows, size_t cols) {
    return rows * (cols / q4k::kQK_K) * q4k::kBlockBytes;
  };
  const GdnParams g = cfg.gdn();

  for (int L = 0; L < d.n_layer; ++L) {
    LayerWeights& lw = layers_[L];
    size_t b = 0;
    bool linear = cfg.is_linear(L);
    if (linear) {
      // carve: in_qkvz | in_ba | out_proj
      b += q4bytes(static_cast<size_t>(g.proj_rows()), H);
      b += q4bytes(static_cast<size_t>(g.ba_rows()), H);
      b += q4bytes(H, static_cast<size_t>(g.y_rows()));
    } else {
      b += q4bytes(QKV, H);                    // wq
      b += q4bytes(KV, H) * 2;                 // wk, wv
      b += q4bytes(H, QKV);                    // wo
    }
    void* base = nullptr;
    if (cudaMalloc(&base, b) != cudaSuccess) return Status::kOomVram;
    uint8_t* p = static_cast<uint8_t*>(base);
    if (linear) {
      lw.lin_in_qkvz = reinterpret_cast<uint16_t*>(p);
      p += q4bytes(static_cast<size_t>(g.proj_rows()), H);
      lw.lin_in_ba = reinterpret_cast<uint16_t*>(p);
      p += q4bytes(static_cast<size_t>(g.ba_rows()), H);
      lw.lin_out = reinterpret_cast<uint16_t*>(p);
    } else {
      lw.wq = reinterpret_cast<uint16_t*>(p);  p += q4bytes(QKV, H);
      lw.wk = reinterpret_cast<uint16_t*>(p);  p += q4bytes(KV, H);
      lw.wv = reinterpret_cast<uint16_t*>(p);  p += q4bytes(KV, H);
      lw.wo = reinterpret_cast<uint16_t*>(p);
    }
    // norms: fp16
    void *n1 = nullptr, *n2 = nullptr;
    if (dmalloc(&n1, H * 2) != Status::kOk) return Status::kOomVram;
    lw.attn_norm = static_cast<uint16_t*>(n1);
    if (dmalloc(&n2, H * 2) != Status::kOk) return Status::kOomVram;
    lw.ffn_norm = static_cast<uint16_t*>(n2);
    // router fp32
    void* r = nullptr;
    if (dmalloc(&r, static_cast<size_t>(d.n_expert) * H * 4) != Status::kOk)
      return Status::kOomVram;
    lw.router = static_cast<float*>(r);
    // shared experts Q4_K
    const size_t sh = q4bytes(static_cast<size_t>(d.n_ff_exp), H) * 2 +
                      q4bytes(H, static_cast<size_t>(d.n_ff_exp));
    void* s1 = nullptr; void* s2 = nullptr; void* s3 = nullptr;
    if (d.shared_expert > 0) {
      if (dmalloc(&s1, sh) != Status::kOk || dmalloc(&s2, sh) != Status::kOk ||
          dmalloc(&s3, sh) != Status::kOk)
        return Status::kOomVram;
      lw.shared_gate = static_cast<uint16_t*>(s1);
      lw.shared_up   = static_cast<uint16_t*>(s2);
      lw.shared_out  = static_cast<uint16_t*>(s3);
    }
    // ---- GDN side tensors (linear layers only) ----
    if (linear) {
      void* cw = nullptr; void* cb = nullptr; void* al = nullptr;
      void* db = nullptr; void* qn = nullptr; void* kn = nullptr;
      void* on = nullptr;
      if (dmalloc(&cw, static_cast<size_t>(g.conv_dim()) * g.conv_kernel * 2) != Status::kOk ||
          dmalloc(&cb, static_cast<size_t>(g.conv_dim()) * 2) != Status::kOk ||
          dmalloc(&al, static_cast<size_t>(g.n_k_heads) * 4) != Status::kOk ||
          dmalloc(&db, static_cast<size_t>(g.n_k_heads) * 4) != Status::kOk ||
          dmalloc(&qn, static_cast<size_t>(g.key_dim) * 2) != Status::kOk ||
          dmalloc(&kn, static_cast<size_t>(g.key_dim) * 2) != Status::kOk ||
          dmalloc(&on, static_cast<size_t>(g.value_dim) * 2) != Status::kOk)
        return Status::kOomVram;
      lw.lin_conv_w  = static_cast<uint16_t*>(cw);
      lw.lin_conv_b  = static_cast<uint16_t*>(cb);
      lw.lin_A_log   = static_cast<float*>(al);
      lw.lin_dt_bias = static_cast<float*>(db);
      lw.lin_q_norm  = static_cast<uint16_t*>(qn);
      lw.lin_k_norm  = static_cast<uint16_t*>(kn);
      lw.lin_o_norm  = static_cast<uint16_t*>(on);
    }
  }

  // Embedding + head
  void* emb = nullptr;
  if (dmalloc(&emb, static_cast<size_t>(d.n_vocab) * H * 2) != Status::kOk)
    return Status::kOomVram;
  tok_embd = static_cast<uint16_t*>(emb);
  void* onorm = nullptr;
  if (dmalloc(&onorm, H * 2) != Status::kOk) return Status::kOomVram;
  output_norm = static_cast<uint16_t*>(onorm);
  // output head stays in its native quant (Q6_K from the converter) - no fp16
  // copy: the head GEMV reads Q4_K/Q6_K directly.
  return Status::kOk;
#else
  (void)cfg;
  return Status::kUnsupported;  // engine requires CUDA in v0.1
#endif
}

Status Weights::copy_attention(gguf::File& f, const ModelConfig& cfg) {
#if defined(SAGER_HAVE_CUDA)
  const EngineDims& d = dims;
  auto copyq4 = [&](void* dst, const gguf::TensorInfo* t) -> Status {
    if (!t) return Status::kGgufParseError;
    // The VRAM carve (alloc_vram) is sized for Q4_K (type 12) blocks only.
    // A Q6_K tensor would overflow the carve - refuse it loudly (converter
    // contract: attention/shared/router mats are Q4_K; see tools docs).
    if (t->type != 12) return Status::kGgufParseError;
    SAGER_CUDA_CHECK(cudaMemcpy(dst, f.tensor_data(*t), t->nbytes,
                                cudaMemcpyHostToDevice));
    return Status::kOk;
  };
  for (int L = 0; L < d.n_layer; ++L) {
    LayerWeights& lw = layers_[L];
    // norms (fp16 in GGUF for qwen3 converters; f32 -> convert)
    if (const auto* t = f.tensor(cat("blk.", L, ".attn_norm"))) {
      if (t->type == 1) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.attn_norm, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
      } else {
        std::vector<float> tmp(t->nbytes / 4);
        std::memcpy(tmp.data(), f.tensor_data(*t), tmp.size() * 4);
        std::vector<uint16_t> h(tmp.size());
        f32_to_f16_rows(tmp.data(), h.data(), tmp.size());
        SAGER_CUDA_CHECK(cudaMemcpy(lw.attn_norm, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
      }
    }
    if (const auto* t = f.tensor(cat("blk.", L, ".ffn_norm"))) {
      if (t->type == 1) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.ffn_norm, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
      }
    }
    Status s = Status::kOk;
    if (cfg.is_linear(L)) {
      // ---- gated delta net layer: fused qkvz + ba + out_proj (Q4_K) ----
      s = copyq4(lw.lin_in_qkvz, f.tensor(cat("blk.", L, ".lin_in_qkvz")));
      if (s != Status::kOk) return s;
      s = copyq4(lw.lin_in_ba, f.tensor(cat("blk.", L, ".lin_in_ba")));
      if (s != Status::kOk) return s;
      s = copyq4(lw.lin_out, f.tensor(cat("blk.", L, ".lin_out")));
      if (s != Status::kOk) return s;
      // conv taps (fp16) + optional bias
      if (const auto* t = f.tensor(cat("blk.", L, ".lin_conv_w"))) {
        if (t->type != 1) return Status::kGgufParseError;
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_conv_w, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
      if (const auto* t = f.tensor(cat("blk.", L, ".lin_conv_b"))) {
        if (t->type == 1) {
          SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_conv_b, f.tensor_data(*t),
                                      t->nbytes, cudaMemcpyHostToDevice));
        }
      }  // bias optional: kernels treat null as zeros? no - zero it here
      else {
        const GdnParams g = cfg.gdn();
        cudaMemset(lw.lin_conv_b, 0,
                   static_cast<size_t>(g.conv_dim()) * sizeof(uint16_t));
      }
      // A_log / dt_bias (fp32)
      if (const auto* t = f.tensor(cat("blk.", L, ".lin_A_log"))) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_A_log, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
      if (const auto* t = f.tensor(cat("blk.", L, ".lin_dt_bias"))) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_dt_bias, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
      // norms (fp16)
      auto copyf16 = [&](uint16_t* dst, const char* key) -> Status {
        const auto* t = f.tensor(key);
        if (!t || t->type != 1) return Status::kGgufParseError;
        SAGER_CUDA_CHECK(cudaMemcpy(dst, f.tensor_data(*t), t->nbytes,
                                    cudaMemcpyHostToDevice));
        return Status::kOk;
      };
      s = copyf16(lw.lin_q_norm, (cat("blk.", L, ".lin_q_norm")).c_str());
      if (s != Status::kOk) return s;
      s = copyf16(lw.lin_k_norm, (cat("blk.", L, ".lin_k_norm")).c_str());
      if (s != Status::kOk) return s;
      s = copyf16(lw.lin_o_norm, (cat("blk.", L, ".lin_o_norm")).c_str());
      if (s != Status::kOk) return s;
    } else {
    s = copyq4(lw.wq, f.tensor(cat("blk.", L, ".attn_q.weight")));   if (s != Status::kOk) return s;
    s = copyq4(lw.wk, f.tensor(cat("blk.", L, ".attn_k.weight")));   if (s != Status::kOk) return s;
    s = copyq4(lw.wv, f.tensor(cat("blk.", L, ".attn_v.weight")));   if (s != Status::kOk) return s;
    s = copyq4(lw.wo, f.tensor(cat("blk.", L, ".attn_output.weight"))); if (s != Status::kOk) return s;
    }
    // router (fp32)
    if (const auto* t = f.tensor(cat("blk.", L, ".ffn_gate_inp"))) {
      SAGER_CUDA_CHECK(cudaMemcpy(lw.router, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
    }
    // shared experts (Q4_K)
    if (d.shared_expert > 0) {
      s = copyq4(lw.shared_gate, f.tensor(cat("blk.", L, ".ffn_gate_shexp"))); if (s != Status::kOk) return s;
      s = copyq4(lw.shared_up,   f.tensor(cat("blk.", L, ".ffn_up_shexp")));   if (s != Status::kOk) return s;
      s = copyq4(lw.shared_out,  f.tensor(cat("blk.", L, ".ffn_down_shexp"))); if (s != Status::kOk) return s;
    }
  }
  // embedding (convert f32/bf16 -> fp16 if needed; most files ship f16/bf16)
  if (const auto* t = f.tensor("token_embd.weight")) {
    if (t->type == 1) {
      SAGER_CUDA_CHECK(cudaMemcpy(tok_embd, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
    } else {
      // BF16/F32 -> fp16 on host then upload (one-time cost, ~0.5 s)
      const size_t n = t->nbytes / (t->type == 0 ? 4 : 2);
      std::vector<float> tmp(n);
      if (t->type == 0) std::memcpy(tmp.data(), f.tensor_data(*t), n * 4);
      else {
        // bf16 -> f32
        const uint16_t* bf = reinterpret_cast<const uint16_t*>(f.tensor_data(*t));
        for (size_t i = 0; i < n; ++i) {
          uint32_t bits = static_cast<uint32_t>(bf[i]) << 16;
          std::memcpy(&tmp[i], &bits, 4);
        }
      }
      std::vector<uint16_t> h(n);
      f32_to_f16_rows(tmp.data(), h.data(), n);
      SAGER_CUDA_CHECK(cudaMemcpy(tok_embd, h.data(), n * 2, cudaMemcpyHostToDevice));
    }
  }
  if (const auto* t = f.tensor("output_norm.weight")) {
    if (t->type == 1) {
      SAGER_CUDA_CHECK(cudaMemcpy(output_norm, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
    }
  }
  // output head: uploaded to device as raw quant bytes.
  // v0.1 kernel surface is Q4_K-only, so a Q6_K head would silently produce
  // garbage logits - reject it here instead.
  if (const auto* t = f.tensor("output.weight")) {
    if (t->type == 12) {
      void* p = nullptr;
      if (cudaMalloc(&p, t->nbytes) != cudaSuccess) return Status::kOomVram;
      output = static_cast<uint16_t*>(p);
      SAGER_CUDA_CHECK(cudaMemcpy(p, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
      head_type_ = t->type;
      head_rows_ = static_cast<int>(t->dims[1]);
    } else {
      return Status::kGgufParseError;  // v0.1 contract: quantized head only
    }
  } else {
    return Status::kGgufParseError;    // untied head required
  }
  (void)cfg;
  return Status::kOk;
#else
  return Status::kUnsupported;
#endif
}

Status Weights::load_expert_pool(gguf::File& f, const ModelConfig& cfg) {
  const EngineDims& d = dims;
  const size_t ebytes = cfg.expert_bytes();
  expert_pool_bytes_ = static_cast<size_t>(d.experts_total()) * ebytes;
  expert_pool_host_ = static_cast<uint8_t*>(
      host_alloc(expert_pool_bytes_, /*pinned=*/true));
  if (!expert_pool_host_) return Status::kOomRam;

  auto tile = [&](size_t rows, size_t cols) { return rows * (cols / q4k::kQK_K) * q4k::kBlockBytes; };
  const size_t gu = tile(static_cast<size_t>(d.n_ff_exp), static_cast<size_t>(d.hidden));
  const size_t dn = tile(static_cast<size_t>(d.hidden), static_cast<size_t>(d.n_ff_exp));

  for (int L = 0; L < d.n_moe_layer; ++L) {
    for (int e = 0; e < d.n_expert; ++e) {
      uint8_t* dst = expert_pool_host_ +
                     (static_cast<size_t>(L) * d.n_expert + e) * ebytes;
      const auto* g = f.tensor("blk." + std::to_string(L) + ".ffn_gate_exps");
      const auto* u = f.tensor("blk." + std::to_string(L) + ".ffn_up_exps");
      const auto* dnw = f.tensor("blk." + std::to_string(L) + ".ffn_down_exps");
      if (!g || !u || !dnw) return Status::kGgufParseError;
      if (g->type != 12 || u->type != 12 || dnw->type != 12) return Status::kGgufParseError;
      // expert-major slices of the 3D stacked tensors [n_expert, rows, cols]
      const uint8_t* gb = f.tensor_data(*g) + static_cast<size_t>(e) * gu;
      const uint8_t* ub = f.tensor_data(*u) + static_cast<size_t>(e) * gu;
      const uint8_t* db = f.tensor_data(*dnw) + static_cast<size_t>(e) * dn;
      std::memcpy(dst, gb, gu);
      std::memcpy(dst + gu, ub, gu);
      std::memcpy(dst + gu * 2, db, dn);
    }
  }
  SLOG_I("weights", "expert pool: %d experts x %.2f MB = %.1f GB pinned host RAM",
         d.experts_total(), ebytes / 1048576.0, expert_pool_bytes_ / 1073741824.0);
  return Status::kOk;
}

Status Weights::load(gguf::File& f, const ModelConfig& cfg, size_t) {
#if defined(SAGER_HAVE_CUDA)
  dims = cfg.dims;
  Status s = alloc_vram(cfg);
  if (s != Status::kOk) return s;
  s = copy_attention(f, cfg);
  if (s != Status::kOk) return s;
  s = load_expert_pool(f, cfg);
  if (s != Status::kOk) return s;
  // MTP head (optional; drafter refuses politely if absent)
  mtp_present = false;
  if (const auto* t = f.tensor("mtp.fc.weight")) {
    void* p1 = nullptr; void* p2 = nullptr;
    if (cudaMalloc(&p1, t->nbytes) == cudaSuccess) {
      SAGER_CUDA_CHECK(cudaMemcpy(p1, f.tensor_data(*t), t->nbytes, cudaMemcpyHostToDevice));
      mtp_proj_in = static_cast<uint16_t*>(p1);
    }
    if (const auto* t2 = f.tensor("mtp.proj_out.weight")) {
      if (cudaMalloc(&p2, t2->nbytes) == cudaSuccess) {
        SAGER_CUDA_CHECK(cudaMemcpy(p2, f.tensor_data(*t2), t2->nbytes, cudaMemcpyHostToDevice));
        mtp_proj_out = static_cast<uint16_t*>(p2);
        mtp_present = true;
      }
    }
  }
  SLOG_I("weights", "load complete: MTP=%s", mtp_present ? "yes" : "no");
  return Status::kOk;
#else
  (void)f; (void)cfg;
  return Status::kUnsupported;
#endif
}

void Weights::unload() {
#if defined(SAGER_HAVE_CUDA)
  for (auto& lw : layers_) {
    // The 4 attention mats were carved from ONE cudaMalloc - free via wq.
    if (lw.wq) { cudaFree(lw.wq); lw.wq = nullptr; }
    if (lw.attn_norm) { cudaFree(lw.attn_norm); lw.attn_norm = nullptr; }
    if (lw.ffn_norm)  { cudaFree(lw.ffn_norm);  lw.ffn_norm = nullptr; }
    if (lw.router)    { cudaFree(lw.router);    lw.router = nullptr; }
    if (lw.shared_gate) { cudaFree(lw.shared_gate); lw.shared_gate = nullptr; }
    if (lw.shared_up)   { cudaFree(lw.shared_up);   lw.shared_up = nullptr; }
    if (lw.shared_out)  { cudaFree(lw.shared_out);  lw.shared_out = nullptr; }
    // GDN tensors: the 3 Q4 mats were carved from ONE cudaMalloc (lin_in_qkvz)
    if (lw.lin_in_qkvz) { cudaFree(lw.lin_in_qkvz); lw.lin_in_qkvz = nullptr; }
    if (lw.lin_conv_w)  { cudaFree(lw.lin_conv_w);  lw.lin_conv_w = nullptr; }
    if (lw.lin_conv_b)  { cudaFree(lw.lin_conv_b);  lw.lin_conv_b = nullptr; }
    if (lw.lin_A_log)   { cudaFree(lw.lin_A_log);   lw.lin_A_log = nullptr; }
    if (lw.lin_dt_bias) { cudaFree(lw.lin_dt_bias); lw.lin_dt_bias = nullptr; }
    if (lw.lin_q_norm)  { cudaFree(lw.lin_q_norm);  lw.lin_q_norm = nullptr; }
    if (lw.lin_k_norm)  { cudaFree(lw.lin_k_norm);  lw.lin_k_norm = nullptr; }
    if (lw.lin_o_norm)  { cudaFree(lw.lin_o_norm);  lw.lin_o_norm = nullptr; }
  }
  layers_.clear();
  if (tok_embd) { cudaFree(tok_embd); tok_embd = nullptr; }
  if (output_norm) { cudaFree(output_norm); output_norm = nullptr; }
  if (output) { cudaFree(output); output = nullptr; }
  if (mtp_proj_in) { cudaFree(mtp_proj_in); mtp_proj_in = nullptr; }
  if (mtp_proj_out) { cudaFree(mtp_proj_out); mtp_proj_out = nullptr; }
#endif
  if (expert_pool_host_) { host_free(expert_pool_host_, true); expert_pool_host_ = nullptr; }
  expert_pool_bytes_ = 0;
}

}  // namespace sager
