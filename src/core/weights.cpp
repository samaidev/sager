// weights.cpp - residency planning + the one-pass loader.
//
// Attention/router/norms/embedding go to VRAM in their native GGUF dtype
// (Q4_K stays Q4_K: T4 decode is bandwidth-bound, so shipping fewer bytes is
// a win even with dequant fused into the GEMV). Routed experts are copied
// into the pinned host pool (tier 1). Shared experts go to VRAM (tier 0).
#include "sager/core/weights.hpp"

#include <cstring>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

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
  const size_t Q = static_cast<size_t>(d.n_head * d.head_dim);
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
    // The q carve always reserves 2*Q rows: qwen3.5-style gated attention
    // stores [q | gate] there; legacy files use the first Q rows only.
    if (linear) {
      // carve: in_qkvz | in_ba | out_proj
      b += q4bytes(static_cast<size_t>(g.proj_rows()), H);
      b += q4bytes(static_cast<size_t>(g.ba_rows()), H);
      b += q4bytes(H, static_cast<size_t>(g.y_rows()));
    } else {
      b += q4bytes(2 * Q, H);                  // wq + attn_gate
      b += q4bytes(KV, H) * 2;                 // wk, wv
      b += q4bytes(H, Q);                      // wo
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
      lw.wq = reinterpret_cast<uint16_t*>(p);        p += q4bytes(Q, H);
      lw.attn_gate = reinterpret_cast<uint16_t*>(p); p += q4bytes(Q, H);
      lw.wk = reinterpret_cast<uint16_t*>(p);        p += q4bytes(KV, H);
      lw.wv = reinterpret_cast<uint16_t*>(p);        p += q4bytes(KV, H);
      lw.wo = reinterpret_cast<uint16_t*>(p);
    }
    // norms: fp16
    void *n1 = nullptr, *n2 = nullptr;
    if (dmalloc(&n1, H * 2) != Status::kOk) return Status::kOomVram;
    lw.attn_norm = static_cast<uint16_t*>(n1);
    if (dmalloc(&n2, H * 2) != Status::kOk) return Status::kOomVram;
    lw.ffn_norm = static_cast<uint16_t*>(n2);
    // per-head q/k RMSNorm weights (full-attention layers; 512 B, negligible)
    void* qn = nullptr; void* kn = nullptr;
    if (dmalloc(&qn, static_cast<size_t>(d.head_dim) * 2) != Status::kOk ||
        dmalloc(&kn, static_cast<size_t>(d.head_dim) * 2) != Status::kOk)
      return Status::kOomVram;
    lw.attn_q_norm = static_cast<uint16_t*>(qn);
    lw.attn_k_norm = static_cast<uint16_t*>(kn);
    // router fp32 + shared-expert scalar router
    void* r = nullptr;
    if (dmalloc(&r, static_cast<size_t>(d.n_expert) * H * 4) != Status::kOk)
      return Status::kOomVram;
    lw.router = static_cast<float*>(r);
    void* sgw = nullptr;
    if (dmalloc(&sgw, H * 4) != Status::kOk) return Status::kOomVram;
    lw.shared_gate_w = static_cast<float*>(sgw);
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
      void* db = nullptr; void* on = nullptr;
      if (dmalloc(&cw, static_cast<size_t>(g.conv_dim()) * g.conv_kernel * 2) != Status::kOk ||
          dmalloc(&cb, static_cast<size_t>(g.conv_dim()) * 2) != Status::kOk ||
          dmalloc(&al, static_cast<size_t>(g.n_v_heads) * 4) != Status::kOk ||
          dmalloc(&db, static_cast<size_t>(g.n_v_heads) * 4) != Status::kOk ||
          dmalloc(&on, static_cast<size_t>(g.value_dim) * 2) != Status::kOk)
        return Status::kOomVram;
      lw.lin_conv_w  = static_cast<uint16_t*>(cw);
      lw.lin_conv_b  = static_cast<uint16_t*>(cb);
      lw.lin_A_log   = static_cast<float*>(al);   // holds ssm_a (pre-negated)
      lw.lin_dt_bias = static_cast<float*>(db);
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
  // The head is allocated in copy_attention(): Q4_K files keep the native
  // quant, Q6_K files are dequantized to fp16 (gemv_f16 path).
  return Status::kOk;
#else
  (void)cfg;
  return Status::kUnsupported;  // engine requires CUDA in v0.1
#endif
}

// dequant_row_q6 - host Q6_K -> fp32 for one row (cols = multiple of 256).
// block_q6_K layout (210 B / 256 weights): ql[128] | qh[64] | scales int8[16]
// | d fp16 - byte-identical to llama.cpp's block_q6_K.
void dequant_row_q6(const uint8_t* src, float* out, int cols) {
  const int nb = cols / 256;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* ql = src + static_cast<size_t>(b) * 210;
    const uint8_t* qh = ql + 128;
    const int8_t*  sc = reinterpret_cast<const int8_t*>(qh + 64);
    uint16_t dh; std::memcpy(&dh, qh + 64, 2);
    const float dd = q4k::half_to_float(dh);
    for (int n = 0; n < 2; ++n) {          // two 128-weight halves
      float* y = out + static_cast<size_t>(b) * 256 + n * 128;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = static_cast<int>((ql[l +  0] & 0xF) |
                                        (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[l + 32] & 0xF) |
                                        (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[l +  0] >> 4) |
                                        (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[l + 32] >> 4) |
                                        (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l +  0] = dd * sc[n * 8 + is + 0] * q1;
        y[l + 32] = dd * sc[n * 8 + is + 2] * q2;
        y[l + 64] = dd * sc[n * 8 + is + 4] * q3;
        y[l + 96] = dd * sc[n * 8 + is + 6] * q4;
      }
    }
  }
}

Status Weights::copy_attention(gguf::File& f, const ModelConfig& cfg) {
#if defined(SAGER_HAVE_CUDA)
  const EngineDims& d = dims;
  auto copyq4 = [&](void* dst, const gguf::TensorInfo* t) -> Status {
    if (!t) return Status::kGgufParseError;
    if (t->type != 12) return Status::kGgufParseError;
    SAGER_CUDA_CHECK(cudaMemcpy(dst, f.tensor_data(*t), t->nbytes,
                                cudaMemcpyHostToDevice));
    return Status::kOk;
  };
  // Requant a Q6_K tensor into a device Q4_K buffer, row by row.
  auto copyq6_as_q4 = [&](void* dst, const gguf::TensorInfo* t,
                          const char* what) -> Status {
    if (!t || t->type != 14) return Status::kGgufParseError;
    const int rows = static_cast<int>(t->dims[1]);
    const int cols = static_cast<int>(t->dims[0]);
    const size_t row_bytes = static_cast<size_t>(t->row_bytes);
    const size_t blocks_per_row = static_cast<size_t>(cols) / q4k::kQK_K;
    std::vector<q4k::BlockQ4K> out(static_cast<size_t>(rows) * blocks_per_row);
    std::vector<float> row(cols);
    const uint8_t* src = static_cast<const uint8_t*>(f.tensor_data(*t));
    for (int r = 0; r < rows; ++r) {
      dequant_row_q6(src + static_cast<size_t>(r) * row_bytes, row.data(), cols);
      auto blocks = q4k::quantize_row(row.data(), cols);
      std::memcpy(out.data() + static_cast<size_t>(r) * blocks_per_row,
                  blocks.data(), blocks_per_row * q4k::kBlockBytes);
      if ((r & 1023) == 0)
        SLOG_I("weights", "requant Q6_K->Q4_K %s: %d/%d rows", what, r, rows);
    }
    SAGER_CUDA_CHECK(cudaMemcpy(dst, out.data(),
                                out.size() * q4k::kBlockBytes,
                                cudaMemcpyHostToDevice));
    return Status::kOk;
  };
  // Matrix copy that accepts Q4_K (direct) or Q6_K (dequant + requant).
  auto copymat = [&](void* dst, const gguf::TensorInfo* t,
                     const char* what) -> Status {
    if (!t) return Status::kGgufParseError;
    if (t->type == 12) return copyq4(dst, t);
    if (t->type == 14) return copyq6_as_q4(dst, t, what);
    return Status::kGgufParseError;
  };
  // fp32 -> fp16 upload
  auto copyf32_as_f16 = [&](void* dst, const gguf::TensorInfo* t) -> Status {
    if (!t) return Status::kGgufParseError;
    const size_t n = t->nbytes / 4;
    std::vector<float> tmp(n);
    std::memcpy(tmp.data(), f.tensor_data(*t), n * 4);
    std::vector<uint16_t> h(n);
    f32_to_f16_rows(tmp.data(), h.data(), n);
    SAGER_CUDA_CHECK(cudaMemcpy(dst, h.data(), n * 2, cudaMemcpyHostToDevice));
    return Status::kOk;
  };
  // norms: fp16 native or fp32 -> fp16
  auto copynorm = [&](void* dst, const gguf::TensorInfo* t) -> Status {
    if (!t) return Status::kGgufParseError;
    if (t->type == 1) {
      SAGER_CUDA_CHECK(cudaMemcpy(dst, f.tensor_data(*t), t->nbytes,
                                  cudaMemcpyHostToDevice));
      return Status::kOk;
    }
    if (t->type == 0) return copyf32_as_f16(dst, t);
    return Status::kGgufParseError;
  };
  const GdnParams g = cfg.gdn();
  const int nk = g.n_k_heads;
  const int nv = g.n_v_heads;
  const int K = g.key_dim;
  const int V = g.value_dim;
  const int Qrows = d.n_head * d.head_dim;
  const int H = d.hidden;

  for (int L = 0; L < d.n_layer; ++L) {
    LayerWeights& lw = layers_[L];
    const std::string lp = "blk." + std::to_string(L) + ".";
    // ---- input + post-attention norms -----------------------------------
    if (const auto* t = f.tensor((lp + "attn_norm").c_str()))
      if (copynorm(lw.attn_norm, t) != Status::kOk) return Status::kGgufParseError;
    const auto* ffn_t = f.tensor((lp + "ffn_norm").c_str());
    if (!ffn_t) ffn_t = f.tensor((lp + "post_attention_norm").c_str());
    if (ffn_t)
      if (copynorm(lw.ffn_norm, ffn_t) != Status::kOk)
        return Status::kGgufParseError;

    Status s = Status::kOk;
    if (cfg.is_linear(L)) {
      // ---- gated delta net layer ----------------------------------------
      // Native qwen35moe layout: attn_qkv [conv_dim x H] + attn_gate
      // [nv*V x H] row-concatenated into the fused [q|k|v|z] carve; the
      // legacy converter layout is a single lin_in_qkvz [proj_rows x H].
      const auto* qkv = f.tensor((lp + "attn_qkv.weight").c_str());
      const auto* gate = f.tensor((lp + "attn_gate.weight").c_str());
      if (qkv && gate && qkv->dims.size() >= 2 &&
          static_cast<int>(qkv->dims[1]) == 2 * nk * K + nv * V &&
          static_cast<int>(gate->dims[1]) == nv * V) {
        s = copymat(lw.lin_in_qkvz, qkv, "attn_qkv");
        if (s != Status::kOk) return s;
        // row offset of the z slice inside the fused carve (uint16 units)
        auto* zdst = reinterpret_cast<uint16_t*>(lw.lin_in_qkvz) +
                     static_cast<size_t>(2 * nk * K) *
                         (static_cast<size_t>(H) / q4k::kQK_K) *
                         (q4k::kBlockBytes / sizeof(uint16_t));
        s = copymat(zdst, gate, "attn_gate");
        if (s != Status::kOk) return s;
      } else {
        s = copymat(lw.lin_in_qkvz,
                    f.tensor((lp + "lin_in_qkvz").c_str()), "lin_in_qkvz");
        if (s != Status::kOk) return s;
      }
      // ba: native = ssm_beta | ssm_alpha (per v-head) row-concatenated;
      // legacy = one lin_in_ba tensor.
      const auto* beta = f.tensor((lp + "ssm_beta.weight").c_str());
      const auto* alpha = f.tensor((lp + "ssm_alpha.weight").c_str());
      if (beta && alpha && beta->type == 12 && alpha->type == 12) {
        s = copyq4(lw.lin_in_ba, beta);
        if (s != Status::kOk) return s;
        auto* adst = reinterpret_cast<uint16_t*>(lw.lin_in_ba) +
                     static_cast<size_t>(nv) *
                         (static_cast<size_t>(H) / q4k::kQK_K) *
                         (q4k::kBlockBytes / sizeof(uint16_t));
        s = copyq4(adst, alpha);
        if (s != Status::kOk) return s;
      } else {
        s = copymat(lw.lin_in_ba,
                    f.tensor((lp + "lin_in_ba").c_str()), "lin_in_ba");
        if (s != Status::kOk) return s;
      }
      const auto* out_t = f.tensor((lp + "ssm_out.weight").c_str());
      if (!out_t) out_t = f.tensor((lp + "lin_out").c_str());
      s = copymat(lw.lin_out, out_t, "lin_out");
      if (s != Status::kOk) return s;
      // conv taps: native ssm_conv1d is fp32 [ck x conv_dim] (row = taps);
      // legacy lin_conv_w is fp16.
      if (const auto* t = f.tensor((lp + "ssm_conv1d.weight").c_str())) {
        if (t->type != 0) return Status::kGgufParseError;
        s = copyf32_as_f16(lw.lin_conv_w, t);
        if (s != Status::kOk) return s;
      } else if (const auto* t = f.tensor((lp + "lin_conv_w").c_str())) {
        if (t->type != 1) return Status::kGgufParseError;
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_conv_w, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
      if (const auto* t = f.tensor((lp + "ssm_conv1d.bias").c_str())) {
        s = copyf32_as_f16(lw.lin_conv_b, t);
        if (s != Status::kOk) return s;
      } else if (const auto* t = f.tensor((lp + "lin_conv_b").c_str())) {
        if (t->type == 1) {
          SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_conv_b, f.tensor_data(*t),
                                      t->nbytes, cudaMemcpyHostToDevice));
        }
      } else {
        cudaMemset(lw.lin_conv_b, 0,
                   static_cast<size_t>(g.conv_dim()) * sizeof(uint16_t));
      }
      // gate params: A (pre-negated) + dt_bias, fp32, per v-head
      if (const auto* t = f.tensor((lp + "ssm_a").c_str())) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_A_log, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else if (const auto* t = f.tensor((lp + "lin_A_log").c_str())) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_A_log, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
      if (const auto* t = f.tensor((lp + "ssm_dt.bias").c_str())) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_dt_bias, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else if (const auto* t = f.tensor((lp + "lin_dt_bias").c_str())) {
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_dt_bias, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
      // output-gate norm: native ssm_norm fp32; legacy lin_o_norm fp16
      if (const auto* t = f.tensor((lp + "ssm_norm.weight").c_str())) {
        s = copyf32_as_f16(lw.lin_o_norm, t);
        if (s != Status::kOk) return s;
      } else if (const auto* t = f.tensor((lp + "lin_o_norm").c_str())) {
        if (t->type != 1) return Status::kGgufParseError;
        SAGER_CUDA_CHECK(cudaMemcpy(lw.lin_o_norm, f.tensor_data(*t),
                                    t->nbytes, cudaMemcpyHostToDevice));
      } else {
        return Status::kGgufParseError;
      }
    } else {
      // ---- full attention (GQA) layer -----------------------------------
      const auto* q = f.tensor((lp + "attn_q.weight").c_str());
      if (!q) return Status::kGgufParseError;
      const int q_rows = q->dims.size() >= 2
                             ? static_cast<int>(q->dims[1]) : Qrows;
      s = copymat(lw.wq, q, "attn_q");
      if (s != Status::kOk) return s;
      // qwen3.5 gated attention: the q tensor carries [q | gate]; the full
      // 2Q-row copy above already filled the gate half of the carve.
      lw.attn_gated = (q_rows == 2 * Qrows);
      s = copymat(lw.wk, f.tensor((lp + "attn_k.weight").c_str()), "attn_k");
      if (s != Status::kOk) return s;
      s = copymat(lw.wv, f.tensor((lp + "attn_v.weight").c_str()), "attn_v");
      if (s != Status::kOk) return s;
      s = copymat(lw.wo, f.tensor((lp + "attn_output.weight").c_str()),
                  "attn_output");
      if (s != Status::kOk) return s;
      // per-head q/k norms (fp32 in the file -> fp16)
      if (const auto* t = f.tensor((lp + "attn_q_norm.weight").c_str())) {
        s = copynorm(lw.attn_q_norm, t);
        if (s != Status::kOk) return s;
      }
      if (const auto* t = f.tensor((lp + "attn_k_norm.weight").c_str())) {
        s = copynorm(lw.attn_k_norm, t);
        if (s != Status::kOk) return s;
      }
    }
    // router (fp32)
    if (const auto* t = f.tensor((lp + "ffn_gate_inp").c_str())) {
      SAGER_CUDA_CHECK(cudaMemcpy(lw.router, f.tensor_data(*t), t->nbytes,
                                  cudaMemcpyHostToDevice));
    }
    // shared-expert scalar router (fp32, one logit per token)
    if (const auto* t = f.tensor((lp + "ffn_gate_inp_shexp.weight").c_str())) {
      SAGER_CUDA_CHECK(cudaMemcpy(lw.shared_gate_w, f.tensor_data(*t),
                                  t->nbytes, cudaMemcpyHostToDevice));
    }
    // shared experts (Q4_K; down may be Q6_K in Q4_K_M builds)
    if (d.shared_expert > 0) {
      s = copymat(lw.shared_gate,
                  f.tensor((lp + "ffn_gate_shexp.weight").c_str()),
                  "ffn_gate_shexp");
      if (s != Status::kOk) return s;
      s = copymat(lw.shared_up,
                  f.tensor((lp + "ffn_up_shexp.weight").c_str()),
                  "ffn_up_shexp");
      if (s != Status::kOk) return s;
      s = copymat(lw.shared_out,
                  f.tensor((lp + "ffn_down_shexp.weight").c_str()),
                  "ffn_down_shexp");
      if (s != Status::kOk) return s;
    }
  }
  // embedding (fp16 / bf16 / f32 / Q4_K -> fp16)
  if (const auto* t = f.tensor("token_embd.weight")) {
    if (t->type == 1) {
      SAGER_CUDA_CHECK(cudaMemcpy(tok_embd, f.tensor_data(*t), t->nbytes,
                                  cudaMemcpyHostToDevice));
    } else if (t->type == 12) {
      // Q4_K embedding: dequant row by row into the fp16 table.
      const int rows = static_cast<int>(t->dims[1]);
      const int cols = static_cast<int>(t->dims[0]);
      const size_t row_bytes = static_cast<size_t>(t->row_bytes);
      const size_t blocks_per_row = static_cast<size_t>(cols) / q4k::kQK_K;
      std::vector<uint16_t> out(static_cast<size_t>(rows) * cols);
      std::vector<float> row(cols);
      const uint8_t* src = static_cast<const uint8_t*>(f.tensor_data(*t));
      for (int r = 0; r < rows; ++r) {
        q4k::dequant_row(reinterpret_cast<const q4k::BlockQ4K*>(
                             src + static_cast<size_t>(r) * row_bytes),
                         row.data(), cols);
        f32_to_f16_rows(row.data(),
                        out.data() + static_cast<size_t>(r) * cols, cols);
        if ((r & 8191) == 0)
          SLOG_I("weights", "embedding Q4_K->F16: %d/%d rows", r, rows);
      }
      SAGER_CUDA_CHECK(cudaMemcpy(tok_embd, out.data(),
                                  out.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice));
    } else {
      // BF16/F32 -> fp16 on host then upload
      const size_t n = t->nbytes / (t->type == 0 ? 4 : 2);
      std::vector<float> tmp(n);
      if (t->type == 0) std::memcpy(tmp.data(), f.tensor_data(*t), n * 4);
      else {
        const uint16_t* bf = reinterpret_cast<const uint16_t*>(f.tensor_data(*t));
        for (size_t i = 0; i < n; ++i) {
          uint32_t bits = static_cast<uint32_t>(bf[i]) << 16;
          std::memcpy(&tmp[i], &bits, 4);
        }
      }
      std::vector<uint16_t> h(n);
      f32_to_f16_rows(tmp.data(), h.data(), n);
      SAGER_CUDA_CHECK(cudaMemcpy(tok_embd, h.data(), n * 2,
                                  cudaMemcpyHostToDevice));
    }
  }
  if (const auto* t = f.tensor("output_norm.weight")) {
    if (copynorm(output_norm, t) != Status::kOk) return Status::kGgufParseError;
  }
  // output head: Q4_K native, or Q6_K dequantized to fp16.
  if (const auto* t = f.tensor("output.weight")) {
    if (t->type == 12) {
      void* p = nullptr;
      if (cudaMalloc(&p, t->nbytes) != cudaSuccess) return Status::kOomVram;
      output = static_cast<uint16_t*>(p);
      SAGER_CUDA_CHECK(cudaMemcpy(p, f.tensor_data(*t), t->nbytes,
                                  cudaMemcpyHostToDevice));
      head_type_ = 12;
      head_rows_ = static_cast<int>(t->dims[1]);
    } else if (t->type == 14) {
      const int rows = static_cast<int>(t->dims[1]);
      const int cols = static_cast<int>(t->dims[0]);
      void* p = nullptr;
      if (cudaMalloc(&p, static_cast<size_t>(rows) * cols * 2) != cudaSuccess)
        return Status::kOomVram;
      output = static_cast<uint16_t*>(p);
      const size_t row_bytes = static_cast<size_t>(t->row_bytes);
      std::vector<uint16_t> out(static_cast<size_t>(rows) * cols);
      std::vector<float> row(cols);
      const uint8_t* src = static_cast<const uint8_t*>(f.tensor_data(*t));
      for (int r = 0; r < rows; ++r) {
        dequant_row_q6(src + static_cast<size_t>(r) * row_bytes, row.data(),
                       cols);
        f32_to_f16_rows(row.data(),
                        out.data() + static_cast<size_t>(r) * cols, cols);
        if ((r & 8191) == 0)
          SLOG_I("weights", "head Q6_K->F16: %d/%d rows", r, rows);
      }
      SAGER_CUDA_CHECK(cudaMemcpy(p, out.data(), out.size() * 2,
                                  cudaMemcpyHostToDevice));
      head_type_ = 1;
      head_rows_ = rows;
    } else {
      return Status::kGgufParseError;
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
  // SAGER_EXPERT_POOL_SWAP=1: back the pool with swappable (non-pinned)
  // memory instead of page-locked RAM. For boxes whose RAM is smaller than
  // the full expert set (e.g. a 12 GB Colab against a 17 GB expert pool):
  // the kernel pages cold experts out, the pump faults them back on demand.
  // Slower than pinned RAM, but it keeps every address stable.
  const auto pool_file_opt = env_get("SAGER_EXPERT_POOL_FILE");
  if (pool_file_opt) {
    const char* pool_file = pool_file_opt->c_str();
#ifndef _WIN32
    // File-backed pool (SAGER_EXPERT_POOL_FILE=<path>): mmap a disk file
    // MAP_SHARED - the kernel pages it in/out through the page cache, so a
    // box whose RAM is smaller than the expert set still loads (no swap
    // needed). Cold experts fault back from the file on demand.
    const int fd = ::open(pool_file, O_RDWR | O_CREAT, 0644);
    if (fd >= 0 && ::ftruncate(fd, static_cast<off_t>(expert_pool_bytes_)) == 0) {
      void* m = ::mmap(nullptr, expert_pool_bytes_, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
      if (m != MAP_FAILED) {
        expert_pool_host_ = static_cast<uint8_t*>(m);
        pool_map_fd_ = fd;
        pool_map_size_ = expert_pool_bytes_;
        SLOG_I("weights", "expert pool backing: file %s (%.1f GB, page cache)",
               pool_file, expert_pool_bytes_ / 1073741824.0);
      }
    }
    if (!expert_pool_host_ && fd >= 0) ::close(fd);
#endif
  }
  if (!expert_pool_host_) {
    const bool pinned = !env_flag("SAGER_EXPERT_POOL_SWAP");
    expert_pool_host_ = static_cast<uint8_t*>(
        host_alloc(expert_pool_bytes_, pinned));
    if (!expert_pool_host_) return Status::kOomRam;
    SLOG_I("weights", "expert pool backing: %s (%.1f GB)",
           pinned ? "pinned RAM" : "swappable (SAGER_EXPERT_POOL_SWAP=1)",
           expert_pool_bytes_ / 1073741824.0);
  }

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
      if (g->type != 12 || u->type != 12) return Status::kGgufParseError;
      if (dnw->type != 12 && dnw->type != 14) return Status::kGgufParseError;
      // expert-major slices of the 3D stacked tensors [n_expert, rows, cols]
      const uint8_t* gb = f.tensor_data(*g) + static_cast<size_t>(e) * gu;
      const uint8_t* ub = f.tensor_data(*u) + static_cast<size_t>(e) * gu;
      std::memcpy(dst, gb, gu);
      std::memcpy(dst + gu, ub, gu);
      uint8_t* dbdst = dst + gu * 2;
      if (dnw->type == 12) {
        const uint8_t* db = f.tensor_data(*dnw) + static_cast<size_t>(e) * dn;
        std::memcpy(dbdst, db, dn);
      } else {
        // Q6_K down projection: dequant + requant this expert's slice
        const int rows = static_cast<int>(d.hidden);
        const int cols = static_cast<int>(d.n_ff_exp);
        const size_t row_bytes = static_cast<size_t>(dnw->row_bytes);
        const size_t blocks_per_row = static_cast<size_t>(cols) / q4k::kQK_K;
        const uint8_t* src = static_cast<const uint8_t*>(f.tensor_data(*dnw)) +
                             static_cast<size_t>(e) * row_bytes * rows;
        std::vector<float> row(cols);
        for (int r = 0; r < rows; ++r) {
          dequant_row_q6(src + static_cast<size_t>(r) * row_bytes,
                         row.data(), cols);
          auto blocks = q4k::quantize_row(row.data(), cols);
          std::memcpy(dbdst + static_cast<size_t>(r) * blocks_per_row *
                          q4k::kBlockBytes,
                      blocks.data(), blocks_per_row * q4k::kBlockBytes);
        }
      }
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
    // The attention mats (wq + gate + k + v + o) are carved from ONE
    // cudaMalloc - free via wq. GDN mats carve via lin_in_qkvz.
    if (lw.wq) { cudaFree(lw.wq); lw.wq = nullptr; lw.attn_gate = nullptr; }
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
    if (lw.lin_o_norm)  { cudaFree(lw.lin_o_norm);  lw.lin_o_norm = nullptr; }
    if (lw.attn_q_norm) { cudaFree(lw.attn_q_norm); lw.attn_q_norm = nullptr; }
    if (lw.attn_k_norm) { cudaFree(lw.attn_k_norm); lw.attn_k_norm = nullptr; }
    if (lw.shared_gate_w) { cudaFree(lw.shared_gate_w); lw.shared_gate_w = nullptr; }
  }
  layers_.clear();
  if (tok_embd) { cudaFree(tok_embd); tok_embd = nullptr; }
  if (output_norm) { cudaFree(output_norm); output_norm = nullptr; }
  if (output) { cudaFree(output); output = nullptr; }
  if (mtp_proj_in) { cudaFree(mtp_proj_in); mtp_proj_in = nullptr; }
  if (mtp_proj_out) { cudaFree(mtp_proj_out); mtp_proj_out = nullptr; }
#endif
  if (expert_pool_host_) {
#ifndef _WIN32
    if (pool_map_fd_ >= 0) {
      ::munmap(expert_pool_host_, pool_map_size_);
      ::close(pool_map_fd_);
      pool_map_fd_ = -1;
      pool_map_size_ = 0;
    } else
#endif
    {
      const bool was_pinned = !env_flag("SAGER_EXPERT_POOL_FILE");
      host_free(expert_pool_host_, was_pinned);
    }
    expert_pool_host_ = nullptr;
  }
  expert_pool_bytes_ = 0;
}

}  // namespace sager
