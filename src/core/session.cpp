// session.cpp - the session loop: owns decode CUDA graphs, the KV cache and
// the eager MoE dispatch half (hot-expert hits on VRAM via fused pointer-table
// GEMVs, misses on the CPU pool through the C1-C6 doorbell contract).
//
// Layout of one decode step (see session.hpp for the big picture):
//   embed -> [per layer: replay dense graph | eager MoE dispatch] -> head -> logits
//
// The dense graph body (rmsnorm -> QKV -> RoPE -> KV append -> attention ->
// o_proj -> residual -> ffn norm) is captured once per layer with fixed
// addresses; the per-step values (position, token) arrive through the
// d_params_ / d_pos_ids_ device blocks, so a replay is 48 launches -> 48
// graph launches with none of the per-kernel launch overhead that dominates
// small-batch Turing decode.
#include "layer_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

#include "sager/log.hpp"

#if defined(SAGER_HAVE_CUDA)
#include "sager/kernels/cuda_ops.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace sager {

#if defined(SAGER_HAVE_CUDA)
// Attention/head/shared-expert weights live in VRAM as raw Q4_K bytes
// (weights.cpp carves them that way); the LayerWeights members expose them
// as uint16_t* and the block type is restored at each launch site.
#define Q4W(p) reinterpret_cast<const q4k::BlockQ4K*>(p)
#endif

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------
Status KVCache::init(const Params& p) {
#if defined(SAGER_HAVE_CUDA)
  n_layer_ = p.n_layer;
  n_head_kv_ = p.n_head_kv;
  head_dim_ = p.head_dim;
  max_seq_ = p.max_seq;
  q8_ = p.q8;
  len_ = 0;
  const int kv_dim = n_head_kv_ * head_dim_;
  // q8: one 34-byte Q8_0 row per 32 values (scale folded in, see
  // attention.cu); fp16: 2 bytes per value.
  const size_t row_bytes = q8_ ? static_cast<size_t>(kv_dim / 32) * 34
                               : static_cast<size_t>(kv_dim) * 2;
  slab_bytes_ = static_cast<size_t>(max_seq_) * row_bytes;
  const size_t total = slab_bytes_ * static_cast<size_t>(n_layer_);
  k_.assign(static_cast<size_t>(n_layer_), nullptr);
  v_.assign(static_cast<size_t>(n_layer_), nullptr);
  uint8_t* kc = static_cast<uint8_t*>(host_alloc(0, false));  // (no-op warmup)
  (void)kc;
  void* ka = nullptr;
  void* va = nullptr;
  if (cudaMalloc(&ka, total) != cudaSuccess) return Status::kOomVram;
  if (cudaMalloc(&va, total) != cudaSuccess) { cudaFree(ka); return Status::kOomVram; }
  uint8_t* kbase = static_cast<uint8_t*>(ka);
  uint8_t* vbase = static_cast<uint8_t*>(va);
  for (int L = 0; L < n_layer_; ++L) {
    k_[static_cast<size_t>(L)] = kbase + static_cast<size_t>(L) * slab_bytes_;
    v_[static_cast<size_t>(L)] = vbase + static_cast<size_t>(L) * slab_bytes_;
  }
  SLOG_I("kv", "kv cache: %d layers x %.1f MB/slab (%s) = %.0f MB",
         n_layer_, slab_bytes_ / 1048576.0, q8_ ? "q8_0" : "f16",
         total * 2 / 1048576.0);
  return Status::kOk;
#else
  (void)p;
  return Status::kUnsupported;
#endif
}

void KVCache::reset() { len_ = 0; }

size_t KVCache::bytes() const {
#if defined(SAGER_HAVE_CUDA)
  return slab_bytes_ * static_cast<size_t>(n_layer_) * 2;
#else
  return 0;
#endif
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
#if defined(SAGER_HAVE_CUDA)

Session::Session(Weights& w, ExpertCache& cache, CpuPool& pool,
                 const ModelConfig& mc, const CtxParams& cp)
    : w_(w), cache_(cache), pool_(pool), mc_(mc), cp_(cp) {}

Session::~Session() {
  if (prof.on) {
    SLOG_I(
        "prof",
        "dec: steps=%ld upload=%.0f dense=%.0f moe=%.0f head=%.0f samp=%.0f ms | "
        "fw: calls=%ld toks=%ld upload=%.0f dense=%.0f moe=%.0f head=%.0f ms | "
        "fw-moe-path: device=%ld batched=%ld rowwalk=%ld | mtp: calls-by-time %.0fms | "
        "spec: windows=%ld drafted=%ld accepted=%ld",
        (long)prof.dec_steps, prof.dec_upload, prof.dec_dense, prof.dec_moe,
        prof.dec_head, prof.dec_samp, (long)prof.fw_calls,
        (long)prof.fw_tokens, prof.fw_upload, prof.fw_dense, prof.fw_moe,
        prof.fw_head, (long)prof.fw_device, (long)prof.fw_batched,
        (long)prof.fw_rows, prof.mtp_draft, (long)prof.spec_windows,
        (long)prof.drafted, (long)prof.accepted);
  }
  if (prof_ev_) {
    for (int i = 0; i < prof_ev_n_; ++i) cudaEventDestroy(prof_ev_[i]);
    delete[] prof_ev_;
    prof_ev_ = nullptr;
  }
  if (fw_ev_) {
    for (int i = 0; i < fw_ev_n_; ++i) cudaEventDestroy(fw_ev_[i]);
    delete[] fw_ev_;
    fw_ev_ = nullptr;
  }
  if (cache_.capacity() > 0) {
    // Persist the usage profile so the next start skips the cold ramp.
    if (const auto p = env_get("SAGER_PROFILE_PATH")) cache_.save_profile(*p);
  }
  pool_.stop();
}

const float* Session::last_logits() const { return logits_host_.get(); }

Status Session::alloc_scratch() {
  const EngineDims& d = mc_.dims;
  { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
    SLOG_I("session", "scratch: %.0f MiB free of %.0f MiB before alloc",
           f / 1048576.0, t / 1048576.0); }
  auto ok = [](bool b, const char* what) {
    if (!b) {
      size_t f = 0, t = 0;
      cudaMemGetInfo(&f, &t);
      SLOG_E("session", "scratch OOM at %s (%.0f MiB free)",
             what, f / 1048576.0);
    }
    return b ? Status::kOk : Status::kOomVram;
  };
  const int H = d.hidden;
  const int Q = d.n_head * d.head_dim;
  const int KVd = d.n_embd_gqa();
  const int FFX = d.n_ff_exp;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  const int T = cp_.chunk_prefill > 0 ? cp_.chunk_prefill : 1;
  const int V = d.n_vocab;

  // v0.9: fp16 dequant scratch for the TC prefill GEMM path - ONE reuse
  // buffer sized for the largest dense projection (GDN qkvz proj_rows x H
  // dominates; lin_out is H x y_rows, smaller in elements).
  // v0.10: the MoE TC ring lives here too (two disjoint regions of G slots
  // each) - allocate it whenever either TC path is on.
  {
    const auto g = mc_.gdn();
    size_t need = static_cast<size_t>(std::max(g.proj_rows(), g.y_rows())) *
                  static_cast<size_t>(H);
    need = std::max(need, static_cast<size_t>(Q) * H);
    need = std::max(need, static_cast<size_t>(H) * g.y_rows());
    if (pf_moe_tc_) {
      // ring: moe_grp_ x (gate+up slot + down slot) fp16 elems
      const size_t ring_need =
          static_cast<size_t>(8) * 3 * static_cast<size_t>(FFX) * H;
      need = std::max(need, ring_need);
    }
    if (pf_tc_ || pf_moe_tc_) {
      if (!ok(d_wtc_.alloc(need), "d_wtc_")) return Status::kOomVram;
    } else {
      d_wtc_.alloc(0);
    }
  }

  // Activation slabs sized for the prefill chunk (fixed addresses: the same
  // buffers serve the 1-row decode path).
  if (!ok(x_.alloc(static_cast<size_t>(T) * H), "x_")) return Status::kOomVram;
  if (!ok(xn_.alloc(static_cast<size_t>(T) * H), "xn_")) return Status::kOomVram;
  x2_.alloc(0);            // repurposed below: compact hit-out staging [K*H]
  if (!ok(x2_.alloc(static_cast<size_t>(K) * H), "x2_")) return Status::kOomVram;
  xn2_.alloc(0);           // reserved
  if (!ok(q_.alloc(static_cast<size_t>(T) * Q), "q_")) return Status::kOomVram;
  if (!ok(k_.alloc(static_cast<size_t>(T) * KVd), "k_")) return Status::kOomVram;
  if (!ok(v_.alloc(static_cast<size_t>(T) * KVd), "v_")) return Status::kOomVram;
  if (!ok(attn_out_.alloc(static_cast<size_t>(T) * Q), "attn_out_")) return Status::kOomVram;
  if (!ok(d_gate_.alloc(static_cast<size_t>(T) * Q), "d_gate_")) return Status::kOomVram;
  if (!ok(d_shared_gate_.alloc(1), "d_shared_gate_")) return Status::kOomVram;
  if (!ok(o_proj_.alloc(static_cast<size_t>(T) * H), "o_proj_")) return Status::kOomVram;
  if (!ok(xf_.alloc(H), "xf_")) return Status::kOomVram;        // MTP hidden stash
  if (!ok(d_cat_.alloc(2 * H), "d_cat_")) return Status::kOomVram;
  if (!ok(d_m1_.alloc(H), "d_m1_")) return Status::kOomVram;

  // MoE working set
  // v0.2.3: slabs sized for the batched prefill walk (RT-token chunks; the
  // per-row decode path only ever touches the first K rows of each).
  const int RT = pf_batched_
      ? std::max(1, std::min(T > 0 ? T : 1, pf_chunk_cap_))
      : 1;
  pf_rt_ = RT;
  const long Rmax = static_cast<long>(RT) * K;
  if (!ok(expert_g2_.alloc(static_cast<size_t>(Rmax) * FFX), "expert_g2_")) return Status::kOomVram;
  if (!ok(expert_u2_.alloc(static_cast<size_t>(Rmax) * FFX), "expert_u2_")) return Status::kOomVram;
  if (!ok(expert_g_.alloc(static_cast<size_t>(K) * FFX), "expert_g_")) return Status::kOomVram;
  if (!ok(expert_u_.alloc(static_cast<size_t>(K) * FFX), "expert_u_")) return Status::kOomVram;
  if (!ok(expert_y_.alloc(static_cast<size_t>(RT) * (K + 1) * H), "expert_y_")) return Status::kOomVram;
  if (!ok(shared_yb_.alloc(static_cast<size_t>(RT) * H), "shared_yb_")) return Status::kOomVram;
  if (!ok(sh_g_.alloc(static_cast<size_t>(RT) * FFX), "sh_g_")) return Status::kOomVram;
  if (!ok(sh_u_.alloc(static_cast<size_t>(RT) * FFX), "sh_u_")) return Status::kOomVram;
  if (!ok(y_grouped_.alloc(static_cast<size_t>(Rmax) * H), "y_grouped_")) return Status::kOomVram;
  if (!ok(d_xqg_.alloc(static_cast<size_t>(Rmax) * (H / q4k::kQK_K)), "d_xqg_")) return Status::kOomVram;
  if (!ok(d_hqg_.alloc(std::max<size_t>(static_cast<size_t>(Rmax) * (FFX / q4k::kQK_K),
                                        static_cast<size_t>(RT) * (FFX / q4k::kQK_K))),
         "d_hqg_")) return Status::kOomVram;
  if (!ok(d_sgate_.alloc(RT), "d_sgate_")) return Status::kOomVram;
  if (!ok(d_grows_.alloc(Rmax), "d_grows_")) return Status::kOomVram;
  if (!ok(d_gslot_.alloc(static_cast<size_t>(RT) * K), "d_gslot_")) return Status::kOomVram;
  // v0.10 MoE TC path: zero new VRAM - the fp16 routed-row slab reuses
  // expert_y_ and the fp16 expert ring reuses d_wtc_ (see session.hpp).
  // h_moe_meta_ is 16 B of pinned host memory.
  if (pf_moe_tc_) {
    moe_slot_elems_ = static_cast<size_t>(3) * FFX * H;  // gate+up rows then down
    // ring capacity from the dense TC scratch: floor to a power of two
    int g = static_cast<int>(d_wtc_.count() / moe_slot_elems_);
    while (g > 1 && (g & (g - 1)) != 0) --g;
    moe_grp_ = g < 1 ? 1 : (g > 8 ? 8 : g);
    if (!h_moe_meta_.alloc(4)) return Status::kOomRam;
  } else {
    moe_slot_elems_ = 0;
    moe_grp_ = 0;
    h_moe_meta_.alloc(0);
  }
  if (!ok(d_cflat_.alloc(pf_cpu_cap_), "d_cflat_")) return Status::kOomVram;
  if (!ok(shared_y_.alloc(static_cast<size_t>(H) + 2 * FFX), "shared_y_")) return Status::kOomVram;
  if (!ok(route_logits_.alloc(static_cast<size_t>(RT) * E), "route_logits_")) return Status::kOomVram;
  if (!ok(route_idx_.alloc(static_cast<size_t>(RT) * (K + 1)), "route_idx_")) return Status::kOomVram;
  if (!ok(route_w_.alloc(static_cast<size_t>(RT) * (K + 1)), "route_w_")) return Status::kOomVram;
  if (!ok(d_miss_y_.alloc(static_cast<size_t>(pf_cpu_cap_) * H), "d_miss_y_")) return Status::kOomVram;
  if (!ok(d_slot_of_k_.alloc(K), "d_slot_of_k_")) return Status::kOomVram;
  if (!ok(d_hit_rows_.alloc(K), "d_hit_rows_")) return Status::kOomVram;
  if (!ok(d_ptrs_hit_.alloc(3 * K), "d_ptrs_hit_")) return Status::kOomVram;

  // Activation quant slabs: worst case = chunked batched GEMV activations.
  const size_t q8_blocks = std::max({static_cast<size_t>(T) * H,
                                     static_cast<size_t>(T) * Q,
                                     static_cast<size_t>(K) * FFX}) /
                           q4k::kQK_K;
  if (!ok(d_xq_.alloc(q8_blocks), "d_xq_")) return Status::kOomVram;
  if (!ok(d_hq_.alloc(q8_blocks), "d_hq_")) return Status::kOomVram;
  if (!ok(d_mq_.alloc(H / q4k::kQK_K), "d_mq_")) return Status::kOomVram;

  // Head + io
  if (!ok(logits_.alloc(V), "logits_")) return Status::kOomVram;
  if (!ok(d_logits16_.alloc(V), "d_logits16_")) return Status::kOomVram;
  mtp_logits_.alloc(0);
  if (!ok(d_params_.alloc(1), "d_params_")) return Status::kOomVram;
  if (!ok(d_ids_.alloc(T), "d_ids_")) return Status::kOomVram;
  if (!ok(d_pos_ids_.alloc(3 * T), "d_pos_ids_")) return Status::kOomVram;
  if (!ok(d_recent_.alloc(kMaxRecent), "d_recent_")) return Status::kOomVram;
  if (!ok(d_recent_n_.alloc(1), "d_recent_n_")) return Status::kOomVram;

  // v0.2 expert pump staging
  if (!ok(d_ptrs_pump_.alloc(3 * K), "d_ptrs_pump_")) return Status::kOomVram;
  if (!ok(d_pump_rows_.alloc(K), "d_pump_rows_")) return Status::kOomVram;
  if (!ok(d_xq_moe_.alloc(static_cast<size_t>(RT) * (H / q4k::kQK_K)), "d_xq_moe_")) return Status::kOomVram;

  // v0.2 GDN working set (fixed addresses; state slabs are per linear layer)
  const GdnParams g = mc_.gdn();
  const int n_gdn = mc_.n_linear();
  if (!ok(lin_proj_.alloc(static_cast<size_t>(g.proj_rows())), "lin_proj_")) return Status::kOomVram;
  if (!ok(lin_ba_.alloc(static_cast<size_t>(g.ba_rows())), "lin_ba_")) return Status::kOomVram;
  if (!ok(lin_conv_.alloc(static_cast<size_t>(g.conv_dim())), "lin_conv_")) return Status::kOomVram;
  if (!ok(lin_qn_.alloc(static_cast<size_t>(g.n_k_heads) * g.key_dim), "lin_qn_")) return Status::kOomVram;
  if (!ok(lin_kn_.alloc(static_cast<size_t>(g.n_k_heads) * g.key_dim), "lin_kn_")) return Status::kOomVram;
  if (!ok(lin_y_.alloc(static_cast<size_t>(g.y_rows())), "lin_y_")) return Status::kOomVram;
  if (!ok(lin_on_.alloc(static_cast<size_t>(g.y_rows())), "lin_on_")) return Status::kOomVram;
  if (!ok(lin_ab_.alloc(static_cast<size_t>(2 * g.n_v_heads)), "lin_ab_")) return Status::kOomVram;
  if (!ok(lin_conv_state_.alloc(static_cast<size_t>(n_gdn) *
                                g.conv_state_elems()), "lin_conv_state_")) return Status::kOomVram;
  if (!ok(lin_delta_state_.alloc(static_cast<size_t>(n_gdn) *
                                 g.state_elems()), "lin_delta_state_")) return Status::kOomVram;
  // v0.3 chunk (batched) GDN slabs: [RT x rows] so prefill/verify windows
  // run the whole layer in ~10 launches instead of ~12 per token.
  if (!ok(lin_proj_b_.alloc(static_cast<size_t>(RT) * g.proj_rows()), "lin_proj_b_")) return Status::kOomVram;
  if (!ok(lin_conv_b_.alloc(static_cast<size_t>(RT) * g.conv_dim()), "lin_conv_b_")) return Status::kOomVram;
  if (!ok(lin_qn_b_.alloc(static_cast<size_t>(RT) * g.n_k_heads * g.key_dim), "lin_qn_b_")) return Status::kOomVram;
  if (!ok(lin_kn_b_.alloc(static_cast<size_t>(RT) * g.n_k_heads * g.key_dim), "lin_kn_b_")) return Status::kOomVram;
  if (!ok(lin_y_b_.alloc(static_cast<size_t>(RT) * g.y_rows()), "lin_y_b_")) return Status::kOomVram;
  if (!ok(lin_on_b_.alloc(static_cast<size_t>(RT) * g.y_rows()), "lin_on_b_")) return Status::kOomVram;
  if (!ok(lin_ba_b_.alloc(static_cast<size_t>(RT) * g.ba_rows()), "lin_ba_b_")) return Status::kOomVram;
  if (!ok(lin_ab_b_.alloc(static_cast<size_t>(RT) * 2 * g.n_v_heads), "lin_ab_b_")) return Status::kOomVram;
  // v0.3 grouped MoE job tables (upper bound: one job per expert; tiles are
  // bounded by Rmax/tile + n_expert)
  const int job_cap = E + 1;
  const int tile_cap = Rmax / 4 + E + 16;
  if (!ok(d_job_wg_.alloc(job_cap), "d_job_wg_") ||
      !ok(d_job_wu_.alloc(job_cap), "d_job_wu_") ||
      !ok(d_job_wo_.alloc(job_cap), "d_job_wo_") ||
      !ok(d_job_off_.alloc(job_cap), "d_job_off_") ||
      !ok(d_job_m_.alloc(job_cap), "d_job_m_") ||
      !ok(d_job_tbase_.alloc(job_cap + 1), "d_job_tbase_") ||
      !ok(d_job_jot_.alloc(tile_cap), "d_job_jot_")) return Status::kOomVram;

  // v0.4 device-side dispatch (P1): chunk pointer table + device plan slabs
  const size_t tbl_entries =
      static_cast<size_t>(d.n_moe_layer) * 3 * E;
  if (!ok(d_ptr_table_.alloc(tbl_entries), "d_ptr_table_")) return Status::kOomVram;
  const int Lm = d.n_moe_layer;
  if (!ok(d_off_.alloc(E + 1), "d_off_") ||
      !ok(d_toff_.alloc(E + 1), "d_toff_") ||
      !ok(d_meta_.alloc(4), "d_meta_") ||
      !ok(d_mail_.alloc(E), "d_mail_") ||
      !ok(d_mail_n_.alloc(1), "d_mail_n_") ||
      !ok(d_rk_.alloc(static_cast<size_t>(RT) * K), "d_rk_")) {
    return Status::kOomVram;
  }
  // v0.6 full-layer decode graphs: per-layer mailbox slots
  if (!ok(d_mail_all_.alloc(static_cast<size_t>(Lm) * E), "d_mail_all_") ||
      !ok(d_mail_n_all_.alloc(Lm), "d_mail_n_all_") ||
      !ok(d_off_all_.alloc(static_cast<size_t>(Lm) * (E + 1)), "d_off_all_") ||
      !ok(d_toff_all_.alloc(static_cast<size_t>(Lm) * (E + 1)), "d_toff_all_") ||
      !ok(d_meta_all_.alloc(static_cast<size_t>(Lm) * 4), "d_meta_all_") ||
      !ok(d_gslot_all_.alloc(static_cast<size_t>(Lm) * K), "d_gslot_all_") ||
      !ok(d_grows_all_.alloc(static_cast<size_t>(Lm) * K), "d_grows_all_") ||
      !ok(d_x16_all_.alloc(static_cast<size_t>(Lm) * H), "d_x16_all_") ||
      !ok(d_route_w_all_.alloc(static_cast<size_t>(Lm) * (K + 1)),
          "d_route_w_all_")) {
    return Status::kOomVram;
  }
  if (!h_ptr_table_.alloc(tbl_entries) ||
      !h_mail_n_.alloc(1) || !h_mail_.alloc(E) ||
      !h_off_.alloc(E + 1) || !h_rk_.alloc(static_cast<size_t>(RT) * K) ||
      !h_touch_idx_.alloc(static_cast<size_t>(d.n_moe_layer) * RT *
                          (K + 1)) ||
      !h_mail_n_all_.alloc(Lm) ||
      !h_mail_all_.alloc(static_cast<size_t>(Lm) * E)) {
    return Status::kOomRam;
  }
  absent_cnt_.assign(static_cast<size_t>(Lm), 0);
  used_full_.assign(static_cast<size_t>(Lm), 0);
  topk_seg_slots_ = (V + 2047) / 2048 * 64;
  // v0.5 decode fast path staging (device sampler + step params)
  const int topk_seg_slots = (V + 2047) / 2048 * 64;
  if (!ok(d_sample_logits_.alloc(V), "d_sample_logits_") ||
      !ok(d_topk_v_.alloc(64), "d_topk_v_") ||
      !ok(d_topk_i_.alloc(64), "d_topk_i_") ||
      !ok(d_topk_seg_v_.alloc(topk_seg_slots), "d_topk_seg_v_") ||
      !ok(d_topk_seg_i_.alloc(topk_seg_slots), "d_topk_seg_i_")) {
    return Status::kOomVram;
  }
  if (!h_topk_v_.alloc(64) || !h_topk_i_.alloc(64) ||
      !h_step_stage_.alloc(32) || !h_recent_stage_.alloc(kMaxRecent + 1)) {
    return Status::kOomRam;
  }
  if (cudaEventCreateWithFlags(&ev_scan_, cudaEventDisableTiming) !=
      cudaSuccess) {
    ev_scan_ = nullptr;
    cudaGetLastError();
  }
  if (pf_moe_tc_ &&
      cudaEventCreateWithFlags(&ev_moe_plan_, cudaEventDisableTiming) !=
          cudaSuccess) {
    ev_moe_plan_ = nullptr;
    cudaGetLastError();
  }
  // v0.6 profiler: event rings (2 records per layer + 4 markers per pass).
  if (prof.on) {
    prof_ev_n_ = 2 * d.n_layer + 5;
    fw_ev_n_ = 2 * d.n_layer + 5;
    prof_ev_ = new cudaEvent_t[prof_ev_n_]{};
    fw_ev_ = new cudaEvent_t[fw_ev_n_]{};
    for (int i = 0; i < prof_ev_n_; ++i)
      cudaEventCreate(&prof_ev_[i]);
    for (int i = 0; i < fw_ev_n_; ++i) cudaEventCreate(&fw_ev_[i]);
  }
  if (!h_job_wg_.alloc(job_cap) || !h_job_wu_.alloc(job_cap) ||
      !h_job_wo_.alloc(job_cap) || !h_job_off_.alloc(job_cap) ||
      !h_job_m_.alloc(job_cap) || !h_job_tbase_.alloc(job_cap + 1) ||
      !h_job_jot_.alloc(tile_cap)) return Status::kOomRam;

  // Host staging (pinned, C1: the SAME regions every call)
  if (!h_route_idx_.alloc(static_cast<size_t>(RT) * (K + 1)) ||
      !h_route_w_.alloc(static_cast<size_t>(RT) * (K + 1)) ||
      !h_grows_.alloc(Rmax) || !h_gslot_.alloc(static_cast<size_t>(RT) * K) ||
      !h_cflat_.alloc(pf_cpu_cap_) ||
      !h_x16_.alloc(static_cast<size_t>(RT) * H) ||
      !miss_x_.alloc(static_cast<size_t>(pf_cpu_cap_) * H) ||
      !miss_y_.alloc(static_cast<size_t>(pf_cpu_cap_) * H) ||
      !logits_host_.alloc(V) || !h_ptrs_.alloc(3 * K) ||
      !h_hit_rows_.alloc(K) || !h_slot_of_k_.alloc(K) ||
      !h_pump_ptrs_.alloc(3 * K) || !h_pump_rows_.alloc(K)) {
    return Status::kOomRam;
  }
  return Status::kOk;
}

Status Session::prepare() {
  if (prepared_) return Status::kOk;
  if (!w_.tok_embd) {
    SLOG_E("session", "prepare called before weights load");
    return Status::kInvalidArgument;
  }

  // v0.2.3 prefill knobs (before alloc_scratch - they size the route slabs).
  if (const auto p = env_get("SAGER_PROF")) prof.on = (*p != "0");
  if (const auto p = env_get("SAGER_STEP_GRAPH")) step_graph_on_ = (*p != "0");
  if (const auto p = env_get("SAGER_FAST_SAMPLE")) fast_sample_ = (*p != "0");
  {
    const auto b = env_get("SAGER_PF_BATCHED");
    if (b) pf_batched_ = (*b != "0");
    const auto mt = env_get("SAGER_MTOK");
    if (mt) pf_mtok_ = (*mt != "0");
    const auto sm = env_get("SAGER_PF_STREAM_MIN");
    if (sm) pf_stream_min_ = std::max(1, std::atoi(sm->c_str()));
    const auto cc = env_get("SAGER_PF_CHUNK");
    if (cc) pf_chunk_cap_ = std::max(64, std::atoi(cc->c_str()));
    const auto crc = env_get("SAGER_PF_CPU_ROWS");
    if (crc) pf_cpu_cap_ = std::clamp(std::atoi(crc->c_str()), 16, 4096);
    const auto dev = env_get("SAGER_PF_DEVICE");
    if (dev) pf_device_ = (*dev != "0");
    // v0.5 decode fast-path knobs (read here, logged in the ready line)
    {
      const auto dd = env_get("SAGER_DECODE_DEVICE");
      if (dd) dec_device_ = (*dd != "0");
      const auto ds = env_get("SAGER_DEVICE_SAMPLER");
      if (ds) dev_sample_ = (*ds != "0");
      // v0.6 knobs (all default on; each flips cleanly back to v0.5 paths)
      const auto fg = env_get("SAGER_FULL_GRAPH");
      if (fg) graphs_full_valid_ = (*fg != "0");
      const auto hq = env_get("SAGER_HEAD_Q6K");
      if (hq) head_q6k_ = (*hq != "0");
      const auto tk = env_get("SAGER_TOPK2");
      if (tk) topk2_ = (*tk != "0");
      const auto gw = env_get("SAGER_GDN_WS");
      if (gw) gdn_ws_ = (*gw != "0");
      // v0.12 knob (default on; SAGER_GDN_R4=0 -> the v0.11 warp scan)
      const auto gr4 = env_get("SAGER_GDN_R4");
      if (gr4) gdn_r4_ = (*gr4 != "0");
      const auto mdb = env_get("SAGER_MOE_DEBUG");
      if (mdb) moe_dbg_ = (*mdb != "0");
      // v0.8 knobs (all default on; each flips cleanly back to the v0.7 path)
      const auto mq = env_get("SAGER_PF_MMQ");
      if (mq) pf_mmq_ = (*mq != "0");
      const auto df = env_get("SAGER_DECODE_FUSE");
      if (df) dec_fuse_ = (*df != "0");
      const auto hs = env_get("SAGER_HEAD_SMX");
      if (hs) head_smx_ = (*hs != "0");
      const auto gf = env_get("SAGER_GDN_FFN_NORM");
      if (gf) gdn_ffn_norm_ = (*gf != "0");
      // v0.9 knob (default on; SAGER_PF_TC=0 reverts to the v0.8 dp4a GEMVs)
      const auto tc = env_get("SAGER_PF_TC");
      if (tc) pf_tc_ = (*tc != "0");
      // v0.10 knob (default on; SAGER_MOE_TC=0 reverts to the grouped dp4a
      // GEMV family for the MoE experts)
      const auto mtc = env_get("SAGER_MOE_TC");
      if (mtc) pf_moe_tc_ = (*mtc != "0");
      // v0.12 knobs (default on; the parity bench arms pin both off)
      const auto stc = env_get("SAGER_SH_TC");
      if (stc) pf_sh_tc_ = (*stc != "0");
      const auto rtc = env_get("SAGER_ROUTER_TC");
      if (rtc) pf_router_tc_ = (*rtc != "0");
      if (const auto dlp = env_get("SAGER_DUMP_LOGITS")) {
        dbg_layers_ = (*dlp == "2");
      }
    }
    const EngineDims& dd0 = mc_.dims;
    if (pf_batched_ &&
        (dd0.hidden % q4k::kQK_K != 0 || dd0.n_ff_exp % q4k::kQK_K != 0)) {
      SLOG_W("session", "batched prefill needs 256-aligned hidden/ff "
             "(%d/%d) - per-row walk stays", dd0.hidden, dd0.n_ff_exp);
      pf_batched_ = false;
    }
    SLOG_I("session", "prefill v0.12: batched=%d mtok=%d mmq=%d tc=%d "
           "moe_tc=%d sh_tc=%d rt_tc=%d gdn_r4=%d "
           "stream_min=%d chunk_cap=%d cpu_rows=%d device=%d fuse=%d "
           "head_smx=%d",
           (int)pf_batched_, (int)pf_mtok_, (int)pf_mmq_, (int)pf_tc_,
           (int)pf_moe_tc_, (int)pf_sh_tc_, (int)pf_router_tc_, (int)gdn_r4_,
           pf_stream_min_, pf_chunk_cap_, pf_cpu_cap_, (int)pf_device_,
           (int)dec_fuse_, (int)head_smx_);
  }

  if (w_.head_type_ != 12 && w_.head_type_ != 1 && w_.head_type_ != 14) {
    SLOG_E("session", "unsupported output head dtype %u (need Q4_K=12); "
                      "re-convert with tools/convert_hf_to_sager.py",
           w_.head_type_);
    return Status::kUnsupported;
  }

  const EngineDims& d = mc_.dims;
  const int E = d.n_expert;

  // Expert cache + hot tier (VRAM slab + pointer tables).
  const size_t budget = static_cast<size_t>(cp_.vram_budget_mb) << 20;
  cache_.init(d.n_moe_layer, E, budget, mc_.expert_bytes());
  if (const auto pp = env_get("SAGER_PROFILE_PATH")) {
    if (cache_.load_profile(*pp)) {
      SLOG_I("session", "loaded expert usage profile from %s", pp->c_str());
    }
  }
  cache_.rebalance();

  hot_ = std::make_unique<detail::HotTier>();
  if (!hot_->pool.alloc(budget)) return Status::kOomVram;
  const size_t table_entries =
      static_cast<size_t>(d.n_moe_layer) * 3 * E;
  if (!hot_->tables.alloc(table_entries)) return Status::kOomVram;
  hot_->tables_host.resize(table_entries, nullptr);
  Status s = hot_->rebuild(cache_, w_, d.n_moe_layer, E, mc_.expert_bytes());
  if (s != Status::kOk) return s;

  // mrope section boundaries -> device constant memory (graph-safe: fixed).
  cuda::set_mrope_section(mc_.mrope_section.data());

  // v0.2 layer-kind indirection: GQA layers share the KV slab pool; GDN
  // layers get a state slab index instead and touch no KV at all.
  const int n_gdn_layers = mc_.n_linear();
  kv_slot_of_layer_.assign(static_cast<size_t>(d.n_layer), -1);
  gdn_idx_of_layer_.assign(static_cast<size_t>(d.n_layer), -1);
  int kv_i = 0, gdn_i = 0;
  for (int L = 0; L < d.n_layer; ++L) {
    if (mc_.is_linear(L)) gdn_idx_of_layer_[static_cast<size_t>(L)] = gdn_i++;
    else                  kv_slot_of_layer_[static_cast<size_t>(L)] = kv_i++;
  }

  // v0.2 expert pump (C7 contract; arena next to the hot pool in VRAM).
  if (cp_.expert_pump) {
    pump_ = std::make_unique<ExpertPump>();
    ExpertPump::Config pc;
    pc.n_layers = d.n_moe_layer;
    pc.n_experts = E;
    pc.per_expert_bytes = mc_.expert_bytes();
    pc.arena_bytes = cp_.pump_arena_mb << 20;
    s = pump_->init(pc, w_.host_expert_base());
    if (s != Status::kOk) {
      SLOG_W("session", "expert pump init failed (%s) - CPU pool only",
             status_str(s));
      pump_.reset();
    }
  }

  // CPU expert pool (C4: geometry validated at bind time).
  const PoolGeometry want{d.hidden, d.n_ff_exp, E, q4k::kBlockBytes};
  s = pool_.start(cp_.batch_threads, want);
  if (s != Status::kOk) return s;

  s = alloc_scratch();
  if (s != Status::kOk) return s;

  // Zero the graph-safe decode param block and position ids BEFORE any warmup
  // or capture: the dense body indexes the KV slab at SeqParams.pos, and an
  // uninitialized block sends the warmup launches out of bounds, which
  // poisons the CUDA context (every later kernel silently no-ops).
  cudaMemset(d_params_.get(), 0, sizeof(SeqParams));
  cudaMemset(d_pos_ids_.get(), 0, 3 * sizeof(int));
  cudaMemset(d_recent_.get(), 0, kMaxRecent * sizeof(int));
  cudaMemset(x_.get(), 0, static_cast<size_t>(cp_.chunk_prefill) * d.hidden *
                              sizeof(uint16_t));
  cudaMemset(xf_.get(), 0, static_cast<size_t>(d.hidden) * sizeof(uint16_t));
  cudaMemset(route_idx_.get(), 0, (d.n_expert_used + 1) * sizeof(int));

  // Static combine extension: slot K carries the shared expert with weight 1.
  // (Buffers above are sized [K+1]; these writes must come after alloc_scratch,
  //  which allocates both the pinned staging and the device route slots.)
  h_route_idx_[d.n_expert_used] = 0;
  h_route_w_[d.n_expert_used] = 1.0f;
  cudaMemcpy(route_idx_.get(), h_route_idx_.get(),
             (d.n_expert_used + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(route_w_.get(), h_route_w_.get(),
             (d.n_expert_used + 1) * sizeof(float), cudaMemcpyHostToDevice);

  kv_.init({d.n_layer - n_gdn_layers, d.n_head_kv, d.head_dim, cp_.n_ctx,
            cp_.kv_q8});

  if (cp_.capture_graphs) {
    s = capture_graphs();
    if (s != Status::kOk) {
      SLOG_W("session", "graph capture failed (%s) - running eager",
             status_str(s));
      graphs_.set_valid(false);
      // A sticky device error here means the context is unusable - surface it
      // instead of letting the eager path fail mysteriously later.
      cudaError_t ds = cudaDeviceSynchronize();
      SLOG_W("session", "post-capture device sync: %s",
             ds == cudaSuccess ? "clean"
                               : cudaGetErrorString(cudaGetLastError()));
    }
  } else {
    graphs_.set_valid(false);
  }

  SLOG_I("session",
         "ready: ctx=%d chunk=%d graphs=%s hot_experts=%d/%d gdn_layers=%d "
         "pump=%s(%d slots) dec_device=%d dev_sampler=%d",
         cp_.n_ctx, cp_.chunk_prefill, graphs_.valid() ? "on" : "off",
         hot_->resident_count, d.experts_total(), n_gdn_layers,
         pump_ ? "on" : "off", pump_ ? pump_->slot_count() : 0,
         (int)dec_device_, (int)dev_sample_);
  prepared_ = true;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Dense layer body (graph-captured for decode, eager-batched for prefill)
// ---------------------------------------------------------------------------
void Session::dense_layer_launch(int layer, int n_tokens) {
  const EngineDims& d = mc_.dims;
  const LayerWeights& lw = w_.layers()[static_cast<size_t>(layer)];
  const int H = d.hidden;
  const int Q = d.n_head * d.head_dim;
  const int KVd = d.n_embd_gqa();
  const int rope_dims = static_cast<int>(mc_.rope_partial *
                                         static_cast<float>(d.head_dim));
  cudaStream_t s = stream_.get();

  if (n_tokens > 1 && pf_tc_ && pf_batched_) {
    // v0.9 TC path: activations stay fp16 - no Q8_K requant needed here; the
    // MoE half quantizes xn itself (moe_xq_in_ stays nullptr; d_xq_ is only
    // filled by the fallback below or the n==1 decode path).
    cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.attn_norm, H,
                              mc_.norm_eps, n_tokens, s);
  } else if (dec_fuse_ && n_tokens == 1) {
    // v0.8: rmsnorm + Q8_K quant in one launch (bit-identical to the pair);
    // decode-only - at n>1 the two-kernel sequence's spread-out quantize
    // grid wins (the fused kernel is barrier-heavy per row)
    cuda::rmsnorm_quant_f16(xn_.get(), x_.get(), lw.attn_norm, d_xq_.get(),
                            H, mc_.norm_eps, n_tokens, s);
  } else {
    if (n_tokens == 1) {
      cuda::rmsnorm_f16(xn_.get(), x_.get(), lw.attn_norm, H, mc_.norm_eps, s);
    } else {
      cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.attn_norm, H,
                                mc_.norm_eps, n_tokens, s);
    }
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n_tokens * H, s);
  }

  if (n_tokens == 1) {
    if (dec_fuse_) {
      // v0.8: q|k|v projections in ONE launch (per-row math unchanged)
      cuda::gemv3m_q4_k_q8_f16out(Q4W(lw.wq), Q4W(lw.wk), Q4W(lw.wv),
                                  d_xq_.get(), q_.get(), k_.get(), v_.get(),
                                  Q, KVd, KVd, H, layer, s);
    } else {
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H, s);
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H, s);
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H, s);
    }
  } else if (pf_tc_ && pf_batched_) {
    // v0.9: real GEMM via cuBLAS tensor cores - weights dequantize once per
    // chunk into d_wtc_ (fp16), activations stay fp16, fp32 accumulate.
    if (!cuda::tc_gemm_q4_k_f16(Q4W(lw.wq), xn_.get(), q_.get(),
                                d_wtc_.get(), Q, H, n_tokens, s) ||
        !cuda::tc_gemm_q4_k_f16(Q4W(lw.wk), xn_.get(), k_.get(),
                                d_wtc_.get(), KVd, H, n_tokens, s) ||
        !cuda::tc_gemm_q4_k_f16(Q4W(lw.wv), xn_.get(), v_.get(),
                                d_wtc_.get(), KVd, H, n_tokens, s)) {
      SLOG_E("session", "TC prefill qkv GEMM failed - dp4a fallback");
      cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n_tokens * H, s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H,
                                     n_tokens, s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H,
                                     n_tokens, s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H,
                                     n_tokens, s);
    }
  } else if (pf_mtok_ && pf_batched_ && pf_mmq_) {
    // v0.8: MMQ prefill GEMM - weights staged once per token tile, the
    // per-token dp4a chains interleave (bit-identical to the mtok path).
    cuda::mmq_q4_k_q8_f16out(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H,
                             n_tokens, s);
    cuda::mmq_q4_k_q8_f16out(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H,
                             n_tokens, s);
    cuda::mmq_q4_k_q8_f16out(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H,
                             n_tokens, s);
  } else if (pf_mtok_ && pf_batched_) {
    // v0.3: tiled multi-token GEMMs - weights are staged in shared memory
    // per token-tile instead of re-read per token (bit-identical outputs).
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H,
                                   n_tokens, s);
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H,
                                   n_tokens, s);
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H,
                                   n_tokens, s);
  } else {
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H,
                                      n_tokens, s);
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H,
                                      n_tokens, s);
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H,
                                      n_tokens, s);
  }
  // qwen3.5 gated attention: separate GEMV for the sigmoid-gate half of the
  // q projection (legacy files: attn_gated == false, gate stays zero).
  if (lw.attn_gated) {
    if (n_tokens == 1) {
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.attn_gate), d_xq_.get(), d_gate_.get(),
                                Q, H, s);
    } else if (pf_tc_ && pf_batched_) {
      // v0.9: TC path (fp16 activations; same numerics contract as qkv).
      if (!cuda::tc_gemm_q4_k_f16(Q4W(lw.attn_gate), xn_.get(),
                                  d_gate_.get(), d_wtc_.get(), Q, H,
                                  n_tokens, s)) {
        SLOG_E("session", "TC prefill attn_gate GEMM failed - dp4a fallback");
        cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n_tokens * H, s);
        cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.attn_gate), d_xq_.get(),
                                       d_gate_.get(), Q, H, n_tokens, s);
      }
    } else if (pf_mtok_ && pf_batched_ && pf_mmq_) {
      cuda::mmq_q4_k_q8_f16out(Q4W(lw.attn_gate), d_xq_.get(),
                               d_gate_.get(), Q, H, n_tokens, s);
    } else if (pf_mtok_ && pf_batched_) {
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.attn_gate), d_xq_.get(),
                                     d_gate_.get(), Q, H, n_tokens, s);
    } else {
      cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.attn_gate), d_xq_.get(),
                                        d_gate_.get(), Q, H, n_tokens, s);
    }
  }
  // per-head q/k RMSNorm (full-attention layers only; no-op when absent)
  if (lw.attn_q_norm) {
    cuda::rmsnorm_batched_f16(q_.get(), q_.get(), lw.attn_q_norm, d.head_dim,
                              mc_.norm_eps, n_tokens * d.n_head, s);
    cuda::rmsnorm_batched_f16(k_.get(), k_.get(), lw.attn_k_norm, d.head_dim,
                              mc_.norm_eps, n_tokens * d.n_head_kv, s);
  }
  cuda::rope_f16(q_.get(), k_.get(), d_pos_ids_.get(), d.n_head, d.n_head_kv,
                 d.head_dim, rope_dims, mc_.rope_theta, mc_.rope_style,
                 n_tokens, s);
  cuda::kv_append(k_.get(), v_.get(),
                  kv_.k_slab(kv_slot_of_layer_[static_cast<size_t>(layer)]),
                  kv_.v_slab(kv_slot_of_layer_[static_cast<size_t>(layer)]),
                  nullptr, nullptr, kv_.q8() ? 1 : 0, d_params_.get(),
                  n_tokens, d.n_head_kv, d.head_dim, s);
  if (n_tokens == 1) {
    cuda::attention_decode_f16(attn_out_.get(), q_.get(),
                               kv_.k_slab(kv_slot_of_layer_[
                                   static_cast<size_t>(layer)]),
                               kv_.v_slab(kv_slot_of_layer_[
                                   static_cast<size_t>(layer)]),
                               nullptr, nullptr,
                               kv_.q8() ? 1 : 0, d_params_.get(), d.n_head,
                               d.n_head_kv, d.head_dim,
                               1.0f / std::sqrt(static_cast<float>(d.head_dim)),
                               s);
  } else {
    cuda::attention_prefill_f16(attn_out_.get(), q_.get(),
                                kv_.k_slab(kv_slot_of_layer_[
                                    static_cast<size_t>(layer)]),
                                kv_.v_slab(kv_slot_of_layer_[
                                    static_cast<size_t>(layer)]),
                                nullptr, nullptr,
                                kv_.q8() ? 1 : 0, d_params_.get(), d.n_head,
                                d.n_head_kv, d.head_dim,
                                1.0f / std::sqrt(static_cast<float>(d.head_dim)),
                                n_tokens, s);
  }
  if (lw.attn_gated) {
    cuda::sigmoid_mul_rows_f16(attn_out_.get(), d_gate_.get(),
                               n_tokens * Q, Q, Q, 0, s);
  }
  if (n_tokens == 1) {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), Q, s);
    if (dec_fuse_) {
      // v0.8: o_proj with the residual add as epilogue (one launch less the
      // o_proj buffer round trip; add_kernel semantics preserved)
      cuda::gemv_q4_k_q8_f16out_resadd(Q4W(lw.wo), d_hq_.get(),
                                       o_proj_.get(), x_.get(), H, Q, s);
    } else {
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q, s);
    }
  } else if (pf_tc_ && pf_batched_) {
    // v0.9: TC path - attn_out stays fp16 (no requant); the residual add
    // stays a separate add_f16 (epilogue fusion is a later micro-opt).
    if (!cuda::tc_gemm_q4_k_f16(Q4W(lw.wo), attn_out_.get(), o_proj_.get(),
                                d_wtc_.get(), H, Q, n_tokens, s)) {
      SLOG_E("session", "TC prefill o_proj GEMM failed - dp4a fallback");
      cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), n_tokens * Q, s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wo), d_hq_.get(), o_proj_.get(),
                                     H, Q, n_tokens, s);
    }
  } else if (pf_mtok_ && pf_batched_ && pf_mmq_) {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), n_tokens * Q, s);
    cuda::mmq_q4_k_q8_f16out(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q,
                             n_tokens, s);
  } else if (pf_mtok_ && pf_batched_) {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), n_tokens * Q, s);
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q,
                                   n_tokens, s);
  } else {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), n_tokens * Q, s);
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q,
                                      n_tokens, s);
  }
  if (!(n_tokens == 1 && dec_fuse_)) {
    cuda::add_f16(x_.get(), o_proj_.get(), n_tokens * H, s);
  }
  if (dec_fuse_ && n_tokens == 1) {
    cuda::rmsnorm_quant_f16(xn_.get(), x_.get(), lw.ffn_norm, d_xq_.get(),
                            H, mc_.norm_eps, n_tokens, s);
    moe_xq_in_ = d_xq_.get();   // the MoE quantizes xn - reuse the fused q8
  } else {
    if (n_tokens == 1) {
      cuda::rmsnorm_f16(xn_.get(), x_.get(), lw.ffn_norm, H, mc_.norm_eps, s);
    } else {
      cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.ffn_norm, H,
                                mc_.norm_eps, n_tokens, s);
    }
    moe_xq_in_ = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Gated-delta layer body (graph-capturable: fixed shapes, fixed addresses).
// For n_tokens > 1 (prefill / verify) the recurrence is walked token by
// token - correct first, chunked-parallel scan is the v0.3 upgrade
// (docs/gdn-kernel.md). The MoE half after this body is IDENTICAL to the
// dense layer's (Sage-A1.5: every block is MoE).
// ---------------------------------------------------------------------------
void Session::linear_layer_launch(int layer, int n_tokens) {
  // v0.3: whole-chunk batched walk (projections batched, recurrence fused
  // into one kernel) - bit-identical to the per-token loop below. Gated on
  // pf_batched_ so SAGER_PF_BATCHED=0 (arm B) stays the exact v0.2.2 walk.
  if (n_tokens > 1 && pf_mtok_ && pf_batched_) {
    linear_layer_chunk(layer, n_tokens);
    return;
  }
  const EngineDims& d = mc_.dims;
  const GdnParams g = mc_.gdn();
  const int nv = g.n_v_heads;
  const LayerWeights& lw = w_.layers()[static_cast<size_t>(layer)];
  const int H = d.hidden;
  const int nk = g.n_k_heads;
  const int K = g.key_dim;
  const int rows = g.ratio() * g.value_dim;
  cudaStream_t s = stream_.get();
  const int slab = gdn_idx_of_layer_[static_cast<size_t>(layer)];
  uint16_t* conv_st = lin_conv_state_.get() +
      static_cast<size_t>(slab) * g.conv_state_elems();
  float* delta_st = lin_delta_state_.get() +
      static_cast<size_t>(slab) * g.state_elems();

  for (int t = 0; t < n_tokens; ++t) {
    uint16_t* x_row = x_.get() + static_cast<size_t>(t) * H;
    uint16_t* xn_row = xn_.get() + static_cast<size_t>(t) * H;
    uint16_t* op_row = o_proj_.get() + static_cast<size_t>(t) * H;
    if (dec_fuse_) {
      // v0.8: rmsnorm + Q8_K quant in one launch (bit-identical to the pair)
      cuda::rmsnorm_quant_f16(xn_row, x_row, lw.attn_norm, d_xq_.get(), H,
                              mc_.norm_eps, 1, s);
    } else {
      cuda::rmsnorm_f16(xn_row, x_row, lw.attn_norm, H, mc_.norm_eps, s);
      cuda::quantize_q8_k(xn_row, d_xq_.get(), H, s);
    }
    // fused projections: qkvz -> [q|k|v|z], ba -> [b|a]
    if (n_tokens == 1 && dec_fuse_) {
      // v0.8: qkvz|ba projections in ONE launch (per-row math unchanged)
      cuda::gemv2m_q4_k_q8_f16out(Q4W(lw.lin_in_qkvz), Q4W(lw.lin_in_ba),
                                  d_xq_.get(), lin_proj_.get(), lin_ba_.get(),
                                  g.proj_rows(), g.ba_rows(), H, layer, s);
    } else {
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.lin_in_qkvz), d_xq_.get(),
                                lin_proj_.get(), g.proj_rows(), H, s);
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.lin_in_ba), d_xq_.get(),
                                lin_ba_.get(), g.ba_rows(), H, s);
    }
    // conv (first conv_dim rows of proj: q|k|v; z bypasses) + SiLU
    cuda::gdn_conv_step_f16(lin_conv_.get(), conv_st, lin_proj_.get(),
                            lw.lin_conv_w, lw.lin_conv_b, g.conv_dim(),
                            g.conv_kernel, s);
    // data-dependent gates: alpha/beta per k-head
    cuda::gdn_gate_f32(lin_ba_.get(), lw.lin_A_log, lw.lin_dt_bias,
                       lin_ab_.get(), nv, s);
    // per-head L2 normalization of q/k (C-GDN-1; no learnable weight)
    cuda::gdn_l2_norm_f16(lin_qn_.get(), lin_kn_.get(), lin_conv_.get(),
                          nk, K, mc_.norm_eps, s);
    // rank-1 gated delta update of the persistent state (C-GDN-4)
    cuda::gdn_delta_step_f16(delta_st, lin_y_.get(), lin_qn_.get(),
                             lin_kn_.get(),
                             lin_conv_.get() + static_cast<size_t>(2) * nk * K,
                             lin_ab_.get(), nk, K, g.ratio(), g.value_dim,
                             s);
    // output gating: o = rmsnorm(y) * silu(z)  (C-GDN-6)
    cuda::gdn_out_norm_f16(lin_on_.get(), lin_y_.get(),
                           lin_proj_.get() + g.conv_dim(), lw.lin_o_norm,
                           g.n_v_heads, g.value_dim, mc_.norm_eps, s);
    // out_proj -> residual
    cuda::quantize_q8_k(lin_on_.get(), d_hq_.get(), g.y_rows(), s);
    if (n_tokens == 1 && dec_fuse_) {
      // v0.8: out_proj with the residual add as epilogue
      cuda::gemv_q4_k_q8_f16out_resadd(Q4W(lw.lin_out), d_hq_.get(), op_row,
                                       x_row, H, g.y_rows(), s);
    } else {
      cuda::gemv_q4_k_q8_f16out(Q4W(lw.lin_out), d_hq_.get(), op_row, H,
                                g.y_rows(), s);
      cuda::add_f16(x_row, op_row, H, s);
    }
  }
  // v0.8 FIX (SAGER_GDN_FFN_NORM=0 restores the v0.2-0.7 behavior): the MoE
  // half consumes ffn_norm(x) - on GDN layers this norm was never applied,
  // so the router/experts read the STALE attn_norm row from before the
  // recurrence update. The converter writes blk.L.ffn_norm for every block
  // (post_attention_layernorm), so the norm belongs here exactly as it is
  // applied at the end of dense_layer_launch.
  if (gdn_ffn_norm_) {
    if (dec_fuse_) {
      cuda::rmsnorm_quant_f16(xn_.get(), x_.get(), lw.ffn_norm, d_xq_.get(),
                              H, mc_.norm_eps, n_tokens, s);
      moe_xq_in_ = d_xq_.get();
    } else {
      cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.ffn_norm, H,
                                mc_.norm_eps, n_tokens, s);
      cuda::quantize_q8_k(xn_.get(), d_xq_.get(),
                          static_cast<size_t>(n_tokens) * H, s);
      moe_xq_in_ = nullptr;
    }
  } else if (dec_fuse_ && n_tokens == 1) {
    // bug-compatible: the fused attn-norm quant produced q8 of the same
    // stale xn row the v0.7 path re-quantized in moe_dispatch
    moe_xq_in_ = d_xq_.get();
  } else {
    moe_xq_in_ = nullptr;
  }
  (void)rows;
}

void Session::zero_gdn_states() {
  if (lin_conv_state_.count()) {
    cudaMemset(lin_conv_state_.get(), 0,
               lin_conv_state_.bytes());
  }
  if (lin_delta_state_.count()) {
    cudaMemset(lin_delta_state_.get(), 0,
               lin_delta_state_.bytes());
  }
}

// ---------------------------------------------------------------------------
// v0.3 batched GDN walk (n tokens at once; bit-identical to the per-token
// loop). Projections + conv + gates + norms run batched over the window;
// the sequential rank-1 recurrence fuses into ONE kernel whose blocks keep
// their state slice in shared memory for the whole chunk.
// ---------------------------------------------------------------------------
#if defined(SAGER_HAVE_CUDA)
// v0.12 debug helper: health of one fp16 slab row (SAGER_DUMP_LOGITS=2).
static void dbg_dump_f16(const char* tag, const uint16_t* dev, size_t off,
                         int cnt, cudaStream_t s) {
  static std::vector<uint16_t> buf;
  if (buf.size() < static_cast<size_t>(cnt)) buf.resize(static_cast<size_t>(cnt));
  cudaMemcpyAsync(buf.data(), dev + off, static_cast<size_t>(cnt) * sizeof(uint16_t),
                  cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  float sabs = 0.f, mx = 0.f;
  int nans = 0;
  for (int i = 0; i < cnt; ++i) {
    const float v = __half2float(*reinterpret_cast<const __half*>(&buf[i]));
    if (std::isnan(v)) ++nans;
    sabs += fabsf(v);
    mx = fmaxf(mx, fabsf(v));
  }
  fprintf(stderr, "[sager/dbg] %-18s sum|x|=%.4f max|x|=%.4f nan=%d\n",
          tag, sabs, mx, nans);
}
// v0.12 debug: fp32 variant (route logits / combine weights).
static void dbg_dump_f32(const char* tag, const float* dev, size_t off,
                         int cnt, cudaStream_t s) {
  static std::vector<float> buf;
  if (buf.size() < static_cast<size_t>(cnt)) buf.resize(static_cast<size_t>(cnt));
  cudaMemcpyAsync(buf.data(), dev + off, static_cast<size_t>(cnt) * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  float sabs = 0.f, mx = 0.f;
  int nans = 0;
  for (int i = 0; i < cnt; ++i) {
    const float v = buf[i];
    if (std::isnan(v)) ++nans;
    sabs += fabsf(v);
    mx = fmaxf(mx, fabsf(v));
  }
  fprintf(stderr, "[sager/dbg] %-18s sum=%.4f max=%.4f nan=%d v0-5=%.4g/%.4g/%.4g/%.4g/%.4g/%.4g\n",
          tag, sabs, mx, nans, buf[0], buf[1 % cnt], buf[2 % cnt],
          buf[3 % cnt], buf[4 % cnt], buf[5 % cnt]);
}
#endif

void Session::linear_layer_chunk(int layer, int n) {
  const EngineDims& d = mc_.dims;
  const GdnParams g = mc_.gdn();
  const LayerWeights& lw = w_.layers()[static_cast<size_t>(layer)];
  const int H = d.hidden;
  cudaStream_t s = stream_.get();
  const int slab = gdn_idx_of_layer_[static_cast<size_t>(layer)];
  uint16_t* conv_st = lin_conv_state_.get() +
      static_cast<size_t>(slab) * g.conv_state_elems();
  float* delta_st = lin_delta_state_.get() +
      static_cast<size_t>(slab) * g.state_elems();

  if (pf_tc_) {
    // v0.9 TC path: xn stays fp16 for the TC GEMMs; the MoE half quantizes
    // xn itself (moe_xq_in_ is nullptr on the n>1 path).
    cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.attn_norm, H,
                              mc_.norm_eps, n, s);
  } else if (dec_fuse_) {
    cuda::rmsnorm_quant_f16(xn_.get(), x_.get(), lw.attn_norm, d_xq_.get(),
                            H, mc_.norm_eps, n, s);
  } else {
    cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.attn_norm, H,
                              mc_.norm_eps, n, s);
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n * H, s);
  }
  if (dbg_layers_) {
    dbg_dump_f16("xn(norm)", xn_.get(),
                 static_cast<size_t>(n - 1) * H, H, s);
    // raw view of the first Q4_K blocks of the qkvz carve (weight sanity)
    static PinnedBuffer<uint8_t> dbg_w;
    if (dbg_w.count() < 8 * 144) dbg_w.alloc(8 * 144);
    cudaMemcpyAsync(dbg_w.get(), lw.lin_in_qkvz, 8 * 144,
                    cudaMemcpyDeviceToHost, s);
    stream_.sync();
    for (int b = 0; b < 4; ++b) {
      const uint8_t* p = dbg_w.get() + b * 144;
      const float d = __half2float(
          *reinterpret_cast<const __half*>(p));
      const float dmin = __half2float(
          *reinterpret_cast<const __half*>(p + 2));
      fprintf(stderr,
              "[sager/dbg] carve blk %d d=%.6f dmin=%.6f sc=%d/%d/%d q0=%u\n",
              b, d, dmin, (int)p[4], (int)p[5], (int)p[6],
              (unsigned)(p[16] & 0xF));
    }
  }
  // fused projections for the whole window
  if (pf_tc_) {
    if (!cuda::tc_gemm_q4_k_f16(Q4W(lw.lin_in_qkvz), xn_.get(),
                                lin_proj_b_.get(), d_wtc_.get(),
                                g.proj_rows(), H, n, s) ||
        !cuda::tc_gemm_q4_k_f16(Q4W(lw.lin_in_ba), xn_.get(),
                                lin_ba_b_.get(), d_wtc_.get(),
                                g.ba_rows(), H, n, s)) {
      SLOG_E("session", "TC prefill qkvz/ba GEMM failed - dp4a fallback");
      cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n * H, s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_in_qkvz), d_xq_.get(),
                                     lin_proj_b_.get(), g.proj_rows(), H, n, s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_in_ba), d_xq_.get(),
                                     lin_ba_b_.get(), g.ba_rows(), H, n, s);
    }
  } else if (pf_mmq_ && pf_mtok_) {
    cuda::mmq_q4_k_q8_f16out(Q4W(lw.lin_in_qkvz), d_xq_.get(),
                             lin_proj_b_.get(), g.proj_rows(), H, n, s);
    cuda::mmq_q4_k_q8_f16out(Q4W(lw.lin_in_ba), d_xq_.get(),
                             lin_ba_b_.get(), g.ba_rows(), H, n, s);
  } else {
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_in_qkvz), d_xq_.get(),
                                   lin_proj_b_.get(), g.proj_rows(), H, n, s);
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_in_ba), d_xq_.get(),
                                   lin_ba_b_.get(), g.ba_rows(), H, n, s);
  }
  if (dbg_layers_) {
    dbg_dump_f16("qkvz_proj", lin_proj_b_.get(),
                 static_cast<size_t>(n - 1) * g.proj_rows(), g.proj_rows(), s);
  }
  // causal conv over the chunk, then commit the state tail
  cuda::gdn_conv_chunk_f16(lin_conv_b_.get(), lin_proj_b_.get(), conv_st,
                           lw.lin_conv_w, lw.lin_conv_b, g.conv_dim(),
                           g.conv_kernel, n, s);
  cuda::gdn_conv_state_commit_f16(conv_st, lin_proj_b_.get(), g.conv_dim(),
                                  g.conv_kernel, n, s);
  if (dbg_layers_) {
    dbg_dump_f16("conv_out", lin_conv_b_.get(),
                 static_cast<size_t>(n - 1) * g.conv_dim(), g.conv_dim(), s);
  }
  // data-dependent gates + per-head L2 norm, batched
  cuda::gdn_gate_chunk_f32(lin_ba_b_.get(), lw.lin_A_log, lw.lin_dt_bias,
                           lin_ab_b_.get(), g.n_v_heads, n, s);
  cuda::gdn_l2_norm_chunk_f16(lin_qn_b_.get(), lin_kn_b_.get(),
                              lin_conv_b_.get(), g.conv_dim(), g.n_k_heads,
                              g.key_dim, mc_.norm_eps, n, s);
  // fused recurrence scan (state stays in shared memory across the chunk)
  if (gdn_ws_ && gdn_r4_) {
    cuda::gdn_scan_chunk_f16_warp4(delta_st, lin_y_b_.get(), lin_qn_b_.get(),
                                   lin_kn_b_.get(), lin_conv_b_.get(),
                                   g.conv_dim(), lin_ab_b_.get(), g.n_k_heads,
                                   g.key_dim, g.ratio(), g.value_dim, n, s);
  } else if (gdn_ws_) {
    cuda::gdn_scan_chunk_f16_warp(delta_st, lin_y_b_.get(), lin_qn_b_.get(),
                                  lin_kn_b_.get(), lin_conv_b_.get(),
                                  g.conv_dim(), lin_ab_b_.get(), g.n_k_heads,
                                  g.key_dim, g.ratio(), g.value_dim, n, s);
  } else {
    cuda::gdn_scan_chunk_f16(delta_st, lin_y_b_.get(), lin_qn_b_.get(),
                             lin_kn_b_.get(), lin_conv_b_.get(), g.conv_dim(),
                             lin_ab_b_.get(), g.n_k_heads, g.key_dim,
                             g.ratio(), g.value_dim, n, s);
  }
  if (dbg_layers_) {
    dbg_dump_f16("scan_y", lin_y_b_.get(),
                 static_cast<size_t>(n - 1) * g.y_rows(), g.y_rows(), s);
  }
  // output gating + out projection + residual
  cuda::gdn_out_norm_chunk_f16(lin_on_b_.get(), lin_y_b_.get(),
                               lin_proj_b_.get(), g.proj_rows(), g.conv_dim(),
                               lw.lin_o_norm, g.n_v_heads, g.value_dim,
                               mc_.norm_eps, n, s);
  if (pf_tc_) {
    // v0.9: lin_on stays fp16 into the TC GEMM (no requant).
    if (!cuda::tc_gemm_q4_k_f16(Q4W(lw.lin_out), lin_on_b_.get(),
                                o_proj_.get(), d_wtc_.get(), H, g.y_rows(),
                                n, s)) {
      SLOG_E("session", "TC prefill lin_out GEMM failed - dp4a fallback");
      cuda::quantize_q8_k(lin_on_b_.get(), d_hq_.get(), n * g.y_rows(), s);
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_out), d_hq_.get(),
                                     o_proj_.get(), H, g.y_rows(), n, s);
    }
  } else {
    cuda::quantize_q8_k(lin_on_b_.get(), d_hq_.get(), n * g.y_rows(), s);
    if (pf_mmq_ && pf_mtok_) {
      cuda::mmq_q4_k_q8_f16out(Q4W(lw.lin_out), d_hq_.get(), o_proj_.get(),
                               H, g.y_rows(), n, s);
    } else {
      cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_out), d_hq_.get(), o_proj_.get(),
                                     H, g.y_rows(), n, s);
    }
  }
  cuda::add_f16(x_.get(), o_proj_.get(), n * H, s);
  if (dbg_layers_) {
    dbg_dump_f16("lin_o_proj", o_proj_.get(),
                 static_cast<size_t>(n - 1) * H, H, s);
    dbg_dump_f16("x_after", x_.get(),
                 static_cast<size_t>(n - 1) * H, H, s);
  }
  // v0.8 FIX: the MoE half consumes ffn_norm(x) - never applied on GDN
  // layers before v0.8 (see linear_layer_launch for the full note).
  if (gdn_ffn_norm_) {
    if (dec_fuse_) {
      cuda::rmsnorm_quant_f16(xn_.get(), x_.get(), lw.ffn_norm, d_xq_.get(),
                              H, mc_.norm_eps, n, s);
    } else {
      cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.ffn_norm, H,
                                mc_.norm_eps, n, s);
      cuda::quantize_q8_k(xn_.get(), d_xq_.get(), static_cast<size_t>(n) * H, s);
    }
    if (dbg_layers_ && layer == 0) {
      dbg_dump_f16("ffn_norm_w", lw.ffn_norm, 0, 16, s);
      dbg_dump_f16("x_last_row", x_.get(),
                   static_cast<size_t>(n - 1) * H, 16, s);
      dbg_dump_f16("xn_ffn_out", xn_.get(),
                   static_cast<size_t>(n - 1) * H, 16, s);
    }
  }
}

// ---------------------------------------------------------------------------
// Graph capture
// ---------------------------------------------------------------------------
Status Session::capture_graphs() {
  const EngineDims& d = mc_.dims;
  graphs_.init(d.n_layer);

  // Warm every kernel once (module load) outside capture. Warm ONE layer of
  // each kind: dense_layer_launch reads lw.wq which is null on GDN layers -
  // so the warmup layer must be chosen by type.
  int warm_attn = -1, warm_gdn = -1;
  for (int L = 0; L < d.n_layer; ++L) {
    if (mc_.is_linear(L)) { if (warm_gdn < 0) warm_gdn = L; }
    else                  { if (warm_attn < 0) warm_attn = L; }
  }
  if (warm_attn >= 0) dense_layer_launch(warm_attn, 1);
  if (warm_gdn >= 0)  linear_layer_launch(warm_gdn, 1);
  stream_.sync();
  if (cudaError_t we = cudaGetLastError(); we != cudaSuccess) {
    SLOG_W("session", "capture warmup left error: %s", cudaGetErrorString(we));
    return Status::kCudaError;
  }

  for (int L = 0; L < d.n_layer; ++L) {
    if (cudaStreamBeginCapture(stream_.get(), cudaStreamCaptureModeThreadLocal)
        != cudaSuccess) {
      SLOG_W("session", "L%d: begin capture failed: %s", L,
             cudaGetErrorString(cudaGetLastError()));
      return Status::kCudaError;
    }
    if (mc_.is_linear(L)) linear_layer_launch(L, 1);
    else                  dense_layer_launch(L, 1);
    cudaGraph_t graph = nullptr;
    cudaError_t ec = cudaStreamEndCapture(stream_.get(), &graph);
    if (ec != cudaSuccess) {
      SLOG_W("session", "L%d: end capture failed: %s", L,
             cudaGetErrorString(ec));
      cudaGetLastError();   // pop the capture error: keep the stream usable
      return Status::kCudaError;   // for the eager fallback path
    }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
      cudaError_t ei = cudaGetLastError();
      SLOG_W("session", "L%d: instantiate failed: %s", L, cudaGetErrorString(ei));
      cudaGraphDestroy(graph);
      return Status::kCudaError;
    }
    cudaGraphDestroy(graph);
    graphs_.set_layer_graph(L, exec);
  }
  stream_.sync();
  graphs_.set_valid(true);

  // ---------------------------------------------------------------------------
  // v0.6 full-layer decode graphs: dense/GDN body + the device-dispatch MoE
  // half in ONE graph per layer. The MoE body is graph-safe at n=1: the plan
  // (histogram/prefix/scatter) is computed on device, the grouped GEMV grids
  // are host-fixed (tiles_max), weights are read through the indirect decode
  // residency table (updated in place between steps), and the mailbox writes
  // go to per-layer device slots whose pinned mirrors become memcpy nodes.
  // Replay is gated per layer on absent_cnt_[layer] == 0, so a replayed MoE
  // body can never mail. Gated layers keep the v0.5 split (dense graph +
  // eager dispatch) until promotions/rebalances open the gate.
  // ---------------------------------------------------------------------------
  if (graphs_full_valid_ && dec_device_ && pf_device_ && pump_) {
    if (cudaError_t pe0 = cudaGetLastError(); pe0 != cudaSuccess) {
      SLOG_W("session", "pre-warmup pending error: %s", cudaGetErrorString(pe0));
    }
    build_decode_table();
    stream_.sync();
    if (cudaError_t pe1 = cudaGetLastError(); pe1 != cudaSuccess) {
      SLOG_W("session", "post-table error: %s", cudaGetErrorString(pe1));
    }
    // Module-load warmup for the MoE kernels (capture records launches but
    // never executes them; the warmup runs once with zeroed routes so the
    // routed experts are the lowest ids - absent ones simply zero their rows).
    cudaMemsetAsync(route_idx_.get(), 0, (d.n_expert_used + 1) * sizeof(int),
                    stream_.get());
    cudaMemsetAsync(route_w_.get(), 0, (d.n_expert_used + 1) * sizeof(float),
                    stream_.get());
    stream_.sync();
    if (cudaError_t pe2 = cudaGetLastError(); pe2 != cudaSuccess) {
      SLOG_W("session", "post-memset error: %s", cudaGetErrorString(pe2));
    }
    moe_dispatch_device(0, 1);
    stream_.sync();
    mail_pending_ = -1;   // the warmup's mailbox state is stale - drop it
    if (cudaError_t we = cudaGetLastError(); we != cudaSuccess) {
      SLOG_W("session", "moe warmup left error: %s", cudaGetErrorString(we));
      graphs_full_valid_ = false;
    }
  }
  int full_ok = 0;
  if (graphs_full_valid_ && dec_device_ && pf_device_ && pump_) {
    for (int L = 0; L < d.n_layer; ++L) {
      if (cudaStreamBeginCapture(stream_.get(),
                                 cudaStreamCaptureModeThreadLocal) !=
          cudaSuccess) {
        cudaGetLastError();
        graphs_full_valid_ = false;
        break;
      }
      if (mc_.is_linear(L)) linear_layer_launch(L, 1);
      else                  dense_layer_launch(L, 1);
      moe_dispatch_device(L, 1);
      cudaGraph_t graph = nullptr;
      cudaError_t ec = cudaStreamEndCapture(stream_.get(), &graph);
      if (ec != cudaSuccess) {
        SLOG_W("session", "L%d: full capture failed: %s", L,
               cudaGetErrorString(cudaGetLastError()));
        cudaGetLastError();
        graphs_full_valid_ = false;
        break;
      }
      cudaGraphExec_t exec = nullptr;
      if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) !=
          cudaSuccess) {
        SLOG_W("session", "L%d: full instantiate failed: %s", L,
               cudaGetErrorString(cudaGetLastError()));
        cudaGraphDestroy(graph);
        cudaGetLastError();
        graphs_full_valid_ = false;
        break;
      }
      cudaGraphDestroy(graph);
      graphs_.set_full_layer_graph(L, exec);
      ++full_ok;
    }
    stream_.sync();
    graphs_.set_full_valid(graphs_full_valid_ && full_ok == d.n_layer);
    SLOG_I("session", "captured %d/%d full-layer decode graphs (%s)",
           full_ok, d.n_layer,
           graphs_.full_valid() ? "gate: absent_cnt==0" : "off");
  } else {
    graphs_full_valid_ = false;
    graphs_.set_full_valid(false);
  }

  // Warmup + capture DON'T execute the recorded kernels, but the warmup DID
  // mutate GDN state (one token through a linear layer) - reset it so real
  // inference starts from clean recurrence state.
  zero_gdn_states();
  SLOG_I("session", "captured %d per-layer decode graphs", d.n_layer);

  // v0.7: whole-step graph (embed + every layer) + head graph (head +
  // penalties + top-k + mirrors). Two cudaGraphLaunch calls per decode step
  // instead of ~40 layer-graph launches + eager calls; on this 2-vCPU host
  // the launch storm left the GPU idle 68% of the step.
  if (step_graph_on_ && graphs_.full_valid()) {
    const Status ssg = capture_step_graph();
    if (ssg != Status::kOk) {
      SLOG_W("session", "step-graph capture failed - per-layer path stays");
      step_graph_ok_ = false;
    }
  }
  return Status::kOk;
}

Status Session::capture_step_graph() {
  const EngineDims& d = mc_.dims;
  cudaStream_t s = stream_.get();
  int* st = h_step_stage_.get();

  // (a) step graph: pinned-stage H2D + embedding + every layer (dense/GDN
  // body + the graph-safe device-dispatch MoE with per-layer slots).
  if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess) {
    cudaGetLastError();
    return Status::kCudaError;
  }
  capturing_step_ = true;
  cudaMemcpyAsync(d_ids_.get(), st, sizeof(int), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_pos_ids_.get(), st + 1,
                  (mc_.use_mrope ? 3 : 1) * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_params_.get(), st + 8, sizeof(SeqParams),
                  cudaMemcpyHostToDevice, s);
  cuda::gather_embed(x_.get(), w_.tok_embd, d_ids_.get(), 1, d.hidden, s);
  for (int L = 0; L < d.n_layer; ++L) {
    if (mc_.is_linear(L)) linear_layer_launch(L, 1);
    else                  dense_layer_launch(L, 1);
    moe_dispatch_device(L, 1);
  }
  // v0.7: TWO bulk D2H mirror nodes replace ~120 per-layer CE round trips.
  cudaMemcpyAsync(h_mail_n_all_.get(), d_mail_n_all_.get(),
                  static_cast<size_t>(d.n_moe_layer) * sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_mail_all_.get(), d_mail_all_.get(),
                  static_cast<size_t>(d.n_moe_layer) * d.n_expert *
                      sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  capturing_step_ = false;
  cudaGraph_t g = nullptr;
  if (cudaStreamEndCapture(s, &g) != cudaSuccess) {
    SLOG_W("session", "step capture end failed: %s",
           cudaGetErrorString(cudaGetLastError()));
    return Status::kCudaError;
  }
  cudaGraphExec_t sg = nullptr;
  if (cudaGraphInstantiate(&sg, g, nullptr, nullptr, 0) != cudaSuccess) {
    SLOG_W("session", "step instantiate failed: %s",
           cudaGetErrorString(cudaGetLastError()));
    cudaGraphDestroy(g);
    cudaGetLastError();
    return Status::kCudaError;
  }
  cudaGraphDestroy(g);
  step_graph_ = sg;

  // (b) head graph: output head + penalties (device-driven count and (rep,
  // pres) params) + two-phase top-k + pinned topk mirrors + MTP stash.
  if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess) {
    cudaGetLastError();
    return Status::kCudaError;
  }
  head_launch();
  cudaMemcpyAsync(d_sample_logits_.get(), logits_.get(),
                  d.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(d_recent_.get(), h_recent_stage_.get(),
                  kMaxRecent * sizeof(int), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_recent_n_.get(), h_recent_stage_.get() + kMaxRecent,
                  sizeof(int), cudaMemcpyHostToDevice, s);
  cuda::apply_penalties_f32_devn(d_sample_logits_.get(), d_recent_.get(),
                                 d_recent_n_.get(),
                                 reinterpret_cast<const float*>(st + 10), s);
  const int ksel = 32;
  cuda::topk_select_f32_2p(d_sample_logits_.get(), d_topk_seg_v_.get(),
                           d_topk_seg_i_.get(), topk_seg_slots_,
                           d_topk_v_.get(), d_topk_i_.get(), d.n_vocab, ksel,
                           s);
  cudaMemcpyAsync(h_topk_v_.get(), d_topk_v_.get(), ksel * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_topk_i_.get(), d_topk_i_.get(), ksel * sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  // Full-logits D2H for the host-sampler route (SAGER_DEVICE_SAMPLER=0);
  // harmless overhead in the device route (mirrors are used there).
  cudaMemcpyAsync(logits_host_.get(), logits_.get(),
                  d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                  cudaMemcpyDeviceToDevice, s);
  cudaGraph_t hg = nullptr;
  if (cudaStreamEndCapture(s, &hg) != cudaSuccess) {
    SLOG_W("session", "head capture end failed: %s",
           cudaGetErrorString(cudaGetLastError()));
    return Status::kCudaError;
  }
  cudaGraphExec_t he = nullptr;
  if (cudaGraphInstantiate(&he, hg, nullptr, nullptr, 0) != cudaSuccess) {
    SLOG_W("session", "head instantiate failed: %s",
           cudaGetErrorString(cudaGetLastError()));
    cudaGraphDestroy(hg);
    cudaGetLastError();
    return Status::kCudaError;
  }
  cudaGraphDestroy(hg);
  head_graph_ = he;
  stream_.sync();
  step_graph_ok_ = true;
  SLOG_I("session",
         "captured whole-step decode graph (+head) - 2 launches/step");
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Eager MoE dispatch (one row)
// ---------------------------------------------------------------------------
void Session::moe_dispatch(int layer) { moe_dispatch_row(layer, 0); }

void Session::moe_dispatch_row(int layer, int row) {
  const EngineDims& d = mc_.dims;
  const LayerWeights& lw = w_.layers()[static_cast<size_t>(layer)];
  const int H = d.hidden;
  const int FFX = d.n_ff_exp;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  cudaStream_t s = stream_.get();

  uint16_t* xn_row = xn_.get() + static_cast<size_t>(row) * H;
  uint16_t* x_row = x_.get() + static_cast<size_t>(row) * H;

  // Router: logits -> softmax top-k (renormalized weights).
  cuda::quantize_q8_k(xn_row, d_hq_.get(), H, s);
  // Snapshot of the quantized activation for the pump path: the hits' out
  // GEMV overwrites d_hq_ with the silu'd gate activations, but pumped
  // experts are computed AFTER those launches and still need x.
  if (pump_) {
    cudaMemcpyAsync(d_xq_moe_.get(), d_hq_.get(),
                    (H / q4k::kQK_K) * sizeof(q4k::BlockQ8K),
                    cudaMemcpyDeviceToDevice, s);
  }
  cuda::gemv_f32_out_f16x(lw.router, xn_row, route_logits_.get(), E, H, s);
  cuda::router_topk(route_logits_.get(), route_idx_.get(), route_w_.get(), E,
                    K, s);

  // Shared expert into combine row K (weight 1, see prepare()).
  detail::shared_expert_forward(w_, layer, d_hq_.get(), xn_row, shared_y_.get(),
                                H, FFX, s);

  // Router decision to the host (C1). The miss-path activation row is only
  // pulled when a CPU-pool miss actually happens (saves a D2H + host wake
  // per layer per step at high pump residency).
  cudaMemcpyAsync(h_route_idx_.get(), route_idx_.get(), K * sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_route_w_.get(), route_w_.get(), K * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  stream_.sync();

  // Host-side guard: a corrupted router output must never reach the cache or
  // the pointer tables (turns a would-be OOB into a logged, skippable row).
  for (int kk = 0; kk < K; ++kk) {
    const int e0 = h_route_idx_[kk];
    if (e0 < 0 || e0 >= E) {
      SLOG_E("moe", "layer %d row %d: routed expert id %d outside [0,%d) - "
                    "skipping MoE for this row", layer, row, e0, E);
      return;
    }
  }

  int n_hit = 0, n_miss = 0, n_pump = 0;
  for (int kk = 0; kk < K; ++kk) {
    const int e = h_route_idx_[kk];
    cache_.touch(layer, e);
    if (cache_.resident(layer, e)) {
      // tier 0a: rebalance-driven hot slab (pointer tables)
      const size_t base = (static_cast<size_t>(layer) * 3 + 0) * E + e;
      h_ptrs_[0 * K + n_hit] = hot_->tables_host[base];
      h_ptrs_[1 * K + n_hit] = hot_->tables_host[base + static_cast<size_t>(E)];
      h_ptrs_[2 * K + n_hit] =
          hot_->tables_host[base + 2 * static_cast<size_t>(E)];
      h_hit_rows_[n_hit] = kk;
      h_slot_of_k_[kk] = -1;
      ++n_hit;
    } else if (pump_) {
      // tier 0b: streaming arena - DMA from the pinned RAM pool, compute on
      // the GPU after the event fence (C7a). Reuses the slot when the expert
      // was already pumped (temporal locality across tokens).
      const int slot = pump_->submit(layer, e, perf.steps + 1);
      if (slot >= 0) {
        h_pump_rows_[n_pump] = kk;
        h_pump_ptrs_[0 * K + n_pump] =
            const_cast<q4k::BlockQ4K*>(pump_->dev_base(slot));
        h_pump_ptrs_[1 * K + n_pump] = h_pump_ptrs_[0 * K + n_pump] +
            (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
        h_pump_ptrs_[2 * K + n_pump] = h_pump_ptrs_[0 * K + n_pump] +
            2 * (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
        h_slot_of_k_[kk] = -2;
        ++n_pump;
      } else {
        h_slot_of_k_[kk] = n_miss++;   // C7b fallback: CPU pool
      }
    } else {
      h_slot_of_k_[kk] = n_miss++;
    }
  }

  if (n_hit > 0) {
    cudaMemcpyAsync(d_ptrs_hit_.get(), h_ptrs_.get(),
                    3 * K * sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_hit_rows_.get(), h_hit_rows_.get(), n_hit * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    // All hits of the layer fuse into 3 GEMV launches (gate / up / out).
    cuda::gemv_ptrs_q4_k_q8_f16out(d_ptrs_hit_.get(), d_hq_.get(),
                                   expert_g_.get(), n_hit, FFX, H, s);
    cuda::gemv_ptrs_q4_k_q8_f16out(d_ptrs_hit_.get() + K, d_hq_.get(),
                                   expert_u_.get(), n_hit, FFX, H, s);
    cuda::silu_mul_f16(expert_g_.get(), expert_g_.get(), expert_u_.get(), 1,
                       n_hit * FFX, s);
    cuda::quantize_q8_k(expert_g_.get(), d_hq_.get(), n_hit * FFX, s);
    cuda::gemv_ptrs_q4_k_q8_f16out(d_ptrs_hit_.get() + 2 * K, d_hq_.get(),
                                   x2_.get(), n_hit, H, FFX, s);
    cuda::scatter_rows_f16(expert_y_.get(), x2_.get(), d_hit_rows_.get(),
                           n_hit, H, s);
  }
  cudaMemcpyAsync(d_slot_of_k_.get(), h_slot_of_k_.get(), K * sizeof(int),
                  cudaMemcpyHostToDevice, s);

  // v0.2 pump group: fence the async copies into the compute stream, then
  // run the SAME fused ptr-GEMV chain as the hits (pointers now aim at the
  // pump arena). Overlap story: the copies raced ahead on the copy stream
  // while the hits + shared expert were being computed above.
  if (n_pump > 0) {
    pump_->arm_wait(s);
    cudaMemcpyAsync(d_ptrs_pump_.get(), h_pump_ptrs_.get(),
                    3 * K * sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_pump_rows_.get(), h_pump_rows_.get(),
                    n_pump * sizeof(int), cudaMemcpyHostToDevice, s);
    cuda::gemv_ptrs_q4_k_q8_f16out(d_ptrs_pump_.get(), d_xq_moe_.get(),
                                   expert_g_.get(), n_pump, FFX, H, s);
    cuda::gemv_ptrs_q4_k_q8_f16out(d_ptrs_pump_.get() + K, d_xq_moe_.get(),
                                   expert_u_.get(), n_pump, FFX, H, s);
    cuda::silu_mul_f16(expert_g_.get(), expert_g_.get(), expert_u_.get(), 1,
                       n_pump * FFX, s);
    cuda::quantize_q8_k(expert_g_.get(), d_hq_.get(), n_pump * FFX, s);
    cuda::gemv_ptrs_q4_k_q8_f16out(d_ptrs_pump_.get() + 2 * K, d_hq_.get(),
                                   x2_.get(), n_pump, H, FFX, s);
    cuda::scatter_rows_f16(expert_y_.get(), x2_.get(), d_pump_rows_.get(),
                           n_pump, H, s);
  }

  // CPU misses (C2: no router weights here; C3: fixed pinned slots; C6:
  // submitted before the GPU finishes hits+shared so both run in parallel).
  // Expert layout in the pinned pool: [gate | up | out], each gu bytes.
  if (n_miss > 0) {
    cudaMemcpy(h_x16_.get(), xn_row, H * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < H; ++i) {
      miss_x_[i] = q4k::half_to_float(h_x16_[i]);
    }
    const size_t gu = hot_->per_expert_bytes / 3;
    std::vector<MissJob> jobs;
    jobs.reserve(static_cast<size_t>(n_miss));
    for (int kk = 0; kk < K; ++kk) {
      const int slot = h_slot_of_k_[kk];
      if (slot < 0) continue;
      const int e = h_route_idx_[kk];
      const uint8_t* exp = w_.host_expert_base() +
                           (static_cast<size_t>(layer) * E + e) *
                               hot_->per_expert_bytes;
      jobs.push_back(MissJob{layer, e,
                             reinterpret_cast<const q4k::BlockQ4K*>(exp),
                             reinterpret_cast<const q4k::BlockQ4K*>(exp + gu),
                             reinterpret_cast<const q4k::BlockQ4K*>(exp + 2 * gu),
                             miss_x_.get(),
                             miss_y_.get() + static_cast<size_t>(slot) * H,
                             H, FFX});
    }
    pool_.submit(std::move(jobs));
  }

  // Shared expert row into the combine buffer.
  cudaMemcpyAsync(expert_y_.get() + static_cast<size_t>(K) * H,
                  shared_y_.get(), H * sizeof(uint16_t),
                  cudaMemcpyDeviceToDevice, s);

  if (n_miss > 0) {
    pool_.wait_done(nullptr);  // CPU worked while the GPU ran hits/shared
    cudaMemcpyAsync(d_miss_y_.get(), miss_y_.get(),
                    static_cast<size_t>(n_miss) * H * sizeof(float),
                    cudaMemcpyHostToDevice, s);
    cuda::scatter_miss_f32(expert_y_.get(), d_slot_of_k_.get(),
                           d_miss_y_.get(), K, H, s);
  }
  cuda::moe_combine_f16(x_row, expert_y_.get(), route_idx_.get(),
                        route_w_.get(), K + 1, H, s);
}

// ---------------------------------------------------------------------------
// v0.2.3 batched whole-chunk MoE (Strata-style streamed walk)
//
// One route sync per LAYER instead of per row: quantize+route every row of
// the chunk, bring the routing decision to the host once, then
//   * hot experts      -> computed in place from the hot-slab pointer table,
//   * stream-all ring  -> slot waited via its ready event, consumed after the
//                         down projection frees it for the loader,
//   * pump LRU         -> (no active session: verify windows / tiny chunks)
//                         per-EXPERT submit + one arm_wait,
//   * CPU pool         -> last resort, capped rows (pf_cpu_cap_).
// All routed rows of an expert are compacted expert-major and computed with
// the batched GEMV kernels - no per-row dispatch, no per-row sync.
// ---------------------------------------------------------------------------
void Session::moe_dispatch_batched(int layer, int n) {
  const EngineDims& d = mc_.dims;
  const LayerWeights& lw = w_.layers()[static_cast<size_t>(layer)];
  const int H = d.hidden;
  const int FFX = d.n_ff_exp;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  cudaStream_t s = stream_.get();
  const int gu_blocks = FFX / q4k::kQK_K;
  const int h_blocks = H / q4k::kQK_K;

  // 1) quantize the whole chunk's post-attention rows once.
  cuda::quantize_q8_k(xn_.get(), d_xq_moe_.get(),
                      static_cast<size_t>(n) * H, s);

  // 2) batched router: logits + top-k for every row (stride K+1 so the
  //    shared-expert combine slot stays host-managed).
  if (pf_mtok_) {
    cuda::gemv_f32_out_f16x_mtok(lw.router, xn_.get(), route_logits_.get(),
                                 E, H, n, s);
  } else {
    cuda::gemv_f32_out_f16x_batched(lw.router, xn_.get(), route_logits_.get(),
                                    E, H, n, s);
  }
  cuda::router_topk_batched(route_logits_.get(), route_idx_.get(),
                            route_w_.get(), E, K, K + 1, n, s);

  // 3) the ONE per-layer sync: routing decision to the host.
  cudaMemcpyAsync(h_route_idx_.get(), route_idx_.get(),
                  static_cast<size_t>(n) * (K + 1) * sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_route_w_.get(), route_w_.get(),
                  static_cast<size_t>(n) * (K + 1) * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  stream_.sync();

  // 4) host pass: validate, count, classify. Everything below is plain host
  //    bookkeeping over n*K routes (~4k ints for a 512-token chunk).
  std::vector<int> count(static_cast<size_t>(E), 0);
  bool row_bad = false;
  for (int r = 0; r < n; ++r) {
    int* ids = h_route_idx_.get() + static_cast<size_t>(r) * (K + 1);
    float* wts = h_route_w_.get() + static_cast<size_t>(r) * (K + 1);
    row_bad = false;
    for (int kk = 0; kk < K; ++kk) {
      if (ids[kk] < 0 || ids[kk] >= E) { row_bad = true; break; }
    }
    if (row_bad) {
      SLOG_E("moe", "layer %d row %d: router id out of range - zeroing row",
             layer, r);
      for (int kk = 0; kk <= K; ++kk) { ids[kk] = 0; wts[kk] = 0.f; }
      continue;
    }
    wts[K] = lw.shared_gate ? 1.0f : 0.0f;   // shared-expert combine slot
    for (int kk = 0; kk < K; ++kk) ++count[static_cast<size_t>(ids[kk])];
  }

  // Expert-major compaction plan: offset per expert (id order), source rows.
  std::vector<int> off(static_cast<size_t>(E) + 1, 0);
  {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      off[static_cast<size_t>(e)] = acc;
      acc += count[static_cast<size_t>(e)];
    }
    off[static_cast<size_t>(E)] = acc;
  }
  const int R_gpu = off[static_cast<size_t>(E)];   // all valid routes (GPU +
                                                   // CPU rows both compact;
                                                   // CPU rows scatter nothing)
  // gslot map: route (r, kk) -> grouped slot; grows: slot -> source row.
  {
    std::vector<int> fill(off.begin(), off.end() - 1);
    int* gs = h_gslot_.get();
    int* gr = h_grows_.get();
    for (int r = 0; r < n; ++r) {
      const int* ids = h_route_idx_.get() + static_cast<size_t>(r) * (K + 1);
      for (int kk = 0; kk < K; ++kk) {
        const int e = ids[kk];
        int* slotp = gs + static_cast<size_t>(r) * K + kk;
        if (e < 0) { *slotp = -1; continue; }
        const int pos = fill[static_cast<size_t>(e)]++;
        *slotp = pos;
        gr[pos] = r;
      }
    }
  }
  cudaMemcpyAsync(d_gslot_.get(), h_gslot_.get(),
                  static_cast<size_t>(n) * K * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_grows_.get(), h_grows_.get(),
                  static_cast<size_t>(R_gpu) * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  // validated weights back to the device (combine reads them) - uploaded
  // after the CPU-fallback cap decision so dropped rows carry weight 0.

  // 5) classify experts + build the per-expert job list (id order).
  struct Job {
    const q4k::BlockQ4K* wg;
    const q4k::BlockQ4K* wu;
    const q4k::BlockQ4K* wo;
    int off;
    int m;
    int ring_slot;   // >= 0: consume after the down projection
  };
  std::vector<Job> jobs;
  std::vector<int> cpu_rows;      // (r * K + kk) packed, CPU-pool fallback
  jobs.reserve(static_cast<size_t>(E));
  // v0.2.4d: require LIVE loader threads, not just an active session flag -
  // a session whose loaders never spawned can never publish a slot, and
  // waiting on its lookups hangs the whole prefill (observed on Colab).
  const bool streaming = pump_ && pump_->stream_active() &&
                         pump_->stream_loader_count() > 0;
  const uint64_t step = static_cast<uint64_t>(perf.steps) + 1;
  for (int e = 0; e < E; ++e) {
    const int m = count[static_cast<size_t>(e)];
    if (m <= 0) {
      // Zero-row pair inside a live session: release its ring slot early so
      // the loaders never clog on pairs the router skipped.
      if (streaming && !cache_.resident(layer, e)) {
        pump_->stream_consume_drop(layer, e);
      }
      continue;
    }
    cache_.touch(layer, e);
    Job j{};
    j.off = off[static_cast<size_t>(e)];
    j.m = m;
    j.ring_slot = -1;
    if (cache_.resident(layer, e)) {
      // tier 0a: hot slab, computed in place (tables hold raw uint16_t*)
      const size_t base = (static_cast<size_t>(layer) * 3 + 0) * E + e;
      j.wg = reinterpret_cast<const q4k::BlockQ4K*>(hot_->tables_host[base]);
      j.wu = reinterpret_cast<const q4k::BlockQ4K*>(
          hot_->tables_host[base + static_cast<size_t>(E)]);
      j.wo = reinterpret_cast<const q4k::BlockQ4K*>(
          hot_->tables_host[base + 2 * static_cast<size_t>(E)]);
    } else if (streaming) {
      // tier 0b: stream-all ring (the pair was planned at chunk start)
      // v0.2.5b: parked survivors of an earlier chunk are pump-resident but
      // NOT in the plan - consume their existing slot directly instead of
      // waiting on a copy that never comes (16 s per pair on the Colab T4).
      if (pump_->resident(layer, e)) {
        const int pslot = pump_->stream_slot_of(layer, e);
        if (pslot >= 0) {
          pump_->stream_wait(s, pslot);
          j.ring_slot = pslot;
          auto* base = const_cast<q4k::BlockQ4K*>(pump_->dev_base(pslot));
          j.wg = base;
          j.wu = base + (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
          j.wo = base + 2 * (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
        }
      }
      if (j.ring_slot < 0) {
      // v0.2.4b: a lookup timeout on the CURRENT layer's pair means the
      // loader is merely BEHIND (the walk is sequential and the pair is
      // still ahead of it - the loader only skips pairs of layers that
      // already passed, and we ARE the current layer). v0.2.5b: the lookup
      // itself now waits up to 2 x arm_timeout (30 s default) for the
      // copy while the readers are live; retries exist only to re-arm the
      // clock after a self-reclaim pass, so 2 is plenty.
      int slot = pump_->stream_lookup(layer, e);
      for (int retry = 0; slot < 0 && retry < 2; ++retry) {
        slot = pump_->stream_lookup(layer, e);
      }
      if (slot >= 0) {
        pump_->stream_wait(s, slot);
        j.ring_slot = slot;
        auto* base = const_cast<q4k::BlockQ4K*>(pump_->dev_base(slot));
        j.wg = base;
        j.wu = base + (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
        j.wo = base + 2 * (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
      } else {
        // Loader genuinely failed (read error / wedged session): give up
        // on the pair for this chunk AND release its slot if the loader
        // did land it meanwhile (no GEMM will read it).
        pump_->stream_consume_drop(layer, e);
      }
      }  // v0.2.5b: if (j.ring_slot < 0)
    } else if (pump_) {
      // tier 0c: per-expert pump submit (verify windows / small chunks)
      const int slot = pump_->submit(layer, e, step);
      if (slot >= 0) {
        auto* base = const_cast<q4k::BlockQ4K*>(pump_->dev_base(slot));
        j.wg = base;
        j.wu = base + (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
        j.wo = base + 2 * (hot_->per_expert_bytes / 3) / q4k::kBlockBytes;
      }
    }
    if (!j.wg) {
      for (int r = 0; r < n; ++r) {
        const int* ids = h_route_idx_.get() + static_cast<size_t>(r) * (K + 1);
        for (int kk = 0; kk < K; ++kk) {
          if (ids[kk] == e) cpu_rows.push_back(r * K + kk);
        }
      }
    } else {
      jobs.push_back(j);
    }
  }
  if (pump_ && pump_->pending()) pump_->arm_wait(s);

  // CPU-fallback cap decision BEFORE the weight upload: rows beyond the cap
  // get their routed weight zeroed (their combine slots stay stale). Build
  // the (r, kk) list first, then upload the final weights.
  int n_cpu = 0;
  if (!cpu_rows.empty()) {
    if (static_cast<int>(cpu_rows.size()) > pf_cpu_cap_) {
      SLOG_W("moe", "layer %d: %zu CPU rows exceed cap %d - dropping tail",
             layer, cpu_rows.size(), pf_cpu_cap_);
      for (size_t i = static_cast<size_t>(pf_cpu_cap_); i < cpu_rows.size(); ++i) {
        const int rk = cpu_rows[i];
        h_route_w_.get()[static_cast<size_t>(rk / K) * (K + 1) + rk % K] = 0.f;
      }
      cpu_rows.resize(static_cast<size_t>(pf_cpu_cap_));
    }
    n_cpu = static_cast<int>(cpu_rows.size());
    // host copy of the chunk rows (fp16) for the CPU GEMVs
    cudaMemcpyAsync(h_x16_.get(), xn_.get(),
                    static_cast<size_t>(n) * H * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost, s);
  }
  cudaMemcpyAsync(route_w_.get(), h_route_w_.get(),
                  static_cast<size_t>(n) * (K + 1) * sizeof(float),
                  cudaMemcpyHostToDevice, s);

  // 6) compact the routed activation rows expert-major (one D2D gather).
  if (R_gpu > 0) {
    cuda::gather_rows_q8(d_xqg_.get(), d_xq_moe_.get(), d_grows_.get(),
                         R_gpu, h_blocks, s);
  }

  // 6b) v0.3 grouped job tables: ALL experts of the layer fuse into ONE
  //     gate+up launch and ONE down launch (tiles flattened over jobs).
  const int n_jobs = static_cast<int>(jobs.size());
  int n_tiles = 0;
  int tile_gu = 8;
  if (pf_mtok_ && n_jobs > 0) {
    int t = (49152 - 16 * h_blocks * static_cast<int>(sizeof(q4k::BlockQ4K)) -
             512) / (h_blocks * static_cast<int>(sizeof(q4k::BlockQ8K)));
    tile_gu = t < 4 ? 4 : (t > 16 ? 16 : t);
    for (int j = 0; j < n_jobs; ++j) {
      h_job_wg_[j] = const_cast<q4k::BlockQ4K*>(jobs[j].wg);
      h_job_wu_[j] = const_cast<q4k::BlockQ4K*>(jobs[j].wu);
      h_job_wo_[j] = const_cast<q4k::BlockQ4K*>(jobs[j].wo);
      h_job_off_[j] = jobs[j].off;
      h_job_m_[j] = jobs[j].m;
      h_job_tbase_[j] = n_tiles;
      const int tiles = (jobs[j].m + tile_gu - 1) / tile_gu;
      for (int q = 0; q < tiles; ++q) h_job_jot_[n_tiles + q] = j;
      n_tiles += tiles;
    }
    cudaMemcpyAsync(d_job_wg_.get(), h_job_wg_.get(),
                    n_jobs * sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_job_wu_.get(), h_job_wu_.get(),
                    n_jobs * sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_job_wo_.get(), h_job_wo_.get(),
                    n_jobs * sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_job_off_.get(), h_job_off_.get(),
                    n_jobs * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_job_m_.get(), h_job_m_.get(),
                    n_jobs * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_job_tbase_.get(), h_job_tbase_.get(),
                    n_jobs * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_job_jot_.get(), h_job_jot_.get(),
                    n_tiles * sizeof(int), cudaMemcpyHostToDevice, s);
  }

  // 7) gate + up for every GPU expert (id order = stream order: the waits
  //    were armed above, so copies stream ahead of these launches).
  if (pf_mtok_) {
    if (n_tiles > 0) {
      cuda::gemv_grouped2_q4_k_q8_f16out_mtok(
          d_job_wg_.get(), d_job_wu_.get(), d_job_off_.get(), d_job_m_.get(),
          d_job_tbase_.get(), d_job_jot_.get(), n_tiles, d_xqg_.get(),
          expert_g2_.get(), expert_u2_.get(), FFX, H, tile_gu, s);
    }
  } else {
    for (const Job& j : jobs) {
      cuda::gemv_q4_k_q8_f16out_batched(
          j.wg, d_xqg_.get() + static_cast<size_t>(j.off) * h_blocks,
          expert_g2_.get() + static_cast<size_t>(j.off) * FFX, FFX, H, j.m, s);
      cuda::gemv_q4_k_q8_f16out_batched(
          j.wu, d_xqg_.get() + static_cast<size_t>(j.off) * h_blocks,
          expert_u2_.get() + static_cast<size_t>(j.off) * FFX, FFX, H, j.m, s);
    }
  }

  // 8) fused activation + requant over the whole compacted slab.
  if (R_gpu > 0) {
    cuda::silu_mul_f16(expert_g2_.get(), expert_g2_.get(), expert_u2_.get(),
                       1, R_gpu * FFX, s);
    cuda::quantize_q8_k(expert_g2_.get(), d_hqg_.get(),
                        static_cast<size_t>(R_gpu) * FFX, s);
  }

  // 9) down projection per expert; ring slots are consumed right after.
  if (pf_mtok_) {
    if (n_tiles > 0) {
      cuda::gemv_grouped_q4_k_q8_f16out_mtok(
          d_job_wo_.get(), d_job_off_.get(), d_job_m_.get(),
          d_job_tbase_.get(), d_job_jot_.get(), n_tiles, d_hqg_.get(),
          y_grouped_.get(), H, FFX, tile_gu, s);
    }
  } else {
    for (const Job& j : jobs) {
      cuda::gemv_q4_k_q8_f16out_batched(
          j.wo, d_hqg_.get() + static_cast<size_t>(j.off) * gu_blocks,
          y_grouped_.get() + static_cast<size_t>(j.off) * H, H, FFX, j.m, s);
    }
  }
  for (const Job& j : jobs) {
    if (j.ring_slot >= 0) pump_->stream_consume(s, j.ring_slot);
  }

  // 10) compact outputs into the combine slab (expert_y_ [n x (K+1) x H]).
  if (R_gpu > 0) {
    cuda::scatter_routed_f16(expert_y_.get(), y_grouped_.get(),
                             d_gslot_.get(), n, K, H, s);
  }

  // 11) shared expert, batched (dense MLP over all chunk rows).
  if (lw.shared_gate) {
    if (pf_mtok_) {
      cuda::gemv2_q4_k_q8_f16out_mtok(
          reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate),
          reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up),
          d_xq_moe_.get(), sh_g_.get(), sh_u_.get(), FFX, H, n, s);
    } else {
      cuda::gemv_q4_k_q8_f16out_batched(
          reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate),
          d_xq_moe_.get(), sh_g_.get(), FFX, H, n, s);
      cuda::gemv_q4_k_q8_f16out_batched(
          reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up),
          d_xq_moe_.get(), sh_u_.get(), FFX, H, n, s);
    }
    cuda::silu_mul_f16(sh_g_.get(), sh_g_.get(), sh_u_.get(), 1, n * FFX, s);
    cuda::quantize_q8_k(sh_g_.get(), d_hqg_.get(),
                        static_cast<size_t>(n) * FFX, s);
    if (pf_mtok_) {
      cuda::gemv_q4_k_q8_f16out_mtok(
          reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out), d_hqg_.get(),
          shared_yb_.get(), H, FFX, n, s);
    } else {
      cuda::gemv_q4_k_q8_f16out_batched(
          reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out), d_hqg_.get(),
          shared_yb_.get(), H, FFX, n, s);
    }
    if (lw.shared_gate_w) {
      cuda::gemv_f32_out_f16x_batched(lw.shared_gate_w, xn_.get(),
                                      d_sgate_.get(), 1, H, n, s);
      cuda::sigmoid_gate_scalar_batched_f16(shared_yb_.get(), d_sgate_.get(),
                                            n, H, s);
    }
    // shared plane (slot K) of the combine slab, strided D2D copy
    cudaMemcpy2DAsync(
        expert_y_.get() + static_cast<size_t>(K) * H,
        static_cast<size_t>(K + 1) * H * sizeof(uint16_t),
        shared_yb_.get(), H * sizeof(uint16_t),
        H * sizeof(uint16_t), n, cudaMemcpyDeviceToDevice, s);
  }

  // 12) CPU-pool fallback (rare: arm timeouts / read errors). Capped rows.
  if (n_cpu > 0) {
    stream_.sync();   // h_x16_ D2H complete
    const size_t gu = hot_->per_expert_bytes / 3;
    std::vector<MissJob> mjobs;
    mjobs.reserve(cpu_rows.size());
    for (int i = 0; i < n_cpu; ++i) {
      const int rk = cpu_rows[static_cast<size_t>(i)];
      const int r = rk / K, kk = rk % K;
      const int e = h_route_idx_.get()[static_cast<size_t>(r) * (K + 1) + kk];
      float* mx = miss_x_.get() + static_cast<size_t>(i) * H;   // per-row slice
      const uint16_t* x16 = h_x16_.get() + static_cast<size_t>(r) * H;
      for (int t = 0; t < H; ++t) mx[t] = q4k::half_to_float(x16[t]);
      const uint8_t* exp = w_.host_expert_base() +
          (static_cast<size_t>(layer) * E + e) * hot_->per_expert_bytes;
      mjobs.push_back(MissJob{
          layer, e, reinterpret_cast<const q4k::BlockQ4K*>(exp),
          reinterpret_cast<const q4k::BlockQ4K*>(exp + gu),
          reinterpret_cast<const q4k::BlockQ4K*>(exp + 2 * gu), mx,
          miss_y_.get() + static_cast<size_t>(i) * H, H, FFX});
      h_cflat_.get()[i] = static_cast<int>(
          (static_cast<size_t>(r) * (K + 1) + kk) * H);
    }
    pool_.submit(std::move(mjobs));
    pool_.wait_done(nullptr);
    cudaMemcpyAsync(d_miss_y_.get(), miss_y_.get(),
                    static_cast<size_t>(n_cpu) * H * sizeof(float),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_cflat_.get(), h_cflat_.get(),
                    static_cast<size_t>(n_cpu) * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    cuda::scatter_rows_generic_f16(expert_y_.get(), d_miss_y_.get(),
                                   d_cflat_.get(), n_cpu, H, s);
  }

  // 13) combine: x_ = residual + sum_k w * expert_y (shared slot folded in).
  cuda::moe_combine_batched_f16(x_.get(), expert_y_.get(), route_w_.get(),
                                K, H, n, s);

  // 14) session progress: every slot owned by a pair of an EARLIER layer is
  // reclaimable now (consumed ones once their event completes; parked ones
  // immediately; loaders skip pairs of past layers). Without this sweep,
  // zero-count pairs loaded after their layer passed clog the ring until it
  // starves - the 2 tok/s failure mode of the first streamed run.
  if (streaming) pump_->stream_progress(layer + 1);
}

// ---------------------------------------------------------------------------
// v0.4 device-side MoE dispatch (stream-all prefill). Routing, grouping and
// tier selection stay ON THE DEVICE - the per-layer D2H sync, the host
// bookkeeping and the job-table uploads of the v0.2.3 path are gone. Weight
// sources come from the chunk pointer table: hot slab / parked arena slots
// were written at chunk start; streaming pairs are patched by their loader
// before the pair's publish event, and the compute stream waits on exactly
// those events (device-side, host never blocks). Absent experts contribute
// zero and are mailed to the host; resolve_cpu_mailbox() patches them later
// without stalling the healthy path.
// ---------------------------------------------------------------------------
#define MOE_DBG(tag) \
  do { \
    if (!moe_dbg_ || capturing) break; \
    cudaError_t e_ = cudaStreamSynchronize(s); \
    cudaError_t e2_ = cudaGetLastError(); \
    if (e_ != cudaSuccess || e2_ != cudaSuccess) \
      SLOG_E("moe", "dbg %s: sync=%s last=%s", tag, cudaGetErrorString(e_), \
             cudaGetErrorString(e2_)); \
  } while (0)

void Session::moe_dispatch_device(int layer, int n) {
  const EngineDims& d = mc_.dims;
  const LayerWeights& lw = w_.layers()[static_cast<size_t>(layer)];
  const int H = d.hidden;
  const int FFX = d.n_ff_exp;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  cudaStream_t s = stream_.get();
  const int h_blocks = H / q4k::kQK_K;
  const long Rmax = static_cast<long>(n) * K;

  // v0.6: during full-layer graph capture the MoE body runs against the
  // per-layer mailbox slots (fixed addresses, one slice per layer) and skips
  // the host-side event/mail_pending_ handshake - the captured graph has no
  // host interplay at all; the pinned mirrors below become memcpy nodes and
  // the host reads them after the step sync.
  cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
  const bool capturing =
      cudaStreamIsCapturing(s, &cap_st) == cudaSuccess &&
      cap_st == cudaStreamCaptureStatusActive;
  int* mail_l = capturing ? d_mail_all_.get() + static_cast<size_t>(layer) * E
                          : d_mail_.get();
  int* mail_n_l = capturing ? d_mail_n_all_.get() + layer : d_mail_n_.get();
  // v0.6: in capture mode every plan buffer is a per-layer slice so the
  // step-end patch can still see this layer's dispatch state after later
  // layers overwrote the shared single-step buffers.
  int* gslot_l = capturing
      ? d_gslot_all_.get() + static_cast<size_t>(layer) * K : d_gslot_.get();
  int* grows_l = capturing
      ? d_grows_all_.get() + static_cast<size_t>(layer) * K : d_grows_.get();
  int* off_l = capturing
      ? d_off_all_.get() + static_cast<size_t>(layer) * (E + 1)
      : d_off_.get();
  int* toff_l = capturing
      ? d_toff_all_.get() + static_cast<size_t>(layer) * (E + 1)
      : d_toff_.get();
  int* meta_l = capturing
      ? d_meta_all_.get() + static_cast<size_t>(layer) * 4 : d_meta_.get();

  // 1) quantize + route (same kernels as the host path - bit-identical).
  // v0.8: the layer body's fused norm+quant already produced q8(xn) in
  // d_xq_ - reuse it instead of re-quantizing (bit-identical values).
  if (!moe_xq_in_) {
    cuda::quantize_q8_k(xn_.get(), d_xq_moe_.get(),
                        static_cast<size_t>(n) * H, s);
  }
  const q4k::BlockQ8K* xq_moe = moe_xq_in_ ? moe_xq_in_ : d_xq_moe_.get();
  if (capturing) {
    // preserve this layer's input row for the step-end CPU patch (the next
    // layer's rmsnorm overwrites xn_ inside the same graph)
    cudaMemcpyAsync(d_x16_all_.get() + static_cast<size_t>(layer) * H,
                    xn_.get(), H * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                    s);
  }
  // v0.8: the router GEMV stays multi-block (2 MB of fp32 W through one SM
  // is a loss); the win is the batched top-k kernel's warp-parallel
  // selection (below in moe.cu) - the v0.7 kernel selected serially in
  // thread 0 (~20-40 us per layer).
  // v0.12: TC router for large chunks (SAGER_ROUTER_TC) - the dp4a GEMV
  // re-reads the 2 MB fp32 W per token tile; the TC path casts W to fp16
  // once (1 MB scratch at the front of d_wtc_, which is free until the MoE
  // ring builds below) and runs one GemmEx with fp32 C. Same output dtype;
  // logits differ at the fp16-rounding level (top-k near-ties can flip).
  if (pf_router_tc_ && n >= 256 && !capturing &&
      d_wtc_.count() >= static_cast<size_t>(E) * H &&
      cuda::tc_handle() != nullptr) {
    cuda::tc_router_f32(lw.router, xn_.get(), route_logits_.get(),
                        d_wtc_.get(), E, H, n, s);
  } else {
    cuda::gemv_f32_out_f16x_mtok(lw.router, xn_.get(), route_logits_.get(),
                                 E, H, n, s);
  }
  cuda::router_topk_batched(route_logits_.get(), route_idx_.get(),
                            route_w_.get(), E, K, K + 1, n, s);
  // v0.12 debug: router health on the first MoE layer (SAGER_DUMP_LOGITS=2)
  if (dbg_layers_ && layer == 0 && !capturing) {
    dbg_dump_f32("route_logits0", route_logits_.get(),
                 static_cast<size_t>(n - 1) * E, 16, s);
    dbg_dump_f32("route_w0", route_w_.get(),
                 static_cast<size_t>(n - 1) * (K + 1), K + 1, s);
  }
  (void)dec_fuse_;
  if (capturing) {
    cudaMemcpyAsync(d_route_w_all_.get() +
                        static_cast<size_t>(layer) * (K + 1),
                    route_w_.get(), (K + 1) * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
  }

  // 2) device plan: histogram + prefix sums + grouped scatter + mailbox.
  int tile_gu = 8;
  {
    int t = (49152 - 16 * h_blocks * static_cast<int>(sizeof(q4k::BlockQ4K)) -
             512) / (h_blocks * static_cast<int>(sizeof(q4k::BlockQ8K)));
    tile_gu = t < 4 ? 4 : (t > 16 ? 16 : t);
  }
  cuda::moe_dispatch_scan(route_idx_.get(), route_w_.get(), gslot_l,
                          grows_l, off_l, toff_l,
                          meta_l, mail_l, mail_n_l,
                          n, K, E, tile_gu, s);
  MOE_DBG("scan");
  if (capturing && !capturing_step_) {
    // Pinned mirrors as memcpy nodes; the host checks them after the step
    // sync and runs the CPU-pool patch for any absent experts (rare - the
    // decode residency covers ~99% of routed rows; see patch_graph_mailbox).
    // v0.7: SKIPPED while capturing the whole-step graph - 40 layers x 3
    // tiny D2H nodes cost ~100 us of copy-engine round trip EACH on T4
    // (~24 ms/step!). The whole-step graph copies both packed mirrors with
    // TWO bulk D2H nodes at its end instead.
  } else if (!capturing) {
    // mailbox count mirror (lands before ev_scan_ fires)
    cudaMemcpyAsync(h_mail_n_.get(), d_mail_n_.get(), sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev_scan_, s);
    mail_pending_ = layer;
  }
  // chunk-level touch feed: async route-id mirror (consumed by
  // drain_expert_touch at the chunk boundary - no per-layer sync). Layer
  // index == MoE-layer index for this arch (every block is MoE).
  // v0.7: not emitted during whole-step capture (CE round trips); the step
  // path skips drain_expert_touch, cache decay via tick() still runs.
  if (!capturing_step_) {
    cudaMemcpyAsync(
        h_touch_idx_.get() + static_cast<size_t>(layer) * pf_rt_ * (K + 1),
        route_idx_.get(), static_cast<size_t>(n) * (K + 1) * sizeof(int),
        cudaMemcpyDeviceToHost, s);
  }

  // 3) compact routed rows expert-major (row count read on device).
  //    v0.10: the MoE tensor-core path compacts straight from the fp16 input
  //    (GemmEx consumes fp16 activations - no Q8 detour, no requant).
  //    Large chunks only (m_e ~ n*K/E makes the GemmEx loop pay off; small
  //    verify/decode windows stay on the dp4a grouped GEMV).
  const size_t lstride = static_cast<size_t>(d.n_moe_layer) * E;
  const bool moe_tc = pf_moe_tc_ && ev_moe_plan_ && n >= 256 && !capturing &&
                      moe_grp_ >= 2 && moe_slot_elems_ > 0 &&
                      d_wtc_.count() >=
                          static_cast<size_t>(moe_grp_) * moe_slot_elems_ &&
                      cuda::tc_handle() != nullptr;
  if (moe_tc) {
    cuda::gather_rows_f16_n(expert_y_.get(), xn_.get(), grows_l,
                            meta_l, static_cast<int>(Rmax), H, s);
  } else {
    cuda::gather_rows_q8_n(d_xqg_.get(), xq_moe, grows_l,
                           meta_l, static_cast<int>(Rmax), h_blocks, s);
  }

  // 4) device-side wait: this layer's planned pairs must have landed (their
  //    loaders patch the pointer table before firing the publish events).
  //    Decode fast path runs WITHOUT a stream session - the pointer table is
  //    the decode-built residency table, nothing ever patches it mid-step.
  if (pump_->stream_active()) pump_->stream_wait_layer(layer, s);

  if (moe_tc) {
    // 5t) plan + residency mirrors, one host sync per layer (the grouped
    //     GemmEx loop needs per-expert m and table residency on the host;
    //     the D2H lands after the pump publishes this layer's pairs - the
    //     stream only has to drain through the gather at this point).
    const int Lm = d.n_moe_layer;
    cudaMemcpyAsync(h_off_.get(), off_l, (E + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_moe_meta_.get(), meta_l, 2 * sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    for (int t = 0; t < 3; ++t) {
      cudaMemcpyAsync(
          h_ptr_table_.get() + (static_cast<size_t>(t) * Lm + layer) * E,
          d_ptr_table_.get() + (static_cast<size_t>(t) * Lm + layer) * E,
          E * sizeof(q4k::BlockQ4K*), cudaMemcpyDeviceToHost, s);
    }
    cudaEventRecord(ev_moe_plan_, s);
    cudaEventSynchronize(ev_moe_plan_);
    MOE_DBG("moe_tc_plan");

    uint16_t* xg = expert_y_.get();   // fp16 routed rows (life-disjoint)
    uint16_t* yg2 = expert_g2_.get();
    uint16_t* yu2 = expert_u2_.get();
    uint16_t* yd = y_grouped_.get();
    const size_t ls = static_cast<size_t>(Lm) * E;
    const q4k::BlockQ4K* const* hg =
        h_ptr_table_.get() + static_cast<size_t>(layer) * E;
    moe_absent_.clear();
    // 5u) grouped two-phase pipeline. The fp16 ring inside d_wtc_ has two
    //     regions: [G slots of gate+up][G slots of down]. Phase 1 walks the
    //     experts in groups of G, dequantizing gate+up into region 0 and
    //     immediately enqueueing the group's GemmEx pairs - the next group's
    //     dequant reuses the slots only after those GEMMs are enqueued
    //     (in-order stream keeps the ring coherent). Phase 2 repeats the
    //     pattern for the down matrix in region 1, after the silu.
    const int G = moe_grp_;   // power of two
    const size_t gu_slot = 2ull * FFX * H;              // fp16 elems / slot
    const size_t dn_slot = static_cast<size_t>(H) * FFX;
    uint16_t* ring_gu = d_wtc_.get();
    uint16_t* ring_dn = d_wtc_.get() + G * gu_slot;
    for (int e = 0; e < E; ++e) {
      const int o = h_off_.get()[e];
      if (h_off_.get()[e + 1] > o &&
          (!hg[e] || !hg[ls + e] || !hg[2 * ls + e])) {
        moe_absent_.push_back(e);
      }
    }
    if (!moe_absent_.empty()) {
      // absent experts: zero their down-output rows so the combine adds 0
      // (the AVX2 CPU-pool patch re-adds them at the next layer boundary).
      SLOG_W("moe", "layer %d: %zu absent expert(s) - TC mailbox patch",
             layer, moe_absent_.size());
      for (int e : moe_absent_) {
        const int o = h_off_.get()[e];
        const int m_e = h_off_.get()[e + 1] - o;
        if (m_e > 0) {
          cudaMemsetAsync(y_grouped_.get() + static_cast<size_t>(o) * H, 0,
                          static_cast<size_t>(m_e) * H * sizeof(uint16_t), s);
        }
      }
    }
    // gate + up: C_row[m_e x FFX] = X_row[m_e x H] * W_row[FFX x H]^T
    for (int g0 = 0; g0 < E; g0 += G) {
      const int ge = std::min(G, E - g0);
      cuda::dequant_experts_q4_k_f16(
          d_ptr_table_.get() + 0 * lstride + g0,
          d_ptr_table_.get() + 1 * lstride + g0, off_l + g0,
          ring_gu, 0, 10, H, ge, G,
          static_cast<long long>(gu_slot), s);
      for (int e = g0; e < g0 + ge; ++e) {
        const int o = h_off_.get()[e];
        const int m_e = h_off_.get()[e + 1] - o;
        if (m_e <= 0 || !hg[e] || !hg[ls + e] || !hg[2 * ls + e]) continue;
        const uint16_t* A =
            ring_gu + static_cast<size_t>(e & (G - 1)) * gu_slot;
        const uint16_t* B = xg + static_cast<size_t>(o) * H;
        if (!cuda::tc_gemm_f16_raw(A, B, yg2 + static_cast<size_t>(o) * FFX,
                                   FFX, H, m_e, s) ||
            !cuda::tc_gemm_f16_raw(A + FFX * H, B,
                                   yu2 + static_cast<size_t>(o) * FFX,
                                   FFX, H, m_e, s)) {
          SLOG_E("moe", "layer %d: MoE TC gate/up GemmEx failed (e=%d)",
                 layer, e);
        }
      }
    }
    // SiLU(g)*U over the compacted slab (device row count, unchanged math).
    cuda::silu_mul_f16_n(expert_g2_.get(), expert_g2_.get(),
                         expert_u2_.get(), meta_l, static_cast<int>(Rmax),
                         FFX, s);
    MOE_DBG("silu_n");
    // down: C_row[m_e x H] = Hsilu_row[m_e x FFX] * W_row[H x FFX]^T
    for (int g0 = 0; g0 < E; g0 += G) {
      const int ge = std::min(G, E - g0);
      cuda::dequant_experts_q4_k_f16(
          d_ptr_table_.get() + 2 * lstride + g0, nullptr, off_l + g0,
          ring_dn, 0, 11, FFX, ge, G,
          static_cast<long long>(dn_slot), s);
      for (int e = g0; e < g0 + ge; ++e) {
        const int o = h_off_.get()[e];
        const int m_e = h_off_.get()[e + 1] - o;
        if (m_e <= 0 || !hg[e] || !hg[ls + e] || !hg[2 * ls + e]) continue;
        const uint16_t* A =
            ring_dn + static_cast<size_t>(e & (G - 1)) * dn_slot;
        const uint16_t* B = yg2 + static_cast<size_t>(o) * FFX;
        if (!cuda::tc_gemm_f16_raw(A, B, yd + static_cast<size_t>(o) * H,
                                   H, FFX, m_e, s)) {
          SLOG_E("moe", "layer %d: MoE TC down GemmEx failed (e=%d)",
                 layer, e);
        }
      }
    }
    // the Q4_K pump ring slots were last read by the down dequant above
    if (pump_->stream_active()) pump_->stream_consume_layer(layer, s);
    if (!moe_absent_.empty()) {
      // Feed the standard mailbox: resolve_cpu_mailbox(layer) fires at the
      // next layer boundary - the combine is already enqueued, this layer's
      // plan state (d_off_/d_gslot_/d_grows_) is still intact and xn_ has
      // not been overwritten yet - and runs the AVX2 CPU-pool patch with the
      // exact v0.4 semantics (combine saw zeros; the patch adds the rows).
      const int cnt = static_cast<int>(moe_absent_.size());
      cudaMemcpyAsync(d_mail_.get(), moe_absent_.data(),
                      static_cast<size_t>(cnt) * sizeof(int),
                      cudaMemcpyHostToDevice, s);
      cudaMemcpyAsync(d_mail_n_.get(), &cnt, sizeof(int),
                      cudaMemcpyHostToDevice, s);
      cudaMemcpyAsync(h_mail_n_.get(), d_mail_n_.get(), sizeof(int),
                      cudaMemcpyDeviceToHost, s);
      cudaEventRecord(ev_scan_, s);
      mail_pending_ = layer;
    }
  } else {
  // 5) gate+up for EVERY routed expert of the layer in ONE launch.
  //    tiles <= total routed rows (each tile covers >= 1 row) - for the 1-row
  //    decode step this caps the grid at Rmax (=K) tiles instead of the
  //    residency-worst-case K/tile + E + 16, which is 94% empty blocks.
  const int tiles_max = std::min(static_cast<int>(Rmax) / tile_gu + E + 16,
                                 static_cast<int>(Rmax));
  cuda::gemv_grouped2_tab_q4_k_q8_f16out_mtok(
      d_ptr_table_.get() + 0 * lstride,
      d_ptr_table_.get() + 1 * lstride,
      off_l, toff_l, meta_l, mail_l, mail_n_l,
      tiles_max, d_xqg_.get(), expert_g2_.get(),
      expert_u2_.get(), FFX, H, E, tile_gu, s);
  MOE_DBG("gu_tab");

  // 6) activation + requant over the compacted slab (device row count).
  cuda::silu_mul_f16_n(expert_g2_.get(), expert_g2_.get(), expert_u2_.get(),
                       meta_l, static_cast<int>(Rmax), FFX, s);
  MOE_DBG("silu_n");
  cuda::quantize_q8_k_n(expert_g2_.get(), d_hqg_.get(), meta_l, FFX,
                        static_cast<int>(Rmax) * (FFX / q4k::kQK_K), s);
  MOE_DBG("requant_n");

  // 7) down projection (last reader of the ring slots; absent experts'
  //    output rows are zeroed in-kernel so the combine adds 0).
  cuda::gemv_grouped_tab_q4_k_q8_f16out_mtok(
      d_ptr_table_.get() + 2 * lstride, off_l, toff_l,
      meta_l, tiles_max, d_hqg_.get(), y_grouped_.get(), H, FFX, E,
      tile_gu, s);
  MOE_DBG("down_tab");

  // 8) ring slots of this layer's planned pairs free after the down GEMM.
  if (pump_->stream_active()) pump_->stream_consume_layer(layer, s);
  }

  // 9) compact outputs into the combine slab (gslot built on device).
  cuda::scatter_routed_f16(expert_y_.get(), y_grouped_.get(), gslot_l,
                           n, K, H, s);

  // 10) shared expert, batched (identical to the host path).
  //     v0.12: TC path for large chunks (SAGER_SH_TC). The dp4a GEMV family
  //     here costs ~18% of the prefill GPU time (gemv2 gate+up + down
  //     mtok); the TC path dequantizes each weight once into the front of
  //     d_wtc_ (the MoE fp16 ring is dead after the down phase above -
  //     in-stream reuse is safe) and needs NO Q8 requant for the down
  //     input (the fp16 silu output feeds GemmEx directly).
  if (lw.shared_gate) {
    const bool sh_tc = pf_sh_tc_ && n >= 256 && !capturing &&
                       d_wtc_.count() >= 3ull * FFX * H &&
                       cuda::tc_handle() != nullptr;
    // v0.8 (n=1): write the shared row straight into the combine slab's
    // reserved slot K instead of a separate buffer + 2D memcpy node.
    uint16_t* sh_out = (n == 1)
        ? expert_y_.get() + static_cast<size_t>(K) * H
        : shared_yb_.get();
    if (sh_tc) {
      uint16_t* wsc = d_wtc_.get();
      if (!cuda::tc_gemm_q4_k_f16(
              reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate),
              xn_.get(), sh_g_.get(), wsc, FFX, H, n, s) ||
          !cuda::tc_gemm_q4_k_f16(
              reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up),
              xn_.get(), sh_u_.get(), wsc + static_cast<size_t>(FFX) * H,
              FFX, H, n, s)) {
        SLOG_E("moe", "layer %d: shared gate/up TC failed - dp4a fallback",
               layer);
        cuda::quantize_q8_k(xn_.get(), d_xq_moe_.get(),
                            static_cast<size_t>(n) * H, s);
        cuda::gemv2_q4_k_q8_f16out_mtok(
            reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate),
            reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up),
            d_xq_moe_.get(), sh_g_.get(), sh_u_.get(), FFX, H, n, s);
      }
      cuda::silu_mul_f16(sh_g_.get(), sh_g_.get(), sh_u_.get(), 1, n * FFX, s);
      if (!cuda::tc_gemm_q4_k_f16(
              reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out),
              sh_g_.get(), sh_out, wsc + 2ull * FFX * H, H, FFX, n, s)) {
        SLOG_E("moe", "layer %d: shared down TC failed - dp4a fallback",
               layer);
        cuda::quantize_q8_k(sh_g_.get(), d_hqg_.get(),
                            static_cast<size_t>(n) * FFX, s);
        cuda::gemv_q4_k_q8_f16out_mtok(
            reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out),
            d_hqg_.get(), sh_out, H, FFX, n, s);
      }
    } else {
    cuda::gemv2_q4_k_q8_f16out_mtok(
        reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate),
        reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up),
        xq_moe, sh_g_.get(), sh_u_.get(), FFX, H, n, s);
    cuda::silu_mul_f16(sh_g_.get(), sh_g_.get(), sh_u_.get(), 1, n * FFX, s);
    cuda::quantize_q8_k(sh_g_.get(), d_hqg_.get(),
                        static_cast<size_t>(n) * FFX, s);
    cuda::gemv_q4_k_q8_f16out_mtok(
        reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out), d_hqg_.get(),
        sh_out, H, FFX, n, s);
    }
    if (lw.shared_gate_w) {
      cuda::gemv_f32_out_f16x_batched(lw.shared_gate_w, xn_.get(),
                                      d_sgate_.get(), 1, H, n, s);
      cuda::sigmoid_gate_scalar_batched_f16(sh_out, d_sgate_.get(),
                                            n, H, s);
    }
    if (n > 1) {
      cudaMemcpy2DAsync(
          expert_y_.get() + static_cast<size_t>(K) * H,
          static_cast<size_t>(K + 1) * H * sizeof(uint16_t),
          shared_yb_.get(), H * sizeof(uint16_t),
          H * sizeof(uint16_t), n, cudaMemcpyDeviceToDevice, s);
    }
  }

  // 11) combine: x_ = residual + sum_k w * expert_y (absent experts add 0).
  cuda::moe_combine_batched_f16(x_.get(), expert_y_.get(), route_w_.get(),
                                K, H, n, s);
  // v0.12 debug: combine health on the first MoE layer (SAGER_DUMP_LOGITS=2)
  if (dbg_layers_ && layer == 0 && !capturing) {
    dbg_dump_f32("router_W0", lw.router, 0, 16, s);
    dbg_dump_f16("expert_y_slot0", expert_y_.get(), 0, 16, s);
    dbg_dump_f16("expert_y_shared", expert_y_.get(),
                 static_cast<size_t>(K) * H, 16, s);
  }
  MOE_DBG("combine");
}

// ---------------------------------------------------------------------------
// v0.4 chunk-level expert-touch feed for the device path: consume the async
// route-id mirrors (one D2H per layer, enqueued by moe_dispatch_device) and
// update the residency decay scores. Called at the chunk boundary where the
// compute stream syncs anyway; replaces the per-route cache_.touch() of the
// host path with one bulk pass per chunk (no per-layer syncs).
// ---------------------------------------------------------------------------
void Session::drain_expert_touch(int n_rows) {
  if (!pf_device_ || !pump_ || pf_rt_ <= 0 || n_rows <= 0) return;
  const EngineDims& d = mc_.dims;
  const int K = d.n_expert_used;
  stream_.sync();   // the chunk's D2H mirrors have landed
  if (prof.on && fw_ev_pe_ > 0) prof_accum_fw();   // chunk fw events complete
  for (int L = 0; L < d.n_moe_layer; ++L) {
    const int* ids = h_touch_idx_.get() +
        static_cast<size_t>(L) * pf_rt_ * (K + 1);
    for (int r = 0; r < n_rows; ++r) {
      for (int kk = 0; kk < K; ++kk) {
        const int e = ids[static_cast<size_t>(r) * (K + 1) + kk];
        if (e >= 0 && e < d.n_expert) cache_.touch(L, e);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// v0.4 CPU-pool patch for the device path's absent-expert mailbox. Healthy
// path: cudaEventQuery succeeds with a zero count - one host-side query, no
// sync, no work. Anomaly path (loader failure / dropped pair): pull the
// mailbox + plan, rebuild the (row, slot) list on the host, run the AVX2
// pool and ADD the missing contributions (the combine already ran with the
// absent experts at zero, and the next layer's kernels are enqueued AFTER
// the patch because the resolution happens at the next layer boundary).
// ---------------------------------------------------------------------------
void Session::resolve_cpu_mailbox(int layer, int n) {
  if (layer < 0 || mail_pending_ != layer || !pump_ ||
      !ev_scan_ || (!pump_->stream_active() && !dec_mail_ok_)) {
    return;
  }
  // Non-blocking first: the scan kernel of layer `layer` is near the head of
  // the queue; if it has not run yet, retry at the next boundary and force
  // at the end of the window.
  if (cudaEventQuery(ev_scan_) != cudaSuccess) return;
  mail_pending_ = -1;
  const int cnt = *h_mail_n_.get();
  if (cnt <= 0) return;

  const EngineDims& d = mc_.dims;
  const int H = d.hidden;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  SLOG_W("moe", "layer %d: %d absent expert(s) mailed - CPU-pool patch",
         layer, cnt);

  // Device state of this layer is still intact (nothing overwrote it).
  cudaMemcpy(h_mail_.get(), d_mail_.get(),
             static_cast<size_t>(cnt) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_off_.get(), d_off_.get(), (E + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gslot_.get(), d_gslot_.get(),
             static_cast<size_t>(n) * K * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grows_.get(), d_grows_.get(),
             static_cast<size_t>(h_off_.get()[E]) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_x16_.get(), xn_.get(),
             static_cast<size_t>(n) * H * sizeof(uint16_t),
             cudaMemcpyDeviceToHost);

  // Dedup mailed experts (each absent expert mails once per tile-block).
  std::vector<uint8_t> mailed(static_cast<size_t>(E), 0);
  std::vector<int> uniq;
  for (int i = 0; i < cnt && i < E; ++i) {
    const int e = h_mail_.get()[i];
    if (e >= 0 && e < E && !mailed[static_cast<size_t>(e)]) {
      mailed[static_cast<size_t>(e)] = 1;
      uniq.push_back(e);
    }
  }
  // v0.5 decode: queue arena promotion for the step end (GPU idle -> safe
  // slot steals); the next step's table rebuild picks the slots up.
  if (dec_mail_ok_) {
    for (int e : uniq) dec_promote_.emplace_back(layer, e);
  }
  cpu_pool_patch_rows(layer, n, uniq.data(), static_cast<int>(uniq.size()));
}

// ---------------------------------------------------------------------------
// v0.10: shared CPU-pool patch core (mailbox path + MoE-TC absent path).
// The pinned mirrors h_off_ / h_grows_ / h_gslot_ / h_x16_ must already hold
// the layer's plan and fp16 input; runs the AVX2 pool for the given absent
// experts and adds their contributions (the combine must already have run
// with the absent experts at zero).
// ---------------------------------------------------------------------------
void Session::cpu_pool_patch_rows(int layer, int n, const int* uniq,
                                  int n_uniq) {
  (void)n;
  const EngineDims& d = mc_.dims;
  const int H = d.hidden;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  const int FFX = d.n_ff_exp;
  cudaStream_t s = stream_.get();
  // Rebuild (row, kk) for the absent experts and run the AVX2 pool.
  const size_t gu = hot_->per_expert_bytes / 3;
  int n_patch = 0;
  struct Patch { int r, kk, e; };
  std::vector<Patch> patches;
  for (int i = 0; i < n_uniq; ++i) {
    const int e = uniq[i];
    if (e < 0 || e >= E) continue;
    for (int pos = h_off_.get()[e]; pos < h_off_.get()[e + 1]; ++pos) {
      const int r = h_grows_.get()[pos];
      for (int kk = 0; kk < K; ++kk) {
        if (h_gslot_.get()[static_cast<size_t>(r) * K + kk] == pos) {
          patches.push_back({r, kk, e});
          break;
        }
      }
    }
  }
  if (static_cast<int>(patches.size()) > pf_cpu_cap_) {
    SLOG_W("moe", "layer %d: %zu patch rows exceed cap %d - dropping tail",
           layer, patches.size(), pf_cpu_cap_);
    patches.resize(static_cast<size_t>(pf_cpu_cap_));
  }
  if (patches.empty()) return;

  std::vector<MissJob> mjobs;
  mjobs.reserve(patches.size());
  for (const Patch& pc : patches) {
    float* mx = miss_x_.get() + static_cast<size_t>(n_patch) * H;
    const uint16_t* x16 = h_x16_.get() + static_cast<size_t>(pc.r) * H;
    for (int t = 0; t < H; ++t) {
      mx[t] = q4k::half_to_float(x16[t]);
    }
    const uint8_t* exp = w_.host_expert_base() +
        (static_cast<size_t>(layer) * E + pc.e) * hot_->per_expert_bytes;
    mjobs.push_back(MissJob{
        layer, pc.e, reinterpret_cast<const q4k::BlockQ4K*>(exp),
        reinterpret_cast<const q4k::BlockQ4K*>(exp + gu),
        reinterpret_cast<const q4k::BlockQ4K*>(exp + 2 * gu), mx,
        miss_y_.get() + static_cast<size_t>(n_patch) * H, H, FFX});
    // expert_y_ element offset for the scatter, and the packed slot for the add
    h_cflat_.get()[n_patch] = static_cast<int>(
        (static_cast<size_t>(pc.r) * (K + 1) + pc.kk) * H);
    h_rk_.get()[n_patch] =
        static_cast<int>(static_cast<size_t>(pc.r) * (K + 1) + pc.kk);
    ++n_patch;
  }
  pool_.submit(std::move(mjobs));
  pool_.wait_done(nullptr);
  cudaMemcpyAsync(d_miss_y_.get(), miss_y_.get(),
                  static_cast<size_t>(n_patch) * H * sizeof(float),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_cflat_.get(), h_cflat_.get(),
                  static_cast<size_t>(n_patch) * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_rk_.get(), h_rk_.get(),
                  static_cast<size_t>(n_patch) * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  cuda::scatter_rows_generic_f16(expert_y_.get(), d_miss_y_.get(),
                                 d_cflat_.get(), n_patch, H, s);
  cuda::moe_combine_add_slots_f16(x_.get(), expert_y_.get(), route_w_.get(),
                                  d_rk_.get(), K, H, n_patch, s);
}

// ---------------------------------------------------------------------------
// v0.6 step-end CPU-pool patch for full-layer graph replay. Same math as
// resolve_cpu_mailbox(), but reads the PRESERVED per-layer dispatch state
// (d_*_all_ slices + the layer's input snapshot) because the shared
// single-step buffers were overwritten by later layers while the graphs ran.
// Runs after the step sync (GPU idle - plain blocking copies are fine) and
// BEFORE the head, so the step's logits are exact.
// ---------------------------------------------------------------------------
void Session::patch_graph_mailbox(int layer, int cnt) {
  const EngineDims& d = mc_.dims;
  const int H = d.hidden;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  const int FFX = d.n_ff_exp;
  cudaStream_t s = stream_.get();
  const size_t le = static_cast<size_t>(layer) * E;

  // Device state of this layer was preserved by the captured graph.
  cudaMemcpy(h_mail_.get(), d_mail_all_.get() + le,
             static_cast<size_t>(cnt) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_off_.get(), d_off_all_.get() +
                 static_cast<size_t>(layer) * (E + 1),
             (E + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gslot_.get(), d_gslot_all_.get() +
                 static_cast<size_t>(layer) * K,
             static_cast<size_t>(K) * sizeof(int), cudaMemcpyDeviceToHost);
  const int R = h_off_.get()[E];
  if (R > 0 && R <= K) {
    cudaMemcpy(h_grows_.get(), d_grows_all_.get() +
                   static_cast<size_t>(layer) * K,
               static_cast<size_t>(R) * sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(h_x16_.get(), d_x16_all_.get() + static_cast<size_t>(layer) * H,
             static_cast<size_t>(H) * sizeof(uint16_t),
             cudaMemcpyDeviceToHost);

  // Dedup mailed experts (each absent expert mails once per tile-block).
  std::vector<uint8_t> mailed(static_cast<size_t>(E), 0);
  std::vector<int> uniq;
  for (int i = 0; i < cnt && i < E; ++i) {
    const int e = h_mail_.get()[i];
    if (e >= 0 && e < E && !mailed[static_cast<size_t>(e)]) {
      mailed[static_cast<size_t>(e)] = 1;
      uniq.push_back(e);
    }
  }
  // Queue arena promotion for the step end (the GPU is idle here; the next
  // step's table rebuild picks the slots up).
  for (int e : uniq) dec_promote_.emplace_back(layer, e);

  // Rebuild (row, kk) for the absent experts and run the AVX2 pool. At the
  // decode step n == 1, so the single routed row is row 0.
  const size_t gu = hot_->per_expert_bytes / 3;
  int n_patch = 0;
  struct Patch { int r, kk, e; };
  std::vector<Patch> patches;
  for (int e : uniq) {
    for (int pos = h_off_.get()[e]; pos < h_off_.get()[e + 1]; ++pos) {
      if (pos < 0 || pos >= R) continue;
      const int r = h_grows_.get()[pos];
      for (int kk = 0; kk < K; ++kk) {
        if (h_gslot_.get()[kk] == pos) {
          patches.push_back({r, kk, e});
          break;
        }
      }
    }
  }
  if (static_cast<int>(patches.size()) > pf_cpu_cap_) {
    SLOG_W("moe", "layer %d: %zu patch rows exceed cap %d - dropping tail",
           layer, patches.size(), pf_cpu_cap_);
    patches.resize(static_cast<size_t>(pf_cpu_cap_));
  }
  if (patches.empty()) return;

  std::vector<MissJob> mjobs;
  mjobs.reserve(patches.size());
  for (const Patch& pc : patches) {
    float* mx = miss_x_.get() + static_cast<size_t>(n_patch) * H;
    const uint16_t* x16 = h_x16_.get() + static_cast<size_t>(pc.r) * H;
    for (int t = 0; t < H; ++t) {
      mx[t] = q4k::half_to_float(x16[t]);
    }
    const uint8_t* exp = w_.host_expert_base() +
        (static_cast<size_t>(layer) * E + pc.e) * hot_->per_expert_bytes;
    mjobs.push_back(MissJob{
        layer, pc.e, reinterpret_cast<const q4k::BlockQ4K*>(exp),
        reinterpret_cast<const q4k::BlockQ4K*>(exp + gu),
        reinterpret_cast<const q4k::BlockQ4K*>(exp + 2 * gu), mx,
        miss_y_.get() + static_cast<size_t>(n_patch) * H, H, FFX});
    h_cflat_.get()[n_patch] = static_cast<int>(
        (static_cast<size_t>(pc.r) * (K + 1) + pc.kk) * H);
    h_rk_.get()[n_patch] =
        static_cast<int>(static_cast<size_t>(pc.r) * (K + 1) + pc.kk);
    ++n_patch;
  }
  pool_.submit(std::move(mjobs));
  pool_.wait_done(nullptr);
  cudaMemcpyAsync(d_miss_y_.get(), miss_y_.get(),
                  static_cast<size_t>(n_patch) * H * sizeof(float),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_cflat_.get(), h_cflat_.get(),
                  static_cast<size_t>(n_patch) * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_rk_.get(), h_rk_.get(),
                  static_cast<size_t>(n_patch) * sizeof(int),
                  cudaMemcpyHostToDevice, s);
  cuda::scatter_rows_generic_f16(expert_y_.get(), d_miss_y_.get(),
                                 d_cflat_.get(), n_patch, H, s);
  cuda::moe_combine_add_slots_f16(x_.get(), expert_y_.get(),
                                  d_route_w_all_.get() +
                                      static_cast<size_t>(layer) * (K + 1),
                                  d_rk_.get(), K, H, n_patch, s);
  stream_.sync();
}

// ---------------------------------------------------------------------------
// Head
// ---------------------------------------------------------------------------
void Session::head_launch() {
  const EngineDims& d = mc_.dims;
  cudaStream_t s = stream_.get();
  cuda::rmsnorm_f16(xn_.get(), x_.get(), w_.output_norm, d.hidden,
                    mc_.norm_eps, s);
  if (w_.head_type_ == 1) {
    // fp16 head (Q6_K dequantized at load)
    cuda::gemv_f16_f16out(w_.output, xn_.get(), d_logits16_.get(), d.n_vocab,
                          d.hidden, s);
    cuda::convert_f16_to_f32(d_logits16_.get(), logits_.get(), d.n_vocab, s);
  } else if (w_.head_type_ == 14 && head_q6k_) {
    // v0.6: native Q6_K head - 401 MB of weight traffic per step instead of
    // the 1017 MB fp16 copy, no separate convert launch.
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
    if (head_smx_) {
      cuda::gemv_q6_k_q8_smx(w_.output, d_xq_.get(), logits_.get(), d.n_vocab,
                             d.hidden, s);
    } else {
      cuda::gemv_q6_k_q8(w_.output, d_xq_.get(), logits_.get(), d.n_vocab,
                         d.hidden, s);
    }
  } else {
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
    cuda::gemv_q4_k_q8(Q4W(w_.output), d_xq_.get(), logits_.get(), d.n_vocab,
                       d.hidden, s);
  }
}

void Session::head_launch_row(int row) {
  const EngineDims& d = mc_.dims;
  cudaStream_t s = stream_.get();
  // v0.12 debug: pre-norm residual row health (SAGER_DUMP_LOGITS=1)
  if (const char* dl = std::getenv("SAGER_DUMP_LOGITS")) {
    if (*dl && *dl != '0') {
      static PinnedBuffer<float> dbg_row;
      if (dbg_row.count() < static_cast<size_t>(d.hidden)) {
        dbg_row.alloc(static_cast<size_t>(d.hidden));
      }
      const uint16_t* xrow = x_.get() + static_cast<size_t>(row) * d.hidden;
      cudaMemcpyAsync(dbg_row.get(), xrow, d.hidden * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost, s);
      stream_.sync();
      float sabs = 0.f, mx = 0.f;
      for (int i = 0; i < d.hidden; ++i) {
        const float v = __half2float(
            *reinterpret_cast<const __half*>(&dbg_row.get()[i]));
        sabs += fabsf(v);
        mx = fmaxf(mx, fabsf(v));
      }
      fprintf(stderr, "[sager/dbg] x_row sum|x|=%.4f max|x|=%.4f (H=%d)\n",
              sabs, mx, d.hidden);
    }
  }
  cuda::rmsnorm_f16(xn_.get(), x_.get() + static_cast<size_t>(row) * d.hidden,
                    w_.output_norm, d.hidden, mc_.norm_eps, s);
  // MTP stash: prefill's tail row is the hidden the first draft conditions on.
  cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                  cudaMemcpyDeviceToDevice, s);
  if (w_.head_type_ == 1) {
    cuda::gemv_f16_f16out(w_.output, xn_.get(), d_logits16_.get(), d.n_vocab,
                          d.hidden, s);
    cuda::convert_f16_to_f32(d_logits16_.get(), logits_.get(), d.n_vocab, s);
  } else if (w_.head_type_ == 14 && head_q6k_) {
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
    if (head_smx_) {
      cuda::gemv_q6_k_q8_smx(w_.output, d_xq_.get(), logits_.get(), d.n_vocab,
                             d.hidden, s);
    } else {
      cuda::gemv_q6_k_q8(w_.output, d_xq_.get(), logits_.get(), d.n_vocab,
                         d.hidden, s);
    }
  } else {
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
    cuda::gemv_q4_k_q8(Q4W(w_.output), d_xq_.get(), logits_.get(), d.n_vocab,
                       d.hidden, s);
  }
  cudaMemcpyAsync(logits_host_.get(), logits_.get(),
                  d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
  stream_.sync();
  // v0.12 debug: dump the prefill tail-row logits distribution
  // (SAGER_DUMP_LOGITS=1). One row per generate - cheap, stderr only.
  if (const char* dl = std::getenv("SAGER_DUMP_LOGITS")) {
    if (*dl && *dl != '0') {
      const int nv = d.n_vocab;
      const float* lg = logits_host_.get();
      float mx = -1e30f, mn = 1e30f, sum = 0.f;
      for (int i = 0; i < nv; ++i) {
        const float v = lg[i];
        if (v > mx) mx = v;
        if (v < mn) mn = v;
        sum += v;
      }
      int top[8];
      float tv[8];
      for (int k = 0; k < 8; ++k) { tv[k] = -1e30f; top[k] = -1; }
      for (int i = 0; i < nv; ++i) {
        const float v = lg[i];
        for (int k = 0; k < 8; ++k) {
          if (v > tv[k]) {
            for (int q = 7; q > k; --q) { tv[q] = tv[q - 1]; top[q] = top[q - 1]; }
            tv[k] = v; top[k] = i;
            break;
          }
        }
      }
      fprintf(stderr, "[sager/dbg] logits n=%d max=%.3f min=%.3f mean=%.4f top:",
              nv, mx, mn, sum / static_cast<float>(nv));
      for (int k = 0; k < 8; ++k) {
        fprintf(stderr, " (%d,%.3f)", top[k], tv[k]);
      }
      fprintf(stderr, "\n");
    }
  }
}

// ---------------------------------------------------------------------------
// v0.5 decode fast path: residency-only pointer table for the device-dispatch
// decode (no stream plan; hot slab + parked arena slots; absent experts mail
// to the CPU pool). Rebuilt lazily and after any residency change.
// ---------------------------------------------------------------------------
void Session::build_decode_table() {
  const EngineDims& d = mc_.dims;
  const int E = d.n_expert;
  const int Lm = d.n_moe_layer;
  for (size_t i = 0; i < h_ptr_table_.count(); ++i) {
    h_ptr_table_.get()[i] = nullptr;
  }
  for (int L = 0; L < Lm; ++L) {
    for (int e = 0; e < E; ++e) {
      q4k::BlockQ4K* tri[3] = {nullptr, nullptr, nullptr};
      if (cache_.resident(L, e)) {
        const size_t base = (static_cast<size_t>(L) * 3 + 0) * E + e;
        tri[0] = reinterpret_cast<q4k::BlockQ4K*>(hot_->tables_host[base]);
        tri[1] = reinterpret_cast<q4k::BlockQ4K*>(
            hot_->tables_host[base + static_cast<size_t>(E)]);
        tri[2] = reinterpret_cast<q4k::BlockQ4K*>(
            hot_->tables_host[base + 2 * static_cast<size_t>(E)]);
      } else if (pump_ && pump_->resident(L, e)) {
        const int pslot = pump_->stream_slot_of(L, e);
        if (pslot >= 0) {
          auto* base = const_cast<q4k::BlockQ4K*>(pump_->dev_base(pslot));
          const size_t gu = hot_->per_expert_bytes / 3 / q4k::kBlockBytes;
          tri[0] = base;
          tri[1] = base + gu;
          tri[2] = base + 2 * gu;
        }
      }
      for (int t = 0; t < 3; ++t) {
        h_ptr_table_.get()[(static_cast<size_t>(t) * Lm + L) * E + e] = tri[t];
      }
    }
  }
  cudaMemcpyAsync(d_ptr_table_.get(), h_ptr_table_.get(),
                  h_ptr_table_.count() * sizeof(q4k::BlockQ4K*),
                  cudaMemcpyHostToDevice, stream_.get());
  dec_tbl_built_ = true;
  // v0.6: per-layer absent-expert counts drive the full-graph replay gate
  // (a layer is only replayed from its captured graph when every expert is
  // resident, so the captured MoE body can never hit the mailbox).
  for (int L = 0; L < Lm; ++L) {
    int cnt = 0;
    for (int e = 0; e < E; ++e) {
      if (!h_ptr_table_.get()[static_cast<size_t>(L) * E + e]) ++cnt;
    }
    absent_cnt_[static_cast<size_t>(L)] = cnt;
  }
}

// ---------------------------------------------------------------------------
// v0.5 device sampler host finish: softmax/temp + sort + min_p + top_p +
// multinomial over the device-selected top-k candidates (same policy and
// rng discipline as sample_host, which stays available as the fallback).
// ---------------------------------------------------------------------------
int Session::finish_device_sample(float temp, int top_k, float top_p,
                                  float min_p, float rep_penalty,
                                  float presence_penalty, uint64_t& rng) {
  (void)rep_penalty;
  (void)presence_penalty;   // penalties were applied on device pre-top-k
  if (!dev_sample_valid_ || dev_sample_k_ <= 0) return -1;
  dev_sample_valid_ = false;
  struct Cand { float p; int id; };
  Cand cand[64];
  int nc = 0;
  for (int i = 0; i < dev_sample_k_ && i < 64; ++i) {
    if (h_topk_i_[i] < 0) break;
    cand[nc].p = h_topk_v_[i];
    cand[nc].id = h_topk_i_[i];
    ++nc;
  }
  if (nc <= 0) return -1;
  if (nc == 1) return cand[0].id;

  const float t = temp > 1e-4f ? temp : 1e-4f;
  float mx = -1e30f;
  for (int i = 0; i < nc; ++i) mx = std::max(mx, cand[i].p);
  float z = 0.f;
  for (int i = 0; i < nc; ++i) {
    cand[i].p = std::exp((cand[i].p - mx) / t);
    z += cand[i].p;
  }
  for (int i = 0; i < nc; ++i) cand[i].p /= z;
  std::sort(cand, cand + nc,
            [](const Cand& a, const Cand& b) { return a.p > b.p; });

  if (min_p > 0.f) {
    const float floor_ = min_p * cand[0].p;
    int keep = 0;
    while (keep < nc && cand[keep].p >= floor_) ++keep;
    nc = std::max(keep, 1);
  }
  if (top_p < 1.0f) {
    float cum = 0.f;
    int keep = nc;
    for (int i = 0; i < nc; ++i) {
      cum += cand[i].p;
      if (cum >= top_p) { keep = i + 1; break; }
    }
    nc = std::max(keep, 1);
  }

  z = 0.f;
  for (int i = 0; i < nc; ++i) z += cand[i].p;
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  float r = static_cast<float>(
      static_cast<double>(rng >> 11) / 9007199254740992.0) * z;
  for (int i = 0; i < nc; ++i) {
    r -= cand[i].p;
    if (r <= 0.f) return cand[i].id;
  }
  return cand[0].id;
}

// ---------------------------------------------------------------------------
// Decode step (graph path)
// ---------------------------------------------------------------------------
const float* Session::decode_step(int last_id) {
  if (!prepared_) return nullptr;
  const EngineDims& d = mc_.dims;
  const double t0 = now_ms();
  cudaStream_t s = stream_.get();
  const bool dev_path = dec_device_ && pf_device_ && pump_ != nullptr;

  if (dev_path && step_graph_ok_) {
    // ---- v0.7 whole-step decode: TWO cudaGraphLaunch calls carry the whole
    // step (embedding + 40 layers, then head + penalties + top-k + mirrors).
    // The pinned stage buffers are read by memcpy nodes at replay time, so
    // the CPU only writes them, launches, and syncs once. Absent experts
    // still zero + mail in-kernel; the step-end CPU patch re-runs the head
    // graph (rare at ~99% row residency) to keep logits exact.
    dev_sample_valid_ = false;
    if (!dec_tbl_built_) build_decode_table();
    const int p7 = pos_;
    int* st7 = h_step_stage_.get();
    st7[0] = last_id;
    st7[1] = p7; st7[2] = p7; st7[3] = p7;
    st7[8] = p7; st7[9] = 1;
    {
      float* sf7 = reinterpret_cast<float*>(st7 + 10);
      sf7[0] = sam_rep_;
      sf7[1] = sam_pres_;
    }
    const int nr7 = static_cast<int>(recent_ids_.size());
    for (int i = 0; i < nr7 && i < kMaxRecent; ++i)
      h_recent_stage_[i] = recent_ids_[i];
    h_recent_stage_[kMaxRecent] = nr7;

    const double t07 = now_ms();
    if (prof.on) cudaEventRecord(prof_ev_[0], s);
    pump_->begin_step(static_cast<uint64_t>(perf.steps) + 1);
    dec_mail_ok_ = true;
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(step_graph_), s);
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(head_graph_), s);
    if (prof.on) cudaEventRecord(prof_ev_[1], s);
    stream_.sync();
    dec_mail_ok_ = false;

    // Rare absent-expert patch: recompute the missing rows from the CPU pool
    // into x_, then replay the head graph so logits stay exact.
    bool patched7 = false;
    const int Lm7 = mc_.dims.n_moe_layer;
    for (int L = 0; L < mc_.dims.n_layer &&
                    L < static_cast<int>(h_mail_n_all_.count()); ++L) {
      const int cnt7 = h_mail_n_all_.get()[L];
      if (cnt7 > 0) {
        patch_graph_mailbox(L, cnt7);
        patched7 = true;
      }
    }
    if (patched7) {
      cudaGraphLaunch(static_cast<cudaGraphExec_t>(head_graph_), s);
      stream_.sync();
    }
    dev_sample_k_ = dev_sample_ ? 32 : 0;
    dev_sample_valid_ = dev_sample_;

    pos_ = p7 + 1;
    push_recent(last_id);

    // Residency maintenance: decay + periodic rebalance (off the steady path).
    cache_.tick(0.9995f);
    if (perf.steps > 0 && perf.steps % 4096 == 0) {
      if (cache_.rebalance() > 0) {
        hot_->rebuild(cache_, w_, mc_.dims.n_moe_layer, mc_.dims.n_expert,
                      mc_.expert_bytes());
        dec_tbl_built_ = false;
      }
    }
    // Arena promotion of mailed experts (GPU idle here - race-free).
    if (!dec_promote_.empty()) {
      for (const auto& le : dec_promote_) {
        pump_->submit(le.first, le.second,
                      static_cast<uint64_t>(perf.steps) + 1);
      }
      pump_->arm_wait(s);
      dec_promote_.clear();
      dec_tbl_built_ = false;
    }
    // v0.7: no touch feed on the whole-step path (CE mirrors removed); the
    // decay via cache_.tick still runs, so residency stays stable.

    if (prof.on) {
      ++prof.dec_steps;
      float ms7 = 0.f;
      cudaEventElapsedTime(&ms7, prof_ev_[0], prof_ev_[1]);
      prof.dec_dense += ms7;
    }
    const double ms7b = now_ms() - t07;
    perf.decode_tps = perf.decode_tps > 0
                          ? 0.9 * perf.decode_tps + 0.1 * (1000.0 / ms7b)
                          : 1000.0 / ms7b;
    ++perf.steps;
    return logits_host_.get();
  }

  if (dev_path) {
    // ---- v0.5 device-dispatch decode: the whole step is enqueued with NO
    // per-layer host sync. Layers replay their captured dense/GDN graphs;
    // MoE runs the v0.4 device dispatch (device plan + grouped _tab GEMVs)
    // against the decode residency table; absent experts zero + mail and
    // the CPU pool patches them at the next boundary (stream-ordered).
    dev_sample_valid_ = false;
    if (!dec_tbl_built_) build_decode_table();

    // per-step values via pinned staging (async uploads)
    int* st = h_step_stage_.get();
    st[0] = last_id;
    const int p = pos_;
    st[1] = p;
    st[2] = p;
    st[3] = p;
    st[8] = p;
    st[9] = 1;
    if (prof.on) cudaEventRecord(prof_ev_[0], s);
    cudaMemcpyAsync(d_ids_.get(), st, sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_pos_ids_.get(), st + 1,
                    (mc_.use_mrope ? 3 : 1) * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_params_.get(), st + 8, sizeof(SeqParams),
                    cudaMemcpyHostToDevice, s);

    cuda::gather_embed(x_.get(), w_.tok_embd, d_ids_.get(), 1, d.hidden, s);

    pump_->begin_step(static_cast<uint64_t>(perf.steps) + 1);
    dec_mail_ok_ = true;
    int pe = 1;
    std::fill(used_full_.begin(), used_full_.end(), 0);
    for (int L = 0; L < d.n_layer; ++L) {
      resolve_cpu_mailbox(L - 1, 1);
      // v0.6: replay the FULL layer (dense + MoE) from one graph. Absent
      // experts zero their rows in-kernel and mail to the per-layer slot;
      // the step-end patch recomputes them from the CPU pool before the
      // head runs (same exact-logits semantics as the v0.4 boundary patch).
      const bool use_full =
          graphs_.full_valid() && graphs_.full_layer_graph(L) != nullptr;
      if (use_full) {
        cudaGraphLaunch(
            static_cast<cudaGraphExec_t>(graphs_.full_layer_graph(L)), s);
        if (L < static_cast<int>(used_full_.size())) used_full_[L] = 1;
      } else if (graphs_.valid() && graphs_.layer_graph(L)) {
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphs_.layer_graph(L)),
                        s);
      } else if (mc_.is_linear(L)) {
        linear_layer_launch(L, 1);
      } else {
        dense_layer_launch(L, 1);
      }
      if (prof.on) cudaEventRecord(prof_ev_[pe++], s);
      if (!use_full) moe_dispatch_device(L, 1);
      if (prof.on) cudaEventRecord(prof_ev_[pe++], s);
    }
    if (prof.on) cudaEventRecord(prof_ev_[pe++], s);   // layers done
    resolve_cpu_mailbox(d.n_layer - 1, 1);
    dec_mail_ok_ = false;

    // v0.6: land the per-layer mail mirrors, then patch any absent-expert
    // rows (CPU pool + AVX2) into x_ BEFORE the head - the step's logits stay
    // exact even when a routed expert is not resident (v0.4 semantics).
    stream_.sync();
    if (graphs_.full_valid()) {
      for (int L = 0;
           L < d.n_layer && L < static_cast<int>(used_full_.size()); ++L) {
        if (!used_full_[L] ||
            L >= static_cast<int>(h_mail_n_all_.count())) {
          continue;
        }
        const int cnt = h_mail_n_all_.get()[L];
        if (cnt > 0) patch_graph_mailbox(L, cnt);
      }
    }

    head_launch();
    if (prof.on) cudaEventRecord(prof_ev_[pe++], s);   // head done

    if (dev_sample_) {
      // Device sampler: scratch copy + penalties + top-k; generate.cpp
      // finishes softmax/top-p/multinomial after the sync below. Skips the
      // ~1 MB full-logits D2H of the host-sampler path.
      cudaMemcpyAsync(d_sample_logits_.get(), logits_.get(),
                      d.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, s);
      {
        const int nr = static_cast<int>(recent_ids_.size());
        for (int i = 0; i < kMaxRecent; ++i)
          h_recent_stage_[i] = (i < nr) ? recent_ids_[i] : -1;
        h_recent_stage_[kMaxRecent] = nr;
        float* sf = reinterpret_cast<float*>(h_step_stage_.get() + 10);
        sf[0] = sam_rep_;
        sf[1] = sam_pres_;
        cudaMemcpyAsync(d_recent_.get(), h_recent_stage_.get(),
                        kMaxRecent * sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_recent_n_.get(), h_recent_stage_.get() + kMaxRecent,
                        sizeof(int), cudaMemcpyHostToDevice, s);
        cuda::apply_penalties_f32_devn(
            d_sample_logits_.get(), d_recent_.get(), d_recent_n_.get(),
            reinterpret_cast<const float*>(h_step_stage_.get() + 10), s);
      }
      const int ksel = 32;
      if (topk2_) {
        cuda::topk_select_f32_2p(
            d_sample_logits_.get(), d_topk_seg_v_.get(), d_topk_seg_i_.get(),
            topk_seg_slots_, d_topk_v_.get(), d_topk_i_.get(), d.n_vocab,
            ksel, s);
      } else {
        cuda::topk_select_f32(d_sample_logits_.get(), d.n_vocab,
                              d_topk_v_.get(), d_topk_i_.get(), ksel, s);
      }
      cudaMemcpyAsync(h_topk_v_.get(), d_topk_v_.get(),
                      ksel * sizeof(float), cudaMemcpyDeviceToHost, s);
      cudaMemcpyAsync(h_topk_i_.get(), d_topk_i_.get(), ksel * sizeof(int),
                      cudaMemcpyDeviceToHost, s);
      dev_sample_k_ = ksel;
      dev_sample_valid_ = true;
    } else {
      cudaMemcpyAsync(logits_host_.get(), logits_.get(),
                      d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
    }
    if (prof.on) cudaEventRecord(prof_ev_[pe++], s);   // sampler done

    // Stash the final hidden row for the MTP drafter (fixed buffer, D2D).
    cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, s);
    stream_.sync();

    if (prof.on) {
      ++prof.dec_steps;
      float ms = 0.f;
      cudaEventElapsedTime(&ms, prof_ev_[0], prof_ev_[1]);
      prof.dec_upload += ms;
      for (int i = 1; i + 1 < pe; i += 2) {
        cudaEventElapsedTime(&ms, prof_ev_[i], prof_ev_[i + 1]);
        prof.dec_moe += ms;
        if (i + 2 < pe) {
          cudaEventElapsedTime(&ms, prof_ev_[i + 1], prof_ev_[i + 2]);
          prof.dec_dense += ms;
        }
      }
      // pe-3: layers-done marker, pe-2: head-done, pe-1: sampler-done
      cudaEventElapsedTime(&ms, prof_ev_[pe - 3], prof_ev_[pe - 2]);
      prof.dec_head += ms;
      cudaEventElapsedTime(&ms, prof_ev_[pe - 2], prof_ev_[pe - 1]);
      prof.dec_samp += ms;
    }

    pos_ = p + 1;
    push_recent(last_id);

    // Residency maintenance: decay + periodic rebalance (off the steady path).
    cache_.tick(0.9995f);
    if (perf.steps > 0 && perf.steps % 4096 == 0) {
      if (cache_.rebalance() > 0) {
        hot_->rebuild(cache_, w_, d.n_moe_layer, d.n_expert,
                      mc_.expert_bytes());
        dec_tbl_built_ = false;
      }
    }
    // Arena promotion of this step's mailed experts: the GPU is idle here,
    // so slot steals are race-free; copies fence in-stream via arm_wait and
    // the table rebuilds before the next step.
    if (!dec_promote_.empty()) {
      for (const auto& le : dec_promote_) {
        pump_->submit(le.first, le.second,
                      static_cast<uint64_t>(perf.steps) + 1);
      }
      pump_->arm_wait(s);
      dec_promote_.clear();
      dec_tbl_built_ = false;
    }
    drain_expert_touch(1);

    const double ms = now_ms() - t0;
    perf.decode_tps = perf.decode_tps > 0
                          ? 0.9 * perf.decode_tps + 0.1 * (1000.0 / ms)
                          : 1000.0 / ms;
    ++perf.steps;
    return logits_host_.get();
  }

  // Per-step values through fixed-address device blocks (graph contract).
  cudaMemcpy(d_ids_.get(), &last_id, sizeof(int), cudaMemcpyHostToDevice);
  const int p = pos_;
  if (mc_.use_mrope) {
    const int pos3[3] = {p, p, p};   // text-only: t/h/w streams identical
    cudaMemcpy(d_pos_ids_.get(), pos3, 3 * sizeof(int), cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(d_pos_ids_.get(), &p, sizeof(int), cudaMemcpyHostToDevice);
  }
  const SeqParams sp{p, 1};
  cudaMemcpy(d_params_.get(), &sp, sizeof(sp), cudaMemcpyHostToDevice);

  cuda::gather_embed(x_.get(), w_.tok_embd, d_ids_.get(), 1, d.hidden, s);

  pump_ ? pump_->begin_step(static_cast<uint64_t>(perf.steps) + 1)
        : void(0);
  for (int L = 0; L < d.n_layer; ++L) {
    const bool linear = mc_.is_linear(L);
    if (graphs_.valid() && graphs_.layer_graph(L)) {
      cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphs_.layer_graph(L)), s);
    } else if (linear) {
      linear_layer_launch(L, 1);
    } else {
      dense_layer_launch(L, 1);
    }
    moe_dispatch(L);
  }
  head_launch();

  // Stash the final hidden row for the MTP drafter (fixed buffer, D2D).
  cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                  cudaMemcpyDeviceToDevice, s);
  if (fast_sample_) {
    // v0.7: device top-k on the legacy path. Replaces the ~10 ms host
    // softmax/argmax over the 248k vocab with a ~25 us two-phase top-k.
    // Penalties run on device with the same order as sample_host (rep m
    // times, then pres m times, per id); the full logits still land on the
    // host as the finish_device_sample fallback.
    cudaMemcpyAsync(d_sample_logits_.get(), logits_.get(),
                    d.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, s);
    {
      const int nr = static_cast<int>(recent_ids_.size());
      for (int i = 0; i < kMaxRecent; ++i)
        h_recent_stage_[i] = (i < nr) ? recent_ids_[i] : -1;
      h_recent_stage_[kMaxRecent] = nr;
      float* sf = reinterpret_cast<float*>(h_step_stage_.get() + 10);
      sf[0] = sam_rep_;
      sf[1] = sam_pres_;
      cudaMemcpyAsync(d_recent_.get(), h_recent_stage_.get(),
                      kMaxRecent * sizeof(int), cudaMemcpyHostToDevice, s);
      cudaMemcpyAsync(d_recent_n_.get(), h_recent_stage_.get() + kMaxRecent,
                      sizeof(int), cudaMemcpyHostToDevice, s);
      cuda::apply_penalties_f32_devn(d_sample_logits_.get(), d_recent_.get(),
                                     d_recent_n_.get(),
                                     reinterpret_cast<const float*>(
                                         h_step_stage_.get() + 10), s);
    }
    cuda::topk_select_f32_2p(d_sample_logits_.get(), d_topk_seg_v_.get(),
                             d_topk_seg_i_.get(), topk_seg_slots_,
                             d_topk_v_.get(), d_topk_i_.get(), d.n_vocab, 32,
                             s);
    cudaMemcpyAsync(h_topk_v_.get(), d_topk_v_.get(), 32 * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_topk_i_.get(), d_topk_i_.get(), 32 * sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    dev_sample_k_ = 32;
    dev_sample_valid_ = true;
  } else {
    dev_sample_valid_ = false;
  }
  cudaMemcpyAsync(logits_host_.get(), logits_.get(),
                  d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
  stream_.sync();

  pos_ = p + 1;
  push_recent(last_id);

  // Residency maintenance: decay + periodic rebalance (off the steady path).
  cache_.tick(0.9995f);
  if (perf.steps > 0 && perf.steps % 4096 == 0) {
    if (cache_.rebalance() > 0) {
      hot_->rebuild(cache_, w_, d.n_moe_layer, d.n_expert, mc_.expert_bytes());
    }
  }

  const double ms = now_ms() - t0;
  perf.decode_tps = perf.decode_tps > 0
                        ? 0.9 * perf.decode_tps + 0.1 * (1000.0 / ms)
                        : 1000.0 / ms;
  ++perf.steps;
  return logits_host_.get();
}

// ---------------------------------------------------------------------------
// Eager window forward (verify / prefill inner path)
// ---------------------------------------------------------------------------
Status Session::forward_window(const std::vector<int>& window,
                               float* logits_out,
                               const std::atomic<bool>* abort_flag) {
  if (!prepared_) return Status::kInvalidArgument;
  const EngineDims& d = mc_.dims;
  const int n = static_cast<int>(window.size());
  if (n <= 0) return Status::kInvalidArgument;
  if (n > cp_.chunk_prefill) return Status::kInvalidArgument;
  cudaStream_t s = stream_.get();

  const int p = pos_;
  if (pos_ + n > cp_.n_ctx) return Status::kInvalidArgument;

  cudaMemcpy(d_ids_.get(), window.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  if (mc_.use_mrope) {
    h_pos_tmp_.resize(static_cast<size_t>(3) * n);
    for (int t = 0; t < n; ++t) {
      h_pos_tmp_[t] = p + t;
      h_pos_tmp_[n + t] = p + t;
      h_pos_tmp_[2 * n + t] = p + t;
    }
    cudaMemcpy(d_pos_ids_.get(), h_pos_tmp_.data(),
               3 * n * sizeof(int), cudaMemcpyHostToDevice);
  } else {
    h_pos_tmp_.resize(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) h_pos_tmp_[t] = p + t;
    cudaMemcpy(d_pos_ids_.get(), h_pos_tmp_.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);
  }
  const SeqParams sp{p, n};
  cudaMemcpy(d_params_.get(), &sp, sizeof(sp), cudaMemcpyHostToDevice);

  cuda::gather_embed(x_.get(), w_.tok_embd, d_ids_.get(), n, d.hidden, s);
  // v0.12 debug: post-embed health of the last row (SAGER_DUMP_LOGITS=1)
  if (const char* dl = std::getenv("SAGER_DUMP_LOGITS")) {
    if (*dl && *dl != '0') {
      static PinnedBuffer<uint16_t> dbg_emb;
      if (dbg_emb.count() < static_cast<size_t>(d.hidden)) {
        dbg_emb.alloc(static_cast<size_t>(d.hidden));
      }
      const uint16_t* xrow = x_.get() +
          static_cast<size_t>(n - 1) * d.hidden;
      cudaMemcpyAsync(dbg_emb.get(), xrow, d.hidden * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost, s);
      stream_.sync();
      float sabs = 0.f, mx = 0.f;
      for (int i = 0; i < d.hidden; ++i) {
        const float v = __half2float(
            *reinterpret_cast<const __half*>(&dbg_emb.get()[i]));
        sabs += fabsf(v);
        mx = fmaxf(mx, fabsf(v));
      }
      fprintf(stderr,
              "[sager/dbg] post_embed row n=%d sum|x|=%.4f max|x|=%.4f\n",
              n, sabs, mx);
    }
  }

  pump_ ? pump_->begin_step(static_cast<uint64_t>(pos_) + 1) : void(0);
  // Batched route slabs are sized for pf_chunk_cap_ tokens; a caller that
  // bypasses prefill()'s chunking (verify windows) with more rows than that
  // stays on the per-row walk rather than overflowing the slabs.
  const bool batched = pf_batched_ && pump_ != nullptr &&
                       n <= pf_chunk_cap_;
  // v0.4: the device-dispatch path needs an ACTIVE stream-all session (the
  // chunk pointer table + per-pair publish events); verify windows and small
  // chunks keep the v0.2.3 host path (per-expert pump submits).
  const bool device_dispatch = batched && pf_device_ &&
                               pump_->stream_active() &&
                               pump_->stream_loader_count() > 0;
  if (prof.on) {
    ++prof.fw_calls;
    prof.fw_tokens += n;
    if (device_dispatch)      ++prof.fw_device;
    else if (batched)         ++prof.fw_batched;
    else                      prof.fw_rows += n;
    fw_ev_pe_ = 1;
    cudaEventRecord(fw_ev_[0], s);
  }
  for (int L = 0; L < d.n_layer; ++L) {
    if (abort_flag && *abort_flag) return Status::kAborted;
    resolve_cpu_mailbox(L - 1, n);   // no-op unless the prior layer mailed
    if (mc_.is_linear(L)) linear_layer_launch(L, n);
    else                  dense_layer_launch(L, n);
    // v0.12 debug: per-layer residual health (SAGER_DUMP_LOGITS=2)
    if (dbg_layers_) {
      static PinnedBuffer<uint16_t> dbg_l;
      if (dbg_l.count() < static_cast<size_t>(d.hidden)) {
        dbg_l.alloc(static_cast<size_t>(d.hidden));
      }
      const uint16_t* xrow = x_.get() +
          static_cast<size_t>(n - 1) * d.hidden;
      cudaMemcpyAsync(dbg_l.get(), xrow, d.hidden * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost, s);
      stream_.sync();
      float sabs = 0.f, mx = 0.f;
      int nans = 0;
      for (int i = 0; i < d.hidden; ++i) {
        const float v = __half2float(
            *reinterpret_cast<const __half*>(&dbg_l.get()[i]));
        if (std::isnan(v)) ++nans;
        sabs += fabsf(v);
        mx = fmaxf(mx, fabsf(v));
      }
      fprintf(stderr, "[sager/dbg] L=%d(%s) post sum|x|=%.4f max|x|=%.4f nan=%d\n",
              L, mc_.is_linear(L) ? "gdn" : "attn", sabs, mx, nans);
    }
    if (prof.on) cudaEventRecord(fw_ev_[fw_ev_pe_++], s);
    if (device_dispatch) {
      moe_dispatch_device(L, n);
    } else if (batched) {
      moe_dispatch_batched(L, n);
    } else {
      for (int t = 0; t < n; ++t) moe_dispatch_row(L, t);
    }
    // v0.12 debug: after-MoE residual health (SAGER_DUMP_LOGITS=2)
    if (dbg_layers_) {
      static PinnedBuffer<uint16_t> dbg_m;
      if (dbg_m.count() < static_cast<size_t>(d.hidden)) {
        dbg_m.alloc(static_cast<size_t>(d.hidden));
      }
      const uint16_t* xrow2 = x_.get() +
          static_cast<size_t>(n - 1) * d.hidden;
      cudaMemcpyAsync(dbg_m.get(), xrow2, d.hidden * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost, s);
      stream_.sync();
      float sabs2 = 0.f, mx2 = 0.f;
      int nans2 = 0;
      for (int i = 0; i < d.hidden; ++i) {
        const float v = __half2float(
            *reinterpret_cast<const __half*>(&dbg_m.get()[i]));
        if (std::isnan(v)) ++nans2;
        sabs2 += fabsf(v);
        mx2 = fmaxf(mx2, fabsf(v));
      }
      fprintf(stderr, "[sager/dbg] L=%d(%s) moe  sum|x|=%.4f max|x|=%.4f nan=%d\n",
              L, mc_.is_linear(L) ? "gdn" : "attn", sabs2, mx2, nans2);
    }
    if (prof.on) cudaEventRecord(fw_ev_[fw_ev_pe_++], s);
  }
  resolve_cpu_mailbox(d.n_layer - 1, n);   // force the last layer's mailbox
  pos_ = p + n;
  // NOTE: recent_ids_ is NOT touched here - rejected speculative tokens must
  // not pollute the sampler's repetition window. decode_step and
  // verify_window push exactly the tokens that were committed.

  if (logits_out) {
    // Batched head: all rows at once, then one D2H.
    if (prof.on) cudaEventRecord(fw_ev_[fw_ev_pe_++], s);   // layers done
    cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), w_.output_norm, d.hidden,
                              mc_.norm_eps, n, s);
    // MTP stash: the drafter must condition on the pass actually executed
    // last (verify windows included), not on a stale decode_step hidden.
    cudaMemcpyAsync(xf_.get(),
                    xn_.get() + static_cast<size_t>(n - 1) * d.hidden,
                    d.hidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice, s);
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(),
                        static_cast<size_t>(n) * d.hidden, s);
    DeviceBuffer<float> lwin(static_cast<size_t>(n) * d.n_vocab);
    if (!lwin) return Status::kOomVram;
    cuda::gemv_q4_k_q8_batched(Q4W(w_.output), d_xq_.get(), lwin.get(), d.n_vocab,
                               d.hidden, n, s);
    cudaMemcpy(logits_out, lwin.get(),
               static_cast<size_t>(n) * d.n_vocab * sizeof(float),
               cudaMemcpyDeviceToHost);
    // Keep last_logits() coherent with the horizon row.
    cudaMemcpyAsync(logits_host_.get(),
                    lwin.get() + static_cast<size_t>(n - 1) * d.n_vocab,
                    d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
    stream_.sync();
    if (prof.on) {
      cudaEventRecord(fw_ev_[fw_ev_pe_++], s);   // head done (synced above)
      prof_accum_fw();
    }
  } else if (prof.on) {
    cudaEventRecord(fw_ev_[fw_ev_pe_++], s);   // head runs at prefill tail
  }
  return Status::kOk;
}

// v0.6: fold the pending forward_window event ring into prof. Called after a
// stream sync that is guaranteed to be ordered after the recorded events
// (verify head sync, or drain_expert_touch's chunk-boundary sync).
void Session::prof_accum_fw() {
  if (!prof.on || fw_ev_pe_ <= 1) return;
  const int pe = fw_ev_pe_;
  float ms = 0.f;
  cudaEventElapsedTime(&ms, fw_ev_[0], fw_ev_[1]);
  prof.fw_upload += ms;
  for (int i = 1; i + 1 < pe; i += 2) {
    cudaEventElapsedTime(&ms, fw_ev_[i], fw_ev_[i + 1]);
    prof.fw_moe += ms;
    if (i + 2 < pe) {
      cudaEventElapsedTime(&ms, fw_ev_[i + 1], fw_ev_[i + 2]);
      prof.fw_dense += ms;
    }
  }
  cudaEventElapsedTime(&ms, fw_ev_[pe - 2], fw_ev_[pe - 1]);
  prof.fw_head += ms;
  fw_ev_pe_ = 0;
}

void Session::reset() {
  kv_.reset();
  pos_ = 0;
  recent_ids_.clear();
#if defined(SAGER_HAVE_CUDA)
  // Recurrence state must follow the conversation: a retained-expert reset
  // with stale GDN states would leak the previous context into the new one.
  zero_gdn_states();
  mail_pending_ = -1;
#endif
  if (pump_) pump_->clear();   // residency tied to the previous conversation
}

void Session::push_recent(int id) {
  recent_ids_.push_back(id);
  if (static_cast<int>(recent_ids_.size()) > kMaxRecent) {
    recent_ids_.erase(recent_ids_.begin());
  }
}

#else  // ---- CPU-only build: v0.1 requires CUDA -----------------------------

Session::Session(Weights& w, ExpertCache& c, CpuPool& p, const ModelConfig&,
                 const CtxParams&)
    : w_(w), cache_(c), pool_(p) {}
Session::~Session() = default;
Status Session::alloc_scratch() { return Status::kUnsupported; }
Status Session::prepare() { return Status::kUnsupported; }
void Session::dense_layer_launch(int, int) {}
void Session::linear_layer_launch(int, int) {}
void Session::zero_gdn_states() {}
void Session::moe_dispatch(int) {}
void Session::moe_dispatch_row(int, int) {}
void Session::head_launch() {}
void Session::head_launch_row(int) {}
const float* Session::decode_step(int) { return nullptr; }
Status Session::forward_window(const std::vector<int>&, float*,
                               const std::atomic<bool>*) {
  return Status::kUnsupported;
}
void Session::reset() {}
void Session::push_recent(int) {}
void Session::moe_dispatch_batched(int, int) {}
void Session::linear_layer_chunk(int, int) {}
void Session::moe_dispatch_device(int, int) {}
void Session::resolve_cpu_mailbox(int, int) {}
void Session::drain_expert_touch(int) {}
const float* Session::last_logits() const { return nullptr; }

#endif  // SAGER_HAVE_CUDA

}  // namespace sager
