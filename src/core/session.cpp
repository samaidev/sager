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
  if (!ok(d_off_.alloc(E + 1), "d_off_") ||
      !ok(d_toff_.alloc(E + 1), "d_toff_") ||
      !ok(d_meta_.alloc(4), "d_meta_") ||
      !ok(d_mail_.alloc(E), "d_mail_") ||
      !ok(d_mail_n_.alloc(1), "d_mail_n_") ||
      !ok(d_rk_.alloc(static_cast<size_t>(RT) * K), "d_rk_")) {
    return Status::kOomVram;
  }
  if (!h_ptr_table_.alloc(tbl_entries) ||
      !h_mail_n_.alloc(1) || !h_mail_.alloc(E) ||
      !h_off_.alloc(E + 1) || !h_rk_.alloc(static_cast<size_t>(RT) * K) ||
      !h_touch_idx_.alloc(static_cast<size_t>(d.n_moe_layer) * RT *
                          (K + 1))) {
    return Status::kOomRam;
  }
  // v0.5 decode fast path staging (device sampler + step params)
  if (!ok(d_sample_logits_.alloc(V), "d_sample_logits_") ||
      !ok(d_topk_v_.alloc(64), "d_topk_v_") ||
      !ok(d_topk_i_.alloc(64), "d_topk_i_")) {
    return Status::kOomVram;
  }
  if (!h_topk_v_.alloc(64) || !h_topk_i_.alloc(64) ||
      !h_step_stage_.alloc(32) || !h_recent_stage_.alloc(kMaxRecent)) {
    return Status::kOomRam;
  }
  if (cudaEventCreateWithFlags(&ev_scan_, cudaEventDisableTiming) !=
      cudaSuccess) {
    ev_scan_ = nullptr;
    cudaGetLastError();
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
    }
    const EngineDims& dd0 = mc_.dims;
    if (pf_batched_ &&
        (dd0.hidden % q4k::kQK_K != 0 || dd0.n_ff_exp % q4k::kQK_K != 0)) {
      SLOG_W("session", "batched prefill needs 256-aligned hidden/ff "
             "(%d/%d) - per-row walk stays", dd0.hidden, dd0.n_ff_exp);
      pf_batched_ = false;
    }
    SLOG_I("session", "prefill v0.2.3: batched=%d mtok=%d stream_min=%d chunk_cap=%d "
           "cpu_rows=%d device=%d", (int)pf_batched_, (int)pf_mtok_, pf_stream_min_, pf_chunk_cap_,
           pf_cpu_cap_, (int)pf_device_);
  }

  if (w_.head_type_ != 12 && w_.head_type_ != 1) {
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

  if (n_tokens == 1) {
    cuda::rmsnorm_f16(xn_.get(), x_.get(), lw.attn_norm, H, mc_.norm_eps, s);
  } else {
    cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.attn_norm, H,
                              mc_.norm_eps, n_tokens, s);
  }
  cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n_tokens * H, s);

  if (n_tokens == 1) {
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H, s);
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H, s);
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H, s);
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
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q, s);
  } else if (pf_mtok_ && pf_batched_) {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), n_tokens * Q, s);
    cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q,
                                   n_tokens, s);
  } else {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), n_tokens * Q, s);
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q,
                                      n_tokens, s);
  }
  cuda::add_f16(x_.get(), o_proj_.get(), n_tokens * H, s);
  if (n_tokens == 1) {
    cuda::rmsnorm_f16(xn_.get(), x_.get(), lw.ffn_norm, H, mc_.norm_eps, s);
  } else {
    cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.ffn_norm, H,
                              mc_.norm_eps, n_tokens, s);
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
    cuda::rmsnorm_f16(xn_row, x_row, lw.attn_norm, H, mc_.norm_eps, s);
    cuda::quantize_q8_k(xn_row, d_xq_.get(), H, s);
    // fused projections: qkvz -> [q|k|v|z], ba -> [b|a]
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.lin_in_qkvz), d_xq_.get(),
                              lin_proj_.get(), g.proj_rows(), H, s);
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.lin_in_ba), d_xq_.get(),
                              lin_ba_.get(), g.ba_rows(), H, s);
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
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.lin_out), d_hq_.get(), op_row, H,
                              g.y_rows(), s);
    cuda::add_f16(x_row, op_row, H, s);
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

  cuda::rmsnorm_batched_f16(xn_.get(), x_.get(), lw.attn_norm, H,
                            mc_.norm_eps, n, s);
  cuda::quantize_q8_k(xn_.get(), d_xq_.get(), n * H, s);
  // fused projections for the whole window
  cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_in_qkvz), d_xq_.get(),
                                 lin_proj_b_.get(), g.proj_rows(), H, n, s);
  cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_in_ba), d_xq_.get(),
                                 lin_ba_b_.get(), g.ba_rows(), H, n, s);
  // causal conv over the chunk, then commit the state tail
  cuda::gdn_conv_chunk_f16(lin_conv_b_.get(), lin_proj_b_.get(), conv_st,
                           lw.lin_conv_w, lw.lin_conv_b, g.conv_dim(),
                           g.conv_kernel, n, s);
  cuda::gdn_conv_state_commit_f16(conv_st, lin_proj_b_.get(), g.conv_dim(),
                                  g.conv_kernel, n, s);
  // data-dependent gates + per-head L2 norm, batched
  cuda::gdn_gate_chunk_f32(lin_ba_b_.get(), lw.lin_A_log, lw.lin_dt_bias,
                           lin_ab_b_.get(), g.n_v_heads, n, s);
  cuda::gdn_l2_norm_chunk_f16(lin_qn_b_.get(), lin_kn_b_.get(),
                              lin_conv_b_.get(), g.conv_dim(), g.n_k_heads,
                              g.key_dim, mc_.norm_eps, n, s);
  // fused recurrence scan (state stays in shared memory across the chunk)
  cuda::gdn_scan_chunk_f16(delta_st, lin_y_b_.get(), lin_qn_b_.get(),
                           lin_kn_b_.get(), lin_conv_b_.get(), g.conv_dim(),
                           lin_ab_b_.get(), g.n_k_heads, g.key_dim, g.ratio(),
                           g.value_dim, n, s);
  // output gating + out projection + residual
  cuda::gdn_out_norm_chunk_f16(lin_on_b_.get(), lin_y_b_.get(),
                               lin_proj_b_.get(), g.proj_rows(), g.conv_dim(),
                               lw.lin_o_norm, g.n_v_heads, g.value_dim,
                               mc_.norm_eps, n, s);
  cuda::quantize_q8_k(lin_on_b_.get(), d_hq_.get(), n * g.y_rows(), s);
  cuda::gemv_q4_k_q8_f16out_mtok(Q4W(lw.lin_out), d_hq_.get(), o_proj_.get(),
                                 H, g.y_rows(), n, s);
  cuda::add_f16(x_.get(), o_proj_.get(), n * H, s);
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
  // Warmup + capture DON'T execute the recorded kernels, but the warmup DID
  // mutate GDN state (one token through a linear layer) - reset it so real
  // inference starts from clean recurrence state.
  zero_gdn_states();
  SLOG_I("session", "captured %d per-layer decode graphs", d.n_layer);
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

  // 1) quantize + route (same kernels as the host path - bit-identical).
  cuda::quantize_q8_k(xn_.get(), d_xq_moe_.get(),
                      static_cast<size_t>(n) * H, s);
  cuda::gemv_f32_out_f16x_mtok(lw.router, xn_.get(), route_logits_.get(),
                               E, H, n, s);
  cuda::router_topk_batched(route_logits_.get(), route_idx_.get(),
                            route_w_.get(), E, K, K + 1, n, s);

  // 2) device plan: histogram + prefix sums + grouped scatter + mailbox.
  int tile_gu = 8;
  {
    int t = (49152 - 16 * h_blocks * static_cast<int>(sizeof(q4k::BlockQ4K)) -
             512) / (h_blocks * static_cast<int>(sizeof(q4k::BlockQ8K)));
    tile_gu = t < 4 ? 4 : (t > 16 ? 16 : t);
  }
  cuda::moe_dispatch_scan(route_idx_.get(), route_w_.get(), d_gslot_.get(),
                          d_grows_.get(), d_off_.get(), d_toff_.get(),
                          d_meta_.get(), d_mail_.get(), d_mail_n_.get(),
                          n, K, E, tile_gu, s);
  // mailbox count mirror (lands before ev_scan_ fires)
  cudaMemcpyAsync(h_mail_n_.get(), d_mail_n_.get(), sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  cudaEventRecord(ev_scan_, s);
  mail_pending_ = layer;
  // chunk-level touch feed: async route-id mirror (consumed by
  // drain_expert_touch at the chunk boundary - no per-layer sync). Layer
  // index == MoE-layer index for this arch (every block is MoE).
  cudaMemcpyAsync(
      h_touch_idx_.get() + static_cast<size_t>(layer) * pf_rt_ * (K + 1),
      route_idx_.get(), static_cast<size_t>(n) * (K + 1) * sizeof(int),
      cudaMemcpyDeviceToHost, s);

  // 3) compact routed rows expert-major (row count read on device).
  cuda::gather_rows_q8_n(d_xqg_.get(), d_xq_moe_.get(), d_grows_.get(),
                         d_meta_.get(), static_cast<int>(Rmax), h_blocks, s);

  // 4) device-side wait: this layer's planned pairs must have landed (their
  //    loaders patch the pointer table before firing the publish events).
  //    Decode fast path runs WITHOUT a stream session - the pointer table is
  //    the decode-built residency table, nothing ever patches it mid-step.
  if (pump_->stream_active()) pump_->stream_wait_layer(layer, s);

  // 5) gate+up for EVERY routed expert of the layer in ONE launch.
  //    tiles <= total routed rows (each tile covers >= 1 row) - for the 1-row
  //    decode step this caps the grid at Rmax (=K) tiles instead of the
  //    residency-worst-case K/tile + E + 16, which is 94% empty blocks.
  const int tiles_max = std::min(static_cast<int>(Rmax) / tile_gu + E + 16,
                                 static_cast<int>(Rmax));
  const size_t lstride = static_cast<size_t>(d.n_moe_layer) * E;
  cuda::gemv_grouped2_tab_q4_k_q8_f16out_mtok(
      d_ptr_table_.get() + 0 * lstride,
      d_ptr_table_.get() + 1 * lstride,
      d_off_.get(), d_toff_.get(), d_meta_.get(), d_mail_.get(),
      d_mail_n_.get(), tiles_max, d_xqg_.get(), expert_g2_.get(),
      expert_u2_.get(), FFX, H, E, tile_gu, s);

  // 6) activation + requant over the compacted slab (device row count).
  cuda::silu_mul_f16_n(expert_g2_.get(), expert_g2_.get(), expert_u2_.get(),
                       d_meta_.get(), static_cast<int>(Rmax), FFX, s);
  cuda::quantize_q8_k_n(expert_g2_.get(), d_hqg_.get(), d_meta_.get(), FFX,
                        static_cast<int>(Rmax) * (FFX / q4k::kQK_K), s);

  // 7) down projection (last reader of the ring slots; absent experts'
  //    output rows are zeroed in-kernel so the combine adds 0).
  cuda::gemv_grouped_tab_q4_k_q8_f16out_mtok(
      d_ptr_table_.get() + 2 * lstride, d_off_.get(), d_toff_.get(),
      d_meta_.get(), tiles_max, d_hqg_.get(), y_grouped_.get(), H, FFX, E,
      tile_gu, s);

  // 8) ring slots of this layer's planned pairs free after the down GEMM.
  if (pump_->stream_active()) pump_->stream_consume_layer(layer, s);

  // 9) compact outputs into the combine slab (gslot built on device).
  cuda::scatter_routed_f16(expert_y_.get(), y_grouped_.get(), d_gslot_.get(),
                           n, K, H, s);

  // 10) shared expert, batched (identical to the host path).
  if (lw.shared_gate) {
    cuda::gemv2_q4_k_q8_f16out_mtok(
        reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_gate),
        reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_up),
        d_xq_moe_.get(), sh_g_.get(), sh_u_.get(), FFX, H, n, s);
    cuda::silu_mul_f16(sh_g_.get(), sh_g_.get(), sh_u_.get(), 1, n * FFX, s);
    cuda::quantize_q8_k(sh_g_.get(), d_hqg_.get(),
                        static_cast<size_t>(n) * FFX, s);
    cuda::gemv_q4_k_q8_f16out_mtok(
        reinterpret_cast<const q4k::BlockQ4K*>(lw.shared_out), d_hqg_.get(),
        shared_yb_.get(), H, FFX, n, s);
    if (lw.shared_gate_w) {
      cuda::gemv_f32_out_f16x_batched(lw.shared_gate_w, xn_.get(),
                                      d_sgate_.get(), 1, H, n, s);
      cuda::sigmoid_gate_scalar_batched_f16(shared_yb_.get(), d_sgate_.get(),
                                            n, H, s);
    }
    cudaMemcpy2DAsync(
        expert_y_.get() + static_cast<size_t>(K) * H,
        static_cast<size_t>(K + 1) * H * sizeof(uint16_t),
        shared_yb_.get(), H * sizeof(uint16_t),
        H * sizeof(uint16_t), n, cudaMemcpyDeviceToDevice, s);
  }

  // 11) combine: x_ = residual + sum_k w * expert_y (absent experts add 0).
  cuda::moe_combine_batched_f16(x_.get(), expert_y_.get(), route_w_.get(),
                                K, H, n, s);
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
  const int FFX = d.n_ff_exp;
  cudaStream_t s = stream_.get();
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

  // Rebuild (row, kk) for the absent experts and run the AVX2 pool.
  const size_t gu = hot_->per_expert_bytes / 3;
  int n_patch = 0;
  struct Patch { int r, kk, e; };
  std::vector<Patch> patches;
  for (int e : uniq) {
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
  } else {
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
    cuda::gemv_q4_k_q8(Q4W(w_.output), d_xq_.get(), logits_.get(), d.n_vocab,
                       d.hidden, s);
  }
}

void Session::head_launch_row(int row) {
  const EngineDims& d = mc_.dims;
  cudaStream_t s = stream_.get();
  cuda::rmsnorm_f16(xn_.get(), x_.get() + static_cast<size_t>(row) * d.hidden,
                    w_.output_norm, d.hidden, mc_.norm_eps, s);
  // MTP stash: prefill's tail row is the hidden the first draft conditions on.
  cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                  cudaMemcpyDeviceToDevice, s);
  if (w_.head_type_ == 1) {
    cuda::gemv_f16_f16out(w_.output, xn_.get(), d_logits16_.get(), d.n_vocab,
                          d.hidden, s);
    cuda::convert_f16_to_f32(d_logits16_.get(), logits_.get(), d.n_vocab, s);
  } else {
    cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
    cuda::gemv_q4_k_q8(Q4W(w_.output), d_xq_.get(), logits_.get(), d.n_vocab,
                       d.hidden, s);
  }
  cudaMemcpyAsync(logits_host_.get(), logits_.get(),
                  d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
  stream_.sync();
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
    cudaMemcpyAsync(d_ids_.get(), st, sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_pos_ids_.get(), st + 1,
                    (mc_.use_mrope ? 3 : 1) * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_params_.get(), st + 8, sizeof(SeqParams),
                    cudaMemcpyHostToDevice, s);

    cuda::gather_embed(x_.get(), w_.tok_embd, d_ids_.get(), 1, d.hidden, s);

    pump_->begin_step(static_cast<uint64_t>(perf.steps) + 1);
    dec_mail_ok_ = true;
    for (int L = 0; L < d.n_layer; ++L) {
      resolve_cpu_mailbox(L - 1, 1);
      if (graphs_.valid() && graphs_.layer_graph(L)) {
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphs_.layer_graph(L)),
                        s);
      } else if (mc_.is_linear(L)) {
        linear_layer_launch(L, 1);
      } else {
        dense_layer_launch(L, 1);
      }
      moe_dispatch_device(L, 1);
    }
    resolve_cpu_mailbox(d.n_layer - 1, 1);
    dec_mail_ok_ = false;
    head_launch();

    if (dev_sample_) {
      // Device sampler: scratch copy + penalties + top-k; generate.cpp
      // finishes softmax/top-p/multinomial after the sync below. Skips the
      // ~1 MB full-logits D2H of the host-sampler path.
      cudaMemcpyAsync(d_sample_logits_.get(), logits_.get(),
                      d.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, s);
      const int nr = static_cast<int>(recent_ids_.size());
      if (nr > 0) {
        for (int i = 0; i < nr; ++i) h_recent_stage_[i] = recent_ids_[i];
        cudaMemcpyAsync(d_recent_.get(), h_recent_stage_.get(),
                        nr * sizeof(int), cudaMemcpyHostToDevice, s);
        cuda::apply_penalties_f32(d_sample_logits_.get(), d_recent_.get(), nr,
                                  sam_rep_, sam_pres_, s);
      }
      const int ksel = 32;
      cuda::topk_select_f32(d_sample_logits_.get(), d.n_vocab,
                            d_topk_v_.get(), d_topk_i_.get(), ksel, s);
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

    // Stash the final hidden row for the MTP drafter (fixed buffer, D2D).
    cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, s);
    stream_.sync();

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
  for (int L = 0; L < d.n_layer; ++L) {
    if (abort_flag && *abort_flag) return Status::kAborted;
    resolve_cpu_mailbox(L - 1, n);   // no-op unless the prior layer mailed
    if (mc_.is_linear(L)) linear_layer_launch(L, n);
    else                  dense_layer_launch(L, n);
    if (device_dispatch) {
      moe_dispatch_device(L, n);
    } else if (batched) {
      moe_dispatch_batched(L, n);
    } else {
      for (int t = 0; t < n; ++t) moe_dispatch_row(L, t);
    }
  }
  resolve_cpu_mailbox(d.n_layer - 1, n);   // force the last layer's mailbox
  pos_ = p + n;
  // NOTE: recent_ids_ is NOT touched here - rejected speculative tokens must
  // not pollute the sampler's repetition window. decode_step and
  // verify_window push exactly the tokens that were committed.

  if (logits_out) {
    // Batched head: all rows at once, then one D2H.
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
  }
  return Status::kOk;
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
