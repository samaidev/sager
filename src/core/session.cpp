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
  const int H = d.hidden;
  const int Q = d.n_head * d.head_dim;
  const int KVd = d.n_embd_gqa();
  const int FFX = d.n_ff_exp;
  const int E = d.n_expert;
  const int K = d.n_expert_used;
  const int T = cp_.chunk_prefill > 0 ? cp_.chunk_prefill : 1;
  const int V = d.n_vocab;

  auto ok = [](bool b) { return b ? Status::kOk : Status::kOomVram; };

  // Activation slabs sized for the prefill chunk (fixed addresses: the same
  // buffers serve the 1-row decode path).
  if (!ok(x_.alloc(static_cast<size_t>(T) * H))) return Status::kOomVram;
  if (!ok(xn_.alloc(static_cast<size_t>(T) * H))) return Status::kOomVram;
  x2_.alloc(0);            // repurposed below: compact hit-out staging [K*H]
  if (!ok(x2_.alloc(static_cast<size_t>(K) * H))) return Status::kOomVram;
  xn2_.alloc(0);           // reserved
  if (!ok(q_.alloc(static_cast<size_t>(T) * Q))) return Status::kOomVram;
  if (!ok(k_.alloc(static_cast<size_t>(T) * KVd))) return Status::kOomVram;
  if (!ok(v_.alloc(static_cast<size_t>(T) * KVd))) return Status::kOomVram;
  if (!ok(attn_out_.alloc(static_cast<size_t>(T) * Q))) return Status::kOomVram;
  if (!ok(o_proj_.alloc(static_cast<size_t>(T) * H))) return Status::kOomVram;
  if (!ok(xf_.alloc(H))) return Status::kOomVram;        // MTP hidden stash
  if (!ok(d_cat_.alloc(2 * H))) return Status::kOomVram;
  if (!ok(d_m1_.alloc(H))) return Status::kOomVram;

  // MoE working set
  if (!ok(expert_g_.alloc(static_cast<size_t>(K) * FFX))) return Status::kOomVram;
  if (!ok(expert_u_.alloc(static_cast<size_t>(K) * FFX))) return Status::kOomVram;
  if (!ok(expert_y_.alloc(static_cast<size_t>(K + 1) * H))) return Status::kOomVram;
  if (!ok(shared_y_.alloc(static_cast<size_t>(H) + 2 * FFX))) return Status::kOomVram;
  if (!ok(route_logits_.alloc(E))) return Status::kOomVram;
  if (!ok(route_idx_.alloc(K + 1))) return Status::kOomVram;
  if (!ok(route_w_.alloc(K + 1))) return Status::kOomVram;
  if (!ok(d_miss_y_.alloc(static_cast<size_t>(K) * H))) return Status::kOomVram;
  if (!ok(d_slot_of_k_.alloc(K))) return Status::kOomVram;
  if (!ok(d_hit_rows_.alloc(K))) return Status::kOomVram;
  if (!ok(d_ptrs_hit_.alloc(3 * K))) return Status::kOomVram;

  // Activation quant slabs: worst case = chunked batched GEMV activations.
  const size_t q8_blocks = std::max({static_cast<size_t>(T) * H,
                                     static_cast<size_t>(T) * Q,
                                     static_cast<size_t>(K) * FFX}) /
                           q4k::kQK_K;
  if (!ok(d_xq_.alloc(q8_blocks))) return Status::kOomVram;
  if (!ok(d_hq_.alloc(q8_blocks))) return Status::kOomVram;
  if (!ok(d_mq_.alloc(H / q4k::kQK_K))) return Status::kOomVram;

  // Head + io
  if (!ok(logits_.alloc(V))) return Status::kOomVram;
  mtp_logits_.alloc(0);
  if (!ok(d_params_.alloc(1))) return Status::kOomVram;
  if (!ok(d_ids_.alloc(T))) return Status::kOomVram;
  if (!ok(d_pos_ids_.alloc(3 * T))) return Status::kOomVram;
  if (!ok(d_recent_.alloc(kMaxRecent))) return Status::kOomVram;

  // Host staging (pinned, C1: the SAME regions every call)
  if (!h_route_idx_.alloc(K + 1) || !h_route_w_.alloc(K + 1) ||
      !h_x16_.alloc(H) || !miss_x_.alloc(H) ||
      !miss_y_.alloc(static_cast<size_t>(K) * H) ||
      !logits_host_.alloc(V) || !h_ptrs_.alloc(3 * K) ||
      !h_hit_rows_.alloc(K) || !h_slot_of_k_.alloc(K)) {
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
  if (w_.head_type_ != 12) {
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

  kv_.init({d.n_layer, d.n_head_kv, d.head_dim, cp_.n_ctx, cp_.kv_q8});

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

  SLOG_I("session", "ready: ctx=%d chunk=%d graphs=%s hot_experts=%d/%d",
         cp_.n_ctx, cp_.chunk_prefill, graphs_.valid() ? "on" : "off",
         hot_->resident_count, d.experts_total());
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
  } else {
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wq), d_xq_.get(), q_.get(), Q, H,
                                      n_tokens, s);
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wk), d_xq_.get(), k_.get(), KVd, H,
                                      n_tokens, s);
    cuda::gemv_q4_k_q8_f16out_batched(Q4W(lw.wv), d_xq_.get(), v_.get(), KVd, H,
                                      n_tokens, s);
  }
  cuda::rope_f16(q_.get(), k_.get(), d_pos_ids_.get(), d.n_head, d.n_head_kv,
                 d.head_dim, rope_dims, mc_.rope_theta, mc_.rope_style,
                 n_tokens, s);
  cuda::kv_append(k_.get(), v_.get(), kv_.k_slab(layer), kv_.v_slab(layer),
                  nullptr, nullptr, kv_.q8() ? 1 : 0, d_params_.get(),
                  n_tokens, d.n_head_kv, d.head_dim, s);
  if (n_tokens == 1) {
    cuda::attention_decode_f16(attn_out_.get(), q_.get(), kv_.k_slab(layer),
                               kv_.v_slab(layer), nullptr, nullptr,
                               kv_.q8() ? 1 : 0, d_params_.get(), d.n_head,
                               d.n_head_kv, d.head_dim,
                               1.0f / std::sqrt(static_cast<float>(d.head_dim)),
                               s);
  } else {
    cuda::attention_prefill_f16(attn_out_.get(), q_.get(), kv_.k_slab(layer),
                                kv_.v_slab(layer), nullptr, nullptr,
                                kv_.q8() ? 1 : 0, d_params_.get(), d.n_head,
                                d.n_head_kv, d.head_dim,
                                1.0f / std::sqrt(static_cast<float>(d.head_dim)),
                                n_tokens, s);
  }
  if (n_tokens == 1) {
    cuda::quantize_q8_k(attn_out_.get(), d_hq_.get(), Q, s);
    cuda::gemv_q4_k_q8_f16out(Q4W(lw.wo), d_hq_.get(), o_proj_.get(), H, Q, s);
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
// Graph capture
// ---------------------------------------------------------------------------
Status Session::capture_graphs() {
  const EngineDims& d = mc_.dims;
  graphs_.init(d.n_layer);

  // Warm every kernel once (module load) outside capture.
  dense_layer_launch(0, 1);
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
    dense_layer_launch(L, 1);
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
  cuda::gemv_f32_out_f16x(lw.router, xn_row, route_logits_.get(), E, H, s);
  cuda::router_topk(route_logits_.get(), route_idx_.get(), route_w_.get(), E,
                    K, s);

  // Shared expert into combine row K (weight 1, see prepare()).
  detail::shared_expert_forward(w_, layer, d_hq_.get(), shared_y_.get(), H,
                                FFX, s);

  // Router decision to the host + the miss-path activation row (C1).
  cudaMemcpyAsync(h_route_idx_.get(), route_idx_.get(), K * sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_route_w_.get(), route_w_.get(), K * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_x16_.get(), xn_row, H * sizeof(uint16_t),
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

  int n_hit = 0, n_miss = 0;
  for (int kk = 0; kk < K; ++kk) {
    const int e = h_route_idx_[kk];
    cache_.touch(layer, e);
    if (cache_.resident(layer, e)) {
      const size_t base = (static_cast<size_t>(layer) * 3 + 0) * E + e;
      h_ptrs_[0 * K + n_hit] = hot_->tables_host[base];
      h_ptrs_[1 * K + n_hit] = hot_->tables_host[base + static_cast<size_t>(E)];
      h_ptrs_[2 * K + n_hit] =
          hot_->tables_host[base + 2 * static_cast<size_t>(E)];
      h_hit_rows_[n_hit] = kk;
      h_slot_of_k_[kk] = -1;
      ++n_hit;
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

  // CPU misses (C2: no router weights here; C3: fixed pinned slots; C6:
  // submitted before the GPU finishes hits+shared so both run in parallel).
  // Expert layout in the pinned pool: [gate | up | out], each gu bytes.
  if (n_miss > 0) {
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
// Head
// ---------------------------------------------------------------------------
void Session::head_launch() {
  const EngineDims& d = mc_.dims;
  cudaStream_t s = stream_.get();
  cuda::rmsnorm_f16(xn_.get(), x_.get(), w_.output_norm, d.hidden,
                    mc_.norm_eps, s);
  cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
  cuda::gemv_q4_k_q8(Q4W(w_.output), d_xq_.get(), logits_.get(), d.n_vocab,
                     d.hidden, s);
}

void Session::head_launch_row(int row) {
  const EngineDims& d = mc_.dims;
  cudaStream_t s = stream_.get();
  cuda::rmsnorm_f16(xn_.get(), x_.get() + static_cast<size_t>(row) * d.hidden,
                    w_.output_norm, d.hidden, mc_.norm_eps, s);
  // MTP stash: prefill's tail row is the hidden the first draft conditions on.
  cudaMemcpyAsync(xf_.get(), xn_.get(), d.hidden * sizeof(uint16_t),
                  cudaMemcpyDeviceToDevice, s);
  cuda::quantize_q8_k(xn_.get(), d_xq_.get(), d.hidden, s);
  cuda::gemv_q4_k_q8(Q4W(w_.output), d_xq_.get(), logits_.get(), d.n_vocab,
                     d.hidden, s);
  cudaMemcpyAsync(logits_host_.get(), logits_.get(),
                  d.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
  stream_.sync();
}

// ---------------------------------------------------------------------------
// Decode step (graph path)
// ---------------------------------------------------------------------------
const float* Session::decode_step(int last_id) {
  if (!prepared_) return nullptr;
  const EngineDims& d = mc_.dims;
  const double t0 = now_ms();
  cudaStream_t s = stream_.get();

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

  for (int L = 0; L < d.n_layer; ++L) {
    if (graphs_.valid() && graphs_.layer_graph(L)) {
      cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphs_.layer_graph(L)), s);
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

  for (int L = 0; L < d.n_layer; ++L) {
    if (abort_flag && *abort_flag) return Status::kAborted;
    dense_layer_launch(L, n);
    for (int t = 0; t < n; ++t) moe_dispatch_row(L, t);
  }
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
const float* Session::last_logits() const { return nullptr; }

#endif  // SAGER_HAVE_CUDA

}  // namespace sager
