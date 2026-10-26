// session.hpp - the session loop: walks every layer, owns state that
// outlives a layer (KV cache, residual stream, hot-expert residency), and
// orchestrates decode CUDA graphs + the CPU expert pool doorbell (C6).
//
// One token's journey (generate.cpp):
//   embed -> for L in layers: {rmsnorm -> rope(QKV) -> attention -> o_proj ->
//            residual -> rmsnorm -> router -> [VRAM batched MoE + CPU misses]
//            -> moe_combine -> residual} -> out_norm -> head -> sample -> next
//
// The per-layer DENSE op sequence is captured into a replayable CUDA graph
// (fixed addresses; per-step values arrive through the d_params_/d_pos_ids_
// device blocks). The MoE dispatch stays EAGER on the host because it is
// data-dependent (router top-k changes per token) - the same split Strata
// makes between captured layer graphs and host-decided expert scheduling.
//
// Prefill is chunked (default 8192) and runs an eager path with overlapped
// H2D; conversation state is retained so follow-ups re-prefill only the new
// suffix (core/conversation.hpp).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "sager/common.hpp"
#include "sager/core/config.hpp"
#include "sager/core/expert_cache.hpp"
#include "sager/core/expert_pump.hpp"
#include "sager/core/expert_source.hpp"
#include "sager/core/weights.hpp"
#include "sager/core/tensor.hpp"
#include "sager/kernels/gdn.hpp"
#include "sager/kernels/q4_k.hpp"

#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
#include "sager/kernels/cuda_ops.hpp"
#endif

namespace sager {

#if defined(SAGER_HAVE_CUDA)
using sager::cuda::SeqParams;   // decode-step parameter block
#endif

// KV cache with optional per-token Q8 quantization (kv_q8).
class KVCache {
 public:
  struct Params { int n_layer, n_head_kv, head_dim, max_seq; bool q8; };
  Status init(const Params& p);
  void reset();
  // Raw slabs for the attention kernels (device pointers).
  uint8_t*  k_slab(int layer) { return k_[static_cast<size_t>(layer)]; }
  uint8_t*  v_slab(int layer) { return v_[static_cast<size_t>(layer)]; }
  int  len() const { return len_; }
  void set_len(int l) { len_ = l; }
  int  max_seq() const { return max_seq_; }
  bool q8() const { return q8_; }
  size_t bytes() const;   // for /props memory report
 private:
  int n_layer_ = 0, n_head_kv_ = 0, head_dim_ = 0, max_seq_ = 0, len_ = 0;
  bool q8_ = false;
  std::vector<uint8_t*> k_, v_;
  size_t slab_bytes_ = 0;
};

// Decode CUDA graphs: one graph per layer, baked at prepare(). v0.6 adds a
// second, FULL-layer graph per layer (dense/GDN body + the device-dispatch
// MoE half) captured at prepare with the decode residency table in place;
// replay is gated per layer on absent_cnt_[layer] == 0 (every expert of the
// layer resident in the table), so the captured MoE body can never hit the
// absent-expert mailbox. Gated layers replay the dense-only graph + eager
// MoE dispatch exactly as v0.5.
class GraphSet {
 public:
  bool valid() const { return valid_; }
  void set_valid(bool v) { valid_ = v; }
  bool full_valid() const { return full_valid_; }
  void set_full_valid(bool v) { full_valid_ = v; }
  void* layer_graph(int i) const { return graphs_[static_cast<size_t>(i)]; }
  void  set_layer_graph(int i, void* g) { graphs_[static_cast<size_t>(i)] = g; }
  void* full_layer_graph(int i) const {
    return full_graphs_[static_cast<size_t>(i)];
  }
  void  set_full_layer_graph(int i, void* g) {
    full_graphs_[static_cast<size_t>(i)] = g;
  }
  int   count() const { return static_cast<int>(graphs_.size()); }
  void  init(int n) {
    graphs_.assign(static_cast<size_t>(n), nullptr);
    full_graphs_.assign(static_cast<size_t>(n), nullptr);
  }
 private:
  bool valid_ = false;
  bool full_valid_ = false;
  std::vector<void*> graphs_;
  std::vector<void*> full_graphs_;
};

// Internal hot-tier state (defined in layer.cpp / layer_internal.hpp).
namespace detail { struct HotTier; }

class Session {
 public:
  struct HostCallbacks {
    std::function<void(int)> on_token;                 // decoded token id
    std::function<void(int, int)> on_prefill_progress; // (chunk, total)
  };

  Session(Weights& w, ExpertCache& cache, CpuPool& pool,
          const ModelConfig& mc, const CtxParams& cp);
  ~Session();

  Status prepare();   // alloc buffers, warmup, capture graphs

  // Chunked prefill of `ids` starting at current position (defined in
  // prefill.cpp). Aborts early if abort_flag set.
  Status prefill(const std::vector<int>& ids,
                 const std::atomic<bool>* abort_flag = nullptr);

  // One decode step for token `last_id`; returns host logits (n_vocab).
  const float* decode_step(int last_id);

  // Eager (non-graph) forward of a short token window at the current position:
  // appends KV for all tokens, returns per-row logits [n x n_vocab] in
  // `logits_out` (caller-allocated) and advances pos(). Used by the
  // speculative verify pass (core/verify.hpp) and by prefill's inner loop.
  Status forward_window(const std::vector<int>& window, float* logits_out,
                        const std::atomic<bool>* abort_flag = nullptr);

  // Host logits of the LAST completed forward (decode_step / prefill tail).
  const float* last_logits() const;
#if defined(SAGER_HAVE_CUDA)
  // v0.5 device sampler accessors (public: used by generate.cpp)
  bool device_sample_valid() const { return dev_sample_valid_; }
  void set_sampler_penalties(float rep, float pres) {
    sam_rep_ = rep;
    sam_pres_ = pres;
  }
  // Finish the device sample (softmax/top-p/min_p/multinomial on the host
  // over the k device-selected candidates). Returns the id, or -1 when no
  // valid device sample is pending.
  int  finish_device_sample(float temp, int top_k, float top_p, float min_p,
                            float rep_penalty, float presence_penalty,
                            uint64_t& rng);
#endif

  void reset();  // clear KV + positions (keep expert residency)

  KVCache&   kv()     { return kv_; }
  GraphSet&  graphs() { return graphs_; }
  Weights&   weights() { return w_; }
  const ModelConfig& model_config() const { return mc_; }
  const CtxParams& ctx_params() const { return cp_; }

  int  pos() const { return pos_; }
  void set_pos(int p) { pos_ = p; }
  void push_recent(int id);          // sampler repetition window
  const std::vector<int>& recent() const { return recent_ids_; }

  // MTP drafter plumbing (friend class MtpDrafter).
#if defined(SAGER_HAVE_CUDA)
  bool prepared_state() const { return prepared_; }
  CudaStream& stream() { return stream_; }
  uint16_t* hidden_stash() { return xf_.get(); }   // final hidden row (fp16)
#else
  bool prepared_state() const { return false; }
#endif

  struct Perf {
    double prefill_tps = 0;
    double decode_tps = 0;
    uint64_t steps = 0;
  };
  Perf perf;

  // v0.6 dev profiler (SAGER_PROF=1): accumulated segment timings in ms.
  // decode_step: per-layer dense-graph vs MoE-half split + head + sampler.
  // forward_window: same split (prefill chunks AND spec-decode verify rows)
  // plus which MoE dispatch path each call took. Accumulated from cudaEvents
  // after the stream syncs; ~100 event records per step, PROF runs only.
  struct Prof {
    bool on = false;
    double dec_upload = 0, dec_dense = 0, dec_moe = 0, dec_head = 0,
           dec_samp = 0;
    double fw_upload = 0, fw_dense = 0, fw_moe = 0, fw_head = 0;
    double mtp_draft = 0;
    int64_t dec_steps = 0;
    int64_t fw_calls = 0, fw_tokens = 0;
    int64_t fw_device = 0, fw_batched = 0, fw_rows = 0;
    int64_t spec_windows = 0, drafted = 0, accepted = 0;
  };
  Prof prof;

  HostCallbacks cb;

 private:
  friend struct detail::HotTier;
  friend class MtpDrafter;   // drafter reuses the stream + hidden stash
  Status alloc_scratch();
  Status capture_graphs();
  void   dense_layer_launch(int layer, int n_tokens);  // graph body (GQA)
  void   linear_layer_launch(int layer, int n_tokens); // graph body (GDN, v0.2)
  void   moe_dispatch(int layer);                       // eager MoE half (row 0)
  void   moe_dispatch_row(int layer, int row);          // eager MoE for row r
  void   moe_dispatch_batched(int layer, int n);        // v0.2.3 whole-chunk MoE
  void   moe_dispatch_device(int layer, int n);         // v0.4 device-side dispatch
  void   resolve_cpu_mailbox(int layer, int n);         // v0.4 CPU-pool patch
  void   patch_graph_mailbox(int layer, int cnt);       // v0.6 step-end patch
  void   drain_expert_touch(int n_rows);                // v0.4 chunk-level touch feed
  void   prof_accum_fw();                               // v0.6 fold fw events
  void   linear_layer_chunk(int layer, int n);          // v0.3 batched GDN walk
  void   head_launch();                                 // final norm+head
  void   head_launch_row(int row);                      // per-row head (prefill tail)
  void   zero_gdn_states();                             // conv + delta state reset

  Weights&      w_;
  ExpertCache&  cache_;
  CpuPool&      pool_;
  ModelConfig   mc_;
  CtxParams     cp_;

  KVCache   kv_;
  GraphSet  graphs_;
  std::unique_ptr<detail::HotTier> hot_;
  std::unique_ptr<ExpertPump> pump_;      // v0.2 per-token expert streaming

  // v0.2 layer-kind indirection (built in prepare()):
  //   kv_slot_of_layer_[L]  KV slab for full-attention layers (-1 for GDN)
  //   gdn_idx_of_layer_[L]  GDN state slab for linear layers (-1 for GQA)
  std::vector<int> kv_slot_of_layer_;
  std::vector<int> gdn_idx_of_layer_;

#if defined(SAGER_HAVE_CUDA)
  CudaStream stream_;
  DeviceBuffer<SeqParams> d_params_;
  DeviceBuffer<int> d_ids_, d_pos_ids_;
  DeviceBuffer<uint16_t> x_, x2_, xn_, xn2_, q_, k_, v_, attn_out_, o_proj_;
  DeviceBuffer<uint16_t> d_gate_;        // attn sigmoid-gate activations
  DeviceBuffer<float>    d_shared_gate_; // shared-expert scalar logit
  DeviceBuffer<uint16_t> d_logits16_;    // F16 head output (Q6_K->F16 files)
  DeviceBuffer<uint16_t> expert_g_, expert_u_, expert_y_, shared_y_;
  DeviceBuffer<uint16_t> xf_, d_cat_, d_m1_;
  DeviceBuffer<float>    route_logits_, d_miss_y_;
  DeviceBuffer<int>      route_idx_, d_slot_of_k_, d_hit_rows_;
  DeviceBuffer<q4k::BlockQ4K*> d_ptrs_hit_;   // fused ptr-table (3*K rows)
  DeviceBuffer<float>    route_w_;
  DeviceBuffer<q4k::BlockQ8K> d_xq_, d_hq_, d_mq_;
  DeviceBuffer<float>    logits_, mtp_logits_;
  DeviceBuffer<int>      d_recent_;
  DeviceBuffer<int>  d_recent_n_;             // v0.7: recent-window count (graph-safe)
  // v0.2 expert pump staging (C7: fixed addresses, session-thread only)
  DeviceBuffer<q4k::BlockQ4K*> d_ptrs_pump_;  // compact pump ptr table (3*K)
  DeviceBuffer<int>      d_pump_rows_;         // pumped row -> routed slot k
  DeviceBuffer<q4k::BlockQ8K> d_xq_moe_;        // row activation snapshot
  // v0.2.3 batched prefill (Strata-style streamed walk) - route slabs REUSE
  // route_logits_/route_idx_/route_w_ + their pinned mirrors (resized to
  // RT*(K+1) in alloc_scratch; the per-row decode path only ever touches
  // row 0, which keeps its exact v0.2 layout). All slabs are sized by
  // min(chunk_prefill, pf_chunk_cap_) so a big configured chunk cannot eat
  // VRAM the arena needs; prefill() chunks at the same cap.
  DeviceBuffer<int>      d_grows_;              // [Rmax] gather source rows
  DeviceBuffer<int>      d_gslot_;              // [RT * K] route -> grouped slot
  PinnedBuffer<int>      h_grows_;
  PinnedBuffer<int>      h_gslot_;
  DeviceBuffer<q4k::BlockQ8K> d_xqg_;          // [Rmax * H/256] compacted rows
  DeviceBuffer<q4k::BlockQ8K> d_hqg_;          // [Rmax * FFX/256] silu'd rows
  DeviceBuffer<uint16_t> expert_g2_, expert_u2_;  // [Rmax * FFX] gate/up outs
  DeviceBuffer<uint16_t> y_grouped_;            // [Rmax * H] expert downs
  DeviceBuffer<uint16_t> shared_yb_;            // [RT * H] shared expert outs
  DeviceBuffer<uint16_t> sh_g_, sh_u_;          // [RT * FFX] shared gate/up
  DeviceBuffer<float>    d_sgate_;              // [RT] shared scalar gates
  DeviceBuffer<int>      d_cflat_;              // CPU-fallback dst offsets
  PinnedBuffer<int>      h_cflat_;
  PinnedBuffer<int>      h_route_idx_, h_slot_of_k_;
  PinnedBuffer<float>    h_route_w_;
  PinnedBuffer<uint16_t> h_x16_;                 // xn_ row staging (miss path)
  PinnedBuffer<float>    miss_x_, miss_y_;
  PinnedBuffer<float>    logits_host_;
  PinnedBuffer<q4k::BlockQ4K*> h_ptrs_;          // compact hit ptr staging
  PinnedBuffer<int>      h_hit_rows_;            // hit slot -> routed slot k
  PinnedBuffer<q4k::BlockQ4K*> h_pump_ptrs_;     // compact pump ptr staging
  PinnedBuffer<int>      h_pump_rows_;           // pumped slot -> routed slot k
  // v0.2 GDN working set (fixed addresses -> graph-capturable)
  DeviceBuffer<uint16_t> lin_proj_, lin_ba_, lin_conv_, lin_qn_, lin_kn_;
  DeviceBuffer<uint16_t> lin_y_, lin_on_;
  DeviceBuffer<float>    lin_ab_;
  DeviceBuffer<uint16_t> lin_conv_state_;   // n_gdn * conv_dim * (ck-1)
  DeviceBuffer<float>    lin_delta_state_;  // n_gdn * nk * (ratio*V) * K
  // v0.3 chunk (batched) GDN slabs, sized [RT x rows]
  DeviceBuffer<uint16_t> lin_proj_b_, lin_conv_b_, lin_qn_b_, lin_kn_b_;
  DeviceBuffer<uint16_t> lin_y_b_, lin_on_b_, lin_ba_b_;
  DeviceBuffer<float>    lin_ab_b_;
  // v0.3 grouped MoE job tables (one GEMM launch per layer phase)
  DeviceBuffer<q4k::BlockQ4K*> d_job_wg_, d_job_wu_, d_job_wo_;
  DeviceBuffer<int>      d_job_off_, d_job_m_, d_job_tbase_, d_job_jot_;
  PinnedBuffer<q4k::BlockQ4K*> h_job_wg_, h_job_wu_, h_job_wo_;
  PinnedBuffer<int>      h_job_off_, h_job_m_, h_job_tbase_, h_job_jot_;
  bool pf_mtok_ = true;      // SAGER_MTOK=0 -> v0.2.3 batched kernels
  // v0.4 device-side expert dispatch (P1): chunk pointer table + device plan
  DeviceBuffer<q4k::BlockQ4K*> d_ptr_table_;   // [3][n_moe_layer][E] weights
  PinnedBuffer<q4k::BlockQ4K*> h_ptr_table_;   // chunk-start staging
  DeviceBuffer<int>      d_off_, d_toff_;      // [E+1] row / tile prefixes
  DeviceBuffer<int>      d_meta_;              // {R, tiles, bad_rows}
  DeviceBuffer<int>      d_mail_, d_mail_n_;   // absent-expert mailbox
  PinnedBuffer<int>      h_mail_n_;            // async mirror of d_mail_n_
  // v0.6 full-layer decode graphs: per-layer mailbox slots (the captured MoE
  // body writes its own layer's mail state; the host checks the pinned
  // mirrors after the step sync - by construction zero for replayed layers).
  DeviceBuffer<int>      d_mail_all_;          // [n_moe_layer * E]
  DeviceBuffer<int>      d_mail_n_all_;        // [n_moe_layer]
  PinnedBuffer<int>      h_mail_n_all_;        // [n_moe_layer] post-step mirror
  PinnedBuffer<int>      h_mail_all_;          // [n_moe_layer * E] anomaly detail
  // v0.6 preserved per-layer dispatch plan: lets the step-end CPU-pool patch
  // recompute absent-expert rows after the graphs ran (the single-buffer plan
  // is overwritten by later layers mid-step). Sizes are tiny (< 1 MB total).
  DeviceBuffer<int>      d_off_all_;           // [Lm][E+1]
  DeviceBuffer<int>      d_toff_all_;          // [Lm][E+1]
  DeviceBuffer<int>      d_meta_all_;          // [Lm][4]
  DeviceBuffer<int>      d_gslot_all_;         // [Lm][K]
  DeviceBuffer<int>      d_grows_all_;         // [Lm][K]
  DeviceBuffer<uint16_t> d_x16_all_;           // [Lm][H] layer input snapshot
  DeviceBuffer<float>    d_route_w_all_;       // [Lm][K+1]
  std::vector<int>       absent_cnt_;          // per MoE layer, from the table
  std::vector<uint8_t>   used_full_;           // per MoE layer, this step
  bool                   graphs_full_valid_ = true;   // SAGER_FULL_GRAPH=0 off
  bool                   head_q6k_ = true;     // SAGER_HEAD_Q6K native head
  bool                   topk2_ = true;        // SAGER_TOPK2=0 -> single-block
  bool                   gdn_ws_ = true;       // SAGER_GDN_WS=0 -> scalar scan
  int                    topk_seg_slots_ = 0;  // two-phase topk scratch slots
  bool                   moe_dbg_ = false;     // SAGER_MOE_DEBUG=1 step probes
  // CPU-pool patch staging (anomaly path only)
  PinnedBuffer<int>      h_mail_, h_off_;
  DeviceBuffer<int>      d_rk_;
  PinnedBuffer<int>      h_rk_;
  PinnedBuffer<int>      h_touch_idx_;         // [n_moe_layer][RT][K+1] async
                                               // route-id mirror for touch
  cudaEvent_t            ev_scan_ = nullptr;   // dispatch-scan completion
  // v0.6 profiler event rings (allocated in prepare() when prof.on)
  cudaEvent_t*           prof_ev_ = nullptr;   // decode ring
  cudaEvent_t*           fw_ev_ = nullptr;     // forward_window ring
  int                    prof_ev_n_ = 0, fw_ev_n_ = 0;
  int                    fw_ev_pe_ = 0;   // >0: forward_window events pending
  bool pf_device_ = true;    // SAGER_PF_DEVICE=0 -> v0.3 host dispatch
  int  mail_pending_ = -1;   // v0.4: layer whose CPU mailbox is unresolved
  int  pf_rt_ = 0;           // v0.4: batched route-slab row capacity (RT)

  // v0.5 decode fast path: whole-step pipelined decode on the v0.4 device
  // dispatch (no per-layer route sync) + device top-k sampler.
  bool dec_device_ = false;      // SAGER_DECODE_DEVICE=0 -> per-row walk
  bool dec_tbl_built_ = false;   // decode pointer table is current
  bool dec_mail_ok_ = false;     // mailbox patch allowed w/o stream session
  bool dev_sample_ = false;      // SAGER_DEVICE_SAMPLER=0 -> host sampler
  int  dev_sample_id_ = -1;      // sampled id of the last decode_step
  bool dev_sample_valid_ = false;
  int  dev_sample_k_ = 0;        // candidates produced by the device top-k
  float sam_rep_ = 1.0f;         // sampler penalties (set per request)
  float sam_pres_ = 0.f;
  std::vector<std::pair<int, int>> dec_promote_;  // mailed (layer, expert)
  DeviceBuffer<float>    d_sample_logits_;  // [n_vocab] sampler scratch
  DeviceBuffer<float>    d_topk_v_;         // [64] final candidates
  DeviceBuffer<int>      d_topk_i_;         // [64]
  DeviceBuffer<float>    d_topk_seg_v_;     // v0.6 two-phase scratch
  DeviceBuffer<int>      d_topk_seg_i_;
  PinnedBuffer<float>    h_topk_v_;         // [64]
  PinnedBuffer<int>      h_topk_i_;         // [64]
  PinnedBuffer<int>      h_step_stage_;     // {id | pos3 | SeqParams} staging
  PinnedBuffer<int>      h_recent_stage_;   // recent-id upload staging

  void build_decode_table();   // decode pointer table from residency
#endif

  int pos_ = 0;
  std::vector<int> recent_ids_;
  std::vector<int> h_pos_tmp_;   // pinned-less pos staging for window forwards
  static constexpr int kMaxRecent = 64;

  // v0.7 whole-step decode graph (embed + all layers in ONE graph; head +
  // sampler in a second graph). Cuts per-step CPU dispatch from ~40 graph
  // launches + eager calls to 2 launches - the 2-vCPU host can otherwise not
  // feed the T4 (68% GPU idle measured at 643 launches/step).
  Status capture_step_graph();
  void* step_graph_ = nullptr;    // cudaGraphExec_t, owned raw
  void* head_graph_ = nullptr;    // cudaGraphExec_t, owned raw
  bool  step_graph_ok_ = false;
  bool  step_graph_on_ = true;
  bool  fast_sample_ = true;   // device top-k on the legacy decode path
  bool  capturing_step_ = false;  // whole-step capture in progress

  // v0.2.3 prefill knobs (read once in prepare())
  bool pf_batched_ = true;    // SAGER_PF_BATCHED=0 -> legacy per-row walk
  int  pf_stream_min_ = 384;  // SAGER_PF_STREAM_MIN: tokens to arm stream-all
  int  pf_chunk_cap_ = 2048;  // SAGER_PF_CHUNK: batched route-buffer chunk cap
  int  pf_cpu_cap_ = 256;     // SAGER_PF_CPU_ROWS: CPU-pool fallback row cap

  bool prepared_ = false;
};

}  // namespace sager
