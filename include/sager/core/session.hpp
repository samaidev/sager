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
#include "sager/core/expert_source.hpp"
#include "sager/core/weights.hpp"
#include "sager/core/tensor.hpp"
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

// Decode CUDA graphs: one graph per layer, baked at prepare().
class GraphSet {
 public:
  bool valid() const { return valid_; }
  void set_valid(bool v) { valid_ = v; }
  void* layer_graph(int i) const { return graphs_[static_cast<size_t>(i)]; }
  void  set_layer_graph(int i, void* g) { graphs_[static_cast<size_t>(i)] = g; }
  int   count() const { return static_cast<int>(graphs_.size()); }
  void  init(int n) { graphs_.assign(static_cast<size_t>(n), nullptr); }
 private:
  bool valid_ = false;
  std::vector<void*> graphs_;
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

  HostCallbacks cb;

 private:
  friend struct detail::HotTier;
  friend class MtpDrafter;   // drafter reuses the stream + hidden stash
  Status alloc_scratch();
  Status capture_graphs();
  void   dense_layer_launch(int layer, int n_tokens);  // graph body
  void   moe_dispatch(int layer);                       // eager MoE half (row 0)
  void   moe_dispatch_row(int layer, int row);          // eager MoE for row r
  void   head_launch();                                 // final norm+head
  void   head_launch_row(int row);                      // per-row head (prefill tail)

  Weights&      w_;
  ExpertCache&  cache_;
  CpuPool&      pool_;
  ModelConfig   mc_;
  CtxParams     cp_;

  KVCache   kv_;
  GraphSet  graphs_;
  std::unique_ptr<detail::HotTier> hot_;

#if defined(SAGER_HAVE_CUDA)
  CudaStream stream_;
  DeviceBuffer<SeqParams> d_params_;
  DeviceBuffer<int> d_ids_, d_pos_ids_;
  DeviceBuffer<uint16_t> x_, x2_, xn_, xn2_, q_, k_, v_, attn_out_, o_proj_;
  DeviceBuffer<uint16_t> expert_g_, expert_u_, expert_y_, shared_y_;
  DeviceBuffer<uint16_t> xf_, d_cat_, d_m1_;
  DeviceBuffer<float>    route_logits_, d_miss_y_;
  DeviceBuffer<int>      route_idx_, d_slot_of_k_, d_hit_rows_;
  DeviceBuffer<q4k::BlockQ4K*> d_ptrs_hit_;   // fused ptr-table (3*K rows)
  DeviceBuffer<float>    route_w_;
  DeviceBuffer<q4k::BlockQ8K> d_xq_, d_hq_, d_mq_;
  DeviceBuffer<float>    logits_, mtp_logits_;
  DeviceBuffer<int>      d_recent_;
  PinnedBuffer<int>      h_route_idx_, h_slot_of_k_;
  PinnedBuffer<float>    h_route_w_;
  PinnedBuffer<uint16_t> h_x16_;                 // xn_ row staging (miss path)
  PinnedBuffer<float>    miss_x_, miss_y_;
  PinnedBuffer<float>    logits_host_;
  PinnedBuffer<q4k::BlockQ4K*> h_ptrs_;          // compact hit ptr staging
  PinnedBuffer<int>      h_hit_rows_;            // hit slot -> routed slot k
#endif

  int pos_ = 0;
  std::vector<int> recent_ids_;
  std::vector<int> h_pos_tmp_;   // pinned-less pos staging for window forwards
  static constexpr int kMaxRecent = 64;

  bool prepared_ = false;
};

}  // namespace sager
