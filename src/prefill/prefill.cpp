// prefill.cpp - chunked prompt ingestion (Strata-style).
//
// The prompt walks through forward_window() in chunks of cp_.chunk_prefill
// (default 8192): KV grows incrementally, activation slabs stay fixed-size,
// and an abort flag is honored between chunks so a disconnected HTTP client
// stops burning GPU within one chunk. Only the FINAL row's logits are
// computed (head_launch_row), which the generate loop samples immediately.
//
// Conversation retention (core/conversation.hpp) lives ABOVE this: the server
// checks reusable_prefix() and calls prefill() with just the new suffix.
#include "sager/core/session.hpp"

#include <vector>

#include "sager/log.hpp"

namespace sager {

Status Session::prefill(const std::vector<int>& ids,
                        const std::atomic<bool>* abort_flag) {
  if (!prepared_) return Status::kInvalidArgument;
  if (ids.empty()) return Status::kOk;
  if (pos() + static_cast<int>(ids.size()) > cp_.n_ctx) {
    SLOG_E("prefill", "prompt %zu + pos %d exceeds ctx %d", ids.size(), pos(),
           cp_.n_ctx);
    return Status::kInvalidArgument;
  }

  const int eff_chunk = pf_batched_
      ? std::min(cp_.chunk_prefill > 0 ? cp_.chunk_prefill
                                       : static_cast<int>(ids.size()),
                 pf_chunk_cap_)
      : (cp_.chunk_prefill > 0 ? cp_.chunk_prefill
                               : static_cast<int>(ids.size()));
  const int chunk = std::max(1, std::min<int>(eff_chunk,
                                              static_cast<int>(ids.size())));
  const double t0 = now_ms();
  size_t done = 0;
  int n_chunks = static_cast<int>((ids.size() + chunk - 1) / chunk);

  // Strata-style stream-all sessions, one PER CHUNK: for a chunk big enough
  // that nearly every expert is routed, copies do not need the routing - all
  // hot-tier AND arena misses stream in fixed (layer, id) order through the
  // pump arena ring while the compute stream consumes that same order layer
  // by layer. Arena survivors of the previous chunk are parked (reused
  // without a copy) and the chunk-boundary fence keeps copies off the slots
  // the previous chunk's GEMMs still read.
#if defined(SAGER_HAVE_CUDA)
  const bool streaming = pf_batched_ && pump_ &&
                         static_cast<int>(ids.size()) >= pf_stream_min_ &&
                         w_.expert_pool_fd() >= 0;
  auto stream_begin_chunk = [this]() {
    if (!pump_) return;
    std::vector<uint8_t> miss(
        static_cast<size_t>(mc_.dims.n_moe_layer) * mc_.dims.n_expert, 0);
    for (int L = 0; L < mc_.dims.n_moe_layer; ++L) {
      for (int e = 0; e < mc_.dims.n_expert; ++e) {
        miss[static_cast<size_t>(L) * mc_.dims.n_expert + e] =
            (cache_.resident(L, e) || pump_->resident(L, e)) ? 0 : 1;
      }
    }
    ExpertPump::StreamPlan plan = ExpertPump::build_stream_plan(
        miss, mc_.dims.n_moe_layer, mc_.dims.n_expert);
    pump_->begin_stream(plan, stream_.get());
    pump_->stream_start(w_.expert_pool_fd());
  };
  auto stream_stop = [this]() {
    if (pump_ && pump_->stream_active()) pump_->stream_finish();
  };
#else
  auto stream_begin_chunk = []() {};
  auto stream_stop = []() {};
#endif

  for (size_t off = 0; off < ids.size(); off += static_cast<size_t>(chunk)) {
    if (abort_flag && *abort_flag) {
      stream_stop();
      return Status::kAborted;
    }
    const size_t end = std::min(ids.size(), off + static_cast<size_t>(chunk));
    std::vector<int> part(ids.begin() + static_cast<long>(off),
                          ids.begin() + static_cast<long>(end));
#if defined(SAGER_HAVE_CUDA)
    if (streaming) stream_begin_chunk();
#endif
    // forward_window: batched dense path + per-LAYER MoE dispatch; no
    // logits (nullptr) - the tail row's head runs once after the last chunk.
    Status s = forward_window(part, nullptr, abort_flag);
#if defined(SAGER_HAVE_CUDA)
    if (streaming) stream_stop();
#endif
    if (s != Status::kOk) {
      stream_stop();
      return s;
    }
    done = end;
    if (cb.on_prefill_progress) {
      cb.on_prefill_progress(static_cast<int>(done), n_chunks * chunk);
    }
  }

  // Tail logits: distribution for the position right after the prompt.
  const int last_row = static_cast<int>(
      ids.size() % static_cast<size_t>(chunk) == 0
          ? chunk - 1
          : static_cast<int>(ids.size() % static_cast<size_t>(chunk)) - 1);
  head_launch_row(last_row);

  const double ms = now_ms() - t0;
  perf.prefill_tps = ms > 0 ? 1000.0 * static_cast<double>(ids.size()) / ms
                            : 0.0;
  SLOG_I("prefill", "%zu tokens in %.0f ms (%.0f tok/s, pos now %d)",
         ids.size(), ms, perf.prefill_tps, pos());
  return Status::kOk;
}

}  // namespace sager
