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

  const int chunk = cp_.chunk_prefill > 0 ? cp_.chunk_prefill : ids.size();
  const double t0 = now_ms();
  size_t done = 0;
  int n_chunks = static_cast<int>((ids.size() + chunk - 1) / chunk);

  for (size_t off = 0; off < ids.size(); off += static_cast<size_t>(chunk)) {
    if (abort_flag && *abort_flag) return Status::kAborted;
    const size_t end = std::min(ids.size(), off + static_cast<size_t>(chunk));
    std::vector<int> part(ids.begin() + static_cast<long>(off),
                          ids.begin() + static_cast<long>(end));
    // forward_window: eager dense-batched path + per-row MoE dispatch; no
    // logits (nullptr) - the tail row's head runs once after the last chunk.
    Status s = forward_window(part, nullptr, abort_flag);
    if (s != Status::kOk) return s;
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
