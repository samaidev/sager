// conversation.hpp - conversation state retention.
//
// After the first message, the token history (and per-turn KV checkpoints)
// is kept so follow-ups start in seconds instead of re-reading the whole
// prompt. Checkpointing policy: one KV snapshot per completed user turn
// (pos barrier), LRU-capped at `max_checkpoints` positions; a follow-up that
// shares a checkpointed prefix skips prefill for it entirely.
#pragma once

#include <deque>
#include <vector>

#include "sager/core/session.hpp"

namespace sager {

class ConversationMemory {
 public:
  void set_max_tokens(int max_tokens) { max_tokens_ = max_tokens; }
  void set_max_checkpoints(int n) { max_checkpoints_ = n; }

  // Longest checkpointed prefix length for this token history (0 if none).
  // `ids` is the FULL prompt (template applied) for this request.
  int reusable_prefix(const std::vector<int>& ids) const;

  // Register a completed turn: KV position + token count.
  void checkpoint(int kv_pos, const std::vector<int>& turn_ids);

  void reset();

  const std::vector<int>& history() const { return history_; }
  size_t history_tokens() const { return history_.size(); }

 private:
  struct Ckpt { int kv_pos; std::vector<int> ids; };
  std::vector<int> history_;
  std::deque<Ckpt> ckpts_;
  int max_tokens_ = 262144;
  int max_checkpoints_ = 4;
};

}  // namespace sager
