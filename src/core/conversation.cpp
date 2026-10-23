// conversation.cpp - conversation state retention (checkpointed prefixes).
//
// Follow-up requests that share a checkpointed prefix skip prefill for it:
// the server trims the prompt to the non-reusable suffix and prefills only
// that. Checkpoints are taken per completed user turn (one KV position
// barrier each) and LRU-capped; history is capped by max_tokens so long
// sessions evict whole turns instead of growing without bound.
#include "sager/core/conversation.hpp"

#include <algorithm>

namespace sager {

int ConversationMemory::reusable_prefix(const std::vector<int>& ids) const {
  int best = 0;
  for (const auto& c : ckpts_) {
    const size_t n = std::min(c.ids.size(), ids.size());
    size_t i = 0;
    while (i < n && c.ids[i] == ids[i]) ++i;
    if (i == c.ids.size() && static_cast<int>(i) > best) {
      best = static_cast<int>(i);   // full checkpointed prefix matches
    }
  }
  return best;
}

void ConversationMemory::checkpoint(int kv_pos,
                                    const std::vector<int>& turn_ids) {
  history_.insert(history_.end(), turn_ids.begin(), turn_ids.end());
  while (static_cast<int>(history_.size()) > max_tokens_) {
    const size_t drop = std::min(history_.size(),
                                 static_cast<size_t>(std::max(1, max_tokens_ / 8)));
    history_.erase(history_.begin(), history_.begin() + static_cast<long>(drop));
  }
  ckpts_.push_back(Ckpt{kv_pos, turn_ids});
  while (static_cast<int>(ckpts_.size()) > max_checkpoints_) ckpts_.pop_front();
}

void ConversationMemory::reset() {
  history_.clear();
  ckpts_.clear();
}

}  // namespace sager
