// mtp.hpp - multi-token-prediction drafter.
//
// Sage-A1.5 ships its own MTP head (the model card documents speculative
// config `{"method":"mtp","num_speculative_tokens":3}`), so the drafter needs
// no auxiliary model: one MTP forward proposes the next k tokens, which the
// verify window checks in one batched pass. Expected gain 1.6-1.8x when
// acceptance is healthy (Strata figures; measured per-session here).
#pragma once

#include <vector>

#include "sager/core/session.hpp"

namespace sager {

class MtpDrafter {
 public:
  explicit MtpDrafter(Session& s, int max_draft) : s_(s), max_draft_(max_draft) {}

  bool ready() const { return ready_; }
  void set_ready(bool r) { ready_ = r; }

  // Draft up to max_draft() continuations of `last_token`. Returns ids
  // (possibly fewer if confidence decays - see controller thresholds).
  std::vector<int> draft(int last_token);

  int max_draft() const { return max_draft_; }

 private:
  Session& s_;
  int max_draft_;
  bool ready_ = false;
};

}  // namespace sager
