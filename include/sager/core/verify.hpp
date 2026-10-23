// verify.hpp - one-pass speculative verification window.
//
// Draft tokens (MTP and/or suffix drafter) are verified in a single batched
// forward: the window replays the draft prefix through the model, and each
// position is accepted iff its sampled token equals the draft token
// (greedy verify => bit-for-bit identical output to plain decoding).
// The commit step then folds the accepted prefix into the KV state.
#pragma once

#include <vector>

#include "sager/core/session.hpp"

namespace sager {

struct VerifyResult {
  int accepted = 0;                 // number of draft tokens accepted
  int next_token = -1;              // first non-accepted position's token
  std::vector<float> top_logits;    // logits at the accept horizon (for sampling)
};

// Verify up to `n_draft` draft tokens against the model, starting at the
// current position. `draft_ids` holds the candidate tokens; the final token
// of the window (model's own choice) is returned in next_token.
VerifyResult verify_window(Session& s, const std::vector<int>& draft_ids);

}  // namespace sager
