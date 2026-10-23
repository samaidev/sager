// verify.cpp - one-pass speculative verification.
//
// Contract (see core/verify.hpp): draft_ids[0] is the token the caller
// sampled and commits unconditionally; draft_ids[1..] are the drafter's
// candidates, verified greedily against the model's own distribution:
//   candidate c_i accepted  iff  argmax(R_{i-1}) == c_i
// where R_j is the logits row the model produced after consuming window
// token j. The horizon token (the model's own choice at the accept point)
// comes back in next_token; the KV cache is rolled back to the accepted
// prefix so nothing beyond it leaks into the next pass.
//
// Greedy verification keeps the output BIT-IDENTICAL to plain decoding:
// a draft can only accelerate, never alter, the token stream.
#include "sager/core/verify.hpp"

#include <cstring>
#include <vector>

#include "sager/log.hpp"

namespace sager {

VerifyResult verify_window(Session& s, const std::vector<int>& draft_ids) {
  VerifyResult out;
  const int n_vocab = s.model_config().dims.n_vocab;
  const int start_pos = s.pos();
  const int n = static_cast<int>(draft_ids.size());
  if (n <= 0) return out;

  // Rows: R_i = distribution after consuming window token i.
  // R_{-1} (the row that judged draft_ids[0]) is the session's last logits.
  const float* r_prev = s.last_logits();

  std::vector<float> rows(static_cast<size_t>(n) * n_vocab);
  Status st = s.forward_window(draft_ids, rows.data());
  if (st != Status::kOk) {
    // Window forward failed: restore position and fall back to plain decode.
    s.set_pos(start_pos);
    out.next_token = -1;
    return out;
  }

  auto argmax_of = [&](const float* v) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n_vocab; ++i) {
      if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
  };

  // Walk the accept chain. Window token 0 is committed unconditionally.
  int accepted = 0;                       // accepted candidates beyond [0]
  int horizon = 0;                        // index of the deciding row
  for (int i = 1; i < n; ++i) {
    const int greedy = argmax_of(rows.data() + static_cast<size_t>(i - 1) * n_vocab);
    if (greedy == draft_ids[i]) {
      ++accepted;
      horizon = i;
    } else {
      horizon = i - 1;                    // rejection at candidate i
      break;
    }
    if (i == n - 1) horizon = n - 1;      // all accepted: bonus from last row
  }

  // Rollback: KV keeps window[0] + accepted candidates only.
  s.set_pos(start_pos + 1 + accepted);

  // Sampler repetition window sees exactly the committed tokens.
  s.push_recent(draft_ids[0]);
  for (int i = 1; i <= accepted; ++i) s.push_recent(draft_ids[i]);

  out.accepted = accepted;
  const float* hrow = rows.data() + static_cast<size_t>(horizon) * n_vocab;
  out.next_token = argmax_of(hrow);
  out.top_logits.assign(hrow, hrow + n_vocab);

  SLOG_D("verify", "window=%d accepted=%d next=%d (pos %d -> %d)", n, accepted,
         out.next_token, start_pos, s.pos());
  (void)r_prev;
  return out;
}

}  // namespace sager
