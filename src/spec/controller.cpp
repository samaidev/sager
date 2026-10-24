// controller.cpp - speculative decoding cost model.
//
// The controller owns the draft/no-draft decision. Its expected-gain model:
//
//   E[gain](d) = A(d) * d * t_decode - (d * t_mtp + E[t_miss] + t_sync)
//
// where A(d) is the per-depth acceptance EWMA (seeded from Strata's published
// 2.4-3.2 tokens/pass priors, learned per session via observe()), t_decode is
// the measured plain decode time per token, and the cost terms cover drafting
// plus the verify-window overhead. Drafting happens only when the expected
// gain is positive - speculation never slows the session down on this model.
#include "sager/spec/controller.hpp"

#include <algorithm>
#include <cmath>

namespace sager {

bool SpecController::should_draft(int d, double pool_miss_ms_now) const {
  if (d <= 0) return false;
  const double accept = acceptance_prior(d);
  if (accept <= 0.0) return false;
  const double gain = accept * static_cast<double>(d) * cfg_.decode_ms_per_tok -
                      (static_cast<double>(d) * cfg_.mtp_ms_per_draft +
                       std::max(pool_miss_ms_now, 0.0) + cfg_.sync_ms);
  return gain > 0.0;
}

int SpecController::choose_depth(double pool_miss_ms_now) const {
  for (int d = cfg_.max_depth; d >= 1; --d) {
    if (should_draft(d, pool_miss_ms_now)) return d;
  }
  return 0;
}

void SpecController::observe(int depth, int drafted, int accepted) {
  if (depth <= 0 || depth >= static_cast<int>(priors_.size())) return;
  if (drafted <= 0) return;
  const double rate = static_cast<double>(accepted) / drafted;
  priors_[static_cast<size_t>(depth)] +=
      cfg_.accept_tau * (rate - priors_[static_cast<size_t>(depth)]);
  priors_[static_cast<size_t>(depth)] =
      std::clamp(priors_[static_cast<size_t>(depth)], 0.05, 0.95);
}

}  // namespace sager
