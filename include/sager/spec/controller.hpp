// controller.hpp - the speculative-decoding controller.
//
// Drafting is not free: each draft token costs one MTP forward, misses cost
// CPU pool time, and the verify window serializes a bigger batch. So the
// controller carries an explicit cost model and only drafts when the
// expected gain beats the measured cost:
//
//   E[gain] = A(d) * d * t_decode  -  (cost_dense(d) + cost_miss(d) + sync)
//   draft iff E[gain] > 0
//
// where A(d) is the per-depth acceptance prior, initialized from Strata's
// published priors (2.4-3.2 tokens/pass) and LEARNED per session from
// observe() feedback. It starts from priors, never from guesses.
#pragma once

#include <cstdint>
#include <vector>

namespace sager {

class SpecController {
 public:
  struct Config {
    int   max_depth        = 3;      // MTP n-max (model card: 3)
    float decode_ms_per_tok = 45.0;  // measured; updated by the session
    float mtp_ms_per_draft  = 4.0;   // MTP head forward cost
    float miss_ms_expected  = 6.0;   // E[miss cost] from pool stats
    float sync_ms           = 1.5;   // verify window sync overhead
    float accept_tau        = 0.15;  // EWMA smoothing for acceptance
  };

  explicit SpecController(Config cfg) : cfg_(cfg) {
    priors_.assign(cfg_.max_depth + 1, 0.0);
    if (priors_.size() > 1) {
      // Strata priors: ~2.4-3.2 tokens accepted per pass -> per-depth rate
      // slightly decreasing in depth.
      for (size_t d = 1; d < priors_.size(); ++d) {
        priors_[d] = 0.62 - 0.04 * static_cast<double>(d);
        if (priors_[d] < 0.35) priors_[d] = 0.35;
      }
    }
  }

  // Should we draft a window of depth `d` right now?
  bool should_draft(int d, double pool_miss_ms_now) const;

  // Depth to draft (1..max_depth), given the current state.
  int  choose_depth(double pool_miss_ms_now) const;

  // Learn: `accepted` of `drafted` tokens were accepted at this depth.
  void observe(int depth, int drafted, int accepted);

  // Session updates live measurements.
  void set_decode_ms(double ms_per_token) { cfg_.decode_ms_per_tok = float(ms_per_token); }
  void set_miss_ms(double ms)             { cfg_.miss_ms_expected = float(ms); }

  double acceptance_prior(int depth) const {
    if (depth <= 0 || depth >= static_cast<int>(priors_.size())) return 0.0;
    return priors_[depth];
  }

  const Config& config() const { return cfg_; }

 private:
  Config cfg_;
  std::vector<double> priors_;   // per-depth acceptance rate (EWMA)
};

}  // namespace sager
