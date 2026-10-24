// generate.hpp - the generation loop driver (engine-side, program-local).
//
// Bridges the session (one forward = one token) to the request semantics:
// sampling (Sage-A1.5 defaults per model card), speculative drafting driven
// by the cost-model controller, stop conditions, aborts and token callbacks.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "sager/common.hpp"
#include "sager/core/session.hpp"
#include "sager/core/mtp.hpp"
#include "sager/spec/controller.hpp"
#include "sager/spec/suffix_drafter.hpp"

namespace sager {

struct GenParams {
  std::vector<int> prompt_ids;      // template-applied token ids (from serve/)
  int   max_tokens = 512;
  // Sampling defaults = Sage-A1.5 card recommendations.
  float temperature      = 1.0f;
  int   top_k            = 20;
  float top_p            = 0.95f;
  float min_p            = 0.0f;
  float presence_penalty = 1.5f;
  float repeat_penalty   = 1.0f;
  bool  greedy           = false;   // argmax (used by bench + MTP verify)
  uint64_t seed          = 0;       // 0 = time-seeded
  std::vector<int> stop_ids;        // EOS etc. (checked after each commit)
};

struct GenCallbacks {
  // Return false to stop generation gracefully (client disconnect).
  std::function<bool(int)> on_token;
};

struct GenResult {
  int    tokens = 0;
  double prefill_tps = 0;
  double decode_tps = 0;
  int    drafted = 0;
  int    accepted = 0;
  Status status = Status::kOk;
};

Status run_generation(Session& s, SpecController& ctrl, MtpDrafter& mtp,
                      SuffixDrafter& suffix, const GenParams& p,
                      const std::atomic<bool>* abort, GenCallbacks cb,
                      GenResult* out);

// Host-side sampler: penalties -> temperature -> top-k -> top-p -> min_p.
int sample_host(const float* logits, int n_vocab, const std::vector<int>& recent,
                const GenParams& p, uint64_t& rng_state);

}  // namespace sager
