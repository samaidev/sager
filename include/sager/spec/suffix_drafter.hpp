// suffix_drafter.hpp - prompt-lookup / suffix drafting (no model involved).
//
// Repeated text (quoted code, document echoes, template answers) is common
// in Sage-A1.5 workloads: agentic traces re-emit tool call frames, JSON
// scaffolding and doc strings. A 3-gram hash over the recent context can
// then propose up to 5 tokens for free, which the verify window checks.
// This mirrors Strata's suffix drafter and typically pays off for copy-heavy
// segments where MTP acceptance is lower.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sager {

class SuffixDrafter {
 public:
  void set_max_draft(int n) { max_draft_ = n; }

  // Rebuild (incrementally) the n-gram index from `context` (recent tokens).
  void update(const std::vector<int>& context, int from_pos);

  // Given the LAST 3 tokens, propose up to max_draft continuations.
  std::vector<int> draft(const int* last3);

  void reset();

 private:
  static uint64_t hash3(int a, int b, int c) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](int v) {
      h ^= static_cast<uint64_t>(static_cast<uint32_t>(v)) + 0x9e3779b97f4a7c15ull;
      h *= 1099511628211ull;
    };
    mix(a); mix(b); mix(c);
    return h;
  }

  int max_draft_ = 5;
  std::unordered_map<uint64_t, std::vector<int>> table_;  // 3-gram -> next ids
  const std::vector<int>* context_ = nullptr;
  int indexed_up_to_ = 0;
};

}  // namespace sager
