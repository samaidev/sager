// suffix_drafter.cpp - prompt-lookup / suffix drafting (model-free).
//
// A 3-gram hash over the recent token context proposes continuations for
// free. Agentic traces are full of exact repeats (tool-call frames, JSON
// scaffolding, echoed documents), which is where this drafter beats MTP on
// acceptance - and it costs zero model time to propose.
#include "sager/spec/suffix_drafter.hpp"

#include <algorithm>

namespace sager {

void SuffixDrafter::update(const std::vector<int>& context, int from_pos) {
  // (Re)point at the caller-owned context; index forward from from_pos.
  context_ = &context;
  const int n = static_cast<int>(context.size());
  int start = std::max(from_pos, 3);   // need 3 tokens of history
  for (int i = start; i < n; ++i) {
    const uint64_t h = hash3(context[static_cast<size_t>(i) - 3],
                             context[static_cast<size_t>(i) - 2],
                             context[static_cast<size_t>(i) - 1]);
    auto& v = table_[h];
    const int nxt = context[static_cast<size_t>(i)];
    if (std::find(v.begin(), v.end(), nxt) == v.end()) {
      if (v.size() < 4) v.push_back(nxt);   // cap alternates per 3-gram
    }
  }
  indexed_up_to_ = n;
}

std::vector<int> SuffixDrafter::draft(const int* last3) {
  std::vector<int> out;
  if (!last3) return out;
  const auto it = table_.find(hash3(last3[0], last3[1], last3[2]));
  if (it == table_.end()) return out;
  out = it->second;
  if (static_cast<int>(out.size()) > max_draft_) {
    out.resize(static_cast<size_t>(max_draft_));
  }
  return out;
}

void SuffixDrafter::reset() {
  table_.clear();
  context_ = nullptr;
  indexed_up_to_ = 0;
}

}  // namespace sager
