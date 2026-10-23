// expert_cache.hpp - the VRAM hot-expert residency table (Strata's heart).
//
// Model: per (layer, expert) pair keep an exponentially-decayed frequency
// score. The rebalance() pass fills free VRAM with the currently hottest
// experts and evicts cold ones. This is the kitchen-counter metaphor:
//   "常用的东西留在流理台，其余收进储藏室"
//
// Numbers to reason about (documented in docs/t4-guide.md):
//   - one expert is 3 mats x [n_ff_exp x hidden] Q4_K: with the 35B-A3B-class
//     geometry that is ~2.5-3 MB, with 4K-hidden variants up to ~7 MB;
//     expert_bytes() in core/config.hpp is the single source of truth
//   - roughly 350-400 more experts fit per extra GB of card memory
//   - a 10 GB pool holds ~60% of all routed experts, which - thanks to the
//     hot-expert locality of agentic workloads - covers >90% of requests
//     once warm (measured by bench/bench.py, see docs/performance.md)
//
// Persistence: the usage profile is written every 10 minutes and at exit
// (save_profile), and loaded at startup to skip the cold-start ramp.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "sager/common.hpp"
#include "sager/util/json.hpp"

namespace sager {

class ExpertCache {
 public:
  ExpertCache() = default;

  void init(int n_layers, int n_experts, size_t budget_bytes,
            size_t per_expert_bytes);

  // Query + record usage. resident() is const and lock-free on the hot path;
  // touch() bumps the decayed counter atomically.
  bool resident(int layer, int expert) const {
    const size_t i = idx(layer, expert);
    return resident_bits_[i >> 6] & (1ull << (i & 63));
  }
  void touch(int layer, int expert);

  // Decay every frequency score by factor (called once per generation step;
  // factor = exp(-dt / tau), tau ~ 2 min). Cheap: one FMUL over n_experts.
  void tick(float decay_factor);

  // Re-resolve residency against the budget. Returns number of experts
  // swapped. Called: after load, every N steps, and on demand.
  int rebalance();

  // Profile persistence (JSON via util/json.hpp).
  json::Value profile_to_json() const;
  void        profile_from_json(const json::Value& v);
  bool save_profile(const std::string& path) const;
  bool load_profile(const std::string& path);

  // Stats for /props and the web monitor.
  struct Stats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t rebalances = 0;
    int resident_count = 0;
    int capacity = 0;             // max experts the budget can hold
    double hit_rate() const { return hits + misses ? double(hits) / double(hits + misses) : 1.0; }
  };
  Stats stats() const;

  int  n_layers()  const { return n_layers_; }
  int  n_experts() const { return n_experts_; }
  int  capacity()  const { return capacity_; }

  // Direct residency control used by the loader (initial fill) and tests.
  void set_resident(int layer, int expert, bool on);

 private:
  size_t idx(int layer, int expert) const {
    return static_cast<size_t>(layer) * n_experts_ + expert;
  }

  int n_layers_ = 0;
  int n_experts_ = 0;
  size_t budget_bytes_ = 0;
  size_t per_expert_bytes_ = 0;
  int capacity_ = 0;

  std::vector<uint64_t> resident_bits_;
  std::vector<float>    freq_;          // decayed request counters
  std::vector<float>    persisted_;     // snapshot base for save/restart

  mutable std::atomic<uint64_t> hits_{0};
  mutable std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> rebalances_{0};
};

}  // namespace sager
