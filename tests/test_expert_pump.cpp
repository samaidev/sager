// test_expert_pump.cpp - the residency POLICY half of the expert pump,
// exercised on a CPU-only build (copies are no-ops, events presumed done).
// Covers the contract clauses that can fail silently in production:
//   - reuse: a second submit of the same expert returns the SAME slot and
//     does not count as a new copy
//   - eviction: full arena evicts the least-recently-used completed slot
//   - stale residency: after eviction, resident() is false and the old
//     slot id is recycled to the new owner
//   - clear(): reset() semantics - everything becomes non-resident, stats stay
//   - fallback: with arena_bytes < per_expert_bytes every submit falls back
#include "sager/core/expert_pump.hpp"

#include <cstdio>
#include <set>
#include <vector>

using namespace sager;

static int g_failures = 0;

#define CHECK(cond, what)                                                    \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL %s (line %d)\n", what, __LINE__);                    \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int kLayers = 2;
constexpr int kExperts = 8;
constexpr size_t kSlab = 1024;  // per-expert bytes (tiny; policy is size-free)

ExpertPump::Config make_cfg(size_t arena_bytes) {
  ExpertPump::Config c;
  c.n_layers = kLayers;
  c.n_experts = kExperts;
  c.per_expert_bytes = kSlab;
  c.arena_bytes = arena_bytes;
  return c;
}

}  // namespace

int main() {
  // Valid dummy pool backing: on CUDA builds the pump performs real (tiny)
  // DMAs, so a null host base would fail init() and the submit path would
  // then run against an uninitialized pump (segfault). 2 layers x 8 experts
  // x 1 KiB = 16 KiB.
  static uint8_t dummy_pool[kLayers * kExperts * kSlab];
  for (size_t i = 0; i < sizeof(dummy_pool); ++i)
    dummy_pool[i] = static_cast<uint8_t>(i * 7 + 3);

  // ---- reuse: same expert submitted twice -> one slot, one copy ----------
  {
    ExpertPump p;
    CHECK(p.init(make_cfg(4 * kSlab), dummy_pool) == Status::kOk, "init");
    const int s1 = p.submit(0, 3, 1);
    CHECK(s1 >= 0, "first submit gets a slot");
    CHECK(p.resident(0, 3), "resident after submit");
    const int s2 = p.submit(0, 3, 2);
    CHECK(s2 == s1, "reuse returns the same slot");
    CHECK(p.stats().submits == 1 && p.stats().reuses == 1,
          "stats: one submit, one reuse");
    CHECK(p.resident_count() == 1, "one resident expert");
  }

  // ---- LRU eviction: fill the arena, touch oldest, submit new ------------
  {
    ExpertPump p;
    CHECK(p.init(make_cfg(2 * kSlab), dummy_pool) == Status::kOk, "init 2 slots");
    const int a = p.submit(0, 0, 1);
    const int b = p.submit(0, 1, 2);
    CHECK(a >= 0 && b >= 0 && a != b, "two distinct slots");
    CHECK(!p.resident(1, 0), "not resident yet");
    // arena full: submitting (1,0) must evict the LRU slot (a, step 1)
    const int c = p.submit(1, 0, 3);
    CHECK(c >= 0, "eviction made room");
    CHECK(c == a, "evicted the least recently used slot");
    CHECK(!p.resident(0, 0), "evicted expert no longer resident");
    CHECK(p.resident(0, 1), "touched expert stays");
    CHECK(p.resident(1, 0), "new expert resident");
    CHECK(p.stats().evictions == 1, "one eviction recorded");
    // the recycled slot belongs to (1,0) now
    CHECK(p.submit(1, 0, 4) == c, "new owner reuses its slot");
  }

  // ---- in-flight safety would need CUDA events; on CPU-only the policy is
  //      exercised above. The C7b guard itself is compile-checked.

  // ---- clear(): reset semantics ------------------------------------------
  {
    ExpertPump p;
    CHECK(p.init(make_cfg(4 * kSlab), dummy_pool) == Status::kOk, "init");
    p.submit(0, 0, 1);
    p.submit(1, 2, 1);
    CHECK(p.resident_count() == 2, "two residents before clear");
    p.clear();
    CHECK(p.resident_count() == 0, "clear drops all residency");
    CHECK(!p.resident(0, 0) && !p.resident(1, 2), "no stale residency");
    // slots are free again: same arena serves new owners
    const int s = p.submit(0, 5, 2);
    CHECK(s >= 0, "post-clear submit works");
    // stats survive (lifetime counters)
    CHECK(p.stats().submits == 3, "stats are lifetime counters");
  }

  // ---- fallback: arena too small for even one expert ---------------------
  {
    ExpertPump p;
    CHECK(p.init(make_cfg(kSlab / 2), dummy_pool) == Status::kOk, "init tiny");
    CHECK(p.submit(0, 0, 1) == -1, "no arena -> fallback (-1)");
    CHECK(p.stats().fallbacks == 1, "fallback counted");
    CHECK(!p.resident(0, 0), "nothing resident after fallback");
  }

  // ---- stress: more unique experts than slots, all submits succeed --------
  {
    ExpertPump p;
    CHECK(p.init(make_cfg(4 * kSlab), dummy_pool) == Status::kOk, "init 4");
    std::set<int> slots;
    uint64_t step = 1;
    for (int L = 0; L < kLayers; ++L) {
      for (int e = 0; e < kExperts; ++e) {
        const int s = p.submit(L, e, step++);
        CHECK(s >= 0, "submit always finds a slot under churn");
        slots.insert(s);
      }
    }
    CHECK(slots.size() <= 4u, "slot count respected");
    CHECK(p.stats().evictions + p.stats().fallbacks >= 12,
          "evictions or fallbacks covered the overflow");
  }

  if (g_failures == 0) {
    std::printf("expert_pump_policy: ALL OK\n");
    return 0;
  }
  std::printf("expert_pump_policy: %d FAILURES\n", g_failures);
  return 1;
}
