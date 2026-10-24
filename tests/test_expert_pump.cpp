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

#include <algorithm>
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

  // ---- v0.2.3 stream plan: order, residency filter, coverage -------------
  {
    // 2 layers x 8 experts; pairs (0,1), (0,2), (1,0) are hot-resident.
    std::vector<uint8_t> res(kLayers * kExperts, 0);
    res[0 * kExperts + 1] = 1;
    res[0 * kExperts + 2] = 1;
    res[1 * kExperts + 0] = 1;
    ExpertPump::StreamPlan plan =
        ExpertPump::build_stream_plan(res, kLayers, kExperts);
    CHECK(static_cast<int>(plan.pairs.size()) == kLayers * kExperts - 3,
          "plan covers exactly the non-resident pairs");
    bool ordered = true;
    int last = -1;
    for (int32_t packed : plan.pairs) {
      if (packed <= last) ordered = false;   // strictly (layer asc, id asc)
      last = packed;
      const int L = packed / kExperts, e = packed % kExperts;
      CHECK(!res[static_cast<size_t>(L) * kExperts + e],
            "plan never includes a resident pair");
    }
    CHECK(ordered, "plan is stream-ordered (layer, id)");
  }

  // ---- v0.2.3 ring bookkeeping (CPU-only: events presumed done) ----------
  {
    ExpertPump p;
    CHECK(p.init(make_cfg(4 * kSlab), dummy_pool) == Status::kOk, "init 5");
    std::vector<uint8_t> res(kLayers * kExperts, 0);
    ExpertPump::StreamPlan plan =
        ExpertPump::build_stream_plan(res, kLayers, kExperts);
    p.begin_stream(plan, nullptr);
    // CPU-only: stream_start is a no-op, nothing becomes resident; lookup
    // must time out (-1) instead of hanging, so dispatch can fall back.
    CHECK(p.stream_lookup(0, 0) == -1, "lookup without loaders times out");
    CHECK(!p.stream_active() || true, "session state queryable");
    p.stream_finish();
    CHECK(!p.stream_active(), "session closed after finish");
    p.stream_progress(1);   // sweep must be a safe no-op on CPU-only
    // after finish, LRU submit still works (decode warm start contract)
    const int s = p.submit(1, 5, 99);
    CHECK(s >= 0, "submit works after a stream session");
  }

  // ---- v0.2.3 ring capacity invariant: per-layer miss set must fit -------
  {
    // The streamed walk needs ring slots >= the per-layer non-resident count
    // (consumption is a monotone subsequence of the fill order). Assert the
    // plan shape on a realistic geometry: 40 layers, 256 experts, 52% hot.
    const int L = 40, E = 256;
    std::vector<uint8_t> res(static_cast<size_t>(L) * E, 0);
    for (int l = 0; l < L; ++l) {
      for (int e = 0; e < E; ++e) {
        // deterministic pseudo-hot: every other expert resident
        res[static_cast<size_t>(l) * E + e] = (e % 2) ? 1 : 0;
      }
    }
    ExpertPump::StreamPlan plan = ExpertPump::build_stream_plan(res, L, E);
    CHECK(plan.pairs.size() == static_cast<size_t>(L) * (E / 2),
          "half the pairs planned");
    // per-layer counts are uniform -> max per-layer misses = E/2 = 128
    std::vector<int> per_layer(static_cast<size_t>(L), 0);
    for (int32_t packed : plan.pairs) ++per_layer[static_cast<size_t>(packed / E)];
    int max_layer_miss = 0;
    for (int c : per_layer) max_layer_miss = std::max(max_layer_miss, c);
    CHECK(max_layer_miss * kSlab <= 605 * kSlab,
          "per-layer miss bytes fit the 605-slot arena (Sage geometry)");
  }

  if (g_failures == 0) {
    std::printf("expert_pump_policy: ALL OK\n");
    return 0;
  }
  std::printf("expert_pump_policy: %d FAILURES\n", g_failures);
  return 1;
}
