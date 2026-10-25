// expert_pump.cpp - see expert_pump.hpp for the design contract (C7a-C7d).
//
// The file is deliberately split so the residency policy is unit-testable
// without a GPU: everything below `#if defined(SAGER_HAVE_CUDA)` is copy
// plumbing; everything above is bookkeeping that runs identically on a
// CPU-only build (where copies are no-ops and events are presumed complete).
#include "sager/core/expert_pump.hpp"

#include <cstring>

#include <algorithm>
#include <chrono>
#include <thread>

#include "sager/log.hpp"

#if defined(SAGER_HAVE_CUDA)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace sager {

// SAGER_PF_* knobs (read once at init; seconds/counts are deliberately
// conservative defaults tuned on a 2-vCPU Colab T4).
namespace {
int env_int(const char* name, int dflt, int lo, int hi) {
  const auto v = env_get(name);
  if (!v) return dflt;
  const int x = std::atoi(v->c_str());
  return x < lo ? lo : (x > hi ? hi : x);
}
}  // namespace

ExpertPump::~ExpertPump() {
#if defined(SAGER_HAVE_CUDA)
  stream_finish();  // no-op unless a session is somehow still live
  for (cudaEvent_t e : events_) {
    if (e) cudaEventDestroy(e);
  }
  for (cudaEvent_t e : consumed_ev_) {
    if (e) cudaEventDestroy(e);
  }
  for (cudaEvent_t e : pair_ev_) {
    if (e) cudaEventDestroy(e);
  }
  for (auto p : tbl_stage_) {
    if (p) cudaFreeHost(p);
  }
  for (auto& slabs : loader_slabs_) {
    for (Slab& sb : slabs) {
      if (sb.ev) cudaEventDestroy(sb.ev);
      if (sb.ptr) cudaFreeHost(sb.ptr);
    }
  }
  if (copy_stream_) cudaStreamDestroy(copy_stream_);
  if (arena_) cudaFree(arena_);
#endif
}

Status ExpertPump::init(const Config& cfg, const uint8_t* host_pool_base) {
  if (cfg.n_layers <= 0 || cfg.n_experts <= 0 || cfg.per_expert_bytes == 0) {
    return Status::kInvalidArgument;
  }
  if (!host_pool_base) {
#if defined(SAGER_HAVE_CUDA)
    return Status::kInvalidArgument;
#else
    // CPU-only build: the pointer is never dereferenced (copies are no-ops);
    // the policy unit tests rely on being able to pass null here.
    SLOG_W("pump", "null host pool on CPU-only build - copies are no-ops");
#endif
  }
  cfg_ = cfg;
  host_base_ = host_pool_base;
  n_experts_ = cfg.n_experts;
  per_expert_bytes_ = cfg.per_expert_bytes;
  slot_count_ = cfg.arena_bytes / cfg.per_expert_bytes;
  // v0.2.5b: the walk WAITS for the loader instead of CPU-fallbacking on a
  // 2 s timeout. The loader is strictly ahead-bound (plan order == walk
  // order), so the current layer's pairs arrive within seconds; the cv
  // wakes the lookup the moment a copy publishes, so a long timeout costs
  // nothing on the fast path. It only bounds a dead-loader scenario, and
  // even then the live_loaders_ check below fails fast.
  arm_timeout_ms_ = env_int("SAGER_PF_ARM_TIMEOUT_MS", 30000, 100, 120000);
  n_readers_ = env_int("SAGER_PF_READERS", 2, 1, 4);
  slabs_per_reader_ = env_int("SAGER_PF_SLABS", 16, 2, 64);

  const size_t pairs =
      static_cast<size_t>(cfg.n_layers) * static_cast<size_t>(cfg.n_experts);
  slot_of_.assign(pairs, -1);
  last_step_.assign(pairs, 0);
  owner_l_.assign(slot_count_, -1);
  owner_e_.assign(slot_count_, -1);
  last_use_.assign(slot_count_, 0);
  cursor_ = 0;
  stats_ = Stats{};

#if defined(SAGER_HAVE_CUDA)
  if (slot_count_ > 0) {
    if (cudaMalloc(&arena_, slot_count_ * per_expert_bytes_) != cudaSuccess) {
      arena_ = nullptr;
      SLOG_W("pump", "arena alloc (%.1f MB) failed - pump disabled",
             slot_count_ * per_expert_bytes_ / 1048576.0);
      slot_count_ = 0;
      return Status::kOomVram;
    }
    cudaStreamCreate(&copy_stream_);
    cudaEventCreateWithFlags(&fence_ev_, cudaEventDisableTiming);
    events_.resize(slot_count_, nullptr);
    consumed_ev_.resize(slot_count_, nullptr);
    for (size_t i = 0; i < slot_count_; ++i) {
      cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&consumed_ev_[i], cudaEventDisableTiming);
    }
    ring_state_.assign(slot_count_, kRingFree);
    ring_owned_.assign(slot_count_, 0);
    consumed_rec_.assign(slot_count_, 0);
    parked_.assign(slot_count_, 0);
  }
  SLOG_I("pump", "arena ready: %zu slots x %.2f MB (%.0f MB VRAM), stream-all "
         "prefill: %d readers x %d slabs, arm timeout %d ms",
         slot_count_, per_expert_bytes_ / 1048576.0,
         slot_count_ * per_expert_bytes_ / 1048576.0,
         n_readers_, slabs_per_reader_, arm_timeout_ms_);
#else
  SLOG_I("pump", "CPU-only build: pump bookkeeping active, copies are no-ops");
#endif
  return Status::kOk;
}

int ExpertPump::resident_count() const {
  int n = 0;
  for (int s : slot_of_) n += (s >= 0);
  return n;
}

int ExpertPump::find_victim_locked() {
  // 1) any free slot (clock hand makes this O(slots) worst case)
  for (size_t i = 0; i < slot_count_; ++i) {
    const size_t s = (cursor_ + i) % slot_count_;
    if (owner_l_[s] < 0) {
      cursor_ = (s + 1) % slot_count_;
      return static_cast<int>(s);
    }
  }
  // 2) LRU among slots whose copy has COMPLETED (C7b: never tear an
  //    in-flight copy). On CUDA builds an event query tells us; on CPU-only
  //    builds events are presumed done.
  int victim = -1;
  uint64_t oldest = ~0ull;
  for (size_t i = 0; i < slot_count_; ++i) {
    const size_t s = (cursor_ + i) % slot_count_;
#if defined(SAGER_HAVE_CUDA)
    if (cudaEventQuery(events_[s]) != cudaSuccess) continue;
#endif
    if (last_use_[s] < oldest) {
      oldest = last_use_[s];
      victim = static_cast<int>(s);
    }
  }
  if (victim >= 0) {
    const size_t s = static_cast<size_t>(victim);
    slot_of_[static_cast<size_t>(owner_l_[s]) * n_experts_ + owner_e_[s]] = -1;
    owner_l_[s] = -1;
    owner_e_[s] = -1;
    ++stats_.evictions;
    cursor_ = (s + 1) % slot_count_;
  }
  return victim;
}

int ExpertPump::submit(int layer, int expert, uint64_t step) {
  const size_t p = sidx(layer, expert);
  if (slot_of_[p] >= 0) {
    // already resident: touch and reuse (no copy, no event wait needed -
    // the prior copy completed before it was last used by construction of
    // the wait chain; arm_wait() still fences it for this stream).
    last_step_[p] = step;
    last_use_[static_cast<size_t>(slot_of_[p])] = step;
    ++stats_.reuses;
    return slot_of_[p];
  }
  ++stats_.submits;
  const int slot = find_victim_locked();
  if (slot < 0) {
    ++stats_.fallbacks;   // arena exhausted + all copies in flight (C7b)
    return -1;
  }
  const size_t s = static_cast<size_t>(slot);
  slot_of_[p] = slot;
  owner_l_[s] = layer;
  owner_e_[s] = expert;
  last_use_[s] = step;
  last_step_[p] = step;
#if defined(SAGER_HAVE_CUDA)
  const uint8_t* src = host_base_ +
      (static_cast<size_t>(layer) * n_experts_ + expert) * per_expert_bytes_;
  uint8_t* dst = static_cast<uint8_t*>(arena_) + s * per_expert_bytes_;
  if (cudaMemcpyAsync(dst, src, per_expert_bytes_, cudaMemcpyHostToDevice,
                      copy_stream_) != cudaSuccess) {
    // fall back cleanly: undo the bookkeeping, let the CPU pool take it
    slot_of_[p] = -1;
    owner_l_[s] = -1;
    owner_e_[s] = -1;
    ++stats_.fallbacks;
    cudaGetLastError();
    return -1;
  }
  cudaEventRecord(events_[s], copy_stream_);
  pending_.push_back(slot);
  stats_.bytes_copied += per_expert_bytes_;
#endif
  return slot;
}

#if defined(SAGER_HAVE_CUDA)
const q4k::BlockQ4K* ExpertPump::dev_base(int slot) const {
  return reinterpret_cast<const q4k::BlockQ4K*>(
      static_cast<const uint8_t*>(arena_) +
      static_cast<size_t>(slot) * per_expert_bytes_);
}

void ExpertPump::arm_wait(cudaStream_t compute) {
  for (int slot : pending_) {
    cudaStreamWaitEvent(compute, events_[static_cast<size_t>(slot)], 0);
  }
  pending_.clear();
}
#endif

void ExpertPump::clear() {
  for (size_t p = 0; p < slot_of_.size(); ++p) slot_of_[p] = -1;
  for (size_t s = 0; s < owner_l_.size(); ++s) {
    owner_l_[s] = -1;
    owner_e_[s] = -1;
  }
  pending_.clear();
  cursor_ = 0;
#if defined(SAGER_HAVE_CUDA)
  // No live session may reach here (stream_finish() runs first in every
  // lifecycle); reset ring bookkeeping so a stale session can't steal slots.
  ring_state_.assign(slot_count_, kRingFree);
  ring_owned_.assign(slot_count_, 0);
  consumed_rec_.assign(slot_count_, 0);
  parked_.assign(slot_count_, 0);
#endif
  // NOTE: stats are intentionally kept across reset() - they are lifetime
  // counters for the /props monitor.
}

// ---------------------------------------------------------------------------
// v0.2.3 prefill stream-all ring (Strata-style streamed walk)
// ---------------------------------------------------------------------------
ExpertPump::StreamPlan ExpertPump::build_stream_plan(
    const std::vector<uint8_t>& miss, int n_layers, int n_experts) {
  // v0.3 FIX: the caller (Session::prefill) passes the MISS bitmap
  // (1 = needs streaming). The old `!resident[p]` condition streamed the
  // HOT pairs instead - the plan held exactly the resident set (5309 pairs,
  // never the ~4931 misses), so the streamed prefill never streamed a miss:
  // with the old layer-major residency the walk's lookups fast-failed past
  // the plan (masked), and with uniform residency they stalled 60 s each.
  StreamPlan plan;
  plan.n_layers = n_layers;
  plan.n_experts = n_experts;
  plan.pairs.reserve(miss.size());
  for (int l = 0; l < n_layers; ++l) {
    for (int e = 0; e < n_experts; ++e) {
      const size_t p = static_cast<size_t>(l) * n_experts + e;
      if (p < miss.size() && miss[p]) {
        plan.pairs.push_back(static_cast<int32_t>(p));
      }
    }
  }
  return plan;
}

void ExpertPump::begin_stream(const StreamPlan& plan, void* compute_stream) {
#if defined(SAGER_HAVE_CUDA)
  if (stream_active_) stream_finish();   // defensive: never two live sessions
  // ONE chunk's session: the arena is NOT cleared. Resident slots (survivors
  // of the previous chunk / decode LRU) are parked into the ring - reusable
  // without a copy - and the plan covers only the misses. The fence orders
  // every copy of this chunk behind the compute stream, so parking can never
  // tear a GEMM that is still draining from the previous chunk.
  if (compute_stream && fence_ev_) {
    cudaEventRecord(fence_ev_, static_cast<cudaStream_t>(compute_stream));
    cudaStreamWaitEvent(copy_stream_, fence_ev_, 0);
  }
  for (size_t s = 0; s < slot_count_; ++s) {
    if (owner_l_[s] >= 0) {
      ring_owned_[s] = 1;
      ring_state_[s] = kRingLoaded;
      parked_[s] = 1;
      consumed_rec_[s] = 0;
    } else {
      ring_owned_[s] = 0;
      ring_state_[s] = kRingFree;
      parked_[s] = 0;
      consumed_rec_[s] = 0;
    }
  }
  pending_.clear();
  cursor_ = 0;
  plan_ = plan;
  stream_next_ = 0;
  done_layer_.store(-1, std::memory_order_relaxed);
  // v0.4 (P1): per-pair publish events + per-layer plan index ranges. Events
  // are REUSED across chunks (re-recording is safe: a StreamWaitEvent keeps
  // the state captured at its own enqueue time).
  if (pair_ev_.size() < plan.pairs.size()) {
    const size_t old = pair_ev_.size();
    pair_ev_.resize(plan.pairs.size(), nullptr);
    for (size_t i = old; i < pair_ev_.size(); ++i) {
      if (cudaEventCreateWithFlags(&pair_ev_[i], cudaEventDisableTiming) !=
          cudaSuccess) {
        pair_ev_[i] = nullptr;
        cudaGetLastError();
      }
    }
  }
  plan_layer_start_.assign(static_cast<size_t>(plan.n_layers) + 1, 0);
  for (int32_t packed : plan.pairs) {
    const int l = packed / n_experts_;
    if (l >= 0 && l < plan.n_layers) ++plan_layer_start_[l + 1];
  }
  for (size_t l = 1; l < plan_layer_start_.size(); ++l) {
    plan_layer_start_[l] += plan_layer_start_[l - 1];
  }
  // v0.2.4c: a loader killed mid-hunt leaves a stale in-flight marker;
  // never carry it into a new session (the lookup would wait forever).
  for (size_t p = 0; p < slot_of_.size(); ++p) {
    if (slot_of_[p] == -2) slot_of_[p] = -1;
  }
  stream_active_ = true;
#else
  (void)plan; (void)compute_stream;
#endif
}

#if defined(SAGER_HAVE_CUDA)
namespace {

// pread the full blob; returns false on short read / error (caller marks the
// pair failed and the dispatch layer falls back to the CPU pool).
bool pread_full(int fd, void* dst, size_t n, off_t off) {
  uint8_t* p = static_cast<uint8_t*>(dst);
  size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pread(fd, p + done, n - done, off + static_cast<off_t>(done));
    if (r <= 0) {
      if (r < 0 && (errno == EINTR)) continue;
      return false;
    }
    done += static_cast<size_t>(r);
  }
  return true;
}

}  // namespace

void ExpertPump::stream_start(int pool_fd) {
  if (!stream_active_) return;
  // v0.2.4d: entry observability - the stream-all path dead-ends silently
  // if this bails; make every bail visible on stderr.
  SLOG_I("pump", "stream_start: active=%d pool_fd=%d pairs=%zu slots=%zu",
         stream_active_.load(std::memory_order_relaxed), pool_fd,
         plan_.pairs.size(), slot_count_);
  {
    // v0.3 diag: plan composition - first pairs + per-layer counts
    std::string head;
    for (size_t i = 0; i < plan_.pairs.size() && i < 10; ++i) {
      const int32_t pp = plan_.pairs[i];
      head += " " + std::to_string(pp / n_experts_) + ":" +
              std::to_string(pp % n_experts_);
    }
    int l0 = 0;
    for (int32_t pp : plan_.pairs) if (pp / n_experts_ == 0) ++l0;
    SLOG_I("pump", "plan head:%s  layer0_pairs=%d first=%d last=%d",
           head.c_str(), l0,
           plan_.pairs.empty() ? -1 : (int)plan_.pairs.front(),
           plan_.pairs.empty() ? -1 : (int)plan_.pairs.back());
  }
  if (pool_fd < 0 || plan_.pairs.empty() || slot_count_ == 0) {
    SLOG_W("pump", "stream-all prefill: nothing to stream (fd=%d pairs=%zu slots=%zu)",
           pool_fd, plan_.pairs.size(), slot_count_);
    stream_active_ = false;
    return;
  }
  pool_fd_ = pool_fd;
  loader_slabs_.assign(static_cast<size_t>(n_readers_), {});
  int readers_with_slabs = 0;
  for (int t = 0; t < n_readers_; ++t) {
    auto& slabs = loader_slabs_[static_cast<size_t>(t)];
    slabs.resize(static_cast<size_t>(slabs_per_reader_));
    int ok = 0;
    for (Slab& sb : slabs) {
      if (cudaHostAlloc(&sb.ptr, per_expert_bytes_, cudaHostAllocDefault) !=
          cudaSuccess) {
        sb.ptr = nullptr;
        cudaGetLastError();
        break;   // this reader runs with fewer slabs; pread still works
      }
      cudaEventCreateWithFlags(&sb.ev, cudaEventDisableTiming);
      sb.used = false;
      ++ok;
    }
    if (ok > 0) ++readers_with_slabs;
  }
  if (readers_with_slabs == 0) {
    // No pinned staging at all: a pageable copy would serialize the whole
    // walk anyway - disable the session, dispatch falls back to submit()/CPU.
    SLOG_W("pump", "stream-all prefill: pinned slab alloc failed - session off");
    stream_active_ = false;
    return;
  }
  loaders_.reserve(static_cast<size_t>(readers_with_slabs));
  for (int t = 0; t < n_readers_; ++t) {
    bool any = false;
    for (const Slab& sb : loader_slabs_[static_cast<size_t>(t)]) any = sb.ptr != nullptr;
    if (!any) continue;
    // v0.4 (P1): per-reader pinned 3-pointer stage for the table patch.
    if (static_cast<int>(tbl_stage_.size()) <= t) tbl_stage_.resize(t + 1, nullptr);
    if (!tbl_stage_[static_cast<size_t>(t)] && d_table_) {
      if (cudaMallocHost(&tbl_stage_[static_cast<size_t>(t)],
                         3 * sizeof(q4k::BlockQ4K*)) != cudaSuccess) {
        tbl_stage_[static_cast<size_t>(t)] = nullptr;
        cudaGetLastError();
      }
    }
    try {
      loaders_.emplace_back([this, t] { loader_loop(t); });
      // v0.2.5: count the thread here (not inside loader_loop) so a slow
      // start cannot race the session's live-loader check; loader_loop
      // decrements on every exit path.
      live_loaders_.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      break;   // thread creation failed; live count stays honest
    }
  }
  SLOG_I("pump", "stream-all prefill: %zu pairs (%.1f GB), ring %zu slots, "
         "%zu readers, %d slabs/reader",
         plan_.pairs.size(),
         plan_.pairs.size() * per_expert_bytes_ / 1073741824.0,
         slot_count_, loaders_.size(), slabs_per_reader_);
}

int ExpertPump::stream_lookup(int layer, int expert) {
  if (!stream_active_) return -1;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(arm_timeout_ms_);
  std::unique_lock<std::mutex> lk(stream_mu_);
  const size_t p = sidx(layer, expert);
  // v0.2.5b: a pair that is not in the plan is never copied by any loader
  // (parked survivors are consumed via stream_slot_of). Waiting for it
  // used to burn the whole retry budget.
  // v0.3 FIX: `lower_bound(...) == end()` only rejected pairs ABOVE the
  // plan's max - a pair missing from the MIDDLE of the plan (a hot expert
  // the walk still routed to) fell into the full 60 s wait. Membership test.
  if (!std::binary_search(plan_.pairs.begin(), plan_.pairs.end(),
                          static_cast<int32_t>(p))) {
    return -1;
  }
  bool extended = false;
  while (slot_of_[p] < 0) {
    if (slot_of_[p] != -2) {
      // v0.2.4c: -1 with the walk already past this pair (take/publish-time
      // skip or read failure) means the pair can never publish - fail fast
      // instead of burning the session's retry budget on it. Without this
      // the cold-disk path livelocks: compute waits on a pair the loader
      // skipped, slots stay owned, loaders stall, nothing publishes.
      const auto it = std::lower_bound(plan_.pairs.begin(),
                                       plan_.pairs.end(),
                                       static_cast<int32_t>(p));
      if (it == plan_.pairs.end()) {
        // not in plan at all: never coming (mirrors the head check above)
        ++stats_.arm_timeouts;
        return -1;
      }
      const bool taken =
          *it == static_cast<int32_t>(p) &&
          stream_next_ > static_cast<size_t>(it - plan_.pairs.begin());
      if (taken || layer < done_layer_.load(std::memory_order_relaxed)) {
        ++stats_.arm_timeouts;
        return -1;
      }
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      // v0.2.5b: the deadline bounds a wedged walk, not the loader's lead -
      // with live readers the pair is bound to arrive (plan order == walk
      // order), so extend the wait ONCE (2 x arm_timeout) before giving up.
      // Readers gone: give up immediately.
      if (live_loaders_.load(std::memory_order_relaxed) == 0 || extended) {
        ++stats_.arm_timeouts;
        int loaded = 0, loading = 0, free_s = 0;
        for (size_t s2 = 0; s2 < slot_count_; ++s2) {
          if (ring_state_[s2] == kRingLoaded) ++loaded;
          else if (ring_state_[s2] == kRingLoading) ++loading;
          else ++free_s;
        }
        size_t plan_pos = plan_.pairs.size();
        for (size_t pi = 0; pi < plan_.pairs.size(); ++pi) {
          if (static_cast<size_t>(plan_.pairs[pi]) == p) { plan_pos = pi; break; }
        }
        SLOG_W("pump", "lookup FAIL: L=%d e=%d slot_of=%d plan_pos=%zu done_layer=%d live=%d ring loaded=%d loading=%d free=%d next=%zu plan=%zu reads=%llu drops=%llu",
               layer, expert, slot_of_[p], plan_pos,
               done_layer_.load(std::memory_order_relaxed),
               live_loaders_.load(std::memory_order_relaxed),
               loaded, loading, free_s, stream_next_, plan_.pairs.size(),
               (unsigned long long)stats_.read_errors,
               (unsigned long long)stats_.stream_drop);
        return -1;   // loader stalled or gone: CPU-pool fallback
      }
      deadline += std::chrono::milliseconds(arm_timeout_ms_);
      extended = true;
    }
    stream_cv_.wait_for(lk, std::chrono::milliseconds(2));
  }
  return slot_of_[p];
}

void ExpertPump::stream_progress(int layer) {
#if defined(SAGER_HAVE_CUDA)
  if (!stream_active_) return;
  done_layer_.store(layer, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(stream_mu_);
    reclaim_locked();
  }
  stream_cv_.notify_all();
#else
  (void)layer;
#endif
}

// Caller must hold stream_mu_. Frees every ring slot owned by a pair of a
// layer the session has already completed (owner_l_ < done_layer_):
//   - LOADED + consumed: released once the GEMM event actually fires
//   - LOADED + unconsumed (walk CPU-fallbacked the pair): nobody will ever
//     read the slot again - release immediately
//   - LOADING: still owned by a live reader; its publish-time staleness
//     check (session check 2) drops it there.
void ExpertPump::reclaim_locked() {
#if defined(SAGER_HAVE_CUDA)
  const int dl = done_layer_.load(std::memory_order_relaxed);
  if (dl < 0) return;
  for (size_t s = 0; s < slot_count_; ++s) {
    if (!ring_owned_[s] || owner_l_[s] < 0 || owner_l_[s] >= dl) continue;
    if (ring_state_[s] == kRingLoading) continue;
    if (!parked_[s] && consumed_rec_[s]) {
      if (cudaEventQuery(consumed_ev_[s]) != cudaSuccess) continue;
    }
    slot_of_[sidx(owner_l_[s], owner_e_[s])] = -1;
    owner_l_[s] = -1;
    owner_e_[s] = -1;
    ring_state_[s] = kRingFree;
    ring_owned_[s] = 0;
    parked_[s] = 0;
    consumed_rec_[s] = 0;
  }
#endif
}

void ExpertPump::stream_wait(cudaStream_t compute, int slot) {
  cudaStreamWaitEvent(compute, events_[static_cast<size_t>(slot)], 0);
}

void ExpertPump::stream_consume(cudaStream_t compute, int slot) {
  const size_t s = static_cast<size_t>(slot);
  cudaEventRecord(consumed_ev_[s], compute);
  std::lock_guard<std::mutex> lk(stream_mu_);
  consumed_rec_[s] = 1;
  parked_[s] = 0;
}

void ExpertPump::stream_consume_drop(int layer, int expert) {
  std::lock_guard<std::mutex> lk(stream_mu_);
  const size_t p = sidx(layer, expert);
  const int slot = slot_of_[p];
  if (slot < 0) return;
  const size_t s = static_cast<size_t>(slot);
  if (ring_state_[s] == kRingLoading) return;  // copy in flight; leave it be
  // The pair will never be consumed on the compute stream: park it as free
  // for the loaders (no event exists - the ring_state_ guard above is what
  // keeps an in-flight copy from being torn).
  ring_state_[s] = kRingFree;
  ring_owned_[s] = 0;
  consumed_rec_[s] = 0;
  parked_[s] = 0;
  owner_l_[s] = -1;
  owner_e_[s] = -1;
  slot_of_[p] = -1;
  ++stats_.stream_drop;
}

// ---- v0.4 (P1): device-side dispatch support -------------------------------

void ExpertPump::stream_set_table(void* d_table_base, int n_layers,
                                  int n_experts) {
  d_table_ = d_table_base;
  tbl_layers_ = n_layers;
  tbl_experts_ = n_experts;
}

void ExpertPump::stream_wait_layer(int layer, void* compute_stream) {
  if (layer < 0 ||
      layer + 1 >= static_cast<int>(plan_layer_start_.size())) {
    return;
  }
  auto* s = static_cast<cudaStream_t>(compute_stream);
  for (int i = plan_layer_start_[layer];
       i < plan_layer_start_[layer + 1]; ++i) {
    if (i < static_cast<int>(pair_ev_.size()) && pair_ev_[i]) {
      cudaStreamWaitEvent(s, pair_ev_[i], 0);
    }
  }
}

void ExpertPump::stream_consume_layer(int layer, void* compute_stream) {
  if (layer < 0 ||
      layer + 1 >= static_cast<int>(plan_layer_start_.size())) {
    return;
  }
  auto* s = static_cast<cudaStream_t>(compute_stream);
  // Snapshot the published slots under the mutex (publishes happen on the
  // loader threads), then record the consume events stream-side.
  std::vector<int> slots;
  {
    std::lock_guard<std::mutex> lk(stream_mu_);
    slots.reserve(static_cast<size_t>(plan_layer_start_[layer + 1] -
                                      plan_layer_start_[layer]));
    for (int i = plan_layer_start_[layer];
         i < plan_layer_start_[layer + 1]; ++i) {
      const int32_t packed = plan_.pairs[static_cast<size_t>(i)];
      const int slot = slot_of_[static_cast<size_t>(packed)];
      if (slot >= 0) {
        slots.push_back(slot);
        consumed_rec_[static_cast<size_t>(slot)] = 1;
      }
    }
  }
  for (int slot : slots) {
    cudaEventRecord(consumed_ev_[static_cast<size_t>(slot)], s);
  }
}

void ExpertPump::stream_finish() {
  if (!stream_active_) return;
  stream_active_ = false;
  {
    std::lock_guard<std::mutex> lk(stream_mu_);
    stream_cv_.notify_all();
  }
  for (auto& t : loaders_) t.join();
  loaders_.clear();
  for (auto& slabs : loader_slabs_) {
    for (Slab& sb : slabs) {
      if (sb.ev) cudaEventDestroy(sb.ev);
      if (sb.ptr) cudaFreeHost(sb.ptr);
      sb.ptr = nullptr;
      sb.ev = nullptr;
      sb.used = false;
    }
  }
  loader_slabs_.clear();
  pool_fd_ = -1;
  done_layer_.store(-1, std::memory_order_relaxed);
  // Surviving ring slots are plain LRU residents now (kept for decode AND
  // for the next chunk's parked reuse): stamp them so the eviction clock
  // sees them as freshly used.
  for (size_t s = 0; s < owner_l_.size(); ++s) {
    if (owner_l_[s] >= 0) {
      last_use_[s] = step_;
      last_step_[sidx(owner_l_[s], owner_e_[s])] = step_;
    }
    ring_owned_[s] = 0;
    consumed_rec_[s] = 0;
    parked_[s] = 0;
    ring_state_[s] = kRingFree;
  }
  SLOG_I("pump", "stream-all chunk done: %llu pairs, %.2f GB H2D total, "
         "%llu steals, %llu drops, %llu skipped, %llu arm timeouts, %llu read errors",
         (unsigned long long)stats_.stream_pairs,
         stats_.bytes_copied / 1073741824.0,
         (unsigned long long)stats_.stream_steals,
         (unsigned long long)stats_.stream_drop,
         (unsigned long long)stats_.stream_skipped,
         (unsigned long long)stats_.arm_timeouts,
         (unsigned long long)stats_.read_errors);
}

// Loader body: pull the next pair, take a slot (free, or steal a consumed
// one), stage pool -> pinned slab -> VRAM on the copy stream, publish.
// Each reader owns its slabs (no cross-thread slab sharing).
// v0.4 (P1): EVERY exit path records the plan pair's publish event (the
// compute stream's stream_wait_layer waits on it; a dropped pair must fire
// it too so the walk never hangs - its rows go to the CPU-pool mailbox),
// and a successful publish PATCHES the device pointer table row before the
// event is recorded (same stream -> ordered).
void ExpertPump::loader_loop(int tid) {
  // v0.2.5: every exit path (plan drained, session off, wedged hunt) must
  // decrement the live count - dispatch gates streaming on it.
  struct LiveGuard {
    std::atomic<int>& a;
    ~LiveGuard() { a.fetch_sub(1, std::memory_order_relaxed); }
  } lg{live_loaders_};
  auto record_pair = [this](size_t i) {
    if (i < pair_ev_.size() && pair_ev_[i]) {
      cudaEventRecord(pair_ev_[i], copy_stream_);
    }
  };
  auto& slabs = loader_slabs_[static_cast<size_t>(tid)];
  size_t slab_i = 0;
  for (;;) {
    if (!stream_active_.load(std::memory_order_relaxed)) return;
    size_t i;
    size_t p;
    {
      std::lock_guard<std::mutex> lk(stream_mu_);
      if (stream_next_ >= plan_.pairs.size()) return;
      i = stream_next_++;
      // v0.2.4c: publish intent BEFORE the (unbounded) slot hunt so
      // stream_lookup can tell "in flight" (-2) apart from "never coming"
      // (-1 with the walk already past this pair).
      p = static_cast<size_t>(plan_.pairs[i]);
      slot_of_[p] = -2;
    }
    const int32_t packed = static_cast<int32_t>(p);
    const int L = packed / n_experts_;
    const int e = packed % n_experts_;

    // v0.2.4: prime the page cache AHEAD of the walk (fd-based, no mmap).
    // Steady state on hosts whose RAM is smaller than the pool (12.7 GB RAM
    // vs the 16.9 GB pool on a Colab T4) re-reads the cold fraction of every
    // chunk from disk (~261 MB/s measured) - a POSIX_FADV_WILLNEED hint ~64
    // pairs (~108 MB) ahead turns those cold preads into overlapping
    // readahead instead of loader-thread stalls. Fire-and-forget: the
    // syscall is a hint and its return value carries no actionable error.
    {
      const size_t ahead = i + 64;
      if (ahead < plan_.pairs.size()) {
        const size_t pa = static_cast<size_t>(plan_.pairs[ahead]);
        posix_fadvise(pool_fd_,
                      static_cast<off_t>(pa) *
                          static_cast<off_t>(per_expert_bytes_),
                      per_expert_bytes_, POSIX_FADV_WILLNEED);
      }
    }

    // Session progress check 1: the session already passed this layer - the
    // pair will never be consumed, loading it would only clog a slot.
    if (L < done_layer_.load(std::memory_order_relaxed)) {
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        slot_of_[p] = -1;   // clear the in-flight marker: this is a final state
        ++stats_.stream_skipped;
      }
      record_pair(i);   // v0.4: the wait must not hang on a skipped pair
      continue;
    }

    // 1. arena slot: a FREE one (never owned, or a dropped pair parked),
    //    else steal a LOADED slot whose consumer event has completed.
    //    v0.3: BOUNDED hunt. The walk's tier loop blocks on lookups for the
    //    CURRENT layer BEFORE its GEMVs run, so slots only free after the
    //    whole tier loop completes; if the ring fills while this loader
    //    hunts, waiting forever deadlocks the walk on this very pair
    //    (measured: ring 606/606 full, both readers hunting, walk in
    //    stream_lookup). Give the pair up instead: slot_of_ -> -1, the
    //    walk's lookup fast-fails on `taken` and routes the rows to the CPU
    //    pool - a handful of rows per stall, and the walk's stream_progress
    //    frees this layer's slots so the loaders resume.
    int slot = -1;
    int hunts = 0;
    for (;;) {
      if (!stream_active_.load(std::memory_order_relaxed)) {
        record_pair(i);
        return;
      }
      if (++hunts > 20000) {
        std::lock_guard<std::mutex> lk(stream_mu_);
        slot_of_[p] = -1;
        ++stats_.stream_drop;
        SLOG_W("pump", "loader hunt gave up pair L=%d e=%d after %d spins "
                       "(ring full; walk will CPU-fallback this pair)",
               L, e, hunts);
        break;
      }
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        // v0.2.5: self-recovery. A reader wedged on a full ring used to wait
        // for the session's next layer boundary (stream_progress) - but the
        // session can BE blocked in a lookup waiting for this very reader,
        // a deadlock that only drained at ~120 s per pair on a cold Colab
        // T4. Reclaiming behind-the-session slots here breaks the cycle.
        reclaim_locked();
        for (size_t k = 0; k < slot_count_; ++k) {
          const size_t s = (cursor_ + k) % slot_count_;
          if (ring_state_[s] != kRingFree) continue;
          // FREE means no valid payload and nothing in flight (drop() and
          // the failure paths below never free an LOADING slot).
          slot = static_cast<int>(s);
          break;
        }
        if (slot < 0) {
          for (size_t k = 0; k < slot_count_; ++k) {
            const size_t s = (cursor_ + k) % slot_count_;
            if (!ring_owned_[s] || ring_state_[s] != kRingLoaded ||
                !consumed_rec_[s]) {
              continue;
            }
            if (cudaEventQuery(consumed_ev_[s]) != cudaSuccess) continue;
            // invalidate the old pair mapping, then reuse the slot
            slot_of_[sidx(owner_l_[s], owner_e_[s])] = -1;
            ++stats_.stream_steals;
            slot = static_cast<int>(s);
            break;
          }
        }
        if (slot >= 0) {
          const size_t s = static_cast<size_t>(slot);
          ring_state_[s] = kRingLoading;
          ring_owned_[s] = 1;
          consumed_rec_[s] = 0;
          cursor_ = (s + 1) % slot_count_;
          owner_l_[s] = L;
          owner_e_[s] = e;
          slot_of_[p] = slot;
          break;
        }
      }
      ++hunt_spins_;
      if ((hunt_spins_ % 20000) == 0) {
        int loaded = 0, loading = 0, free_s = 0;
        for (size_t s2 = 0; s2 < slot_count_; ++s2) {
          if (ring_state_[s2] == kRingLoaded) ++loaded;
          else if (ring_state_[s2] == kRingLoading) ++loading;
          else ++free_s;
        }
        SLOG_W("pump", "loader hunt stall: spins=%llu loaded=%d loading=%d free=%d done_layer=%d live=%d",
               (unsigned long long)hunt_spins_, loaded, loading, free_s,
               done_layer_.load(std::memory_order_relaxed),
               live_loaders_.load(std::memory_order_relaxed));
      }
      std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
    if (slot < 0) {
      record_pair(i);   // v0.4: dropped pair fires its event (CPU fallback)
      continue;
    }

    // 2. pinned slab of this reader (its DMA must have completed). Bounded
    //    wait: slabs free up as the copy engine drains; a pathological stall
    //    fails the pair instead of hanging the session.
    Slab* sb = nullptr;
    for (int attempt = 0; attempt < 2000 && !sb; ++attempt) {
      if (!stream_active_.load(std::memory_order_relaxed)) {
        record_pair(i);
        return;
      }
      for (int k = 0; k < static_cast<int>(slabs.size()); ++k) {
        Slab& cand = slabs[static_cast<size_t>((slab_i + k) % slabs.size())];
        if (!cand.ptr) continue;
        if (cand.used && cudaEventQuery(cand.ev) != cudaSuccess) continue;
        sb = &cand;
        slab_i = (slab_i + k + 1) % slabs.size();
        break;
      }
      if (!sb) std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    if (!sb) {
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        slot_of_[p] = -1;
        const size_t s = static_cast<size_t>(slot);
        ring_state_[s] = kRingFree;
        ring_owned_[s] = 0;
        owner_l_[s] = -1;
        owner_e_[s] = -1;
        ++stats_.read_errors;
      }
      record_pair(i);
      continue;   // dispatch layer's arm timeout routes this pair to the CPU
    }

    // 3. disk -> pinned slab -> VRAM (copy engine), then publish.
    const off_t off = static_cast<off_t>(p) * static_cast<off_t>(per_expert_bytes_);
    bool copy_ok = pread_full(pool_fd_, sb->ptr, per_expert_bytes_, off);
    if (copy_ok) {
      uint8_t* dst = static_cast<uint8_t*>(arena_) +
                     static_cast<size_t>(slot) * per_expert_bytes_;
      copy_ok = cudaMemcpyAsync(dst, sb->ptr, per_expert_bytes_,
                                cudaMemcpyHostToDevice, copy_stream_) ==
                cudaSuccess;
      if (!copy_ok) cudaGetLastError();
    }
    if (!copy_ok) {
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        slot_of_[p] = -1;
        const size_t s = static_cast<size_t>(slot);
        ring_state_[s] = kRingFree;
        ring_owned_[s] = 0;
        owner_l_[s] = -1;
        owner_e_[s] = -1;
        ++stats_.read_errors;
      }
      record_pair(i);
      continue;   // dispatch layer's arm timeout routes this pair to the CPU
    }
    cudaEventRecord(events_[static_cast<size_t>(slot)], copy_stream_);
    cudaEventRecord(sb->ev, copy_stream_);
    sb->used = true;
    bool published = false;
    {
      std::lock_guard<std::mutex> lk(stream_mu_);
      // Session progress check 2: the layer went by while we were copying.
      // The data is in VRAM but nobody will read it - release the slot.
      if (L < done_layer_.load(std::memory_order_relaxed)) {
        slot_of_[p] = -1;
        ring_state_[static_cast<size_t>(slot)] = kRingFree;
        ring_owned_[static_cast<size_t>(slot)] = 0;
        owner_l_[static_cast<size_t>(slot)] = -1;
        owner_e_[static_cast<size_t>(slot)] = -1;
        ++stats_.stream_skipped;
      } else {
        ring_state_[static_cast<size_t>(slot)] = kRingLoaded;
        ++stats_.stream_pairs;
        stats_.bytes_copied += per_expert_bytes_;
        published = true;
        stream_cv_.notify_all();
      }
    }
    // v0.4 (P1): patch the device pointer table row, THEN fire the pair's
    // publish event (same stream -> the compute stream's wait_layer sees a
    // valid pointer). Single stage per reader is safe: one stream, FIFO.
    if (published && d_table_ && tbl_stage_[static_cast<size_t>(tid)]) {
      auto* stage =
          reinterpret_cast<q4k::BlockQ4K**>(tbl_stage_[static_cast<size_t>(tid)]);
      const uint8_t* base = static_cast<const uint8_t*>(arena_) +
                            static_cast<size_t>(slot) * per_expert_bytes_;
      const size_t gu = per_expert_bytes_ / 3;
      stage[0] = reinterpret_cast<q4k::BlockQ4K*>(const_cast<uint8_t*>(base));
      stage[1] = reinterpret_cast<q4k::BlockQ4K*>(const_cast<uint8_t*>(base + gu));
      stage[2] = reinterpret_cast<q4k::BlockQ4K*>(const_cast<uint8_t*>(base + 2 * gu));
      cudaMemcpyAsync(static_cast<q4k::BlockQ4K**>(d_table_) +
                          (static_cast<size_t>(0) * tbl_layers_ + L) * tbl_experts_ + e,
                      stage, sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice,
                      copy_stream_);
      cudaMemcpyAsync(static_cast<q4k::BlockQ4K**>(d_table_) +
                          (static_cast<size_t>(1) * tbl_layers_ + L) * tbl_experts_ + e,
                      stage + 1, sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice,
                      copy_stream_);
      cudaMemcpyAsync(static_cast<q4k::BlockQ4K**>(d_table_) +
                          (static_cast<size_t>(2) * tbl_layers_ + L) * tbl_experts_ + e,
                      stage + 2, sizeof(q4k::BlockQ4K*), cudaMemcpyHostToDevice,
                      copy_stream_);
    }
    record_pair(i);
  }
}
#endif  // SAGER_HAVE_CUDA

#if !defined(SAGER_HAVE_CUDA)
// CPU-only builds: the streamer has no loaders and never activates, so the
// session API degenerates to "not found / nothing to join" (the policy tests
// exercise exactly these fallbacks).
int ExpertPump::stream_lookup(int, int) { return -1; }
void ExpertPump::stream_progress(int) {}
void ExpertPump::stream_finish() {}
void ExpertPump::stream_set_table(void*, int, int) {}
void ExpertPump::stream_wait_layer(int, void*) {}
void ExpertPump::stream_consume_layer(int, void*) {}
#endif

}  // namespace sager
