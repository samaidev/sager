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
  arm_timeout_ms_ = env_int("SAGER_PF_ARM_TIMEOUT_MS", 2000, 100, 60000);
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
    const std::vector<uint8_t>& resident, int n_layers, int n_experts) {
  StreamPlan plan;
  plan.n_layers = n_layers;
  plan.n_experts = n_experts;
  plan.pairs.reserve(resident.size());
  for (int l = 0; l < n_layers; ++l) {
    for (int e = 0; e < n_experts; ++e) {
      const size_t p = static_cast<size_t>(l) * n_experts + e;
      if (p < resident.size() && !resident[p]) {
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
    loaders_.emplace_back([this, t] { loader_loop(t); });
  }
  SLOG_I("pump", "stream-all prefill: %zu pairs (%.1f GB), ring %zu slots, "
         "%zu readers, %d slabs/reader",
         plan_.pairs.size(),
         plan_.pairs.size() * per_expert_bytes_ / 1073741824.0,
         slot_count_, loaders_.size(), slabs_per_reader_);
}

int ExpertPump::stream_lookup(int layer, int expert) {
  if (!stream_active_) return -1;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(arm_timeout_ms_);
  std::unique_lock<std::mutex> lk(stream_mu_);
  const size_t p = sidx(layer, expert);
  while (slot_of_[p] < 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      ++stats_.arm_timeouts;
      return -1;   // loader stalled (or pair dropped): CPU-pool fallback
    }
    stream_cv_.wait_for(lk, std::chrono::milliseconds(2));
  }
  return slot_of_[p];
}

void ExpertPump::stream_progress(int layer) {
#if defined(SAGER_HAVE_CUDA)
  if (!stream_active_) return;
  done_layer_.store(layer, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk(stream_mu_);
  for (size_t s = 0; s < slot_count_; ++s) {
    if (!ring_owned_[s] || owner_l_[s] < 0 || owner_l_[s] >= layer) continue;
    // Past-layer slots: parked pairs have no in-flight work (the chunk-bound
    // fence covers prior-chunk GEMMs); consumed pairs are released once the
    // compute stream actually finished their down projection; LOADED pairs
    // whose GEMM was never launched this chunk (drop-at-loading case) have
    // no in-flight reader either. LOADING slots wait for the publish-time
    // staleness check.
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
  stream_cv_.notify_all();
#else
  (void)layer;
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
void ExpertPump::loader_loop(int tid) {
  auto& slabs = loader_slabs_[static_cast<size_t>(tid)];
  size_t slab_i = 0;
  for (;;) {
    if (!stream_active_.load(std::memory_order_relaxed)) return;
    size_t i;
    {
      std::lock_guard<std::mutex> lk(stream_mu_);
      if (stream_next_ >= plan_.pairs.size()) return;
      i = stream_next_++;
    }
    const int32_t packed = plan_.pairs[i];
    const int L = packed / n_experts_;
    const int e = packed % n_experts_;
    const size_t p = static_cast<size_t>(packed);

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
      std::lock_guard<std::mutex> lk(stream_mu_);
      ++stats_.stream_skipped;
      continue;
    }

    // 1. arena slot: a FREE one (never owned, or a dropped pair parked),
    //    else steal a LOADED slot whose consumer event has completed.
    int slot = -1;
    for (;;) {
      if (!stream_active_.load(std::memory_order_relaxed)) return;
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
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
      std::this_thread::sleep_for(std::chrono::microseconds(150));
    }

    // 2. pinned slab of this reader (its DMA must have completed). Bounded
    //    wait: slabs free up as the copy engine drains; a pathological stall
    //    fails the pair instead of hanging the session.
    Slab* sb = nullptr;
    for (int attempt = 0; attempt < 2000 && !sb; ++attempt) {
      if (!stream_active_.load(std::memory_order_relaxed)) return;
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
      std::lock_guard<std::mutex> lk(stream_mu_);
      slot_of_[p] = -1;
      const size_t s = static_cast<size_t>(slot);
      ring_state_[s] = kRingFree;
      ring_owned_[s] = 0;
      owner_l_[s] = -1;
      owner_e_[s] = -1;
      ++stats_.read_errors;
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
      std::lock_guard<std::mutex> lk(stream_mu_);
      slot_of_[p] = -1;
      const size_t s = static_cast<size_t>(slot);
      ring_state_[s] = kRingFree;
      ring_owned_[s] = 0;
      owner_l_[s] = -1;
      owner_e_[s] = -1;
      ++stats_.read_errors;
      continue;   // dispatch layer's arm timeout routes this pair to the CPU
    }
    cudaEventRecord(events_[static_cast<size_t>(slot)], copy_stream_);
    cudaEventRecord(sb->ev, copy_stream_);
    sb->used = true;
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
        stream_cv_.notify_all();
      }
    }
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
#endif

}  // namespace sager
