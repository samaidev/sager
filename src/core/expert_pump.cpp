// expert_pump.cpp - see expert_pump.hpp for the design contract (C7a-C7d).
//
// The file is deliberately split so the residency policy is unit-testable
// without a GPU: everything below `#if defined(SAGER_HAVE_CUDA)` is copy
// plumbing; everything above is bookkeeping that runs identically on a
// CPU-only build (where copies are no-ops and events are presumed complete).
#include "sager/core/expert_pump.hpp"

#include <cstring>

#include "sager/log.hpp"

namespace sager {

ExpertPump::~ExpertPump() {
#if defined(SAGER_HAVE_CUDA)
  for (cudaEvent_t e : events_) {
    if (e) cudaEventDestroy(e);
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
    events_.resize(slot_count_, nullptr);
    for (size_t i = 0; i < slot_count_; ++i) {
      cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming);
    }
  }
  SLOG_I("pump", "arena ready: %zu slots x %.2f MB (%.0f MB VRAM)",
         slot_count_, per_expert_bytes_ / 1048576.0,
         slot_count_ * per_expert_bytes_ / 1048576.0);
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
  // NOTE: stats are intentionally kept across reset() - they are lifetime
  // counters for the /props monitor.
}

}  // namespace sager
