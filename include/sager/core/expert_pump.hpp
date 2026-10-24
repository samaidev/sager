// expert_pump.hpp - per-token expert streaming: the "miss" half of the MoE
// dispatch learns to DMA experts from the pinned RAM pool straight into a
// VRAM arena instead of (only) falling back to the CPU pool.
//
// Why: v0.1's miss path computes W_e @ x on the host (CpuPool). That is fine
// for a couple of misses, but on a 2-core host (Colab T4) the CPU pool is the
// decode bottleneck - the same wall llama.cpp hits with whole-layer offload.
// The pump instead treats an expert as DATA: one cudaMemcpyAsync of
// per_expert_bytes (~1.7 MB for the 512x2048 geometry) from the already-pinned
// expert pool into a rotating VRAM arena slot, overlapped with the GPU's
// hit-expert GEMVs, and computed ON the GPU. The expert stays in the arena,
// so the next token that routes to it is a pure VRAM hit.
//
// Three-tier view after v0.2:
//   tier 0  VRAM  hot experts (rebalance-driven residency table, unchanged)
//                 + pump arena (LRU slots filled on the decode path)
//                 + attention/router/norm/shared/embedding/head
//   tier 1  RAM   every routed expert, pinned (the pump's source)
//   tier 2  SSD   roadmap v0.3 (the pump's slot logic is where n-gram
//                 prefetch will plug in)
//
// Contracts (extending core/expert_source.hpp, clause C7):
//   C7a. The pump NEVER blocks the caller on a copy: submit() only records
//        and enqueues; the compute stream waits via events at GEMV time.
//   C7b. A slot evicted while its copy is still in flight is NOT evicted
//        (event-queried before reuse); if no slot is free the caller falls
//        back to the CPU pool for that expert - never a torn read.
//   C7c. Residency is hierarchical: hot tier first, then pump arena. The
//        dispatch layer owns that OR; the pump only manages its own slots.
//   C7d. All pump state is touched from the session thread only (single
//        producer), like the rest of the MoE dispatch.
//
// The bookkeeping (slot allocation, LRU eviction, reuse detection) is plain
// host code and is unit-tested on CPU-only builds; the CUDA copy path is a
// thin memcpyAsync + event layer on top.
#pragma once

#include <cstdint>
#include <vector>

#include "sager/common.hpp"
#include "sager/kernels/q4_k.hpp"

#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
#include "sager/core/tensor.hpp"
#endif

namespace sager {

class ExpertPump {
 public:
  struct Config {
    int n_layers = 0;
    int n_experts = 0;
    size_t per_expert_bytes = 0;   // one expert: [gate|up|out] Q4_K bytes
    size_t arena_bytes = 0;        // VRAM budget for streaming slots
  };

  ExpertPump() = default;
  ~ExpertPump();

  ExpertPump(const ExpertPump&) = delete;
  ExpertPump& operator=(const ExpertPump&) = delete;

  // host_pool_base must stay alive and page-locked for the pump's lifetime
  // (Weights::host_expert_base() qualifies - the loader pins it).
  Status init(const Config& cfg, const uint8_t* host_pool_base);

#if defined(SAGER_HAVE_CUDA)
  // Compute stream is the session's; copies go on the pump's own stream.
  cudaStream_t copy_stream() const { return copy_stream_; }
  const q4k::BlockQ4K* dev_base(int slot) const;
#endif

  // ---- bookkeeping (host-only; unit-tested on CPU-only builds) -----------

  bool resident(int layer, int expert) const {
    return slot_of_[sidx(layer, expert)] >= 0;
  }

  // Make (layer, expert) pump-resident and return its arena slot, or -1 if
  // no slot could be freed (caller must fall back to the CPU pool).
  // Reuse path: if already resident, returns the existing slot WITHOUT
  // recopying (the data is already in VRAM).
  int submit(int layer, int expert, uint64_t step);

  // True if submit() enqueued at least one new copy since the last arm_wait.
  bool pending() const { return !pending_.empty(); }

#if defined(SAGER_HAVE_CUDA)
  // Order the compute stream behind every copy submitted since the last
  // arm_wait (one cudaStreamWaitEvent per touched slot; copies share one
  // stream, so waiting on the newest event would suffice - we wait on all
  // touched slots to keep the reasoning local and auditable).
  void arm_wait(cudaStream_t compute);
#endif

  // Step bookkeeping (last-use stamps for LRU).
  void begin_step(uint64_t step) { step_ = step; }

  // Drop everything (reset(): conversation cleared, residency is stale).
  void clear();

  struct Stats {
    uint64_t submits = 0;     // submit() calls that found the expert absent
    uint64_t reuses = 0;      // submit() calls served from an existing slot
    uint64_t evictions = 0;
    uint64_t fallbacks = 0;   // no free slot -> CPU pool handled the miss
    size_t   bytes_copied = 0;
  };
  Stats stats() const { return stats_; }

  int slot_count() const { return static_cast<int>(slot_count_); }
  size_t per_expert_bytes() const { return per_expert_bytes_; }
  int resident_count() const;

 private:
  size_t sidx(int layer, int expert) const {
    return static_cast<size_t>(layer) * n_experts_ + expert;
  }
  int find_victim_locked();

  Config cfg_{};
  const uint8_t* host_base_ = nullptr;
  int n_experts_ = 0;
  size_t per_expert_bytes_ = 0;
  size_t slot_count_ = 0;
  uint64_t step_ = 0;
  Stats stats_;

  std::vector<int> slot_of_;          // (layer,expert) -> slot | -1
  std::vector<int> owner_l_, owner_e_; // slot -> (layer, expert) | -1
  std::vector<uint64_t> last_use_;     // slot -> step stamp
  std::vector<uint64_t> last_step_;    // per (layer,expert) stamp mirror
  std::vector<int> pending_;           // slots copied since last arm_wait
  size_t cursor_ = 0;                  // clock hand for eviction scans

#if defined(SAGER_HAVE_CUDA)
  void* arena_ = nullptr;
  cudaStream_t copy_stream_ = nullptr;
  std::vector<cudaEvent_t> events_;    // per-slot completion events
#endif
};

}  // namespace sager
