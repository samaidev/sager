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
//   tier 1  RAM   every routed expert (the pump's source; the prefill
//                 streamer stages it through its own pinned slabs)
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
//        producer), like the rest of the MoE dispatch - EXCEPT the v0.2.3
//        prefill streamer, whose loader threads mutate only the fields
//        guarded by stream_mu_ (see below).
//
// The bookkeeping (slot allocation, LRU eviction, reuse detection) is plain
// host code and is unit-tested on CPU-only builds; the CUDA copy path is a
// thin memcpyAsync + event layer on top.
//
// ---------------------------------------------------------------------------
// v0.2.3: the prefill stream-all ring (Strata-style streamed walk)
// ---------------------------------------------------------------------------
// Prefill was per-row: every (token, layer) dispatched its own top-k, synced
// the whole device, and submitted its own copies. A 512-token chunk paid
// 20k device syncs and the copy engine only raced single-row GEMVs. The fix
// follows Strata's streamed walk: for a chunk big enough that nearly every
// expert is routed (>= SAGER_PF_STREAM_MIN tokens, top-8 of 256), the exact
// routing is not needed to schedule copies - ALL hot-tier misses stream in
// fixed (layer asc, expert asc) order through the arena acting as a ring,
// while the compute stream consumes that same order layer by layer. Copies
// then overlap the attention halves; consumption is a monotone subsequence
// of the fill order, so the ring cannot deadlock (a slot the loader wants is
// only stolen after its "consumed" event - recorded by the compute stream
// after the expert's GEMMs - has completed).
//
// Session lifecycle (per prompt, from Session::prefill):
//   begin_stream(plan, compute) : plan = hot-tier AND arena misses in stream
//                        order. Does NOT clear the arena: resident slots are
//                        parked into the ring (previous chunk's survivors stay
//                        reusable); a fence event orders the copy stream
//                        behind the compute stream so no parked expert is
//                        overwritten while a prior-chunk GEMM still reads it.
//   stream_start(fd)   : starts SAGER_PF_READERS loader threads. Each loader:
//                        take next pair (skipping pairs whose layer the
//                        session already passed) -> take arena slot (free, or
//                        steal a slot whose consumed event completed) -> take
//                        a pinned slab -> pread(pool) -> cudaMemcpyAsync H2D
//                        on the pump's copy stream -> record ready event ->
//                        publish (or free the slot again if the pair went
//                        stale during the copy).
//   stream_progress(L) : called by the session after layer L-1's dispatch:
//                        frees every slot owned by a pair of an earlier
//                        layer. This is what keeps zero-count pairs (loaded
//                        after their layer passed) and stale survivors from
//                        clogging the ring - the failure mode that stalled
//                        the first streamed run at 2 tok/s.
//   stream_lookup(L,e) : slot if the pair is at least enqueued (blocks up to
//                        SAGER_PF_ARM_TIMEOUT_MS, then -1 -> caller falls back
//                        to the CPU pool; also the guard against a stalled
//                        loader hanging prefill).
//   stream_wait(s,slot): cudaStreamWaitEvent on the slot's ready event.
//   stream_consume(s,slot): record the slot's consumed event on the compute
//                        stream (frees the slot for the loader).
//   stream_finish()    : join loaders; surviving slots stay LRU-resident for
//                        decode and for the next chunk's parked reuse.
//
// Zero-row pairs (an expert no token of this prompt routes to) would clog
// slots forever, so the dispatch layer consumes them early: after a layer's
// routing is known, stream_consume_drop() releases the layer's untouched
// pairs. With SAGER_PF_STREAM_MIN >= 384 the expected zero-row count is ~0
// anyway; the drop call is the belt to that braces.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
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
    // v0.2.3 prefill streamer
    uint64_t stream_pairs = 0;    // pairs completed H2D this prompt
    uint64_t stream_steals = 0;   // slots taken from consumed experts
    uint64_t stream_drop = 0;     // zero-row pairs released early
    uint64_t stream_skipped = 0;  // pairs skipped (session already past)
    uint64_t arm_timeouts = 0;    // stream_lookup gave up -> CPU pool
    uint64_t read_errors = 0;     // pread() short/failed
  };
  Stats stats() const { return stats_; }

  int slot_count() const { return static_cast<int>(slot_count_); }
  size_t per_expert_bytes() const { return per_expert_bytes_; }
  int resident_count() const;

  // ---- v0.2.3 prefill stream-all ring ------------------------------------

  // Pure host planning: the hot-tier misses in (layer asc, expert asc) order.
  // `resident` is a packed L*E bitmap of the HOT tier only (the arena side is
  // reset by begin_stream). Unit-testable without CUDA.
  struct StreamPlan {
    std::vector<int32_t> pairs;    // packed layer*n_experts+expert, stream order
    int n_layers = 0;
    int n_experts = 0;
  };
  static StreamPlan build_stream_plan(const std::vector<uint8_t>& resident,
                                      int n_layers, int n_experts);

  // Hand the arena to the ring for ONE chunk: resident slots are parked (no
  // copy needed), the plan covers only the misses. The fence event orders
  // the copy stream behind the compute stream across the chunk boundary.
  // No-op on CPU-only builds.
  void begin_stream(const StreamPlan& plan, void* compute_stream);

  // Start the loaders against a pool FILE (pread source; mmap host_base_ is
  // pageable and defeats cudaMemcpyAsync - the slabs fix that). No-op on
  // CPU-only builds and when the fd is invalid.
  void stream_start(int pool_fd);

  bool stream_active() const {
#if defined(SAGER_HAVE_CUDA)
    return stream_active_.load(std::memory_order_relaxed);
#else
    return false;
#endif
  }

  // v0.2.4d: live loader-thread count. A stream session whose loaders never
  // spawned (or already exited) can never publish a slot - dispatch must
  // treat it as non-streaming (per-pump submit path) instead of waiting on
  // lookups forever.
  int stream_loader_count() const {
#if defined(SAGER_HAVE_CUDA)
    // v0.2.5: LIVE count - loader_loop decrements on exit, so a session whose
    // walkers finished (or never spawned) reads 0 and dispatch falls back to
    // the per-pump submit path instead of waiting on lookups forever.
    return live_loaders_.load(std::memory_order_relaxed);
#else
    return 0;
#endif
  }

  // Session progress: release every slot owned by a pair of a layer below
  // `layer` (the session never revisits earlier layers within a chunk).
  void stream_progress(int layer);

  // Slot of (layer, expert) once its copy is enqueued, else -1 after a
  // bounded wait. Session thread only.
  int stream_lookup(int layer, int expert);

#if defined(SAGER_HAVE_CUDA)
  // Order the compute stream behind the slot's ready copy (after lookup).
  void stream_wait(cudaStream_t compute, int slot);
  // Freed-for-refill marker, recorded ON the compute stream after the
  // expert's last GEMM (stream-ordered, so the GEMMs really finished).
  void stream_consume(cudaStream_t compute, int slot);
#endif

  // Release a pair the router never picked (zero-row) WITHOUT touching the
  // compute stream: the slot becomes stealable, the pair unresidable.
  void stream_consume_drop(int layer, int expert);

  // Join the loaders. Slots still holding pairs stay LRU-resident (decode
  // warm start); ring bookkeeping resets to plain pump state.
  void stream_finish();

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

  // SAGER_PF_* knobs (plain ints: init() reads them on every build)
  int arm_timeout_ms_ = 2000;          // SAGER_PF_ARM_TIMEOUT_MS
  int n_readers_ = 2;                  // SAGER_PF_READERS
  int slabs_per_reader_ = 16;          // SAGER_PF_SLABS

#if defined(SAGER_HAVE_CUDA)
  void* arena_ = nullptr;
  cudaStream_t copy_stream_ = nullptr;
  std::vector<cudaEvent_t> events_;    // per-slot completion events

  // ---- stream-all ring state ---------------------------------------------
  // Loader threads mutate slot_of_/owner_/ring_* under stream_mu_; the
  // session thread reads them there too (stream_lookup). Everything else
  // stays session-thread only (C7d).
  enum RingState : uint8_t { kRingFree = 0, kRingLoading = 1, kRingLoaded = 2 };
  std::vector<uint8_t> ring_state_;    // per slot
  std::vector<uint8_t> ring_owned_;    // slot belongs to the live session
  std::vector<uint8_t> consumed_rec_;  // consume event recorded (host flag)
  std::vector<uint8_t> parked_;        // resident at chunk start (no copy)
  std::vector<cudaEvent_t> consumed_ev_;
  cudaEvent_t fence_ev_ = nullptr;     // chunk-boundary compute -> copy fence
  std::atomic<int> done_layer_{-1};    // session progress (layers < this die)
  StreamPlan plan_;
  int pool_fd_ = -1;
  std::atomic<bool> stream_active_{false};
  std::vector<std::thread> loaders_;
  std::atomic<int> live_loaders_{0};   // v0.2.5: spawned minus exited

  // Requires stream_mu_ held. Releases every slot owned by a pair whose
  // layer is already behind the session (done_layer_): consumed pairs wait
  // for their GEMM event, unconsumed ones (walk took the CPU fallback)
  // can never be read again and free up immediately. Called by the session
  // thread (stream_progress) AND by loader threads stuck in the take loop -
  // otherwise a wedged ring can only recover at the next layer boundary,
  // which never comes while the walk itself is blocked in a lookup.
  void reclaim_locked();

  void loader_loop(int tid);

  struct Slab {
    void* ptr = nullptr;
    cudaEvent_t ev = nullptr;
    bool used = false;                 // ev ever recorded (query safety)
  };
  std::vector<std::vector<Slab>> loader_slabs_;   // per loader thread
  size_t stream_next_ = 0;             // next plan index (atomic via mu)

  std::mutex stream_mu_;               // guards ring_* + slot_of_ + owner_ +
                                       // stream_next_ while a session lives
  std::condition_variable stream_cv_;  // publish signal for stream_lookup
#endif
};

}  // namespace sager
