// expert_source.hpp - the verified contract between the session loop, the
// router and the CPU expert pool.
//
// This header documents the interface CLAUSE BY CLAUSE (a Strata practice we
// copy deliberately - see docs/architecture.md). Violating any clause is a
// bug class the engine refuses at runtime rather than misindexing.
//
// Contract (v0.1):
//
//  C1. The activation buffer handed to the pool MUST be the same pinned
//      staging region every layer (PinnedBuffer, allocated once at session
//      start). The pool reads it via raw pointer; no per-call allocation.
//
//  C2. The CPU pool computes y = W_e @ x for the missing experts and MUST
//      NOT apply router weights. moe_combine_f16() (device side) is the only
//      place the weighted sum happens, so a CPU/GPU weight mismatch is
//      structurally impossible.
//
//  C3. The pool MUST write results into the fixed pinned output slot
//      assigned by the session (one contiguous [max_miss x hidden] region),
//      from where an async H2D copy lands them into the combine buffer.
//
//  C4. Geometry is validated once at bind time: hidden, n_ff_exp and the
//      Q4_K block layout of every expert the pool will see. A wrong expert
//      geometry MUST be refused (Status::kGeometryMismatch), never misindexed.
//
//  C5. The pool MUST stay lock-free on the submit path: a single-producer
//      (session thread) MPMC-free design - jobs go into a per-worker
//      work-stealing-free ring, completions are counted atomically.
//
//  C6. Misses for layer N are submitted when the GPU STARTS layer N-1, so
//      CPU work overlaps device work (doorbell protocol).
#pragma once

#include <memory>
#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "sager/common.hpp"
#include "sager/kernels/q4_k.hpp"

namespace sager {

// One missing expert's job. x/y point into the fixed pinned regions (C1/C3).
struct MissJob {
  int      layer;
  int      expert;
  const q4k::BlockQ4K* w_gate;  // [n_ff x hidden]
  const q4k::BlockQ4K* w_up;    // [n_ff x hidden]
  const q4k::BlockQ4K* w_out;   // [hidden x n_ff]
  const float* x;               // [hidden]      (pinned, C1)
  float*       y;               // [hidden]      (pinned slot, C3)
  int          hidden;
  int          n_ff;
};

// Validation performed once at bind time (C4).
struct PoolGeometry {
  int hidden;
  int n_ff;
  int n_expert;
  int block_bytes;   // must equal q4k::kBlockBytes
};
Status validate_pool_geometry(const PoolGeometry& want, const PoolGeometry& got);

// The pool interface. The session owns one CpuPool; workers are pinned to
// alternating physical cores to avoid SMT contention with the H2D copy thread.
class CpuPool {
 public:
  using DoneFn = std::function<void(const std::vector<MissJob>&)>;

  virtual ~CpuPool() = default;

  virtual Status start(int workers, const PoolGeometry& geo) = 0;
  virtual void   stop() = 0;

  // Submit a batch of misses for ONE layer. Non-blocking (C5). Returns the
  // number of jobs actually enqueued (rest were pre-resident - caller bug).
  virtual int submit(std::vector<MissJob> jobs) = 0;

  // Wait until all outstanding jobs of the current generation complete and
  // invoke the done-callback exactly once. Called by the session at combine
  // time for the layer the misses belonged to.
  virtual void wait_done(DoneFn on_done) = 0;

  virtual int  outstanding() const = 0;
  virtual void bind_geometry(const PoolGeometry& geo) = 0;
};

// Factory (AVX2 build uses src/kernels/cpu/pool.cpp implementation).
std::unique_ptr<CpuPool> make_cpu_pool();

}  // namespace sager
