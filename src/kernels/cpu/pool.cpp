// pool.cpp - the CPU expert pool: worker threads that compute VRAM misses
// in parallel with the GPU (contract C1-C6 in core/expert_source.hpp).
//
// Design: single producer (session thread), N workers, fixed per-worker
// scratch (no allocation on the hot path), atomic completion counting.
// Results land directly in the session's pinned output slots so the combine
// step is a plain async H2D copy.
#include "sager/core/expert_source.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "sager/kernels/cpu_ops.hpp"
#include "sager/log.hpp"

namespace sager {

Status validate_pool_geometry(const PoolGeometry& want, const PoolGeometry& got) {
  if (want.hidden != got.hidden || want.n_ff != got.n_ff ||
      want.n_expert != got.n_expert || want.block_bytes != got.block_bytes) {
    return Status::kGeometryMismatch;  // C4: refuse, never misindex
  }
  return Status::kOk;
}

namespace {

class PoolImpl final : public CpuPool {
 public:
  ~PoolImpl() override { stop(); }

  Status start(int workers, const PoolGeometry& geo) override {
    bind_geometry(geo);
    workers_ = workers > 0 ? workers
                           : static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2));
    running_ = true;
    // Per-worker scratch: gate+up halves = 2 x n_ff floats (reused, C1).
    for (int i = 0; i < workers_; ++i) {
      scratch_.emplace_back(static_cast<size_t>(2 * geo_.n_ff));
    }
    threads_.reserve(static_cast<size_t>(workers_));
    for (int i = 0; i < workers_; ++i) {
      threads_.emplace_back([this, i] { worker_loop(i); });
    }
    SLOG_I("pool", "started %d worker threads (avx2=%s)", workers_,
           cpu::have_avx2() ? "yes" : "no");
    return Status::kOk;
  }

  void stop() override {
    {
      // Flip the flag under mu_: a worker sitting between its predicate
      // check and the futex sleep would otherwise lose the notify below and
      // sleep forever while stop() joins it (observed as a load_cmd hang on
      // a 2-vCPU box where that window is wide).
      std::lock_guard<std::mutex> lk(mu_);
      running_ = false;
    }
    cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
    threads_.clear();
  }

  void bind_geometry(const PoolGeometry& geo) override {
    geo_ = geo;
    bound_ = true;
  }

  int submit(std::vector<MissJob> jobs) override {
    if (jobs.empty()) return 0;
    if (!bound_) return 0;
    // Double-checked geometry: every job must match the bound geometry (C4).
    for (const auto& j : jobs) {
      PoolGeometry want{geo_.hidden, geo_.n_ff, geo_.n_expert, q4k::kBlockBytes};
      PoolGeometry got{j.hidden, j.n_ff, 0, q4k::kBlockBytes};
      if (validate_pool_geometry(want, got) != Status::kOk) {
        SLOG_E("pool", "geometry mismatch on submit - refusing job (C4)");
        return 0;
      }
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& j : jobs) ring_.emplace_back(std::move(j));
    outstanding_.fetch_add(static_cast<int>(jobs.size()));
    cv_.notify_all();
    return static_cast<int>(jobs.size());
  }

  void wait_done(DoneFn on_done) override {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return outstanding_.load() == 0; });
    if (on_done && !completed_.empty()) {
      on_done(completed_);
      completed_.clear();
    }
  }

  int outstanding() const override { return outstanding_.load(); }

 private:
  void worker_loop(int id) {
    float* scratch = scratch_[static_cast<size_t>(id)].data();
    while (true) {
      MissJob* job = nullptr;
      {
        std::unique_lock<std::mutex> lk(mu_);
        // Bounded wait: even if a notify were lost, the worker re-checks the
        // predicate every 200 ms and exits, so shutdown cannot deadlock.
        cv_.wait_for(lk, std::chrono::milliseconds(200),
                     [&] { return !running_.load() || next_idx_ < ring_.size(); });
        if (!running_.load()) return;
        if (next_idx_ >= ring_.size()) continue;
        job = &ring_[next_idx_++];
      }
      run_job(*job, scratch);
      {
        std::lock_guard<std::mutex> lk(mu_);
        completed_.push_back(*job);
        if (outstanding_.fetch_sub(1) == 1) done_cv_.notify_one();
      }
    }
  }

  // C2: no router weights here - pure expert FFN. C3: y is a fixed pinned slot.
  static void run_job(const MissJob& j, float* scratch) {
    cpu::expert_ffn_q4(j.w_gate, j.w_up, j.w_out, j.x, j.y, scratch, j.hidden, j.n_ff);
  }

  PoolGeometry geo_{};
  bool bound_ = false;
  int workers_ = 0;
  std::vector<std::thread> threads_;
  std::vector<std::vector<float>> scratch_;   // per-worker, fixed (C1)
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  std::vector<MissJob> ring_;
  size_t next_idx_ = 0;
  std::atomic<int> outstanding_{0};
  std::vector<MissJob> completed_;
  std::atomic<bool> running_{false};
};

}  // namespace

std::unique_ptr<CpuPool> make_cpu_pool() {
  return std::make_unique<PoolImpl>();
}

}  // namespace sager
