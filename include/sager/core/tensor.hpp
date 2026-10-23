// tensor.hpp - RAII device/host buffers, pinned memory and stream helpers.
//
// Two rules keep the CUDA-graph story simple:
//  1. Decode graphs bake addresses, so every per-step buffer is allocated
//     ONCE at session start and never moved (fixed-address contract).
//  2. Host staging buffers used by the CPU expert pool are page-locked
//     (pinned) and reused, so H2D copies are async-cheap and the pool can
//     write results without extra allocations (same-address contract from
//     core/expert_source.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sager/log.hpp"

#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
#define SAGER_CUDA_CHECK(call)                                              \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      SLOG_E("cuda", "%s failed: %s", #call, cudaGetErrorString(err_));     \
    }                                                                       \
  } while (0)
#else
#define SAGER_CUDA_CHECK(call) do { (void)sizeof(call); } while (0)
#endif

namespace sager {

// Non-CUDA-safe allocation helpers (work in CPU-only builds too).
void* host_alloc(size_t bytes, bool pinned);
void  host_free(void* p, bool pinned);

#if defined(SAGER_HAVE_CUDA)

struct DeviceDeleter { void operator()(void* p) const { if (p) cudaFree(p); } };

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) { alloc(count); }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept { swap(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept { reset(); swap(o); return *this; }

  bool alloc(size_t count) {
    reset();
    if (count == 0) return true;
    if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) {
      ptr_ = nullptr; count_ = 0; return false;
    }
    count_ = count;
    return true;
  }

  void reset() { if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; count_ = 0; } }
  void swap(DeviceBuffer& o) noexcept {
    std::swap(ptr_, o.ptr_); std::swap(count_, o.count_);
  }

  T* get() const { return ptr_; }
  size_t count() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

class CudaStream {
 public:
  CudaStream()  { cudaStreamCreate(&s_); }
  ~CudaStream() { if (s_) cudaStreamDestroy(s_); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  cudaStream_t get() const { return s_; }
  void sync() { cudaStreamSynchronize(s_); }
 private:
  cudaStream_t s_ = nullptr;
};

// Pinned host buffer for async H2D/D2H and CPU-pool scratch.
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t count) { alloc(count); }
  ~PinnedBuffer() { reset(); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  bool alloc(size_t count) {
    reset();
    if (count == 0) return true;
    ptr_ = static_cast<T*>(host_alloc(count * sizeof(T), true));
    count_ = ptr_ ? count : 0;
    return ptr_ != nullptr;
  }
  void reset() { if (ptr_) { host_free(ptr_, true); ptr_ = nullptr; count_ = 0; } }

  T* get() const { return ptr_; }
  size_t count() const { return count_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

#else  // CPU-only build: keep host buffers, make device buffers compile out.

template <typename T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t) {}
  bool alloc(size_t) { return false; }
  void reset() {}
  T* get() const { return nullptr; }
  size_t count() const { return 0; }
  explicit operator bool() const { return false; }
};

struct CudaStream { void* get() const { return nullptr; } void sync() {} };

template <typename T> using PinnedBuffer = std::vector<T>;

#endif  // SAGER_HAVE_CUDA

}  // namespace sager
