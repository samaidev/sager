// test_cuda_q4.cu - CUDA-vs-host numeric validation of the Q4_K matmul stack.
//
// The engine bugs found in v0.12 debugging (GDN conv raw-bit taps, Q6 head
// missing the Q8 activation scale) were both invisible to the host-only
// ctest suite. This test runs the REAL device kernels against the host
// reference on seeded random data:
//   1. quantize_q8_k (CUDA) vs host quantize_row_q8_K - bit-exact
//   2. gemv_q4_k_q8 (dp4a) vs host dequant+dot
//   3. mmq_q4_k_q8_f16out (batched prefill) vs host ref
//   4. gemv_q4_k_q8_f16out_mtok vs host ref
//   5. tc_gemm_q4_k_f16 (tensor core, fp16 path) vs host ref
// Exit code 0 = all green.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "sager/kernels/q4_k.hpp"
#include "sager/kernels/cuda_ops.hpp"

using namespace sager;          // NOLINT
using namespace sager::q4k;     // NOLINT
using namespace sager::cuda;    // NOLINT

static int g_failures = 0;

#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

static float h2f(uint16_t h) { return q4k::half_to_float(h); }
static uint16_t f2h(float f) { return q4k::float_to_half(f); }

namespace {

struct Dev {
  cudaStream_t s = nullptr;
  Dev() { cudaStreamCreate(&s); }
  ~Dev() { if (s) cudaStreamDestroy(s); }
};

template <typename T>
T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return p;
}

double cos_vec(const std::vector<float>& a, const std::vector<float>& b) {
  double da = 0, db = 0, ab = 0;
  for (size_t i = 0; i < a.size(); ++i) { da += double(a[i]) * a[i];
    db += double(b[i]) * b[i]; ab += double(a[i]) * b[i]; }
  return ab / (std::sqrt(da * db) + 1e-30);
}

double rel_rmse(const std::vector<float>& a, const std::vector<float>& b) {
  double se = 0, se2 = 0;
  for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i];
    se += d * d; se2 += b[i] * b[i]; }
  return std::sqrt(se / a.size()) / (std::sqrt(se2 / a.size()) + 1e-30);
}

}  // namespace

int main() {
  Dev dev;
  cudaStream_t s = dev.s;
  const int K = 2048, ROWS = 512, TOK = 5, NB = K / kQK_K;

  // ---- seeded random data --------------------------------------------------
  std::mt19937 rng(1234);
  std::normal_distribution<float> nx(0.f, 0.6f);    // activations
  std::normal_distribution<float> nw(0.f, 0.03f);   // weights

  std::vector<float> x(TOK * K);
  for (auto& v : x) v = nx(rng);
  std::vector<BlockQ4K> wblocks(ROWS * NB);
  for (int r = 0; r < ROWS; ++r) {
    std::vector<float> row(K);
    for (int i = 0; i < K; ++i) row[i] = nw(rng);
    auto bl = quantize_row(row.data(), K);
    std::memcpy(&wblocks[size_t(r) * NB], bl.data(), NB * kBlockBytes);
  }

  // host reference: dequant W, fp32 dot with x[t]
  std::vector<float> wref(ROWS * K);
  for (int r = 0; r < ROWS; ++r)
    dequant_row(&wblocks[size_t(r) * NB], &wref[size_t(r) * K], K);
  std::vector<float> yref(TOK * ROWS);
  for (int t = 0; t < TOK; ++t)
    for (int r = 0; r < ROWS; ++r) {
      double acc = 0;
      for (int i = 0; i < K; ++i)
        acc += double(wref[size_t(r) * K + i]) * double(x[size_t(t) * K + i]);
      yref[size_t(t) * ROWS + r] = float(acc);
    }

  // ---- 1. quantize_q8_k CUDA vs host ---------------------------------------
  std::vector<uint16_t> x_f16(TOK * K);
  for (size_t i = 0; i < x.size(); ++i) x_f16[i] = f2h(x[i]);
  // v0.12 fix: the host reference must consume the SAME fp16-rounded values
  // the kernel sees. Comparing against the raw fp32 tensor makes every fp16
  // amax/code rounding difference look like a kernel bug (40/40 mismatch).
  std::vector<float> x_h(TOK * K);
  for (size_t i = 0; i < x.size(); ++i) x_h[i] = h2f(x_f16[i]);
  uint16_t* d_x16 = upload(x_f16);
  BlockQ8K* d_xq = nullptr;
  cudaMalloc(&d_xq, size_t(TOK) * NB * sizeof(BlockQ8K));
  for (int t = 0; t < TOK; ++t)
    quantize_q8_k(d_x16 + size_t(t) * K, d_xq + size_t(t) * NB, K, s);
  std::vector<BlockQ8K> xq_gpu(size_t(TOK) * NB);
  cudaMemcpy(xq_gpu.data(), d_xq, xq_gpu.size() * sizeof(BlockQ8K),
             cudaMemcpyDeviceToHost);
  int q8_mismatch = 0;
  for (int t = 0; t < TOK; ++t) {
    std::vector<BlockQ8K> xq_host(NB);
    quantize_row_q8_K(&x_h[size_t(t) * K], xq_host.data(), K);
    for (int b = 0; b < NB; ++b) {
      const BlockQ8K& a = xq_gpu[size_t(t) * NB + b];
      const BlockQ8K& e = xq_host[b];
      if (memcmp(&a, &e, sizeof(BlockQ8K)) != 0) ++q8_mismatch;
    }
  }
  CHECK(q8_mismatch == 0, "quantize_q8_k CUDA != host");
  std::printf("quantize_q8_k: %d/%d blocks mismatch\n", q8_mismatch,
              TOK * NB);

  // ---- 2. gemv_q4_k_q8 (dp4a, fp32 out, single token) ----------------------
  {
    BlockQ4K* d_w = upload(wblocks);
    float* d_y = nullptr;
    cudaMalloc(&d_y, ROWS * sizeof(float));
    gemv_q4_k_q8(d_w, d_xq, d_y, ROWS, K, s);
    std::vector<float> y(ROWS);
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> ref(y.size());
    for (int r = 0; r < ROWS; ++r) ref[r] = yref[r];  // token 0
    const double c = cos_vec(y, ref), rr = rel_rmse(y, ref);
    std::printf("gemv_q4_k_q8:      cos=%.6f relRMSE=%.4f\n", c, rr);
    CHECK(c > 0.999 && rr < 0.02, "dp4a gemv vs host ref");
    cudaFree(d_w); cudaFree(d_y);
  }

  // ---- 3. mmq_q4_k_q8_f16out (batched) --------------------------------------
  {
    BlockQ4K* d_w = upload(wblocks);
    uint16_t* d_y = nullptr;
    cudaMalloc(&d_y, size_t(TOK) * ROWS * 2);
    mmq_q4_k_q8_f16out(d_w, d_xq, d_y, ROWS, K, TOK, s);
    std::vector<uint16_t> yh(size_t(TOK) * ROWS);
    cudaMemcpy(yh.data(), d_y, yh.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<float> y(yh.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = h2f(yh[i]);
    const double c = cos_vec(y, yref), rr = rel_rmse(y, yref);
    std::printf("mmq_q4_k_q8_f16out: cos=%.6f relRMSE=%.4f\n", c, rr);
    CHECK(c > 0.998 && rr < 0.03, "mmq batched vs host ref");
    cudaFree(d_w); cudaFree(d_y);
  }

  // ---- 4. gemv_q4_k_q8_f16out_mtok ------------------------------------------
  {
    BlockQ4K* d_w = upload(wblocks);
    uint16_t* d_y = nullptr;
    cudaMalloc(&d_y, size_t(TOK) * ROWS * 2);
    gemv_q4_k_q8_f16out_mtok(d_w, d_xq, d_y, ROWS, K, TOK, s);
    std::vector<uint16_t> yh(size_t(TOK) * ROWS);
    cudaMemcpy(yh.data(), d_y, yh.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<float> y(yh.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = h2f(yh[i]);
    const double c = cos_vec(y, yref), rr = rel_rmse(y, yref);
    std::printf("gemv_mtok_f16out:   cos=%.6f relRMSE=%.4f\n", c, rr);
    CHECK(c > 0.998 && rr < 0.03, "mtok gemv vs host ref");
    cudaFree(d_w); cudaFree(d_y);
  }

  // ---- 5. tc_gemm_q4_k_f16 (tensor core) ------------------------------------
  {
    BlockQ4K* d_w = upload(wblocks);
    uint16_t* d_y = nullptr;
    cudaMalloc(&d_y, size_t(TOK) * ROWS * 2);
    const size_t scratch_elems = tc_scratch_elems(ROWS, K);
    uint16_t* d_scratch = nullptr;
    cudaMalloc(&d_scratch, scratch_elems * 2);
    const bool ok = tc_gemm_q4_k_f16(d_w, d_x16, d_y, d_scratch, ROWS, K, TOK, s);
    CHECK(ok, "tc_gemm returned false");
    std::vector<uint16_t> yh(size_t(TOK) * ROWS);
    cudaMemcpy(yh.data(), d_y, yh.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<float> y(yh.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = h2f(yh[i]);
    const double c = cos_vec(y, yref), rr = rel_rmse(y, yref);
    std::printf("tc_gemm_q4_k_f16:   cos=%.6f relRMSE=%.4f\n", c, rr);
    CHECK(c > 0.99 && rr < 0.10, "tc gemm vs host ref");
    cudaFree(d_w); cudaFree(d_y); cudaFree(d_scratch);
  }

  cudaFree(d_x16); cudaFree(d_xq);
  if (g_failures == 0) std::printf("ALL CUDA Q4 TESTS GREEN\n");
  return g_failures == 0 ? 0 : 1;
}
