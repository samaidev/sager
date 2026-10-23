// test_q4_host.cpp - host-side correctness tests for the Q4_K stack.
//
// These run anywhere (no GPU, no model): CI builds them to guard the
// quantization contract that the CUDA and AVX2 kernels must match bit-for-bit
// on the integer path. Exit code 0 = all green.
//
//   1. fp16 <-> fp32 conversion (round trip, extremes)
//   2. K-quant scale pack/unpack round trip (get/set_scale_min_k4)
//   3. quantize -> dequant round trip within Q4_K_M-class error budget
//   4. scalar dot vs naive reference; AVX2 dot vs scalar (bit-exact int path)
//   5. Q8_K activation quant exactness (amax/127 invariant)
//   6. compare_rows stats sanity (cosine ~1 for self, ~0 for noise)
//   7. pool geometry validation (C4)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "sager/core/expert_source.hpp"
#include "sager/kernels/cpu_ops.hpp"
#include "sager/kernels/q4_k.hpp"

using namespace sager;          // NOLINT
using namespace sager::q4k;     // NOLINT

static int g_failures = 0;

#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

static void test_fp16() {
  for (float f : {0.0f, 1.0f, -1.0f, 0.5f, -1234.5f, 65504.0f, 1e-6f}) {
    const float back = half_to_float(float_to_half(f));
    // fp16 subnormals (|f| < 6e-5) quantize coarsely: allow an absolute term.
    CHECK(std::fabs(back - f) <= std::fabs(f) * 1e-3f + 2e-7f, "fp16 roundtrip");
  }
  CHECK(float_to_half(1e10f) == 0x7C00u, "fp16 overflow -> inf");
}

static void test_scale_pack() {
  uint8_t q[12] = {0};   // 16 x 6-bit values packed into 12 bytes (BlockQ4K layout)
  std::mt19937 rng(7);
  for (int iter = 0; iter < 1000; ++iter) {
    uint8_t d[8], m[8];
    for (int j = 0; j < 8; ++j) {
      d[j] = static_cast<uint8_t>(rng() & 63);
      m[j] = static_cast<uint8_t>(rng() & 63);
    }
    std::memset(q, 0, 12);
    for (int j = 0; j < 8; ++j) set_scale_min_k4(j, q, d[j], m[j]);
    for (int j = 0; j < 8; ++j) {
      uint8_t gd = 0, gm = 0;
      get_scale_min_k4(j, q, &gd, &gm);
      CHECK(gd == d[j], "scale d roundtrip");
      CHECK(gm == m[j], "scale m roundtrip");
    }
  }
}

static void test_quant_dequant() {
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int n = 4096;
  std::vector<float> x(n);
  for (auto& v : x) v = nd(rng);

  const std::vector<BlockQ4K> q = quantize_row(x.data(), n);
  std::vector<float> y(n);
  dequant_row(q.data(), y.data(), n);

  double se = 0, se_ref = 0;
  for (int i = 0; i < n; ++i) {
    se += (y[i] - x[i]) * (y[i] - x[i]);
    se_ref += x[i] * x[i];
  }
  const double rmse = std::sqrt(se / n);
  const double rel = rmse / std::sqrt(se_ref / n);
  // 4-bit rate-distortion floor for IID gaussian is ~7-8% rel RMSE (the
  // grid search lands within a few % of it). Real LLM weight distributions
  // are far more peaked and land closer to 2-4%; tools/validate_quant.py
  // measures per-tensor quality on actual weights.
  CHECK(rel < 0.10, "q4 quant relative rmse near floor");
}

static void test_dot() {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int n = 2048;
  std::vector<float> x(n);
  for (auto& v : x) v = nd(rng);

  const std::vector<BlockQ4K> w = quantize_row(x.data(), n);
  std::vector<BlockQ8K> xq(n / kQK_K);
  quantize_row_q8_K(x.data(), xq.data(), n);

  // Exact float reference against the DEQUANTIZED weights.
  std::vector<float> wd(n);
  dequant_row(w.data(), wd.data(), n);
  float ref = 0.f;
  for (int i = 0; i < n; ++i) {
    const int blk = i / kQK_K, off = i % kQK_K;
    ref += wd[i] * static_cast<float>(xq[blk].qs[off]) * xq[blk].d;
  }

  const float got_scalar = vec_dot_ref(w.data(), xq.data(), n);
  CHECK(std::fabs(got_scalar - ref) <= 1e-4f * std::fabs(ref) + 1e-2f,
        "scalar dot matches dequant reference");

  if (cpu::have_avx2()) {
    std::vector<float> y_avx(n / 256, 0.f), y_sc(n / 256, 0.f);
    // Slice into 256-length rows to exercise the row API.
    for (int r = 0; r < 8; ++r) {
      cpu::gemv_q4_k_q8_avx2(w.data() + r, xq.data(), &y_avx[r], 1, 256);
      cpu::gemv_q4_k_q8_scalar(w.data() + r, xq.data(), &y_sc[r], 1, 256);
    }
    for (int r = 0; r < 8; ++r) {
      CHECK(std::fabs(y_avx[r] - y_sc[r]) <= 1e-4f * std::fabs(y_sc[r]) + 1e-5f,
            "avx2 dot matches scalar");
    }
  }
}

static void test_q8_quant() {
  std::mt19937 rng(99);
  std::normal_distribution<float> nd(0.f, 3.f);
  const int n = 1024;
  std::vector<float> x(n);
  for (auto& v : x) v = nd(rng);
  std::vector<BlockQ8K> b(n / kQK_K);
  quantize_row_q8_K(x.data(), b.data(), n);
  for (int bi = 0; bi < n / kQK_K; ++bi) {
    float amax = 1e-30f;
    for (int i = 0; i < kQK_K; ++i) {
      amax = std::max(amax, std::fabs(x[bi * kQK_K + i]));
    }
    CHECK(std::fabs(b[bi].d - amax / 127.f) < 1e-6f, "q8 scale = amax/127");
    for (int i = 0; i < kQK_K; ++i) {
      CHECK(std::fabs(b[bi].qs[i]) <= 127, "q8 clamp");
    }
  }
}

static void test_compare_rows() {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> a(512), b(512), noise(512);
  for (auto& v : a) v = nd(rng);
  b = a;
  for (auto& v : noise) v = nd(rng);
  const QuantStats s_same = compare_rows(a.data(), b.data(), 512);
  CHECK(s_same.cosine > 0.999, "cosine(self)=1");
  const QuantStats s_noise = compare_rows(a.data(), noise.data(), 512);
  CHECK(s_noise.cosine < 0.5, "cosine(noise)~0");
}

static void test_pool_geometry() {
  const PoolGeometry want{2048, 768, 128, kBlockBytes};
  PoolGeometry got = want;
  CHECK(validate_pool_geometry(want, got) == Status::kOk, "matching geometry ok");
  got.n_ff = 4096;
  CHECK(validate_pool_geometry(want, got) == Status::kGeometryMismatch,
        "wrong n_ff refused (C4)");
}

int main() {
  test_fp16();
  test_scale_pack();
  test_quant_dequant();
  test_dot();
  test_q8_quant();
  test_compare_rows();
  test_pool_geometry();

  if (g_failures == 0) {
    std::printf("q4 host tests: ALL PASSED\n");
    return 0;
  }
  std::printf("q4 host tests: %d FAILURES\n", g_failures);
  return 1;
}
