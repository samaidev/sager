// q4_equiv_test.cpp - standalone equivalence harness for the CPU expert FFN
// kernel (suspect #2 / candidate root cause 13).
//
//   1. builds random fp32 gate/up/down weights + input x,
//   2. quantizes them with the engine's own Q4_K/Q8_K quantizers,
//   3. runs sager::cpu::expert_ffn_q4 (AVX2) AND gemv_q4_k_q8_scalar,
//   4. dumps every blob for the numpy reference:
//        x_f32[hidden], gate_q4 bytes, up_q4 bytes, out_q4 bytes,
//        y_avx2[hidden], y_scalar[hidden], g_raw[n_ff], u_raw[n_ff], h[n_ff]
//
//   g++ -O2 -mavx2 -DSAGER_HAVE_AVX2 -I include q4_equiv_test.cpp \
//       src/kernels/cpu/q4_avx2.cpp src/kernels/q4_k.cpp -o q4_equiv_test
#include "sager/kernels/cpu_ops.hpp"
#include "sager/kernels/q4_k.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sager;

static void dump(const char* path, const void* p, size_t n) {
  FILE* f = fopen(path, "wb");
  fwrite(p, 1, n, f);
  fclose(f);
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : "/tmp/q4eq";
  const int H = 2048, FFX = 512;
  std::mt19937 rng(20260718);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> x(H), wg(FFX * H), wu(FFX * H), wo(H * FFX);
  for (auto& v : x) v = dist(rng) * 0.5f;
  for (auto& v : wg) v = dist(rng) * 0.05f;
  for (auto& v : wu) v = dist(rng) * 0.05f;
  for (auto& v : wo) v = dist(rng) * 0.05f;

  auto qwg = q4k::quantize_row(wg.data(), FFX * H);
  auto qwu = q4k::quantize_row(wu.data(), FFX * H);
  auto qwo = q4k::quantize_row(wo.data(), H * FFX);

  std::vector<float> y_avx2(H), y_scalar(H), scratch(2 * FFX);
  cpu::expert_ffn_q4(qwg.data(), qwu.data(), qwo.data(), x.data(),
                     y_avx2.data(), scratch.data(), H, FFX);

  // scalar cross-check of every stage
  std::vector<q4k::BlockQ8K> xq(H / q4k::kQK_K), hq(FFX / q4k::kQK_K);
  cpu::quantize_row_q8_k_avx2(x.data(), xq.data(), H);
  std::vector<float> g(FFX), u(FFX), h(FFX);
  cpu::gemv_q4_k_q8_scalar(qwg.data(), xq.data(), g.data(), FFX, H);
  cpu::gemv_q4_k_q8_scalar(qwu.data(), xq.data(), u.data(), FFX, H);
  for (int i = 0; i < FFX; ++i)
    h[i] = g[i] / (1.f + expf(-g[i])) * u[i];
  cpu::quantize_row_q8_k_avx2(h.data(), hq.data(), FFX);
  cpu::gemv_q4_k_q8_scalar(qwo.data(), hq.data(), y_scalar.data(), H, FFX);

  double dg = 0, du = 0, dy = 0, ys = 0;
  for (int i = 0; i < H; ++i) dy += fabs(y_avx2[i] - y_scalar[i]);
  for (int i = 0; i < FFX; ++i) { dg += fabs(g[i]); du += fabs(u[i]); }
  for (int i = 0; i < H; ++i) ys += fabs(y_avx2[i]);
  printf("sum|g|=%.4f sum|u|=%.4f sum|y_avx2|=%.4f sum|y_sc-y_avx2|=%.6f\n",
         dg, du, ys, dy);

  dump(std::string(std::string(dir) + "/x_f32.bin").c_str(), x.data(),
       H * 4);
  dump(std::string(std::string(dir) + "/wg_q4.bin").c_str(), qwg.data(),
       qwg.size() * 144);
  dump(std::string(std::string(dir) + "/wu_q4.bin").c_str(), qwu.data(),
       qwu.size() * 144);
  dump(std::string(std::string(dir) + "/wo_q4.bin").c_str(), qwo.data(),
       qwo.size() * 144);
  dump(std::string(std::string(dir) + "/xq_q8.bin").c_str(), xq.data(),
       xq.size() * 276);
  dump(std::string(std::string(dir) + "/hq_q8.bin").c_str(), hq.data(),
       hq.size() * 276);
  dump(std::string(std::string(dir) + "/g_f32.bin").c_str(), g.data(),
       FFX * 4);
  dump(std::string(std::string(dir) + "/u_f32.bin").c_str(), u.data(),
       FFX * 4);
  dump(std::string(std::string(dir) + "/h_f32.bin").c_str(), h.data(),
       FFX * 4);
  dump(std::string(std::string(dir) + "/y_avx2.bin").c_str(), y_avx2.data(),
       H * 4);
  dump(std::string(std::string(dir) + "/y_scalar.bin").c_str(),
       y_scalar.data(), H * 4);
  printf("dumped to %s\n", dir);
  return 0;
}
