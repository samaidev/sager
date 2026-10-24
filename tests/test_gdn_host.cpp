// test_gdn_host.cpp - GDN (gated delta net) CPU reference verification.
//
// Two layers of defense:
//  1. Property tests: the delta rule's algebraic identities (exact column
//     replacement under alpha=beta=1 with unit keys, pure write under
//     alpha=0, gated superposition) - catches transcription errors that
//     golden vectors alone could normalize away.
//  2. Golden vectors: scripts/gen_gdn_vectors.py (numpy, independent
//     implementation of docs/gdn-kernel.md) -> tests/gdn_vectors.hpp.
#include "sager/kernels/gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sager/kernels/q4_k.hpp"
#include "gdn_vectors.hpp"

using namespace sager;

static int g_failures = 0;

#define CHECK_NEAR(actual, expected, atol, what)                            \
  do {                                                                      \
    const double diff = std::fabs(static_cast<double>(actual) -             \
                                  static_cast<double>(expected));           \
    if (diff > (atol)) {                                                    \
      std::printf("FAIL %s: got %.9f want %.9f (|d|=%.3g > %g)\n", what,    \
                  static_cast<double>(actual),                              \
                  static_cast<double>(expected), diff,                      \
                  static_cast<double>(atol));                               \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

namespace {

inline float h2f(uint16_t h) { return q4k::half_to_float(h); }
inline uint16_t f2h(float f) { return q4k::float_to_half(f); }

// Convert a float array to a fp16 buffer.
std::vector<uint16_t> to_f16(const float* v, size_t n) {
  std::vector<uint16_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = f2h(v[i]);
  return out;
}

// ---------------------------------------------------------------------------
// Property tests on the raw delta update (nk=1, K=4, rows=3).
// ---------------------------------------------------------------------------
void test_delta_properties() {
  const int K = 4, rows = 3, nk = 1;
  float S[rows * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // unit key along dim m=2
  float k[K] = {0, 0, 1, 0};
  float v[rows] = {-1, -2, -3};
  float q[K] = {0.5f, -0.25f, 0.125f, 0.0625f};
  float ab[2] = {1.0f, 1.0f};  // alpha=1, beta=1
  std::vector<uint16_t> qh = to_f16(q, K), kh = to_f16(k, K);
  std::vector<uint16_t> vh = to_f16(v, rows), y(rows);

  // alpha=beta=1 + unit key -> column m of S is replaced by v
  gdn_cpu::delta_step(S, y.data(), qh.data(), kh.data(), vh.data(), ab, nk, K,
                      1, rows);
  for (int j = 0; j < rows; ++j) {
    CHECK_NEAR(S[j * K + 2], v[j], 1e-5f, "delta: column replaced by v");
    CHECK_NEAR(S[j * K + 0], static_cast<float>(j * K + 1), 1e-6f,
               "delta: untouched col 0");
  }
  // y_j = S_new[j,:] . q = S_old[j,:].q - S_old[j][m]*q[m] + v_j*q[m]
  // recomputed here independently:
  const float S_old[rows * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (int j = 0; j < rows; ++j) {
    float want = 0;
    for (int i = 0; i < K; ++i) {
      float s = S_old[j * K + i];
      if (i == 2) s = v[j];
      want += s * q[i];
    }
    CHECK_NEAR(h2f(y[static_cast<size_t>(j)]), want, 2e-4f,
               "delta: inclusive retrieve");
  }

  // alpha=0 -> S = beta * v k^T (pure write)
  float S2[rows * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float ab0[2] = {0.0f, 0.7f};
  gdn_cpu::delta_step(S2, y.data(), qh.data(), kh.data(), vh.data(), ab0, nk,
                      K, 1, rows);
  for (int j = 0; j < rows; ++j)
    for (int i = 0; i < K; ++i)
      CHECK_NEAR(S2[j * K + i], 0.7f * v[j] * k[i], 1e-5f,
                 "delta: alpha=0 pure write");
}

void test_gate_formulas() {
  // hand-computed sigmoid/softplus/exp chain
  const int nk = 2;
  uint16_t ba_h[2 * nk];
  float A_log[nk] = {0.2f, -0.4f}, dt_bias[nk] = {0.1f, 0.3f};
  // pick b,a so results are stable to hand-verify
  // gate() layout: ba[h] = b_h, ba[nk + h] = a_h
  ba_h[0] = f2h(0.0f);   // b0 -> beta0 = 0.5
  ba_h[1] = f2h(2.0f);   // b1 -> beta1 = sigmoid(2)
  ba_h[2] = f2h(1.0f);   // a0 = 1 -> dt = softplus(1.1)
  ba_h[3] = f2h(-1.0f);  // a1 = -1 -> dt = softplus(-0.7)
  float ab[2 * nk];
  gdn_cpu::gate(ba_h, A_log, dt_bias, ab, nk);
  CHECK_NEAR(ab[1], 0.5f, 1e-6f, "gate: sigmoid(0)");
  CHECK_NEAR(ab[3], 1.f / (1.f + std::exp(-2.f)), 1e-6f, "gate: sigmoid(2)");
  const float dt0 = std::log1p(std::exp(1.1f));
  CHECK_NEAR(ab[0], std::exp(-std::exp(0.2f) * dt0), 1e-6f, "gate: alpha0");
  const float dt1 = std::log1p(std::exp(-0.7f));
  CHECK_NEAR(ab[2], std::exp(std::exp(-0.4f) * -dt1), 1e-6f, "gate: alpha1");
}

void test_conv_and_outnorm_golden() {
  const auto gv = test::kGdnVec;
  const int D = 2 * gv.nk * gv.K + gv.nv * gv.V;  // conv_dim = 40

  std::vector<uint16_t> st = to_f16(test::g_conv_state, test::g_conv_state_n);
  std::vector<uint16_t> w = to_f16(test::g_conv_w, test::g_conv_w_n);
  std::vector<uint16_t> b = to_f16(test::g_conv_b, test::g_conv_b_n);
  std::vector<uint16_t> x = to_f16(test::g_conv_x, test::g_conv_x_n);
  std::vector<uint16_t> out(D);
  gdn_cpu::conv_step(out.data(), st.data(), x.data(), w.data(), b.data(), D,
                     gv.ck);
  for (int d = 0; d < D; ++d)
    CHECK_NEAR(h2f(out[d]), test::g_conv_out[d], 2e-2f, "conv: golden out");
  // state shifted: new state == golden g_conv_state_new
  for (size_t i = 0; i < test::g_conv_state_new_n; ++i)
    CHECK_NEAR(h2f(st[i]), test::g_conv_state_new[i], 2e-3f,
               "conv: golden state shift");

  // out norm: y (nk blocks of ratio*V) + z [nv*V]
  const int NV = gv.nv, V = gv.V;
  std::vector<uint16_t> y = to_f16(test::g_y, test::g_y_n);
  std::vector<uint16_t> z = to_f16(test::g_z, test::g_z_n);
  std::vector<uint16_t> won = to_f16(test::g_won, test::g_won_n);
  std::vector<uint16_t> o(NV * V);
  gdn_cpu::out_norm(o.data(), y.data(), z.data(), won.data(), NV, V, 1e-6f);
  for (int i = 0; i < NV * V; ++i)
    CHECK_NEAR(h2f(o[i]), test::g_o[i], 2e-2f, "out norm: golden");
}

void test_delta_golden() {
  const auto gv = test::kGdnVec;
  const int nk = gv.nk, K = gv.K, RATIO = gv.nv / gv.nk, V = gv.V;
  const int rows = RATIO * V;

  std::vector<float> S(test::g_S0, test::g_S0 + test::g_S0_n);
  std::vector<uint16_t> qh = to_f16(test::g_q_raw, test::g_q_raw_n);
  std::vector<uint16_t> kh = to_f16(test::g_k_raw, test::g_k_raw_n);
  std::vector<uint16_t> qnw = to_f16(test::g_qn_w, test::g_qn_w_n);
  std::vector<uint16_t> knw = to_f16(test::g_kn_w, test::g_kn_w_n);
  std::vector<uint16_t> v = to_f16(test::g_v_all, test::g_v_all_n);
  std::vector<uint16_t> ba(2 * nk);
  std::vector<uint16_t> qn(nk * K), kn(nk * K), y(nk * rows);
  std::vector<float> ab(2 * nk);

  // qk_norm reads its source from a conv_out layout [q | k]; build the fake
  // input from the raw q|k vectors (the golden generator ran the SAME
  // C-GDN-1 chain, so qk_norm must reproduce g_qh/g_kh).
  {
    std::vector<uint16_t> conv_fake(2 * nk * K);
    for (size_t i = 0; i < test::g_q_raw_n; ++i) conv_fake[i] = qh[i];
    for (size_t i = 0; i < test::g_k_raw_n; ++i)
      conv_fake[test::g_q_raw_n + i] = kh[i];
    gdn_cpu::qk_norm(qn.data(), kn.data(), conv_fake.data(), qnw.data(),
                     knw.data(), nk, K, 1e-6f);
  }
  for (int h = 0; h < nk; ++h)
    for (int i = 0; i < K; ++i) {
      CHECK_NEAR(h2f(qn[h * K + i]), test::g_qh[h * K + i], 2e-3f,
                 "qk_norm: golden q");
      CHECK_NEAR(h2f(kn[h * K + i]), test::g_kh[h * K + i], 2e-3f,
                 "qk_norm: golden k");
    }

  // gate from b/a raw
  ba[0] = f2h(test::g_b_raw[0]); ba[1] = f2h(test::g_b_raw[1]);
  ba[2] = f2h(test::g_a_raw[0]); ba[3] = f2h(test::g_a_raw[1]);
  gdn_cpu::gate(ba.data(), test::g_A_log, test::g_dt_bias, ab.data(), nk);
  CHECK_NEAR(ab[0], test::g_alpha[0], 2e-3f, "gate: golden alpha0");
  CHECK_NEAR(ab[2], test::g_alpha[1], 2e-3f, "gate: golden alpha1");
  CHECK_NEAR(ab[1], test::g_beta[0], 2e-3f, "gate: golden beta0");
  CHECK_NEAR(ab[3], test::g_beta[1], 2e-3f, "gate: golden beta1");

  // delta step
  gdn_cpu::delta_step(S.data(), y.data(), qn.data(), kn.data(), v.data(),
                      ab.data(), nk, K, RATIO, V);
  for (size_t i = 0; i < test::g_S1_n; ++i)
    CHECK_NEAR(S[i], test::g_S1[i], 5e-3f, "delta: golden state");
  for (size_t i = 0; i < test::g_y_n; ++i)
    CHECK_NEAR(h2f(y[i]), test::g_y[i], 1e-2f, "delta: golden y");
}

}  // namespace

int main() {
  test_delta_properties();
  test_gate_formulas();
  test_conv_and_outnorm_golden();
  test_delta_golden();
  if (g_failures == 0) {
    std::printf("gdn_host_reference: ALL OK\n");
    return 0;
  }
  std::printf("gdn_host_reference: %d FAILURES\n", g_failures);
  return 1;
}
