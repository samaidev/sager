// q4_k.cpp - host reference implementations for the Q4_K format:
// dequant, quantize (M-flavor scale refinement), reference dot, and stats.
// These are the ground truth for tests/test_q4_host.cpp and the numerical
// contract shared by the CUDA and AVX2 kernels.
#include "sager/kernels/q4_k.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sager::q4k {

void dequant_row(const BlockQ4K* blocks, float* out, int n) {
  const int nb = n / kQK_K;
  for (int b = 0; b < nb; ++b) {
    const BlockQ4K& bl = blocks[b];
    const float d = half_to_float(bl.d);
    const float dmin = half_to_float(bl.dmin);
    float* ob = out + static_cast<size_t>(b) * kQK_K;
    // ggml Q4_K nibble order (v0.12 fix): the 128 qs bytes split into four
    // 32-byte groups; group g's LOW nibbles are sub-block 2g (weights
    // 64g..64g+31) and its HIGH nibbles are sub-block 2g+1 (weights
    // 64g+32..64g+63). The previous adjacent-pair reading was Q4_0 semantics
    // and silently permuted every community/llama.cpp-converted Q4_K file.
    for (int g = 0; g < kNumSubBlocks / 2; ++g) {
      const uint8_t* q = bl.qs + g * 32;
      uint8_t d0, m0, d1, m1;
      get_scale_min_k4(2 * g, bl.scales, &d0, &m0);
      get_scale_min_k4(2 * g + 1, bl.scales, &d1, &m1);
      const float sd0 = d * static_cast<float>(d0);
      const float sm0 = dmin * static_cast<float>(m0);
      const float sd1 = d * static_cast<float>(d1);
      const float sm1 = dmin * static_cast<float>(m1);
      float* o0 = ob + g * 64;
      float* o1 = o0 + kQK;
      for (int l = 0; l < 32; ++l) {
        o0[l] = sd0 * static_cast<float>(q[l] & 0xF) - sm0;
        o1[l] = sd1 * static_cast<float>(q[l] >> 4) - sm1;
      }
    }
  }
}

void quantize_row_q8_K(const float* x, BlockQ8K* out, int n) {
  const int nb = n / kQK_K;
  for (int b = 0; b < nb; ++b) {
    const float* xb = x + static_cast<size_t>(b) * kQK_K;
    BlockQ8K& ob = out[b];
    float amax = 1e-30f;
    for (int i = 0; i < kQK_K; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
    const float d = amax / 127.f;
    ob.d = d;
    const float id = d > 0 ? 1.f / d : 0.f;
    for (int s = 0; s < kNumSubBlocks; ++s) ob.bsums[s] = 0;
    for (int i = 0; i < kQK_K; ++i) {
      const int q = static_cast<int>(std::nearbyint(xb[i] * id));
      ob.qs[i] = static_cast<int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
      ob.bsums[i >> 5] = static_cast<int16_t>(ob.bsums[i >> 5] + ob.qs[i]);
    }
  }
}

// M-flavor quantization: per-sub-block (scale, min) grid search minimizing
// RMSE of the reconstruction w ~ d*q - m with q CLAMPED to [0,15] (negative
// codes would wrap the nibble pack - a bug class the unit test guards).
// Coarse-to-fine grid keeps the converter fast: a 35B model has ~1.3e4
// matrices and the naive 63x64x32 search would take hours.
std::vector<BlockQ4K> quantize_row(const float* x, int n) {
  const int nb = n / kQK_K;
  std::vector<BlockQ4K> out(static_cast<size_t>(nb));
  uint8_t codes[kNumSubBlocks][kQK];

  auto eval_err = [](const float* sb, float dd, float mm) {
    const float id = dd > 0.f ? 1.f / dd : 0.f;
    float err = 0.f;
    for (int j = 0; j < kQK; ++j) {
      int q = static_cast<int>(std::nearbyint((sb[j] + mm) * id));
      q = std::max(0, std::min(15, q));
      const float w = dd * static_cast<float>(q) - mm;
      const float e = sb[j] - w;
      err += e * e;
    }
    return err;
  };

  for (int b = 0; b < nb; ++b) {
    const float* xb = x + static_cast<size_t>(b) * kQK_K;
    BlockQ4K& bl = out[static_cast<size_t>(b)];
    std::memset(&bl, 0, sizeof(bl));
    std::memset(codes, 0, sizeof(codes));

    // Super scale: max sub-block (max-min) amplitude keeps 6-bit scales in range.
    float max_abs = 1e-30f;
    for (int s = 0; s < kNumSubBlocks; ++s) {
      float mn = xb[s * kQK], mx = xb[s * kQK];
      for (int j = 1; j < kQK; ++j) {
        mn = std::fmin(mn, xb[s * kQK + j]);
        mx = std::fmax(mx, xb[s * kQK + j]);
      }
      max_abs = std::fmax(max_abs, mx - mn);
    }
    const float d = max_abs / 63.f;
    const float dmin = max_abs / 63.f;  // symmetric start; refined per sub-block
    bl.d = float_to_half(d);
    bl.dmin = float_to_half(dmin);

    for (int s = 0; s < kNumSubBlocks; ++s) {
      const float* sb = xb + s * kQK;
      float mn = sb[0], mx = sb[0];
      for (int j = 1; j < kQK; ++j) {
        mn = std::fmin(mn, sb[j]);
        mx = std::fmax(mx, sb[j]);
      }
      if (mx == mn) {
        set_scale_min_k4(s, bl.scales, 1, 0);
        continue;
      }

      // Anchor the search at the direct (near-optimal) choice:
      //   dd ~ range/15, mm ~ -mn   (w = dd*q - mm covers [mn, mx] exactly)
      // then refine on a coarse/fine grid around it. For iid gaussian the
      // 4-bit rate-distortion floor is ~7% rel RMSE - the anchor already
      // sits within a few percent of the exhaustive 63x64 minimum.
      const float d_ideal = (mx - mn) / 15.f;
      const int is_ideal =
          std::max(1, std::min(63, static_cast<int>(std::nearbyint(d_ideal / d))));
      const int im_ideal =
          std::max(0, std::min(63, static_cast<int>(std::nearbyint(-mn / dmin))));

      float best = eval_err(sb, d * static_cast<float>(is_ideal),
                            dmin * static_cast<float>(im_ideal));
      int b_is = is_ideal, b_im = im_ideal;
      // Coarse pass: step 4 around the anchor.
      for (int is = std::max(1, is_ideal - 16); is <= std::min(63, is_ideal + 16); is += 4) {
        for (int im = std::max(0, im_ideal - 16); im <= std::min(63, im_ideal + 16); im += 4) {
          const float err = eval_err(sb, d * static_cast<float>(is),
                                     dmin * static_cast<float>(im));
          if (err < best) { best = err; b_is = is; b_im = im; }
        }
      }
      // Fine pass: +-2 around the coarse winner at step 1.
      for (int is = std::max(1, b_is - 2); is <= std::min(63, b_is + 2); ++is) {
        for (int im = std::max(0, b_im - 2); im <= std::min(63, b_im + 2); ++im) {
          const float err = eval_err(sb, d * static_cast<float>(is),
                                     dmin * static_cast<float>(im));
          if (err < best) { best = err; b_is = is; b_im = im; }
        }
      }

      // Final encode with the winning (d_s, m_s): keep the 32 codes; the
      // ggml-order byte pack happens after all sub-blocks are quantized.
      const float bd = d * static_cast<float>(b_is);
      const float bm = dmin * static_cast<float>(b_im);
      const float id = bd > 0.f ? 1.f / bd : 0.f;
      for (int j = 0; j < kQK; ++j) {
        codes[s][j] = static_cast<uint8_t>(std::max(0, std::min(15,
            static_cast<int>(std::nearbyint((sb[j] + bm) * id)))));
      }
      set_scale_min_k4(s, bl.scales, static_cast<uint8_t>(b_is),
                       static_cast<uint8_t>(b_im));
    }

    // ggml Q4_K byte pack: group g's 32 bytes carry sub-block 2g in the LOW
    // nibbles and sub-block 2g+1 in the HIGH nibbles (one weight per byte).
    for (int g = 0; g < kNumSubBlocks / 2; ++g) {
      uint8_t* q = bl.qs + g * 32;
      for (int l = 0; l < 32; ++l) {
        q[l] = static_cast<uint8_t>((codes[2 * g][l] & 0xF) |
                                    ((codes[2 * g + 1][l] & 0xF) << 4));
      }
    }
  }
  return out;
}

float vec_dot_ref(const BlockQ4K* w, const BlockQ8K* x, int n) {
  const int nb = n / kQK_K;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const BlockQ4K& bl = w[b];
    const BlockQ8K& xb = x[b];
    const float d = half_to_float(bl.d);
    const float dmin = half_to_float(bl.dmin);
    // ggml Q4_K order: group g = one 32-byte chunk, low nibbles -> sub-block
    // 2g, high nibbles -> sub-block 2g+1 (mirrors dequant_row).
    for (int g = 0; g < kNumSubBlocks / 2; ++g) {
      const uint8_t* q = bl.qs + g * 32;
      uint8_t d0, m0, d1, m1;
      get_scale_min_k4(2 * g, bl.scales, &d0, &m0);
      get_scale_min_k4(2 * g + 1, bl.scales, &d1, &m1);
      const float sd0 = d * static_cast<float>(d0);
      const float sm0 = dmin * static_cast<float>(m0);
      const float sd1 = d * static_cast<float>(d1);
      const float sm1 = dmin * static_cast<float>(m1);
      int p0 = 0, p1 = 0;
      for (int l = 0; l < 32; ++l) {
        p0 += (q[l] & 0xF) * xb.qs[g * 64 + l];
        p1 += (q[l] >> 4) * xb.qs[g * 64 + 32 + l];
      }
      acc += xb.d * (sd0 * static_cast<float>(p0) -
                     sm0 * static_cast<float>(xb.bsums[2 * g]));
      acc += xb.d * (sd1 * static_cast<float>(p1) -
                     sm1 * static_cast<float>(xb.bsums[2 * g + 1]));
    }
  }
  return acc;
}

QuantStats compare_rows(const float* ref, const float* got, int n) {
  QuantStats s{};
  double se = 0, mx = 0;
  double dot = 0, nr = 0, ng = 0;
  for (int i = 0; i < n; ++i) {
    const double e = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    se += e * e;
    mx = std::fmax(mx, std::fabs(e));
    dot += static_cast<double>(ref[i]) * static_cast<double>(got[i]);
    nr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    ng += static_cast<double>(got[i]) * static_cast<double>(got[i]);
  }
  s.rmse = static_cast<float>(std::sqrt(se / n));
  s.max_abs_err = static_cast<float>(mx);
  s.cosine = static_cast<float>(dot / (std::sqrt(nr) * std::sqrt(ng) + 1e-30));
  return s;
}

}  // namespace sager::q4k
