// q4_avx2.cpp - AVX2 Q4_K dot products for the CPU expert pool + scalar
// reference paths (used by tests on any machine).
//
// The CPU pool serves VRAM misses, so this path is already PCIe/memory-bound;
// we optimize for EXACT per-sub-block scale handling (fp32 carries per 32
// weights, identical rounding to the CUDA path) and verifiable lane order
// rather than last-few-percent throughput.
#include "sager/kernels/cpu_ops.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sager::cpu {

bool have_avx2() {
#if defined(SAGER_HAVE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
  return true;
#else
  return false;
#endif
}

namespace {

inline void unpack_scales(const q4k::BlockQ4K& b, float* sd, float* sm) {
  const float d = q4k::half_to_float(b.d);
  const float dmin = q4k::half_to_float(b.dmin);
  for (int j = 0; j < 8; ++j) {
    uint8_t d6, m6;
    q4k::get_scale_min_k4(j, b.scales, &d6, &m6);
    sd[j] = d * static_cast<float>(d6);
    sm[j] = dmin * static_cast<float>(m6);
  }
}

inline float sub_sum8(const q4k::BlockQ8K& xb, int s) {
  return static_cast<float>(xb.bsums[s]);
}

}  // namespace

void gemv_q4_k_q8_scalar(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x,
                         float* y, int n_rows, int k) {
  const int nb = k / q4k::kQK_K;
  for (int r = 0; r < n_rows; ++r) {
    const q4k::BlockQ4K* row = w + static_cast<size_t>(r) * nb;
    float acc = 0.f;
    for (int blk = 0; blk < nb; ++blk) {
      const q4k::BlockQ4K& b = row[blk];
      const q4k::BlockQ8K& xb = x[blk];
      float sd[8], sm[8];
      unpack_scales(b, sd, sm);
      float sum = 0.f;
      for (int s = 0; s < 8; ++s) {
        float psum = 0.f;
        for (int j = 0; j < q4k::kQK; ++j) {
          const int idx = s * q4k::kQK + j;
          const int nib = (idx & 1) ? (b.qs[idx >> 1] >> 4) : (b.qs[idx >> 1] & 0xF);
          psum += static_cast<float>(nib) * static_cast<float>(xb.qs[idx]);
        }
        sum += xb.d * (sd[s] * psum - sm[s] * sub_sum8(xb, s));
      }
      acc += sum;
    }
    y[r] = acc;
  }
}

#if defined(SAGER_HAVE_AVX2) && (defined(__x86_64__) || defined(_M_X64))

void quantize_row_q8_k_avx2(const float* x, q4k::BlockQ8K* out, int n) {
  const int nb = n / q4k::kQK_K;
  for (int b = 0; b < nb; ++b) {
    const float* xb = x + static_cast<size_t>(b) * q4k::kQK_K;
    q4k::BlockQ8K& ob = out[b];
    float amax = 1e-30f;
    for (int i = 0; i < q4k::kQK_K; ++i) amax = std::fmaxf(amax, std::fabsf(xb[i]));
    const float d = amax / 127.f;
    ob.d = d;
    const float id = d > 0 ? 1.f / d : 0.f;
    for (int s = 0; s < 8; ++s) ob.bsums[s] = 0;
    for (int i = 0; i < q4k::kQK_K; ++i) {
      const int q = static_cast<int>(std::nearbyintf(xb[i] * id));
      ob.qs[i] = static_cast<int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
      ob.bsums[i >> 5] = static_cast<int16_t>(ob.bsums[i >> 5] + ob.qs[i]);
    }
  }
}

// One sub-block (32 weights) dot with exact scale handling.
// qs bytes (16) hold the nibbles: low nibble = even index, high = odd index.
// We restore natural lane order with two byte interleaves, then use
// maddubs (u8*s8 -> s16 pairs) exactly like the on-device kernel.
static inline float dot_subblock_avx2(const uint8_t* qs16,
                                      const int8_t* xq32,
                                      float sd, float sm, float xd,
                                      float xsum) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs16));
  const __m128i lowMask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(raw, lowMask);                       // w0,w2,...
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lowMask);    // w1,w3,...
  // natural order: [w0..w15] and [w16..w31]
  const __m128i w_lo = _mm_unpacklo_epi8(lo, hi);
  const __m128i w_hi = _mm_unpackhi_epi8(lo, hi);
  const __m256i w = _mm256_set_m128i(w_hi, w_lo);                       // 32 u8 lanes
  const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq32));
  const __m256i p = _mm256_maddubs_epi16(w, xv);                        // 16 x s16 pairs

  // widen s16 pairs to s32, sum to one i32
  __m128i pl = _mm256_castsi256_si128(p);
  __m128i ph = _mm256_extracti128_si256(p, 1);
  __m128i s1 = _mm_add_epi32(_mm_cvtepi16_epi32(pl), _mm_cvtepi16_epi32(_mm_unpackhi_epi64(pl, pl)));
  __m128i s2 = _mm_add_epi32(_mm_cvtepi16_epi32(ph), _mm_cvtepi16_epi32(_mm_unpackhi_epi64(ph, ph)));
  __m128i st = _mm_add_epi32(s1, s2);
  st = _mm_add_epi32(st, _mm_shuffle_epi32(st, 0x0E));
  st = _mm_add_epi32(st, _mm_shuffle_epi32(st, 0x01));
  const int32_t ip = _mm_cvtsi128_si32(st);

  return xd * (sd * static_cast<float>(ip) - sm * xsum);
}

void gemv_q4_k_q8_avx2(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x,
                       float* y, int n_rows, int k) {
  const int nb = k / q4k::kQK_K;
  for (int r = 0; r < n_rows; ++r) {
    const q4k::BlockQ4K* row = w + static_cast<size_t>(r) * nb;
    float acc = 0.f;
    for (int blk = 0; blk < nb; ++blk) {
      const q4k::BlockQ4K& b = row[blk];
      const q4k::BlockQ8K& xb = x[blk];
      float sd[8], sm[8];
      unpack_scales(b, sd, sm);
      float sum = 0.f;
      for (int s = 0; s < 8; ++s) {
        sum += dot_subblock_avx2(b.qs + s * 16, xb.qs + s * q4k::kQK,
                                 sd[s], sm[s], xb.d,
                                 static_cast<float>(xb.bsums[s]));
      }
      acc += sum;
    }
    y[r] = acc;
  }
}

#else  // no AVX2 available: forward to scalar

void quantize_row_q8_k_avx2(const float* x, q4k::BlockQ8K* out, int n) {
  const int nb = n / q4k::kQK_K;
  for (int b = 0; b < nb; ++b) {
    const float* xb = x + static_cast<size_t>(b) * q4k::kQK_K;
    q4k::BlockQ8K& ob = out[b];
    float amax = 1e-30f;
    for (int i = 0; i < q4k::kQK_K; ++i) amax = std::fmaxf(amax, std::fabsf(xb[i]));
    const float d = amax / 127.f;
    ob.d = d;
    const float id = d > 0 ? 1.f / d : 0.f;
    for (int s = 0; s < 8; ++s) ob.bsums[s] = 0;
    for (int i = 0; i < q4k::kQK_K; ++i) {
      const int q = static_cast<int>(std::nearbyintf(xb[i] * id));
      ob.qs[i] = static_cast<int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
      ob.bsums[i >> 5] = static_cast<int16_t>(ob.bsums[i >> 5] + ob.qs[i]);
    }
  }
}

void gemv_q4_k_q8_avx2(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x,
                       float* y, int n_rows, int k) {
  gemv_q4_k_q8_scalar(w, x, y, n_rows, k);
}

#endif  // AVX2

void expert_ffn_q4(const q4k::BlockQ4K* w_gate, const q4k::BlockQ4K* w_up,
                   const q4k::BlockQ4K* w_out, const float* x, float* y,
                   float* scratch, int hidden, int n_ff) {
  // Quantize activations once for both gate/up projections.
  std::vector<q4k::BlockQ8K> xq(hidden / q4k::kQK_K);
  quantize_row_q8_k_avx2(x, xq.data(), hidden);

  float* g = scratch;             // [n_ff]
  float* u = scratch + n_ff;      // [n_ff]
  gemv_q4_k_q8_avx2(w_gate, xq.data(), g, n_ff, hidden);
  gemv_q4_k_q8_avx2(w_up,   xq.data(), u, n_ff, hidden);

  // h = silu(g) * u  (reuse the gate half as scratch)
  for (int i = 0; i < n_ff; ++i) {
    const float gv = g[i];
    g[i] = gv / (1.f + std::exp(-gv)) * u[i];
  }

  std::vector<q4k::BlockQ8K> hq(n_ff / q4k::kQK_K);
  quantize_row_q8_k_avx2(g, hq.data(), n_ff);
  gemv_q4_k_q8_avx2(w_out, hq.data(), y, hidden, n_ff);
}

}  // namespace sager::cpu
