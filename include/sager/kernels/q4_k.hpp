// q4_k.hpp - SAGER's 4-bit super-block quantization format and host helpers.
//
// The container layout is byte-compatible with the widely used Q4_K
// super-block ("K-quant") scheme, so engine builds can consume existing
// community GGUF Q4_K_M weights of Sage-A1.5 AND produce identical files with
// tools/convert_hf_to_sager.py. The 自研 (in-house) part of SAGER's 4-bit stack
// is the set of kernels (CUDA sm_75 path in src/kernels/cuda/q4_k.cu, AVX2 CPU
// path in src/kernels/cpu/q4_avx2.cpp) plus the expert-granularity tooling
// (tools/validate_quant.py), not a gratuitously different on-disk format.
//
// Layout, one super-block = 256 weights (QK_K), 144 bytes total (4.5 bpw):
//   fp16 d            : super-block scale
//   fp16 dmin         : super-block min scale
//   12 bytes scales   : 8 sub-block 6-bit scales (d) interleaved with 8 x 6-bit
//                       mins (m), packed in the standard K-quant bit order
//                       (see scales_pack/scales_unpack below)
//   128 bytes qs      : 8 sub-blocks x 32 x 4-bit codes (low nibble first)
//
// Reconstruction: w[i] = sub_d[b] * q[i] - sub_m[b],  b = i / 32
//   sub_d[b] = d * scales[b], sub_m[b] = dmin * mins[b]
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

#include "sager/common.hpp"

namespace sager::q4k {

inline constexpr int kQK_K = 256;           // weights per super-block
inline constexpr int kQK = 32;              // weights per sub-block
inline constexpr int kNumSubBlocks = 8;     // kQK_K / kQK
inline constexpr size_t kBlockBytes = 144;  // bytes per super-block
inline constexpr float kFlOneOverMaxQ = 1.f / 15.f;  // 4-bit: 0..15

#pragma pack(push, 1)
struct BlockQ4K {
  uint16_t d;        // fp16 super scale
  uint16_t dmin;     // fp16 super min
  uint8_t  scales[12];
  uint8_t  qs[128];  // 8 x 32 nibbles
};
static_assert(sizeof(BlockQ4K) == kBlockBytes, "BlockQ4K must be exactly 144 bytes");

// Activation side (per 256 inputs): fp32 scale + 256 int8 values + 8 int16
// sub-block sums. ggml-compatible layout so vec_dot kernels can be diffed
// against reference code. NOTE: bsums must be WIDE (int16): a 32-value sum
// reaches +-4064 and silently wraps int8 - a bug the host tests guard.
struct BlockQ8K {
  float    d;
  int8_t   qs[kQK_K];
  int16_t  bsums[kNumSubBlocks];  // per sub-block sum of qs
};
#pragma pack(pop)

// fp16 <-> fp32 without <cuda_fp16.h> on the host side.
inline float half_to_float(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t man  = (h & 0x03FFu);
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) { bits = sign; }
    else {
      // value = man * 2^-24; normalize into [1024, 2048) (1.frac x 2^10)
      int e = 0;                 // shifts applied
      uint32_t m = man;
      while ((m & 0x400u) == 0) { m <<= 1; ++e; }
      bits = sign | ((113u - static_cast<uint32_t>(e)) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t float_to_half(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t man  = bits & 0x7FFFFFu;
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);           // inf
  if (exp <= 0) {                                                           // sub/zero
    if (exp < -10) return static_cast<uint16_t>(sign);
    man |= 0x800000u;
    int shift = 14 - exp;
    uint32_t m = man >> shift;
    return static_cast<uint16_t>(sign | m);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

// ---- K-quant scale packing (ggml-compatible "get_scale_min_k4" layout) ----
// The 16 6-bit values (8 sub-block scales d, 8 mins m) are packed into 12
// bytes. Canonical decode below; quantize uses the inverse.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

inline void set_scale_min_k4(int j, uint8_t* q, uint8_t d, uint8_t m) {
  if (j < 4) {
    q[j]     = static_cast<uint8_t>((q[j]     & 0xC0) | (d & 63));
    q[j + 4] = static_cast<uint8_t>((q[j + 4] & 0xC0) | (m & 63));
  } else {
    q[j + 4] = static_cast<uint8_t>((q[j + 4] & 0xF0) | (d & 0xF));
    q[j - 4] = static_cast<uint8_t>((q[j - 4] & 0x3F) | ((d >> 4) << 6));
    q[j + 4] = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((m & 0xF) << 4));
    q[j - 0] = static_cast<uint8_t>((q[j - 0] & 0x3F) | ((m >> 4) << 6));
  }
}

// ---- Reference dequant (host, scalar) - the ground truth used by tests ----
void dequant_row(const BlockQ4K* blocks, float* out, int n);

// Reference quantization of one row (n must be a multiple of kQK_K).
// Uses the "M" flavor: per-sub-block min+scale RMSE refinement (a simplified
// make_qkx2_quants), matching the quality tier of community Q4_K_M files.
std::vector<BlockQ4K> quantize_row(const float* x, int n);

// Scalar reference dot: sum(w_q4 * x_q8) * block scales.
float vec_dot_ref(const BlockQ4K* w, const BlockQ8K* x, int n);

// Activation quant (host reference).
void quantize_row_q8_K(const float* x, BlockQ8K* out, int n);

// Human-readable name for tensor-level stats in tools/validate_quant.py.
struct QuantStats { float rmse; float max_abs_err; float cosine; };
QuantStats compare_rows(const float* ref, const float* got, int n);

}  // namespace sager::q4k
