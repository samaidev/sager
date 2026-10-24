// cpu_ops.hpp - AVX2 kernels for the CPU expert pool (Strata-style).
//
// When a routed expert misses the VRAM hot cache, the host computes
// y = W_expert @ x directly in system RAM (Q4_K AVX2 dot products) in
// parallel with the GPU working on other layers. Only Q4_K x Q8_K + SiLU-mul
// is needed on the CPU path, so the pool stays small and cache-friendly.
#pragma once

#include <cstdint>
#include <cstddef>

#include "sager/kernels/q4_k.hpp"

namespace sager::cpu {

// Returns true if the binary was built with AVX2 (SAGER_HAVE_AVX2).
bool have_avx2();

// y[k] += w_q4[n x k] @ x_q8[k] for ONE expert FFN projection row-group.
// Simple, allocation-free; the pool caller slices n into tiles.
void gemv_q4_k_q8_avx2(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x,
                       float* y, int n_rows, int k);

// Quantize one activation row to BlockQ8K (n multiple of 256).
void quantize_row_q8_k_avx2(const float* x, q4k::BlockQ8K* out, int n);

// Full expert FFN on CPU: h = silu(W_g @ x) * (W_u @ x); y = W_o @ h
//   W_g/W_u: [n_ff x hidden] Q4_K, W_o: [hidden x n_ff] Q4_K
//   x: hidden floats, y: hidden floats (overwritten, not accumulated)
// scratch: caller-provided, >= n_ff floats.
void expert_ffn_q4(const q4k::BlockQ4K* w_gate, const q4k::BlockQ4K* w_up,
                   const q4k::BlockQ4K* w_out, const float* x, float* y,
                   float* scratch, int hidden, int n_ff);

// Non-AVX2 scalar fallbacks (correct, slow; used on old hosts and in tests).
void gemv_q4_k_q8_scalar(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x,
                         float* y, int n_rows, int k);

}  // namespace sager::cpu
