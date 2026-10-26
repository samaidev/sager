// tc_gemm.cu - v0.9 prefill dense-path: Q4_K -> fp16 dequant + cuBLAS
// tensor-core GEMM (SAGER_PF_TC).
//
// The v0.3 mtok GEMV family (and the v0.8 MMQ variant of it) runs the dense
// projections at ~0.8-1.1 TFLOPS on the T4 (~3% of the dp4a peak), and the
// measured prefill wall is ~2/3 dense (SAGER_PROF: fw dense=21316 ms vs
// moe=12148 ms over 6656 chunk tokens). This path dequantizes a projection
// weight ONCE per chunk into a fp16 scratch (bandwidth-bound, ~0.31 ms for
// the 12288x2048 GDN qkvz after the v8 linear-store rewrite) and calls
// cublasGemmEx with CUBLAS_COMPUTE_32F (fp16 in, fp32 accumulate -> HMMA
// tensor cores on sm_75).
//
// Layout contract (cuBLAS is column-major, everything here is row-major):
//   C_row[n x N] = X_row[n x K] * W_row[N x K]^T
//   C_row^T = W * X^T  ->  GemmEx(OP_T, OP_N, m=N, n=n_tok, k=K,
//                                 A=scratch lda=K, B=x ldb=K, C=y ldc=N)
//   because memory-wise C_row[n x N] == C_col[N x n], X_row[n x K] == X_col[K x n],
//   and W_row[N x K] == W_col[K x N] (so op(A)=A^T yields W).
//
// Numerics: NOT bit-identical to the dp4a path (weights round to fp16,
// activations stay fp16 instead of being requantized to Q8_K, and the
// accumulation order changes). Decode (n==1) never takes this path; the
// v0.7/v0.8 dp4a GEMV path stays available via SAGER_PF_TC=0.
#include "sager/kernels/cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace sager::cuda {
namespace {

__device__ __forceinline__ void unpack_scales6_tc(const uint8_t* sc, int s,
                                                  float d, float dmin,
                                                  float* sd, float* sm) {
  uint8_t d6, m6;
  if (s < 4) {
    d6 = static_cast<uint8_t>(sc[s] & 63);
    m6 = static_cast<uint8_t>(sc[s + 4] & 63);
  } else {
    d6 = static_cast<uint8_t>((sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4));
    m6 = static_cast<uint8_t>((sc[s + 4] >> 4) | ((sc[s] >> 6) << 4));
  }
  *sd = d * static_cast<float>(d6);
  *sm = dmin * static_cast<float>(m6);
}

// One thread dequantizes one 16 B output slab (8 weights) with a perfectly
// LINEAR store stream: thread L writes uint4[L]. Probe9/probe10 on sm_75
// showed the write pattern is everything here: 512 B-per-thread chunked
// stores run at 18-50 GB/s while linear 16 B-per-thread stores hit 166 GB/s
// (reads were never the issue: 218 GB/s). The 144 B Q4_K header and the 4
// nibble bytes are re-read per thread but stay hot in L1 (32 threads share
// each header). k must be a power of two (2048/4096 in this engine) so the
// index math is all shifts; verified bit-identical to the sub-block kernel
// (probe10: 0 / 25165824 mismatches) and 3.4x faster (1.06 -> 0.31 ms for
// the 12288x2048 qkvz projection).
__global__ void dequant_q4_k_f16_kernel(const q4k::BlockQ4K* __restrict__ w,
                                        __half* __restrict__ out,
                                        long long n_u4, int sl2) {
  const long long L = static_cast<long long>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (L >= n_u4) return;
  const int k = 1 << sl2;
  const long long row = L >> (sl2 - 3);
  const int u4r = static_cast<int>(L & ((1 << (sl2 - 3)) - 1));
  const long long bi = row * (k >> 8) + (u4r >> 5);
  const int sb = (u4r >> 2) & 7;         // 32-weight sub-block within the row
  const int q = u4r & 3;                 // quarter of it: 8 weights
  const uint8_t* base = reinterpret_cast<const uint8_t*>(w + bi);
  const uint4 hdr = *reinterpret_cast<const uint4*>(base);
  const float d = __half2float(
      __ushort_as_half(static_cast<unsigned short>(hdr.x & 0xFFFFu)));
  const float dmin = __half2float(
      __ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
  uint8_t sc[q4k::kQK_K / 32 + 4];  // 12 scales bytes
  #pragma unroll
  for (int b = 0; b < 12; ++b) {
    const unsigned word = b < 4 ? hdr.y : (b < 8 ? hdr.z : hdr.w);
    sc[b] = static_cast<uint8_t>((word >> (8 * (b & 3))) & 0xFFu);
  }
  float sd, sm;
  unpack_scales6_tc(sc, sb, d, dmin, &sd, &sm);
  const int qi = sb * (q4k::kQK / 2) + q * 4;
  const unsigned word =
      *reinterpret_cast<const unsigned*>(base + (q4k::kQK_K / 16) + qi);
  const unsigned n0 = (word >> 0) & 0xFFu, n1 = (word >> 8) & 0xFFu;
  const unsigned n2 = (word >> 16) & 0xFFu, n3 = (word >> 24) & 0xFFu;
  // byte j -> halves {2j, 2j+1} = (low nibble, high nibble), matching the
  // reference kernel's buf[2*j] / buf[2*j+1] layout.
  union H2U { __half2 h; unsigned u; };
  auto pk_half = [&](unsigned byte) -> unsigned {
    H2U c;
    c.h = __halves2half2(
        __float2half(sd * static_cast<float>(byte & 0xF) - sm),
        __float2half(sd * static_cast<float>(byte >> 4) - sm));
    return c.u;
  };
  const unsigned u0 = pk_half(n0);
  const unsigned u1 = pk_half(n1);
  const unsigned u2 = pk_half(n2);
  const unsigned u3 = pk_half(n3);
  uint4* o4 = reinterpret_cast<uint4*>(out) + L;
  *o4 = make_uint4(u0, u1, u2, u3);
}

cublasHandle_t g_tc_handle = nullptr;

}  // namespace

void* tc_handle() {
  if (!g_tc_handle) {
    if (cublasCreate(&g_tc_handle) != CUBLAS_STATUS_SUCCESS) return nullptr;
    cublasSetMathMode(g_tc_handle, CUBLAS_DEFAULT_MATH);
  }
  return g_tc_handle;
}

// Largest useful fp16 dequant scratch: max over the dense projections of
// (rows x k) elements.
size_t tc_scratch_elems(int max_rows, int max_k) {
  return static_cast<size_t>(max_rows > 0 ? max_rows : 1) *
         static_cast<size_t>(max_k > 0 ? max_k : 1);
}

// out[t * n_rows + row] = sum_k x[t * k + kk] * w[row * k + kk]
// (fp16 I/O, fp32 accumulate). Returns false if cuBLAS is unavailable, k is
// not a power of two, or the weight pointer is not 16 B aligned; callers
// fall back to the dp4a GEMV path.
bool tc_gemm_q4_k_f16(const q4k::BlockQ4K* w, const uint16_t* x, uint16_t* y,
                      uint16_t* scratch, int n_rows, int k, int n_tokens,
                      cudaStream_t stream) {
  if (!g_tc_handle && !tc_handle()) return false;
  if ((k & (k - 1)) != 0 || k < 256) return false;      // need log2 shifts
  if ((reinterpret_cast<uintptr_t>(w) & 15) != 0) return false;
  if (cublasSetStream(g_tc_handle, stream) != CUBLAS_STATUS_SUCCESS)
    return false;
  int sl2 = 0;
  while ((1 << sl2) < k) ++sl2;
  const long long n_u4 =
      static_cast<long long>(n_rows) * k / (q4k::kQK / 4);
  const int blocks = static_cast<int>((n_u4 + 255) / 256);
  dequant_q4_k_f16_kernel<<<blocks, 256, 0, stream>>>(
      w, reinterpret_cast<__half*>(scratch), n_u4, sl2);
  const float one = 1.0f, zero = 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      g_tc_handle, CUBLAS_OP_T, CUBLAS_OP_N,
      n_rows, n_tokens, k,
      &one, scratch, CUDA_R_16F, k,
      x, CUDA_R_16F, k,
      &zero, y, CUDA_R_16F, n_rows,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return st == CUBLAS_STATUS_SUCCESS;
}

}  // namespace sager::cuda
