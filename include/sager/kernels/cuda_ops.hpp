// cuda_ops.hpp - launch API of the sm_75 (T4) kernels.
//
// These declarations are the ONLY surface the session loop sees; the .cu files
// are compiled into libsager_cuda and provide the implementations. All
// launchers are stream-ordered and never synchronize - the session owns the
// stream. Buffers are raw device pointers registered by the Weights/Session
// allocators (see core/tensor.hpp).
//
// Graph-capture contract: every value that CHANGES between decode steps
// (sequence position, past length) is read by kernels from fixed-address
// DEVICE parameter blocks (d_pos / d_seq), never from baked scalar launch
// arguments. That is what makes per-layer CUDA graphs replayable with
// updated state - the "per-token values arrive through fixed-address device
// buffers" trick Strata uses.
//
// Numerics: T4 (Turing, sm_75) has no bf16 units, so all activations are
// fp16. Accumulation in the Q4 GEMV and attention is fp32 (via dp4a or
// fmaf). The optional int4-tensor-core path (guarded by __CUDA_ARCH__ >= 750)
// is off by default and enabled with SAGER_T4_INT4_TC=1 (see docs/t4-guide.md).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "sager/kernels/q4_k.hpp"

namespace sager::cuda {

// One-time upload of the mrope section boundaries (device constant memory).
void set_mrope_section(const int sec[3]);

// ---- parameter blocks (fixed addresses for graph capture) ------------------
struct SeqParams {
  int pos;       // absolute position of the FIRST token of this step
  int n_tokens;  // tokens in this step (1 for decode)
};
static_assert(sizeof(SeqParams) == 8, "");

// ---- Q4_K x activations ----------------------------------------------------
// y[n] = W_q4[n x k] @ x_q8[k], one thread block per 8 output rows.
void gemv_q4_k_q8(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x_q8, float* y,
                  int n_rows, int k, cudaStream_t stream);

// fp16-output variant (activations already fp16; saves the f32->f16 convert
// for the small per-expert projections).
void gemv_q4_k_q8_f16out(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x_q8,
                         uint16_t* y, int n_rows, int k, cudaStream_t stream);

// Pointer-table batched variant for the hot-expert cache: `ptrs[r / rows_per]
// ` gives the base of the matrix serving row-group r, so ALL resident experts
// of a layer fuse into ONE launch (rows = n_hit * rows_per_mat).
// out rows: y[expert * rows_per + r].
void gemv_ptrs_q4_k_q8_f16out(const q4k::BlockQ4K* const* ptrs,
                              const q4k::BlockQ8K* x_q8, uint16_t* y,
                              int n_ptrs, int rows_per, int k,
                              cudaStream_t stream);

// Batched variants (prefill / verify window): one grid dim over tokens.
// x_q8 holds n_tokens contiguous rows of k values; y is [n_tokens x n_rows].
void gemv_q4_k_q8_f16out_batched(const q4k::BlockQ4K* w,
                                 const q4k::BlockQ8K* x_q8, uint16_t* y,
                                 int n_rows, int k, int n_tokens,
                                 cudaStream_t stream);

void gemv_q4_k_q8_batched(const q4k::BlockQ4K* w, const q4k::BlockQ8K* x_q8,
                          float* y, int n_rows, int k, int n_tokens,
                          cudaStream_t stream);

// Router logits, batched: W fp32 [n_rows x k] @ x fp16 -> y fp32.
void gemv_f32_out_f16x_batched(const float* w_f32, const uint16_t* x_f16,
                               float* y, int n_rows, int k, int n_tokens,
                               cudaStream_t stream);

// Plain fp16 GEMV, fp16 out (MTP head projections; attention stays Q4_K).
void gemv_f16_f16out(const uint16_t* w_f16, const uint16_t* x_f16, uint16_t* y,
                     int n_rows, int k, cudaStream_t stream);

// Quantize a fp16 row into BlockQ8K on device (n multiple of 256).
void quantize_q8_k(const uint16_t* x_f16, q4k::BlockQ8K* out, int n,
                   cudaStream_t stream);

// y = W_q4 @ x_f16 fused: quantizes x on the fly (small rows; used for
// router logits with fp32 output).
void gemv_f32_out_f16x(const float* w_f32, const uint16_t* x_f16, float* y,
                       int n_rows, int k, cudaStream_t stream);

// ---- elementwise / norm / rope ----------------------------------------------
void rmsnorm_f16(uint16_t* out, const uint16_t* x, const uint16_t* w, int n,
                 float eps, cudaStream_t stream);

// Batched RMSNorm (prefill / verify window): x is [n_rows x n] fp16.
void rmsnorm_batched_f16(uint16_t* out, const uint16_t* x, const uint16_t* w,
                         int n, float eps, int n_rows, cudaStream_t stream);

void add_f16(uint16_t* in_out, const uint16_t* residual, int n,
             cudaStream_t stream);

void convert_f32_to_f16(const float* src, uint16_t* dst, int n,
                        cudaStream_t stream);

// Embedding gather: out[i, :] = table[ids[i], :]
void gather_embed(uint16_t* out, const uint16_t* table, const int* ids,
                  int n_tokens, int hidden, cudaStream_t stream);

// RoPE for Sage-A1.5: partial rotary (rope_dims = head_dim * partial factor)
// + optional mrope (pos_ids = 3 x n_tokens when style==2).
void rope_f16(uint16_t* q, uint16_t* k, const int* pos_ids, int n_head,
              int n_head_kv, int head_dim, int rope_dims, float theta,
              int style /*0=neoX,1=interleaved,2=mrope*/, int n_tokens,
              cudaStream_t stream);

void silu_mul_f16(uint16_t* out, const uint16_t* g, const uint16_t* u,
                  int rows, int n_ff, cudaStream_t stream);

// v0.2.1: qwen3.5 gated attention + F16 head support.
// out (rows x row_len, contiguous) *= sigmoid(gate row), where gate is
// [rows x gate_stride] and each row's gates start at column gate_off.
void sigmoid_mul_rows_f16(uint16_t* out, const uint16_t* gate, int total,
                          int row_len, int gate_stride, int gate_off,
                          cudaStream_t stream);
// out[i] *= sigmoid(g[0]) - shared-expert scalar router (fp32 logit).
void sigmoid_gate_scalar_f16(uint16_t* out, const float* g, int n,
                             cudaStream_t stream);
// fp16 -> fp32 (logits after the F16 head GEMV).
void convert_f16_to_f32(const uint16_t* src, float* dst, int n,
                        cudaStream_t stream);

// ---- KV cache ----------------------------------------------------------------
// Append K/V rows (per layer) into the cache slabs at the device-position.
// q8 KV writes 34-byte Q8_0 rows; fp16 KV writes 2B rows. n_tokens_host
// sizes the launch grid; positions come from the device param block.
void kv_append(const uint16_t* k, const uint16_t* v, uint8_t* k_cache,
               uint8_t* v_cache, uint16_t* k_scale, uint16_t* v_scale,
               int kv_dtype, const SeqParams* d_params, int n_tokens_host,
               int n_head_kv, int head_dim, cudaStream_t stream);

// GQA attention (decode: 1 token; prefill: n_tokens, causal). Reads K/V
// straight from the layer's cache slabs. sm_scale = 1/sqrt(head_dim).
void attention_decode_f16(uint16_t* out, const uint16_t* q,
                          const uint8_t* k_cache, const uint8_t* v_cache,
                          const uint16_t* k_scale, const uint16_t* v_scale,
                          int kv_dtype, const SeqParams* d_params,
                          int n_head, int n_head_kv, int head_dim,
                          float sm_scale, cudaStream_t stream);

void attention_prefill_f16(uint16_t* out, const uint16_t* q,
                           const uint8_t* k_cache, const uint8_t* v_cache,
                           const uint16_t* k_scale, const uint16_t* v_scale,
                           int kv_dtype, const SeqParams* d_params,
                           int n_head, int n_head_kv, int head_dim,
                           float sm_scale, int n_tokens_host,
                           cudaStream_t stream);

// ---- MoE ----------------------------------------------------------------------
// Softmax over logits then top-k (values sorted descending).
void router_topk(const float* logits, int* idx, float* weight, int n_expert,
                 int n_expert_used, cudaStream_t stream);

// v0.2.3 batched prefill path:
// Top-k for n_rows routers at once; logits [n_rows x n_expert], results at
// [row * idx_stride + kk] (idx_stride is usually K+1 so the shared-expert
// combine slot stays host-managed).
void router_topk_batched(const float* logits, int* idx, float* weight,
                         int n_expert, int n_expert_used, int idx_stride,
                         int n_rows, cudaStream_t stream);

// dst[r] = src[src_rows[r]] for whole BlockQ8K activation rows (the routed
// rows are compacted expert-major before the per-expert GEMMs).
void gather_rows_q8(q4k::BlockQ8K* dst, const q4k::BlockQ8K* src,
                    const int* src_rows, int n_rows, int row_blocks,
                    cudaStream_t stream);

// Compact per-expert outputs back into the combine slab:
// expert_y[row * (k_slots) * hidden + kk * hidden + i] =
//     grouped_out[gslot[row * n_used + kk] * hidden + i]   (gslot < 0: skip)
// where k_slots = n_used + 1 (last slot = shared expert, host-copied).
void scatter_routed_f16(uint16_t* expert_y, const uint16_t* grouped_out,
                        const int* gslot, int n_rows, int n_used, int hidden,
                        cudaStream_t stream);

// Batched combine: out[row] = residual + sum_kk weight[row, kk] * expert_y
// [row, kk]; slots are positional (k_slots = n_used, shared folded in by the
// caller), no idx indirection needed.
void moe_combine_batched_f16(uint16_t* out, const uint16_t* expert_y,
                             const float* weight, int n_used, int hidden,
                             int n_rows, cudaStream_t stream);

// Batched shared-expert scalar gate: out[row * hidden + i] *= sigmoid(g[row]).
void sigmoid_gate_scalar_batched_f16(uint16_t* out, const float* g,
                                     int n_rows, int hidden,
                                     cudaStream_t stream);

// CPU-pool fallback results into the batched combine slab:
// dst[dst_flat[i]] = fp16(src row i) (src is fp32, converted in-kernel);
// dst_flat holds full element offsets.
void scatter_rows_generic_f16(uint16_t* dst, const float* src,
                              const int* dst_flat, int n_list, int hidden,
                              cudaStream_t stream);

// Combine: out = residual + sum_k weight[k] * expert_out[k]
// (+ shared_out already folded into expert_out by the caller's row setup).
void moe_combine_f16(uint16_t* out, const uint16_t* expert_out,
                     const int* idx, const float* weight, int n_expert_used,
                     int hidden, cudaStream_t stream);

// Scatter CPU-pool miss results (fp32, pinned-sourced device buffer) into
// the fp16 expert_out rows. d_slot_of_k[k] = miss slot or -1 for hits.
void scatter_miss_f32(uint16_t* expert_out, const int* d_slot_of_k,
                      const float* miss_y, int k_rows, int hidden,
                      cudaStream_t stream);

// Scatter compact hit rows into their routed slots: dst[dst_rows[i]] = src[i].
void scatter_rows_f16(uint16_t* dst, const uint16_t* src, const int* dst_rows,
                      int n_rows, int hidden, cudaStream_t stream);

// ---- sampling -----------------------------------------------------------------
// Sampling: repetition/presence penalty, temperature, top-k, top-p, min_p.
// logits [n_vocab] fp32, mutated in place. Returns chosen id.
int sample_topk_topp(const float* logits, int n_vocab, const int* recent_ids,
                     int n_recent, float temp, int top_k, float top_p,
                     float min_p, float rep_penalty, float presence_penalty,
                     uint64_t* rng_state, cudaStream_t stream);

// Greedy argmax (used by the speculative verify window - bit-exact decoding).
int argmax_f32(const float* logits, int n_vocab, cudaStream_t stream);

}  // namespace sager::cuda
