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

// v0.6: native Q6_K x q8 GEMV (single token - the decode head). w_q6 points
// at raw block_q6_K bytes (210 B per 256 weights, llama.cpp layout); the
// result matches the f16-dequant head up to weight-rounding differences (the
// native dot is the more faithful of the two). One warp per row, dp4a math.
void gemv_q6_k_q8(const void* w_q6, const q4k::BlockQ8K* x_q8, float* y,
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

// ---- v0.3 mtok tiled kernels (prefill / verify; weight-traffic fix) ---------
// Same math and per-(row, token) accumulation order as the batched kernels
// above (bit-identical outputs), but tokens are TILED through shared memory
// so weight rows are read from DRAM once per tile instead of once per token.
// y layout is unchanged: y[tok * n_rows + row].

// Single matrix, fp16 out.
void gemv_q4_k_q8_f16out_mtok(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                              uint16_t* y, int n_rows, int k, int n_tokens,
                              cudaStream_t stream);

// ---- v0.8 MMQ prefill kernels (bit-identical to the mtok kernels above;
// the per-token dp4a chains are interleaved for ILP instead of serialized).
// y layout: y[tok * n_rows + row].
void mmq_q4_k_q8_f16out(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                        uint16_t* y, int n_rows, int k, int n_tokens,
                        cudaStream_t stream);

// Fused gate+up MMQ (two matrices, same n_rows): one launch.
void mmq2_q4_k_q8_f16out(const q4k::BlockQ4K* wg, const q4k::BlockQ4K* wu,
                         const q4k::BlockQ8K* xq, uint16_t* yg, uint16_t* yu,
                         int n_rows, int k, int n_tokens, cudaStream_t stream);

// ---- v0.9 prefill tensor-core path (SAGER_PF_TC; 0 reverts to the dp4a
// GEMV family) -------------------------------------------------------------
// Q4_K weights dequantize into a fp16 scratch ONCE per chunk, then cuBLAS
// GemmEx (fp16 in / fp32 accumulate, sm_75 HMMA) runs the projection as a
// real GEMM: out[t * n_rows + row] = sum_k x[t * k + kk] * w[row * k + kk].
// NOT bit-identical to the dp4a path (fp16 weight rounding, fp16 activations
// instead of Q8_K requant, different accumulation order); decode never takes
// it. Returns false on cuBLAS failure - callers fall back to the GEMV path.
bool tc_gemm_q4_k_f16(const q4k::BlockQ4K* w, const uint16_t* x, uint16_t* y,
                      uint16_t* scratch, int n_rows, int k, int n_tokens,
                      cudaStream_t stream);
void* tc_handle();
size_t tc_scratch_elems(int max_rows, int max_k);

// ---- v0.10 MoE tensor-core path (SAGER_MOE_TC; 0 reverts to the grouped
// dp4a GEMV family) ----------------------------------------------------------
// Routed experts' Q4_K weights dequantize into a small ring of fp16 slots,
// then one cublasGemmEx per (expert, matrix) runs the grouped GEMM with the
// same layout contract as tc_gemm_q4_k_f16. Not bit-identical to the dp4a
// grouped GEMV (same reasons as the dense TC path); runs for prefill chunks
// of >= 256 tokens only.
bool tc_gemm_f16_raw(const void* w16, const void* x16, void* y16,
                     int n_rows, int k, int n_tokens, cudaStream_t stream);
void dequant_experts_q4_k_f16(const q4k::BlockQ4K* const* tab_a,
                              const q4k::BlockQ4K* const* tab_b,
                              const int* off, void* slots, long long dst_row0,
                              int rows_sl2, int k, int n_experts, int ring,
                              long long slot_elems, cudaStream_t stream);

// ---- v0.12 shared-expert / router TC (SAGER_SH_TC / SAGER_ROUTER_TC; 0
// reverts to the dp4a GEMV family) ------------------------------------------
// fp32-out GemmEx primitive (router logits keep the fp32 output dtype).
bool tc_gemm_f16_f32out_raw(const void* w16, const void* x16, float* y32,
                            int n_rows, int k, int n_tokens,
                            cudaStream_t stream);
// MoE router: cast the fp32 [n_rows x k] weight to fp16 in `scratch` once,
// then tc_gemm_f16_f32out_raw. Logits differ at the fp16-rounding level.
void tc_router_f32(const float* w_f32, const uint16_t* x, float* y,
                   uint16_t* scratch, int n_rows, int k, int n_tokens,
                   cudaStream_t stream);

// ---- v0.8 decode fusion kernels (all bit-identical to what they replace) ----

// rmsnorm + Q8_K quantization in one launch (n_rows blocks of one row each).
// Falls back to the two-kernel sequence for unsupported geometries.
void rmsnorm_quant_f16(uint16_t* out, const uint16_t* x, const uint16_t* w,
                       q4k::BlockQ8K* xq, int n, float eps, int n_rows,
                       cudaStream_t stream);

// The layer's 2 (GDN qkvz|ba) or 3 (GQA q|k|v) projections in ONE launch.
// `slot` identifies the layer (0..63): the parameter block is written once
// per layer at capture time, so graph replay needs no host interplay.
void gemv2m_q4_k_q8_f16out(const q4k::BlockQ4K* w1, const q4k::BlockQ4K* w2,
                           const q4k::BlockQ8K* xq, uint16_t* y1, uint16_t* y2,
                           int rows1, int rows2, int k, int slot,
                           cudaStream_t stream);
void gemv3m_q4_k_q8_f16out(const q4k::BlockQ4K* w1, const q4k::BlockQ4K* w2,
                           const q4k::BlockQ4K* w3, const q4k::BlockQ8K* xq,
                           uint16_t* y1, uint16_t* y2, uint16_t* y3, int rows1,
                           int rows2, int rows3, int k, int slot,
                           cudaStream_t stream);

// GEMV with the residual add fused as an epilogue (single token):
// y = W @ x (fp16, same as gemv_q4_k_q8_f16out), then
// res_inout[i] = f16(f32(res_inout[i]) + f32(y[i])) - add_kernel semantics.
void gemv_q4_k_q8_f16out_resadd(const q4k::BlockQ4K* w, const q4k::BlockQ8K* xq,
                                uint16_t* y, uint16_t* res_inout, int n_rows,
                                int k, cudaStream_t stream);

// Router GEMV (fp32 W, fp16 x) + warp-parallel top-k in ONE launch (n=1).
// Selection/softmax semantics match router_topk_batched_kernel row 0.
void router_topk_fused(const float* w_f32, const uint16_t* x_f16, int* idx,
                       float* weight, int n_expert, int n_expert_used,
                       int hidden, cudaStream_t stream);

// Q6_K head GEMV with x staged in shared memory (bit-identical to
// gemv_q6_k_q8; lower DRAM/L2 traffic on the 248k-row head).
void gemv_q6_k_q8_smx(const void* w_q6, const q4k::BlockQ8K* x_q8, float* y,
                      int n_rows, int k, cudaStream_t stream);

// Fused gate+up: two matrices share the staged X rows (one launch, x read
// once). yg/yu are [n_tokens x n_rows] each.
void gemv2_q4_k_q8_f16out_mtok(const q4k::BlockQ4K* wg, const q4k::BlockQ4K* wu,
                               const q4k::BlockQ8K* xq, uint16_t* yg,
                               uint16_t* yu, int n_rows, int k, int n_tokens,
                               cudaStream_t stream);

// Grouped fused gate+up: ALL routed experts of a layer in ONE launch.
// Job j serves activations xq + off[j] rows (m[j] rows) with weights
// wg[j]/wu[j]; tile_base[j] is the prefix of ceil(m[j]/TILE) over jobs and
// job_of_tile[t] maps flattened tile t back to its job (n_tiles = total).
void gemv_grouped2_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* wg, const q4k::BlockQ4K* const* wu,
    const int* off, const int* m, const int* tile_base,
    const int* job_of_tile, int n_tiles, const q4k::BlockQ8K* xq,
    uint16_t* yg, uint16_t* yu, int n_rows, int k, int tok_per,
    cudaStream_t stream);

// Grouped single matrix (down projection).
void gemv_grouped_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* w, const int* off, const int* m,
    const int* tile_base, const int* job_of_tile, int n_tiles,
    const q4k::BlockQ8K* xq, uint16_t* y, int n_rows, int k, int tok_per,
    cudaStream_t stream);

// ---- v0.4 device-side dispatch kernels (P1) ---------------------------------
// The grouped plan is built ON DEVICE by moe_dispatch_scan (histogram +
// prefix sums + row scatter); the GEMMs derive their (expert, tile) mapping
// from off[]/toff[] and read per-expert weights from the chunk pointer table
// (null pointer = expert absent -> outputs zeroed, expert id mailed to the
// host for the CPU-pool patch). All bit-exact vs the host-compacted path.
//
// One block over the routed ids of a layer: writes off[E+1] (row prefix),
// toff[E+1] (tile prefix), meta{R, tiles, bad_rows}, gslot/grows (grouped
// order), and zeroes the weights of bounds-invalid rows. tok_per = GEMM tile.
void moe_dispatch_scan(const int* idx, float* weight, int* gslot, int* grows,
                       int* off, int* toff, int* meta, int* mail, int* mail_n,
                       int n_rows, int k, int n_expert, int tok_per,
                       cudaStream_t stream);

// Device-count variants: the row/element count lives in device memory
// (meta[0] resp. meta[0]*n_ff); the host only bounds the launch grid.
void gather_rows_q8_n(q4k::BlockQ8K* dst, const q4k::BlockQ8K* src,
                      const int* src_rows, const int* meta, int r_max,
                      int row_blocks, cudaStream_t stream);
void gather_rows_f16_n(uint16_t* dst, const uint16_t* src,
                       const int* src_rows, const int* meta, int r_max,
                       int hidden, cudaStream_t stream);
void silu_mul_f16_n(uint16_t* out, const uint16_t* g, const uint16_t* u,
                    const int* d_rows, int rows_max, int n_ff,
                    cudaStream_t stream);
void quantize_q8_k_n(const uint16_t* x_f16, q4k::BlockQ8K* out,
                     const int* d_rows, int row_mult, int n_max,
                     cudaStream_t stream);

// Grouped fused gate+up driven by the device plan (meta[1] tiles).
void gemv_grouped2_tab_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* wg_tab, const q4k::BlockQ4K* const* wu_tab,
    const int* off, const int* toff, const int* meta, int* mail, int* mail_n,
    int tiles_max, const q4k::BlockQ8K* xq, uint16_t* yg, uint16_t* yu,
    int n_rows, int k, int n_expert, int tok_per, cudaStream_t stream);

// Grouped down projection driven by the device plan.
void gemv_grouped_tab_q4_k_q8_f16out_mtok(
    const q4k::BlockQ4K* const* w_tab, const int* off, const int* toff,
    const int* meta, int tiles_max, const q4k::BlockQ8K* xq, uint16_t* y,
    int n_rows, int k, int n_expert, int tok_per, cudaStream_t stream);

// CPU-pool patch: x[row] += weight[slot] * expert_y[slot] for the packed
// (row * (n_used+1) + kk) slots in rk[0..n_list) - late add of the rows the
// device path left at zero (absent experts), after the batched combine ran.
void moe_combine_add_slots_f16(uint16_t* out, const uint16_t* expert_y,
                               const float* weight, const int* rk, int n_used,
                               int hidden, int n_list, cudaStream_t stream);

// Router logits, tiled multi-token (fp32 W, fp16 x, fp32 out; accumulation
// order per (row, token) preserved -> logits bit-identical to the batched
// router).
void gemv_f32_out_f16x_mtok(const float* w_f32, const uint16_t* x_f16,
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
// shared_w is written into weight[n_expert_used] - the shared-expert combine
// slot (v0.12.4: the router owns this slot; see moe.cu root-cause-8 note).
void router_topk(const float* logits, int* idx, float* weight, int n_expert,
                 int n_expert_used, float shared_w, cudaStream_t stream);

// v0.2.3 batched prefill path:
// Top-k for n_rows routers at once; logits [n_rows x n_expert], results at
// [row * idx_stride + kk] (idx_stride is usually K+1 so the shared-expert
// combine slot stays router-managed - shared_w lands in out_w[K]).
void router_topk_batched(const float* logits, int* idx, float* weight,
                         int n_expert, int n_expert_used, int idx_stride,
                         int n_rows, float shared_w, cudaStream_t stream);

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

// v0.5 device top-k selection: writes the top-`k` (value, index) pairs of
// logits[0..n) into out_v/out_i (device buffers of >= k slots), ordered by
// the iterative max+mask pass (descending values). Ties break toward the
// lower index. k must be <= 64. NOTE: the logits buffer IS modified (picked
// entries are masked) - pass a scratch copy. The host finishes
// softmax/top-p/min_p/multinomial over the k candidates (see generate.cpp).
void topk_select_f32(const float* logits, int n, float* out_v, int* out_i,
                     int k, cudaStream_t stream);

// v0.6 two-phase top-k: same selected set and tie rule as topk_select_f32,
// computed by parallel per-segment scans + one merge block (~25 us at
// n=248320, k=32 on T4). seg_v/seg_i: scratch for the per-segment candidates
// (>= ceil(n/2048) * k slots); falls back to the single-block kernel when the
// scratch is too small.
void topk_select_f32_2p(const float* logits, float* seg_v, int* seg_i,
                        int seg_slots, float* out_v, int* out_i, int n, int k,
                        cudaStream_t stream);

// v0.5: repetition/presence penalty pass over the recent-id window (the
// decode device sampler applies it before the top-k selection).
void apply_penalties_f32_devn(float* logits, const int* recent_ids,
                              const int* n_recent, const float* params,
                              cudaStream_t s);
void apply_penalties_f32(float* logits, const int* recent_ids, int n_recent,
                         float rep_penalty, float presence_penalty,
                         cudaStream_t stream);

}  // namespace sager::cuda
