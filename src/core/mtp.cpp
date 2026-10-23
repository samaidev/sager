// mtp.cpp - MTP (multi-token prediction) drafter.
//
// Sage-A1.5 ships a dedicated MTP head; the drafter runs the recurrence
//   cat  = [ emb(t) ; h ]              (h = final hidden of the last pass)
//   h'   = rmsnorm( W_fc  @ cat )      (mtp.fc.weight,  [H x 2H] fp16)
//   h''  = rmsnorm( W_o   @ h' )       (mtp.proj_out,   [H x H]  fp16)
//   t'   = argmax( lm_head @ h'' )     (the main Q4_K head)
// iterating up to max_draft tokens. Each step costs two small GEMVs plus one
// lm_head stream (~0.6 ms on T4) - far below a full 48-layer pass, which is
// what makes the verify window profitable (see spec/controller.hpp).
#include "sager/core/mtp.hpp"

#include <vector>

#if defined(SAGER_HAVE_CUDA)
#include <cuda_runtime.h>
#include "sager/kernels/cuda_ops.hpp"
#include "sager/log.hpp"
#endif

namespace sager {

#if defined(SAGER_HAVE_CUDA)
namespace {

// Lazily-grown device scratch for the drafter (single-session engine; a
// per-instance pimpl would buy nothing here - see docs/architecture.md).
struct MtpScratch {
  DeviceBuffer<uint16_t> cat, h1, h2;
  DeviceBuffer<q4k::BlockQ8K> q;
  DeviceBuffer<float> lg;
  std::vector<float> host_lg;
  int H = 0, V = 0;

  void ensure(int h, int v) {
    if (H == h && V == v) return;
    H = h; V = v;
    cat.alloc(2 * static_cast<size_t>(h));
    h1.alloc(h);
    h2.alloc(h);
    q.alloc(h / q4k::kQK_K);
    lg.alloc(v);
    host_lg.resize(v);
  }
};

int argmax_host(const float* v, int n) {
  int best = 0;
  float bv = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] > bv) { bv = v[i]; best = i; }
  }
  return best;
}

}  // namespace
#endif

std::vector<int> MtpDrafter::draft(int last_token) {
  std::vector<int> out;
#if defined(SAGER_HAVE_CUDA)
  if (!ready_ || max_draft_ <= 0) return out;
  Weights& w = s_.weights();
  if (!w.mtp_present || !w.mtp_proj_in || !w.mtp_proj_out) return out;
  if (!s_.prepared_state()) return out;

  const int H = w.dims.hidden;
  const int V = w.dims.n_vocab;
  const float eps = s_.model_config().norm_eps;
  static MtpScratch sc;               // function-local, grown on first use
  sc.ensure(H, V);
  cudaStream_t st = s_.stream().get();

  uint16_t* h_cur = s_.hidden_stash();      // final hidden of the last pass
  if (!h_cur) return out;

  for (int step = 0; step < max_draft_; ++step) {
    // cat = [ emb(t) ; h_cur ]
    cudaMemcpyAsync(sc.cat.get(), w.tok_embd + static_cast<size_t>(last_token) * H,
                    H * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(sc.cat.get() + H, h_cur, H * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, st);
    cuda::gemv_f16_f16out(w.mtp_proj_in, sc.cat.get(), sc.h1.get(), H, 2 * H, st);
    cuda::rmsnorm_f16(sc.h1.get(), sc.h1.get(), w.output_norm, H, eps, st);
    cuda::gemv_f16_f16out(w.mtp_proj_out, sc.h1.get(), sc.h2.get(), H, H, st);
    cuda::rmsnorm_f16(sc.h2.get(), sc.h2.get(), w.output_norm, H, eps, st);
    cuda::quantize_q8_k(sc.h2.get(), sc.q.get(), H, st);
    cuda::gemv_q4_k_q8(reinterpret_cast<const q4k::BlockQ4K*>(w.output),
                       sc.q.get(), sc.lg.get(), V, H, st);
    cudaMemcpy(sc.host_lg.data(), sc.lg.get(), V * sizeof(float),
               cudaMemcpyDeviceToHost);
    const int tok = argmax_host(sc.host_lg.data(), V);
    out.push_back(tok);
    last_token = tok;
    h_cur = sc.h2.get();              // chain: next draft conditions on h2
  }
  return out;
#else
  (void)last_token;
  return out;
#endif
}

}  // namespace sager
