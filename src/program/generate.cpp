// generate.cpp - sampling + the speculative generation loop.
//
// Per iteration:
//   1. sample t from the current logits (host sampler over n_vocab floats)
//   2. ask the controller whether drafting pays; if yes, collect candidates
//      from the suffix drafter (free) and the MTP head (~0.6 ms/token)
//   3. verify [t, c1..cd] in ONE batched forward; commit the accepted prefix
//   4. emit committed tokens; the horizon distribution becomes the next logits
//
// Greedy verification keeps output bit-identical to plain decoding.
#include "generate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "sager/core/verify.hpp"
#include "sager/log.hpp"

namespace sager {

// ---------------------------------------------------------------------------
// Host sampler (single O(n) selection pass - no full sort of 152k floats)
// ---------------------------------------------------------------------------
int sample_host(const float* logits, int n_vocab, const std::vector<int>& recent,
                const GenParams& p, uint64_t& rng_state) {
  thread_local std::vector<float> v;
  v.assign(logits, logits + n_vocab);

  // Penalties over the recent window.
  if (p.presence_penalty != 0.f || p.repeat_penalty != 1.0f) {
    thread_local std::vector<int> seen;
    seen.assign(n_vocab, 0);
    for (int id : recent) {
      if (id >= 0 && id < n_vocab) {
        seen[id] = 1;
        if (p.repeat_penalty != 1.0f) {
          v[id] = v[id] > 0.f ? v[id] / p.repeat_penalty
                              : v[id] * p.repeat_penalty;
        }
      }
    }
    if (p.presence_penalty != 0.f) {
      for (int id : recent) {
        if (id >= 0 && id < n_vocab) v[id] -= p.presence_penalty;
      }
      (void)seen;
    }
  }

  if (p.greedy) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n_vocab; ++i) {
      if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
  }

  const float temp = p.temperature > 1e-4f ? p.temperature : 1e-4f;

  // Top-k by selection (k is small; O(n) nth_element).
  int k = p.top_k > 0 ? std::min(p.top_k, n_vocab) : n_vocab;
  thread_local std::vector<int> idx;
  idx.resize(n_vocab);
  for (int i = 0; i < n_vocab; ++i) idx[i] = i;
  std::nth_element(idx.begin(), idx.begin() + k - 1, idx.end(),
                   [&](int a, int b) { return v[a] > v[b]; });
  idx.resize(static_cast<size_t>(k));

  // Softmax over the kept candidates.
  float mx = -1e30f;
  for (int i : idx) mx = std::max(mx, v[i]);
  float z = 0.f;
  for (int i : idx) {
    v[i] = std::exp((v[i] - mx) / temp);
    z += v[i];
  }
  for (int i : idx) v[i] /= z;

  // Sort the k candidates by prob desc for top-p / min_p truncation.
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });

  // min_p: keep p_i >= min_p * p_max.
  if (p.min_p > 0.f) {
    const float floor_ = p.min_p * v[idx[0]];
    size_t keep = 0;
    while (keep < idx.size() && v[idx[keep]] >= floor_) ++keep;
    idx.resize(std::max<size_t>(keep, 1));
  }

  // top-p: smallest prefix with cumulative mass >= top_p.
  if (p.top_p < 1.0f) {
    float cum = 0.f;
    size_t keep = idx.size();
    for (size_t i = 0; i < idx.size(); ++i) {
      cum += v[idx[i]];
      if (cum >= p.top_p) { keep = i + 1; break; }
    }
    idx.resize(std::max<size_t>(keep, 1));
  }

  // Renormalize over the surviving prefix and draw.
  z = 0.f;
  for (int i : idx) z += v[i];
  auto next01 = [&]() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return static_cast<float>(static_cast<double>(rng_state >> 11) / 9007199254740992.0);
  };
  float r = next01() * z;
  for (size_t i = 0; i < idx.size(); ++i) {
    r -= v[idx[i]];
    if (r <= 0.f) return idx[i];
  }
  return idx[0];
}

// ---------------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------------
Status run_generation(Session& s, SpecController& ctrl, MtpDrafter& mtp,
                      SuffixDrafter& suffix, const GenParams& p,
                      const std::atomic<bool>* abort, GenCallbacks cb,
                      GenResult* out) {
  const int n_vocab = s.model_config().dims.n_vocab;
  const double t0 = now_ms();

  // ---- prefill (conversation-aware trimming happens above this layer) ----
  Status st = s.prefill(p.prompt_ids, abort);
  if (st != Status::kOk) {
    out->status = st;
    return st;
  }
  out->prefill_tps = s.perf.prefill_tps;

  uint64_t rng = p.seed ? p.seed : static_cast<uint64_t>(now_ms() * 1000.0) ^ 0x9E3779B97F4A7C15ull;

  // Local context for the suffix drafter (capped sliding window). The
  // drafter is caller-owned so its n-gram table can outlive a request.
  thread_local std::vector<int> ctx;
  ctx.clear();
  ctx.insert(ctx.end(), p.prompt_ids.end() - std::min<size_t>(p.prompt_ids.size(), 4096),
             p.prompt_ids.end());
  suffix.reset();
  // max_draft comes from the load-time config (enable_suffix_draft /
  // suffix_max_draft); do not override it here - the disable switch and the
  // MTP-isolation smoke both rely on it being respected.
  suffix.update(ctx, 3);

  std::vector<float> cur_logits(s.last_logits(), s.last_logits() + n_vocab);
  const double decode_t0 = now_ms();
#if defined(SAGER_HAVE_CUDA)
  // v0.5: the device sampler applies penalties on device before the top-k
  // selection; the host finish reuses the same policy for the tail.
  s.set_sampler_penalties(p.repeat_penalty, p.presence_penalty);
#endif
  SLOG_I("gen", "drafters: suffix_max=%d mtp_ready=%d mtp_max=%d",
         suffix.max_draft(), mtp.ready() ? 1 : 0, mtp.max_draft());
  // v0.7 fix: the device-side penalties read these - v0.5/v0.6 never set
  // them, so the device route sampled without the request's penalties.
  s.set_sampler_penalties(p.repeat_penalty, p.presence_penalty);

  while (out->tokens < p.max_tokens) {
    if (abort && *abort) { out->status = Status::kAborted; break; }

    // 1. sample the next token (device top-k when the last decode_step
    //    produced one; the prefill tail always goes through the host path)
    int t;
#if defined(SAGER_HAVE_CUDA)
    if (s.device_sample_valid()) {
      t = s.finish_device_sample(p.temperature, p.greedy ? 1 : p.top_k,
                                 p.top_p, p.min_p, p.repeat_penalty,
                                 p.presence_penalty, rng);
      if (t < 0) t = sample_host(cur_logits.data(), n_vocab, s.recent(), p, rng);
    } else {
      t = sample_host(cur_logits.data(), n_vocab, s.recent(), p, rng);
    }
#else
    t = sample_host(cur_logits.data(), n_vocab, s.recent(), p, rng);
#endif
    ++out->tokens;
    bool stop = false;
    if (cb.on_token && !cb.on_token(t)) { stop = true; out->status = Status::kAborted; }
    if (!stop) {
      for (int sid : p.stop_ids) {
        if (t == sid) { stop = true; break; }
      }
    }

    // Keep the suffix index fresh with committed tokens only.
    ctx.push_back(t);
    if (ctx.size() > 4096) {
      ctx.erase(ctx.begin(), ctx.begin() + 2048);
      suffix.reset();
      suffix.update(ctx, 3);
    } else {
      suffix.update(ctx, static_cast<int>(ctx.size()) - 1);
    }

    if (stop) break;

    // 2. drafting decision (cost model gates everything)
    const auto& recent = s.recent();
    const int last3[3] = {recent.size() >= 3 ? recent[recent.size() - 3] : t,
                          recent.size() >= 2 ? recent[recent.size() - 2] : t,
                          t};
    std::vector<int> cand;
    const int depth = ctrl.choose_depth(ctrl.config().miss_ms_expected);
    if (depth > 0) {
      const double mtp_t0 = now_ms();
      cand = suffix.draft(last3);
      if (static_cast<int>(cand.size()) > depth) cand.resize(static_cast<size_t>(depth));
      if (mtp.ready()) {
        auto m = mtp.draft(t);
        const size_t room = static_cast<size_t>(std::max(0, depth - static_cast<int>(cand.size())));
        if (m.size() > room) m.resize(room);
        cand.insert(cand.end(), m.begin(), m.end());
      }
#if defined(SAGER_HAVE_CUDA)
      s.prof.mtp_draft += now_ms() - mtp_t0;
#endif
    }

    // 3. verify or plain decode
    if (cand.empty()) {
      const float* lg = s.decode_step(t);
      if (!lg) { out->status = Status::kCudaError; break; }
#if defined(SAGER_HAVE_CUDA)
      if (!s.device_sample_valid())
#endif
      {
        std::memcpy(cur_logits.data(), lg, n_vocab * sizeof(float));
      }
      continue;
    }

    std::vector<int> window;
    window.reserve(cand.size() + 1);
    window.push_back(t);
    window.insert(window.end(), cand.begin(), cand.end());
    VerifyResult vr = verify_window(s, window);
#if defined(SAGER_HAVE_CUDA)
    ++s.prof.spec_windows;
    s.prof.drafted += static_cast<int>(cand.size());
#endif
    if (vr.next_token < 0) {
      // Verify pass failed - fall back to a plain decode of t.
      const float* lg = s.decode_step(t);
      if (!lg) { out->status = Status::kCudaError; break; }
#if defined(SAGER_HAVE_CUDA)
      if (!s.device_sample_valid())
#endif
      {
        std::memcpy(cur_logits.data(), lg, n_vocab * sizeof(float));
      }
      continue;
    }

    ctrl.observe(static_cast<int>(cand.size()), static_cast<int>(cand.size()),
                 vr.accepted);
    out->drafted += static_cast<int>(cand.size());
    out->accepted += vr.accepted;
#if defined(SAGER_HAVE_CUDA)
    s.prof.accepted += vr.accepted;
#endif

    // 4. emit the accepted candidates (window[0]=t already emitted).
    bool broke = false;
    for (int i = 1; i <= vr.accepted; ++i) {
      const int c = window[i];
      ++out->tokens;
      if (cb.on_token && !cb.on_token(c)) { out->status = Status::kAborted; broke = true; }
      for (int sid : p.stop_ids) {
        if (c == sid) { stop = true; break; }
      }
      ctx.push_back(c);
      if (ctx.size() > 4096) {
        ctx.erase(ctx.begin(), ctx.begin() + 2048);
        suffix.reset();
        suffix.update(ctx, 3);
      } else {
        suffix.update(ctx, static_cast<int>(ctx.size()) - 1);
      }
      if (stop || broke) break;
    }
    if (stop || broke) break;

    std::memcpy(cur_logits.data(), vr.top_logits.data(), n_vocab * sizeof(float));
  }

  const double ms = now_ms() - decode_t0;
  out->decode_tps = ms > 0 ? 1000.0 * out->tokens / ms : 0.0;
  s.perf.decode_tps = out->decode_tps;
  return out->status;
}

}  // namespace sager
