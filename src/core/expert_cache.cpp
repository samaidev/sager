// expert_cache.cpp - decayed-frequency hot-expert residency (see header).
#include "sager/core/expert_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "sager/log.hpp"

namespace sager {

void ExpertCache::init(int n_layers, int n_experts, size_t budget_bytes,
                       size_t per_expert_bytes) {
  n_layers_ = n_layers;
  n_experts_ = n_experts;
  budget_bytes_ = budget_bytes;
  per_expert_bytes_ = per_expert_bytes;
  capacity_ = per_expert_bytes_ > 0
                  ? static_cast<int>(budget_bytes_ / per_expert_bytes_)
                  : 0;
  const size_t n = idx(n_layers_ - 1, n_experts_ - 1) + 1;
  resident_bits_.assign((n + 63) / 64, 0);
  freq_.assign(n, 0.f);
  persisted_.assign(n, 0.f);
  SLOG_I("cache", "init: %d layers x %d experts, budget %.1f GB, per-expert %.2f MB, capacity %d/%d",
         n_layers_, n_experts_, budget_bytes_ / 1073741824.0,
         per_expert_bytes_ / 1048576.0, capacity_, n);
}

void ExpertCache::touch(int layer, int expert) {
  const size_t i = idx(layer, expert);
  // Frequency bump happens regardless of residency: misses are exactly the
  // signal we want for the next rebalance (a missed expert got hotter).
  freq_[i] += 1.0f;
  if (resident_bits_[i >> 6] & (1ull << (i & 63))) {
    hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    misses_.fetch_add(1, std::memory_order_relaxed);
  }
}

void ExpertCache::tick(float decay_factor) {
  // Called once per generation step. One pass over ~6k floats is negligible
  // next to a decode step; keep it branch-free.
  for (auto& f : freq_) f *= decay_factor;
}

int ExpertCache::rebalance() {
  if (capacity_ <= 0) return 0;

  // Rank ALL experts by frequency; keep the top-capacity resident, evict the
  // rest. Simple, deterministic, and cheap enough off the hot path.
  struct Ent { float f; size_t i; };
  const size_t n = freq_.size();
  std::vector<Ent> ents;
  ents.reserve(n);
  double mass = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const float f = freq_[i] + persisted_[i];
    mass += f;
    ents.push_back({f, i});
  }
  if (mass <= 0.0) {
    // v0.3: COLD-START UNIFORM RESIDENCY. With no usage profile every score
    // is 0 and a plain partial_sort keeps the FIRST capacity_ pairs - which
    // are layer-major, i.e. the first ~21 layers fully resident and the rest
    // empty. The streamed prefill then races through the all-hot layers
    // (stream_progress advances done_layer_ for free) while the loaders are
    // still on their first cold preads, and every remaining plan pair dies
    // to the take-time "past layer" skip (measured: 5307/5309 skipped, whole
    // layers dropped to the CPU pool). Seeding EXPERT-MAJOR round-robin
    // instead gives every layer the same hit rate, so the walk paces against
    // the loaders layer by layer - and decode sees a uniform miss pattern.
    int kept = 0;
    std::vector<uint64_t> new_bits(resident_bits_.size(), 0);
    for (int e = 0; e < n_experts_ && kept < capacity_; ++e) {
      for (int L = 0; L < n_layers_ && kept < capacity_; ++L) {
        const size_t i = idx(L, e);
        new_bits[i >> 6] |= (1ull << (i & 63));
        ++kept;
      }
    }
    int evicted = 0;
    for (size_t i = 0; i < n; ++i) {
      const bool was = resident_bits_[i >> 6] & (1ull << (i & 63));
      const bool now = new_bits[i >> 6] & (1ull << (i & 63));
      if (was && !now) ++evicted;
    }
    resident_bits_ = std::move(new_bits);
    ++rebalances_;
    SLOG_I("cache", "rebalance #%llu (cold-start uniform): resident=%d/%d, evicted=%d",
           static_cast<unsigned long long>(rebalances_.load()), kept,
           static_cast<int>(n), evicted);
    return 0;
  }
  std::partial_sort(ents.begin(), ents.begin() + std::min<size_t>(capacity_, n),
                    ents.end(), [](const Ent& a, const Ent& b) { return a.f > b.f; });

  int moved = 0;
  const size_t keep = std::min<size_t>(capacity_, n);
  std::vector<uint64_t> new_bits(resident_bits_.size(), 0);
  for (size_t r = 0; r < keep; ++r) {
    const size_t i = ents[r].i;
    new_bits[i >> 6] |= (1ull << (i & 63));
    const bool was = resident_bits_[i >> 6] & (1ull << (i & 63));
    if (!was && ents[r].f > 0.f) ++moved;  // insertion (DMA cost on caller)
  }
  int evicted = 0;
  for (size_t i = 0; i < n; ++i) {
    const bool was = resident_bits_[i >> 6] & (1ull << (i & 63));
    const bool now = new_bits[i >> 6] & (1ull << (i & 63));
    if (was && !now) ++evicted;
  }
  resident_bits_ = std::move(new_bits);
  ++rebalances_;
  SLOG_D("cache", "rebalance #%llu: +%d in, -%d out, resident=%d/%d",
         static_cast<unsigned long long>(rebalances_.load()), moved, evicted,
         capacity_, static_cast<int>(n));
  return moved;
}

void ExpertCache::set_resident(int layer, int expert, bool on) {
  const size_t i = idx(layer, expert);
  if (on) resident_bits_[i >> 6] |= (1ull << (i & 63));
  else    resident_bits_[i >> 6] &= ~(1ull << (i & 63));
}

ExpertCache::Stats ExpertCache::stats() const {
  Stats s;
  s.hits = hits_.load(std::memory_order_relaxed);
  s.misses = misses_.load(std::memory_order_relaxed);
  s.rebalances = rebalances_.load(std::memory_order_relaxed);
  s.resident_count = 0;
  for (const auto b : resident_bits_) {
    s.resident_count += __builtin_popcountll(b);
  }
  s.capacity = capacity_;
  return s;
}

// ---- profile persistence ----------------------------------------------------
json::Value ExpertCache::profile_to_json() const {
  json::Value v, f, p;
  for (size_t i = 0; i < freq_.size(); ++i) f.push(static_cast<double>(freq_[i]));
  for (size_t i = 0; i < persisted_.size(); ++i) p.push(static_cast<double>(persisted_[i]));
  v.set("version", json::Value(1));
  v.set("n_layers", json::Value(n_layers_));
  v.set("n_experts", json::Value(n_experts_));
  v.set("freq", json::Value(std::move(f)));
  v.set("persisted", json::Value(std::move(p)));
  return v;
}

void ExpertCache::profile_from_json(const json::Value& v) {
  if (v.find("freq") == nullptr) return;
  const json::Array& fa = v.find("freq")->as_array();
  const json::Array& pa = v.find("persisted") ? v.find("persisted")->as_array() : json::Array{};
  const size_t n = freq_.size();
  for (size_t i = 0; i < n && i < fa.size(); ++i) freq_[i] = static_cast<float>(fa[i].as_double());
  for (size_t i = 0; i < n && i < pa.size(); ++i) persisted_[i] = static_cast<float>(pa[i].as_double());
}

bool ExpertCache::save_profile(const std::string& path) const {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) return false;
  const std::string s = profile_to_json().dump();
  const bool ok = std::fwrite(s.data(), 1, s.size(), fp) == s.size();
  std::fclose(fp);
  return ok;
}

bool ExpertCache::load_profile(const std::string& path) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  std::string buf;
  char tmp[4096];
  size_t r;
  while ((r = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.append(tmp, r);
  std::fclose(fp);
  json::Value v;
  if (!json::Value::parse(buf, &v)) return false;
  profile_from_json(v);
  return true;
}

}  // namespace sager
