// common.hpp - shared types, constants and small helpers for the SAGER engine.
//
// SAGER is a Strata-inspired, single-model inference engine purpose-built for
// InternScience/Sage-A1.5 (35B MoE, ~3B active parameters, qwen3_5_moe
// architecture family). It runs a custom 4-bit (Q4_K super-block) build of the
// model "smoothly" on one Tesla T4 (16 GB, sm_75) by exploiting the MoE
// property: a token needs only a few experts, so most experts can live in
// system RAM while the hottest ones stay resident in VRAM.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <optional>

namespace sager {

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------
inline constexpr const char* kSagerVersion = "0.1.0";
inline constexpr const char* kSagerTargetModel = "InternScience/Sage-A1.5";

// ---------------------------------------------------------------------------
// Status codes - the engine prefers explicit, greppable failures over
// silent misindexing. See docs/architecture.md ("defensive systems design").
// ---------------------------------------------------------------------------
enum class Status : int {
  kOk = 0,
  kInvalidArgument = 1,
  kFileNotFound = 2,
  kGgufParseError = 3,
  kGeometryMismatch = 4,   // refuse to run instead of producing garbage
  kOomVram = 5,
  kOomRam = 6,
  kCudaError = 7,
  kUnsupported = 8,
  kAborted = 9,
};

inline const char* status_str(Status s) {
  switch (s) {
    case Status::kOk: return "ok";
    case Status::kInvalidArgument: return "invalid argument";
    case Status::kFileNotFound: return "file not found";
    case Status::kGgufParseError: return "gguf parse error";
    case Status::kGeometryMismatch: return "geometry mismatch";
    case Status::kOomVram: return "out of VRAM";
    case Status::kOomRam: return "out of system RAM";
    case Status::kCudaError: return "CUDA error";
    case Status::kUnsupported: return "unsupported";
    case Status::kAborted: return "aborted";
  }
  return "unknown";
}

// "!status" reads naturally at the ~30 alloc/init guard sites ("if (!ok(...))").
inline bool operator!(Status s) { return s != Status::kOk; }

// ---------------------------------------------------------------------------
// Geometry. Everything is metadata-driven: values are read from the GGUF
// header at load time, never hard-coded. The numbers below are only
// sanity-check defaults used to reject obviously wrong files.
// ---------------------------------------------------------------------------
struct EngineDims {
  int n_layer        = 48;     // transformer blocks
  int n_head         = 32;     // query heads
  int n_head_kv      = 4;      // GQA key/value heads
  int head_dim       = 128;    // per-head dim (q/k/v)
  int hidden         = 2048;   // model dim
  int n_ff           = 6144;   // dense FFN dim (some layers may be MoE)
  int n_expert       = 128;    // routed experts per MoE layer
  int n_expert_used  = 8;      // top-k routed experts per token
  int n_ff_exp       = 768;    // per-expert FFN dim
  int n_vocab        = 151936;
  int n_ctx_train    = 262144; // 256K native
  int shared_expert  = 1;      // shared expert per MoE layer (0 if none)
  int n_moe_layer    = 48;     // layers that are MoE (Sage-A1.5: all blocks)
  bool mtp           = true;   // MTP (multi-token prediction) head present
  bool vision        = false;  // mmproj present (v0.1 is text-only: skipped)

  int n_embd_gqa()  const { return n_head_kv * head_dim; }
  int experts_total() const { return n_moe_layer * n_expert; }
};

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------
inline constexpr int div_up(int a, int b) { return (a + b - 1) / b; }
inline constexpr size_t align_up(size_t v, size_t a) {
  return (v + a - 1) / a * a;
}

// ---------------------------------------------------------------------------
// Environment helpers (feature flags without recompiling)
// ---------------------------------------------------------------------------
std::optional<std::string> env_get(const char* name);
bool env_flag(const char* name);                 // SAGER_X=1
int env_int(const char* name, int fallback);

// ---------------------------------------------------------------------------
// Monotonic time in milliseconds (for scheduler cost models).
// ---------------------------------------------------------------------------
double now_ms();

// Rough host RAM size in bytes (0 = unknown).
uint64_t host_ram_bytes();

}  // namespace sager
