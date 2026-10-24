#!/usr/bin/env bash
# run_hybrid_chain.sh - full HF -> GGUF -> engine-metadata chain smoke on a
# CPU-only box (no GPU, no real weights):
#
#   1. tests/make_synthetic_hf.py --hybrid   (Sage-A1.5 layout: GDN + GQA
#      layers under model.language_model.*, layer_types + linear_* config)
#   2. tools/convert_hf_to_sager.py --fast   (v0.2 converter: lin_* tensors,
#      sager.layer_is_linear + sager.gdn.* metadata)
#   3. inline C++ checker: ModelConfig::load must see the hybrid layout,
#      per-layer kinds, GDN geometry, and every tensor the engine loads.
#
# This is the contract test between the converter and the engine loader;
# run it after touching either side.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=${1:-/tmp/sager_hybrid_chain}
rm -rf "$WORK"
mkdir -p "$WORK"

echo "== 1. synthetic hybrid HF checkpoint =="
python3 tests/make_synthetic_hf.py "$WORK/hf" --hybrid

echo "== 2. convert (fast quant) =="
python3 tools/convert_hf_to_sager.py "$WORK/hf" "$WORK/hybrid.gguf" --workers 1 --fast

echo "== 3. engine metadata contract =="
cat > "$WORK/check_meta.cpp" <<'EOF'
#include <cstdio>
#include <string>
#include "sager/core/config.hpp"
#include "sager/core/gguf.hpp"
using namespace sager;
int main(int argc, char** argv) {
  ModelConfig mc;
  Status s = ModelConfig::load(argv[1], &mc);
  if (s != Status::kOk) { std::printf("FAIL config load: %s\n", status_str(s)); return 1; }
  int fails = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++fails; }
  };
  const EngineDims& d = mc.dims;
  check(d.n_layer == 8, "n_layer=8");
  check(mc.layer_is_linear.size() == static_cast<size_t>(d.n_layer),
        "layer_is_linear present");
  int nlin = mc.n_linear();
  check(nlin == 6, "6 GDN layers (3+1 pattern)");
  check(!mc.is_linear(3) && !mc.is_linear(7), "layers 3,7 are full attention");
  check(mc.is_linear(0) && mc.is_linear(2), "layers 0,2 are GDN");
  check(d.n_gdn_k_heads == 16 && d.n_gdn_v_heads == 32, "gdn heads 16/32");
  check(d.gdn_key_dim == 128 && d.gdn_value_dim == 128, "gdn dims 128/128");
  check(d.gdn_conv_kernel == 4, "conv kernel 4");
  // tensors the engine will load for a GDN layer
  gguf::File f;
  if (f.open(argv[1]) != Status::kOk) { std::printf("FAIL gguf open\n"); return 1; }
  const char* lin_tensors[] = {"blk.0.lin_in_qkvz", "blk.0.lin_in_ba",
                               "blk.0.lin_out", "blk.0.lin_conv_w",
                               "blk.0.lin_conv_b", "blk.0.lin_A_log",
                               "blk.0.lin_dt_bias", "blk.0.lin_q_norm",
                               "blk.0.lin_k_norm", "blk.0.lin_o_norm"};
  for (const char* t : lin_tensors) {
    const gguf::TensorInfo* ti = f.tensor(t);
    check(ti != nullptr, (std::string("tensor ") + t).c_str());
  }
  const gguf::TensorInfo* q = f.tensor("blk.0.lin_in_qkvz");
  if (q) {
    // rows = 2*16*128 + 2*32*128 = 12288, cols = 2048 (gguf dims reversed)
    check(q->dims.size() == 2 && q->dims[0] == 2048 && q->dims[1] == 12288,
          "lin_in_qkvz shape [12288,2048]");
  }
  const gguf::TensorInfo* cw = f.tensor("blk.0.lin_conv_w");
  if (cw) {
    // conv_dim = 2*16*128 + 32*128 = 8192, ck = 4
    check(cw->dims.size() == 2 && cw->dims[0] == 4 && cw->dims[1] == 8192,
          "lin_conv_w shape [8192,4]");
  }
  // GQA layer keeps its attention tensors
  check(f.tensor("blk.3.attn_q.weight") != nullptr, "blk.3 attn_q present");
  check(f.tensor("blk.3.lin_in_qkvz") == nullptr, "blk.3 has no GDN tensors");
  if (fails == 0) { std::printf("hybrid chain metadata: ALL OK\n"); return 0; }
  std::printf("%d FAILURES\n", fails);
  return 1;
}
EOF
export PATH="$HOME/.local/bin:$PATH"
g++ -std=c++20 -O1 -Iinclude "$WORK/check_meta.cpp" src/core/config.cpp \
    src/core/gguf.cpp src/core/common.cpp -o "$WORK/check_meta" -pthread
"$WORK/check_meta" "$WORK/hybrid.gguf"

echo "== 4. quant validation (spot) =="
python3 tools/validate_quant.py --gguf "$WORK/hybrid.gguf" --hf "$WORK/hf" --rows 4 || true

echo "HYBRID CHAIN SMOKE: DONE"
