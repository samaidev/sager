# Roadmap

Ordered by impact on the single-T4 mission.

## v0.1 (this release)

- Three-tier expert storage: VRAM hot cache + pinned RAM pool (SSD reserved)
- In-house Q4_K kernels: CUDA sm_75 (dp4a GEMV family, streaming GQA,
  fused router/combine) + AVX2 CPU pool
- Per-layer decode CUDA graphs with the fixed-address device-block contract
- MTP (model-native head) + suffix drafting with cost-model gating and
  bit-exact greedy verification
- q8 KV cache, chunked prefill, conversation retention with KV checkpoints
- OpenAI-compatible server, web chat, HF converter, quant validator, bench

## v0.2 - hybrid attention + expert streaming (shipped) & throughput

- **Gated Delta Net kernels** (DONE, v0.2): CPU reference + sm_75 CUDA,
  fp32 recurrence state, graph-capturable; hybrid layer dispatch; KV only
  for full-attention layers. See docs/gdn-kernel.md.
- **Per-token expert pump** (DONE, v0.2): async pinned-RAM -> VRAM arena
  streaming with LRU slots + event fences; CPU pool demoted to fallback.
  See docs/expert-pump.md.
- **Converter v0.2** (DONE, v0.2): real Sage-A1.5 layout (language_model
  prefix, linear_attn.*, mlp/moe aliases) + hybrid metadata; chain smoke in
  tests/run_hybrid_chain.sh.
- **Batched MoE prefill**: chunk-level router GEMV, one D2H for all top-k,
  hit-set grouping, ptr-GEMV per group (closes the prefill gap; see
  performance.md)
- **Turing int4 tensor-core path** behind `SAGER_T4_INT4_TC=1` (sm_75
  supports int4 mma via wmma fragments)
- **Multi-session scheduling**: one engine process, N sessions, graph-safe
  slot swapping (KV slabs are already carved per layer)
- **Sampling-aware verification** (rejection sampling on the drafter's
  distribution) alongside greedy verify

## v0.3 - the third tier

- **SSD expert tier**: n-gram-prefetched experts from NVMe through a
  pinned staging ring; the profile already knows what will be needed
- **Expert-granularity quantization profiles** (`expert_profiles/`): mix
  Q4_K (cold) with Q5_K/Q6_K (hot) per residency tier, hot experts get the
  precision they earn
- **CUDA-graph-aware MoE**: capture the hits path too, with an index
  rewrite pass instead of pointer tables

## Explorations

- AMX/AVX512-VNNI CPU pool variant (server-class hosts)
- ZeRO-style expert sharding across 2x T4 with NVLink-free P2P over PCIe
- Drafting from the suffix table ACROSS sessions (tool-call frames repeat
  fleet-wide, not just per conversation)
