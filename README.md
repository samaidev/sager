# SAGER

**A Strata-inspired, single-card inference engine that runs [Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) (35B MoE, ~3B active) smoothly in 4-bit on one NVIDIA Tesla T4 (16 GB).**

SAGER 把 MoE 的稀疏性当作存储问题来解：一张 T4 装不下 6144 个专家，但一个 token 只用 8 个。于是我们把"常用的专家留在显存流理台上，其余收进内存储藏室"——这正是 [Strata](https://github.com/Niko1221/Strata) 的三层专家存储思想，配上一套自研的 sm_75 4-bit 内核。

> English below each section · 每节先英文后中文

---

## What it is

A complete stack, no external inference framework:

| Layer | Component | Notes |
|---|---|---|
| Kernels | `src/kernels/cuda/*.cu` | In-house Q4_K dp4a GEMV family (single-row, batched, pointer-table fused), streaming online-softmax GQA attention with fp16/Q8 KV, mRoPE + partial rotary, fused router top-k / combine, all tuned for sm_75 (T4) |
| Kernels | `src/kernels/cpu/*.cpp` | AVX2 Q4_K expert FFN + lock-free-submission worker pool for VRAM misses |
| Core | `src/core/*` | GGUF v3 mmap loader, metadata-driven config, hot-expert residency cache (decay + rebalance + persistence), the session loop (per-layer decode CUDA graphs + eager MoE dispatch), chunked prefill, speculative verify, conversation retention |
| Spec | `src/spec/*` | Cost-model speculative controller, MTP drafter (Sage-A1.5's own MTP head), 3-gram suffix drafter |
| Serve | `serve/*` | OpenAI-compatible FastAPI server, engine process manager, tokenizer, web chat UI |
| Tools | `tools/*` | HF → Q4_K GGUF converter (validated byte-compatible format), per-tensor quant validator, CLI chat |
| Bench | `bench/bench.py` | prefill/decode/hit-rate/acceptance reporting |

## The four Strata features (all implemented)

1. **Expert hot cache (VRAM tier)** — exponentially-decayed per-(layer, expert) frequency scores; a rebalance pass fills free VRAM with the hottest experts and the pointer tables are rebuilt with one fused H2D. Usage profile persists to disk so restarts skip the cold ramp. `core/expert_cache.hpp`
2. **MTP speculative decoding** — Sage-A1.5 ships its own multi-token head (`num_speculative_tokens=3`); one MTP forward proposes up to 3 tokens, a batched greedy verify window commits them bit-exactly. A cost model (learned acceptance priors seeded from Strata's 2.4–3.2 tokens/pass) gates drafting so speculation never slows you down. `spec/controller.hpp`, `core/verify.hpp`
3. **CPU expert pool** — misses stream the expert from pinned RAM through AVX2 Q4_K dot products on worker threads, overlapped with the GPU via the doorbell contract (C1–C6 in `core/expert_source.hpp`). The CPU never applies router weights; the device-side combine is the only weighted sum, so CPU/GPU weight mismatch is structurally impossible.
4. **KV reuse + chunked prefill** — 8192-token chunks, q8 KV cache (2× context per GB), per-turn KV checkpoints so follow-ups re-prefill only the new suffix.

## Format: GGUF-compatible Q4_K, in-house kernels

The on-disk format is **byte-compatible with the standard Q4_K super-block** (256 weights / 144 bytes / 4.5 bpw, K-quant scale packing), so community GGUFs load directly — but the kernels are ours:

- one warp per output row, `dp4a` int8 dots (Turing INT32 path, 4× INT8 throughput), fp32 accumulation, exact per-sub-block scale handling — bit-compatible with the AVX2 path and the host reference
- pointer-table batched GEMV fuses **all resident experts of a layer into ONE launch** (3 launches per MoE instead of 3×hits)
- per-layer decode CUDA graphs: fixed addresses, per-step values arrive through device parameter blocks (`SeqParams{pos, n_tokens}`)

## Build

```bash
./setup.sh                      # or:
cmake -B build -DCMAKE_BUILD_TYPE=Release   # CUDA sm_75 + AVX2 auto-detected
cmake --build build -j
ctest --test-dir build          # host-side Q4 contract tests (no GPU needed)
```

Requirements: CUDA toolkit 12.x (compute capability 7.5+), CMake ≥ 3.24, GCC/Clang with C++20. The AVX2 CPU pool needs AVX2/FMA (any x86-64 server since ~2013).

## Run (target: single T4)

```bash
# 1. convert (or grab a community Sage-A1.5 Q4_K_M GGUF)
python tools/convert_hf_to_sager.py /path/to/InternScience/Sage-A1.5 sage-a15-q4_k_m.gguf --workers 16

# 2. serve (OpenAI-compatible + web UI at http://localhost:8000)
python serve/server.py --model sage-a15-q4_k_m.gguf --vram-mb 10240

# 3. benchmark
python bench/bench.py --host http://localhost:8000 --prompt-len 4096 --gen 256
```

Recommended T4 profile (why these numbers: `docs/t4-guide.md`):

```
ctx 16384 · KV q8 · chunked prefill 8192 · hot-expert pool 10 GB
→ ~3900 of 6144 experts hot-resident (63%), >90% hit rate on agentic workloads
→ static VRAM ≈ 1.5 GB weights + 0.9 GB KV + 10 GB pool + <1 GB scratch ≈ 13 GB
```

Sampling defaults follow the Sage-A1.5 model card: `temp 1.0 · top_k 20 · top_p 0.95 · presence 1.5`.

## Repository layout

```
include/sager/          public headers (kernels, core, spec, util)
src/kernels/cuda/       sm_75 kernels (.cu)
src/kernels/cpu/        AVX2 pool
src/core/               loader, expert cache, session, verify, mtp, conversation
src/spec/               controller, suffix drafter
src/prefill/            chunked prefill
src/program/            sager_engine (JSON-lines resident process)
serve/                  FastAPI server + web UI (chat.html)
tools/                  converter, validator, chat CLI
tests/                  host-side Q4 contract tests
docs/                   architecture, T4 guide, performance, roadmap
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — the three-tier memory model, the graph-capture contract, the CPU-pool clauses C1–C6, defensive systems design
- [docs/t4-guide.md](docs/t4-guide.md) — VRAM budget arithmetic, tuning knobs, mrope/partial-rotary notes for Sage-A1.5
- [docs/performance.md](docs/performance.md) — what to measure, what to expect, and the honest v0.1 limitations
- [docs/roadmap.md](docs/roadmap.md) — SSD expert tier, batched MoE prefill, sampling-aware verify

## License

MIT — see [LICENSE](LICENSE).

---

# SAGER（中文）

**受 Strata 启发的单卡推理引擎：让 [Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5)（35B MoE、约 3B 激活）以 4-bit 流畅跑在一张 NVIDIA Tesla T4（16 GB）上。**

SAGER 把 MoE 的稀疏性当作存储问题来解：T4 装不下 6144 个专家，但每个 token 只用 8 个。我们把常用的专家留在显存（流理台），其余放进 pinned 内存（储藏室），再冷门的上 SSD（roadmap）。这套三层专家存储 + 自研 sm_75 4-bit 内核，就是 SAGER 的全部秘密。

## 亮点（无外部推理框架，全套自研）

- **自研 Q4_K 内核（CUDA sm_75 + AVX2 双路径）**：warp-per-row dp4a 整型点积、fp32 累加、逐 sub-block 精确 scale；指针表批处理 GEMV 把一层所有命中专家融合成一次 launch；逐层解码 CUDA 图（固定地址 + 设备参数块传值）。
- **专家热缓存**：指数衰减频率打分 + 预算内 rebalance + 使用画像持久化（重启跳过冷启动）。
- **MTP 投机解码**：用 Sage-A1.5 自带的 MTP 头一次起草 3 个 token，批量贪心验证逐位一致；成本模型学到的接受率先行门控，亏不起就不投机。
- **CPU 专家池**：未命中专家从 pinned 内存走 AVX2 点积，与 GPU 门铃协议重叠（合约 C1–C6）；CPU 永不施加路由权重，结构上杜绝 CPU/GPU 权重错配。
- **KV 复用 + 分块预填充**：q8 KV（同显存上下文翻倍）、8192 分块、逐轮 KV 检查点，追问只补算增量。
- **格式兼容**：磁盘格式与标准 Q4_K 超级块逐字节兼容，社区 GGUF 直接加载；内核、量化工具、验证器全套自研。

## 快速开始

```bash
./setup.sh                    # cmake + build + ctest
python tools/convert_hf_to_sager.py <HF目录> sage-a15-q4_k_m.gguf --workers 16
python serve/server.py --model sage-a15-q4_k_m.gguf --vram-mb 10240
# 浏览器打开 http://localhost:8000 即可对话；或
python tools/chat.py --host http://localhost:8000
```

T4 推荐配置与显存预算、调参说明见 [docs/t4-guide.md](docs/t4-guide.md)；架构与合约见 [docs/architecture.md](docs/architecture.md)。

## 许可

MIT。
