# Colab one-click serving stack (Sage-A1.5 + ZAgent)

Fresh-VM bootstrap for the whole stack on a Google Colab T4 instance:

```
HF GGUF (21.7 GB) ──► sager_engine (200k ctx) ──► OpenAI API :8000 ──► aitun public URL
                                                            ▲
                                             ZAgent (:8384) ┘ (LLM client + web chat + tools)
```

## One-click

```bash
bash colab/oneclick.sh          # runs stages 1-4 in order, all idempotent
```

Stages (also runnable standalone):

| script | does | marker |
|---|---|---|
| `01_bootstrap.sh` | clone repo (if needed), download GGUF + tokenizer from HF, cmake build | `/content/BOOT1_DONE` |
| `02_serve.sh` | start `serve/server.py` with **200k ctx** + expert-pool env (first load builds the 18 GB pool) | `/content/SERVE_DONE` |
| `03_tunnel.sh` | map the API port to the public internet via a **registered aitun tunnel** (needs `AITUN_API_KEY`) | prints public URL |
| `04_zagent.sh` | install ZAgent (`samaidev/zagent_r`), fix the pip-wrapper binary path, start web UI on :8384 pointed at the local API | `/content/ZAGENT_DONE` |

## Environment variables (keys are NEVER stored in this repo)

| var | default | meaning |
|---|---|---|
| `AITUN_API_KEY` | *(required for 03)* | aitun.cc registered key, e.g. `ait_...` |
| `SAGER_CTX` | `204800` | engine context (200k) |
| `SAGER_VRAM_MB` | `8192` | hot-expert VRAM budget (lowered to leave room for 200k KV) |
| `SAGER_PORT` | `8000` | OpenAI API port |
| `SAGER_LLM_KEY` | `sk-sager-local` | API key handed to ZAgent |
| `ZAGENT_PORT` | `8384` | ZAgent web UI port |

## After boot

- OpenAI API: `http://<vm>:8000/v1/chat/completions` (+ public aitun URL from stage 03)
- ZAgent web chat: `http://<vm>:8384` — assign tasks in chat; or drive it directly:

```bash
curl -s http://127.0.0.1:8384/api/conversations
curl -s -N http://127.0.0.1:8384/api/chat/send/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"搜索并整理今天的3大AI新闻","conv_id":"<conv_id>","model_index":0,"mode":"agent"}'
```

- Server logs: `/content/logs/serve.log`, engine protocol errors there too.
- Speed numbers come with every non-streaming response: `sager.prefill_tps` / `sager.decode_tps`.

## Artifacts (VM-local)

| path | what |
|---|---|
| `/content/Sage-A1.5-Q4_K_M.gguf` | model, HF `InternScience/Sage-A1.5-Q4_K_M-GGUF` |
| `/content/sage_pool.bin` | expert pool, built by the engine on first load (18.1 GB) |
| `/content/sage_tokenizer/tokenizer.json` | tokenizer, HF `InternScience/Sage-A1.5` |
| `/content/logs/*.log` | boot/dl/build/serve/tunnel/zagent logs |
