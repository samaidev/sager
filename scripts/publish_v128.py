#!/usr/bin/env python3
# publish_v128.py - publish the v0.12.8 build artifacts:
#   1. GitHub release asset (samaidev/sager, tag sager-v0.12.8-t4)
#   2. HuggingFace mirror (tchbcb/sager-t4-cuda-artifacts)
#   3. ModelScope mirror (luoyunlan168/sager-t4-cuda-artifacts)
#
#   python3 scripts/publish_v128.py /path/sager_v0128_t4_cuda.tar.gz [--gh-only|--skip-gh]
#   Optional env:
#     PUBLISH_POOL=/path/pool.bin   -> upload the expert pool (18.1 GB) to HF+MS
#     PUBLISH_GGUF=/path/model.gguf -> upload the reconverted GGUF (21.7 GB) to HF+MS
#
# The big files exceed the GitHub release asset limit and must go through the
# HF/MS mirrors; both uploads are resumable chunked uploads inside the SDKs.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

GH_TOKEN = os.environ.get("GHTOKEN", "")
HF_TOKEN = os.environ.get("HFTOKEN", "")
MS_TOKEN = os.environ.get("MSTOKEN", "")
GH_REPO = "samaidev/sager"
TAG = "sager-v0.12.8-t4"
TITLE = "sager v0.12.8 - T4 CUDA binaries (word-soup roots 5-16 fixed + converter rebuilt)"
HF_REPO = "tchbcb/sager-t4-cuda-artifacts"
MS_REPO = "luoyunlan168/sager-t4-cuda-artifacts"

BODY = """v0.12.8 patchset (T4 / sm_75). Engine binaries built at commit 768e503
(diag-kv tip; includes every fix from the word-soup investigation, root causes
5 through 16) + the release scripts commit.

Why this release matters: BENCH_RESULTS_v012.md already showed v0.12 perf
(pp512 ~1000 tok/s, tg128 ~40 tok/s on T4), but every engine since v0.1
produced word soup on the real model. The WORKLOG.md investigation found the
real root cause: the InternScience/Sage-A1.5-Q4_K_M-GGUF weights themselves
are corrupt (ssm_conv1d garbage past a fixed boundary, ssm_a stride-2
scramble, blk0 ssm_out corr 0.05 vs HF safetensors). Fixes 1-14 were real
engine bugs, but the soup was baked into the weights.

Fixed in this release (highlights; full chain in WORKLOG.md):
- v0.12.1 fused per-head q|gate deinterleave (root 5)
- v0.12.2 q8 KV scale half->uint16 truncation, scale always 0 (root 6)
- v0.12.3 dense TC scratch overlapped the MoE TC expert ring (root 7)
- v0.12.4 shared-expert combine weight never written on device paths (root 8)
- v0.12.5 quantize_q8_k_n grid shrank by 256x (root 10)
- v0.12.6 absent-expert CPU patch dead on graph paths (root 11)
- v0.12.7 decode MoE read LAYER 0's residency for every layer (root 13) +
  eager mailbox handshake ordered after the down GEMM + combine RMW race
- v0.12.8 GDN q/k RMSNorm never applied + A_log pre-negation (root 14)
- converter rebuilt for the real checkpoint layout (in_proj_qkv/z/a/b),
  raw bf16 safetensors reader, streaming write + in-flight shard release
- roots 15-16: GGUF weights-load blockers for converter output

IMPORTANT (artifacts):
- The engine binaries in this release are sm_75 builds verified with the
  host-side ctest suite. Run tests/test_cuda_q4 + bench/bench_v8.py on the
  T4 to reproduce the numbers (scripts/colab_v128_run.sh automates the whole
  chain: reconvert GGUF -> rebuild pool -> quality gate -> bench -> publish).
- The 18.1 GB expert pool and the 21.7 GB GGUF published BEFORE this release
  mirror the corrupt InternScience weights; rebuild both from HF safetensors
  with the bundled converter and upload them to the HF/MS mirrors
  (PUBLISH_POOL / PUBLISH_GGUF). Do not use the old copies for quality runs.
"""


def gh_req(url, method="GET", data=None, headers=None,
           ctype="application/json"):
    h = {"Accept": "application/vnd.github+json",
         "Authorization": f"Bearer {GH_TOKEN}",
         "X-GitHub-Api-Version": "2022-11-28"}
    if headers:
        h.update(headers)
    if ctype:
        h.setdefault("Content-Type", ctype)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = r.read()
    return json.loads(body) if body and ctype == "application/json" else body


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh_release(path):
    assert GH_TOKEN, "GHTOKEN missing"
    rel = None
    try:
        rel = gh_req(f"https://api.github.com/repos/{GH_REPO}/releases/tags/{TAG}")
    except Exception:
        print(f"release {TAG} not found - creating")
    if not rel:
        rel = gh_req(f"https://api.github.com/repos/{GH_REPO}/releases", "POST",
                     json.dumps({"tag_name": TAG,
                                 "target_commitish": "rel-v0.12.8",
                                 "name": TITLE, "body": BODY,
                                 "prerelease": False}).encode())
    name = os.path.basename(path)
    url = rel["upload_url"].split("{")[0] + f"?name={name}"
    size = os.path.getsize(path)
    print(f"GH: uploading {path} ({size} B) to release {TAG}")
    with open(path, "rb") as f:
        data = f.read()
    resp = gh_req(url, "POST", data, headers={"Content-Length": str(size)},
                  ctype="application/octet-stream")
    print("GH asset:", resp.get("browser_download_url"))
    return rel.get("html_url", "")


def hf_upload(path, name=None):
    from huggingface_hub import HfApi
    assert HF_TOKEN, "HFTOKEN missing"
    api = HfApi(token=HF_TOKEN)
    name = name or os.path.basename(path)
    print(f"HF: uploading {path} -> {HF_REPO}/{name}")
    api.upload_file(path_or_fileobj=path, path_in_repo=name,
                    repo_id=HF_REPO, repo_type="model",
                    commit_message=f"v0.12.8 artifacts: {name}")
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{name}"


def ms_upload(path, name=None):
    from modelscope.hub.api import HubApi
    assert MS_TOKEN, "MSTOKEN missing"
    api = HubApi()
    api.login(MS_TOKEN)
    name = name or os.path.basename(path)
    print(f"MS: uploading {path} -> {MS_REPO}/{name}")
    api.upload_file(repo_id=MS_REPO,
                    src_path=path, dest_path=name,
                    commit_message=f"v0.12.8 artifacts: {name}")
    return f"https://www.modelscope.cn/models/{MS_REPO}/resolve/master/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball")
    ap.add_argument("--gh-only", action="store_true")
    ap.add_argument("--skip-gh", action="store_true")
    args = ap.parse_args()
    path = args.tarball
    assert os.path.isfile(path), f"missing {path}"
    print("sha256:", sha256_file(path))

    urls = {}
    if not args.skip_gh:
        urls["gh"] = gh_release(path)
    if args.gh_only:
        print(json.dumps(urls, indent=2))
        return
    try:
        urls["hf"] = hf_upload(path)
    except Exception as e:
        print("HF upload failed:", e)
    try:
        urls["ms"] = ms_upload(path)
    except Exception as e:
        print("MS upload failed:", e)

    pool = os.environ.get("PUBLISH_POOL", "")
    if pool and os.path.isfile(pool):
        name = "Sage-A1.5-Q4_K_M-expert-pool-v0128.bin"
        try:
            urls["hf_pool"] = hf_upload(pool, name)
        except Exception as e:
            print("HF pool upload failed:", e)
        try:
            urls["ms_pool"] = ms_upload(pool, name)
        except Exception as e:
            print("MS pool upload failed:", e)

    gguf = os.environ.get("PUBLISH_GGUF", "")
    if gguf and os.path.isfile(gguf):
        name = "Sage-A1.5-Q4_K_M.gguf"
        try:
            urls["hf_gguf"] = hf_upload(gguf, name)
        except Exception as e:
            print("HF gguf upload failed:", e)
        try:
            urls["ms_gguf"] = ms_upload(gguf, name)
        except Exception as e:
            print("MS gguf upload failed:", e)

    print(json.dumps(urls, indent=2))


if __name__ == "__main__":
    main()
