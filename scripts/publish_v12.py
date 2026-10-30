#!/usr/bin/env python3
# publish_v12.py - publish the v0.12 build artifacts:
#   1. GitHub release asset (samaidev/sager, tag sager-v0.12.0-t4)
#   2. HuggingFace mirror (tchbcb/sager-t4-cuda-artifacts)
#   3. ModelScope mirror (luoyunlan168/sager-t4-cuda-artifacts)
#
#   python3 publish_v12.py /path/sager_v012_t4_cuda.tar.gz [--gh-only|--skip-gh]
#   Optional env: PUBLISH_POOL=/path/expert-pool.bin -> also upload the fresh
#   (fixed-requant) expert pool to HF + MS (too large for GH release assets).
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

GH_TOKEN = os.environ.get("GHTOKEN", "")
HF_TOKEN = os.environ.get("HFTOKEN", "")
MS_TOKEN = os.environ.get("MSTOKEN", "")
GH_REPO = "samaidev/sager"
TAG = "sager-v0.12.4-t4"
TITLE = "sager v0.12.4 - T4 CUDA binaries (word-soup root causes 5-8 fixed)"
HF_REPO = "tchbcb/sager-t4-cuda-artifacts"
MS_REPO = "luoyunlan168/sager-t4-cuda-artifacts"

BODY = """v0.12.4 patchset (T4 / sm_75, CUDA 13.0 build):

Correctness (word-soup root-cause chain, all CRITICAL - see WORKLOG.md):
- RC6: q8 KV cache scale bytes were stored via __half->uint16 implicit
  truncation - scale always 0, attention output exactly zero (d999de7)
- RC7: d_wtc_ dense TC scratch overlapped the MoE TC fp16 expert ring;
  pump stream-all queueing clobbered every qkvz dequant (5120e0c)
- RC8: shared-expert combine weight (route_w_[row][K]) never written on the
  device paths - decode steps and device-prefill rows >= 1 dropped the
  shared expert entirely (the main word-soup culprit) (640c016)
- plus RC1-5: Q4_K nibble order, Q6_K scale offsets, fused q|gate
  deinterleave, GDN conv stride, per-head q|gate of fused q_proj

Contents: sager_engine binary (sm_75), BUILD_INFO_v12.txt, raw bench JSON.
The expert-pool artifact (HF/MS only, 18.1 GB) is rebuilt with the FIXED
requant pipeline - the pre-v012 pool contains garbage down segments and must
not be used with SAGER_EXPERT_POOL_REUSE=1.
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
                                 "target_commitish": "main",
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
                    commit_message=f"v0.12 artifacts: {name}")
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
                    commit_message=f"v0.12 artifacts: {name}")
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
        name = "Sage-A1.5-Q4_K_M-expert-pool-v012-fixed.bin"
        try:
            urls["hf_pool"] = hf_upload(pool, name)
        except Exception as e:
            print("HF pool upload failed:", e)
        try:
            urls["ms_pool"] = ms_upload(pool, name)
        except Exception as e:
            print("MS pool upload failed:", e)
    print(json.dumps(urls, indent=2))


if __name__ == "__main__":
    main()
