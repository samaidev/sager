#!/usr/bin/env python3
# publish_v08.py - publish the v0.8 build artifacts:
#   1. GitHub release asset (samaidev/sager, tag sager-v0.8.0-t4)
#   2. HuggingFace mirror (tchbcb/sager-t4-cuda-artifacts)
#   3. ModelScope mirror (luoyunlan168/sager-t4-cuda-artifacts)
#
#   python3 publish_v08.py /path/sager_v08_t4_cuda.tar.gz [--gh-only|--skip-gh]
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

GH_TOKEN = os.environ.get("GHTOKEN", "")
HF_TOKEN = os.environ.get("HFTOKEN", "")
MS_TOKEN = os.environ.get("MSTOKEN", "")
GH_REPO = "samaidev/sager"
TAG = "sager-v0.8.0-t4"
TITLE = "sager v0.8.0 - T4 CUDA binaries (prefill MMQ + decode fusions)"
HF_REPO = "tchbcb/sager-t4-cuda-artifacts"
MS_REPO = "luoyunlan168/sager-t4-cuda-artifacts"


def gh_req(url: str, method: str = "GET", data=None, headers=None,
           ctype: str = "application/json"):
    h = {"Accept": "application/vnd.github+json",
         "Authorization": f"Bearer {GH_TOKEN}", "X-GitHub-Api-Version": "2022-11-28"}
    if headers:
        h.update(headers)
    if ctype:
        h.setdefault("Content-Type", ctype)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    return json.loads(body) if body and ctype == "application/json" else body


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh_release(path: str) -> str:
    """Create (or reuse) the release and upload the tarball. Returns URL."""
    assert GH_TOKEN, "GHTOKEN missing"
    rel = None
    try:
        rel = gh_req(f"https://api.github.com/repos/{GH_REPO}/releases/tags/{TAG}")
    except Exception as e:
        print(f"release {TAG} not found ({e}) - creating")
    if not rel:
        rel = gh_req(f"https://api.github.com/repos/{GH_REPO}/releases",
                     "POST",
                     json.dumps({"tag_name": TAG, "target_commitish": "v0.8-opt",
                                 "name": TITLE,
                                 "body": ("v0.8 patchset: prefill MMQ GEMM, decode "
                                          "kernel fusions, GDN ffn-norm fix. See "
                                          "BENCH_RESULTS_v08.md for measured T4 "
                                          "numbers. Contents: BUILD_INFO_v8.txt, "
                                          "raw bench JSON + logs."),
                                 "prerelease": False}).encode())
    url = rel["upload_url"].split("{")[0] + f"?name={os.path.basename(path)}"
    size = os.path.getsize(path)
    print(f"GH: uploading {path} ({size} B) to release {TAG}")
    with open(path, "rb") as f:
        data = f.read()
    resp = gh_req(url, "POST", data,
                  headers={"Content-Length": str(size)},
                  ctype="application/octet-stream")
    print("GH asset:", resp.get("browser_download_url"))
    return rel.get("html_url", "")


def hf_upload(path: str) -> str:
    assert HF_TOKEN, "HFTOKEN missing"
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    name = os.path.basename(path)
    res = api.upload_file(path_or_fileobj=path, path_in_repo=name,
                          repo_id=HF_REPO, repo_type="model",
                          commit_message="v0.8.0 T4 CUDA artifacts")
    print("HF:", res)
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{name}"


def ms_upload(path: str) -> str:
    assert MS_TOKEN, "MSTOKEN missing"
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(MS_TOKEN)
    name = os.path.basename(path)
    api.push_model(model_id=MS_REPO, model_path=os.path.dirname(path) or ".",
                   commit_message="v0.8.0 T4 CUDA artifacts")
    # push_model uploads a directory; fall back to single-file put when needed
    print("MS upload requested for", name)
    return f"https://www.modelscope.cn/models/{MS_REPO}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball")
    ap.add_argument("--skip-gh", action="store_true")
    ap.add_argument("--gh-only", action="store_true")
    a = ap.parse_args()
    p = a.tarball
    assert os.path.isfile(p), p
    print("sha256:", sha256_file(p))

    results = {}
    if not a.skip_gh:
        results["github"] = gh_release(p)
    if not a.gh_only:
        try:
            results["hf"] = hf_upload(p)
        except Exception as e:
            print("HF upload failed:", repr(e))
            results["hf"] = f"FAILED: {e!r}"
        try:
            results["modelscope"] = ms_upload(p)
        except Exception as e:
            print("MS upload failed:", repr(e))
            results["modelscope"] = f"FAILED: {e!r}"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
