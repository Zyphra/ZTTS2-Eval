#!/usr/bin/env python3
"""Publish the benchmark release audio changes to the HF dataset utils.benchmark.HF_REPO_ID.

  1. Upload the held-out ITW target recordings
     (data/in-the-wild/<lang>/target_wav/*.wav).
  2. Delete the prompt wavs of the items removed by the prompt/target near-duplicate
     filter (data/near_duplicates_removed.json), so a fresh `download_data.py` matches the
     committed prompt_wav.scp files.

Idempotent: uploads skip files the hub already has, and the deletion only touches paths
still present in the repo. Needs HF_TOKEN with write access.

    HF_TOKEN=... python scripts/prep/upload_release_data.py
"""
import json
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.benchmark import HF_REPO_ID as REPO_ID  # noqa: E402

DATA = Path(REPO) / "data"


def removed_prompt_paths():
    """Repo-relative-to-data prompt wav paths named in near_duplicates_removed.json."""
    log = json.loads((DATA / "near_duplicates_removed.json").read_text(encoding="utf-8"))
    out = set()
    for entries in log["removed"].values():
        for e in entries:
            p = e.get("prompt_wav", "")
            if not p:
                continue
            p = os.path.normpath(p)                   # collapses the Clean '//' paths
            out.add(p[len("data/"):] if p.startswith("data" + os.sep) else p)
    return out


def main():
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is required")
    from huggingface_hub import CommitOperationDelete, HfApi
    api = HfApi()

    api.upload_large_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(DATA),
                            allow_patterns=["in-the-wild/*/target_wav/*.wav"])

    in_repo = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    stale = sorted(removed_prompt_paths() & in_repo)
    if stale:
        api.create_commit(
            repo_id=REPO_ID, repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=p) for p in stale],
            commit_message="ZTTS2-Eval: drop prompt/target near-duplicate prompts")
        print(f"deleted {len(stale)} near-duplicate prompt wavs")
    else:
        print("no near-duplicate prompt wavs left on the hub")

    print("dataset commit:", api.dataset_info(REPO_ID).sha)


if __name__ == "__main__":
    main()
