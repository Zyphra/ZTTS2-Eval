#!/usr/bin/env python
"""Fetch the ZTTS2-Eval prompt audio from the Hugging Face Hub into data/.

The metadata (text, prompt_text, prompt_wav.scp, ...) is tracked in git; the prompt
wavs (data/<task>/<lang>/waveform/, 24 GB in total) are the dataset
Zyphra/ZTTS2-Eval. This script downloads them to the paths the committed
prompt_wav.scp files already point at, then checks that every prompt resolves.

Usage:
    python scripts/download_data.py                              # both sets, all languages
    python scripts/download_data.py --tasks zero_shot_voxblink --langs en zh
    python scripts/download_data.py --revision <commit>          # pin a dataset revision

Authentication: --token, else HF_TOKEN / a cached `hf auth login`. Downloads resume,
and files already present are skipped (metadata lives in data/.cache/huggingface/).
Exits 1 if any prompt wav listed in a prompt_wav.scp is still missing.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TASKS = ("zero_shot", "zero_shot_voxblink")


def langs_of(task):
    """Languages of a task: the subdirs of data/<task> that carry a prompt_wav.scp."""
    return sorted(p.parent.name for p in (DATA / task).glob("*/prompt_wav.scp"))


def missing_prompts(task, lang):
    """Prompt wav paths listed in data/<task>/<lang>/prompt_wav.scp that are not files."""
    return [path for path in _prompt_paths(task, lang) if not Path(path).is_file()]


def _prompt_paths(task, lang):
    """Prompt wav paths of data/<task>/<lang>/prompt_wav.scp, repo-relative ones resolved."""
    paths = []
    with open(DATA / task / lang / "prompt_wav.scp", encoding="utf-8") as fh:
        for line in fh:
            _, _, path = line.strip().partition(" ")
            if path:
                paths.append(str(Path(path) if Path(path).is_absolute() else REPO / path))
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="Zyphra/ZTTS2-Eval",
                    help="Hugging Face dataset repo (default %(default)s)")
    ap.add_argument("--revision", default=None,
                    help="dataset revision (branch, tag or commit); default = main. "
                         "Pin a commit for reproducibility")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS),
                    help="sets to fetch (default: both)")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="languages to fetch (default: every language of each task)")
    ap.add_argument("--token", default=None,
                    help="Hugging Face token (default: HF_TOKEN / cached login)")
    ap.add_argument("--max-workers", type=int, default=8, help="parallel downloads")
    args = ap.parse_args()

    selected = {}
    for task in args.tasks:
        available = langs_of(task)
        for lang in args.langs or ():
            if lang not in available:
                ap.error(f"{task}: no data/{task}/{lang}/prompt_wav.scp")
        selected[task] = list(args.langs) if args.langs else available

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=DATA,
        allow_patterns=[f"{task}/{lang}/waveform/*.wav"
                        for task, langs in selected.items() for lang in langs],
        token=args.token,
        max_workers=args.max_workers,
    )

    n_missing = 0
    for task, langs in selected.items():
        for lang in langs:
            total = len(_prompt_paths(task, lang))
            missing = missing_prompts(task, lang)
            n_missing += len(missing)
            print(f"{task}/{lang}: {total - len(missing)}/{total} prompt wavs present")
    if n_missing:
        print(f"missing {n_missing} prompt wavs", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
