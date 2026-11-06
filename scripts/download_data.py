#!/usr/bin/env python
"""Fetch the ZTTS2-Eval audio from the Hugging Face Hub into data/.

The metadata (text, prompt_text, prompt_wav.scp, target_wav.scp, ...) is tracked in
git; the audio is the dataset Zyphra/ZTTS2-Eval: the prompt wavs
(data/<task>/<lang>/waveform/, 24 GB in total) and, for in-the-wild, the
held-out target recordings (data/in-the-wild/<lang>/target_wav/) behind the
speaker-similarity primary score. This script downloads them to the paths the
committed .scp files already point at, then checks that every listed wav resolves.

Usage:
    python scripts/download_data.py                              # both sets, all languages
    python scripts/download_data.py --tasks in-the-wild --langs en zh
    python scripts/download_data.py --revision <commit>          # pin a dataset revision

Authentication: --token, else HF_TOKEN / a cached `hf auth login`. Downloads resume,
and files already present are skipped (metadata lives in data/.cache/huggingface/).
Exits 1 if any wav listed in a prompt_wav.scp / target_wav.scp is still missing.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TASKS = ("clean", "in-the-wild")


def langs_of(task):
    """Languages of a task: the subdirs of data/<task> that carry a prompt_wav.scp."""
    return sorted(p.parent.name for p in (DATA / task).glob("*/prompt_wav.scp"))


def missing_prompts(task, lang):
    """Prompt wav paths listed in data/<task>/<lang>/prompt_wav.scp that are not files."""
    return [path for path in _scp_paths(task, lang, "prompt_wav.scp") if not Path(path).is_file()]


def missing_targets(task, lang):
    """Target wav paths listed in data/<task>/<lang>/target_wav.scp that are not files."""
    return [path for path in _scp_paths(task, lang, "target_wav.scp") if not Path(path).is_file()]


def _scp_paths(task, lang, scp):
    """Wav paths of data/<task>/<lang>/<scp> (empty when absent), repo-relative ones resolved."""
    paths = []
    p = DATA / task / lang / scp
    if not p.is_file():
        return paths
    with open(p, encoding="utf-8") as fh:
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
        allow_patterns=[pat
                        for task, langs in selected.items() for lang in langs
                        for pat in ([f"{task}/{lang}/waveform/*.wav"]
                                    + ([f"{task}/{lang}/target_wav/*.wav"]
                                       if task == "in-the-wild" else []))],
        token=args.token,
        max_workers=args.max_workers,
    )

    n_missing = 0
    for task, langs in selected.items():
        for lang in langs:
            total = len(_scp_paths(task, lang, "prompt_wav.scp"))
            missing = missing_prompts(task, lang)
            n_missing += len(missing)
            print(f"{task}/{lang}: {total - len(missing)}/{total} prompt wavs present")
            t_total = len(_scp_paths(task, lang, "target_wav.scp"))
            if t_total:
                t_missing = missing_targets(task, lang)
                n_missing += len(t_missing)
                print(f"{task}/{lang}: {t_total - len(t_missing)}/{t_total} target wavs present")
    if n_missing:
        print(f"missing {n_missing} wavs", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
