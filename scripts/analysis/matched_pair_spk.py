"""Matched-pair bootstrap on per-utterance speaker-similarity scores.

Comparing two systems by their mean spk throws away the pairing: both systems spoke
the SAME utterances with the SAME reference clips, and most of the variance is
per-utterance (some refs are just easier to clone). Differencing within uttid removes
that variance entirely, so a ~0.3-point effect that the two means cannot separate is
resolvable here.

Usage:
    python scripts/analysis/matched_pair_spk.py A.txt B.txt [C.txt ...] [--labels a,b,c]

Each file is a spk_simi_scores.txt: "<uttid> <x> <score>" per line, plus a trailing
"avg" row (skipped). Every ordered pair is reported as mean delta, bootstrap CI95 and
win% (fraction of utterances where the later system scores higher, ties split evenly —
a sign-consistency readout, not an effect size) over the uttids the pair has in common.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
# Canonical spk_simi_scores.txt parser -- do NOT re-implement it here. The local
# copy this replaced accepted 2-field lines (parse_spk requires the full
# '<uid> <ref> <hyp>' 3-field row), a silent semantic divergence.
from utils.per_uid import parse_spk  # noqa: E402


def load(path):
    scores = parse_spk(path)
    if not scores:
        sys.exit(f"error: no per-utterance scores parsed from {path}")
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="two or more spk_simi_scores.txt")
    ap.add_argument("--labels", default="",
                    help="comma-separated display names (default: parent dir names)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if len(args.files) < 2:
        ap.error("need at least two score files to pair")
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        # results/<model>/<task>/<lang>/spk_simi_scores.txt -> <model>
        labels = [os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(p))))) for p in args.files]
    if len(labels) != len(args.files):
        ap.error(f"{len(labels)} labels for {len(args.files)} files")

    sets = [load(p) for p in args.files]
    for lab, s in zip(labels, sets):
        vals = np.fromiter(s.values(), dtype=float)
        print(f"{lab:<28} n={len(vals):<5} mean={vals.mean():.2f}")

    rng = np.random.default_rng(args.seed)
    print(f"\nmatched-pair bootstrap ({args.resamples} resamples, seed {args.seed})")
    print("win% = sign consistency (share of utterances where the later system scores "
          "higher,\nties split evenly), NOT an effect size — a 0.01-point edge on every "
          "utterance is win=100%.")
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            common = sorted(set(sets[i]) & set(sets[j]))
            if not common:
                print(f"{labels[j]} - {labels[i]}: no shared uttids")
                continue
            d = np.array([sets[j][u] - sets[i][u] for u in common])
            idx = rng.integers(0, len(d), size=(args.resamples, len(d)))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            win = float((d > 0).mean() + 0.5 * (d == 0).mean()) * 100
            print(f"{labels[j] + ' - ' + labels[i]:<38} n={len(d):<5} "
                  f"delta={d.mean():+6.2f}  CI95[{lo:+.2f},{hi:+.2f}]  win={win:.0f}%")


if __name__ == "__main__":
    main()
