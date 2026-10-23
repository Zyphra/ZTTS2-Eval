#!/usr/bin/env python3
"""Validate Delta panels: can a small panel tell two nearby checkpoints apart?

Delta panels answer a different question from Mini panels, so they need a different test.
A Mini panel is judged on how close its *level* is to the full benchmark. A Delta panel is
judged on the **paired difference** between two systems, where the level does not matter
at all -- and where pairing removes the per-item variance that dominates an unpaired
comparison, because both systems spoke the same utterances with the same prompts (the same
reason `scripts/matched_pair_spk.py` can resolve ~0.3-point effects the raw means cannot).

Two things are reported:

  MDD    minimum detectable difference at 80% power, two-sided alpha=0.05, derived from
         the panel's own matched-pair bootstrap SE. This is the honest headline: "on this
         panel you can detect a change of at least X". It needs no near-pair to exist.
  sign / magnitude accuracy on the real near-pairs the repo happens to contain
         (zonos2 accurate-vs-expressive, the two qwen3 sizes, omnivoice with and without
         reference text, the moss variants).

Panels are compared against same-size random and strat_meta panels, since a Delta panel
is only worth shipping if selecting on discrimination actually buys power.

    python scripts/coreset/validate_delta.py --set itw --ks 50,100,200,400
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Fingerprints, Matrix, weighted_boot  # noqa: E402
from panel_select import build_panel  # noqa: E402
from validate import MIN_LEAF_COVERAGE  # noqa: E402

# Pairs of systems from the same family -- the closest stand-ins this repo has for two
# checkpoints of one model, which is what a Delta panel is meant to compare.
NEAR_PAIRS = [
    ("zonos2_accurate", "zonos2_expressive"),
    ("qwen3-tts-1.7b", "qwen3-tts-0.6b"),
    ("omnivoice-reftext", "omnivoice"),
    ("moss-tts", "moss-tts-v1.5"),
    ("moss-tts-v1.5", "moss-tts-local-v1.5"),
]
# 80% power, two-sided alpha=0.05: |delta| must exceed (z_{0.975} + z_{0.80}) * SE.
POWER_Z = 1.959964 + 0.841621
BOOT = 2000
SEED = 20260717


def shared_langs(M, a, b, panel, metric):
    ca = set(M.covered_langs(a, metric=metric, min_coverage=MIN_LEAF_COVERAGE))
    cb = set(M.covered_langs(b, metric=metric, min_coverage=MIN_LEAF_COVERAGE))
    return [lg for lg in panel["langs"] if lg in ca and lg in cb]


def true_delta(M, a, b, metric, langs):
    return M.true_score(a, metric, langs) - M.true_score(b, metric, langs)


def panel_delta(M, a, b, metric, panel, langs, boot=BOOT, seed=SEED):
    """Matched-pair weighted difference + its bootstrap SE."""
    pieces = []
    for lg in langs:
        sel = panel["langs"][lg]
        idx = np.asarray(sel["idx"], dtype=int)
        w = np.asarray(sel["w"], dtype=float)
        ya, yb = M.m[metric][a, idx], M.m[metric][b, idx]
        ok = np.isfinite(ya) & np.isfinite(yb)
        if ok.any():
            pieces.append((ya[ok] - yb[ok], w[ok]))
    if not pieces:
        return float("nan"), float("nan")
    est = float(np.mean([np.sum(w * d) / np.sum(w) for d, w in pieces]))
    draws = weighted_boot(pieces, boot=boot, seed=seed)
    return est, float(np.std(draws, ddof=1))


def measure(set_name, ks, selectors, metric="wer_clip", seeds=(0, 1, 2)):
    """Return (rows, pairs). Each row: (selector, k, MDD, sign_acc, mag_err, n)."""
    M = Matrix(set_name)
    F = Fingerprints(set_name, matrix=M)  # fail loudly on a stale fp_<set>.npz
    have = set(M.systems)
    pairs = [(a, b) for a, b in NEAR_PAIRS if a in have and b in have]

    rows = []
    for selector in selectors:
        for k in ks:
            mdds, signs, errs, n = [], [], [], 0
            for seed in seeds if selector != "delta" else (0,):
                panel = build_panel(M, F, selector, k, kind="delta", seed=seed)
                for a_name, b_name in pairs:
                    a, b = M.si(a_name), M.si(b_name)
                    langs = shared_langs(M, a, b, panel, metric)
                    if len(langs) < 2:
                        continue
                    td = true_delta(M, a, b, metric, langs)
                    pd, se = panel_delta(M, a, b, metric, panel, langs)
                    if not (np.isfinite(pd) and np.isfinite(se) and se > 0):
                        continue
                    mdds.append(POWER_Z * se)
                    signs.append(np.sign(td) == np.sign(pd))
                    errs.append(abs(pd - td))
                    n += 1
            if mdds:
                rows.append((selector, k, float(np.mean(mdds)), float(np.mean(signs)),
                             float(np.mean(errs)), n))
    return rows, pairs


def run(set_name, ks, selectors, metric="wer_clip", seeds=(0, 1, 2)):
    rows, pairs = measure(set_name, ks, selectors, metric=metric, seeds=seeds)

    print(f"=== {set_name}: Delta panel power on {metric} ===")
    print(f"MDD = minimum detectable difference at 80% power (two-sided alpha=0.05),\n"
          f"from the panel's matched-pair bootstrap SE. Lower is better.\n")
    print(f"    {'selector':16s}{'k':>6s}{'MDD':>9s}{'sign ok':>9s}{'|err|':>9s}"
          f"{'n pairs':>9s}")
    for selector, k, mdd, sign, err, n in rows:
        print(f"    {selector:16s}{k:6d}{mdd:9.3f}{sign * 100:8.0f}%{err:9.3f}{n:9d}")

    print(f"\n  near-pairs used ({len(pairs)}): "
          + ", ".join(f"{a} vs {b}" for a, b in pairs))
    print("  These pairs are further apart than two consecutive checkpoints would be, so\n"
          "  the two columns mislead in OPPOSITE directions and should be read that way:\n"
          "    sign ok — optimistic. Large true gaps are easy; it says little about the\n"
          "              small differences a Delta panel actually exists to catch.\n"
          "    MDD     — conservative. It scales with the spread of the paired per-item\n"
          "              differences, and two nearby checkpoints differ far less, and far\n"
          "              more consistently, than these pairs do. Treat it as an upper\n"
          "              bound on what the panel can resolve.\n"
          "  |err| is the magnitude error, and a Delta panel is deliberately biased there:\n"
          "  over-sampling discriminative items inflates the measured difference. Use a\n"
          "  Delta panel to decide DIRECTION, and a Mini panel to quantify.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["clean", "itw"])
    ap.add_argument("--ks", default="50,100,200,400")
    ap.add_argument("--selectors", default="delta,random_perlang,strat_meta")
    ap.add_argument("--metric", default="wer_clip", choices=["wer_clip", "spk", "msr"])
    args = ap.parse_args()
    run(args.set, [int(x) for x in args.ks.split(",")], args.selectors.split(","),
        metric=args.metric)


if __name__ == "__main__":
    main()
