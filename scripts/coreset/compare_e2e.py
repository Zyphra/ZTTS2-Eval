#!/usr/bin/env python3
"""Compare a panel run's estimate against the known full-set score, system by system.

This is the readout of scripts/coreset/verify_e2e.sh: for each system it prints the panel
estimate with its bootstrap CI next to the full-benchmark score computed from the complete
matrix, restricted to exactly the languages the panel run covered — and says whether the
true value fell inside the interval.

CI coverage is the property that matters here. An estimate that is off by 0.3 with a CI of
±0.5 is working as designed; one that is off by 0.3 with a CI of ±0.05 is not.

    python scripts/coreset/compare_e2e.py --task clean_mini --systems "zonos2_accurate kokoro"
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from common import Matrix  # noqa: E402
from score_panel import (SET_OF_TASK, boot_ci, collect, read_panel_manifest,  # noqa: E402
                         weighted_macro)

METRICS = ("wer_clip", "spk", "msr")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--systems", required=True, help="space-separated results/ dir names")
    args = ap.parse_args()

    manifest = read_panel_manifest(args.task)
    set_name = SET_OF_TASK[manifest["source_task"]]
    M = Matrix(set_name)

    print(f"\n{'system':22s}{'metric':10s}{'panel':>9s}{'CI95':>20s}{'full set':>10s}"
          f"{'err':>8s}{'in CI':>7s}{'langs':>7s}")
    n_cov = n_tot = n_unscored = 0
    for sysname in args.systems.split():
        decode = os.path.join(REPO_ROOT, "results", sysname, args.task)
        if not os.path.isdir(decode):
            print(f"{sysname:22s}(no panel run found)")
            n_unscored += 1
            continue
        collected = collect(decode, manifest)
        if not collected:
            print(f"{sysname:22s}(no scored leaves)")
            n_unscored += 1
            continue
        try:
            si = M.si(_alias(M, sysname))
        except ValueError:
            print(f"{sysname:22s}(not in the historical matrix — panel estimate only)")
            si = None

        for metric in METRICS:
            est, pieces = weighted_macro(collected, manifest, metric)
            if not np.isfinite(est):
                continue
            langs = [p[0] for p in pieces]
            lo, hi = boot_ci(pieces)
            if si is None:
                print(f"{sysname:22s}{metric:10s}{est:9.3f}"
                      f"{f'[{lo:.3f}, {hi:.3f}]':>20s}{'—':>10s}{'—':>8s}{'—':>7s}{len(langs):7d}")
                continue
            true = M.true_score(si, metric, langs)
            inside = lo <= true <= hi
            n_cov += int(inside)
            n_tot += 1
            print(f"{sysname:22s}{metric:10s}{est:9.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>20s}{true:10.3f}{est - true:+8.3f}"
                  f"{('yes' if inside else 'NO'):>7s}{len(langs):7d}")
    if n_tot:
        print(f"\nCI95 coverage: {n_cov}/{n_tot} estimates contained the full-set value.")
        print("The full-set value is itself computed over exactly the languages the panel "
              "run covered,\nso this compares like with like rather than penalizing "
              "partial language support.")
    if n_unscored:
        # A system with no scored panel leaves means the e2e run verified nothing for
        # it -- that must fail the calling job, not read as a quiet table footnote.
        sys.exit(f"error: {n_unscored} system(s) had no scored panel run under results/ "
                 f"— the e2e check verified nothing for them")


def _alias(M, results_dir_name):
    """Map a results/ directory name to its matrix row key."""
    if results_dir_name in M.systems:
        return results_dir_name
    # build_matrix.py's registry keys drop the omni_ prefix and some suffixes.
    cand = results_dir_name.replace("omni_", "")
    if cand in M.systems:
        return cand
    raise ValueError(results_dir_name)


if __name__ == "__main__":
    main()
