#!/usr/bin/env python3
"""Leave-family-out validation of coreset panels.

The question a coreset must answer is not "can it reconstruct scores for systems it was
built from" -- that is trivially yes -- but "does it predict full-benchmark results for a
genuinely new kind of system". So folds hold out an entire model **family** (all zonos2
variants together, both qwen3 sizes together, ...), select the panel using only the
remaining families, and score the held-out ones.

Reported per (selector, k):

  score MAE          absolute error of the panel estimate, in the metric's own units
  Pearson/Spearman/Kendall   agreement of the panel ranking with the full one
  pairwise sign accuracy vs true gap   the primary readout -- n = pairs x folds, so it
                     has power where a per-fold rank correlation over ~9 systems does not
  top-K membership   does the panel put the same systems in the top K
  worst-language error   the largest per-language error, not just the average
  in-family vs novel-family   error split, to expose selection overfitting

A caution that belongs in the output rather than a footnote: with ~9 independent families
a per-fold rank correlation is nearly meaningless. Conclusions come from the pooled
pairwise statistics.

    python scripts/coreset/validate.py --set itw --ks 200,500,1000,2000
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import stats

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

from scripts.coreset.common import (Fingerprints, Matrix, OUT_DIR, SIGN_GAP_EDGES, check_weights,  # noqa: E402
                    load_validation_rows, mae_cells, mean_se, pairwise_sign_stats)
from scripts.coreset.panel_select import build_panel  # noqa: E402

EVAL_METRICS = ("wer_clip", "spk", "msr")
STOCHASTIC = {"random_global", "random_perlang", "strat_meta", "strat_response", "strat_quantile"}
# Greedy is O(k x N x M) per language; above this it dominates the sweep's runtime for no
# new insight. The cap is reported in the output rather than silently applied.
GREEDY_MAX_K = 1000
MIN_LEAF_COVERAGE = 0.9


def subject_rows(M):
    """Systems to score: the dense 11 plus the partial-coverage rows as novel families."""
    return [i for i, r in enumerate(M.roles) if r in ("system", "partial")]


def eval_langs(M, si, panel, metric):
    """Languages where both the panel and this system have usable data.

    Scoring a system on languages it never ran would measure coverage, not sampling
    error, so the true score is always taken over exactly the languages compared.
    """
    cov = set(M.covered_langs(si, metric=metric, min_coverage=MIN_LEAF_COVERAGE))
    return [lg for lg in panel["langs"] if lg in cov]


def panel_estimate(M, si, metric, panel, langs):
    vals = []
    for lg in langs:
        sel = panel["langs"][lg]
        idx = np.asarray(sel["idx"], dtype=int)
        w = np.asarray(sel["w"], dtype=float)
        y = M.m[metric][si, idx]
        ok = np.isfinite(y)
        if ok.any():
            vals.append(float(np.sum(w[ok] * y[ok]) / np.sum(w[ok])))
    return float(np.mean(vals)) if vals else float("nan")


def per_lang_errors(M, si, metric, panel, langs):
    out = []
    for lg in langs:
        sel = panel["langs"][lg]
        idx = np.asarray(sel["idx"], dtype=int)
        w = np.asarray(sel["w"], dtype=float)
        y = M.m[metric][si, idx]
        ok = np.isfinite(y)
        full = M.m[metric][si, M.cols(lg)]
        full = full[np.isfinite(full)]
        if ok.any() and full.size:
            out.append(abs(float(np.sum(w[ok] * y[ok]) / np.sum(w[ok])) - float(full.mean())))
    return out


def run(set_name, ks, selectors, seeds, out_csv, verbose=True):
    M = Matrix(set_name)
    F = Fingerprints(set_name, matrix=M)  # fail loudly on a stale fp_<set>.npz
    subjects = subject_rows(M)
    dense = [i for i in subjects if M.roles[i] == "system"]
    families = sorted({M.families[i] for i in dense})

    if verbose:
        print(f"[{set_name}] {len(subjects)} subject systems, {len(families)} families "
              f"for leave-family-out: {', '.join(families)}")
        print(f"[{set_name}] NOTE: {len(families)} families is a thin basis -- per-fold rank "
              f"correlations are underpowered; use the pooled pairwise statistics.")

    rows = []
    t0 = time.time()
    fam_set = set(families)
    for fold_family in families:
        train_si = [i for i in dense if M.families[i] != fold_family]
        # "Novel" means the selector never saw this family. That covers the held-out fold
        # AND every family that exists only as a partial-coverage row (kokoro, glm-tts,
        # neutts, vibevoice, indextts): they are never in `dense`, so they are never in
        # any training pool, which makes them the *most* novel systems available -- and
        # exactly the ones a coreset's generalization claim rests on.
        held = [i for i in subjects
                if M.families[i] == fold_family or M.families[i] not in fam_set]
        for selector in selectors:
            use_seeds = seeds if selector in STOCHASTIC else [0]
            for k in ks:
                if selector == "greedy" and k > GREEDY_MAX_K:
                    continue
                for seed in use_seeds:
                    panel = build_panel(M, F, selector, k, kind="mini", seed=seed,
                                        train_si=train_si)
                    bad = check_weights(panel)
                    if bad:
                        raise AssertionError(f"weights do not sum to 1: {bad[:3]}")
                    for si in subjects:
                        novel = si in held
                        for metric in EVAL_METRICS:
                            if metric == "spk" and M.preset[si]:
                                continue  # preset engines do not clone; spk is meaningless
                            langs = eval_langs(M, si, panel, metric)
                            if len(langs) < 2:
                                continue
                            est = panel_estimate(M, si, metric, panel, langs)
                            true = M.true_score(si, metric, langs)
                            if not (np.isfinite(est) and np.isfinite(true)):
                                continue
                            ple = per_lang_errors(M, si, metric, panel, langs)
                            rows.append({
                                "set": set_name, "fold": fold_family, "selector": selector,
                                "k": k, "k_actual": panel["k_actual"], "seed": seed,
                                "system": M.systems[si], "family": M.families[si],
                                "role": M.roles[si],
                                "novel": int(novel), "metric": metric,
                                "est": est, "true": true, "abs_err": abs(est - true),
                                "n_langs": len(langs),
                                "worst_lang_err": max(ple) if ple else float("nan"),
                            })
        if verbose:
            print(f"  fold {fold_family:12s} done ({time.time() - t0:.0f}s, {len(rows)} rows)",
                  flush=True)

    if not rows:
        sys.exit(f"[{set_name}] error: no evaluation rows produced — check coverage "
                 f"thresholds (MIN_LEAF_COVERAGE={MIN_LEAF_COVERAGE}) and whether the "
                 f"matrix contains any subject systems")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{set_name}] wrote {len(rows)} evaluation rows -> {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def summarize(rows, set_name):
    print(f"\n=== {set_name}: score MAE by selector x k (novel-family systems only) ===")
    print("    mean absolute error, with its standard error in parentheses. Differences "
          "smaller\n    than ~2 SE are noise -- do not read a ranking into them.")
    metrics = sorted({r["metric"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    sels = sorted({r["selector"] for r in rows})
    for metric in metrics:
        by = mae_cells(rows, metric)
        print(f"\n  -- {metric} --")
        print(f"    {'selector':18s}" + "".join(f"{('k=' + str(k)):>16s}" for k in ks))
        for sel in sels:
            cells = []
            for k in ks:
                v = by.get((sel, k))
                if v:
                    m, se = mean_se(v)
                    cells.append(f"{m:10.3f}({se:.3f})")
                else:
                    cells.append(f"{'—':>16s}")
            print(f"    {sel:18s}" + "".join(cells))
        counts = [len(v) for v in by.values() if v]
        if counts:
            print(f"    (n = {min(counts)}-{max(counts)} novel-family evaluations per cell)")


def pairwise_sign(rows, set_name, selector, k, metric="wer_clip"):
    """Sign accuracy of panel-estimated differences, as a function of the true gap.

    This is the actionable statement: for systems separated by at least X on the full
    benchmark, the panel picks the better one Y% of the time. Pairs are pooled across
    folds, so unlike a per-fold rank correlation this has real sample size. Only pairs
    with at least one genuinely held-out (novel) side count -- a pair of two in-pool
    systems was fully visible to the selector and would flatter response-based ones.
    """
    gaps, correct = pairwise_sign_stats(rows, selector, k, metric=metric, min_novel=1)
    if not gaps.size:
        return
    print(f"\n=== {set_name}: pairwise sign accuracy — {selector} k={k} on {metric} "
          f"({len(gaps)} pairs, ≥1 held-out side) ===")
    edges = SIGN_GAP_EDGES
    print(f"    {'true gap':>16s} {'n':>7s} {'sign acc':>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gaps >= lo) & (gaps < hi)
        if m.sum():
            label = f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f">= {lo:g}"
            print(f"    {label:>16s} {int(m.sum()):7d} {correct[m].mean() * 100:9.1f}%")


def rank_corr(rows, set_name, selector, k, metric="wer_clip"):
    sel = [r for r in rows if r["selector"] == selector and r["k"] == k
           and r["metric"] == metric and r["seed"] == 0]
    by_fold = {}
    for r in sel:
        by_fold.setdefault(r["fold"], []).append(r)
    sp, pe, ke = [], [], []
    for _f, g in by_fold.items():
        if len(g) < 4:
            continue
        t = [r["true"] for r in g]
        e = [r["est"] for r in g]
        sp.append(stats.spearmanr(t, e).statistic)
        pe.append(stats.pearsonr(t, e).statistic)
        ke.append(stats.kendalltau(t, e).statistic)
    if sp:
        print(f"\n=== {set_name}: rank agreement — {selector} k={k} on {metric} "
              f"(mean over {len(sp)} folds) ===")
        print(f"    Pearson {np.mean(pe):.4f}   Spearman {np.mean(sp):.4f}   "
              f"Kendall {np.mean(ke):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["clean", "itw"])
    ap.add_argument("--ks", default="200,500,1000,2000")
    ap.add_argument("--selectors",
                    default="random_global,random_perlang,strat_meta,strat_response,"
                            "medoid_response,medoid_raw,highvar,greedy")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--report-selector", default="strat_response")
    ap.add_argument("--report-k", type=int, default=2000)
    ap.add_argument("--from-csv", action="store_true",
                    help="re-summarize an existing validation_<set>.csv without re-running")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    selectors = args.selectors.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    out_csv = os.path.join(OUT_DIR, f"validation_{args.set}.csv")

    if args.from_csv:
        # common.load_validation_rows types the fields and recomputes `novel` from
        # (fold, family, role) rather than trusting the stored column.
        rows, n_fixed = load_validation_rows(args.set)
        print(f"[{args.set}] loaded {len(rows)} rows from {out_csv}"
              + (f" ({n_fixed} novel-flags corrected)" if n_fixed else ""))
    else:
        rows = run(args.set, ks, selectors, seeds, out_csv)
    summarize(rows, args.set)
    rank_corr(rows, args.set, args.report_selector, args.report_k)
    pairwise_sign(rows, args.set, args.report_selector, args.report_k)
    if "greedy" in selectors and max(ks) > GREEDY_MAX_K:
        print(f"\nNOTE: greedy was skipped for k > {GREEDY_MAX_K} (runtime, not a result).")


if __name__ == "__main__":
    main()
