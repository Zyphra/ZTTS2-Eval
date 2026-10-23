#!/usr/bin/env python3
"""Score a system that was run on an exported coreset panel.

Reads the ordinary per-leaf score files the pipeline already wrote under a panel
DECODE_DIR, applies the panel's per-item weights, and reports the estimate of the
**full-benchmark** score -- plus a weighted bootstrap CI so the number arrives with its
uncertainty attached instead of looking more precise than it is.

Two numbers are reported, deliberately:

  raw        the weighted panel score. Transparent, computable by anyone from the
             published weights, and dependent on no private model.
  calibrated a difference estimator that adds back the known full-set mean of the panel
             items' historical difficulty. It is the tinyBenchmarks-style efficiency
             gain, but it depends on this repo's historical matrix, so it is offered
             alongside the raw score rather than replacing it.

For a Delta panel only differences between two systems are meaningful; the tool refuses
to print an absolute level for one.

    python scripts/coreset/score_panel.py --decode_dir results/zonos2_accurate/zero_shot_mini
    python scripts/coreset/score_panel.py --decode_dir A --vs results/other/zero_shot_delta
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from common import CI_BOOT, CI_SEED, Matrix, weighted_boot  # noqa: E402
from utils.per_uid import load_leaf  # noqa: E402

METRIC_KEY = {"wer_clip": "wer_clip", "spk": "spk", "msr": "msr"}
# Metrics where a SMALLER value is the better one (error rates). Everything else
# (spk, msr) is higher-is-better. Drives the direction of the paired "A wins" readout.
LOWER_BETTER = {"wer", "wer_clip"}
SET_OF_TASK = {"zero_shot": "clean", "zero_shot_voxblink": "itw"}


def read_panel_manifest(task):
    path = os.path.join(REPO_ROOT, "data", task, "panel.json")
    if not os.path.isfile(path):
        sys.exit(f"error: {path} not found — is {task} an exported panel?")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect(decode_dir, manifest):
    """Per-language {uid: {metric: value}} for whatever the scorer produced."""
    out = {}
    for lang in sorted(manifest["langs"]):
        leaf = os.path.join(decode_dir, lang)
        if not os.path.isdir(leaf):
            continue
        d = load_leaf(leaf)
        vals = {}
        for uid in manifest["langs"][lang]:
            row = {}
            for m in METRIC_KEY:
                v = d[m].get(uid)
                if v is not None and np.isfinite(v):
                    row[m] = float(v)
            if row:
                vals[uid] = row
        if vals:
            out[lang] = vals
    return out


def weighted_macro(collected, manifest, metric):
    """Panel estimate + the per-language pieces the bootstrap and calibration need.

    Each piece is (lang, uids, y, w, lang_mean) with all three arrays in the same uid
    order, so downstream steps can align on uid rather than on position.
    """
    pieces = []
    for lang, vals in collected.items():
        w_all = manifest["langs"][lang]
        uids, y, w = [], [], []
        for uid in sorted(vals):
            row = vals[uid]
            if metric in row and uid in w_all:
                uids.append(uid)
                y.append(row[metric])
                w.append(w_all[uid])
        if y:
            y, w = np.asarray(y), np.asarray(w)
            pieces.append((lang, uids, y, w, float(np.sum(w * y) / np.sum(w))))
    if not pieces:
        return float("nan"), []
    return float(np.mean([p[4] for p in pieces])), pieces


def boot_ci(pieces, boot=CI_BOOT, seed=CI_SEED):
    if not pieces:
        return float("nan"), float("nan")
    draws = weighted_boot([(y, w) for _lg, _uids, y, w, _m in pieces], boot=boot, seed=seed)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def calibrate(pieces, manifest, metric):
    """Difference estimator against the panel items' historical difficulty.

        S_cal = mean_full(historical) + weighted_mean_panel(y - historical)

    Removes the part of a panel's error that comes from having happened to draw easy or
    hard items, to the extent the historical mean predicts this system. That is a modest
    effect for WER and a larger one for the in-the-wild speaker/MOS metrics, so the gain
    is reported rather than assumed. Returns None when the historical matrix does not
    overlap the panel.

    Both terms are computed over exactly the same items, aligned by uid: the residual has
    to be measured against the difficulty of the items actually scored, not of the items
    originally selected.
    """
    set_name = SET_OF_TASK.get(manifest["source_task"])
    if set_name is None:
        return None
    M = Matrix(set_name)
    train = [i for i, r in enumerate(M.roles) if r == "system"]
    if metric == "spk":
        train = [i for i in train if not M.preset[i]]
    if not train:
        return None

    with np.errstate(invalid="ignore"):
        hist = np.nanmean(M.m[metric][train, :], axis=0)
    pos = {(lg, u): j for j, (lg, u) in enumerate(zip(M.lang, M.uid))}

    full_by_lang, resid_by_lang = [], []
    for lang, uids, y, w, _m in pieces:
        cols = M.cols(lang)
        hf = hist[cols]
        hf = hf[np.isfinite(hf)]
        if not hf.size:
            continue
        rows = [(yi, wi, hist[pos[(lang, u)]])
                for u, yi, wi in zip(uids, y, w)
                if (lang, u) in pos and np.isfinite(hist[pos[(lang, u)]])]
        if not rows:
            continue
        yy = np.array([r[0] for r in rows])
        ww = np.array([r[1] for r in rows])
        hp = np.array([r[2] for r in rows])
        full_by_lang.append(float(hf.mean()))
        resid_by_lang.append(float((np.sum(ww * yy) - np.sum(ww * hp)) / np.sum(ww)))
    if not full_by_lang:
        return None
    return float(np.mean(full_by_lang) + np.mean(resid_by_lang))


def paired_stats(coll_a, coll_b, manifest, metric):
    """Matched-pair difference (A − B) on the shared panel items, for one metric.

    Both systems spoke the SAME utterances with the SAME prompts, so differencing
    within uid removes the per-item variance that dominates the unpaired comparison.

    Returns ``(est, lo, hi, win, n)`` or None when the two runs share no scored items.
    ``win`` is the percentage of shared items where A is genuinely BETTER than B --
    direction-aware per metric (for error metrics like wer_clip, A wins where the
    difference is negative), with exact ties split evenly rather than counted as losses.
    """
    per_lang = []
    for lang in sorted(set(coll_a) & set(coll_b)):
        w_all = manifest["langs"][lang]
        shared = [u for u in coll_a[lang] if u in coll_b[lang]
                  and metric in coll_a[lang][u] and metric in coll_b[lang][u]]
        if not shared:
            continue
        d = np.array([coll_a[lang][u][metric] - coll_b[lang][u][metric] for u in shared])
        w = np.array([w_all[u] for u in shared])
        per_lang.append((d, w))
    if not per_lang:
        return None
    est = float(np.mean([float(np.sum(w * d) / np.sum(w)) for d, w in per_lang]))
    draws = weighted_boot(per_lang, boot=CI_BOOT, seed=CI_SEED)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    d_all = np.concatenate([d for d, _ in per_lang])
    a_better = d_all < 0 if metric in LOWER_BETTER else d_all > 0
    win = float((np.mean(a_better) + 0.5 * np.mean(d_all == 0)) * 100)
    return est, float(lo), float(hi), win, int(d_all.size)


def report(decode_dir, task, label=None):
    manifest = read_panel_manifest(task)
    collected = collect(decode_dir, manifest)
    if not collected:
        sys.exit(f"error: no scored leaves found under {decode_dir}")
    n = sum(len(v) for v in collected.values())
    print(f"\n=== {label or decode_dir} ===")
    print(f"panel: {manifest['kind']} / {manifest['selector']} / k={manifest['k_actual']} "
          f"({manifest['source_task']}); scored {n} items in {len(collected)} languages")
    if manifest["kind"] == "delta":
        print("NOTE: this is a Delta panel — only differences between two systems are\n"
              "      meaningful. Its absolute level is not an estimate of the full benchmark.")
    out = {}
    for metric in METRIC_KEY:
        est, pieces = weighted_macro(collected, manifest, metric)
        if not np.isfinite(est):
            continue
        lo, hi = boot_ci(pieces)
        cal = calibrate(pieces, manifest, metric) if manifest["kind"] == "mini" else None
        cal_s = f"   calibrated {cal:8.3f}" if cal is not None else ""
        print(f"  {metric:9s} {est:8.3f}   CI95 [{lo:.3f}, {hi:.3f}]{cal_s}")
        out[metric] = {"raw": est, "ci95": [lo, hi], "calibrated": cal, "pieces": pieces}
    return manifest, collected, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--task", help="panel task name (default: basename of decode_dir)")
    ap.add_argument("--vs", help="second decode_dir for a paired comparison")
    args = ap.parse_args()

    task = args.task or os.path.basename(os.path.normpath(args.decode_dir))
    manifest, coll_a, res_a = report(args.decode_dir, task)

    if args.vs:
        task_b = args.task or os.path.basename(os.path.normpath(args.vs))
        _mb, coll_b, res_b = report(args.vs, task_b)
        print("\n=== paired difference (A − B) ===")
        print("    'A wins' is direction-aware: for wer_clip A wins where its error is\n"
              "    LOWER; for spk/msr where its score is higher. Ties split evenly.")
        for metric in METRIC_KEY:
            if metric not in res_a or metric not in res_b:
                continue
            st = paired_stats(coll_a, coll_b, manifest, metric)
            if st is None:
                continue
            est, lo, hi, win, _n = st
            print(f"  {metric:9s} {est:+8.3f}   CI95 [{lo:+.3f}, {hi:+.3f}]   A wins {win:5.1f}%")


if __name__ == "__main__":
    main()
