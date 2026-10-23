#!/usr/bin/env python3
"""Paired bootstrap CIs for system-vs-system and system-vs-GT deltas.

The published ranking tables (RESULTS_VOXBLINK.md, RESULTS_NEWLANGS.md) print Δ vs GT
to 0.1 with no uncertainty attached. This tool puts the missing intervals on those
deltas, using the pairing the benchmark already has: every system spoke the SAME items,
so differencing within item removes the per-item difficulty variance that dominates an
unpaired comparison of two means.

Estimator (the published macro-average, scripts/coreset/common.py): per language, keep
the items where BOTH rows are finite and take the mean paired difference; macro Δ is the
unweighted mean over languages. The bootstrap resamples items WITHIN each language --
languages are fixed strata, not a sample -- and takes seeded percentile CI95s with the
repo-wide CI_BOOT/CI_SEED convention (utils/average_wer.py). ``--langs-random`` adds the
other view per pair: the spread of per-language deltas, in how many languages the macro
delta's sign holds, and a CI from resampling *languages* (read it as "would this hold on
a different language basket", not as the basket-level CI).

Scope defaults to the published basket, exactly as build_matrix.py --verify derives it:
itw = all 20 langs; clean = the 26 tail languages of RESULTS_NEWLANGS.md (base-9
excluded). ``--langs`` overrides it.

    python scripts/paired_ci.py --set itw --adjacent
    python scripts/paired_ci.py --set clean --adjacent --langs-random
    python scripts/paired_ci.py --set itw --vs-gt moss-tts --vs-gt zonos2_expressive
    python scripts/paired_ci.py --set itw --metric spk --pair zonos2_accurate voxcpm2 --langs en
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.coreset.common import CI_BOOT, CI_SEED, Matrix  # noqa: E402

# RESULTS_NEWLANGS.md scope = the FLEURS-R tail; must match build_matrix.py verify().
BASE9 = {"en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru"}


def published_scope(M):
    """The language basket the published table for this set aggregates over."""
    if M.set == "clean":
        return [lg for lg in M.langs if lg not in BASE9]
    return list(M.langs)


def pair_delta_ci(M, metric, a, b, langs, boot=CI_BOOT, seed=CI_SEED,
                  langs_random=False, min_items=2):
    """Paired macro Δ = mean_l mean_i (a_i - b_i) with a within-language item bootstrap.

    Returns a dict (delta, lo, hi, n_langs, n_items, langs, + lang stats when
    ``langs_random``), or None when no language has ``min_items`` jointly-scored items.
    """
    ia, ib = M.si(a), M.si(b)
    ds, used = [], []
    for lg in langs:
        cols = M.cols(lg)
        va = M.m[metric][ia, cols]
        vb = M.m[metric][ib, cols]
        ok = np.isfinite(va) & np.isfinite(vb)
        if ok.sum() >= min_items:
            ds.append((va[ok] - vb[ok]).astype(np.float64))
            used.append(lg)
    if not ds:
        return None
    lang_means = np.array([d.mean() for d in ds])
    delta = float(lang_means.mean())

    rng = np.random.default_rng(seed)
    acc = np.zeros(boot)
    for d in ds:
        n = d.size
        # Row-chunked like utils/average_wer.py boot_ci95, to cap peak memory.
        step = max(1, 2_000_000 // n)
        done = 0
        while done < boot:
            m = min(step, boot - done)
            idx = rng.integers(0, n, size=(m, n))
            acc[done:done + m] += d[idx].mean(axis=1)
            done += m
    draws = acc / len(ds)
    lo, hi = np.percentile(draws, [2.5, 97.5])

    out = {"a": a, "b": b, "delta": delta, "lo": float(lo), "hi": float(hi),
           "n_langs": len(used), "n_items": int(sum(d.size for d in ds)), "langs": used}
    if langs_random:
        out["lang_sd"] = float(lang_means.std(ddof=1)) if len(used) > 1 else float("nan")
        out["sign_holds"] = int(np.sum(np.sign(lang_means) == np.sign(delta)))
        rng2 = np.random.default_rng(seed)
        j = rng2.integers(0, len(used), size=(boot, len(used)))
        ldraws = lang_means[j].mean(axis=1)
        llo, lhi = np.percentile(ldraws, [2.5, 97.5])
        out["lang_lo"], out["lang_hi"] = float(llo), float(lhi)
    return out


def ranking_order(M, langs):
    """Role-"system" rows sorted by published Δ vs GT ascending (wer_clip, unpaired
    per-language means over the system's covered basket, GT restricted to it) -- the
    exact aggregation of build_matrix.py verify(), so the order matches the tables."""
    gi = M.si("GT")
    order = []
    for si, (key, role) in enumerate(zip(M.systems, M.roles)):
        if role != "system":
            continue
        wm, gm = [], []
        for lg in langs:
            v = M.m["wer_clip"][si, M.cols(lg)]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            g = M.m["wer_clip"][gi, M.cols(lg)]
            g = g[np.isfinite(g)]
            wm.append(v.mean())
            gm.append(g.mean() if g.size else np.nan)
        if wm:
            order.append((float(np.mean(wm) - np.mean(gm)), key))
    return [key for _d, key in sorted(order)]


def format_table(results, metric, langs_random=False):
    """Compact aligned lines for a list of pair_delta_ci results (None rows skipped)."""
    lines = []
    width = max([len(f"{r['a']} - {r['b']}") for r in results if r] + [4])
    head = (f"{'pair':<{width}} {'langs':>5} {'items':>6} {'Δ':>8} {'CI95':>17}")
    if langs_random:
        head += f" {'sd(Δ_l)':>8} {'sign':>6} {'lang-CI95':>17}"
    lines.append(head)
    for r in results:
        if r is None:
            continue
        line = (f"{r['a'] + ' - ' + r['b']:<{width}} {r['n_langs']:>5d} {r['n_items']:>6d} "
                f"{r['delta']:>+8.2f} [{r['lo']:>+7.2f},{r['hi']:>+7.2f}]")
        if langs_random:
            line += (f" {r['lang_sd']:>8.2f} {r['sign_holds']:>3d}/{r['n_langs']:<2d} "
                     f"[{r['lang_lo']:>+7.2f},{r['lang_hi']:>+7.2f}]")
        if r["lo"] <= 0.0 <= r["hi"]:
            line += "  <- spans 0"
        lines.append(line)
    return lines


def print_adjacent(M, metric="wer_clip", boot=CI_BOOT, seed=CI_SEED,
                   langs_random=False, langs=None):
    """Adjacent-pair ΔCI table in published ranking order (importable by build_matrix)."""
    scope = langs or published_scope(M)
    ranked = ranking_order(M, scope)
    results = [pair_delta_ci(M, metric, a, b, scope, boot=boot, seed=seed,
                             langs_random=langs_random)
               for a, b in zip(ranked[:-1], ranked[1:])]
    print(f"\n=== {M.set}: adjacent-pair paired ΔCI95 on {metric} "
          f"({len(scope)} langs, published ranking order, boot={boot}, seed={seed}) ===")
    print("Δ = higher-ranked minus next system; a CI spanning 0 means the rank gap is "
          "statistically\nunresolved on this basket.")
    for line in format_table(results, metric, langs_random):
        print(line)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=("clean", "itw"))
    ap.add_argument("--metric", default="wer_clip", choices=("wer_clip", "wer", "spk", "msr"))
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), default=[],
                    help="paired ΔCI for A - B (repeatable)")
    ap.add_argument("--vs-gt", action="append", metavar="SYSTEM", default=[],
                    help="paired ΔCI for SYSTEM - GT (repeatable)")
    ap.add_argument("--adjacent", action="store_true",
                    help="all adjacent pairs of the published ranking (role=system, "
                         "sorted by Δ vs GT ascending)")
    ap.add_argument("--langs", default="",
                    help="comma-separated language restriction (default: published basket)")
    ap.add_argument("--langs-random", action="store_true",
                    help="add per-pair: sd of per-lang deltas, #langs where the macro "
                         "delta's sign holds, and a languages-as-random resampling CI")
    ap.add_argument("--boot", type=int, default=CI_BOOT)
    ap.add_argument("--seed", type=int, default=CI_SEED)
    args = ap.parse_args()

    if not (args.pair or args.vs_gt or args.adjacent):
        ap.error("nothing to do: pass --pair, --vs-gt and/or --adjacent")

    M = Matrix(args.set)
    scope = [s.strip() for s in args.langs.split(",") if s.strip()] or published_scope(M)
    unknown = [lg for lg in scope if lg not in M.langs]
    if unknown:
        ap.error(f"unknown languages for --set {args.set}: {' '.join(unknown)}")

    explicit = list(args.pair) + [(s, "GT") for s in args.vs_gt]
    if explicit:
        results = []
        for a, b in explicit:
            for name in (a, b):
                if name not in M.systems:
                    ap.error(f"unknown system {name!r} (rows: {', '.join(M.systems)})")
            r = pair_delta_ci(M, args.metric, a, b, scope, boot=args.boot, seed=args.seed,
                              langs_random=args.langs_random)
            if r is None:
                print(f"{a} - {b}: no language with >=2 jointly scored items", file=sys.stderr)
            results.append(r)
        print(f"\n=== {args.set}: paired ΔCI95 on {args.metric} "
              f"({len(scope)} langs, boot={args.boot}, seed={args.seed}) ===")
        for line in format_table(results, args.metric, args.langs_random):
            print(line)

    if args.adjacent:
        print_adjacent(M, args.metric, boot=args.boot, seed=args.seed,
                       langs_random=args.langs_random, langs=scope)


if __name__ == "__main__":
    main()
