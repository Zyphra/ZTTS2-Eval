#!/usr/bin/env python3
"""Generate RESULTS_CORESET.md from the validation CSVs.

Written as a generator rather than by hand on purpose: a hand-maintained RESULTS table
in this repo once silently dropped a language from three systems, and every number
below is one nobody should be retyping.

    python scripts/coreset/gen_results_md.py > RESULTS_CORESET.md
"""

import argparse
import collections
import os
import sys

import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

from scripts.coreset.common import (Matrix, PANEL_DIR, SIGN_GAP_EDGES, load_panel,  # noqa: E402
                    load_validation_rows, mae_cells, mean_se, pairwise_sign_stats)

SETS = [("clean", "Clean (FLEURS-R, 35 languages, 57,388 items)"),
        ("itw", "In-The-Wild (VoxBlink2, 20 languages, 17,319 items)")]
FULL_N = {"clean": 57366, "itw": 17155}
METRIC_LABEL = {"wer_clip": "WER_CLIP (%)", "spk": "speaker sim (x100)", "msr": "MSR-UTMOS"}
SHIPPED = {"clean": 3500, "itw": 2000}
SHIPPED_SELECTOR = "strat_meta"


def load(set_name):
    """Typed validation rows with the `novel` flag re-derived (see common.py)."""
    try:
        rows, _n_fixed = load_validation_rows(set_name)
    except FileNotFoundError:
        return []
    return rows


def mae_table(rows, metric, out):
    by = mae_cells(rows, metric)
    ks = sorted({k for _s, k in by})
    sels = sorted({s for s, _k in by})
    if not ks:
        return
    out.append(f"\n**{METRIC_LABEL[metric]}** — mean absolute error on held-out families "
               f"(standard error in parentheses)\n")
    out.append("| selector | " + " | ".join(f"k={k}" for k in ks) + " |")
    out.append("|---|" + "--:|" * len(ks))
    for sel in sels:
        cells = []
        for k in ks:
            v = by.get((sel, k))
            if v:
                m, se = mean_se(v)
                cells.append(f"{m:.3f} ({se:.3f})")
            else:
                cells.append("—")
        bold = "**" if sel == SHIPPED_SELECTOR else ""
        out.append(f"| {bold}{sel}{bold} | " + " | ".join(cells) + " |")


def sign_table(rows, set_name, out, metric="wer_clip"):
    # Only pairs with at least one genuinely held-out side: a pair of two in-pool
    # systems was fully visible to the selector, so it would not be the leave-family-out
    # readout the preamble promises.
    gaps, correct, pairs = pairwise_sign_stats(rows, SHIPPED_SELECTOR, SHIPPED[set_name],
                                               metric=metric, min_novel=1, with_pairs=True)
    if not gaps.size:
        return
    out.append(f"\n**Pairwise sign accuracy** — {SHIPPED_SELECTOR}, k={SHIPPED[set_name]}, "
               f"on {METRIC_LABEL[metric]} ({len(gaps)} pairs pooled across folds and "
               f"seeds; every pair has at least one held-out system). Pooled pairs are "
               f"dependent — the same two systems recur across folds/seeds — so the "
               f"distinct-pairs column is the real sample size per bin.\n")
    out.append("| true gap on the full benchmark | pairs | distinct system pairs | "
               "picks the better system |")
    out.append("|---|--:|--:|--:|")
    edges = SIGN_GAP_EDGES
    thin = False
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gaps >= lo) & (gaps < hi)
        if m.sum():
            label = f"{lo:g} – {hi:g}" if np.isfinite(hi) else f"≥ {lo:g}"
            n_distinct = len(set(pairs[m]))
            star = "\\*" if n_distinct < 20 else ""
            out.append(f"| {label} | {int(m.sum())} | {n_distinct}{star} | "
                       f"{correct[m].mean() * 100:.1f}% |")
            thin = thin or bool(star)
    if thin:
        out.append("\n\\* fewer than 20 distinct system pairs behind this bin — too few "
                   "independent\ncomparisons to read the percentage as more than anecdote.")


def worst_lang(rows, set_name, out):
    sel = [r for r in rows if r["selector"] == SHIPPED_SELECTOR
           and r["k"] == SHIPPED[set_name] and r["novel"]]
    by = collections.defaultdict(list)
    for r in sel:
        if np.isfinite(r["worst_lang_err"]):
            by[r["metric"]].append(r["worst_lang_err"])
    if not by:
        return
    out.append(f"\n**Worst single-language error** — {SHIPPED_SELECTOR}, "
               f"k={SHIPPED[set_name]}, held-out families\n")
    out.append("| metric | median | p90 | max |")
    out.append("|---|--:|--:|--:|")
    for metric in ("wer_clip", "spk", "msr"):
        v = by.get(metric)
        if v:
            out.append(f"| {METRIC_LABEL[metric]} | {np.median(v):.3f} | "
                       f"{np.percentile(v, 90):.3f} | {max(v):.3f} |")
    out.append("\nA panel's per-language numbers are much noisier than its macro-average — "
               "the\nlanguage budgets are small by construction. Quote the macro-average; "
               "treat a\nsingle language's panel score as indicative only.")


def panel_file(set_name):
    """Shipped-panel filename, derived from SHIPPED/SHIPPED_SELECTOR so bumping a k
    cannot silently point this doc at a stale panel."""
    return f"{set_name}_mini_{SHIPPED_SELECTOR}_k{SHIPPED[set_name]}.json"


def ci_coverage(set_name, out):
    """Does the SHIPPED panel's bootstrap CI actually contain the full-set value?

    The error tables above describe the *procedure* averaged over seeds and folds. This
    asks the different and more practical question about the *one panel that ships*: if a
    user quotes its CI, how often does that interval really cover the truth? Nominal is
    95%.
    """
    path = os.path.join(PANEL_DIR, panel_file(set_name))
    if not os.path.isfile(path):
        out.append(f"\n_CI95 coverage of the shipped panel unavailable: "
                   f"`{os.path.relpath(path, REPO)}` not found._\n")
        return
    try:
        M, p = Matrix(set_name), load_panel(path)
    except Exception as exc:
        out.append(f"\n_CI95 coverage of the shipped panel unavailable: {exc}_\n")
        return
    hit = tot = 0
    rows = []
    for si, (name, role) in enumerate(zip(M.systems, M.roles)):
        if role not in ("system", "partial"):
            continue
        for met in ("wer_clip", "spk", "msr"):
            if met == "spk" and M.preset[si]:
                continue
            langs = [lg for lg in p["langs"] if lg in set(M.covered_langs(si, met, 0.9))]
            if len(langs) < 2:
                continue
            sub = {"langs": {lg: p["langs"][lg] for lg in langs}}
            lo, hi = M.panel_score_ci(si, met, sub, boot=2000)
            t = M.true_score(si, met, langs)
            ok = lo <= t <= hi
            hit += ok
            tot += 1
            if met == "wer_clip":
                rows.append((name, M.panel_score(si, met, sub), lo, hi, t, ok))
    if not tot:
        out.append("\n_CI95 coverage of the shipped panel unavailable: no system × metric "
                   "combination had enough covered panel languages._\n")
        return
    out.append(f"\n**CI95 coverage of the shipped panel** — {hit}/{tot} = "
               f"**{hit / tot * 100:.0f}%** of estimates contained the full-set value "
               f"(nominal 95%), across every system × metric.\n")
    out.append("| system | panel | CI95 | full set | covered |")
    out.append("|---|--:|--:|--:|:-:|")
    for name, est, lo, hi, t, ok in rows:
        out.append(f"| {name} | {est:.2f} | [{lo:.2f}, {hi:.2f}] | {t:.2f} | "
                   f"{'yes' if ok else '**no**'} |")
    out.append("\nWER_CLIP shown; speaker similarity and MSR-UTMOS are folded into the "
               "headline count.\nNote these rows are *not* independent — they share one "
               "panel draw, so a panel that\nhappens to sample slightly hard items misses "
               "on several systems at once.")


DELTA_K = 200


def delta_table(set_name, out):
    """Delta-panel power, measured live (there is no CSV for it -- it is cheap to redo)."""
    try:
        from scripts.coreset.validate_delta import measure
        rows, pairs = measure(set_name, [100, DELTA_K, 400],
                              ["delta", "random_perlang", "strat_meta"])
    except Exception as exc:                       # missing matrix/fingerprints
        out.append(f"\n_Delta panel validation unavailable: {exc}_\n")
        return
    if not rows:
        return
    out.append(f"\n### Delta panel (k={DELTA_K}) — power to detect a change\n")
    out.append("MDD = minimum detectable difference at 80% power (two-sided α=0.05), from "
               "the panel's\nmatched-pair bootstrap SE. Lower is better.\n")
    out.append("| selector | k | MDD | sign correct | magnitude err | pairs |")
    out.append("|---|--:|--:|--:|--:|--:|")
    for selector, k, mdd, sign, err, n in rows:
        bold = "**" if selector == "delta" and k == DELTA_K else ""
        out.append(f"| {bold}{selector}{bold} | {k} | {mdd:.3f} | {sign * 100:.0f}% | "
                   f"{err:.3f} | {n} |")
    out.append(f"\nNear-pairs used: " + ", ".join(f"`{a}` vs `{b}`" for a, b in pairs) + ".")
    out.append("\nThe `delta` selector buys real power — it reaches a given MDD with roughly "
               "half the\nitems of a random panel — but its **magnitude error is worse by "
               "design**: over-sampling\ndiscriminative items inflates the measured "
               "difference. Use a Delta panel to decide\n*direction*, and a Mini panel to "
               "*quantify*. Also note these near-pairs are further apart\nthan two "
               "consecutive checkpoints, so the sign column is optimistic while MDD is\n"
               "conservative — quote MDD.")


def in_vs_novel(rows, set_name, out):
    sel = [r for r in rows if r["selector"] == SHIPPED_SELECTOR
           and r["k"] == SHIPPED[set_name] and r["metric"] == "wer_clip"]
    a = [r["abs_err"] for r in sel if not r["novel"]]
    b = [r["abs_err"] for r in sel if r["novel"]]
    if not (a and b):
        return
    out.append(f"\n**Selection overfitting check** — WER_CLIP MAE, k={SHIPPED[set_name]}\n")
    out.append("| systems | n | MAE |")
    out.append("|---|--:|--:|")
    out.append(f"| in the selection pool | {len(a)} | {np.mean(a):.3f} |")
    out.append(f"| held-out family (novel) | {len(b)} | {np.mean(b):.3f} |")
    out.append("\nThese should be close. A large gap would mean the selector had tuned "
               "itself to the\nsystems it could see.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    out = []
    out.append("# ZTTS2-Eval coreset panels — validation results\n")
    out.append("Generated by `scripts/coreset/gen_results_md.py` from "
               "`assets/coreset/validation_*.csv`.\nMethod and scope limits are in "
               "[docs/coreset.md](docs/coreset.md).\n")
    out.append("Every number below comes from **leave-family-out** validation: the panel is "
               "selected\nusing only the remaining model families, then scored on the "
               "held-out one. With ~9\nindependent families, per-fold rank correlations are "
               "underpowered — the pairwise\nsign-accuracy table is the readout with real "
               "sample size.\n")

    for set_name, title in SETS:
        rows = load(set_name)
        if not rows:
            out.append(f"\n## {title}\n\n_Not yet published._ No "
                       f"`assets/coreset/validation_{set_name}.csv` is present — regenerate "
                       f"with `scripts/coreset/validate.py --set {set_name}`.\n")
            if set_name == "clean":
                out.append("\nThe Clean panel is **deliberately held back**. Building the "
                           "matrix surfaced that the\n2026-07-30 ASR-routing fix covered "
                           "only 6 of the 28 languages that need Whisper\nrouting, so 22 "
                           "languages still have synth scored by Qwen auto-detect while "
                           "ground\ntruth was Whisper-routed — inflating their WER by up to "
                           "53 points. Selecting a public\npanel from those numbers would "
                           "bake the error in. They are being re-scored;\n"
                           "the Clean panel and this section will be "
                           "regenerated afterwards.\n")
            continue
        k = SHIPPED[set_name]
        out.append(f"\n## {title}\n")
        out.append(f"Shipped Mini panel: **{SHIPPED_SELECTOR}, k={k}** "
                   f"({FULL_N[set_name] / k:.1f}x smaller than the full set).\n")
        for metric in ("wer_clip", "spk", "msr"):
            mae_table(rows, metric, out)
        sign_table(rows, set_name, out)
        worst_lang(rows, set_name, out)
        in_vs_novel(rows, set_name, out)
        ci_coverage(set_name, out)
        delta_table(set_name, out)

    out.append("\n## Reading these tables\n")
    out.append("- **Differences smaller than ~2 SE are noise.** At the shipped sizes every "
               "language-stratified\n  unbiased selector ties; the ones that separate are "
               "`highvar` (catastrophic — it preserves\n  ordering while badly biasing the "
               "level), `random_global` (~2x worse, and it can miss small\n  languages "
               "entirely), and `medoid_raw` vs `medoid_response` (which is what the low-rank\n"
               "  denoising is worth).")
    out.append("- **`strat_meta` ships** not because it wins but because it ties while being "
               "unbiased and\n  defining its strata from *published metadata only* — so the "
               "design is auditable without\n  access to this repo's historical results.")
    out.append("- **Delta panels are validated separately** by "
               "`scripts/coreset/validate_delta.py`, which\n  reports the minimum detectable "
               "difference at 80% power. They are deliberately biased in\n  magnitude: use "
               "them for direction, and a Mini panel to quantify.")

    print("\n".join(out))


if __name__ == "__main__":
    main()
