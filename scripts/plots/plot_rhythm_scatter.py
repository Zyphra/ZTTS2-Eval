"""Rhythm flattened-vs-erratic dot plot: FLEURS-R clean and VoxBlink2 in-the-wild.

One row per system, sorted by in-the-wild rhythm spread (flattened on top, erratic at
the bottom). Left: burstiness std_ratio (system spread / real-speech spread, log scale —
< 1 is rhythm flattened, > 1 erratic). Right: long_pct (% of utterances containing a
>=200 ms pause/stall run). Each row carries a hollow marker for clean and a filled one
for in-the-wild, and the column on the right prints the system's WER_CLIP on both
sets. The blue top row is real speech: the GT target recordings where the set has them
(in-the-wild), else the reference prompts' split-half row (clean, whose GT wavs are the
prompts and are not scored); its values are also drawn as reference lines. Good = close
to the real-speech markers of the same set.

Reads results/rhythm_<set>_<lang>.csv (scripts/prosody/rhythm_dist.py output).

Usage:
    python scripts/plots/plot_rhythm_scatter.py                         # both sets, en
    python scripts/plots/plot_rhythm_scatter.py --test-set clean --lang en
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from scripts.plots.tufte_common import PAPER, INK, MUT, GREY, ENS, MID, BLUE  # noqa: E402
from scripts.prosody.common import CEILING_ROW  # noqa: E402
from utils.benchmark import leaf_dir, wer_footer  # noqa: E402

SHORT = {
    "gt_in-the-wild": "GT (real speech)",
    "omni_fish-s2-pro": "fish-s2-pro",
    "omni_glm-tts": "glm-tts",
    "omni_higgs-audio-v3": "higgs-v3",
    "omni_omnivoice": "omnivoice-noref",
    "omni_omnivoice-reftext": "omnivoice-rt",
    "omni_qwen3-tts-0.6b": "qwen3-0.6b",
    "omni_qwen3-tts-1.7b": "qwen3-1.7b",
    "omni_voxcpm2": "voxcpm2",
    "omni_voxtral-tts": "voxtral",
    "zonos2_accurate": "zonos2-accurate",
    "zonos2_expressive": "zonos2-expressive",
    "moss-tts": "moss-tts",
    "moss-tts-v1.5": "moss-1.5",
    "moss-tts-local": "moss-local",
    "moss-tts-local-v1.5": "moss-local-1.5",
}

# foreground layer: the flattener exemplars, the near-GT case, the two
# counterexamples, the erratic extremes. Everything else stays plotted + labeled
# but recedes to the muted layer.
FOREGROUND = {
    "GT (real speech)", "kokoro", "fish-s2-pro", "qwen3-1.7b", "voxcpm2",
    "moss-tts", "zonos2-accurate", "voxtral", "glm-tts", "zonos2-expressive",
    "omnivoice-noref",
}

SET_LABEL = {"clean": "FLEURS-R clean", "in-the-wild": "VoxBlink2 in-the-wild"}


def wer_clip(system, test_set, lang):
    """WER_CLIP % from the system's .wer footer (nested or flat layout), or None."""
    w = wer_footer(os.path.join(leaf_dir(system, test_set, lang), "wav_res_ref_text.wer"))
    return None if w is None else w[2]


def load(csv_path, test_set, lang):
    """(rows, anchor): one dict per system, and the real-speech anchor -- the GT target
    recordings' row where the set has one, else the reference split-half row."""
    if not os.path.isfile(csv_path):
        sys.exit(f"error: {csv_path} not found -- run scripts/prosody/rhythm_dist.py "
                 f"--test-set {test_set} --lang {lang}")
    gt = f"gt_{test_set}"
    rows, special = [], {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            s = r["system"]
            if s in (CEILING_ROW, gt):
                special[s] = r
                continue
            rows.append(dict(system=SHORT.get(s, s), wer=wer_clip(s, test_set, lang),
                             x=float(r["burstiness_std_ratio"]), y=float(r["long_pct"]),
                             n=int(r["n"])))
    if gt in special:
        r, label = special[gt], "GT (real speech)"
    elif CEILING_ROW in special:
        r, label = special[CEILING_ROW], "real speech (reference prompts)"
    else:
        sys.exit(f"error: {csv_path} has neither {gt} nor {CEILING_ROW}")
    anchor = dict(system=label, wer=wer_clip(gt, test_set, lang), anchor=True,
                  x=float(r["burstiness_std_ratio"]), y=float(r["long_pct"]))
    return rows, anchor


def draw(ax_x, ax_y, data, lang):
    """data = [(test_set, rows, anchor)]. One row per system; with two sets the first is
    drawn hollow and the second filled on the same row."""
    two = len(data) == 2
    sets = [s for s, _, _ in data]
    series = {}
    for test_set, rows, _ in data:
        for r in rows:
            series.setdefault(r["system"], {})[test_set] = r
    key = sets[-1]   # sort by the last set's spread: flattened on top, erratic at the bottom
    order = sorted(series, key=lambda n: (series[n].get(key) or next(iter(series[n].values())))["x"])
    real = "real speech" if two else data[0][2]["system"]
    rows = [(real, {s: a for s, _, a in data})] + [(n, series[n]) for n in order]
    ys = [0] + [i + 1.5 for i in range(len(order))]   # a gap under the real-speech row

    def style(name):
        if name == real:
            return BLUE, True
        return (INK, True) if name in FOREGROUND else (MID, False)

    for ax in (ax_x, ax_y):
        ax.set_facecolor(PAPER)
        for i, y in enumerate(ys[1:]):
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color=ENS, alpha=0.35, lw=0, zorder=0)
        ax.tick_params(colors=MID, labelsize=9, length=3)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
    ax_x.set_ylim(ys[-1] + 0.8, -0.8)

    for (name, by_set), y in zip(rows, ys):
        color, fg = style(name)
        for k, test_set in enumerate(sets):
            r = by_set.get(test_set)
            if r is None:
                continue
            hollow = two and k == 0
            for ax, v in ((ax_x, r["x"]), (ax_y, r["y"])):
                ax.scatter(v, y, s=(46 if fg else 34) if hollow else (26 if fg else 18),
                           facecolors=PAPER if hollow else color,
                           edgecolors=color, linewidths=1.1 if hollow else 0,
                           zorder=3 if hollow else 4)
        wers = [by_set[s]["wer"] if s in by_set else None for s in sets]
        txt = " / ".join("—" if w is None else f"{w:.1f}" for w in wers)
        if name == real:
            txt += "  ASR floor"
        ax_y.annotate(txt, (1.0, y), xycoords=("axes fraction", "data"), xytext=(12, 0),
                      textcoords="offset points", ha="left", va="center", fontsize=8.5,
                      color=color, fontweight="bold" if name == real else "normal")

    ax_x.set_yticks(ys)
    ax_x.set_yticklabels([n for n, _ in rows], fontsize=8.5)
    for lab, (name, _) in zip(ax_x.get_yticklabels(), rows):
        lab.set_color(style(name)[0])
        if name == real:
            lab.set_fontweight("bold")

    # spread: log axis around parity with real speech
    ax_x.set_xscale("log")
    ax_x.set_xlim(0.32, 2.6)
    ax_x.set_xticks([0.4, 0.5, 0.7, 1.0, 1.4, 2.0])
    ax_x.set_xticklabels(["0.4×", "0.5×", "0.7×", "1×", "1.4×", "2×"])
    ax_x.minorticks_off()
    ax_x.axvline(1.0, color=GREY, lw=0.8, zorder=1)
    ax_x.set_xlabel("rhythm spread vs real speech (burstiness std ratio, log scale)",
                    color=INK, fontsize=10)

    # pause incidence: one reference line per set at its real-speech value
    ax_y.set_xlim(-2, 72)
    ax_y.set_xticks([0, 20, 40, 60])
    ax_y.set_xticklabels(["0", "20", "40", "60%"])
    for test_set, _, anchor in data:
        ax_y.axvline(anchor["y"], color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax_y.set_xlabel("utterances containing a ≥200 ms pause/stall", color=INK, fontsize=10)

    # direction words above each column, and the WER column header
    def head(ax, x, s, ha):
        ax.annotate(s, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 5),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=9,
                    color=MUT, style="italic")
    head(ax_x, 0.34, "← flattened", "left")
    head(ax_x, 2.5, "erratic →", "right")
    head(ax_y, 0, "← pause-poor", "left")
    head(ax_y, 70, "stall-heavy →", "right")
    ax_y.annotate("en WER_CLIP %" + ("\nclean / in-the-wild" if two else ""), (1.0, 1.0),
                  xycoords="axes fraction", xytext=(12, 5), textcoords="offset points",
                  ha="left", va="bottom", fontsize=8.5, color=MID)

    if two:
        sub = (f"{lang} · {len(order)} systems, sorted by in-the-wild rhythm spread · "
               f"○ {SET_LABEL[sets[0]]}   ● {SET_LABEL[sets[1]]}")
    else:
        ns = [r["n"] for r in data[0][1]]
        sub = (f"{SET_LABEL[sets[0]]} · {lang} · {len(order)} systems, "
               f"{min(ns)}–{max(ns)} heard utterances each, sorted by rhythm spread")
    for s, dy, size, weight, color in (("Rhythm: flattened vs erratic", 44, 15, "bold", INK),
                                       (sub, 26, 9.5, "normal", MUT)):
        ax_x.annotate(s, (0, 1.0), xycoords="axes fraction", xytext=(0, dy),
                      textcoords="offset points", ha="left", va="bottom", fontsize=size,
                      fontweight=weight, color=color)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", choices=["both", "clean", "in-the-wild"], default="both")
    ap.add_argument("--lang", default="en")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    sets = ["clean", "in-the-wild"] if args.test_set == "both" else [args.test_set]
    tag = "clean_vs_in-the-wild" if args.test_set == "both" else args.test_set
    out = args.out or os.path.join(
        REPO, "assets", "plots", f"rhythm_flatten_vs_stall_{tag}_{args.lang}.png")

    data = []
    for test_set in sets:
        csv_path = os.path.join(REPO, "results", f"rhythm_{test_set}_{args.lang}.csv")
        data.append((test_set, *load(csv_path, test_set, args.lang)))

    n_rows = len({r["system"] for _, rows, _ in data for r in rows}) + 1
    fig, (ax_x, ax_y) = plt.subplots(1, 2, sharey=True, dpi=200,
                                     figsize=(11.0, 1.9 + 0.2 * n_rows),
                                     gridspec_kw=dict(wspace=0.08))
    fig.patch.set_facecolor(PAPER)
    draw(ax_x, ax_y, data, args.lang)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
