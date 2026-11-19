"""Rhythm flattened-vs-erratic scatter: FLEURS-R clean and VoxBlink2 in-the-wild side by side.

One dot per system: x = burstiness std_ratio (system spread / real-speech spread, log
scale — < 1 is rhythm flattened, > 1 erratic), y = long_pct (% of utterances
containing a >=200 ms pause/stall run); each label carries the system's WER_CLIP.
The blue anchor is real speech: the GT target recordings where the set has them
(in-the-wild), else the reference prompts' split-half row (clean, whose GT wavs are
the prompts and are not scored). Lower-left = flattened, pause-poor; upper-right =
erratic, stall-heavy; good = near the anchor. Both panels share axes.

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
from scripts.plots.tufte_common import PAPER, INK, MUT, GREY, MID, RED, BLUE  # noqa: E402
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

# label offset (points) and horizontal alignment per set and system, tuned by eye.
# default is a small right-offset; entries here override collisions.
NUDGE = {
    "in-the-wild": {
        "GT (real speech)": (8, 7, "left"),
        "moss-1.5": (-5, 0, "right"),
        "qwen3-0.6b": (5, 4, "left"),
        "fish-s2-pro": (6, 2, "left"),
        "qwen3-1.7b": (5, -1, "left"),
        "voxcpm2": (5, -5, "left"),
        "zonos2-expressive": (-5, 0, "right"),
        "vibevoice-7b": (-5, 0, "right"),
        "cosyvoice3": (-5, 0, "right"),
        "higgs-v3": (-5, 0, "right"),
    },
    "clean": {
        "real speech (reference prompts)": (6, 9, "left"),
        "kokoro": (-5, 0, "right"),
        "qwen3-1.7b": (-5, 0, "right"),
        "fish-s2-pro": (-5, 0, "right"),
        "moss-tts": (5, -4, "left"),
        "zonos2-accurate": (5, 5, "left"),
        "moss-local": (5, -3, "left"),
        "moss-1.5": (5, -4, "left"),
        "omni_breeze-tts-2": (-5, 0, "right"),
        "omni_drama-3": (-5, 0, "right"),
        "vibevoice-rt-0.5b": (-5, 0, "right"),
        "vibevoice-1.5b": (5, -4, "left"),
        "moss-local-1.5": (-5, 0, "right"),
        "zonos3-mini_reference": (-5, 0, "right"),
        "zonos3-mini_embedding": (5, -4, "left"),
        "zonos3-mini-reftext_embedding": (-5, 0, "right"),
        "ming-3.0-tts-0.5b": (5, -4, "left"),
        "ming-3.0-tts-0.5b-trim": (5, -4, "left"),
        "ming-3.0-tts-tiny": (5, -4, "left"),
        "ming-3.0-tts-tiny-trim": (5, -4, "left"),
    },
}


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
    anchor = dict(system=label, wer=wer_clip(gt, test_set, lang),
                  x=float(r["burstiness_std_ratio"]), y=float(r["long_pct"]))
    return rows, anchor


def draw(ax, rows, anchor, test_set, lang, first):
    ax.set_facecolor(PAPER)
    ax.set_xscale("log")
    ax.set_xlim(0.32, 2.6)
    ax.set_ylim(-2, 74)

    # reference geometry: spread parity line + real-speech pause incidence
    ax.axvline(1.0, color=GREY, lw=0.8, zorder=1)
    ax.axhline(anchor["y"], color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(1.0, 72.5, "same spread as real speech", color=MUT, fontsize=8,
            ha="center", va="top")
    ax.text(0.33, anchor["y"] + 1.2, f"real-speech pause incidence ({anchor['y']:.0f}%)",
            color=MUT, fontsize=8, ha="left", va="bottom")

    # regime annotations
    ax.text(0.335, 3.5, "flattened, pause-poor",
            color=MUT, fontsize=9, ha="left", va="bottom", style="italic")
    ax.text(2.52, 72.5, "erratic, stall-heavy", color=MUT, fontsize=9,
            ha="right", va="top", style="italic")

    nudge = NUDGE.get(test_set, {})
    for r, is_anchor in [(r, False) for r in rows] + [(anchor, True)]:
        fg = is_anchor or r["system"] in FOREGROUND
        color = BLUE if is_anchor else (INK if fg else MUT)
        ax.scatter(r["x"], r["y"], s=46 if is_anchor else (24 if fg else 16),
                   color=color, zorder=3 if fg else 2,
                   edgecolors=PAPER, linewidths=0.8)
        dx, dy, ha = nudge.get(r["system"], (5, 0, "left"))
        # Tufte: WER rides in the label as printed data, not a size/color code
        label = r["system"]
        if r["wer"] is not None:
            label += f"  {r['wer']:.1f}" + ("  (= ASR floor)" if is_anchor else "")
        ax.annotate(label, (r["x"], r["y"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8 if fg else 7.2, color=color,
                    fontweight="bold" if is_anchor else "normal",
                    ha=ha, va="center", zorder=4 if fg else 3)

    ax.set_xticks([0.4, 0.5, 0.7, 1.0, 1.4, 2.0])
    ax.set_xticklabels(["0.4×", "0.5×", "0.7×", "1×", "1.4×", "2×"])
    ax.minorticks_off()
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0", "20", "40", "60%"])
    ax.tick_params(colors=MID, labelsize=9, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)

    ax.set_xlabel("rhythm spread vs real speech (burstiness std ratio, log scale)",
                  color=INK, fontsize=10)
    if first:
        ax.set_ylabel("utterances containing a ≥200 ms pause/stall", color=INK, fontsize=10)

    ns = [r["n"] for r in rows]
    ax.text(0, 1.04, f"{SET_LABEL[test_set]} · {lang} · {len(rows)} systems, "
            f"{min(ns)}–{max(ns)} heard utterances each · label = WER_CLIP %",
            transform=ax.transAxes, color=MUT, fontsize=9.5)


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

    fig, axes = plt.subplots(1, len(sets), figsize=(9.6 * len(sets), 7.0), dpi=200,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(PAPER)
    for i, (ax, test_set) in enumerate(zip(axes[0], sets)):
        csv_path = os.path.join(REPO, "results", f"rhythm_{test_set}_{args.lang}.csv")
        rows, anchor = load(csv_path, test_set, args.lang)
        draw(ax, rows, anchor, test_set, args.lang, first=i == 0)

    axes[0][0].text(0, 1.085, "Rhythm: flattened vs erratic",
                    transform=axes[0][0].transAxes, color=INK, fontsize=15, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
