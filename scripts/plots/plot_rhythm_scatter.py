"""Rhythm flattening-vs-stalling scatter (VoxBlink/en by default).

One dot per system: x = burstiness std_ratio (system spread / real-speech
spread, log scale — < 1 is rhythm flattened, > 1 erratic), y = long_pct (% of
utterances containing a >=200 ms pause/stall run). GT (the real target
recordings) is the visual anchor near (1.0, 31 %); the reference prompt set is
x=1.0 by construction. The causal story — WER/ASR-optimized systems flatten
rhythm — is the lower-left cluster; erratic stallers are upper-right; good
means near GT, not at either extreme.

Reads results/rhythm_<set>_<lang>.csv (scripts/prosody/rhythm_dist.py output).

Usage:
    python scripts/plots/plot_rhythm_scatter.py
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

# foreground layer: the systems the title claim is about (flattener exemplars,
# the near-GT case, the two counterexamples, the erratic extremes). Everything
# else stays plotted + labeled but recedes to the muted layer.
FOREGROUND = {
    "GT (real speech)", "kokoro", "fish-s2-pro", "qwen3-1.7b", "voxcpm2",
    "moss-tts", "zonos2-accurate", "voxtral", "glm-tts", "zonos2-expressive",
    "omnivoice-noref",
}

# label offset (points) and horizontal alignment per system, tuned by eye.
# default is a small right-offset; entries here override collisions.
NUDGE = {
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
}


def wer_clip(system, test_set, lang):
    """WER_CLIP % from the system's .wer summary line (nested or flat layout)."""
    for p in (os.path.join(REPO, "results", system, test_set, lang, "wav_res_ref_text.wer"),
              os.path.join(REPO, "results", system, lang, "wav_res_ref_text.wer")):
        if os.path.isfile(p):
            with open(p) as fh:
                for line in fh:
                    if "WER_CLIP:" in line and "CI95" not in line:
                        return float(line.rsplit("WER_CLIP:", 1)[1].strip().rstrip(","))
    return None


def load(csv_path, test_set, lang):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["system"] == "__CEILING_split_half__":
                continue
            rows.append(dict(system=SHORT.get(r["system"], r["system"]),
                             wer=wer_clip(r["system"], test_set, lang),
                             x=float(r["burstiness_std_ratio"]),
                             y=float(r["long_pct"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", default="in-the-wild")
    ap.add_argument("--lang", default="en")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    csv_path = os.path.join(REPO, "results", f"rhythm_{args.test_set}_{args.lang}.csv")
    rows = load(csv_path, args.test_set, args.lang)
    out = args.out or os.path.join(
        REPO, "assets", "plots", f"rhythm_flatten_vs_stall_{args.test_set}_{args.lang}.png")

    fig, ax = plt.subplots(figsize=(9.6, 7.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    ax.set_xscale("log")
    ax.set_xlim(0.32, 2.6)
    ax.set_ylim(-2, 74)

    # reference geometry: spread parity line + real-speech pause incidence
    gt = next(r for r in rows if r["system"].startswith("GT"))
    ax.axvline(1.0, color=GREY, lw=0.8, zorder=1)
    ax.axhline(gt["y"], color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(1.0, 72.5, "same spread as real speech", color=MUT, fontsize=8,
            ha="center", va="top")
    ax.text(0.33, gt["y"] + 1.2, f"real-speech pause incidence ({gt['y']:.0f}%)",
            color=MUT, fontsize=8, ha="left", va="bottom")

    # regime annotations
    ax.text(0.335, 3.5, "flattened, pause-poor\n(ASR-optimized rhythm)",
            color=MUT, fontsize=9, ha="left", va="bottom", style="italic")
    ax.text(2.52, 66, "erratic, stall-heavy", color=MUT, fontsize=9,
            ha="right", va="top", style="italic")

    for r in rows:
        is_gt = r["system"].startswith("GT")
        fg = r["system"] in FOREGROUND
        color = BLUE if is_gt else (INK if fg else MUT)
        ax.scatter(r["x"], r["y"], s=46 if is_gt else (24 if fg else 16),
                   color=color, zorder=3 if fg else 2,
                   edgecolors=PAPER, linewidths=0.8)
        dx, dy, ha = NUDGE.get(r["system"], (5, 0, "left"))
        # Tufte: WER rides in the label as printed data, not a size/color code
        label = r["system"]
        if r["wer"] is not None:
            label += f"  {r['wer']:.1f}" + ("  (= ASR floor)" if is_gt else "")
        ax.annotate(label, (r["x"], r["y"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8 if fg else 7.2, color=color,
                    fontweight="bold" if is_gt else "normal",
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
    ax.set_ylabel("utterances containing a ≥200 ms pause/stall", color=INK, fontsize=10)

    ax.text(0, 1.085, "WER is won by flattening rhythm",
            transform=ax.transAxes, color=INK, fontsize=15, fontweight="bold")
    ax.text(0, 1.04, "VoxBlink2 in-the-wild · en · n=1500/system · label = "
            "WER_CLIP % · good = near GT, not at either extreme",
            transform=ax.transAxes, color=MUT, fontsize=9.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
