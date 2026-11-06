#!/usr/bin/env python3
"""WER-vs-prosody tradeoff figures: assets/plots/tradeoff_<task>.png.

One figure per task (clean = Clean, in-the-wild = ITW), two panels:

  x: macro WER_CLIP (log scale)      y: macro rhythm pause factor (left) or
                                        macro pitch-range factor (right), both in
                                        units of the real-speech split-half ceiling

so a system that is accurate AND prosodically natural sits at the bottom left. The
dashed horizontal line y = 1 is the real-speech split-half ceiling; the dashed
vertical line is the ground-truth WER_CLIP (the ASR floor). Markers encode the
conditioning class (rd_common.condition); the ground truth is a star.

Reads the snapshot written by build_results.py (same loader and the same macro rule
as gen_md.py: unweighted mean over the task's frozen basket, defined only when every
basket language has the value -- so only full-coverage systems are plotted).
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "assets", "plots")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from utils.benchmark import langs_for  # noqa: E402
from rd_common import (  # noqa: E402
    condition, index_records, label, load_snapshot, macro)

TASKS = (("clean", "Clean (FLEURS-R)"), ("in-the-wild", "ITW (VoxBlink2)"))
PANELS = (("rhythm_x", "Rhythm pause factor (x split-half ceiling)"),
          ("pitch_range_x", "Pitch-range distance (x split-half ceiling)"))
CLASS_MARKER = {"with-transcript": "o", "no-transcript": "^", "x-vector": "D", "preset": "s"}
CLASS_NAME = {"with-transcript": "with transcript", "no-transcript": "no transcript",
              "x-vector": "x-vector only", "preset": "preset voice"}
UNCLASSIFIED_MARKER = "P"


def marker_for(model):
    cond = condition(model)
    return CLASS_MARKER[cond[0]] if cond else UNCLASSIFIED_MARKER


def points(idx, recs, task, field):
    """[(model, macro WER_CLIP, macro `field`)] over full-coverage systems, GT included."""
    langs = langs_for(task)
    models = sorted({r["model"] for r in recs if r["task"] == task}
                    | {f"gt_{task}"})
    out = []
    for m in models:
        x = macro(idx, m, task, "wer", langs)
        y = macro(idx, m, task, field, langs)
        if x is None or y is None or x <= 0:
            continue
        out.append((m, x, y))
    return out


def plot_task(plt, idx, recs, task, title, out_path):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    gt_model = f"gt_{task}"
    gt_x = macro(idx, gt_model, task, "wer", langs_for(task))
    used = set()
    for ax, (field, ylabel) in zip(axes, PANELS):
        pts = points(idx, recs, task, field)
        for m, x, y in pts:
            if m == gt_model:
                ax.scatter([x], [y], marker="*", s=220, c="black", zorder=4)
            else:
                mk = marker_for(m)
                used.add(mk)
                ax.scatter([x], [y], marker=mk, s=55, c="tab:blue", edgecolors="black",
                           linewidths=0.5, zorder=3)
            ax.annotate(label(m), (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=7)
        ax.axhline(1.0, ls="--", c="gray", lw=1, zorder=1)
        if gt_x is not None:
            ax.axvline(gt_x, ls="--", c="gray", lw=1, zorder=1)
        ax.set_xscale("log")
        # plain tick labels: WER spans a narrow range, where the default log
        # formatter prints unreadable 6.4 x 10^0 style ticks
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2g"))
        ax.set_xlabel("macro WER_CLIP % (log scale; lower = more accurate)")
        ax.set_ylabel(ylabel + "\n(lower = closer to real speech)")
        ax.grid(alpha=0.25)
        if not pts:
            ax.text(0.5, 0.5, "no full-coverage system has both values", ha="center",
                    va="center", transform=ax.transAxes, color="gray")
    handles = [Line2D([], [], marker="*", ls="", c="black", ms=13, label="ground truth")]
    for cls, mk in CLASS_MARKER.items():
        if mk in used:
            handles.append(Line2D([], [], marker=mk, ls="", mfc="tab:blue", mec="black",
                                  ms=7, label=CLASS_NAME[cls]))
    if UNCLASSIFIED_MARKER in used:
        handles.append(Line2D([], [], marker=UNCLASSIFIED_MARKER, ls="", mfc="tab:blue",
                              mec="black", ms=7, label="unclassified"))
    handles.append(Line2D([], [], ls="--", c="gray", label="real-speech ceiling / GT WER"))
    axes[0].legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle(f"{title}: accuracy vs prosody (full-coverage systems only)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        recs = load_snapshot()
    except OSError:
        sys.exit("plot_tradeoff: results_snapshot.json missing -- run build_results.py first")
    idx = index_records(recs)
    for task, title in TASKS:
        out = os.path.join(OUT_DIR, f"tradeoff_{task}.png")
        plot_task(plt, idx, recs, task, title, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
