# tufte_survival_smallmultiples.py  --  small multiples of reliability survival
# curves, Tufte style: no gridlines, range-frame axes, titles/stats outside the
# data region. One panel per series (dark) against all others (faint).
#
# Generic over the current eval: a "series" is chosen by --label (language by
# default). The grid auto-sizes to the number of series. See tufte_common.py.
#
#   python tufte_survival_smallmultiples.py               # per-lang GT (clean)
#   python tufte_survival_smallmultiples.py --limit 16 --cols 4
#
# Requires: numpy, matplotlib   (no scipy)
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tufte_common as tc

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
tc.add_common_args(ap, "results/gt_zero_shot/*/wav_res_ref_text.wer.raw",
                   "ztts_reliability_smallmultiples_tufte.png")
ap.add_argument("--cols", type=int, default=0, help="panel columns (0 = auto ~sqrt)")
ap.add_argument("--blowup", type=int, default=40,
                help="edits marking a catastrophic worst-case, flagged red (default: 40)")
args = ap.parse_args()

E, meta = tc.load_series(args.globs, args.label, args.label_re, args.rename, args.limit)
curves = {m: tc.survxy(E[m]) for m in E}
models = sorted(E, key=lambda m: (E[m] >= 3).mean())      # most -> least reliable

PAPER, INK, MUT, ENS, RED = tc.PAPER, tc.INK, tc.MUT, tc.ENS, tc.RED
plt.rcParams.update({"font.family": "DejaVu Serif", "text.color": INK})

ncol = args.cols or max(1, int(np.ceil(np.sqrt(len(models)))))
nrow = int(np.ceil(len(models) / ncol))
fig = plt.figure(figsize=(2.5 * ncol + 1.2, 2.7 * nrow + 2.2), dpi=150)
fig.patch.set_facecolor(PAPER)
gs = GridSpec(nrow, ncol, left=0.075, right=0.975, top=0.86, bottom=0.075,
              hspace=0.55, wspace=0.14)

xmax = max(int(e.max()) for e in E.values())
ymin = min(min(yv) for _, yv in curves.values())
XL, YL = (0.9, xmax * 1.1), (ymin * 0.7, 120)
xt = [t for t in [1, 10, 100, 1000] if t <= XL[1]]
yt = [t for t in [0.01, 0.1, 1, 10, 100] if YL[0] <= t <= YL[1]]

for i, m in enumerate(models):
    r, c = divmod(i, ncol)
    ax = fig.add_subplot(gs[r, c]); ax.set_facecolor(PAPER)
    for mm in models:
        x, yv = curves[mm]; ax.step(x, yv, where="post", color=ENS, lw=0.8, zorder=1)
    x, yv = curves[m]; ax.step(x, yv, where="post", color=INK, lw=1.8, zorder=3)
    mx = int(E[m].max())
    if mx >= args.blowup:
        ax.scatter([x[-1]], [yv[-1]], s=13, color=RED, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*XL); ax.set_ylim(*YL)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUT); ax.spines[s].set_linewidth(0.7)
    ax.spines["bottom"].set_bounds(xt[0], xt[-1])
    ax.spines["left"].set_bounds(yt[0], yt[-1])
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.tick_params(length=2.5, colors=MUT, labelsize=7)
    ax.minorticks_off()
    ax.set_xticklabels([str(t) for t in xt] if r == nrow - 1 else [])
    ax.set_yticklabels([("%g" % t) for t in yt] if c == 0 else [])
    p99 = int(round(np.percentile(E[m], 99)))
    ax.text(0.0, 1.16, m, transform=ax.transAxes, fontsize=9.3, ha="left", va="bottom", color=INK)
    ax.text(0.0, 1.05, f"p99 = {p99}   ·   worst = {mx}   ·   n = {meta[m]['n']}",
            transform=ax.transAxes, fontsize=6.9, ha="left", va="bottom",
            color=(RED if mx >= args.blowup else MUT))

title = args.title or "How reliable is each series?"
sub = args.subtitle or tc.default_subtitle(E, meta)
fig.text(0.075, 0.955, title, color=INK, fontsize=15, ha="left")
fig.text(0.075, 0.925,
         "Each panel: one series (dark) against all others (faint). Lower curve = more reliable; "
         "panels ordered most → least reliable (share with ≥3 edits).",
         color=MUT, fontsize=8.3, ha="left")
fig.text(0.075, 0.900, "P(edits ≥ k) on log–log axes  ·  " + sub, color=MUT, fontsize=8.3, ha="left")
fig.text(0.5, 0.040, "edit distance  k  (edits in one utterance)", color=MUT, fontsize=9, ha="center")
fig.text(0.020, 0.46, "% of utterances at least this bad", color=MUT, fontsize=9, rotation=90, va="center")
fig.text(0.075, 0.016,
         f"Red dot = a catastrophic worst case (≥ {args.blowup} edits): the extreme tail is individual failures, not a rankable rate.",
         color=MUT, fontsize=7.3, ha="left")

plt.savefig(args.out, facecolor=PAPER)
print("wrote", args.out)
