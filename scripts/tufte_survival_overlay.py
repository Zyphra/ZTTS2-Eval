# tufte_survival_overlay.py  --  single-panel reliability survival curves, Tufte
# style: no gridlines, range-frame axes, a few emphasised curves directly
# labelled in colour, the rest a faint grey bundle. P(edits >= k) on log-log.
#
# Generic over the current eval: a "series" is chosen by --label (language by
# default). Emphasised series default to best/worst/most-catastrophic (auto);
# override with --highlight. See tufte_common.py.
#
#   python tufte_survival_overlay.py                      # per-lang GT (clean)
#   python tufte_survival_overlay.py --limit 12 --highlight en,zh,ru
#
# Requires: numpy, matplotlib   (no scipy)
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tufte_common as tc

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
tc.add_common_args(ap, "results/gt_clean/*/wav_res_ref_text.wer.raw",
                   "ztts_reliability_overlay_tufte.png")
args = ap.parse_args()

E, meta = tc.load_series(args.globs, args.label, args.label_re, args.rename, args.limit)
curves = {m: tc.survxy(E[m]) for m in E}
emph = dict(tc.resolve_highlight(E, args.highlight))

PAPER, INK, MUT, GREY = tc.PAPER, tc.INK, tc.MUT, tc.GREY
plt.rcParams.update({"font.family": "DejaVu Serif", "text.color": INK})
fig, ax = plt.subplots(figsize=(10.4, 6.6), dpi=150)
fig.subplots_adjust(left=0.075, right=0.965, top=0.83, bottom=0.165)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)

for mdl in E:                                    # faint bundle first
    if mdl in emph:
        continue
    x, yv = curves[mdl]
    ax.step(x, yv, where="post", color=GREY, lw=0.9, zorder=2)
nhl = max(len(emph) - 1, 1)
for i, (mdl, c) in enumerate(emph.items()):      # emphasised on top, direct-labelled
    x, yv = curves[mdl]
    ax.step(x, yv, where="post", color=c, lw=1.9, zorder=4)
    ax.scatter([x[-1]], [yv[-1]], s=14, color=c, zorder=5)
    j = int(round((0.25 + 0.5 * i / nhl) * (len(x) - 1)))   # stagger anchors so labels don't stack
    up = 1.55 if i % 2 == 0 else 0.62
    ax.text(x[j] * 1.05, yv[j] * up, mdl, color=c, fontsize=9,
            ha="left", va="bottom" if up >= 1 else "top", zorder=6)

# axis extents from the data
xmax = max(int(e.max()) for e in E.values())
ymin = min(min(yv) for _, yv in curves.values())
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.9, xmax * 1.15); ax.set_ylim(ymin * 0.7, 120)


def _decades(lo, hi):
    t, v = [], 10 ** int(np.floor(np.log10(lo)))
    while v <= hi:
        t.append(v); v *= 10
    return t


xt = [t for t in [1, 3, 10, 30, 100, 300, 1000] if t <= xmax * 1.15]
yt = [t for t in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] if t >= ymin * 0.7 and t <= 120]
ax.set_xticks(xt); ax.set_xticklabels([str(t) for t in xt], fontsize=8, color=MUT)
ax.set_yticks(yt)
ax.set_yticklabels([("%g" % t) for t in yt], fontsize=8, color=MUT)
ax.tick_params(length=3, colors=MUT); ax.minorticks_off()
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(MUT); ax.spines[s].set_linewidth(0.8)
ax.spines["bottom"].set_bounds(xt[0], xt[-1])
ax.spines["left"].set_bounds(yt[0], yt[-1])
ax.set_xlabel("edit distance  k   (edits in one utterance)", color=MUT, fontsize=9)
ax.set_ylabel("% of utterances at least this bad     P(edits ≥ k)", color=MUT, fontsize=9)

title = args.title or "How reliable is each series?  —  the tail is the whole story"
sub = args.subtitle or ("Lower curve = more reliable.  " + tc.default_subtitle(E, meta))
fig.text(0.075, 0.945, title, color=INK, fontsize=13, ha="left")
fig.text(0.075, 0.895, sub, color=MUT, fontsize=8.3, ha="left")
fig.text(0.075, 0.045,
         "Curves separate where thousands of events support them; the extreme tail is individual failures, not a rankable rate.",
         color=MUT, fontsize=7.4, ha="left")
rest = [m for m in E if m not in emph]
if rest:
    foot = "Grey: " + ", ".join(rest) if len(rest) <= 12 else f"Grey: the remaining {len(rest)} series."
    fig.text(0.075, 0.020, foot, color=MUT, fontsize=7.4, ha="left")

plt.savefig(args.out, facecolor=PAPER)
print("wrote", args.out)
