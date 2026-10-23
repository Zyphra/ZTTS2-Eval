# tufte_p99.py  --  p99 edit distance per series with bootstrap 95% CI, Tufte
# style: no gridlines, range-frame axis, CI drawn as a thin line gapped at the
# estimate (Tufte's quartile-plot idiom), captions clear of the data region.
#
# Generic over the current eval: a "series" is chosen by --label (language by
# default) and read from whatever --glob points at. See tufte_common.py.
#
#   python tufte_p99.py                                   # per-lang GT (clean)
#   python tufte_p99.py 'results/gt_zero_shot_voxblink/*/wav_res_ref_text.wer.raw'
#   python tufte_p99.py 'results/{gt_zero_shot,whisper_zero_shot}/de/*.wer.raw' \
#          --label set --rename 'gt_zero_shot=Qwen routed,whisper_zero_shot=Whisper'
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
tc.add_common_args(ap, "results/gt_zero_shot/*/wav_res_ref_text.wer.raw", "ztts_p99_tufte.png")
ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (default: 2000)")
args = ap.parse_args()

E, meta = tc.load_series(args.globs, args.label, args.label_re, args.rename, args.limit)

rng = np.random.default_rng(0)
rows = {}
for m, e in E.items():
    p99 = np.percentile(e, 99)
    bs = np.array([np.percentile(rng.choice(e, size=len(e), replace=True), 99)
                   for _ in range(args.boot)])
    rows[m] = (p99, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(e))
order = sorted(rows, key=lambda m: rows[m][0])

PAPER, INK, MUT = tc.PAPER, tc.INK, tc.MUT
plt.rcParams.update({"font.family": "DejaVu Serif", "text.color": INK})
fig, ax = plt.subplots(figsize=(9.2, max(3.2, 0.42 * len(order) + 1.8)), dpi=150)
fig.subplots_adjust(left=0.26, right=0.93, top=0.82, bottom=0.16)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)

xmax = max(hi for _, _, hi, _ in rows.values())
span = max(xmax, 1.0)
GAP = 0.008 * span                              # white gap either side of the dot
lab_x = -0.03 * span                            # series names, left of the frame
y = np.arange(len(order))[::-1]
for yi, m in zip(y, order):
    p, lo, hi, n = rows[m]
    small = n < args.small_n
    c = MUT if small else INK
    if lo < p - GAP:
        ax.plot([lo, p - GAP], [yi, yi], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
    if hi > p + GAP:
        ax.plot([p + GAP, hi], [yi, yi], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
    ax.scatter([p], [yi], s=22, color=c, zorder=4)
    ax.text(lab_x, yi, m, ha="right", va="center", fontsize=9.3, color=INK)
    note = f"   n = {n}" if small else ""
    ax.text(hi + 0.015 * span, yi, f"{p:.0f}{note}", ha="left", va="center", fontsize=8, color=MUT)

# nice round tick ceiling
step = max(1, round(span / 5))
ticks = list(range(0, int(np.ceil(xmax / step)) * step + 1, step))
ax.set_xlim(-0.02 * span, ticks[-1] * 1.02)
ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_yticks([])
ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=8, color=MUT)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_bounds(0, ticks[-1])
ax.spines["bottom"].set_color(MUT); ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(axis="x", length=3, colors=MUT); ax.tick_params(axis="y", length=0)
ax.set_xlabel("p99 edit distance — 99% of utterances are within this many edits (fewer = more reliable)",
              color=MUT, fontsize=8.7, labelpad=8)

title = args.title or "Reliability at the 99th percentile"
sub = args.subtitle or ("Dot = p99 edit distance, line = bootstrap 95% CI. " + tc.default_subtitle(E, meta))
fig.text(0.055, 0.945, title, color=INK, fontsize=13, ha="left")
fig.text(0.055, 0.885, sub, color=MUT, fontsize=8.2, ha="left")
fig.text(0.055, 0.045,
         "Overlapping intervals are not distinguishable. Grey rows rest on fewer than "
         f"{args.small_n} utterances (n shown).",
         color=MUT, fontsize=7.3, ha="left")

plt.savefig(args.out, facecolor=PAPER)
print("wrote", args.out)
