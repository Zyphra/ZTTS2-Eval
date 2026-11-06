# tufte_slopegraph.py  --  slopegraph of P(edits >= k) at several severity
# thresholds. Companion to the survival plots: the curves show the shape of the
# tail, this shows the *ranking* at each severity level, with rank crossings
# visible as line crossings. No axes: the numbers are on the chart (log-spaced
# vertically). A few series are emphasised in colour (auto best/worst, or
# --highlight); the rest are mid-grey.
#
# Generic over the current eval: a "series" is chosen by --label (language by
# default). With many series a slopegraph gets crowded — use --limit to keep the
# worst-N. See tufte_common.py.
#
#   python tufte_slopegraph.py --limit 12                 # per-lang GT (clean)
#   python tufte_slopegraph.py --ks 1,3,5,10 --highlight en,ru,zh
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
                   "ztts_reliability_slopegraph.png")
ap.add_argument("--ks", default="1,3,5,10", help="severity thresholds (default: 1,3,5,10)")
args = ap.parse_args()

KS = [int(x) for x in args.ks.split(",")]
E, meta = tc.load_series(args.globs, args.label, args.label_re, args.rename, args.limit)
if args.limit == 0 and len(E) > 16:
    print(f"note: {len(E)} series in one slopegraph is busy; consider --limit 12")

P = {m: [(e >= k).mean() * 100 for k in KS] for m, e in E.items()}
col_for_order = min(1, len(KS) - 1)              # order by the 2nd threshold column
models = sorted(P, key=lambda m: P[m][col_for_order])
emph = dict(tc.resolve_highlight(E, args.highlight))

PAPER, INK, MUT, MID = tc.PAPER, tc.INK, tc.MUT, tc.MID


def col(m):
    return emph.get(m, MID)


def fmt(v):
    return f"{v:.1f}" if v >= 1 else f"{v:.2f}"


plt.rcParams.update({"font.family": "DejaVu Serif", "text.color": INK})
fig, ax = plt.subplots(figsize=(9.4, max(5.0, 0.42 * len(models) + 2.4)), dpi=150)
fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.10)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)


def nudge(vals, gap=0.082):                       # spread crowded labels, keep exact order
    vals = np.array(vals, dtype=float)
    idx = np.argsort(vals)
    pos = vals.copy()
    prev = -np.inf
    for i in idx:
        pos[i] = max(pos[i], prev + gap)
        prev = pos[i]
    return pos - (pos.mean() - vals.mean())


X = list(range(len(KS)))
floor = min(min(v for v in P[m] if v > 0) for m in models) if any(any(v > 0 for v in P[m]) for m in models) else 0.01
L = {m: [np.log10(max(v, floor * 0.5)) for v in P[m]] for m in models}
tpos = np.column_stack([nudge([L[m][j] for m in models]) for j in range(len(KS))])

for i, m in enumerate(models):
    acc = m in emph
    ax.plot(X, tpos[i], color=col(m), lw=1.7 if acc else 1.0, zorder=3 if acc else 2,
            solid_capstyle="round")
    for j in range(len(KS)):
        ax.text(X[j], tpos[i, j], fmt(P[m][j]), ha="center", va="center",
                fontsize=7.6, color=col(m), zorder=4,
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2))
    ax.text(X[0] - 0.10, tpos[i, 0], m, ha="right", va="center", fontsize=9, color=col(m))
    ax.text(X[-1] + 0.10, tpos[i, -1], m, ha="left", va="center", fontsize=9, color=col(m))

hy = tpos.max() + 0.16
for j, k in enumerate(KS):
    ax.text(X[j], hy, f"≥ {k} edit" + ("s" if k > 1 else ""), ha="center",
            va="bottom", fontsize=9.5, color=INK)

ax.set_xlim(X[0] - 1.15, X[-1] + 1.15); ax.set_ylim(tpos.min() - 0.14, hy + 0.16)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

title = args.title or "The ranking, column by column"
sub = args.subtitle or ("% of utterances with at least k edits, at "
                        f"{len(KS)} thresholds. Vertical order = ranking; crossings = rank changes.")
fig.text(0.03, 0.945, title, color=INK, fontsize=13, ha="left")
fig.text(0.03, 0.900, sub, color=MUT, fontsize=8.3, ha="left")
fig.text(0.03, 0.872, "Spacing ~log, nudged for legibility · " + tc.default_subtitle(E, meta),
         color=MUT, fontsize=8.3, ha="left")

plt.savefig(args.out, facecolor=PAPER)
print("wrote", args.out)
