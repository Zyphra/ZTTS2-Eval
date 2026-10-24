#!/usr/bin/env python3
"""Grid of per-text-length-bucket voxblink WER_CLIP heatmaps (one panel per model).

Models and languages are discovered from the cache JSON (build_perlen.py already
discovers every system in the results tree) — no hard-coded lists to go stale.
"""
import math
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(REPO_ROOT, "assets", "plots", "voxblink_perlen_heatmaps.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import ordered_langs  # noqa: E402

perlen = json.load(open(os.path.join(CACHE, "voxblink_perlen.json")))
if not perlen:
    sys.exit("plot_perlen: voxblink_perlen.json is empty — rerun build_perlen.py")

VOX_LANGS = ordered_langs({k.split("|", 1)[1] for k in perlen})
N_BUCKETS = max((len(v) for v in perlen.values()), default=6)

_labels = sorted({k.split("|", 1)[0] for k in perlen},
                 key=lambda l: (l != "gt", l))
MODELS = [(l, "ground-truth source" if l == "gt" else l) for l in _labels]

def matrix(label):
    M = np.full((len(VOX_LANGS), N_BUCKETS), np.nan)
    for i, lang in enumerate(VOX_LANGS):
        vals = perlen.get(f"{label}|{lang}")
        if not vals: continue
        for j, v in enumerate(vals):
            if v is not None:
                M[i, j] = v
    return M

NCOLS = 4
NROWS = max(1, math.ceil(len(MODELS) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 5.5 * NROWS),
                         constrained_layout=True, squeeze=False)
vmax = 100
cmap = plt.cm.viridis_r

for ax in axes.flat[len(MODELS):]:
    ax.set_visible(False)

for ax, (label, title) in zip(axes.flat, MODELS):
    M = matrix(label)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nmean WER_CLIP per (lang, text-length quantile)", fontsize=10)
    ax.set_xticks(range(N_BUCKETS))
    ax.set_xticklabels([f"q{i}" for i in range(N_BUCKETS)], fontsize=8)
    ax.set_yticks(range(len(VOX_LANGS)))
    ax.set_yticklabels(VOX_LANGS, fontsize=8)
    ax.set_xlabel("text-length quantile (q0=shortest 1/6 → q5=longest 1/6, per-lang)", fontsize=8)
    for i in range(len(VOX_LANGS)):
        for j in range(N_BUCKETS):
            v = M[i, j]
            if np.isnan(v): continue
            ratio = min(v, vmax) / vmax
            color = "white" if ratio > 0.55 else "black"
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                    ha="center", va="center", fontsize=7, color=color)

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
cbar.set_label("WER_CLIP %", fontsize=10)

fig.suptitle(f"Voxblink2 per-text-length-bucket WER_CLIP — rows = {len(VOX_LANGS)} languages, "
             "cols = REF char-count quantiles (per-lang)",
             fontsize=13, y=1.02)

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
