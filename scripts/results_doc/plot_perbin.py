#!/usr/bin/env python3
"""Grid of per-MSR-UTMOS-bin voxblink WER_CLIP heatmaps (one panel per model).

Models and languages are discovered from the cache JSON (build_perbin.py already
discovers every system in the results tree), so a newly scored system shows up
here on the next run instead of silently missing from a hard-coded list.
"""
import math
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(REPO_ROOT, "assets", "plots", "voxblink_perbin_heatmaps.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import ordered_langs, VOX_LANGS_EXCLUDED  # noqa: E402

perbin = json.load(open(os.path.join(CACHE, "voxblink_perbin.json")))
if not perbin:
    sys.exit("plot_perbin: voxblink_perbin.json is empty — rerun build_perbin.py")

# Bin count comes from the cache (build_perbin.py's N_BINS), not a hardcoded value.
N_BINS = max((len(v) for v in perbin.values()), default=5)

# `tl` excluded, matching RESULTS_CURRENT.md's aggregates (the ASR pipeline was
# unreliable on its reference audio — see the excluded-langs callout there).
VOX_LANGS = [l for l in ordered_langs({k.split("|", 1)[1] for k in perbin})
             if l not in VOX_LANGS_EXCLUDED]

# One panel per model in the cache: gt first, then everything else in name order.
_labels = sorted({k.split("|", 1)[0] for k in perbin},
                 key=lambda l: (l != "gt", l))
MODELS = [(l, "ground-truth source" if l == "gt" else l) for l in _labels]

NCOLS = 4
NROWS = max(1, math.ceil(len(MODELS) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 5.5 * NROWS),
                         constrained_layout=True, squeeze=False)
vmax = 100
cmap = plt.cm.viridis_r

def matrix(label):
    M = np.full((len(VOX_LANGS), N_BINS), np.nan)
    for i, lang in enumerate(VOX_LANGS):
        vals = perbin.get(f"{label}|{lang}")
        if not vals: continue
        for j, v in enumerate(vals):
            if v is not None:
                M[i, j] = v
    return M

for ax in axes.flat[len(MODELS):]:
    ax.set_visible(False)

for ax, (label, title) in zip(axes.flat, MODELS):
    M = matrix(label)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nmean WER_CLIP per (lang, MSR-UTMOS bin)", fontsize=10)
    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels([f"b{i}" for i in range(N_BINS)], fontsize=8)
    ax.set_yticks(range(len(VOX_LANGS)))
    ax.set_yticklabels(VOX_LANGS, fontsize=8)
    ax.set_xlabel(f"MSR-UTMOS bin (b0 worst → b{N_BINS - 1} best)", fontsize=8)
    for i in range(len(VOX_LANGS)):
        for j in range(N_BINS):
            v = M[i, j]
            if np.isnan(v): continue
            ratio = min(v, vmax) / vmax
            color = "white" if ratio > 0.55 else "black"
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                    ha="center", va="center", fontsize=6.5, color=color)

cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
cbar.set_label("WER_CLIP %", fontsize=10)

fig.suptitle(f"Voxblink2 per-bin WER_CLIP — rows = {len(VOX_LANGS)} languages, "
             "cols = MSR-UTMOS quality bins",
             fontsize=13, y=1.02)

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
