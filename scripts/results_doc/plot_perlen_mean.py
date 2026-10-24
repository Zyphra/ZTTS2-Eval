#!/usr/bin/env python3
"""Cross-lang means per text-length bucket — simple mean vs weighted, all-lang vs core-9.

A line per model is unreadable past ~8 systems, so this plot draws a curated
headline subset (override with PERLEN_PLOT_MODELS=comma,separated,labels),
filtered to what the cache actually contains — a label absent from the cache is
dropped with a warning instead of silently plotting nothing.
"""
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(REPO_ROOT, "assets", "plots", "voxblink_perlen_crosslang.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import ordered_langs  # noqa: E402

simple = json.load(open(os.path.join(CACHE, "voxblink_perlen.json")))
weighted = json.load(open(os.path.join(CACHE, "voxblink_perlen_weighted.json")))
if not simple:
    sys.exit("plot_perlen_mean: voxblink_perlen.json is empty — rerun build_perlen.py")

VOX_LANGS = ordered_langs({k.split("|", 1)[1] for k in simple})
CORE9 = ["en","zh","de","es","fr","it","ja","ko","ru"]
N_BUCKETS = 6

DEFAULT_MODELS = ["gt", "fish-s2-pro", "voxcpm2", "qwen3-1.7b",
                  "zonos2_accurate", "zonos2_expressive"]
_want = [m for m in os.environ.get("PERLEN_PLOT_MODELS", ",".join(DEFAULT_MODELS)).split(",")
         if m.strip()]
_have = {k.split("|", 1)[0] for k in simple}
for m in _want:
    if m not in _have:
        print(f"plot_perlen_mean: '{m}' not in cache — dropped", file=sys.stderr)
MODELS = [(m, "ground-truth" if m == "gt" else m) for m in _want if m in _have]
if not MODELS:
    sys.exit("plot_perlen_mean: none of the requested models are in the cache")

def crossmean(d, label, lang_subset):
    cols = [[] for _ in range(N_BUCKETS)]
    for lang in lang_subset:
        vals = d.get(f"{label}|{lang}")
        if not vals: continue
        for j, v in enumerate(vals):
            if v is not None:
                cols[j].append(v)
    return [(sum(c)/len(c)) if c else None for c in cols]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

def draw_lines(ax, d, lang_subset, title):
    xs = list(range(N_BUCKETS))
    for label, t in MODELS:
        ys = crossmean(d, label, lang_subset)
        ys_plot = [y if y is not None else np.nan for y in ys]
        ax.plot(xs, ys_plot, marker="o", label=t, linewidth=2 if "zonos2" in t else 1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"q{i}" for i in xs], fontsize=9)
    ax.set_xlabel("text-length quantile (q0 shortest → q5 longest)", fontsize=9)
    ax.set_ylabel("WER_CLIP across langs", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

draw_lines(axes[0,0], simple,   VOX_LANGS, f"Simple mean across utts — All {len(VOX_LANGS)} langs")
draw_lines(axes[0,1], weighted, VOX_LANGS, f"Weighted mean (∑edits/∑ref_len) — All {len(VOX_LANGS)} langs")
draw_lines(axes[1,0], simple,   CORE9, "Simple mean — Core 9 langs")
draw_lines(axes[1,1], weighted, CORE9, "Weighted mean — Core 9 langs")

fig.suptitle(
    "Voxblink2 length-degradation curves — simple vs weighted mean.\n"
    "Both show WER FALLING with length: longer refs in voxblink2 come from cleaner content (scripted/podcast vs casual snippets) — "
    "the trend is dataset selection, not a TTS length-extrapolation property.",
    fontsize=11, y=1.04)

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
