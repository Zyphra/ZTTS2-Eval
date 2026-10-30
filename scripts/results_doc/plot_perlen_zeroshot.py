#!/usr/bin/env python3
"""Plot per-length-bucket WER on the controlled-content zero_shot set side-by-side
with voxblink — exposing voxblink's content-selection confound and the real U-shape."""
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(REPO_ROOT, "assets", "plots", "perlen_zeroshot_vs_voxblink.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

simple_vox = json.load(open(os.path.join(CACHE, "voxblink_perlen.json")))
simple_zs = json.load(open(os.path.join(CACHE, "zeroshot_perlen_simple.json")))

CORE9 = ["en","zh","de","es","fr","it","ja","ko","ru"]
N_BUCKETS = 6

# Curated headline subset (a line per discovered system is unreadable);
# override with PERLEN_PLOT_MODELS=comma,separated,labels. Labels absent from a
# panel's cache are skipped for that panel.
DEFAULT_MODELS = ["fish-s2-pro", "voxcpm2", "qwen3-1.7b",
                  "zonos2_accurate", "zonos2_expressive",
                  "zonos3-mini_reference", "zonos3-mini_embedding"]
MODELS = [(m, m) for m in
          os.environ.get("PERLEN_PLOT_MODELS", ",".join(DEFAULT_MODELS)).split(",")
          if m.strip()]

def cross(d, label, langs):
    cols = [[] for _ in range(N_BUCKETS)]
    for lang in langs:
        v = d.get(f"{label}|{lang}")
        if not v: continue
        for j, x in enumerate(v):
            if x is not None: cols[j].append(x)
    return [(sum(c)/len(c)) if c else np.nan for c in cols]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

def draw(ax, d, langs, title):
    xs = list(range(N_BUCKETS))
    have = {k.split("|", 1)[0] for k in d}
    for label, t in MODELS:
        if label not in have:
            print(f"plot_perlen_zeroshot: '{label}' not in cache for '{title.splitlines()[0]}'"
                  " — skipped", file=sys.stderr)
            continue
        ys = cross(d, label, langs)
        ax.plot(xs, ys, marker="o", label=t, linewidth=2 if "zonos" in t else 1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"q{i}" for i in xs], fontsize=9)
    ax.set_xlabel("text-length quantile (q0 shortest → q5 longest, per-lang)", fontsize=9)
    ax.set_ylabel("mean WER_CLIP across langs", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

draw(axes[0], simple_vox, CORE9, "voxblink2 (in-the-wild, core 9 langs)\nWER falls monotonically — content-selection confound")
draw(axes[1], simple_zs, CORE9, "zero_shot (FLEURS-R, controlled content, core 9 langs)\nU-shape: short refs noisy, mid optimal, longest climb back")

fig.suptitle(
    "Length-degradation: voxblink2 vs controlled-content zero_shot.\n"
    "The voxblink monotonic decrease was a content-selection artifact. On controlled content the real shape is U: "
    "q0 hits the short-ref WER noise floor, mid buckets are best, and q5 (longest refs, >200 chars for EU langs) climbs back +2pp for Zonos2 / voxcpm2 — real length-degradation.",
    fontsize=10.5, y=1.06)

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
