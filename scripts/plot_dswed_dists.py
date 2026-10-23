"""Plot per-utterance DS-WED distributions for every model that has a
DS-WED run on disk (results/<model>/dswed/zero_shot/en/dswed_per_uid.tsv).

Uses a KDE overlay with per-model mean/median markers and a boxplot panel that
makes the long right tail and outliers explicit.
Higher DS-WED = more seed-to-seed prosodic
variation. Writes assets/plots/dswed_distributions.png.
"""
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# dir -> short label, from the shared results_doc map so this can't drift;
# only the per-plot zonos2-stability variant stays local.
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "scripts" / "results_doc"))
from rd_common import label as _shared_label  # noqa: E402

_DSWED_DIRS = ["zonos2", "zonos2_stability", "omni_voxcpm2",
               "omni_qwen3-tts-1.7b", "omni_fish-s2-pro"]
LABELS = {d: _shared_label(d) for d in _DSWED_DIRS}
LABELS["zonos2_stability"] = "zonos2-stability"
# stable, distinguishable palette (Zonos warm, Omni cool)
COLORS = {
    "zonos2": "#9467bd",
    "zonos2-stability": "#17becf",
    "voxcpm2": "#2ca02c",
    "qwen3-1.7b": "#1f77b4",
    "fish-s2-pro": "#ff7f0e",
}


def load_per_uid(tsv):
    vals = []
    for ln in Path(tsv).read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 2 and p[1] not in ("NA", ""):
            try:
                vals.append(float(p[1]))
            except ValueError:
                pass
    return np.array(vals)


def collect():
    rows = []  # (label, color, array)
    for tsv in glob.glob(str(REPO / "results/*/dswed/zero_shot/en/dswed_per_uid.tsv")):
        model_dir = Path(tsv).parts[-5]
        label = LABELS.get(model_dir, model_dir)
        arr = load_per_uid(tsv)
        if arr.size:
            rows.append((label, COLORS.get(label, "#555555"), arr))
    rows.sort(key=lambda r: r[2].mean(), reverse=True)  # most diverse first
    return rows


def kde_or_hist(ax, x, label, color, xrange):
    x = x[np.isfinite(x)]
    if x.size == 0 or np.allclose(x, x[0]):
        return
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x)
        xs = np.linspace(xrange[0], xrange[1], 400)
        ys = kde(xs)
        ax.plot(xs, ys, color=color, lw=2, label=label)
        ax.fill_between(xs, ys, alpha=0.08, color=color)
    except Exception:
        ax.hist(x, bins=40, range=xrange, density=True, histtype="step",
                color=color, lw=2, label=label)


def main():
    rows = collect()
    if not rows:
        print("no DS-WED per-uid files found")
        return

    # KDE x-range: clip to p98 across all models so the bulk is visible
    allv = np.concatenate([a for _, _, a in rows])
    xhi = float(np.percentile(allv, 98))
    xrange = (0, xhi)

    fig, (ax_kde, ax_box) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1.4]})

    # --- top: KDE overlay with mean (solid) + median (dotted) lines ---------
    for label, color, arr in rows:
        kde_or_hist(ax_kde, arr, f"{label}  (μ={arr.mean():.0f}, med={np.median(arr):.0f})",
                    color, xrange)
        ax_kde.axvline(float(np.median(arr)), color=color, ls=":", lw=1.2, alpha=0.7)
    ax_kde.set_xlim(*xrange)
    ax_kde.set_xlabel(f"DS-WED (per utterance)  —  x clipped at p98 = {xhi:.0f}")
    ax_kde.set_ylabel("density")
    ax_kde.set_title("DS-WED per-utterance distributions — prosody diversity "
                     "(K=3 seeds, zero_shot/en; higher = more varied)", fontsize=11)
    ax_kde.grid(alpha=0.25)
    ax_kde.legend(fontsize=8, loc="upper right")

    # --- bottom: boxplot (full range incl. outliers) on log x ---------------
    labels = [r[0] for r in rows]
    data = [r[2] for r in rows]
    bp = ax_box.boxplot(data, vert=False, tick_labels=labels, showfliers=True,
                        patch_artist=True, widths=0.6,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, (_, color, _) in zip(bp["boxes"], rows):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for med in bp["medians"]:
        med.set_color("black")
    ax_box.set_xscale("log")
    ax_box.set_xlabel("DS-WED (log scale, full range incl. outliers)")
    ax_box.invert_yaxis()  # most-diverse on top, matching KDE legend order
    ax_box.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    out = REPO / "assets/plots/dswed_distributions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
