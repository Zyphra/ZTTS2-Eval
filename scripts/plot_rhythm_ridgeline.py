"""Dwell-time ridgeline: where does each system's LISTENING TIME live?

The rhythm analogue of the pitch ridgeline. For each system, the distribution
of token-segment duration (mHuBERT-147 k-means runs, the rhythm metric's
substrate), weighted by duration — i.e. the share of total listening time
spent inside segments that stay the same for X ms. Time-weighting is the key
move: raw run counts are dominated by 1-2-frame segments, but a 1 s stall
occupies 1 s of the listener's time, so weighting shows pause mass at its
perceptual size.

The shapes read as the adjectives directly: a monotonous system (kokoro) is a
needle with an empty right side — it never dwells; real speech has a broad
body plus a fat right shoulder past 200 ms (pauses, stalls, held vowels);
erratic systems (zonos2-expressive, glm-tts) have lobes reaching seconds.

Reads runlen_hist from the --dump-dir npz files written by rhythm_dist.py.

Usage:
    python scripts/plot_rhythm_ridgeline.py
    python scripts/plot_rhythm_ridgeline.py --test-set zero_shot --lang en
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tufte_common import PAPER, INK, MUT, GREY, MID, BLUE, RED
from plot_rhythm_scatter import SHORT

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_MS = 20.0
LONG_MS = 200.0

# display order is computed (sorted by pause-time share); this is just the set
WANT = ["gt_zero_shot_voxblink", "kokoro", "omni_voxcpm2", "omni_qwen3-tts-1.7b",
        "omni_fish-s2-pro", "moss-tts", "vibevoice-7b", "zonos2_accurate",
        "omni_voxtral-tts", "omni_omnivoice", "zonos2_expressive", "omni_glm-tts"]


MIN_MS = 50.0   # the ridge starts where rhythm lives; below this every system
                # shows the same universal 20-40 ms phone-body and the rows
                # become indistinguishable (first-render lesson)


def dwell_curve(hist, xs_log, bw=0.09):
    """Time-weighted run-length density on a log10(ms) grid, ≥ MIN_MS only,
    but normalized by TOTAL listening time — so the AREA of a ridge is the
    share of that system's audio spent holding a sound ≥ 50 ms. A monotonous
    system is a near-empty ridge, a stall-heavy one a mountain.

    hist[f] = count of runs of length f frames; weight each by its duration so
    the curve integrates listening time, then smooth in log-duration space.
    """
    f = np.arange(1, len(hist))
    w = hist[1:] * f                      # frames of audio spent at each length
    w = w / w.sum()                       # normalize by ALL time (incl. <50 ms)
    centers = np.log10(f * FRAME_MS)
    keep = f * FRAME_MS >= MIN_MS
    dens = np.zeros_like(xs_log)
    for c, wi in zip(centers[keep], w[keep]):
        if wi > 0:
            dens += wi * np.exp(-0.5 * ((xs_log - c) / bw) ** 2)
    return dens


def pause_share(hist):
    f = np.arange(1, len(hist))
    w = hist[1:] * f
    return 100.0 * w[f >= LONG_MS / FRAME_MS].sum() / w.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", default="zero_shot_voxblink")
    ap.add_argument("--lang", default="en")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    dump_dir = os.path.join(REPO, "results",
                            f"rhythm_perutt_{args.test_set}_{args.lang}")
    rows = []
    for raw in WANT:
        p = os.path.join(dump_dir, raw.replace("/", "__") + ".npz")
        if not os.path.isfile(p):
            print(f"  skip {raw}: no dump at {p}")
            continue
        z = np.load(p, allow_pickle=True)
        if "runlen_hist" not in z:
            print(f"  skip {raw}: dump has no runlen_hist (re-run rhythm_dist "
                  f"--dump-dir)")
            continue
        rows.append((SHORT.get(raw, raw), z["runlen_hist"]))
    if not rows:
        raise SystemExit("no dumps with runlen_hist found")

    xs_log = np.linspace(np.log10(MIN_MS * 0.82), np.log10(5200), 700)
    curves = {name: dwell_curve(h, xs_log) for name, h in rows}
    shares = {name: pause_share(h) for name, h in rows}
    # driest rhythm on top, sorted by the same number printed at the right
    # margin (≥200 ms time share) so the column reads monotonically
    order = sorted(curves, key=lambda n: shares[n])
    norm = max(c.max() for c in curves.values())

    STEP, AMP = 0.52, 1.10
    fig, ax = plt.subplots(figsize=(9.2, 9.6), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    gt_name = next(n for n in curves if n.startswith("GT"))
    gt_y = curves[gt_name] / norm * AMP

    # the pause zone: everything right of 200 ms
    ax.axvspan(np.log10(LONG_MS), xs_log[-1], color=GREY, alpha=0.22, zorder=0)

    ytrans = ax.get_yaxis_transform()
    for i, name in enumerate(order):
        base = -i * STEP
        y = curves[name] / norm * AMP
        z = (i + 1) * 10
        is_gt = name == gt_name
        c = BLUE if is_gt else INK
        ax.fill_between(xs_log, base, base + y,
                        color=c, alpha=0.30 if is_gt else 0.13, lw=0, zorder=z)
        ax.plot(xs_log, base + y, color=PAPER, lw=3.0, zorder=z + 1)   # halo
        ax.plot(xs_log, base + y, color=c, lw=1.8 if is_gt else 1.1, zorder=z + 2)
        if not is_gt:                                                  # GT ghost
            ax.plot(xs_log, base + gt_y, color=BLUE, lw=0.9,
                    ls=(0, (4, 2.5)), alpha=0.55, zorder=z + 3)
        # lifted off the baseline so each label nests inside its own curve's
        # left flank instead of floating between rows
        ax.text(-0.015, base + 0.13, name, transform=ytrans, ha="right",
                va="bottom", fontsize=9, clip_on=False, zorder=99,
                color=c, fontweight="bold" if is_gt else "normal")
        ax.text(1.015, base + 0.13, f"{shares[name]:.1f}%", transform=ytrans,
                ha="left", va="bottom", fontsize=9, clip_on=False, zorder=99,
                color=c, fontweight="bold" if is_gt else "normal")

    n = len(order)
    ax.text(1.015, AMP * 0.72, "time in\n≥200 ms\nsegments", transform=ytrans,
            ha="left", va="bottom", fontsize=8, color=MUT, clip_on=False)
    ax.text(np.log10(LONG_MS * 1.06), AMP * 1.18, "pauses, stalls, held sounds →",
            fontsize=8.5, color=MID, ha="left", va="bottom", style="italic")

    ticks_ms = [50, 100, 200, 500, 1000, 2000, 5000]
    ax.set_xticks([np.log10(t) for t in ticks_ms])
    ax.set_xticklabels(["50 ms", "100", "200", "500", "1 s", "2 s", "5 s"])
    ax.set_xlim(xs_log[0], xs_log[-1])
    ax.set_ylim(-(n - 1) * STEP - 0.10, AMP * 1.30)
    ax.set_yticks([])
    ax.tick_params(colors=MID, labelsize=9, length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.set_xlabel("how long the sound holds (segment duration, log scale)",
                  color=INK, fontsize=10)

    ax.text(0, 1.045, "Who can hold a sound? The held-time spectrum",
            transform=ax.transAxes, color=INK, fontsize=15, fontweight="bold")
    ax.text(0, 1.015,
            "listening time spent holding a sound ≥50 ms (mHuBERT segments) · "
            "ridge AREA = share of total audio time · VoxBlink2 in-the-wild · en · "
            "dashed = real speech",
            transform=ax.transAxes, color=MUT, fontsize=9)

    fig.tight_layout()
    out = args.out or os.path.join(
        REPO, "assets", "plots", f"rhythm_dwelltime_ridgeline_{args.test_set}_{args.lang}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
