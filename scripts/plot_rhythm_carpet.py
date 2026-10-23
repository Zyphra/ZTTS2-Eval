"""Pause carpet: small multiples of per-utterance rhythm texture.

The middle ground between the dwell-time ridgeline (population, no texture)
and the single-utterance barcode (texture, no population). One panel per
system; inside each panel, ~70 utterances (the SAME uids, in the SAME row
order everywhere) drawn as thin strips: pale = speech, mid-tone = a sound
held 100-200 ms, red = a >=200 ms pause/stall. The eye reads each panel as
cloth — kokoro is unmarked fabric (monotonous), real speech is peppered with
red flecks, glm-tts is streaked.

Rows are sorted by the GT recording's duration, so every panel shows the same
duration envelope and the same utterance sits at the same height everywhere.

Tokenizes ~systems x 70 wavs on the fly against the cached k-means anchor:
needs a GPU (a couple of minutes).

Usage:
    python scripts/plot_rhythm_carpet.py
    python scripts/plot_rhythm_carpet.py --n-utts 100
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tufte_common import PAPER, INK, MUT, GREY, MID, BLUE, RED  # noqa: E402
from plot_rhythm_scatter import SHORT  # noqa: E402
from rhythm_dist import (Tokenizer, run_lengths, anchor_key, CACHE_DIR,  # noqa: E402
                         _load_trim, FRAME_S)
from speaking_rate_dist import find_systems  # noqa: E402

MID_FRAMES = 5     # 100 ms: a held sound worth inking
LONG_FRAMES = 10   # 200 ms: pause (clause-level)
STALL_FRAMES = 25  # 500 ms: stall (pathological — different ink, thicker knot)
DARKRED = "#611A0E"
AMBER = "#D08A18"
# pause/stall ink pairs: redmaroon = escalating reds; yellowred = severity ramp
SCHEMES = {"redmaroon": (RED, DARKRED), "yellowred": (AMBER, RED)}

# panels are re-sorted at plot time by the share computed on the sampled
# utterances (so the printed number and the grid position always agree)
PANELS = ["kokoro", "omni_voxcpm2", "omni_qwen3-tts-1.7b", "omni_fish-s2-pro",
          "moss-tts", "gt_zero_shot_voxblink", "zonos2_accurate",
          "omni_voxtral-tts", "vibevoice-7b", "omni_omnivoice",
          "zonos2_expressive", "omni_glm-tts"]


def segments(labels):
    """[(start_s, dur_s, run_frames), ...] from a label sequence."""
    r = run_lengths(labels)
    starts = np.concatenate(([0], np.cumsum(r)[:-1]))
    return [(s * FRAME_S, d * FRAME_S, d) for s, d in zip(starts, r)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", default="zero_shot_voxblink")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n-utts", type=int, default=70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xmax", type=float, default=9.0, help="seconds shown per strip")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="yellowred",
                    help="pause/stall ink: redmaroon (escalating reds) or "
                         "yellowred (amber warning -> red alarm)")
    ap.add_argument("--row-order", choices=["matched", "duration"], default="matched",
                    help="matched: same uid at the same height in every panel "
                         "(GT-duration sorted); duration: each panel sorted by its "
                         "OWN strip durations (clean staircase per panel, uid "
                         "correspondence across panels lost)")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    systems = find_systems(args.test_set, args.lang)
    dirs = {p: systems[p] for p in PANELS if p in systems}

    ref_dir = os.path.join(REPO, "data", args.test_set, args.lang, "waveform")
    key = anchor_key(ref_dir, 9, 100, 0, 35.0, LONG_FRAMES)
    tok = Tokenizer(9)
    tok.set_centroids(np.load(CACHE_DIR / f"{key}.npz")["centroids"])

    common = sorted(set.intersection(*[
        {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".wav")}
        for d in dirs.values()]))
    rng = np.random.default_rng(args.seed)
    uids = list(rng.choice(common, min(args.n_utts, len(common)), replace=False))

    # shared row order: by the real recording's trimmed duration
    gt_dir = dirs["gt_zero_shot_voxblink"]
    gt_dur = {}
    for u in uids:
        _, w = _load_trim((os.path.join(gt_dir, u + ".wav"), 35.0))
        gt_dur[u] = len(w) / 16000 if w is not None else 0.0
    uids.sort(key=lambda u: gt_dur[u])

    # pass 1: tokenize everything, so panels can be ordered by what is shown
    panels = []
    for raw in PANELS:
        rows, pause_t, total_t = [], 0.0, 0.0
        for i, u in enumerate(uids):
            _, w = _load_trim((os.path.join(dirs[raw], u + ".wav"), 35.0))
            if w is None or len(w) < 8000:
                continue
            segs = segments(tok.labels(w))
            total_t += sum(d for _, d, _ in segs)
            pause_t += sum(d for _, d, f in segs if f >= LONG_FRAMES)
            rows.append((i, segs))
        share = 100 * pause_t / total_t if total_t else 0.0
        panels.append((share, raw, rows))
    panels.sort(key=lambda p: p[0])

    pause_c, stall_c = SCHEMES[args.scheme]
    ncol, nrow = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.6, 9.4), dpi=200)
    fig.patch.set_facecolor(PAPER)

    for ax, (share, raw, rows) in zip(axes.flat, panels):
        ax.set_facecolor(PAPER)
        name = SHORT.get(raw, raw)
        is_gt = raw.startswith("gt_")
        if args.row_order == "duration":
            # each panel gets its own staircase: rank by this system's durations
            ranked = sorted(rows, key=lambda r: sum(d for _, d, _ in r[1]))
            rows = [(rank, segs) for rank, (_, segs) in enumerate(ranked)]
        # the barcode's skyline idea at thread scale: speech is a thin thread,
        # holds are thicker, pauses are the thickest knots in the weave
        for i, segs in rows:
            dur = sum(d for _, d, _ in segs)
            y = -i
            ax.broken_barh([(0, min(dur, args.xmax))], (y - 0.16, 0.32),
                           color="#E4DCCB", lw=0)
            mids = [(s, d) for s, d, f in segs
                    if MID_FRAMES <= f < LONG_FRAMES and s < args.xmax]
            pauses = [(s, d) for s, d, f in segs
                      if LONG_FRAMES <= f < STALL_FRAMES and s < args.xmax]
            stalls = [(s, d) for s, d, f in segs
                      if f >= STALL_FRAMES and s < args.xmax]
            if mids:
                ax.broken_barh(mids, (y - 0.30, 0.60), color="#B4A88F", lw=0)
            if pauses:
                ax.broken_barh(pauses, (y - 0.40, 0.80), color=pause_c, lw=0)
            if stalls:
                ax.broken_barh(stalls, (y - 0.48, 0.96), color=stall_c, lw=0)
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(-len(uids) + 0.5, 0.9)
        ax.set_yticks([])
        ax.set_title(f"{name}   {share:.1f}%", loc="left", fontsize=10,
                     color=BLUE if is_gt else INK,
                     fontweight="bold" if is_gt else "normal", pad=4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
        if ax in axes[-1]:
            ax.set_xticks([0, 3, 6, 9])
            ax.set_xticklabels(["0", "3", "6", "9 s"])
            ax.tick_params(colors=MID, labelsize=8, length=3)
        else:
            ax.set_xticks([])

    fig.text(0.065, 0.985, "The pause carpet", color=INK, fontsize=16,
             fontweight="bold", va="top")
    order_note = ("same uids, same row order in every panel"
                  if args.row_order == "matched"
                  else "same uids, each panel sorted by its own strip durations")
    ink_note = ("red = 200–500 ms pause · dark red = ≥500 ms stall"
                if args.scheme == "redmaroon"
                else "amber = 200–500 ms pause · red = ≥500 ms stall")
    fig.text(0.065, 0.955,
             f"{len(uids)} VoxBlink utterances ({order_note}) · one strip per "
             f"utterance · pale = speech · tan = sound held 100–200 ms · "
             f"{ink_note} · number = % of audio time inside ≥200 ms holds",
             color=MUT, fontsize=9, va="top")
    fig.tight_layout(rect=[0.055, 0.01, 0.985, 0.93])

    suffix = "" if args.row_order == "matched" else "_byduration"
    if args.scheme != "yellowred":
        suffix += f"_{args.scheme}"
    out = args.out or os.path.join(
        REPO, "assets", "plots",
        f"rhythm_carpet_{args.test_set}_{args.lang}{suffix}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
