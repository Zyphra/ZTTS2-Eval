"""Pitch carpet: small multiples of per-utterance MELODY texture.

The pitch sibling of the rhythm carpet (plot_rhythm_carpet.py), and the picture behind
the pitch-range metric (utils/pitch_range.py, scripts/prosody/pitch_range_dist.py; see
docs/pitch_range.md). A per-file MEAN-F0 comparison cannot see melody at all, so this
view shows the contour. One panel per system, the
SAME ~70 VoxBlink utterances as thin strips; each strip is the utterance's F0
track drawn in semitones around its own median (register removed, melody
kept). Monotone systems are flat threads, real speech is waves; the per-panel
number is the median per-utterance pitch range (p10..p90, semitones), and
panels are sorted by it, flattest first.

F0 = librosa.pyin (60-400 Hz, 20 ms hop), median-filtered (width 3) to kill
octave blips; unvoiced frames break the line and show the pale thread
underneath. Tracks are cached per system under
results/pitch_perutt_<set>_<lang>/, so only the first run is slow (CPU-only,
~50 min at 48 workers; no GPU needed).

Usage:
    python scripts/plots/plot_pitch_carpet.py
    python scripts/plots/plot_pitch_carpet.py --row-order duration
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

from scripts.plots.tufte_common import PAPER, INK, MUT, GREY, MID, BLUE  # noqa: E402
from scripts.plots.plot_rhythm_scatter import SHORT  # noqa: E402
from scripts.plots.plot_rhythm_carpet import PANELS, AMBER  # noqa: E402
from scripts.prosody.speaking_rate_dist import find_systems  # noqa: E402
from utils.pitch_range import (SR, HOP, METHOD, f0_track, semitone_dev,  # noqa: E402
                               utt_range_st)

ST_CLIP = 6.0      # semitone display band: ±6 st around the utterance median
ST_BIG = 4.0       # |dev| beyond this is a big melodic move -> amber ink



def load_tracks(cache_dir, raw, wav_dir, uids, workers):
    """{uid: f0 array} for one system, npz-cached; a cache from another METHOD is redone."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, raw.replace("/", "__") + "_f0.npz")
    if os.path.isfile(p):
        z = np.load(p, allow_pickle=False)
        if "__method__" in z and str(z["__method__"]) == METHOD and all(u in z for u in uids):
            return {u: z[u] for u in uids if z[u].size}
    paths = [os.path.join(wav_dir, u + ".wav") for u in uids]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(f0_track, paths, chunksize=1))
    tracks = {u: (f0 if f0 is not None else np.zeros(0, dtype=np.float32))
              for u, f0 in zip(uids, res)}
    np.savez_compressed(p, __method__=np.array(METHOD), **tracks)
    return {u: f for u, f in tracks.items() if f.size}



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", default="in-the-wild")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n-utts", type=int, default=70,
                    help="uids to sample & F0-cache (keep at 70: cache key)")
    ap.add_argument("--n-show", type=int, default=28,
                    help="strips actually drawn per panel — fewer, taller "
                         "strips; 70 squeezed contours into ~7 px dashes and "
                         "every panel looked identical (first-render lesson)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xmax", type=float, default=9.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--row-order", choices=["matched", "duration"], default="matched",
                    help="matched: same uid at the same height in every panel; "
                         "duration: each panel sorted by its own strip durations")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    systems = find_systems(args.test_set, args.lang)
    dirs = {p: systems[p] for p in PANELS if p in systems}
    cache_dir = os.path.join(REPO, "results",
                             f"pitch_perutt_{args.test_set}_{args.lang}")

    # identical uid sampling to the rhythm carpet (same seed, same pool)
    common = sorted(set.intersection(*[
        {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".wav")}
        for d in dirs.values()]))
    rng = np.random.default_rng(args.seed)
    uids = list(rng.choice(common, min(args.n_utts, len(common)), replace=False))

    gt_key = next(p for p in dirs if p.startswith("gt_"))
    gt_tracks = load_tracks(cache_dir, gt_key, dirs[gt_key], uids, args.workers)
    print(f"[f0] {gt_key}: {len(gt_tracks)} tracks")
    uids.sort(key=lambda u: gt_tracks.get(u, np.zeros(0)).size)
    if args.n_show < len(uids):
        # evenly spaced through the duration-sorted list: keeps the staircase
        # envelope while the full 70-uid F0 cache stays valid
        pick = np.linspace(0, len(uids) - 1, args.n_show).round().astype(int)
        uids = [uids[i] for i in pick]

    panels = []
    for raw in PANELS:
        if raw not in dirs:
            continue
        tracks = (gt_tracks if raw == gt_key else
                  load_tracks(cache_dir, raw, dirs[raw], uids, args.workers))
        print(f"[f0] {raw}: {len(tracks)} tracks")
        rows, ranges = [], []
        for i, u in enumerate(uids):
            f0 = tracks.get(u)
            if f0 is None or not f0.size:
                continue
            st = semitone_dev(f0)
            if st is None:
                continue
            rows.append((i, st))
            ranges.append(utt_range_st(st))
        med_range = float(np.median(ranges)) if ranges else 0.0
        panels.append((med_range, raw, rows))
    panels.sort(key=lambda p: p[0])

    ncol, nrow = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.6, 9.4), dpi=200)
    fig.patch.set_facecolor(PAPER)

    for ax, (med_range, raw, rows) in zip(axes.flat, panels):
        ax.set_facecolor(PAPER)
        name = SHORT.get(raw, raw)
        is_gt = raw == gt_key
        if args.row_order == "duration":
            ranked = sorted(rows, key=lambda r: r[1].size)
            rows = [(rank, st) for rank, (_, st) in enumerate(ranked)]
        line_c = BLUE if is_gt else INK
        for i, st in rows:
            y = -i
            t = np.arange(st.size) * (HOP / SR)
            dur = min(t[-1] if t.size else 0.0, args.xmax)
            ax.broken_barh([(0, dur)], (y - 0.06, 0.12), color="#E4DCCB", lw=0)
            keep = t < args.xmax
            sk = st[keep]
            yy = y + np.clip(sk, -ST_CLIP, ST_CLIP) / ST_CLIP * 0.46
            ax.plot(t[keep], yy, color=line_c, lw=0.55, alpha=0.9,
                    solid_capstyle="butt", zorder=3)
            # big melodic moves get amber ink (rhythm-carpet semantics:
            # accent color = the thing the flatteners can't do)
            big = np.where(np.abs(sk) >= ST_BIG, yy, np.nan)
            if np.any(np.isfinite(big)):
                ax.plot(t[keep], big, color=AMBER, lw=1.0, alpha=0.95,
                        solid_capstyle="butt", zorder=4)
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(-len(uids) + 0.5, 0.9)
        ax.set_yticks([])
        ax.set_title(f"{name}   {med_range:.1f} st", loc="left", fontsize=10,
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

    fig.text(0.065, 0.985, "The melody carpet", color=INK, fontsize=16,
             fontweight="bold", va="top")
    order_note = ("same uids, same row order in every panel"
                  if args.row_order == "matched"
                  else "same uids, each panel sorted by its own strip durations")
    fig.text(0.065, 0.955,
             f"{len(uids)} VoxBlink utterances ({order_note}) · one strip per "
             "utterance · line = F0 contour in semitones around the utterance's "
             f"own median (±{ST_CLIP:.0f} st band)\namber = beyond ±{ST_BIG:.0f} st "
             "(big melodic moves) · gaps = unvoiced · number = median "
             "per-utterance pitch range (p10–p90, st) · flattest panel first",
             color=MUT, fontsize=9, va="top")
    fig.tight_layout(rect=[0.055, 0.01, 0.985, 0.93])

    suffix = "" if args.row_order == "matched" else "_byduration"
    out = args.out or os.path.join(
        REPO, "assets", "plots",
        f"pitch_carpet_{args.test_set}_{args.lang}{suffix}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
