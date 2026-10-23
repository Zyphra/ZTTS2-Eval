"""Rhythm barcodes: the SAME utterance, one row per system, texture visible.

Each row renders one system's generation of the same VoxBlink utterance (same
target text, same speaker prompt; the GT row is the real recording) as its
mHuBERT token segments: alternating shades per segment, ≥200 ms segments in
red. Rhythm becomes texture the eye reads directly — real speech is clustered
ticks with gaps, a metronomic system is a picket fence, a stall-prone one has
long red voids.

This is the anecdote companion to the distributional views (the dwell-time
ridgeline / the flatten-vs-stall scatter): one utterance, honestly labeled as
such. The uid is chosen deterministically: among utterances every plotted
system generated, the one where the GT recording shows the most >=200 ms
segments within a 5-9 s duration (i.e. a sentence where real speech actually
pauses).

Needs a GPU (tokenizes ~8 wavs on the fly with the cached k-means anchor).

Usage:
    python scripts/plot_rhythm_barcode.py
    python scripts/plot_rhythm_barcode.py --uid uttid_123
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

LONG_FRAMES = 10   # 200 ms: pause (clause-level, real speech does this)
STALL_FRAMES = 25  # 500 ms: stall (pathological; different ink on purpose)
DARKRED = "#611A0E"
AMBER = "#D08A18"
SCHEMES = {"redmaroon": (RED, DARKRED), "yellowred": (AMBER, RED)}
ROWS = ["gt_zero_shot_voxblink", "kokoro", "omni_qwen3-tts-1.7b",
        "omni_fish-s2-pro", "moss-tts", "omni_voxtral-tts",
        "zonos2_expressive", "omni_glm-tts"]


def pick_uid(dirs, tok, scan):
    """Deterministic exemplar: GT 5-9 s with the most >=200 ms segments,
    scanning at most the first `scan` shared uids (keeps GPU time short)."""
    common = sorted(set.intersection(*[
        {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".wav")}
        for d in dirs.values()]))[:scan]
    gt_dir = dirs["gt_zero_shot_voxblink"]
    best, best_n = None, -1
    for uid in common:
        _, wav = _load_trim((os.path.join(gt_dir, uid + ".wav"), 35.0))
        if wav is None or not (5.0 <= len(wav) / 16000 <= 9.0):
            continue
        n_long = int((run_lengths(tok.labels(wav)) >= LONG_FRAMES).sum())
        if n_long > best_n:
            best, best_n = uid, n_long
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", default="zero_shot_voxblink")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--uid", default=None, help="skip selection, plot this uid")
    ap.add_argument("--scan", type=int, default=120,
                    help="max candidate uids to scan when picking (keeps GPU time short)")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="yellowred",
                    help="pause/stall ink: redmaroon (escalating reds) or "
                         "yellowred (amber warning -> red alarm)")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    systems = find_systems(args.test_set, args.lang)
    dirs = {r: systems[r] for r in ROWS if r in systems}
    missing = [r for r in ROWS if r not in dirs]
    if missing:
        print(f"  missing systems skipped: {missing}")

    ref_dir = os.path.join(REPO, "data", args.test_set, args.lang, "waveform")
    key = anchor_key(ref_dir, 9, 100, 0, 35.0, LONG_FRAMES)
    anchor = np.load(CACHE_DIR / f"{key}.npz")
    tok = Tokenizer(9)
    tok.set_centroids(anchor["centroids"])

    uid = args.uid or pick_uid(dirs, tok, args.scan)
    print(f"uid = {uid}")

    rows = []
    for raw in ROWS:
        if raw not in dirs:
            continue
        _, wav = _load_trim((os.path.join(dirs[raw], uid + ".wav"), 35.0))
        if wav is None:
            continue
        rows.append((SHORT.get(raw, raw), run_lengths(tok.labels(wav))))

    fig, ax = plt.subplots(figsize=(10.4, 0.74 * len(rows) + 1.4), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    # Rhythm skyline: each segment is a block rising from the row's baseline,
    # HEIGHT = how long the sound holds (log-scaled 20 ms -> 2 s), colors =
    # the carpet's 3-class language. Monotony reads as uniform low-rise
    # rubble; pauses as red towers.
    H = 0.92                                       # max tower height
    MID_FRAMES = 5                                 # 100 ms, same as the carpet

    def seg_h(r):
        dur_ms = r * FRAME_S * 1000
        return H * (0.12 + 0.88 * min(1.0, np.log(dur_ms / 20) / np.log(2000 / 20)))

    pause_c, stall_c = SCHEMES[args.scheme]

    def seg_color(r):
        if r >= STALL_FRAMES:
            return stall_c
        if r >= LONG_FRAMES:
            return pause_c
        return "#A89A7E" if r >= MID_FRAMES else "#D2C9B7"

    PITCH = 1.18
    for i, (name, runs) in enumerate(rows):
        y = -i * PITCH                             # row pitch > H: no collisions
        ax.axhline(y, xmin=0, xmax=1, color=GREY, lw=0.6, zorder=1)
        t = 0.0
        for r in runs:
            dur = r * FRAME_S
            ax.add_patch(plt.Rectangle(
                (t, y), dur, seg_h(r), lw=0.15, edgecolor=PAPER,
                facecolor=seg_color(r), zorder=2))
            t += dur
        is_gt = name.startswith("GT")
        ax.text(-0.12, y + 0.06, name, ha="right", va="bottom", fontsize=9,
                color=BLUE if is_gt else INK,
                fontweight="bold" if is_gt else "normal")
        n_pause = int(((runs >= LONG_FRAMES) & (runs < STALL_FRAMES)).sum())
        n_stall = int((runs >= STALL_FRAMES).sum())
        note = f"{t:.1f} s · {n_pause} pause{'s' if n_pause != 1 else ''}"
        if n_stall:
            note += f" · {n_stall} stall{'s' if n_stall != 1 else ''}"
        ax.text(t + 0.12, y + 0.06, note,
                ha="left", va="bottom", fontsize=7.5, color=MUT)

    ax.set_xlim(-0.05, max(sum(r) * FRAME_S for _, r in rows) * 1.14)
    ax.set_ylim(-(len(rows) - 1) * PITCH - 0.25, H + 0.30)
    ax.set_yticks([])
    ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 2))
    ax.set_xticklabels([f"{s} s" if s else "0" for s in ax.get_xticks()])
    ax.tick_params(colors=MID, labelsize=8.5, length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)

    ax.text(0, 1.115, "The same sentence, eight rhythms",
            transform=ax.transAxes, color=INK, fontsize=14, fontweight="bold")
    ax.text(0, 1.03,
            f"one VoxBlink utterance ({uid}) · same text & speaker prompt\n"
            "block = one mHuBERT token segment · HEIGHT = how long the sound "
            "holds (log, 20 ms–2 s) · tan ≥100 ms · "
            + ("red = 200–500 ms pause · dark red = ≥500 ms stall"
               if args.scheme == "redmaroon"
               else "amber = 200–500 ms pause · red = ≥500 ms stall"),
            transform=ax.transAxes, color=MUT, fontsize=8.5, va="top")

    fig.tight_layout()
    suffix = "" if args.scheme == "yellowred" else f"_{args.scheme}"
    out = args.out or os.path.join(
        REPO, "assets", "plots",
        f"rhythm_barcode_{args.test_set}_{args.lang}{suffix}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAPER, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
