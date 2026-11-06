"""Level-sensitivity probe for MSR-UTMOS (gain sweep).

Scores a sample of human reference clips raw and under pure gain changes
(no other processing) with the same scorer the eval uses
(utils/score_msr_utmos.py) and reports the per-clip score delta per
condition. A level-invariant scorer returns deltas at zero; MSR-UTMOS does
not (2026-09-08, 80 clips: -6 dB -> +0.042 mean, -12 dB -> +0.095).

Conditions: raw; -6 dB; -12 dB; +3 dB (only where it does not clip);
peak-normalized to -1 dBFS (to show that peak normalization is itself a
per-clip level change, not a fix).

Usage
-----
    python scripts/probe_utmos_level.py                       # 40 Clean + 40 ITW en clips
    python scripts/probe_utmos_level.py --n 100 --langs en de
    python scripts/probe_utmos_level.py --out assets/probes/utmos_level_<date>.csv

Output CSV: one row per clip and condition (set, lang, uttid, condition,
gain_db, score, delta_vs_raw); a summary per condition is printed.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torchaudio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.score_msr_utmos import load_model, score_wave  # noqa: E402

GAINS_DB = (-6.0, -12.0, 3.0)
PEAK_DBFS = -1.0


def conditions(wav):
    """-> {name: (gain_db or None, waveform)}; +gain skipped where it would clip."""
    peak = wav.abs().max().item()
    out = {"raw": (0.0, wav)}
    for g in GAINS_DB:
        if peak * 10 ** (g / 20) <= 1.0:
            out[f"{g:+.0f}dB"] = (g, wav * 10 ** (g / 20))
    if peak > 0:
        g = PEAK_DBFS - 20 * np.log10(peak)
        out[f"peak{PEAK_DBFS:+.0f}dBFS"] = (float(g), wav * 10 ** (g / 20))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-sets", nargs="*", default=["clean", "in-the-wild"])
    ap.add_argument("--langs", nargs="*", default=["en"])
    ap.add_argument("--n", type=int, default=40, help="clips per (set, lang), taken in sorted order")
    ap.add_argument("--out", default=os.path.join(REPO, "assets", "probes", "utmos_level.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(device=args.device).to(args.device)
    rows = []
    for test_set in args.test_sets:
        for lang in args.langs:
            files = sorted(glob.glob(os.path.join(REPO, "results", f"gt_{test_set}", lang, "wavs", "*.wav")))[: args.n]
            if not files:
                print(f"  no reference wavs for {test_set}/{lang}", file=sys.stderr)
                continue
            for f in files:
                wav, sr = torchaudio.load(f)
                scores = {k: score_wave(model, v, sr, args.device) for k, (_, v) in conditions(wav).items()}
                gains = {k: g for k, (g, _) in conditions(wav).items()}
                for k, s in scores.items():
                    rows.append(dict(test_set=test_set, lang=lang, uttid=os.path.basename(f)[:-4], condition=k,
                                     gain_db=round(gains[k], 3), score=round(s, 4), delta_vs_raw=round(s - scores["raw"], 4)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test_set", "lang", "uttid", "condition", "gain_db", "score", "delta_vs_raw"])
        w.writeheader()
        w.writerows(rows)

    n_clips = sum(1 for r in rows if r["condition"] == "raw")
    print(f"{n_clips} clips -> {args.out}")
    print(f"{'condition':>14s} {'n':>4s} {'mean d':>8s} {'sd':>6s} {'|d|max':>7s}")
    for cond in sorted({r["condition"] for r in rows} - {"raw"}, key=lambda c: (c.startswith("peak"), c)):
        d = np.array([r["delta_vs_raw"] for r in rows if r["condition"] == cond])
        print(f"{cond:>14s} {d.size:4d} {d.mean():+8.3f} {d.std(ddof=1):6.3f} {np.abs(d).max():7.3f}")


if __name__ == "__main__":
    main()
