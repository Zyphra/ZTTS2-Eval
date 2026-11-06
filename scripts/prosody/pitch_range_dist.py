"""Pitch-range distribution score: register-normalized melody spread vs a reference.

Per utterance the pitch range is p90 - p10 of the voiced F0 in semitones around the
utterance's own median (utils/pitch_range.py), so it measures how much the melody moves
and ignores the speaker's register. A system is scored by the 2-Wasserstein distance
between its distribution of per-utterance ranges and the reference's, at the same
split-half geometry as the ceiling (scripts/prosody/common.w2_fair vs ceiling), so
`x_ceiling` = 1 means "as close to real speech as two halves of real speech are to
each other".

pYIN is slow, so every system is scored on a FIXED sample of at most --n-utts uids per
(set, lang): utils.benchmark.sample_uids (seeded, shared with scripts/prosody/paired_itw.py).
The reference is the PROMPT recordings of those uids (data/<set>/<lang>/prompt_wav.scp).

    python scripts/prosody/pitch_range_dist.py --test-set in-the-wild --lang en
    python scripts/prosody/pitch_range_dist.py --test-set clean --lang de --models omni_voxcpm2

Caches (uid<TAB>range, empty = unusable wav): results/_pitch_range_ref/<set>_<lang>.tsv
for the reference and <leaf>/pitch_range.tsv per system leaf. Output CSV columns:
rank, system, n, median_st, median_ratio, d_ref, x_ceiling, benchmark_version, with a
`__CEILING_split_half__` row (rank 0, x_ceiling 1.0). Systems that ARE the reference
audio (the Clean GT) are skipped by find_systems, as for char/s.
"""
import argparse
import csv
import os
import sys

import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from scripts.prosody.common import (CEILING_ROW, add_common_prosody_args, ceiling,  # noqa: E402
                                    find_systems, sample_and_need, w2_fair)
from utils.benchmark import BENCHMARK_VERSION  # noqa: E402
from utils.pitch_range import ranges_cached  # noqa: E402
from utils.scp import load_scp_items  # noqa: E402


def prompt_wavs(test_set, lang, uids):
    """{uid: absolute prompt wav path} for `uids` from data/<set>/<lang>/prompt_wav.scp."""
    scp = dict(load_scp_items(os.path.join(REPO, "data", test_set, lang, "prompt_wav.scp")))
    return {u: (scp[u] if os.path.isabs(scp[u]) else os.path.join(REPO, scp[u]))
            for u in uids if u in scp}


def values(rng_map):
    return np.array([v for v in rng_map.values() if v is not None], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default="clean")
    ap.add_argument("--lang", default="en")
    add_common_prosody_args(ap)
    ap.add_argument("--n-utts", type=int, default=500, help="uid sample size (default 500)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-utts", type=int, default=100,
                    help="minimum usable utterances per system (capped at 80%% of the "
                         "sample so small languages are not skipped)")
    args = ap.parse_args()

    uids, need = sample_and_need(args.test_set, args.lang, args.n_utts, args.seed, args.min_utts)

    ref_wavs = prompt_wavs(args.test_set, args.lang, uids)
    ref_cache = os.path.join(REPO, "results", "_pitch_range_ref",
                             f"{args.test_set}_{args.lang}.tsv")
    ref = values(ranges_cached(ref_wavs, ref_cache, args.workers))
    if ref.size < max(need, 4):
        raise SystemExit(f"only {ref.size} usable reference utterances "
                         f"({args.test_set}/{args.lang})")
    ceil = ceiling(ref)
    print(f"reference: n={ref.size}  median {np.median(ref):.2f} st  "
          f"CEILING (split-half) d_ref = {ceil:.3f}")

    rows = []
    for name, wav_dir in sorted(find_systems(args.test_set, args.lang, args.models).items()):
        wavs = {u: os.path.join(wav_dir, u + ".wav") for u in uids
                if os.path.isfile(os.path.join(wav_dir, u + ".wav"))}
        if len(wavs) < need:
            continue
        leaf = os.path.dirname(wav_dir)
        sysv = values(ranges_cached(wavs, os.path.join(leaf, "pitch_range.tsv"), args.workers))
        if sysv.size < need:
            continue
        d_ref = w2_fair(sysv, ref)
        rows.append(dict(system=name, n=int(sysv.size), median_st=float(np.median(sysv)),
                         median_ratio=float(np.median(sysv) / np.median(ref)),
                         d_ref=d_ref, x_ceiling=d_ref / ceil if ceil > 0 else float("nan"),
                         benchmark_version=BENCHMARK_VERSION))
        print(f"  {name:34s} n={sysv.size:4d} median {np.median(sysv):5.2f} st  "
              f"d_ref {d_ref:6.3f} ({rows[-1]['x_ceiling']:.2f}x)", flush=True)

    rows.sort(key=lambda r: r["d_ref"])
    out = args.out or os.path.join(REPO, "results",
                                   f"pitch_range_{args.test_set}_{args.lang}.csv")
    fields = ["rank", "system", "n", "median_st", "median_ratio", "d_ref", "x_ceiling",
              "benchmark_version"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(rank=0, system=CEILING_ROW, n=int(ref.size),
                        median_st=float(np.median(ref)), median_ratio=1.0, d_ref=ceil,
                        x_ceiling=1.0, benchmark_version=BENCHMARK_VERSION))
        for i, r in enumerate(rows):
            w.writerow(dict(rank=i + 1, **r))
    print(f"\n{len(rows)} systems -> {out}")


if __name__ == "__main__":
    main()
