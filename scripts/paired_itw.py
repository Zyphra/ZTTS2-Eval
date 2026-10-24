"""Per-utterance PAIRED prosody for the ITW set: each generation vs the real recording.

On the in-the-wild set every item has a held-out real recording of the same speaker
saying the target sentence (data/zero_shot_voxblink/<lang>/target_wav.scp), so prosody
can be compared utterance by utterance instead of only as distributions:

  dur_log2    log2(trimmed duration of the generation / of the real recording)
  pause_diff  #pauses(generation) - #pauses(real); a pause is an internal gap >= 0.2 s
              between librosa.effects.split(top_db=35) intervals
  range_log2  log2(pitch range of the generation / of the real recording), utils/pitch_range

Reported per system as medians (means for pause counts) of the signed and the absolute
values: signed = bias (too slow / too few pauses / too flat), absolute = how far off a
typical utterance is. Uses the same fixed uid sample as scripts/pitch_range_dist.py
(utils.benchmark.sample_uids) and shares its pitch caches, so run them in one job.

    python scripts/paired_itw.py --lang en

Output: results/paired_itw_<lang>.csv with columns system, n, dur_log2_med, dur_abs_med,
pause_diff_mean, pause_abs_mean, range_log2_med, range_abs_med, benchmark_version.
The real-recording GT system (gt_zero_shot_voxblink) is not a row: it IS the reference.
"""
import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from speaking_rate_dist import find_systems  # noqa: E402
from utils.benchmark import BENCHMARK_VERSION, sample_uids  # noqa: E402
from utils.pitch_range import ranges_cached  # noqa: E402
from utils.scp import load_scp_items  # noqa: E402

TASK = "zero_shot_voxblink"
GT_NAME = f"gt_{TASK}"
SR = 16000
TOP_DB = 35.0
PAUSE_S = 0.2


def audio_stats(path):
    """(trimmed duration s, #internal pauses >= PAUSE_S), or None for an unusable wav."""
    try:
        import librosa
        wav, _ = librosa.load(path, sr=SR)
        if wav.size == 0:
            return None
        trimmed, _ = librosa.effects.trim(wav, top_db=TOP_DB)
        if trimmed.size == 0:
            return None
        iv = librosa.effects.split(trimmed, top_db=TOP_DB)
        gaps = (iv[1:, 0] - iv[:-1, 1]) / SR if len(iv) > 1 else np.zeros(0)
        return trimmed.size / SR, int(np.sum(gaps >= PAUSE_S))
    except Exception:
        return None


def stats_for(paths, workers):
    """{uid: (dur, n_pauses) | None} for {uid: path}."""
    uids = list(paths)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(audio_stats, [paths[u] for u in uids], chunksize=4))
    return dict(zip(uids, res))


def log2_ratio(a, b):
    return math.log2(a / b) if a and b and a > 0 and b > 0 else None


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="default results/paired_itw_<lang>.csv")
    ap.add_argument("--n-utts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--min-utts", type=int, default=100,
                    help="minimum paired utterances per system (capped at 80%% of the sample)")
    args = ap.parse_args()

    uids = sample_uids(TASK, args.lang, args.n_utts, args.seed)
    need = min(args.min_utts, int(0.8 * len(uids)))
    scp = os.path.join(REPO, "data", TASK, args.lang, "target_wav.scp")
    if not os.path.isfile(scp):
        raise SystemExit(f"missing {scp} (run scripts/build_voxblink_targets.py)")
    wanted = set(uids)
    tgt = {u: (p if os.path.isabs(p) else os.path.join(REPO, p))
           for u, p in load_scp_items(scp) if u in wanted}
    tgt_range = ranges_cached(tgt, os.path.join(
        REPO, "results", "_pitch_range_ref", f"{TASK}_{args.lang}_target.tsv"), args.workers)
    tgt_stats = stats_for(tgt, args.workers)
    print(f"{args.lang}: {len(tgt)} real target recordings in the sample")

    rows = []
    systems = find_systems(TASK, args.lang, args.models)
    systems.pop(GT_NAME, None)
    for name, wav_dir in sorted(systems.items()):
        wavs = {u: os.path.join(wav_dir, u + ".wav") for u in tgt
                if os.path.isfile(os.path.join(wav_dir, u + ".wav"))}
        if len(wavs) < need:
            continue
        st = stats_for(wavs, args.workers)
        rng = ranges_cached(wavs, os.path.join(os.path.dirname(wav_dir), "pitch_range.tsv"),
                            args.workers)
        dur, pause, prange = [], [], []
        for u in wavs:
            s, t = st[u], tgt_stats.get(u)
            if s is None or t is None:
                continue
            dur.append(math.log2(s[0] / t[0]))
            pause.append(s[1] - t[1])
            r = log2_ratio(rng[u], tgt_range.get(u))
            if r is not None:
                prange.append(r)
        if len(dur) < need:
            continue
        rows.append(dict(
            system=name, n=len(dur),
            dur_log2_med=med(dur), dur_abs_med=med(np.abs(dur)),
            pause_diff_mean=float(np.mean(pause)), pause_abs_mean=float(np.mean(np.abs(pause))),
            range_log2_med=med(prange), range_abs_med=med(np.abs(prange)),
            benchmark_version=BENCHMARK_VERSION))
        r = rows[-1]
        print(f"  {name:34s} n={r['n']:4d} dur {r['dur_log2_med']:+.3f} |{r['dur_abs_med']:.3f}|  "
              f"pauses {r['pause_diff_mean']:+.2f} |{r['pause_abs_mean']:.2f}|  "
              f"range {r['range_log2_med']:+.3f} |{r['range_abs_med']:.3f}|", flush=True)

    out = args.out or os.path.join(REPO, "results", f"paired_itw_{args.lang}.csv")
    fields = ["system", "n", "dur_log2_med", "dur_abs_med", "pause_diff_mean", "pause_abs_mean",
              "range_log2_med", "range_abs_med", "benchmark_version"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["system"]):
            w.writerow(r)
    print(f"\n{len(rows)} systems -> {out}")


if __name__ == "__main__":
    main()
