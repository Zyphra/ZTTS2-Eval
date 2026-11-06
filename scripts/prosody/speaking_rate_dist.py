"""Speaking-rate distribution score: trimmed, truncation-filtered char/s vs a reference.

WHAT THIS IS
------------
A distributional distance between a system's per-utterance speaking rate and a
reference set's, in characters per second of ACTUAL SPEECH.

    char/s(utt) = len(target text) / trimmed audio duration
    d_ref       = W2( {char/s of system}, {char/s of reference} )     LOWER = better

There is no 0-100 normalisation on purpose. The scale is runs of char/s, and the
only calibration that matters is the CEILING: split the reference in half and
score half against half. That is what a perfect system scores at finite n.
Everything is reported as d_ref alongside that ceiling.

WHY THIS EXISTS
----------------------------------------------------------------
TTSDS2's two "speaking rate" benchmarks are not usable as-is on this tree:
  - Allosaurus SR = phones / TOTAL duration, silence included. The reference
    (cloning prompts) carries 1.92 s of lead+trail silence = ~21 % of each file,
    so padding more moves you toward the reference and SCORES BETTER
    (rho +0.618 with measured padding). It ranks silence, not rate.
  - Hubert Token SR does track real rate, but ~40 % of its response to a 25 %
    tempo change is reproduced by adding white noise at constant rate.
This metric needs no model, no GPU and no checkpoint, and beats both on
discrimination (49 % of 1081 system pairs separable vs HubSR's 40 %).

TWO THINGS IT DELIBERATELY DOES NOT DO
--------------------------------------
1. It is PACE ONLY. W2 over char/s knows nothing about intonation or rhythm.
   Reversing the audio does not change duration, so this score cannot see it.
2. It matches the DISTRIBUTION of pace, not per-utterance pace. W2 sorts both
   sides, so it is permutation-invariant: a system that reads fast sentences
   slowly and slow sentences fast, with the right overall spread, scores the
   same as one that gets every sentence right. A matched variant is possible
   against results/gt_clean/<lang>/wavs (GT renditions of the SAME uttids,
   unlike the prompts used as reference here) with a derangement floor. Not
   implemented; untested whether it changes the ranking.

TRUNCATION FILTER (--cap)
-------------------------
Trimming turns a stalled generation into a ~0.01 s file, so char/s explodes
(observed max: 9216). These are TRUNCATED generations -- the model stops
mid-sentence -- so the numerator is the full text while the denominator is only
the fragment that was spoken. They are not rate anomalies and must not enter
d_ref: a single one moves d_ref from 0.58 to 2.90. Files above --cap char/s are
excluded from d_ref and reported separately as truncation_pct.

truncation_pct is a CHEAP FLAG, NOT A MEASUREMENT. The cap only catches severe
truncation; a generation dropping 20 % of its text lands inside the normal range
and is invisible. That is what the content filter below is for.

CONTENT FILTER (MAX_WER)
------------------------
The numerator is the TARGET text, so a content error reads as a pace error:
dropped words look fast, a runaway or hallucinated tail looks slow. A few such
utterances dominate W2 (zonos2_accurate ja: 13.2x ceiling on every utterance, 6.4x
on the 93 % the ASR heard; docs/controls_2026-08.md C7). Only utterances whose
WER (the leaf's wav_res_ref_text.wer.raw) is <= MAX_WER are scored, so this
measures pace and intelligibility stays in WER; a leaf with no transcription
gets no char/s. heard_pct reports the share kept. The reference (real prompt
speech, no transcription) is not filtered.

Usage
-----
    python scripts/prosody/speaking_rate_dist.py                      # clean / en
    python scripts/prosody/speaking_rate_dist.py --test-set in-the-wild --lang en
    python scripts/prosody/speaking_rate_dist.py --models gt_clean --out /tmp/x.csv

Reference is data/<test_set>/<lang>/waveform, keyed to prompt_text; systems are
scored against data/<test_set>/<lang>/text. Both sets use the uttid_N uid scheme
(reference file prompt_audio_N.wav -> uttid_N; system file uttid_N.wav), so utt_of
serves both. Two results layouts are walked (see find_systems):
the nested per-model form results/<sys>/<test_set>/<lang>/wavs, and this tree's
flat baseline form results/<backend>_<test_set>/<lang>/wavs (gt_clean,
whisper_clean, ...). NB in this tree gt_clean's wavs are symlinks back to
the reference PROMPT audio but are scored against the TARGET text (a different
sentence), so its char/s is prompt-duration over target-length -- a degenerate
baseline that sits well above the ceiling, not at it. d_ref only becomes a real
pacing score once actual generations (audio that reads the target text) land.

NOTE: char/s is only meaningful where character count tracks phonetic length.
It is fine for en/de/es/fr/it and NOT comparable across scripts -- for zh/ja a
character is a whole syllable/morpheme, so the scale differs. Never compare
char/s across languages; compare systems within one language.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from scripts.prosody.common import (CEILING_ROW, add_common_prosody_args, boot_ci,  # noqa: E402
                                    ceiling, find_systems, heard_files, utt_of, w2_fair)
from utils import vad  # noqa: E402
from utils.atomic import write_text_atomic  # noqa: E402
from utils.scp import load_text  # noqa: E402


def _probe(path):
    try:
        audio, _ = librosa.load(path, sr=vad.SR)
        if audio.size == 0:
            return path, None, None
        trimmed = vad.trim_speech(audio)
        d_trim = len(trimmed) / vad.SR if trimmed is not None and len(trimmed) > 0 else None
        return path, d_trim, len(audio) / vad.SR
    except Exception:
        return path, None, None


def duration_cache_path(wav_dir):
    """Where a wav dir's durations are cached: next to a system leaf's other caches
    (<leaf>/speaking_rate_durations.tsv, beside pitch_range.tsv), or, for a reference
    dir data/<set>/<lang>/waveform, results/_speaking_rate_ref/<set>_<lang>.tsv, so no
    cache file lands in the dataset itself."""
    d = os.path.normpath(os.path.abspath(wav_dir))
    if os.path.basename(d) == "waveform":
        lang, task = os.path.basename(os.path.dirname(d)), os.path.basename(os.path.dirname(os.path.dirname(d)))
        return os.path.join(REPO, "results", "_speaking_rate_ref", f"{task}_{lang}.tsv")
    return os.path.join(os.path.dirname(d), "speaking_rate_durations.tsv")


def _read_cache(path):
    """Cached rows trimmed with the current vad.TRIM_TAG; rows from another trim are ignored."""
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) == 6 and p[0] != "name" and p[3] == vad.TRIM_TAG:
                    out[p[0]] = (int(p[1]), int(p[2]), p[3], _num(p[4]), _num(p[5]))
    return out


def _num(s):
    return None if s in ("", "nan") else float(s)


def _write_cache(path, rows):
    lines = ["name\tsize\tmtime_ns\ttrim\td_trim\td_raw\n"]
    for name in sorted(rows):
        size, mtime, trim, d_trim, d_raw = rows[name]
        lines.append(f"{name}\t{size}\t{mtime}\t{trim}\t{'' if d_trim is None else d_trim}\t"
                     f"{'' if d_raw is None else d_raw}\n")
    write_text_atomic(path, "".join(lines))


def durations(files, workers):
    """{path: (trimmed s, raw s)} (None where unreadable / no speech), from the per-dir cache
    when the file's size, mtime and the trim (vad.TRIM_TAG) all match, else probed and cached.
    Decoding and trimming every wav is the whole cost of char/s, so a rebuild only probes
    what changed."""
    by_dir = {}
    for f in files:
        by_dir.setdefault(os.path.dirname(f), []).append(f)
    out = {}
    for d, fs in by_dir.items():
        cpath = duration_cache_path(d)
        cache = _read_cache(cpath)
        todo, stat = [], {}
        for f in fs:
            st = os.stat(f)
            stat[f] = (st.st_size, st.st_mtime_ns)
            c = cache.get(os.path.basename(f))
            if c and c[:2] == stat[f]:
                out[f] = (c[3], c[4])
            else:
                todo.append(f)
        if not todo:
            continue
        vad.load()                                 # once, before the pool forks
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_probe, todo, chunksize=4))
        for path, d_trim, d_raw in res:
            out[path] = (d_trim, d_raw)
            cache[os.path.basename(path)] = (*stat[path], vad.TRIM_TAG, d_trim, d_raw)
        try:
            _write_cache(cpath, cache)
        except OSError as e:                       # read-only tree: still correct, just uncached
            print(f"  [speaking_rate] cache not written ({cpath}): {e}", file=sys.stderr)
    return out


def char_rates(files, textmap, workers):
    """-> (char/s array, mean lead+trail silence seconds)."""
    dur = durations(files, workers)
    rates, pads = [], []
    for path in files:
        d_trim, d_raw = dur[path]
        text = textmap.get(utt_of(path))
        if not text or not d_trim or d_trim <= 0:
            continue
        rates.append(len(text) / d_trim)
        pads.append(d_raw - d_trim)
    return np.array(rates), (float(np.mean(pads)) if pads else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default="clean")
    ap.add_argument("--lang", default="en")
    add_common_prosody_args(ap)
    ap.add_argument("--cap", type=float, default=40.0,
                    help="char/s above this = truncated generation, excluded from d_ref (default 40; "
                         "reference median is ~16)")
    ap.add_argument("--min-utts", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    base = os.path.join(REPO, "data", args.test_set, args.lang)
    ref_dir = os.path.join(base, "waveform")
    for p in (ref_dir, os.path.join(base, "prompt_text"), os.path.join(base, "text")):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    prompt_text = load_text(os.path.join(base, "prompt_text"))
    target_text = load_text(os.path.join(base, "text"))

    ref_all, ref_pad = char_rates(sorted(glob.glob(ref_dir + "/*.wav")), prompt_text, args.workers)
    ref = ref_all[ref_all <= args.cap]
    ceil = ceiling(ref)
    print(f"reference: {ref_dir}")
    print(f"  n={ref.size}  median {np.median(ref):.2f} char/s (speech-only)  "
          f"lead+trail silence {ref_pad:.3f}s")
    print(f"  CEILING (split-half, {len(ref)//2}v{len(ref)//2}) d_ref = {ceil:.3f}\n")

    systems = find_systems(args.test_set, args.lang, args.models)
    rows = []
    for name, d in sorted(systems.items()):
        files, heard_pct = heard_files(d)
        if len(files) < args.min_utts:
            continue
        rates, pad = char_rates(files, target_text, args.workers)
        if rates.size < args.min_utts:
            continue
        keep = rates[rates <= args.cap]
        trunc = 100.0 * (rates.size - keep.size) / rates.size
        d_ref = w2_fair(keep, ref)
        lo, hi = boot_ci(keep, ref, args.n_boot)
        rows.append(dict(system=name, n=rates.size, heard_pct=100.0 * heard_pct,
                         char_s_median=float(np.median(keep)),
                         pad_s=pad, truncation_pct=trunc, d_ref=d_ref, ci_lo=lo, ci_hi=hi,
                         x_ceiling=d_ref / ceil))
        print(f"  {name:34s} n={rates.size:4d} {np.median(keep):6.2f} char/s  "
              f"trunc {trunc:4.1f}%  d_ref {d_ref:6.3f} [{lo:.2f},{hi:.2f}]", flush=True)

    rows.sort(key=lambda r: r["d_ref"])
    out = args.out or os.path.join(REPO, "results", f"speaking_rate_{args.test_set}_{args.lang}.csv")
    import csv as _csv
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["rank", "system", "n", "heard_pct", "char_s_median", "pad_s",
                                            "truncation_pct", "d_ref", "ci_lo", "ci_hi", "x_ceiling"])
        w.writeheader()
        w.writerow(dict(rank=0, system=CEILING_ROW, n=ref.size, heard_pct=100.0,
                        char_s_median=float(np.median(ref)), pad_s=ref_pad, truncation_pct=0.0,
                        d_ref=ceil, ci_lo="", ci_hi="", x_ceiling=1.0))
        for i, r in enumerate(rows):
            w.writerow(dict(rank=i + 1, **r))
    print(f"\n{len(rows)} systems -> {out}")
    print(f"at/below ceiling: {sum(1 for r in rows if r['d_ref'] <= ceil)}/{len(rows)}")


if __name__ == "__main__":
    main()
