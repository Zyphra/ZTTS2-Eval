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
   against results/gt_zero_shot/<lang>/wavs (GT renditions of the SAME uttids,
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
    python scripts/speaking_rate_dist.py                      # zero_shot / en
    python scripts/speaking_rate_dist.py --test-set zero_shot_voxblink --lang en
    python scripts/speaking_rate_dist.py --models gt_zero_shot --out /tmp/x.csv

Reference is data/<test_set>/<lang>/waveform, keyed to prompt_text; systems are
scored against data/<test_set>/<lang>/text. Both sets use the uttid_N uid scheme
(reference file prompt_audio_N.wav -> uttid_N; system file uttid_N.wav), so utt_of
serves both. Two results layouts are walked (see find_systems):
the nested per-model form results/<sys>/<test_set>/<lang>/wavs, and this tree's
flat baseline form results/<backend>_<test_set>/<lang>/wavs (gt_zero_shot,
whisper_zero_shot, ...). NB in this tree gt_zero_shot's wavs are symlinks back to
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def w2(x, y, n_iter=10, seed=0):
    """Sorted-quantile RMS distance == the 2-Wasserstein distance.

    Byte-for-byte the same procedure as TTSDS2's `ttsds.util.distances.
    wasserstein_distance` (util/distances.py:9-23) so numbers here are directly
    comparable to the TTSDS2 columns. Note their loop is partly degenerate:
    after the first iteration the shapes match, so iterations 2..n recompute an
    identical value. Kept for exact comparability.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = []
    for _ in range(n_iter):
        if len(x) > len(y):
            x = x[rng.choice(len(x), len(y), replace=False)]
        elif len(y) > len(x):
            y = y[rng.choice(len(y), len(x), replace=False)]
        out.append(np.mean((np.sort(x) - np.sort(y)) ** 2) ** 0.5)
    return float(np.mean(out))


def read_kv(path):
    """Read a `<uttid> <rest of line>` mapping (data/<set>/<lang>/text style)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def utt_of(path):
    """Map a wav to its uid. Both test sets use the uttid_N scheme: the reference
    waveform file is prompt_audio_N.wav (-> uttid_N), the system file is uttid_N.wav."""
    base = os.path.basename(path).rsplit(".", 1)[0]
    if base.startswith("prompt_audio_"):
        return "uttid_" + base.split("_")[-1]
    return base


MAX_WER = 0.2


def heard_uids(wav_dir, max_wer=MAX_WER):
    """uids of a system leaf the ASR transcribed with WER <= max_wer, read from the
    leaf's wav_res_ref_text.wer.raw (<wav path>\\t<WER fraction>\\t...); None when the
    leaf has no transcription."""
    raw = os.path.join(os.path.dirname(os.path.normpath(wav_dir)), "wav_res_ref_text.wer.raw")
    if not os.path.isfile(raw):
        return None
    out = set()
    with open(raw, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                if float(parts[1]) <= max_wer:
                    out.add(utt_of(parts[0]))
            except ValueError:
                continue
    return out


def heard_files(wav_dir):
    """(the leaf's wavs the ASR heard, share of its wavs they are) — the content
    filter (module docstring); ([], 0.0) for an untranscribed leaf."""
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    heard = heard_uids(wav_dir)
    if heard is None or not files:
        return [], 0.0
    kept = [f for f in files if utt_of(f) in heard]
    return kept, len(kept) / len(files)


def _probe(args):
    path, top_db = args
    try:
        audio, sr = librosa.load(path, sr=22050)
        if audio.size == 0:
            return path, None, None
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        d_trim = len(trimmed) / sr if len(trimmed) > 0 else None
        return path, d_trim, len(audio) / sr
    except Exception:
        return path, None, None


def char_rates(files, textmap, top_db, workers):
    """-> (char/s array, mean lead+trail silence seconds)."""
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_probe, [(f, top_db) for f in files], chunksize=4))
    rates, pads = [], []
    for path, d_trim, d_raw in res:
        text = textmap.get(utt_of(path))
        if not text or not d_trim or d_trim <= 0:
            continue
        rates.append(len(text) / d_trim)
        pads.append(d_raw - d_trim)
    return np.array(rates), (float(np.mean(pads)) if pads else float("nan"))


def ceiling(ref, n_splits=32, seed=0):
    """What a PERFECT system scores at finite n: disjoint halves of the reference.

    Systems MUST be compared at this same half-vs-half geometry or the ceiling is
    understated and systems are flattered.
    """
    rng = np.random.default_rng(seed)
    half = len(ref) // 2
    return float(np.mean([
        w2(ref[(i := rng.permutation(len(ref)))[:half]], ref[i[half:2 * half]])
        for _ in range(n_splits)
    ]))


def w2_fair(x, ref, n_splits=32, seed=0):
    """W2 at the ceiling's half-vs-half geometry: mean over random n/2-vs-n/2
    subsamples of both sides. Comparing full samples against a split-half
    ceiling flatters systems (the W2 noise floor shrinks with n)."""
    rng = np.random.default_rng(seed)
    half = len(ref) // 2
    return float(np.mean([
        w2(x[rng.choice(len(x), min(half, len(x)), replace=False)],
           ref[rng.choice(len(ref), half, replace=False)])
        for _ in range(n_splits)
    ]))


def boot_ci(x, ref, n_boot=400, seed=0):
    """Bootstrap CI of the split-half-geometry distance (same geometry as d_ref)."""
    rng = np.random.default_rng(seed)
    bs = np.array([
        w2_fair(x[rng.integers(0, x.size, x.size)], ref[rng.integers(0, ref.size, ref.size)],
                n_splits=4, seed=b)
        for b in range(n_boot)
    ])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def is_reference_audio(wav_dir, test_set, lang, probe=8):
    """True when a "system"'s wavs ARE the reference prompt recordings.

    On FLEURS-R, results/gt_zero_shot/<lang>/wavs/uttid_N.wav symlinks straight back
    to data/zero_shot/<lang>/waveform/prompt_audio_N.wav — so that baseline speaks
    prompt_text, while every real system speaks the target text. Pairing its durations
    with the target text measures len(one sentence) / duration(a different one), which
    inflates d_ref without moving the median; pairing them with prompt_text instead
    reproduces the reference sample exactly, for d_ref = 0 by construction. Neither is
    a measurement, so the row is dropped.

    VoxBlink2 is unaffected: there the GT baseline holds the *target* recordings, a
    genuinely different sample from the prompts, and it scores near the ceiling.
    """
    scp = os.path.join(REPO, "data", test_set, lang, "prompt_wav.scp")
    if not os.path.isfile(scp):
        return False
    prompts = {}
    with open(scp, encoding="utf-8") as fh:
        for line in fh:
            uid, _, path = line.strip().partition(" ")
            if uid and path:
                prompts[uid] = os.path.realpath(
                    path if os.path.isabs(path) else os.path.join(REPO, path))
    hits = seen = 0
    for f in sorted(glob.glob(os.path.join(wav_dir, "*.wav")))[:probe]:
        p = prompts.get(utt_of(f))
        if not p:
            continue
        seen += 1
        hits += os.path.realpath(f) == p
    return seen > 0 and hits == seen


def find_systems(test_set, lang, only=None):
    """Locate every system's wav dir for one (test_set, lang).

    Two layouts are walked so the metric runs on both trees:
      - nested (ZTTS1 / per-model gens):  results/<sys>/<test_set>/<lang>/wavs
        (and results/<group>/<sys>/<test_set>/<lang>/wavs) -> name = <sys> / <group>/<sys>
      - flat (ZTTS2 baselines): results/<backend>_<test_set>/<lang>/wavs, e.g.
        gt_zero_shot, whisper_zero_shot -> name = the full dir (backend kept).
        The `*<test_set>` glob keeps the two test-sets disjoint: results/*zero_shot
        does not match gt_zero_shot_voxblink, and vice versa.
    """
    results_root = os.path.join(REPO, "results")
    out = {}
    pats = (
        f"results/*/{test_set}/{lang}/wavs",     # nested: results/<sys>/<set>/<lang>/wavs
        f"results/*/*/{test_set}/{lang}/wavs",   # nested: results/<group>/<sys>/<set>/...
        f"results/*{test_set}/{lang}/wavs",      # flat:   results/<backend>_<set>/<lang>/wavs
    )
    for pat in pats:
        for d in glob.glob(os.path.join(REPO, pat)):
            rel = os.path.relpath(d, results_root)          # <...>/<lang>/wavs
            name = rel[: -len(f"/{lang}/wavs")]             # strip trailing /<lang>/wavs
            if name.endswith(f"/{test_set}"):               # nested -> drop the /<set> segment
                name = name[: -len(f"/{test_set}")]
            out[name] = d
    # whisper_* dirs are ASR-comparison diagnostics, not TTS systems
    for name in [n for n in out if n.startswith("whisper_")]:
        del out[name]
    # a "system" that is really the reference recordings tells us nothing about pacing
    for name in [n for n, d in out.items() if is_reference_audio(d, test_set, lang)]:
        print(f"  skip {name} ({test_set}/{lang}): wavs are the reference prompt audio",
              file=sys.stderr)
        del out[name]
    if only:
        out = {k: v for k, v in out.items() if k in set(only)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default="zero_shot")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--models", nargs="*", default=None, help="subset of system dirs; default = all found")
    ap.add_argument("--out", default=None, help="CSV path; default results/speaking_rate_<set>_<lang>.csv")
    ap.add_argument("--cap", type=float, default=40.0,
                    help="char/s above this = truncated generation, excluded from d_ref (default 40; "
                         "reference median is ~16)")
    ap.add_argument("--top-db", type=float, default=35.0, help="librosa.effects.trim threshold (default 35)")
    ap.add_argument("--min-utts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    base = os.path.join(REPO, "data", args.test_set, args.lang)
    ref_dir = os.path.join(base, "waveform")
    for p in (ref_dir, os.path.join(base, "prompt_text"), os.path.join(base, "text")):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    prompt_text = read_kv(os.path.join(base, "prompt_text"))
    target_text = read_kv(os.path.join(base, "text"))

    ref_all, ref_pad = char_rates(sorted(glob.glob(ref_dir + "/*.wav")), prompt_text, args.top_db, args.workers)
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
        rates, pad = char_rates(files, target_text, args.top_db, args.workers)
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
        w.writerow(dict(rank=0, system="__CEILING_split_half__", n=ref.size, heard_pct=100.0,
                        char_s_median=float(np.median(ref)), pad_s=ref_pad, truncation_pct=0.0,
                        d_ref=ceil, ci_lo="", ci_hi="", x_ceiling=1.0))
        for i, r in enumerate(rows):
            w.writerow(dict(rank=i + 1, **r))
    print(f"\n{len(rows)} systems -> {out}")
    print(f"at/below ceiling: {sum(1 for r in rows if r['d_ref'] <= ceil)}/{len(rows)}")


if __name__ == "__main__":
    main()
