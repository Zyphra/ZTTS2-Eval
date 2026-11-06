"""Shared helpers of the prosody metrics (char/s, rhythm, pitch range, paired ITW, MPM)
and their analysis/plot callers: the TTSDS2 2-Wasserstein and its split-half ceiling,
the WER <= MAX_WER content filter, the system walk, and the shared CLI flags."""
import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.benchmark import iter_leaves, sample_uids  # noqa: E402
from utils.per_uid import parse_wer  # noqa: E402
from utils.scp import load_scp_items  # noqa: E402

CEILING_ROW = "__CEILING_split_half__"


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
    return {utt_of(u) for u, d in parse_wer(raw).items()
            if d["wer"] is not None and d["wer"] <= max_wer}


def heard_files(wav_dir):
    """(the leaf's wavs the ASR heard, share of its wavs they are) — the content
    filter (speaking_rate_dist docstring); ([], 0.0) for an untranscribed leaf."""
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    heard = heard_uids(wav_dir)
    if heard is None or not files:
        return [], 0.0
    kept = [f for f in files if utt_of(f) in heard]
    return kept, len(kept) / len(files)


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


def dmean_ci(x, ref, n_boot, seed=0):
    """95% percentile CI of mean(x) - mean(ref) under independent utterance resamples."""
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    br = rng.choice(ref, size=(n_boot, ref.size), replace=True).mean(axis=1)
    d = bx - br
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def dmean_band(ref, n_draws=1000, seed=0):
    """p95 of |mean(A*) - mean(B*)| over independent size-n bootstrap re-draw pairs
    of the reference: the null distribution of dmean at the n-vs-n geometry dmean
    is computed at (a half-vs-half split would be sqrt(2) wider)."""
    rng = np.random.default_rng(seed)
    a = rng.choice(ref, size=(n_draws, ref.size), replace=True).mean(axis=1)
    b = rng.choice(ref, size=(n_draws, ref.size), replace=True).mean(axis=1)
    return float(np.percentile(np.abs(a - b), 95))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def is_reference_audio(wav_dir, test_set, lang, probe=8):
    """True when a "system"'s wavs ARE the reference prompt recordings.

    On FLEURS-R, results/gt_clean/<lang>/wavs/uttid_N.wav symlinks straight back
    to data/clean/<lang>/waveform/prompt_audio_N.wav — so that baseline speaks
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
    prompts = {uid: os.path.realpath(path if os.path.isabs(path) else os.path.join(REPO, path))
               for uid, path in load_scp_items(scp) if uid and path}
    hits = seen = 0
    for f in sorted(glob.glob(os.path.join(wav_dir, "*.wav")))[:probe]:
        p = prompts.get(utt_of(f))
        if not p:
            continue
        seen += 1
        hits += os.path.realpath(f) == p
    return seen > 0 and hits == seen


def find_systems(test_set, lang, only=None):
    """{system: wav dir} of every system leaf for one (test_set, lang)
    (utils.benchmark.iter_leaves), minus the whisper_* ASR-comparison diagnostics and
    any "system" whose wavs are the reference prompt audio; `only` keeps a subset."""
    out = {name: os.path.join(leaf, "wavs")
           for name, leaf in iter_leaves(test_set, lang, "wavs").items()
           if not name.startswith("whisper_")}
    # a "system" that is really the reference recordings tells us nothing about pacing
    for name in [n for n, d in out.items() if is_reference_audio(d, test_set, lang)]:
        print(f"  skip {name} ({test_set}/{lang}): wavs are the reference prompt audio",
              file=sys.stderr)
        del out[name]
    if only:
        out = {k: v for k, v in out.items() if k in set(only)}
    return out


def add_common_prosody_args(ap, workers=48):
    """--models / --out / --workers, shared by every prosody metric CLI."""
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of system dirs; default = all found")
    ap.add_argument("--out", default=None, help="CSV path (default: see the script docstring)")
    ap.add_argument("--workers", type=int, default=workers)


def sample_and_need(task, lang, n_utts, seed, min_utts):
    """(the deterministic uid sample, the usable-utterance minimum capped at 80% of it)."""
    uids = sample_uids(task, lang, n_utts, seed)
    return uids, min(min_utts, int(0.8 * len(uids)))


def content_hash(ref_dir, tail, keep=None):
    """(md5[:16] of the ref dir CONTENTS (names + sizes + mtimes) + `tail`, wavs list).

    Content, not names alone: regenerating a wav in place must invalidate a cached
    anchor. `keep` (a uid set) filters the list via utt_of, so removing items from
    the benchmark invalidates it too; the caller's `tail` carries its own params/version."""
    wavs = sorted(p for p in Path(ref_dir).iterdir() if p.suffix.lower() == ".wav"
                  and (keep is None or utt_of(str(p)) in keep))
    content = "\n".join(f"{p.name}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}"
                        for p in wavs)
    return hashlib.md5((content + tail).encode()).hexdigest()[:16], wavs
