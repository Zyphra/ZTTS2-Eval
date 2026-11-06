"""Selectivity validation for the rhythm metric (scripts/rhythm_dist.py).

Constructed-ground-truth design (results/prosody_metric_audit + FPD post-mortem:
correlating ~12 systems against listeners has no power; perturbations with a
known answer do). The reference set is split in half:

    A  -> anchor: k-means centroids, reference dim distributions, ceilings
    B  -> probes: perturbed copies scored against A

Probes and what a WORKING metric must do with them:

    null         B untouched.            Every dim ~= the A-vs-B sampling noise
                                         (this row IS the empirical null).
    tempo_1.25   uniform 25% speed-up.   rate moves hard; others move little.
    tempo_0.80   uniform 25% slow-down.  rate moves hard, other direction.
                 Tempo probes use ffmpeg's atempo (WSOLA). librosa's
                 time_stretch is a PHASE VOCODER: it smears transients enough
                 to merge token boundaries, so a 25% speed-up LOWERED the
                 measured segment rate (30.0 -> 27.9 tokens/s) — the probe was
                 broken, not the metric (WSOLA + naive resampling both give
                 the correct monotone response, 30.9 -> 34.6 / 26.7).
    pause_2x300  two 300 ms interior     long_frac + burstiness move hard;
                 silences inserted.      rate dilutes slightly.
    noise_snr20  white noise @ 20 dB.    ROBUSTNESS NULL: if timing dims move,
                                         the metric is reading audio quality,
                                         not rhythm (the HubSR failure mode).
    reverse      time-reversed audio.    NULL: run-length distribution is
                                         direction-blind; nothing should move.
    pad_sil_1s   1 s digital silence     PADDING NULL (the Allosaurus-SR
                 lead+trail, then the    footgun: TTSDS SR *ranked* lead/trail
                 scoring path's cut.     silence, rho +0.6 with padding). The
    pad_hiss_1s  same but -45 dB-rel-RMS probe goes through the same
                 hiss (vocoder floor).   speech-span cut as load_dir, so ~1x null
    pad_noise_1s same but -20 dB-rel-RMS everywhere = the cut actually works.
                 noise (an audible floor
                 the old 35 dB energy
                 rule kept, C8).
    lead_tone_1s 1 s steady 200 Hz hum   EDGE NULL: generated lead-in tones are
                 (5 harmonics, -6 dB     a defect, but not rhythm; they must not
                 rel RMS) before the     read as a stall (C9).
                 speech, then the cut.

Output: selectivity matrix of d_ref per (probe, dim), each also expressed as a
multiple of the null probe's d_ref for that dim. PASS = targeted cells >> null,
off-target + control cells ~ null.

Usage:
    python scripts/analysis/validate_rhythm_metric.py                # clean/en, k=100
    python scripts/analysis/validate_rhythm_metric.py --kmeans-k 500 --n 600
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from speaking_rate_dist import w2, ceiling  # noqa: E402
from rhythm_dist import (Tokenizer, load_dir, fit_kmeans, dims_for_wavs,
                         DIMS, SR, FRAME_S)  # noqa: E402
sys.path.insert(0, REPO)
from utils import vad  # noqa: E402


def atempo(wav, rate):
    """WSOLA tempo change via ffmpeg (transient-preserving; see docstring)."""
    import librosa
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "a.wav"), os.path.join(td, "b.wav")
        sf.write(a, wav, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a,
                        "-filter:a", f"atempo={rate}", b], check=True)
        out, _ = librosa.load(b, sr=SR)
    return out.astype(np.float32)


def insert_pauses(wav, rng, n_pauses=2, pause_s=0.3):
    """Insert n silent gaps at random interior points (10%..90% of the file)."""
    out = wav
    for _ in range(n_pauses):
        pos = rng.integers(int(0.1 * len(out)), int(0.9 * len(out)))
        gap = np.zeros(int(pause_s * SR), dtype=np.float32)
        out = np.concatenate([out[:pos], gap, out[pos:]])
    return out


def add_noise(wav, rng, snr_db=20.0):
    p_sig = float(np.mean(wav ** 2)) or 1e-9
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (wav + rng.normal(0, np.sqrt(p_noise), len(wav)).astype(np.float32))


def cut(wav):
    """The scoring path's edge cut (rhythm_dist._load_trim) on an in-memory wav."""
    span = vad.speech_span(wav)
    return wav if span is None else wav[span[0]:span[1]]


def pad_then_trim(wav, rng, pad_s=1.0, hiss_db=None):
    """Pad lead+trail, then re-cut exactly as the scoring path (load_dir) does. If the
    cut works this is a no-op; hiss_db pads with noise (dB relative to speech RMS)
    instead of digital zeros."""
    n = int(pad_s * SR)
    if hiss_db is None:
        pads = (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
    else:
        amp = (float(np.sqrt(np.mean(wav ** 2))) or 1e-9) * 10 ** (hiss_db / 20)
        pads = (rng.normal(0, amp, n).astype(np.float32),
                rng.normal(0, amp, n).astype(np.float32))
    return cut(np.concatenate([pads[0], wav, pads[1]]))


def lead_tone(wav, pad_s=1.0, f0=200.0, level_db=-6.0):
    """A steady harmonic hum (f0 and 4 harmonics at 1/k amplitude) before the speech, at
    level_db relative to speech RMS, then the scoring path's cut — the constructed version
    of the tones some systems generate before speaking."""
    t = np.arange(int(pad_s * SR)) / SR
    hum = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    rms = float(np.sqrt(np.mean(wav ** 2))) or 1e-9
    hum = (hum / np.sqrt(np.mean(hum ** 2)) * rms * 10 ** (level_db / 20)).astype(np.float32)
    return cut(np.concatenate([hum, wav]))


def make_probes(wavs_b, seed):
    rng = np.random.default_rng(seed)
    probes = {"null": [(p, w) for p, w in wavs_b]}
    probes["tempo_1.25"] = [(p, atempo(w, 1.25)) for p, w in wavs_b]
    probes["tempo_0.80"] = [(p, atempo(w, 0.80)) for p, w in wavs_b]
    probes["pause_2x300"] = [(p, insert_pauses(w, rng)) for p, w in wavs_b]
    probes["noise_snr20"] = [(p, add_noise(w, rng)) for p, w in wavs_b]
    probes["reverse"] = [(p, np.ascontiguousarray(w[::-1])) for p, w in wavs_b]
    probes["pad_sil_1s"] = [(p, pad_then_trim(w, rng)) for p, w in wavs_b]
    probes["pad_hiss_1s"] = [(p, pad_then_trim(w, rng, hiss_db=-45.0)) for p, w in wavs_b]
    probes["pad_noise_1s"] = [(p, pad_then_trim(w, rng, hiss_db=-20.0)) for p, w in wavs_b]
    probes["lead_tone_1s"] = [(p, lead_tone(w)) for p, w in wavs_b]
    return probes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default="clean")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=600,
                    help="total usable utts (half anchor, half probe)")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--kmeans-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long-ms", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None,
                    help="CSV path; default results/rhythm_validation_<set>_<lang>_k<k>.csv")
    args = ap.parse_args()

    long_frames = max(1, round(args.long_ms / 1000.0 / FRAME_S))
    ref_dir = os.path.join(REPO, "data", args.test_set, args.lang, "waveform")
    wavs = load_dir(ref_dir, args.workers)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(wavs))[:args.n]
    half = len(order) // 2
    A = [wavs[i] for i in order[:half]]
    B = [wavs[i] for i in order[half:2 * half]]
    print(f"ref={ref_dir}  usable={len(wavs)}  anchor |A|={len(A)}  probe |B|={len(B)}")

    tok = Tokenizer(args.layer)
    print("[fit] k-means on A ...")
    tok.set_centroids(fit_kmeans((tok.features(w) for _, w in A),
                                 args.kmeans_k, args.seed))
    ref = dims_for_wavs(tok, A, long_frames, desc="anchor A")
    ceil = {d: ceiling(ref[d]) for d in DIMS}
    for d in DIMS:
        print(f"  A {d:<10} median {np.median(ref[d]):8.3f}  split-half ceiling {ceil[d]:.4f}")

    rows = []
    null_dref = None
    for probe, pw in make_probes(B, args.seed).items():
        pd = dims_for_wavs(tok, pw, long_frames, desc=probe)
        row = {"probe": probe, "n": len(pd["rate"])}
        for d in DIMS:
            row[f"{d}_median"] = float(np.median(pd[d]))
            row[f"{d}_d_ref"] = w2(pd[d], ref[d])
        if probe == "null":
            null_dref = {d: max(row[f"{d}_d_ref"], 1e-9) for d in DIMS}
        for d in DIMS:
            row[f"{d}_x_null"] = row[f"{d}_d_ref"] / null_dref[d]
        rows.append(row)

    hdr = f"\n{'probe':<14}" + "".join(f"{d:>22}" for d in DIMS)
    print(hdr + "\n" + "-" * len(hdr))
    for row in rows:
        line = f"{row['probe']:<14}"
        for d in DIMS:
            line += f"{row[f'{d}_d_ref']:>13.4f} ({row[f'{d}_x_null']:>5.1f}x)"
        print(line)
    print("\n(each cell: W2 d_ref vs anchor half, and multiple of the null row)")

    out = args.out or os.path.join(
        REPO, "results",
        f"rhythm_validation_{args.test_set}_{args.lang}_k{args.kmeans_k}.csv")
    import csv
    fields = ["probe", "n"] + [f"{d}_{s}" for d in DIMS
                               for s in ("median", "d_ref", "x_null")]
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        wtr.writeheader()
        for row in rows:
            wtr.writerow(row)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
