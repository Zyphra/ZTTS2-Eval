"""Rhythm distribution score: burstiness & friends over deduped mHuBERT-147 k-means tokens.

WHAT THIS IS
------------
A distributional distance between a system's per-utterance RHYTHM and a
reference set's, on four separately-reported dimensions (no combined mean, on
purpose — same philosophy as the Prosody Profile metric). Each utterance is
tokenized with mHuBERT-147 (layer 9, 20 ms frames) + k-means, consecutive
duplicates are collapsed, and the run lengths of the surviving segments carry
the timing signal:

    rate       deduped tokens per second of trimmed audio (segment rate,
               a phone-rate proxy; complements char/s which needs text)
    burstiness (sigma - mu) / (sigma + mu) over run lengths (Goh & Barabasi).
               -1 = metronome, 0 = Poisson, ->1 = long-stall-dominated
    npvi       normalized Pairwise Variability Index over run durations —
               the classic stress-/syllable-timing rhythm contrast
    long_frac  fraction of frames inside runs >= --long-ms (200 ms default —
               a physically 300 ms pause survives tokenization as only ~220 ms
               of stable frames, so the threshold sits below that):
               pause / stall / drawl mass

Per dimension:  d_ref = W2({system per-utt values}, {reference per-utt values}),
computed at the SAME half-vs-half geometry as the ceiling (mean over 32 random
n/2-vs-n/2 subsamples). Full-sample W2 has a lower noise floor than the
split-half ceiling and flattered every system by ~1.3-2x — fish-s2-pro read
0.86x ceiling full-vs-full but 1.14x at matched geometry. LOWER = better,
reported as x_ceiling = d_ref / ceiling (1.0 = indistinguishable at finite n).

d_ref is direction-blind, so two directional readouts ride along per dim:
  std_ratio   system std / reference std. < 1 = over-regularized (the
              RL-for-WER signature: compressed toward the reference center —
              W2 under-penalizes that).
  long_pct    % of utterances containing any >= --long-ms run, vs the
              reference's (real speech pauses/stalls; ASR-optimized systems
              produce markedly less of this mass).

WHY mHuBERT-147 + own k-means
-----------------------------
mHuBERT-147 is multilingual (147 langs), matching this benchmark's language
spread — unlike the English hubert-base behind DS-WED. The HF repo ships only
the encoder, so centroids are fitted here: MiniBatchKMeans on frames sampled
from the REFERENCE set, cached to ~/.cache/ztts2_eval/rhythm keyed by
(ref wav list, layer, k, seed, top_db). Every system in one (set, lang) is
scored against the SAME cached anchor. Token IDs therefore differ across
languages — irrelevant, since every comparison is within-language (as with
char/s and MPM).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
1. It is TIMING ONLY. Token identity is discarded after dedup; intonation and
   spectral quality are invisible (MPM / Pitch cover those).
2. It matches DISTRIBUTIONS, not per-utterance identity: W2 is
   permutation-invariant across utterances (same caveat as char/s).
3. Lead/trail silence is trimmed (librosa, --top-db); INTERNAL pauses are kept
   — they are rhythm. A system that never pauses or stalls mid-sentence
   diverges from the reference on burstiness / long_frac.

Validated by constructed perturbations (scripts/validate_rhythm_metric.py):
run that before trusting a change to this file.

Usage
-----
    python scripts/rhythm_dist.py                      # zero_shot / en
    python scripts/rhythm_dist.py --test-set zero_shot_voxblink --lang en
    python scripts/rhythm_dist.py --models omni_voxcpm2 --out /tmp/x.csv

Reference is data/<test_set>/<lang>/waveform; systems are discovered with the
same two-layout walk as speaking_rate_dist.py (which also supplies w2 and the
reference-audio skip). Requires a GPU for the mHuBERT forward.
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import glob
import librosa
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from speaking_rate_dist import w2, ceiling, find_systems  # noqa: E402

CACHE_DIR = Path(os.path.expanduser("~/.cache/ztts2_eval/rhythm"))
MODEL_ID = "utter-project/mHuBERT-147"
SR = 16000
FRAME_S = 0.02          # mHuBERT stride = 320 samples @ 16 kHz
DIMS = ("rate", "burstiness", "npvi", "long_frac")


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


# ---------------------------------------------------------------- audio I/O

def _load_trim(args):
    """Worker: load 16 kHz mono, trim lead/trail silence. Returns float32 or None."""
    path, top_db = args
    try:
        wav, _ = librosa.load(path, sr=SR)
        if wav.size == 0:
            return path, None
        wav, _ = librosa.effects.trim(wav, top_db=top_db)
        return path, wav.astype(np.float32)
    except Exception:
        return path, None


def load_dir(wav_dir, top_db, workers, limit=0):
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if limit:
        files = files[:limit]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_load_trim, [(f, top_db) for f in files], chunksize=4))
    return [(p, w) for p, w in res if w is not None and len(w) >= int(0.5 * SR)]


# ------------------------------------------------------------- tokenization

class Tokenizer:
    """mHuBERT-147 layer features -> k-means labels (GPU)."""

    def __init__(self, layer, device="cuda"):
        import torch
        from transformers import HubertModel
        self.torch = torch
        self.layer = layer
        self.device = device
        self.model = HubertModel.from_pretrained(MODEL_ID).to(device).eval()
        self.centroids = None  # (k, 768) torch tensor once fitted/loaded

    def features(self, wav):
        """(T, 768) float32 numpy for one trimmed wav."""
        t = self.torch
        with t.inference_mode():
            x = t.from_numpy(wav).unsqueeze(0).to(self.device)
            h = self.model(x, output_hidden_states=True).hidden_states[self.layer]
        return h[0].float().cpu().numpy()

    def set_centroids(self, c):
        self.centroids = self.torch.from_numpy(np.ascontiguousarray(c)).float().to(self.device)

    def labels(self, wav):
        """1-D int array of k-means labels per 20 ms frame."""
        t = self.torch
        with t.inference_mode():
            x = t.from_numpy(wav).unsqueeze(0).to(self.device)
            h = self.model(x, output_hidden_states=True).hidden_states[self.layer][0].float()
            d = t.cdist(h, self.centroids)
            return d.argmin(dim=1).cpu().numpy()


def fit_kmeans(feats_iter, k, seed, max_frames=2_000_000):
    """MiniBatchKMeans over frames pooled (reservoir-free: cap per utt) from feats_iter."""
    from sklearn.cluster import MiniBatchKMeans
    pool = []
    total = 0
    for f in feats_iter:
        pool.append(f)
        total += len(f)
        if total >= max_frames:
            break
    X = np.concatenate(pool, axis=0)[:max_frames]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=10_000,
                         n_init=3, max_no_improvement=50).fit(X)
    return km.cluster_centers_.astype(np.float32)


# ------------------------------------------------------------- rhythm dims

def run_lengths(labels):
    """Lengths (frames) of consecutive-duplicate runs. This IS the dedup."""
    if len(labels) == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    return np.diff(bounds)


def rhythm_dims(labels, long_frames, min_runs=10):
    """dict of the 4 per-utterance dims, or None if too few runs."""
    r = run_lengths(labels).astype(np.float64)
    if len(r) < min_runs:
        return None
    dur_s = len(labels) * FRAME_S
    mu, sig = r.mean(), r.std()
    npvi = 100.0 * np.mean(2.0 * np.abs(np.diff(r)) / (r[1:] + r[:-1]))
    return dict(
        rate=len(r) / dur_s,
        burstiness=(sig - mu) / (sig + mu) if (sig + mu) > 0 else 0.0,
        npvi=float(npvi),
        long_frac=float(r[r >= long_frames].sum() / r.sum()),
    )


RUNLEN_HIST_MAX = 256  # run-length histogram support in frames (5.12 s); longer runs clip


def dims_for_wavs(tok, wavs, long_frames, desc="", with_uids=False):
    """-> {dim: np.array} over utterances (skipping too-short-token ones).
    with_uids=True additionally returns the kept utterances' file stems and the
    pooled run-length histogram (index = run length in frames, clipped at
    RUNLEN_HIST_MAX; feeds the dwell-time ridgeline)."""
    from tqdm import tqdm
    out = {d: [] for d in DIMS}
    uids = []
    hist = np.zeros(RUNLEN_HIST_MAX + 1, dtype=np.int64)
    for path, wav in tqdm(wavs, desc=desc or "tokenize", leave=False):
        labels = tok.labels(wav)
        dd = rhythm_dims(labels, long_frames)
        if dd is None:
            continue
        uids.append(Path(path).stem)
        for d in DIMS:
            out[d].append(dd[d])
        r = np.minimum(run_lengths(labels), RUNLEN_HIST_MAX)
        hist += np.bincount(r, minlength=RUNLEN_HIST_MAX + 1)
    arrs = {d: np.array(v) for d, v in out.items()}
    return (arrs, uids, hist) if with_uids else arrs


# ------------------------------------------------------------------ anchor

def anchor_key(ref_dir, layer, k, seed, top_db, long_frames):
    # Key over the ref dir CONTENTS (names + sizes + mtimes), not names alone:
    # regenerating a wav in place must invalidate the anchor (same fix as
    # utils/score_mpm_shuffle_floor._key). v2 bumps past name-only entries.
    wavs = sorted(p for p in Path(ref_dir).iterdir() if p.suffix.lower() == ".wav")
    content = "\n".join(f"{p.name}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}"
                        for p in wavs)
    h = hashlib.md5((content +
                     f"|{layer}|{k}|{seed}|{top_db}|{long_frames}|v2").encode())
    return h.hexdigest()[:16]


def build_anchor(tok, ref_dir, layer, k, seed, top_db, long_frames, workers):
    """Return (centroids, {dim: ref values}, {dim: ceiling}); disk-cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = anchor_key(ref_dir, layer, k, seed, top_db, long_frames)
    npz = CACHE_DIR / f"{key}.npz"
    if npz.exists():
        z = np.load(npz)
        print(f"[cache] rhythm anchor {key} (ref={ref_dir})")
        cents = z["centroids"]
        ref = {d: z[f"ref_{d}"] for d in DIMS}
        ceil = {d: float(z[f"ceil_{d}"]) for d in DIMS}
        return cents, ref, ceil

    print(f"[build] rhythm anchor {key} from {ref_dir}")
    wavs = load_dir(ref_dir, top_db, workers)
    if len(wavs) < 100:
        raise SystemExit(f"only {len(wavs)} usable reference wavs in {ref_dir}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(wavs))
    cents = fit_kmeans((tok.features(wavs[i][1]) for i in order), k, seed)
    tok.set_centroids(cents)
    ref = dims_for_wavs(tok, wavs, long_frames, desc="ref")
    ceil = {d: ceiling(ref[d]) for d in DIMS}
    np.savez(npz, centroids=cents,
             **{f"ref_{d}": ref[d] for d in DIMS},
             **{f"ceil_{d}": np.float64(ceil[d]) for d in DIMS})
    return cents, ref, ceil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default="zero_shot")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=None,
                    help="CSV path; default results/rhythm_<set>_<lang>.csv")
    ap.add_argument("--layer", type=int, default=9,
                    help="mHuBERT hidden_states index (9 = the layer its own "
                         "training targets were clustered from)")
    ap.add_argument("--kmeans-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-db", type=float, default=35.0)
    ap.add_argument("--long-ms", type=float, default=200.0,
                    help="run length (ms) above which frames count as long_frac mass")
    ap.add_argument("--min-utts", type=int, default=250)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dump-dir", default=None,
                    help="also write per-utterance dims + uids per system to "
                         "<dump-dir>/<system>.npz (for matched / dispersion audits)")
    args = ap.parse_args()

    long_frames = max(1, round(args.long_ms / 1000.0 / FRAME_S))

    ref_dir = os.path.join(REPO, "data", args.test_set, args.lang, "waveform")
    if not os.path.isdir(ref_dir):
        raise SystemExit(f"missing {ref_dir}")

    tok = Tokenizer(args.layer)
    cents, ref, ceil = build_anchor(tok, ref_dir, args.layer, args.kmeans_k,
                                    args.seed, args.top_db, long_frames, args.workers)
    tok.set_centroids(cents)

    print(f"reference: {ref_dir}  n={len(ref['rate'])}")
    for d in DIMS:
        print(f"  {d:<10} median {np.median(ref[d]):8.3f}   ceiling(W2 split-half) {ceil[d]:.4f}")

    systems = find_systems(args.test_set, args.lang, args.models)
    rows = []
    for name, wav_dir in sorted(systems.items()):
        wavs = load_dir(wav_dir, args.top_db, args.workers)
        if len(wavs) < args.min_utts:
            continue
        sysd, uids, runlen_hist = dims_for_wavs(tok, wavs, long_frames,
                                                desc=name, with_uids=True)
        if len(sysd["rate"]) < args.min_utts:
            continue
        if args.dump_dir:
            os.makedirs(args.dump_dir, exist_ok=True)
            np.savez(os.path.join(args.dump_dir, name.replace("/", "__") + ".npz"),
                     uids=np.array(uids), runlen_hist=runlen_hist, **sysd)
        row = dict(system=name, n=len(sysd["rate"]),
                   long_pct=float(100 * np.mean(sysd["long_frac"] > 0)))
        msg = f"  {name:34s} n={row['n']:4d}"
        for d in DIMS:
            dref = w2_fair(sysd[d], ref[d], seed=args.seed)
            row[f"{d}_median"] = float(np.median(sysd[d]))
            row[f"{d}_d_ref"] = dref
            row[f"{d}_x_ceiling"] = dref / ceil[d] if ceil[d] > 0 else float("nan")
            row[f"{d}_std_ratio"] = float(sysd[d].std() / ref[d].std()) if ref[d].std() > 0 else float("nan")
            msg += f"  {d[:5]} {dref:7.4f} ({row[f'{d}_x_ceiling']:5.1f}x)"
        rows.append(row)
        print(msg, flush=True)

    rows.sort(key=lambda r: r["burstiness_d_ref"])
    out = args.out or os.path.join(REPO, "results",
                                   f"rhythm_{args.test_set}_{args.lang}.csv")
    import csv
    fields = ["rank", "system", "n", "long_pct"] + \
        [f"{d}_{s}" for d in DIMS for s in ("median", "d_ref", "x_ceiling", "std_ratio")]
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        wtr.writeheader()
        ceil_row = dict(rank=0, system="__CEILING_split_half__", n=len(ref["rate"]),
                        long_pct=float(100 * np.mean(ref["long_frac"] > 0)))
        for d in DIMS:
            ceil_row[f"{d}_median"] = float(np.median(ref[d]))
            ceil_row[f"{d}_d_ref"] = ceil[d]
            ceil_row[f"{d}_x_ceiling"] = 1.0
            ceil_row[f"{d}_std_ratio"] = 1.0
        wtr.writerow(ceil_row)
        for i, r in enumerate(rows):
            wtr.writerow(dict(rank=i + 1, **r))
    print(f"\n{len(rows)} systems -> {out}")


if __name__ == "__main__":
    main()
