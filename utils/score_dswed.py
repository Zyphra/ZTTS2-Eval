"""Score prosody diversity via mean pairwise DS-WED across N seed dirs.

DS-WED (arXiv:2509.19928, ICASSP'26) computes a weighted edit distance over
HuBERT/WavLM k-means tokens between two same-text audios. Higher = more
prosodic variation. To turn this into a per-checkpoint metric, we generate K
samples per (text, prompt) pair with different seeds and average pairwise
DS-WED across them.

Inputs are K result directories each containing a wavs/ subdir with shared uids
(i.e. pass <decode_dir>, not <decode_dir>/wavs -- the wavs/ suffix is appended
here), typically one per --seed value. For each uid present in all dirs we compute
C(K, 2) pairwise scores and take the mean; the corpus-level mean is the
prosody-diversity figure-of-merit.

Usage:
    python utils/score_dswed.py \\
        --seed-dirs results/seed0/en results/seed1/en results/seed2/en \\
        --out dswed_per_uid.tsv --mean-out dswed_mean.txt
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
import torch
from tqdm import tqdm

# silence dswed's chatty http logs and progress bars
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from dswed import DSWED  # noqa: E402
from dswed.ds_wed import int_array_to_chinese_unicode  # noqa: E402
from Levenshtein import distance as levenshtein_distance  # noqa: E402

TARGET_SR = 16000


def wav_to_token_str(scorer, wav):
    """Decode one wav to its k-means token string (the per-wav half of DSWED.score).

    DSWED.score(a, b) re-decodes BOTH wavs on every call, so a K-seed uid decodes
    each wav (K-1) times. Decoding once per wav here and running only the cheap
    Levenshtein per pair gives bit-identical results with (K-1)x less HuBERT/VAD/
    k-means work (the CPU-bound bottleneck). Mirrors DSWED.score exactly.
    """
    w = torch.from_numpy(wav).float()
    w = scorer.trim_silence(w)
    w = w.unsqueeze(0).to(scorer.device)
    labels = scorer.decode_label(w)
    return int_array_to_chinese_unicode(labels)


def load_wav_16k_mono(path):
    """Read a wav file, downmix to mono, resample to 16 kHz, return float32 np.array."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).numpy().astype(np.float32)


def find_uids(seed_dirs):
    """Return uids that exist as <dir>/wavs/<uid>.wav in every seed dir."""
    sets = []
    for d in seed_dirs:
        wd = Path(d) / "wavs"
        if not wd.is_dir():
            raise SystemExit(f"missing wav dir: {wd}")
        sets.append({p.stem for p in wd.glob("*.wav")})
    common = sorted(set.intersection(*sets))
    if not common:
        raise SystemExit("no uids common to all seed dirs")
    return common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dirs", nargs="+", required=True,
                    help="K decode-dir/<lang> roots, each with wavs/<uid>.wav")
    ap.add_argument("--out", required=True,
                    help="per-uid TSV: uid\\tmean_dswed\\tk")
    ap.add_argument("--mean-out", required=True,
                    help="single-line file with the corpus mean")
    ap.add_argument("--model-type", default="hubert-base",
                    choices=["hubert-base", "wavlm-base"])
    ap.add_argument("--vocab", type=int, default=50,
                    choices=[20, 50, 100, 200, 500, 1000, 2000])
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="only score the first N common uids")
    args = ap.parse_args()

    k = len(args.seed_dirs)
    if k < 2:
        raise SystemExit("need at least 2 seed dirs for pairwise DS-WED")

    uids = find_uids(args.seed_dirs)
    if args.limit > 0:
        uids = uids[:args.limit]
    print(f"[dswed] {k} seed dirs, {len(uids)} common uids "
          f"({k * (k - 1) // 2} pairs/uid -> {len(uids) * k * (k - 1) // 2} pair scores)")

    scorer = DSWED(model_type=args.model_type, vocab=args.vocab, layer=args.layer)
    # DSWED sets self.device="cuda" but never moves the model there, so HuBERT
    # silently runs on CPU (~15x slower per forward). Force it onto the device.
    scorer.model.to(scorer.device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    per_uid = []
    n_failed = 0
    with torch.inference_mode(), open(args.out, "w", encoding="utf-8") as f:
        f.write("uid\tmean_dswed\tk\n")
        for uid in tqdm(uids, desc="dswed"):
            try:
                # decode each seed's wav to tokens ONCE, then pair cheaply
                toks = [wav_to_token_str(scorer,
                                         load_wav_16k_mono(Path(d) / "wavs" / f"{uid}.wav"))
                        for d in args.seed_dirs]
                scores = []
                for i in range(k):
                    for j in range(i + 1, k):
                        scores.append(levenshtein_distance(
                            toks[i], toks[j], weights=(1, 1, 1)))
                m = float(np.mean(scores))
                per_uid.append(m)
                f.write(f"{uid}\t{m:.4f}\t{k}\n")
            except Exception as e:
                # keep the declared 3-column TSV schema; the error goes to stderr
                n_failed += 1
                print(f"[dswed] {uid}: {type(e).__name__}: {e}", file=sys.stderr)
                f.write(f"{uid}\tNA\t{k}\n")

    if per_uid:
        arr = np.array(per_uid)
        mean = float(arr.mean())
        std = float(arr.std())
        median = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        with open(args.mean_out, "w", encoding="utf-8") as f:
            f.write(f"DS-WED mean: {mean:.4f}\n")
            f.write(f"DS-WED std:  {std:.4f}\n")
            f.write(f"DS-WED median: {median:.4f}\n")
            f.write(f"DS-WED p90: {p90:.4f}\n")
            f.write(f"n_uids: {len(per_uid)}\n")
            f.write(f"n_failed: {n_failed}\n")
            f.write(f"k_seeds: {k}\n")
            f.write(f"model: {args.model_type} vocab={args.vocab} layer={args.layer}\n")
        print(f"[dswed] mean={mean:.4f}  std={std:.4f}  median={median:.4f}  "
              f"p90={p90:.4f}  n={len(per_uid)}  failed={n_failed}  k={k}")
    else:
        print("[dswed] no successful scores")

    min_cov = float(os.environ.get("SCORE_MIN_COVERAGE", "0.98"))
    if not uids or len(per_uid) / len(uids) < min_cov:
        sys.exit(f"error: DS-WED coverage {len(per_uid)}/{len(uids)} < {min_cov} "
                 f"(SCORE_MIN_COVERAGE) -- outputs written but not trustworthy")


if __name__ == "__main__":
    main()
