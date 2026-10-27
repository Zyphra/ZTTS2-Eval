"""Speaker-similarity scoring with ReDimNet2 (https://github.com/PalabraAI/redimnet2).

Loads the ReDimNet2 speaker-verification model via torch.hub, extracts an embedding
for each prompt (reference voice) and each generated utterance, and reports the cosine
similarity. Defaults to ReDimNet2-B6 (vb2+vox2_v0, large-margin) -- the strongest
published variant.

Usage:
    python eval_speaker_similarity.py \\
        --prompt_wavs prompt_wav.scp --hyp_wavs wav.scp \\
        --log_file spk_simi_scores.txt

Several systems scored against the same prompts can share one run: pass the
hyp scps after --hyp_wavs and their log files after --log_file, in the same
order. The model then loads once and each prompt is embedded once and reused
for every system, instead of once per system per process.

The .scp files have '<uttid> <wav_path>' per line. Output: '<uttid> <ref> <hyp>' per
line (cosine similarity x100) plus a final 'avg' line; <ref> is 0 unless --ref_wavs
is given. Two trailing coverage lines ('#N_SCORED: <n>' / '#N_EXPECTED: <n>', shaped
so per-uid parsers skip them) record how many of the given hyp wavs actually scored,
and the exit code is non-zero when coverage falls below SCORE_MIN_COVERAGE (default
0.98) -- an `avg 0.00 0.00` stub must not look like a scored leaf.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Ordered (uttid, path) list parser; '<uttid> <wav_path>' per line.
from utils.scp import load_scp_items as load_scp  # noqa: E402
from utils.benchmark import (BENCHMARK_VERSION, SPK_HUB_REF, SPK_HUB_REPO,  # noqa: E402
                             spk_stamp)
from utils.prefetch import prefetch  # noqa: E402

TARGET_SR = 16000


def cosine(a, b):
    """Embeddings come back on the CPU (see embed in main), so the score never depends
    on which device produced them."""
    return F.cosine_similarity(a, b, dim=-1).item()


def load_wav(wav_path):
    """The scorer's only front end: (1, time) float tensor, mono, 16 kHz."""
    wav, sr = torchaudio.load(wav_path)                # (channels, time)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)            # downmix to mono
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def score_system(hyp_scp, log_file, prompt_wavs, ref_wavs, embed, prompt_cache, ref_cache,
                 stamp):
    """Score one hyp scp into log_file; returns (n_scored, n_expected, avg_hyp).

    prompt_cache (uttid -> prompt embedding) and ref_cache (uttid -> prompt-vs-ref
    score) persist across calls, so prompts shared by several systems are
    embedded once. Failed prompts are not cached and are retried."""
    hyp_wavs = load_scp(hyp_scp)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    ref_scores, hyp_scores = [], []
    todo = [(u, p) for u, p in hyp_wavs if u in prompt_wavs]

    def _load(item):
        """Decode (on a worker thread) every wav this utterance still needs."""
        uttid, hyp_path = item
        return (None if uttid in prompt_cache else load_wav(prompt_wavs[uttid]),
                load_wav(hyp_path),
                load_wav(ref_wavs[uttid]) if uttid in ref_wavs and uttid not in ref_cache
                else None)

    with open(log_file, "w") as out:
        for (uttid, _), wavs, err in prefetch(_load, todo):
            try:
                if err is not None:
                    raise err
                prompt_wav, hyp_wav, ref_wav = wavs
                if uttid not in prompt_cache:
                    prompt_cache[uttid] = embed(prompt_wav)
                prompt_emb = prompt_cache[uttid]
                hyp_emb = embed(hyp_wav)
                hyp_score = cosine(prompt_emb, hyp_emb)
                ref_score = 0.0
                if uttid in ref_wavs:
                    if uttid not in ref_cache:
                        ref_cache[uttid] = cosine(prompt_emb, embed(ref_wav))
                    ref_score = ref_cache[uttid]
            except Exception as e:
                print(f"[spk] {uttid}: {e}")
                continue
            if any(math.isnan(s) or math.isinf(s) for s in (ref_score, hyp_score)):
                print(f"Warning: {uttid}: ref_score={ref_score}, hyp_score={hyp_score}")
                continue
            ref_scores.append(ref_score)
            hyp_scores.append(hyp_score)
            out.write(f"{uttid} {ref_score * 100.0:.2f} {hyp_score * 100.0:.2f}\n")
            out.flush()

        avg_ref = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
        avg_hyp = sum(hyp_scores) / len(hyp_scores) if hyp_scores else 0.0
        out.write(f"avg {avg_ref * 100.0:.2f} {avg_hyp * 100.0:.2f}\n")
        # Coverage footer. '#'-prefixed with exactly TWO whitespace fields so every
        # existing parser skips it: utils.per_uid.parse_spk needs >=3 fields, the
        # build_results/matched-pair readers key on the 'avg' line, and the resume
        # gates grep '^avg '. Strictly additive -- the line formats above are unchanged.
        n_expected = len(hyp_wavs)
        out.write(f"#N_SCORED: {len(hyp_scores)}\n")
        out.write(f"#N_EXPECTED: {n_expected}\n")
        # Version stamps (two-field '#' lines, skipped by every parser): benchmark
        # version and the pinned embedder, checked by utils.benchmark.leaf_problems.
        out.write(f"#BENCHMARK_VERSION: {BENCHMARK_VERSION}\n")
        out.write(f"#SPK_MODEL: {stamp}\n")
    return len(hyp_scores), n_expected, avg_hyp


def add_common_arguments(ap):
    ap.add_argument("--prompt_wavs", required=True, help="scp of prompt (reference) wavs")
    ap.add_argument("--hyp_wavs", required=True, nargs="+",
                    help="scp of generated wavs; several systems may be given")
    ap.add_argument("--ref_wavs", default="", help="optional scp of ground-truth wavs")
    ap.add_argument("--log_file", required=True, nargs="+",
                    help="output score file, one per --hyp_wavs entry in the same order")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def parse_args(ap):
    args = ap.parse_args()
    if len(args.hyp_wavs) != len(args.log_file):
        ap.error(f"{len(args.hyp_wavs)} --hyp_wavs but {len(args.log_file)} --log_file; "
                 "pass one log file per hyp scp, in the same order")
    return args


def main():
    ap = argparse.ArgumentParser(description="Speaker similarity with ReDimNet2.")
    add_common_arguments(ap)
    # ReDimNet2 variant -- defaults to the strongest published model.
    ap.add_argument("--model-name", default="b6")
    ap.add_argument("--train-type", default="lm")
    ap.add_argument("--dataset", default="vb2+vox2_v0")
    args = parse_args(ap)

    # Pinned to a commit of the hub repo so the embedder cannot drift under us.
    def load(device):
        m = torch.hub.load(f"{SPK_HUB_REPO}:{SPK_HUB_REF}", "redimnet2",
                           model_name=args.model_name, train_type=args.train_type,
                           dataset=args.dataset, pretrained=True, trust_repo=True,
                           skip_validation=True, verbose=False)
        return m.to(device).eval()

    model = load(args.device)
    # CPU fallback. A runaway-long generation can exceed the GPU memory left next to the
    # ASR servers; the allocator raises OutOfMemoryError (recoverable: retry that one
    # utterance on the CPU), while an OOM inside a CUDA library raises a plain "CUDA
    # error: out of memory" that poisons the context for the rest of the process, so
    # after one of those every remaining utterance goes to the CPU. Embeddings always
    # come back on the CPU, so cached prompt embeddings survive either case.
    state = {"cpu_model": None, "gpu_ok": args.device != "cpu"}

    def cpu_model():
        if state["cpu_model"] is None:
            state["cpu_model"] = load("cpu")
        return state["cpu_model"]

    @torch.no_grad()
    def embed(wav):
        if state["gpu_ok"]:
            try:
                # (batch=1, emb_dim) -> (emb_dim,)
                return model(wav.to(args.device)).reshape(-1).cpu()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "CUDA" not in str(e):
                    raise
                print(f"[spk] {e}; scoring the rest of this run on the CPU")
                state["gpu_ok"] = False
        return cpu_model()(wav).reshape(-1)

    run(args, embed, spk_stamp(args.model_name, args.train_type, args.dataset))


def run(args, embed, stamp):
    """Score every --hyp_wavs scp into its --log_file; exits non-zero on low coverage."""
    prompt_wavs = dict(load_scp(args.prompt_wavs))
    ref_wavs = (dict(load_scp(args.ref_wavs))
                if args.ref_wavs and os.path.exists(args.ref_wavs) else {})
    prompt_cache, ref_cache = {}, {}
    min_cov = float(os.environ.get("SCORE_MIN_COVERAGE", "0.98"))
    multi = len(args.hyp_wavs) > 1
    failed = []
    for hyp_scp, log_file in zip(args.hyp_wavs, args.log_file):
        start = time.time()
        n_scored, n_expected, avg_hyp = score_system(
            hyp_scp, log_file, prompt_wavs, ref_wavs, embed, prompt_cache, ref_cache, stamp)
        print(f"speaker similarity: avg hyp = {avg_hyp * 100.0:.2f}  "
              f"(n={n_scored}/{n_expected})"
              + (f"  -> {log_file} ({time.time() - start:.0f}s)" if multi else ""), flush=True)
        if n_expected == 0 or n_scored / n_expected < min_cov:
            failed.append(f"speaker-sim coverage {n_scored}/{n_expected} < {min_cov} "
                          f"(SCORE_MIN_COVERAGE) -- {log_file} written but the leaf is "
                          f"not trustworthy")
    if failed:
        sys.exit("error: " + "\nerror: ".join(failed))


if __name__ == "__main__":
    main()
