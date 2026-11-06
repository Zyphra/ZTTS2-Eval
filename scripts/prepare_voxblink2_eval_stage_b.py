"""Stage B of the VoxBlink2-derived eval: score each prompt candidate from the
Stage-A manifest with MSR-UTMOS, stratified-sample across MOS bins per language,
and emit `data/in-the-wild/<lang>/{text,prompt_text,prompt_wav.scp,waveform/}`
in the layout the existing scoring pipeline (run_score.sh) expects.

Stratification is by *prompt* MSR-UTMOS — the audio we feed the TTS — across
N_BINS uniform bins in [MSR_UTMOS_MIN, MSR_UTMOS_MAX]. Prompts are selected round-robin
across bins up to --target-total per language, one pair per speaker (Stage A
already enforced cross-video). A large per-language N matters because TTS error
metrics are tail-driven (dominated by rare catastrophic generations).

MSR-UTMOS scores are cached to <output-dir>/_msr_utmos_cache.tsv; rerunning skips
already-scored candidates.

Usage:
  python scripts/prepare_voxblink2_eval_stage_b.py \\
      [--manifest data/in-the-wild/_stage_a/manifest.tsv] \\
      [--output-dir data/in-the-wild] \\
      [--target-total 1500] [--n-bins 5] [--device cuda]
"""
import argparse
import collections
import csv
import os
import random
import shutil
import sys

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.score_msr_utmos import DEFAULT_CKPT, load_model, score_wave  # noqa: E402
from utils.text_norm import near_duplicate  # noqa: E402

# Local VoxBlink2 audio tree (<spk>/<vid>/<seg>.wav). Set VOXBLINK2_AUDIO_ROOT to
# point at your copy of the dataset.
AUDIO_ROOT = os.environ.get("VOXBLINK2_AUDIO_ROOT", "/path/to/VoxBlink2/audio")
SR = 16000


def wav_path(speaker, vid, seg):
    return f"{AUDIO_ROOT}/{speaker}/{vid}/{seg}.wav"


def load_manifest(path):
    rows = []
    with open(path) as f:
        # QUOTE_NONE: the manifest is raw tab-separated (Stage A never quotes), so a
        # literal '"' in a transcript must NOT open a quoted field (which would
        # swallow following rows/columns and corrupt candidate_ids).
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in r:
            row["prompt_start"] = float(row["prompt_start"])
            row["prompt_end"] = float(row["prompt_end"])
            row["target_start"] = float(row["target_start"])
            row["target_end"] = float(row["target_end"])
            row["candidate_id"] = (row["speaker"], row["prompt_vid"], row["prompt_seg"],
                                   row["prompt_start"], row["prompt_end"])
            rows.append(row)
    return rows


def read_slice(path, start, end):
    """Read a [start, end]-second slice as float32 mono at SR."""
    info = sf.info(path)
    assert info.samplerate == SR and info.channels == 1, \
        f"unexpected wav format {info.samplerate}Hz {info.channels}ch  {path}"
    s = int(round(start * SR))
    e = int(round(end * SR))
    e = min(e, info.frames)
    s = max(0, min(s, e))
    wav, _ = sf.read(path, start=s, stop=e, dtype="float32", always_2d=False)
    return wav


def load_msr_utmos_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6: continue
            spk, vid, seg, s, e, sc = parts
            cache[(spk, vid, seg, float(s), float(e))] = float(sc)
    return cache


def append_msr_utmos_cache(path, key, score):
    with open(path, "a") as f:
        spk, vid, seg, s, e = key
        f.write(f"{spk}\t{vid}\t{seg}\t{s:.3f}\t{e:.3f}\t{score:.4f}\n")


def compute_msr_utmos(rows, device, cache_path):
    """Compute MSR-UTMOS for each unique prompt candidate. Writes to cache_path
    incrementally so partial runs are resumable. Mutates rows with `msr_utmos`."""
    cache = load_msr_utmos_cache(cache_path)
    unique = collections.OrderedDict()
    for r in rows:
        unique.setdefault(r["candidate_id"], r)
    pending = [(k, v) for k, v in unique.items() if k not in cache]
    print(f"MSR-UTMOS: {len(unique):,} unique candidates, {len(cache):,} cached, "
          f"{len(pending):,} to score", file=sys.stderr)

    if pending:
        model = load_model(DEFAULT_CKPT, device)
        with torch.no_grad():
            for key, row in tqdm(pending, desc="MSR-UTMOS"):
                p = wav_path(row["speaker"], row["prompt_vid"], row["prompt_seg"])
                try:
                    wav = read_slice(p, row["prompt_start"], row["prompt_end"])
                    if wav.size < SR:  # <1s -- skip
                        score = float("nan")
                    else:
                        t = torch.from_numpy(wav).unsqueeze(0)  # (1, time) mono
                        score = float(score_wave(model, t, SR, device))
                except Exception as e:
                    print(f"[msr-utmos] {p}: {e}", file=sys.stderr)
                    score = float("nan")
                cache[key] = score
                append_msr_utmos_cache(cache_path, key, score)

    for r in rows:
        r["msr_utmos"] = cache.get(r["candidate_id"], float("nan"))


def pick_per_lang(rows, n_bins, msr_utmos_min, msr_utmos_max, target_total, seed):
    """Round-robin MSR-UTMOS-stratified selection, one prompt per speaker, up to
    `target_total` prompts per language. Returns {lang: [(bin, row), ...]}.

    TTS error metrics (WER especially) are tail-driven — the mean is dominated by
    rare catastrophic generations — so per-language N matters for a stable
    estimate; we take as many prompts as the target/data allow. Round-robin across
    bins keeps the prompt quality distribution as even as the data supports while
    still filling the quota from the fuller (typically lower-MSR-UTMOS) bins once the
    sparse high-MSR-UTMOS bins are exhausted."""
    bin_w = (msr_utmos_max - msr_utmos_min) / n_bins
    rnd = random.Random(seed)
    by_lang = collections.defaultdict(list)
    for r in rows:
        if r["msr_utmos"] != r["msr_utmos"]:  # NaN
            continue
        # prompt transcript (near-)equal to its target: copyable, drop before binning
        if near_duplicate(r["prompt_text"], r["target_text"]):
            continue
        by_lang[r["lang"]].append(r)

    chosen = {}
    for lang, cands in by_lang.items():
        rnd.shuffle(cands)
        bins = [[] for _ in range(n_bins)]
        for r in cands:
            b = int((r["msr_utmos"] - msr_utmos_min) / bin_w)
            if 0 <= b < n_bins:
                bins[b].append(r)

        used_speakers = set()
        sel = []
        ptr = [0] * n_bins
        active = True
        while len(sel) < target_total and active:
            active = False
            for bi in range(n_bins):
                if len(sel) >= target_total:
                    break
                # advance to the next unused-speaker candidate in this bin
                while ptr[bi] < len(bins[bi]):
                    r = bins[bi][ptr[bi]]
                    ptr[bi] += 1
                    if r["speaker"] in used_speakers:
                        continue
                    used_speakers.add(r["speaker"])
                    sel.append((bi, r))
                    active = True
                    break
        chosen[lang] = sel
        per_bin = collections.Counter(bi for bi, _ in sel)
        dist = " ".join(
            f"b{bi}({msr_utmos_min+bi*bin_w:.1f}-{msr_utmos_min+(bi+1)*bin_w:.1f})={per_bin.get(bi, 0)}"
            for bi in range(n_bins))
        print(f"  {lang}: {len(sel)}/{target_total} prompts  [{dist}]", file=sys.stderr)
    return chosen


def materialize(chosen, output_dir):
    for lang, sel in chosen.items():
        lang_dir = os.path.join(output_dir, lang)
        wav_dir = os.path.join(lang_dir, "waveform")
        shutil.rmtree(wav_dir, ignore_errors=True)   # drop stale wavs from prior runs
        os.makedirs(wav_dir, exist_ok=True)
        text_p, prompt_text_p, scp_p = (
            os.path.join(lang_dir, n) for n in ("text", "prompt_text", "prompt_wav.scp"))
        with open(text_p, "w") as ft, open(prompt_text_p, "w") as fp, \
             open(scp_p, "w") as fs:
            for k, (bi, r) in enumerate(sel, start=1):
                # uid scheme matches prepare_fleurs_r.py: id is uttid_N while the
                # reference waveform file is prompt_audio_N.wav (one scheme across sets).
                uid = f"uttid_{k}"
                wav_out = os.path.join(wav_dir, f"prompt_audio_{k}.wav")
                wav = read_slice(wav_path(r["speaker"], r["prompt_vid"], r["prompt_seg"]),
                                 r["prompt_start"], r["prompt_end"])
                sf.write(wav_out, wav, SR, subtype="PCM_16")
                ft.write(f"{uid} {r['target_text']}\n")
                fp.write(f"{uid} {r['prompt_text']}\n")
                # repo-relative, matching the FLEURS sets: absolute paths dangle
                # on any checkout at a different location (utils/scp.resolve_path
                # re-anchors legacy absolute entries at load time).
                fs.write(f"{uid} {os.path.relpath(wav_out)}\n")
        print(f"  {lang}: wrote {len(sel)} items to {lang_dir}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/in-the-wild/_stage_a/manifest.tsv")
    ap.add_argument("--output-dir", default="data/in-the-wild")
    ap.add_argument("--target-total", type=int, default=1500,
                    help="max prompts per language (round-robin across MSR-UTMOS bins)")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--msr-utmos-min", type=float, default=1.0)
    ap.add_argument("--msr-utmos-max", type=float, default=4.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "_msr_utmos_cache.tsv")

    rows = load_manifest(args.manifest)
    print(f"loaded manifest: {len(rows):,} rows", file=sys.stderr)

    compute_msr_utmos(rows, args.device, cache_path)

    chosen = pick_per_lang(
        rows,
        n_bins=args.n_bins,
        msr_utmos_min=args.msr_utmos_min,
        msr_utmos_max=args.msr_utmos_max,
        target_total=args.target_total,
        seed=args.seed,
    )

    materialize(chosen, args.output_dir)


if __name__ == "__main__":
    main()
