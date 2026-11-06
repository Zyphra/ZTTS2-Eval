"""Generate TTS audio with Kokoro-82M for a ZTTS2-Eval test set.

Kokoro (https://huggingface.co/hexgrad/Kokoro-82M) is a small (82 M) open-weights
TTS model. Unlike the zero-shot cloners in this benchmark (zonos2 / omni / echo),
Kokoro does NOT clone a reference speaker -- it synthesises with a fixed built-in
"preset" voice per language. So it ignores <data-dir>/prompt_wav.scp entirely, and
the speaker-similarity metric is not meaningful for it (the output voice is unrelated
to the prompt); WER and MSR-UTMOS still are. It runs in-process (no server), like
generate_echo.py -- a single device, sequential loop.

For each utterance in <data-dir>/text it synthesises with (--lang-code, --voice) and
writes <out-dir>/wavs/<uttid>.wav (24 kHz), ready for scripts/score/run_score.sh.

Run with the dedicated Kokoro venv (scripts/setup/setup_kokoro_env.sh):
    ~/.venvs/kokoro/bin/python scripts/gen/generate_kokoro.py \\
        --data-dir data/clean/en --out-dir results/kokoro/clean/en \\
        --lang-code a --voice af_heart
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, run_generate_loop, save_wav_atomic)
from utils.scp import load_scp  # noqa: E402

SR = 24000                       # Kokoro always outputs 24 kHz mono
REPO_ID = "hexgrad/Kokoro-82M"


def _to_np(audio):
    """Kokoro yields a torch.FloatTensor (or ndarray) per chunk -> 1-D float32."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with Kokoro-82M.")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir containing 'text' (prompt_wav is IGNORED -- "
                         "Kokoro uses a preset voice, it does not clone)")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--lang-code", required=True,
                    help="Kokoro lang_code letter: a=American en, b=British en, "
                         "e=es, f=fr, h=hi, i=it, j=ja, p=pt-br, z=zh")
    ap.add_argument("--voice", required=True,
                    help="Kokoro voice name, e.g. af_heart (see VOICES.md)")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking-rate multiplier")
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda if available else cpu)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--stratified-bins", type=int, default=0,
                    help="if >0, pick K evenly-spaced utts across the full set "
                         "(in-the-wild is sorted into UTMOS bins per lang)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    texts = load_scp(os.path.join(args.data_dir, "text"))
    # Kokoro is not a cloner -- keep EVERY utterance in `text` (do not intersect with
    # prompt_wav.scp), so the WER set matches the cloning models on identical text.
    uids = list(texts.keys())
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[kokoro] loading KPipeline(lang_code={args.lang_code!r}) on {device} "
          f"voice={args.voice} ...", flush=True)
    from kokoro import KPipeline  # heavy import -- after arg parsing
    pipeline = KPipeline(lang_code=args.lang_code, repo_id=REPO_ID, device=device)

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            # Kokoro splits long text into segments; concatenate their audio into one wav.
            chunks = [_to_np(audio) for _, _, audio in
                      pipeline(texts[uid], voice=args.voice, speed=args.speed)]
            wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            if wav.size == 0:
                return uid, "empty audio", 0.0
            dur = save_wav_atomic(wav, SR, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU engine -> sequential loop in this thread.
    run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
