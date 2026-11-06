"""Generate TTS audio with NeuTTS-2E (Neuphonic) for a ZTTS2-Eval test set.

NeuTTS-2E (neuphonic/neutts-2e) is a small (~125M) on-device English emotional TTS
model. It is a PRESET-voice model -- 4 fixed speakers (emily, paul, sophie, steven)
x 7 emotions -- NOT a zero-shot cloner, so it ignores the reference/prompt wav and
speaker-similarity is not meaningful (WER and MSR-UTMOS are). English only. Runs
in-process (no server), like generate_kokoro.py / generate_echo.py.

    ~/.venvs/neutts/bin/python scripts/gen/generate_neutts.py \\
        --data-dir data/clean/en --out-dir results/neutts-2e/clean/en
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, pick_voice, run_generate_loop,
    save_wav_atomic, write_gen_meta_once)

SR = 24000
SPEAKERS = ["emily", "paul", "sophie", "steven"]


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with NeuTTS-2E.")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text' (prompt_wav is IGNORED -- preset voice)")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--speaker", default="rotate",
                    help="fixed speaker (emily|paul|sophie|steven) or 'rotate' "
                         "(deterministic per-utterance rotation across the 4 speakers)")
    ap.add_argument("--emotion", default="neutral",
                    help="emotion (neutral|happy|sad|angry|fearful|disgusted|surprised)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--stratified-bins", type=int, default=0,
                    help="if >0, pick K evenly-spaced utts across the full set")
    ap.add_argument("--seed", type=int, default=0, help="NeuTTS2E sampling seed")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    # preset voice -> every utterance in `text`
    texts, _, _, uids = load_test_set(args.data_dir, need_prompt_wav=False,
                                      need_prompt_text=False)
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    def speaker_for(uid):
        # uid-hash rotation (not list index) so a resume with different
        # --limit/--stratified-bins never reassigns voices mid-set. NOTE: this
        # reassigns voices once vs runs generated before the hash switch.
        if args.speaker != "rotate":
            return args.speaker
        return pick_voice(uid, SPEAKERS)

    # voice manifest -- deterministic, so write it upfront (recoverable later).
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            f.write(f"{uid}\t{speaker_for(uid)}\n")
    print(f"[neutts-2e] voice assignments -> {vpath}", flush=True)

    # NeuTTS2E defaults both the backbone and the NeuCodec to CPU -- put them on GPU.
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    print(f"[neutts-2e] loading NeuTTS2E on {dev} ...", flush=True)
    from neutts import NeuTTS2E  # noqa: E402  -- downloads the model on first run
    tts = NeuTTS2E(backbone_device=dev, codec_device=dev, seed=args.seed)

    write_gen_meta_once(args.out_dir, args.overwrite, model="neuphonic/neutts-2e",
                        seed=args.seed, speaker=args.speaker, emotion=args.emotion)

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        try:
            wav = tts.infer(texts[uid], speaker=speaker_for(uid), emotion=args.emotion)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if wav.size == 0:
                return uid, "empty audio", 0.0
            dur = save_wav_atomic(wav, SR, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU engine -> sequential loop in this thread.
    run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir,
                      overwrite=args.overwrite)


if __name__ == "__main__":
    main()
