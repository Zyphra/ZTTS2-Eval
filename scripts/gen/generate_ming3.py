"""Generate TTS audio with Ming-3.0-tts (native upstream) for a ZTTS2-Eval test set.

Runs the official inclusionAI/Ming-3.0-tts code IN-PROCESS from the checkout on
PYTHONPATH (scripts/gen/run_ming3.sh): `ming_tts.MingTTS` for the 0.5b dense checkpoint,
`ming_tiny.MingTinyTTS` for the tiny MoE. Zero-shot cloning conditions on the prompt wav
and its transcript (both checkpoints require the transcript for cloning), for the seven
languages the models support: zh en (0.5b TTS / tiny `seed`) and de es fr ja ko (0.5b
TTS with language_id / tiny `multilingual`).

Upstream reference settings: greedy text decoding (temperature 0), CFG 2.0, flow sigma
0.25 (tiny: 10 flow steps), trailing-silence trim, and raw_text -- the text is sent as
given, in one piece, "which is how the published benchmarks were run" (ming_tts.py).
The seed is fixed per utterance (0.5b: infer.py's 1895, reset before every utterance;
tiny: its formal seed 1234, which generate() applies per call), so a resumed or partial
leaf reproduces the same audio.

--trim-prompt (the *-trim systems) trims leading/trailing silence off the prompt wav
(librosa.effects.trim, top_db 35 -- the threshold the MPM and speaking-rate metrics use)
before Ming encodes it. Ming continues from the tail of the prompt ("trailing silence
would be what the model continues from", ming_tts.py), and the FLEURS-R prompts end in
~1-2 s of silence, on which it mostly emits ~0.3 s of near-silence. The untrimmed
systems keep the prompts as given.

    ~/.venvs/ming3/bin/python scripts/gen/generate_ming3.py --variant 0.5b --language en \\
        --data-dir data/clean/en --out-dir results/ming-3.0-tts-0.5b/clean/en
"""
import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, run_generate_loop,
    save_wav_atomic)

MODEL_IDS = {"0.5b": "inclusionAI/Ming-3.0-tts-0.5b", "tiny": "inclusionAI/Ming-3.0-tts-tiny"}
LANGS = {"zh", "en", "de", "es", "fr", "ja", "ko"}
TINY_TASK = {"zh": "seed", "en": "seed"}  # every other supported language: multilingual
SEED_05B = 1895  # infer.py --seed default
TRIM_TOP_DB = 35.0  # scripts/prosody/score_mpm_shuffle_floor.py and the speaking-rate metric


def trimmed_copy(path, out_dir):
    """The prompt wav with leading/trailing silence removed, written under out_dir."""
    import librosa
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    out = os.path.join(out_dir, Path(path).stem + ".trim.wav")
    sf.write(out, trimmed if trimmed.size else audio, sr, subtype="FLOAT")
    return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with Ming-3.0-tts (native).")
    ap.add_argument("--variant", required=True, choices=sorted(MODEL_IDS))
    ap.add_argument("--language", required=True, choices=sorted(LANGS),
                    help="benchmark language of the test set")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_wav.scp' and 'prompt_text'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--model-dir", default="", help="HF id or path (default: per variant)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--trim-prompt", action="store_true",
                    help="trim the prompt wav's edge silence (top_db %g) before encoding"
                         % TRIM_TOP_DB)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    uids = apply_limit(uids, args.limit)
    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    model_dir = args.model_dir or MODEL_IDS[args.variant]

    print(f"[ming3] loading {model_dir} ({args.variant}) ...", flush=True)
    if args.variant == "0.5b":
        from ming_tts import MingTTS  # noqa: E402
        # raw_text never runs the text frontend, so skip loading its normalizer.
        tts = MingTTS(model_dir, device="cuda:0", text_normalization=False)
        sr = int(tts.sample_rate)

        def synth(uid, prompt):
            seed_everything(SEED_05B)
            return tts.synthesize(task="TTS", text=texts[uid], language_id=args.language,
                                  prompt_wav_path=prompt, prompt_text=prompt_texts[uid],
                                  use_spk_emb=True, raw_text=True)
    else:
        from ming_tiny import MingTinyTTS  # noqa: E402
        tts = MingTinyTTS.from_pretrained(model_dir, device="cuda:0", text_normalization=False)
        sr = int(tts.sample_rate)
        task = TINY_TASK.get(args.language, "multilingual")

        def synth(uid, prompt):
            return tts.generate(task=task, text=texts[uid], prompt_wav=prompt,
                                prompt_text=prompt_texts[uid], target_language=args.language,
                                raw_text=True).waveform

    tmp = tempfile.TemporaryDirectory(prefix="ming3-prompts-")

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            prompt = (trimmed_copy(prompts[uid], tmp.name) if args.trim_prompt
                      else prompts[uid])
            with torch.no_grad():
                wav = synth(uid, prompt)
            if args.trim_prompt:
                os.remove(prompt)
            wav = (wav.detach().float().cpu().numpy() if isinstance(wav, torch.Tensor)
                   else np.asarray(wav, dtype=np.float32)).reshape(-1)
            if wav.size == 0:
                return uid, "empty audio", 0.0
            return uid, "ok", save_wav_atomic(wav, sr, out_path)
        except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU model -> sequential loop in this thread.
    with tmp:
        run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
