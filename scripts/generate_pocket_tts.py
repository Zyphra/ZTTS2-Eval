"""Generate TTS audio with Kyutai Pocket TTS (kyutai/pocket-tts) for a ZTTS2-Eval test set.

Runs the official `pocket-tts` Python package on CPU (the runtime Kyutai ships and
benchmarks: batch size 1, small model, no GPU speedup). Zero-shot cloning conditions on the
prompt wav alone (`get_state_for_audio_prompt`, no transcript). Each language has its own
checkpoint; this uses the package's default per language (`english` = english_2026-10,
`french`, `german`, `portuguese`, `italian`, `spanish`; not the slower *_24l variants).
Sampling is the package default (temperature 0.3, EOS threshold -4) with a fixed torch seed
per utterance.

Generation is single-stream per model instance, so the leaf is spread over --workers
processes, each with its own model and --threads CPU threads.

    ~/.venvs/pocket-tts/bin/python scripts/generate_pocket_tts.py --language en \\
        --data-dir data/zero_shot/en --out-dir results/pocket-tts/zero_shot/en
"""
import argparse
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, run_generate_loop, save_wav_atomic)

LANGUAGE_MODEL = {"en": "english", "fr": "french", "de": "german", "pt": "portuguese",
                  "it": "italian", "es": "spanish"}
SEED = 42
_W = None


def _init_worker(language, threads):
    global _W
    import torch
    from pocket_tts import TTSModel
    torch.set_num_threads(threads)
    _W = TTSModel.load_model(language=language)


def _synth(job):
    text, prompt, out_path = job
    import torch
    try:
        with torch.no_grad():
            state = _W.get_state_for_audio_prompt(Path(prompt))
            torch.manual_seed(SEED)
            # copy_state must stay True (the default): with copy_state=False every text
            # chunk after the first is lost (a two-sentence utterance comes back as its
            # first sentence only).
            audio = _W.generate_audio(state, text)
        wav = audio.float().mean(dim=0).numpy() if audio.dim() == 2 else audio.float().numpy()
        if wav.size == 0:
            return "empty audio", 0.0
        return "ok", save_wav_atomic(wav, _W.sample_rate, out_path)
    except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
        return format_error(e), 0.0


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with Pocket TTS (CPU).")
    ap.add_argument("--language", required=True, choices=sorted(LANGUAGE_MODEL),
                    help="benchmark language of the test set")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="wavs are written to <out-dir>/wavs/")
    ap.add_argument("--workers", type=int, default=16, help="model processes")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per process")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    texts, prompts, _, uids = load_test_set(args.data_dir, need_prompt_text=False)
    uids = apply_limit(uids, args.limit)
    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    language = LANGUAGE_MODEL[args.language]
    print(f"[pocket-tts] {args.data_dir} -> {wav_dir} | {len(uids)} utterances, model "
          f"'{language}', {args.workers} x {args.threads} threads", flush=True)

    pool = multiprocessing.get_context("spawn").Pool(
        args.workers, initializer=_init_worker, initargs=(language, args.threads))

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        status, dur = pool.apply(_synth, ((texts[uid], prompts[uid], out_path),))
        return uid, status, dur

    # Threads only dispatch; the synthesis runs in the worker processes.
    with pool:
        run_generate_loop(uids, generate, workers=args.workers, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
