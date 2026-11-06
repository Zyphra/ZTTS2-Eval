"""Generate TTS audio with Phonon (gradium/phonon-*) for a ZTTS2-Eval test set.

Runs Gradium's own Rust runtime (`ptts` on PyPI, the xn-ptts stack) on CPU with the
checkpoint's published q8 GGUF weights, the runtime the model card benchmarks and ships.
The runtime also spells out numbers, symbols and dates for `lang` before synthesis.

Phonon cannot clone a voice from a wav locally: the open checkpoints ship only the Mimi
decoder (cloning goes through Gradium's hosted API), so this is a PRESET-voice system. Each
utterance gets a catalog voice of its language (gradium/phonon-voices), picked by a
deterministic per-uid rotation (genclient.pick_voice), and the prompt wav is ignored. The
assignments are written to <out-dir>/voices.txt. Sampling is the runtime's default
(temperature 0.3) with its default seed passed on every call; reruns match closely but not
always bit for bit (CPU thread scheduling).

Synthesis is single-stream per model instance, so the leaf is spread over --workers
processes, each with its own model and --threads CPU threads.

    ~/.venvs/phonon/bin/python scripts/generate_phonon.py --language en \\
        --data-dir data/clean/en --out-dir results/phonon-40m-multilingual/clean/en
"""
import argparse
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, pick_voice, run_generate_loop,
    save_wav_atomic)

SEED = 4242424242424242  # ptts.TTS default seed
_W = None


def _init_worker(model_dir, catalog_dir, lang, quant, threads):
    global _W
    import ptts
    ptts.set_num_threads(threads)
    _W = ptts.TTS(model_dir, lang=lang, quant=quant, voice_catalog=catalog_dir)


def _synth(job):
    text, voice, out_path = job
    try:
        wav = _W.synth(text, voice=voice, seed=SEED)
        if wav.size == 0:
            return "empty audio", 0.0
        return "ok", save_wav_atomic(wav, _W.sample_rate, out_path)
    except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
        return format_error(e), 0.0


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with Phonon (ptts runtime).")
    ap.add_argument("--model-dir", required=True, help="local Phonon checkpoint directory")
    ap.add_argument("--voice-catalog", required=True,
                    help="local gradium/phonon-voices snapshot (voices.jsonl + voices/)")
    ap.add_argument("--language", required=True, help="benchmark language of the test set")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="wavs are written to <out-dir>/wavs/")
    ap.add_argument("--quant", default="q8", help="weight format (model card default q8)")
    ap.add_argument("--workers", type=int, default=16, help="model processes")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per process")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    import ptts
    pool_names = sorted(v["name"] for v in ptts.catalog_voices(args.voice_catalog)
                        if v["language"] == args.language)
    if not pool_names:
        sys.exit(f"error: the voice catalog has no '{args.language}' voices")

    texts, _, _, uids = load_test_set(args.data_dir)
    uids = apply_limit(uids, args.limit)
    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    voices = {u: pick_voice(u, pool_names) for u in uids}
    with open(os.path.join(args.out_dir, "voices.txt"), "w", encoding="utf-8") as f:
        for u in uids:
            f.write(f"{u}\t{voices[u]}\t{args.language}\n")
    print(f"[phonon] {args.data_dir} -> {wav_dir} | {len(uids)} utterances, "
          f"{len(pool_names)} '{args.language}' voices, {args.workers} x {args.threads} threads",
          flush=True)

    pool = multiprocessing.get_context("spawn").Pool(
        args.workers, initializer=_init_worker,
        initargs=(args.model_dir, args.voice_catalog, args.language, args.quant, args.threads))

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        status, dur = pool.apply(_synth, ((texts[uid], voices[uid], out_path),))
        return uid, status, dur

    # Threads only dispatch; the synthesis runs in the worker processes.
    with pool:
        run_generate_loop(uids, generate, workers=args.workers, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
