"""Generate zero-shot *cloned* TTS audio via ElevenLabs' Instant Voice Cloning.

Unlike generate_elevenlabs.py (which rotates preset voices and measures the
model's intrinsic quality), this path clones each utterance's prompt speaker so
the output is comparable to the cloning engines in this harness (Zonos2, Omni,
fal): it yields a real zero-shot speaker-similarity number for ElevenLabs.

Per utterance it: (1) creates an Instant Voice Clone (IVC) from the prompt_wav,
(2) synthesizes the target text with that temp voice + model, (3) decodes to a
PCM_16 WAV in the scripts/score/run_score.sh layout, then (4) DELETES the temp
voice so the account's custom-voice slots are freed for the next utterance.

Synthesis goes through generate_elevenlabs.synth (eleven_v4 via the
Text-to-Dialogue API, language_code from the --data-dir basename, nb->no,
tl->fil). The benchmark prompts are short (~3-20 s) vs ElevenLabs' recommended 1-2 min, so
clone fidelity is on the low end -- a property of the prompts, not the method.

Requires an ElevenLabs API key scoped with text-to-speech, voices read/write
(instant voice cloning) and user-read (a plain TTS key returns 401
missing_permissions on voices.ivc.create):
    uv sync --extra elevenlabs
    export ELEVENLABS_API_KEY=...

The startup/final purge deletes EVERY live ztts_clone_* voice on the account:
run only one instance per account at a time.

Usage:
    python scripts/gen/generate_elevenlabs_clone.py --model v4 \\
        --data-dir data/clean/en \\
        --out-dir results/elevenlabs_v4_clone/clean/en
"""
import argparse
import os
import sys
import threading

from pathlib import Path

# Reuse the preset-voice script's helpers (same dir) -- DRY, single source of truth.
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from scripts.gen import generate_elevenlabs as ge  # noqa: E402
from utils.genclient import (  # noqa: E402
    apply_limit, benchmark_lang, format_error, load_test_set, run_generate_loop,
    write_gen_meta_once)

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None

VOICE_PREFIX = "ztts_clone_"  # temp-voice name prefix, so leftovers are findable


def create_ivc(client, name, wav_path, retries, base_delay):
    """Create an Instant Voice Clone from one reference wav; return its voice_id."""
    def _call():
        with open(wav_path, "rb") as f:
            data = f.read()
        r = client.voices.ivc.create(
            name=name,
            # tuple form (filename, bytes, content_type) avoids dangling handles
            files=[(os.path.basename(wav_path), data, "audio/wav")],
            # samples are already clean 16k mono; isolation on clean audio hurts
            remove_background_noise=False,
        )
        return r.voice_id
    return ge.with_retry(_call, retries, base_delay)


def delete_voice(client, voice_id, retries=3, base_delay=1.0):
    """Best-effort delete of a temp voice; retried so slots don't leak."""
    try:
        ge.with_retry(lambda: client.voices.delete(voice_id), retries, base_delay)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[cleanup] failed to delete voice {voice_id}: {e}")
        return False


def purge_stale(client):
    """Best-effort: delete any leftover ztts_clone_ voices from a prior crashed run.

    Needs voices-read scope; silently skipped if the key can't list voices."""
    try:
        resp = client.voices.get_all()
        voices = getattr(resp, "voices", resp) or []
        stale = [v for v in voices
                 if (getattr(v, "name", "") or "").startswith(VOICE_PREFIX)]
        if stale:
            print(f"[purge] removing {len(stale)} stale {VOICE_PREFIX}* voices")
            for v in stale:
                delete_voice(client, v.voice_id)
    except Exception as e:  # noqa: BLE001
        print(f"[purge] skipped (cannot list voices): {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate zero-shot cloned audio via ElevenLabs Instant Voice Cloning.")
    ap.add_argument("--model", default=ge.DEFAULT_MODEL, choices=sorted(ge.MODEL_IDS),
                    help=f"ElevenLabs model (default: {ge.DEFAULT_MODEL})")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent (create+synth+delete) chains. Bounded by the "
                         "account's custom-voice slots and concurrency limit; default 4")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=4, help="attempts per API call")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--seed", type=int, default=0,
                    help="generation seed (>=0, provider best-effort; default 0); "
                         "negative = not sent (stochastic)")
    ap.add_argument("--keep-voices", action="store_true",
                    help="do NOT delete temp voices after synth (debug; risks hitting "
                         "the custom-voice cap)")
    ap.add_argument("--no-purge", action="store_true",
                    help="skip the startup purge of leftover ztts_clone_* voices")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    if ElevenLabs is None:
        sys.exit("error: elevenlabs is not installed -- run `uv sync --extra elevenlabs`")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("error: ELEVENLABS_API_KEY environment variable is not set")

    model_id = ge.MODEL_IDS[args.model]
    client = ElevenLabs(api_key=api_key, timeout=args.timeout)

    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    uids = apply_limit(uids, args.limit)
    lang = benchmark_lang(args.data_dir)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[elevenlabs-clone:{args.model}] {args.data_dir} -> {wav_dir} | "
          f"{len(uids)} utterances | IVC per-utt, workers={args.workers}")

    # Kept from the FIRST run of a resumed dir unless --overwrite regenerates.
    write_gen_meta_once(args.out_dir, args.overwrite, provider="elevenlabs", model=model_id,
                        endpoint=ge.ENDPOINT, voice_mode="clone", language=lang,
                        language_sent=ge.eleven_lang(lang),
                        output_format=ge.OUTPUT_FORMAT,
                        seed=args.seed if args.seed >= 0 else None,
                        ivc={"remove_background_noise": False,
                             "deleted_after_use": not args.keep_voices})

    if not args.no_purge:
        purge_stale(client)

    # A monotonic counter for unique temp-voice names (avoid collisions across threads).
    lock = threading.Lock()
    counter = {"n": 0}

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        voice_id = None
        try:
            wav_path = prompts[uid]
            if not os.path.exists(wav_path):
                return uid, f"error: prompt wav missing ({wav_path})", 0.0
            with lock:
                counter["n"] += 1
                name = f"{VOICE_PREFIX}{uid}_{counter['n']}"
            voice_id = create_ivc(client, name, wav_path,
                                  args.retries, args.retry_base_delay)
            raw = ge.synth(client, model_id, texts[uid], voice_id, lang,
                           args.retries, args.retry_base_delay, seed=args.seed)
            dur = ge.bytes_to_wav(raw, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0
        finally:
            if voice_id and not args.keep_voices:
                delete_voice(client, voice_id)

    run_generate_loop(uids, generate, args.workers,
                      desc=f"elevenlabs-clone:{args.model}", wav_dir=wav_dir,
                      overwrite=args.overwrite)
    if not args.keep_voices:
        purge_stale(client)  # final sweep in case any delete was missed


if __name__ == "__main__":
    main()
