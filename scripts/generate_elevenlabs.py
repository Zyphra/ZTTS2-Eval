"""Generate TTS audio via ElevenLabs' official API for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, synthesizes the target text with an
ElevenLabs model (via the official `elevenlabs` SDK) and writes
<out-dir>/wavs/<uttid>.wav -- the layout run_infer_zonos2_eval.sh expects.

This path does NOT clone the prompt speaker: like the Gemini engine in this
harness, it uses ElevenLabs' built-in preset voices and picks one per utterance
deterministically (so re-runs are reproducible), recorded in voices.txt. It
measures the model's intrinsic quality, not zero-shot speaker similarity.

Models (--model):
    v3                eleven_v3                most expressive (default)
    multilingual-v2   eleven_multilingual_v2   prior-gen, for comparison

Requires the optional 'elevenlabs' extra and an ElevenLabs API key:
    uv sync --extra elevenlabs
    export ELEVENLABS_API_KEY=...

Usage:
    python scripts/generate_elevenlabs.py --model v3 \\
        --data-dir data/zero_shot/en --out-dir results/elevenlabs_v3/zero_shot/en \\
        --language en
"""
import argparse
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    bytes_to_wav, detect_lang, format_error, load_test_set, normalize_lang,
    pick_voice, run_generate_loop, with_retry)

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None


# --- model ids ---------------------------------------------------------------
# Keys are version-tagged so each model's output lands in a distinct
# results/elevenlabs_<key>/ tree (never conflate two models in one dir).
DEFAULT_MODEL = "v3"
MODEL_IDS = {
    "v3": "eleven_v3",
    "multilingual-v2": "eleven_multilingual_v2",
}

# --- voice pool (rotated per utterance) --------------------------------------
# The official premade ElevenLabs voices (name -> public voice_id).
# Voices are language-agnostic; the model infers / is told the spoken language.
ELEVEN_VOICES = {
    "Aria": "9BWtsMINqrJLrRacOk9x",
    "Roger": "CwhRBWXzGAHq8TQ4Fs17",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Laura": "FGY2WhTYpPnrIDTdsKH5",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "River": "SAz9YHcvj6GT2YYXdXww",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Alice": "Xb7hH8MSUJpSbSDYk0k2",
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "Will": "bIHbv24MWmeRgasZH58o",
    "Jessica": "cgSgspJ2msm6clMCkdW9",
    "Eric": "cjVigY5qzO86Huf0OWal",
    "Chris": "iP95p4xoKVk53GoZ742B",
    "Brian": "nPczCjzI2devNBz1zQrb",
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Lily": "pFZP5JQG7iQjIQuC4Bku",
    "Bill": "pqHfZKP75CvOlQylNhV4",
}
VOICE_NAMES = list(ELEVEN_VOICES)

# eleven_v3 accepts an optional ISO 639-1 language_code; otherwise auto-detected.
# Keys are the --language values the wrapper passes (identical to the ISO codes).
ELEVEN_LANG = {
    "en": "en", "zh": "zh", "ja": "ja", "ko": "ko",
    "de": "de", "es": "es", "fr": "fr", "it": "it", "ru": "ru",
    # voxblink adds these (eleven_v3-supported ISO 639-1 codes)
    "ar": "ar", "hi": "hi", "id": "id", "pl": "pl", "pt": "pt",
    "th": "th", "tr": "tr",
    # note: tl (Tagalog) is NOT in eleven_v3's language_code enum -- it is
    # intentionally omitted so it auto-detects (synth() also falls back on the
    # 'unsupported_language' 400 as a safety net for any other rejected code).
}

# Request MP3 (always available); decoded to PCM_16 WAV for the scorers.
OUTPUT_FORMAT = "mp3_44100_128"


def _is_unsupported_lang(e):
    """True if a convert() error is a 'model does not support language_code X' 400."""
    s = str(e)
    return "unsupported_language" in s or "does not support language_code" in s


def synth(client, model_id, text, voice_id, lang, retries, base_delay, seed=-1):
    """Run one ElevenLabs synthesis and return the result audio bytes, retried.

    If the model rejects the language_code (a hard 400), fall back to the same
    request without it (model auto-detects) so one unsupported voxblink language
    never zeroes out a whole split. A non-negative seed is passed through for
    reproducible / multi-seed (DS-WED) generation."""
    code = ELEVEN_LANG.get(lang)

    def _call(send_lang):
        kwargs = dict(voice_id=voice_id, model_id=model_id, text=text,
                      output_format=OUTPUT_FORMAT)
        if send_lang:
            kwargs["language_code"] = code
        if seed >= 0:
            kwargs["seed"] = seed
        return b"".join(client.text_to_speech.convert(**kwargs))

    def _attempt():
        try:
            return _call(bool(code))
        except Exception as e:  # noqa: BLE001
            if code and _is_unsupported_lang(e):
                return _call(False)  # permanent 400 on the code; retry won't help
            raise

    return with_retry(_attempt, retries, base_delay)


def main():
    ap = argparse.ArgumentParser(
        description="Generate TTS audio via the official ElevenLabs API for a ZTTS2-Eval test set.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_IDS),
                    help=f"ElevenLabs model (default: {DEFAULT_MODEL})")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--language", default="",
                    help="target language code (en/zh/ja/ko/de/es/fr/it/ru); sent as "
                         "language_code when supported, else auto-detected per utterance")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent elevenlabs requests (free/Starter tiers cap at 2; "
                         "raise to 5 on Creator+; default 2)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per elevenlabs call")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--seed", type=int, default=-1,
                    help="eleven_v3 generation seed (>=0); omit for stochastic. "
                         "Use distinct seeds per dir for DS-WED multi-seed diversity.")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    if ElevenLabs is None:
        sys.exit("error: elevenlabs is not installed -- run `uv sync --extra elevenlabs`")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("error: ELEVENLABS_API_KEY environment variable is not set")

    model_id = MODEL_IDS[args.model]
    client = ElevenLabs(api_key=api_key, timeout=args.timeout)

    # Intersect so the uttid set matches the other engines (each test-set dir
    # has all three files); ElevenLabs itself only needs the target text.
    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    if args.limit > 0:
        uids = uids[:args.limit]

    forced_lang = normalize_lang(args.language)
    if args.language and forced_lang not in ELEVEN_LANG:
        print(f"[warn] --language '{args.language}' -> '{forced_lang}' is not a "
              f"supported language code; falling back to auto-detect")
        forced_lang = ""

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[elevenlabs:{args.model}] {args.data_dir} -> {wav_dir} | {len(uids)} utterances "
          f"(text={len(texts)}, prompt_wav={len(prompts)}, prompt_text={len(prompt_texts)})")

    # voice manifest -- deterministic, so write it upfront
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            lg = forced_lang or detect_lang(texts[uid])
            f.write(f"{uid}\t{pick_voice(uid, VOICE_NAMES)}\t{lg}\n")
    print(f"[elevenlabs:{args.model}] voice assignments -> {vpath}")

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            target_text = texts[uid]
            lang = forced_lang or detect_lang(target_text)
            voice_id = ELEVEN_VOICES[pick_voice(uid, VOICE_NAMES)]
            raw = synth(client, model_id, target_text, voice_id, lang,
                        args.retries, args.retry_base_delay, seed=args.seed)
            dur = bytes_to_wav(raw, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers,
                      desc=f"elevenlabs:{args.model}", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
