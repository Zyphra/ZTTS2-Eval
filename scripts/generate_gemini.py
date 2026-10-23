"""Generate TTS audio via Google's native Gemini TTS for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, synthesizes the target text with a
Gemini TTS model (via the official google-genai SDK) and writes
<out-dir>/wavs/<uttid>.wav -- the layout run_infer_zonos2_eval.sh expects.

Gemini TTS has no zero-shot voice cloning: it uses ~30 prebuilt voices and
infers the spoken language from the text itself (no language_code is sent). Like
the other preset-voice engines in this harness, a voice is picked per utterance
deterministically (so re-runs are reproducible) and recorded in voices.txt.

Models (--model):
    3.1-flash  gemini-3.1-flash-tts-preview   latest, faster / cheaper (default)
    2.5-flash  gemini-2.5-flash-preview-tts   prior-gen flash
    2.5-pro    gemini-2.5-pro-preview-tts     prior-gen, higher quality

Requires the optional 'gemini' extra and a Google API key:
    uv sync --extra gemini
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

Usage:
    python scripts/generate_gemini.py --model 3.1-flash \\
        --data-dir data/zero_shot/en --out-dir results/gemini_3.1-flash/zero_shot/en \\
        --language en
"""
import argparse
import os
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    detect_lang, format_error, load_test_set, normalize_lang, parse_l16_rate,
    pick_voice, run_generate_loop, save_wav_atomic, with_retry)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# --- model ids ---------------------------------------------------------------
# Keys are version-tagged so each model's output lands in a distinct
# results/gemini_<key>/ tree (never conflate two models in one dir).
DEFAULT_MODEL = "3.1-flash"
MODEL_IDS = {
    "3.1-flash": "gemini-3.1-flash-tts-preview",
    "2.5-flash": "gemini-2.5-flash-preview-tts",
    "2.5-pro": "gemini-2.5-pro-preview-tts",
}

# --- voice pool (rotated per utterance) --------------------------------------
# The 30 prebuilt Gemini voices. They are language-agnostic; Gemini detects the
# spoken language from the text, so the voice pick does not depend on --language.
GEMINI_VOICES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe",
    "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir",
    "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Pulcherrima",
    "Puck", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat",
    "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi",
]

# Native Gemini TTS returns raw PCM: 24 kHz, 16-bit signed little-endian, mono.
GEMINI_SAMPLE_RATE = 24000


def extract_pcm(resp):
    """Pull the raw inline PCM bytes + mime type out of a generate_content response.

    Returns (data, mime_type); mime_type may be None/'' on SDK paths that omit
    it (the caller then assumes the documented 24 kHz L16 default)."""
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or [] if content else []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline else None
            if data:
                return data, getattr(inline, "mime_type", None)
    # No audio: usually a safety block or empty generation. Surface why if we can.
    reason = ""
    if candidates:
        reason = f" finish_reason={getattr(candidates[0], 'finish_reason', None)}"
    fb = getattr(resp, "prompt_feedback", None)
    if fb is not None:
        reason += f" prompt_feedback={fb}"
    raise RuntimeError(f"no audio in gemini response{reason}")


def pcm_to_wav(raw, mime_type, out_path):
    """Wrap raw 16-bit LE mono PCM bytes in a mono PCM_16 WAV. Returns duration (s).

    The sample rate comes from the response's mime type (e.g.
    'audio/L16;rate=24000'); a non-L16 mime raises instead of silently writing
    valid-looking audio at the wrong speed."""
    sr = parse_l16_rate(mime_type, GEMINI_SAMPLE_RATE)
    data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    return save_wav_atomic(data, sr, out_path)


def synth(client, model_id, text, voice, style_prompt, retries, base_delay):
    """Run one Gemini TTS synthesis; returns (pcm_bytes, mime_type), retried."""
    contents = f"{style_prompt.strip()}: {text}" if style_prompt.strip() else text
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )

    def _call():
        resp = client.models.generate_content(
            model=model_id, contents=contents, config=config)
        return extract_pcm(resp)

    return with_retry(_call, retries, base_delay)


def main():
    ap = argparse.ArgumentParser(
        description="Generate TTS audio via native Gemini TTS for a ZTTS2-Eval test set.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_IDS),
                    help=f"Gemini TTS model (default: {DEFAULT_MODEL})")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--language", default="",
                    help="target language label for voices.txt only (en/zh/ja/ko); "
                         "not sent to the API -- Gemini auto-detects language")
    ap.add_argument("--style-prompt", default="",
                    help="optional natural-language style prefix (e.g. 'Say cheerfully'); "
                         "default empty = raw text, comparable with the other engines")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--workers", type=int, default=4, help="concurrent gemini requests")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per gemini call")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    if genai is None:
        sys.exit("error: google-genai is not installed -- run `uv sync --extra gemini`")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("error: GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set")

    model_id = MODEL_IDS[args.model]
    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=int(args.timeout * 1000)))

    # Intersect so the uttid set matches the other engines (each test-set dir
    # has all three files); Gemini itself only needs the target text.
    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    if args.limit > 0:
        uids = uids[:args.limit]

    # --language only labels voices.txt (Gemini auto-detects the spoken language),
    # so accept the full zero_shot + voxblink language set; unknown values just
    # fall back to per-utterance script detection for the label.
    KNOWN_LANGS = ("en", "zh", "ja", "ko", "de", "es", "fr", "it", "ru")
    forced_lang = normalize_lang(args.language)
    if args.language and forced_lang not in KNOWN_LANGS:
        print(f"[warn] --language '{args.language}' -> '{forced_lang}' is not in "
              f"{'/'.join(KNOWN_LANGS)}; falling back to auto-detect for voices.txt")
        forced_lang = ""

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[gemini:{args.model}] {args.data_dir} -> {wav_dir} | {len(uids)} utterances "
          f"(text={len(texts)}, prompt_wav={len(prompts)}, prompt_text={len(prompt_texts)})")

    # voice manifest -- deterministic, so write it upfront
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            lg = forced_lang or detect_lang(texts[uid])
            f.write(f"{uid}\t{pick_voice(uid, GEMINI_VOICES)}\t{lg}\n")
    print(f"[gemini:{args.model}] voice assignments -> {vpath}")

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            raw, mime = synth(client, model_id, texts[uid],
                              pick_voice(uid, GEMINI_VOICES),
                              args.style_prompt, args.retries, args.retry_base_delay)
            dur = pcm_to_wav(raw, mime, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers,
                      desc=f"gemini:{args.model}", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
