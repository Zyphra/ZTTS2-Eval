"""Generate TTS audio from a local vLLM-Omni speech server for a ZTTS2-Eval test set.

POSTs each utterance in <data-dir>/text to the server's OpenAI-compatible
/v1/audio/speech endpoint and writes the result to <out-dir>/wavs/<uttid>.wav --
the layout run_infer_zonos2_eval.sh expects.

This is an HTTP client only: run_generate_omni.sh (via scripts/serve_omni.sh) starts
and stops the `vllm serve --omni` server; this script talks to one already running.

Synthesis mode -- pick the one the served model supports (run_generate_omni.sh sets it):
    clone   zero-shot voice cloning from the prompt wav: task_type=Base, ref_audio +
            ref_text -- the benchmark's intended task
    preset  built-in speaker, a different one per utterance: task_type=CustomVoice
    design  task_type=VoiceDesign with a fixed style instruction

Usage:
    python scripts/generate_omni.py --server http://127.0.0.1:8091 --mode clone \\
        --data-dir data/clean/en \\
        --out-dir results/omni_qwen3-1.7b-base/clean/en --language en
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    bytes_to_wav as save_wav, detect_lang, format_error, load_test_set,
    normalize_lang, pick_voice, run_generate_loop, with_retry)

SPEECH_PATH = "/v1/audio/speech"
VOICES_PATH = "/v1/audio/voices"

# vLLM-Omni's `language` field accepts these names (among others); we map the
# benchmark's language codes onto them and fall back to "Auto".
LANG_NAMES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
              "de": "German", "es": "Spanish", "fr": "French", "it": "Italian",
              "ru": "Russian", "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi"}

# Voxtral-style preset voice naming uses '<lang>_<gender>' prefixes for non-English
# voices. Restrict the preset pool to voices matching the target language so a
# German request doesn't pick a Spanish voice. For English / unknown targets,
# drop language-coded voices and keep the unprefixed ones (casual/neutral/...).
VOICE_LANG_PREFIXES = {"de": "de_", "es": "es_", "fr": "fr_", "it": "it_",
                       "pt": "pt_", "ar": "ar_", "hi": "hi_", "nl": "nl_"}

DEFAULT_VOICE = "vivian"          # vLLM-Omni's documented default speaker
DESIGN_INSTRUCTION = "A clear, natural voice speaking in a neutral tone."


def normalize_prompt_path(raw):
    """Collapse doubled slashes in prompt_wav.scp paths."""
    return os.path.normpath(raw)


def server_alive(server, timeout=5):
    """Quick GET /health so a wrong --server (or a dead server) fails fast."""
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/health",
                                    timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def http_post(url, payload, timeout):
    """POST a JSON body; return the raw response bytes. Raises with detail on error."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "zonos2-eval"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None


def http_get_json(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "zonos2-eval"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_voices(server, timeout):
    """GET /v1/audio/voices -> list of speaker names. Falls back to [DEFAULT_VOICE]."""
    try:
        obj = http_get_json(server.rstrip("/") + VOICES_PATH, timeout)
    except Exception:
        return [DEFAULT_VOICE]
    if isinstance(obj, dict):
        obj = obj.get("voices") or obj.get("data") or []
    names = []
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, str):
                names.append(v)
            elif isinstance(v, dict):
                n = v.get("name") or v.get("id") or v.get("voice")
                if n:
                    names.append(n)
    return names or [DEFAULT_VOICE]


def filter_voices_by_lang(voices, lang):
    """Restrict a preset-voice pool to voices matching the target language.

    Voxtral-style voices encode language in a prefix (de_male, fr_female, ...);
    English / generic voices have no prefix (casual_*, neutral_*, cheerful_*).
    For a language with an explicit prefix, keep only those voices. For 'en' or
    anything unmapped, drop the language-coded voices so we don't synthesize
    English text with a French preset. Empty selection falls back to the full
    pool so we never block generation.
    """
    if not lang:
        return voices
    if lang in VOICE_LANG_PREFIXES:
        sub = [v for v in voices if v.startswith(VOICE_LANG_PREFIXES[lang])]
    else:
        sub = [v for v in voices
               if not any(v.startswith(p) for p in VOICE_LANG_PREFIXES.values())]
    return sub or voices


def audio_to_data_uri(path):
    """Read an audio file and return a base64 'data:' URI.

    For servers whose clone path cannot open local files -- notably Voxtral, whose
    ref_audio is handed straight to mistral_common, which expects inline base64."""
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
    mime = "audio/wav" if ext == "wav" else f"audio/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_request(mode, target_text, lang, prompt_path, prompt_text, voice,
                  model, max_new_tokens, ref_audio_mode="url",
                  seed=None, temperature=None, no_ref_text=False,
                  x_vector_only=False, send_language=True):
    """Assemble the /v1/audio/speech JSON body for one utterance."""
    body = {
        "input": target_text,
        "language": LANG_NAMES.get(lang, "Auto"),
        "response_format": "wav",
        "max_new_tokens": max_new_tokens,
    }
    if model:
        body["model"] = model
    # vLLM-Omni accepts OpenAI-style sampling params; only send when set so
    # the server defaults still apply for ordinary runs. seed makes runs
    # reproducible per-utt; varying seed across runs gives different
    # prosodic realisations of the same text and prompt.
    if seed is not None:
        body["seed"] = int(seed)
    if temperature is not None:
        body["temperature"] = float(temperature)
    if mode == "clone":
        # task_type=Base clones the speaker from a reference wav + its transcript.
        body["task_type"] = "Base"
        # 'url': a file:// path the server (same host) reads locally -- works for
        # most models. 'base64': an inline data: URI -- required by Voxtral, whose
        # ref_audio goes straight to mistral_common with no file:// support.
        if ref_audio_mode == "base64":
            body["ref_audio"] = audio_to_data_uri(prompt_path)
        else:
            body["ref_audio"] = "file://" + os.path.abspath(prompt_path)
        # Audio-only ablation: omit the reference transcript so cloning uses the
        # speaker audio alone (matches Zonos2's audio-only setup). For Qwen3 the
        # server's Base task requires ref_text unless x_vector_only_mode is set,
        # so --x-vector-only must accompany --no-ref-text there.
        if not no_ref_text:
            body["ref_text"] = prompt_text
        if x_vector_only:
            body["x_vector_only_mode"] = True
    elif mode == "preset":
        body["task_type"] = "CustomVoice"
        body["voice"] = voice
    elif mode == "design":
        body["task_type"] = "VoiceDesign"
        body["instructions"] = DESIGN_INSTRUCTION
    else:
        raise ValueError(f"unknown mode: {mode}")
    if not send_language:
        del body["language"]
    return body


def main():
    ap = argparse.ArgumentParser(
        description="Generate TTS audio from a vLLM-Omni speech server.")
    ap.add_argument("--server", default="http://127.0.0.1:8091",
                    help="vLLM-Omni server base URL")
    ap.add_argument("--mode", choices=["clone", "preset", "design"], default="clone",
                    help="synthesis mode the served model supports")
    ap.add_argument("--ref-audio-mode", choices=["url", "base64"], default="url",
                    help="clone-mode reference audio delivery: 'url' (a file:// path "
                         "the server reads locally) or 'base64' (an inline data: URI "
                         "-- required by Voxtral)")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--model", default="",
                    help="model id for the request 'model' field "
                         "(default: the server's loaded model)")
    ap.add_argument("--language", default="",
                    help="target language override (en/zh/ja/ko); "
                         "default: auto-detect per utterance")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=None,
                    help="sampler seed (passed through to vLLM); omit for "
                         "server-default behaviour")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampler temperature (passed through to vLLM); "
                         "omit to keep server defaults")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout, in seconds")
    ap.add_argument("--retries", type=int, default=4, help="attempts per request")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    ap.add_argument("--no-ref-text", action="store_true",
                    help="clone-mode: omit ref_text (audio-only cloning). Some "
                         "models (Fish, CosyVoice3) reject this server-side.")
    ap.add_argument("--x-vector-only", action="store_true",
                    help="clone-mode: set x_vector_only_mode=True (Qwen3 needs this "
                         "to clone without ref_text).")
    ap.add_argument("--no-language", action="store_true",
                    help="omit the request 'language' field (Breeze-TTS-2 infers the "
                         "language from the text and rejects the field).")
    args = ap.parse_args()

    if not server_alive(args.server):
        sys.exit(f"error: no vLLM-Omni server reachable at {args.server} -- start one "
                 f"with run_generate_omni.sh (or scripts/serve_omni.sh)")

    # Intersect so the uttid set is identical across models/modes (every test-set
    # dir has all three files); clone mode additionally needs the prompt wav + text.
    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    if args.limit > 0:
        uids = uids[:args.limit]

    forced_lang = normalize_lang(args.language)
    if args.language and forced_lang not in LANG_NAMES:
        print(f"[warn] --language '{args.language}' -> '{forced_lang}' is not in "
              f"LANG_NAMES ({','.join(sorted(LANG_NAMES))}); falling back to auto-detect")
        forced_lang = ""

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    speech_url = args.server.rstrip("/") + SPEECH_PATH

    # preset mode rotates the server's built-in voices; fetch them once.
    voices = None
    if args.mode == "preset":
        voices = fetch_voices(args.server, args.timeout)
        if forced_lang:
            voices = filter_voices_by_lang(voices, forced_lang)
        lang_tag = f" for lang={forced_lang}" if forced_lang else ""
        print(f"[omni] preset voices ({len(voices)}{lang_tag}): "
              f"{', '.join(voices[:12])}{' ...' if len(voices) > 12 else ''}")

    print(f"[omni:{args.mode}] {args.data_dir} -> {wav_dir} | {len(uids)} utterances "
          f"(text={len(texts)}, prompt_wav={len(prompts)}, prompt_text={len(prompt_texts)})")

    # voice manifest -- deterministic, so write it upfront for preset mode
    if args.mode == "preset":
        vpath = os.path.join(args.out_dir, "voices.txt")
        with open(vpath, "w", encoding="utf-8") as f:
            for uid in uids:
                f.write(f"{uid}\t{pick_voice(uid, voices)}\n")

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            target_text = texts[uid]
            lang = forced_lang or detect_lang(target_text)
            prompt_path = normalize_prompt_path(prompts[uid])
            voice = pick_voice(uid, voices) if voices else DEFAULT_VOICE
            body = build_request(args.mode, target_text, lang, prompt_path,
                                 prompt_texts[uid], voice, args.model,
                                 args.max_new_tokens, args.ref_audio_mode,
                                 seed=args.seed, temperature=args.temperature,
                                 no_ref_text=args.no_ref_text,
                                 x_vector_only=args.x_vector_only,
                                 send_language=not args.no_language)
            try:
                raw = with_retry(lambda: http_post(speech_url, body, args.timeout),
                                 args.retries, args.retry_base_delay)
            except RuntimeError as e:
                # Servers advertise different language sets: qwen3-tts accepts only 11
                # and rejects e.g. Arabic/Hindi with `HTTP 400: Invalid language
                # 'Arabic'. Supported: Auto, ...`, which failed 1500/1500 utterances on
                # VoxBlink ar/hi. LANG_NAMES is a superset of what some servers take, so
                # fall back to the server-side detector rather than dropping the cell.
                # (clean never hit this only because its driver maps just 9 langs.)
                if "Invalid language" not in str(e) or body.get("language") == "Auto":
                    raise
                print(f"[gen] {uid}: server rejected language "
                      f"{body['language']!r} -- retrying with Auto", flush=True)
                body["language"] = "Auto"
                raw = with_retry(lambda: http_post(speech_url, body, args.timeout),
                                 args.retries, args.retry_base_delay)
            dur = save_wav(raw, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers, desc=args.mode,
                      wav_dir=wav_dir)


if __name__ == "__main__":
    main()
