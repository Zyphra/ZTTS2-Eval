"""Generate TTS audio from a Mini-SGLang / Zonos2 server for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, sends the target text plus the prompt wav
(zero-shot voice cloning, via /tts/generate's speaker_audio_base64) to the server and
writes the synthesized audio to <out-dir>/wavs/<uttid>.wav. <out-dir> is then ready
to be scored by scripts/score/run_score.sh.

Usage:
    python scripts/gen/generate_zonos2.py \\
        --data-dir data/clean/en --out-dir results/clean/en --language en_us
"""
import argparse
import base64
import json
import os

import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, http_post, load_test_set, run_generate_loop,
    save_wav_atomic, with_retry, write_gen_meta_once)

DEFAULT_SR = 44100


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio from a Zonos2 server.")
    ap.add_argument("--server", default="http://127.0.0.1:20234")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir containing 'text' and 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--language", default="en_us", help="server TTS language code")
    ap.add_argument("--text-normalization", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="expand numbers/dates/units to words before synthesis")
    ap.add_argument("--normalizer-language", default="",
                    help="normalizer language code (en, zh, de, es, fr, it, ru, ...); "
                         "use together with --text-normalization")
    ap.add_argument("--byte-tokenize-all", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="tokenize text as raw UTF-8 bytes, skipping phonemization "
                         "(default: on -- this Zonos2 checkpoint requires byte mode)")
    ap.add_argument("--seed", type=int, default=0)
    # Speaking-rate control: index into /tts/capabilities speaking_rate_buckets
    # (the current server exposes 8 buckets). Sets speaking_rate_enabled=True
    # in the payload when given.
    ap.add_argument("--speaking-rate-bucket", type=int, default=None,
                    help="bucket index into the server's speaking_rate_buckets; "
                         "sets speaking_rate_enabled=True when given")
    # Accuracy-mode token: gated by accurate_mode_token_enabled in
    # /tts/capabilities. Verified against the zonos2 server source
    # (server/api_server.py): the request field is `accurate_mode` and it
    # DEFAULTS TO TRUE server-side -- "accurate mode (on) vs expressive mode
    # (off)". So the meaningful client toggle is --no-accurate-mode
    # (expressive); omitting the flag leaves the server default (accurate).
    # Same-seed A/B confirms accurate_mode=false changes the output.
    ap.add_argument("--accurate-mode", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="accurate mode (server default ON); --no-accurate-mode "
                         "switches to expressive mode. Omitted if unset. Requires "
                         "accurate_mode_token_enabled in /tts/capabilities")
    # Speaker-origin ("background") marker token. Training configs call this
    # `speaker_embedding_origin_token_enabled`; the server maps it to the model
    # flag `speaker_background_token_enabled` (TRUE on the released ZONOS2), and
    # scheduler.py always emits ONE of the two bucket tokens right after the
    # speaker slot. The per-request field selecting which bucket is
    # `clean_speaker_background` (server default False = "noisy origin").
    # Passing True asserts the reference clip is clean-studio origin.
    ap.add_argument("--clean-speaker-background", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="speaker-origin marker bucket: True = clean origin, "
                         "False = noisy (server default). Omitted if unset. "
                         "Requires speaker_background_token_enabled in /tts/capabilities")
    # Quality-bucket conditioning. The model exposes 6 features x N buckets
    # (params.json quality_buckets: lufs, estimated_snr, max_pause,
    # estimated_bandlimit_hz, leading_silence_s, trailing_silence_s); the request
    # field `quality_buckets` is a dict of feature -> bucket index.
    # NOTE /tts/generate has NO default quality conditioning: omit this and the
    # server sends no quality tokens at all (api_server._resolve_quality_buckets
    # returns None; prompt.py skips None-bucket features). The
    # _DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3} in the server source
    # applies only to the OpenAI-compatible /v1/audio/speech endpoint. So listing a
    # feature turns it on; unlisted features stay unconditioned, replacing nothing.
    # Accepts JSON, but prefer the quote-free "k=v:k=v" form: utils/parse_options.sh
    # does `eval $name=\"$2\"`, whose quote removal silently strips JSON's inner
    # double quotes (the server then 400s), and sbatch --export splits on commas.
    ap.add_argument("--quality-buckets", default=None,
                    help='quality bucket indices as "estimated_snr=11:'
                         'estimated_bandlimit_hz=7" (colon- or comma-separated k=v), or '
                         'a JSON dict. Only the listed features are conditioned. '
                         'Omitted if unset (= no quality conditioning).')
    # Sampling / decoding parameters. Defaults are the released Zonos2 model's
    # reference (UI-default) eval config:
    #   temperature=1.15 topk=106 top_p=0 min_p=0.18 max_tokens=4096
    #   silence_penalty=0 prepend_silence=on
    #   repetition_window=50 repetition_penalty=1.2 repetition_codebooks=8
    ap.add_argument("--temperature", type=float, default=1.15)
    ap.add_argument("--topk", type=int, default=106)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--min-p", type=float, default=0.18)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--repetition-window", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--repetition-codebooks", type=int, default=8)
    ap.add_argument("--silence-penalty", type=float, default=0.0,
                    help="server silence_penalty (UI default 0)")
    ap.add_argument("--prepend-silence", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="server prepend_silence: include the precomputed silence "
                         "prefix in the prompt (UI default On)")
    # Request fields this client has no flag for: merged into every payload last.
    # A null value drops that field, so the server's own default applies.
    ap.add_argument("--request-extras", default=None,
                    help='JSON object merged into each request last, e.g. '
                         '\'{"temperature": null}\'; '
                         'null drops the field (server default)')
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--stratified-bins", type=int, default=0,
                    help="if >0, pick K evenly-spaced utts across the full set "
                         "(useful when the set is sorted by quality bin -- e.g. "
                         "in-the-wild is 5 MSR-UTMOS bins per lang; --stratified-bins 5 "
                         "samples one utt per bin instead of the worst N first)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request seconds")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per server request (matches the other clients)")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    quality_buckets = None
    if args.quality_buckets:
        spec = args.quality_buckets.strip()
        if spec.startswith("{"):
            try:
                quality_buckets = json.loads(spec)
            except json.JSONDecodeError:
                ap.error(f"--quality-buckets is not valid JSON: {spec!r} (shell quote "
                         "removal? use the quote-free k=v:k=v form instead)")
        else:
            quality_buckets = {}
            for pair in spec.replace(",", ":").split(":"):
                if not pair:
                    continue
                if "=" not in pair:
                    ap.error(f"--quality-buckets: expected feature=index, got {pair!r}")
                k, v = pair.split("=", 1)
                quality_buckets[k.strip()] = int(v)
        if not isinstance(quality_buckets, dict) or not quality_buckets:
            ap.error("--quality-buckets must be a non-empty map of feature -> bucket index")
    request_extras = {}
    if args.request_extras:
        try:
            request_extras = json.loads(args.request_extras)
        except json.JSONDecodeError:
            ap.error(f"--request-extras is not valid JSON: {args.request_extras!r}")
        if not isinstance(request_extras, dict):
            ap.error("--request-extras must be a JSON object")

    texts, prompts, _, uids = load_test_set(args.data_dir, need_prompt_text=False)
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    url = args.server.rstrip("/") + "/tts/generate"

    # Provenance: the payload-shaping params, kept from the FIRST run of a
    # resumed dir unless --overwrite regenerates.
    write_gen_meta_once(
        args.out_dir,
        args.overwrite,
        model="zonos2",
        base_url=args.server,
        language=args.language,
        seed=args.seed,
        temperature=args.temperature,
        topk=args.topk,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        repetition_window=args.repetition_window,
        repetition_penalty=args.repetition_penalty,
        repetition_codebooks=args.repetition_codebooks,
        silence_penalty=args.silence_penalty,
        prepend_silence=args.prepend_silence,
        byte_tokenize_all=args.byte_tokenize_all,
        text_normalization=args.text_normalization,
        speaking_rate_bucket=args.speaking_rate_bucket,
        accurate_mode=args.accurate_mode,
        clean_speaker_background=args.clean_speaker_background,
        quality_buckets=quality_buckets,
        request_extras=request_extras or None,
    )

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        prompt_path = prompts[uid]
        try:
            payload = {
                "text": texts[uid],
                "language": args.language,
                "byte_tokenize_all": args.byte_tokenize_all,
                "text_normalization_enabled": args.text_normalization,
                "stream": True,                 # streamed PCM; read() collects all chunks
                "seed": args.seed,
                "temperature": args.temperature,
                "topk": args.topk,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "max_tokens": args.max_tokens,
                "repetition_window": args.repetition_window,
                "repetition_penalty": args.repetition_penalty,
                "repetition_codebooks": args.repetition_codebooks,
                "silence_penalty": args.silence_penalty,
                "prepend_silence": args.prepend_silence,
            }
            with open(prompt_path, "rb") as pf:
                spk_b64 = base64.b64encode(pf.read()).decode()
            payload["speaker_audio_base64"] = spk_b64
            payload["speaker_audio_name"] = os.path.basename(prompt_path)
            if args.text_normalization and args.normalizer_language:
                payload["normalizer_language"] = args.normalizer_language
            if args.speaking_rate_bucket is not None:
                payload["speaking_rate_enabled"] = True
                payload["speaking_rate_bucket"] = args.speaking_rate_bucket
            if args.accurate_mode is not None:
                payload["accurate_mode"] = args.accurate_mode
            if args.clean_speaker_background is not None:
                payload["clean_speaker_background"] = args.clean_speaker_background
            if quality_buckets is not None:
                payload["quality_enabled"] = True
                payload["quality_buckets"] = quality_buckets
            for key, value in request_extras.items():
                if value is None:
                    payload.pop(key, None)
                else:
                    payload[key] = value

            def _call():
                body, headers = http_post(url, payload, args.timeout)
                return body, int(headers.get("X-Audio-Sample-Rate", DEFAULT_SR))

            # Retried like the other network clients (transient resets / server
            # restarts must not leave silent gaps in the wav set).
            raw, sr = with_retry(_call, args.retries, args.retry_base_delay)
            raw = raw[: len(raw) // 4 * 4]                  # whole float32 frames
            pcm = np.frombuffer(raw, dtype=np.float32)
            if pcm.size == 0:
                return uid, "empty response", 0.0
            dur = save_wav_atomic(pcm, sr, out_path)  # atomic: safe under preemption
            return uid, "ok", dur
        except Exception as e:
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers, desc="generate",
                      wav_dir=wav_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
