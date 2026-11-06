"""Generate TTS audio from a ZONOS3 Mini release server for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, sends the target text plus the prompt wav
(audio-only zero-shot cloning, /tts/generate's speaker_audio_base64) and writes the
synthesized audio to <out-dir>/wavs/<uttid>.wav. <out-dir> is then ready to be
scored (scripts/score/run_score.sh).

Two conditioning systems, chosen by --speaker-mode:
    reference   raw reference prefix (the server's default for this checkpoint)
    embedding   speaker embedding only (the ownmect_s2 row)

Server text normalization is on exactly where the server has a normalizer
(TN_LANG); every other benchmark language is sent raw with it off.

--laci sends laci=true: LACI alignment-guarded decoding (servers started with
--laci-heads; the heads are looked up by the request's server language code).

Usage:
    python scripts/gen/generate_zonos3.py --server http://127.0.0.1:1920 \\
        --data-dir data/clean/en --out-dir results/zonos3-mini_reference/clean/en \\
        --speaker-mode reference
"""
import argparse
import base64
import os
import sys
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, benchmark_lang, format_error, get_json, http_post, load_test_set,
    run_generate_loop, save_wav_atomic, with_retry, write_gen_meta_once)

DEFAULT_SR = 44100

# benchmark language -> the server's TN language code. Every other benchmark
# language is sent as en_us (it passes the server's always-on language validation
# and is unused with TN off) with text_normalization=false.
TN_LANG = {"en": "en_us", "zh": "cmn", "de": "de", "es": "es", "fr": "fr_fr",
           "it": "it", "pt": "pt_br", "ja": "ja", "ko": "ko"}
SPEAKER_MODES = ("reference", "embedding")

# The release server's TTSGenerateRequest defaults, sent explicitly so gen_meta.json
# records what was used.
POLICY = dict(temperature=0.75, topk=80, top_p=0.69, min_p=0.0,
              repetition_window=96, repetition_penalty=2.4, repetition_codebooks=6,
              prepend_silence=True, silence_prefill_frames=3, speaker_cfg_scale=1.0)


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio from a ZONOS3 Mini server.")
    ap.add_argument("--server", default="http://127.0.0.1:1920")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir containing 'text' and 'prompt_wav.scp'; its "
                         "basename is the benchmark language")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--speaker-mode", required=True, choices=SPEAKER_MODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1350,
                    help="1350 = 90 s at 15 frames/s, >= 2x the longest target")
    ap.add_argument("--laci", action="store_true",
                    help="LACI alignment-guarded decoding (laci=true; needs the request "
                         "language in the server's laci_languages)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent requests")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request seconds")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per server request (matches the other clients)")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    lang = benchmark_lang(args.data_dir)
    code = TN_LANG.get(lang)
    tn = code is not None
    language = code or "en_us"

    server = args.server.rstrip("/")
    try:
        caps = get_json(server + "/tts/capabilities")
    except Exception as e:
        sys.exit(f"error: cannot read /tts/capabilities: {format_error(e)} (server {server})")
    problem = None
    if args.speaker_mode not in (caps.get("speaker_modes") or []):
        problem = f"server has no speaker_mode {args.speaker_mode!r} (has {caps.get('speaker_modes')})"
    elif not (caps.get("silence_prefill") or {}).get("available"):
        problem = "server has no silence prefill"
    elif tn and code not in (caps.get("text_norm_languages") or []):
        problem = f"server has no text normalizer for {code!r}"
    elif tn and not caps.get("text_normalization_enabled"):
        problem = "server text normalization is disabled"
    elif args.laci and language not in (caps.get("laci_languages") or []):
        problem = f"server has no LACI heads for {language!r} (has {caps.get('laci_languages')})"
    if problem:
        sys.exit(f"error: {problem} (server {server})")
    try:
        served_model = get_json(server + "/v1/models")["data"][0]["id"]
    except Exception as e:
        sys.exit(f"error: cannot read /v1/models: {format_error(e)} (server {server})")

    texts, prompts, _, uids = load_test_set(args.data_dir, need_prompt_text=False)
    uids = apply_limit(uids, args.limit)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    url = server + "/tts/generate"

    # Kept from the FIRST run of a resumed dir unless --overwrite regenerates.
    write_gen_meta_once(
        args.out_dir,
        args.overwrite,
        model="zonos3-mini",
        base_url=args.server,
        served_model=served_model,
        runtime_rev=os.environ.get("ZONOS3_RUNTIME_REV"),
        speaker_mode=args.speaker_mode,
        language=language,
        text_normalization=tn,
        seed=args.seed,
        max_tokens=args.max_tokens,
        **POLICY,
        # Recorded only when on, so non-LACI runs keep their original gen_meta shape.
        **({"laci": True, "laci_languages": caps.get("laci_languages")} if args.laci else {}),
        server_caps={k: caps.get(k) for k in (
            "audio_codec", "speaker_embedding_codec", "speaker_mode_default",
            "silence_prefill", "max_tokens")},
    )

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        prompt_path = prompts[uid]
        try:
            with open(prompt_path, "rb") as pf:
                spk_b64 = base64.b64encode(pf.read()).decode()
            payload = {
                "text": texts[uid],
                "language": language,
                "text_normalization": tn,
                "speaker_mode": args.speaker_mode,
                "speaker_audio_base64": spk_b64,
                "speaker_audio_name": os.path.basename(prompt_path),
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "stream": True,                 # streamed PCM; read() collects all chunks
                **POLICY,
            }
            if args.laci:
                payload["laci"] = True

            def _call():
                body, headers = http_post(url, payload, args.timeout)
                return body, int(headers.get("X-Audio-Sample-Rate", DEFAULT_SR))

            # Retried like the other network clients (transient resets must not
            # leave silent gaps in the wav set).
            raw, sr = with_retry(_call, args.retries, args.retry_base_delay)
            raw = raw[: len(raw) // 4 * 4]                  # whole float32 frames
            pcm = np.frombuffer(raw, dtype=np.float32)
            if pcm.size == 0:
                return uid, "empty response", 0.0
            dur = save_wav_atomic(pcm, sr, out_path)  # atomic: safe under preemption
            return uid, "ok", dur
        except Exception as e:
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers, desc="generate", wav_dir=wav_dir,
                      overwrite=args.overwrite)


if __name__ == "__main__":
    main()
