"""Generate TTS audio from an Audio8-TTS SGLang-Omni server for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, sends the target text plus the prompt wav
and its transcript (with-transcript zero-shot cloning, one reference per request)
to POST /v1/audio/speech and writes the returned 44.1 kHz PCM16 WAV to
<out-dir>/wavs/<uttid>.wav. <out-dir> is then ready to be scored
(scripts/score/run_score.sh).

The server reads the reference wav from its OWN filesystem, so the prompt path is
sent absolute. Sampling (upstream CLI/adapter defaults) is sent explicitly: omitted
fields fall back to the server's S2-Pro defaults. The adapter has no seed input, so
sampling is unseeded. Text is sent raw (no normalization, no splitting); a
generation that hits --max-new-tokens without EOS is kept truncated and logged.

Usage:
    python scripts/gen/generate_audio8.py --server http://127.0.0.1:8010 \\
        --data-dir data/clean/en --out-dir results/audio8-tts-realistic-0.6b/clean/en
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, bytes_to_wav, format_error, load_test_set, run_generate_loop,
    with_retry, write_gen_meta)

MODEL_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
MODEL_REVISION = "1e857568bfd4f02931aca3988ee773a44d2d26d3"
SPEECH_PATH = "/v1/audio/speech"
# Upstream CLI + adapter defaults, sent explicitly (the server's own defaults are S2-Pro's).
POLICY = dict(temperature=0.8, top_p=0.95, top_k=50)


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio from an Audio8 SGLang-Omni server.")
    ap.add_argument("--server", default="http://127.0.0.1:8010")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir containing 'text', 'prompt_wav.scp' and 'prompt_text'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="codec frames per request (1024 = 47.6 s; fits the 2048 context "
                         "for every benchmark prompt)")
    ap.add_argument("--workers", type=int, default=32,
                    help="concurrent requests (matches the server's max running requests)")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request seconds")
    ap.add_argument("--retries", type=int, default=4, help="attempts per server request")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    server = args.server.rstrip("/")
    try:
        with urllib.request.urlopen(server + "/v1/models", timeout=10) as resp:
            served_model = json.loads(resp.read())["data"][0]["id"]
    except Exception as e:
        sys.exit(f"error: cannot read /v1/models: {format_error(e)} (server {server})")

    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    uids = apply_limit(uids, args.limit)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    url = server + SPEECH_PATH

    # Kept from the FIRST run of a resumed dir unless --overwrite regenerates.
    if args.overwrite or not os.path.exists(os.path.join(args.out_dir, "gen_meta.json")):
        write_gen_meta(
            args.out_dir,
            model=MODEL_ID,
            model_revision=MODEL_REVISION,
            served_model=served_model,
            base_url=args.server,
            runtime_rev=os.environ.get("AUDIO8_RUNTIME_REV"),
            cuda_graph=os.environ.get("AUDIO8_DISABLE_CUDA_GRAPH", "0") != "1",
            conditioning="prompt wav + prompt_text transcript",
            max_new_tokens=args.max_new_tokens,
            seed=None,
            **POLICY,
        )

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        payload = {
            "model": served_model,
            "input": texts[uid],
            "response_format": "wav",
            "max_new_tokens": args.max_new_tokens,
            **POLICY,
            "references": [{"audio_path": os.path.abspath(os.path.normpath(prompts[uid])),
                            "text": prompt_texts[uid]}],
        }
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})

            def _call():
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    return resp.read(), resp.headers.get("X-Completion-Tokens")

            raw, ctoks = with_retry(_call, args.retries, args.retry_base_delay)
            if ctoks and ctoks.isdigit() and int(ctoks) >= args.max_new_tokens:
                # No EOS within the cap; one generation per prompt, so keep it.
                print(f"[audio8] {uid}: hit max_new_tokens ({args.max_new_tokens} frames), "
                      "kept truncated", flush=True)
            dur = bytes_to_wav(raw, out_path)
            return uid, "ok", dur
        except urllib.error.HTTPError as e:
            return uid, f"http {e.code}: {e.read().decode(errors='replace')[:200]}", 0.0
        except Exception as e:
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
