"""Generate TTS audio via Cartesia's REST TTS API (Sonic) for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, synthesizes the target text with a
Cartesia TTS model (via the POST /tts/bytes endpoint) and writes
<out-dir>/wavs/<uttid>.wav -- the layout run_infer_zonos2_eval.sh expects.

Two modes (--mode):

  preset (default) -- like the Gemini / ElevenLabs / Inworld engines, does NOT
    clone the prompt speaker: it uses a fixed pool of Cartesia public preset
    voices and picks one per utterance deterministically (so re-runs are
    reproducible), recorded in voices.txt. Sonic is multilingual -- one voice
    renders any supported language via the `language` field -- so the same
    English-named voice pool serves every language. Speaker similarity to the
    prompt is meaningless here (different speaker by design).

  clone -- true zero-shot, comparable to Zonos2 / Omni: for each utterance it
    instant-voice-clones the per-utterance reference (prompt_wav.scp) via
    POST /voices/clone, synthesizes the target text with that cloned voice, then
    DELETEs the temporary voice. This makes speaker-similarity scores comparable
    with the cloning engines, at the cost of a clone + synth + delete API round
    trip per utterance. voices.txt records the reference clip name.

Models (--model):
    sonic-3.6  sonic-3.6-2026-08-27   pinned stable Sonic 3.6 snapshot (default)

The benchmark language is the basename of --data-dir; it is sent as `language`
where Sonic supports it (nb->no, fil->tl). ca/fa are unsupported: no `language`
is sent and clones are created with language en.

Clone mode's startup/final purge deletes EVERY live evtmp-* voice on the
account: run only one clone-mode instance per account at a time.

Requires the optional 'cartesia' extra and a Cartesia API key:
    uv sync --extra cartesia
    export CARTESIA_API_KEY=...   # the token after 'Bearer ' in the curl

Usage:
    # preset voices (default)
    python scripts/generate_cartesia.py --model sonic-3.6 \\
        --data-dir data/clean/en --out-dir results/cartesia_sonic-3.6/clean/en
    # zero-shot cloning of the prompt speaker
    python scripts/generate_cartesia.py --model sonic-3.6 --mode clone \\
        --data-dir data/clean/en \\
        --out-dir results/cartesia_sonic-3.6_clone/clean/en
"""
import argparse
import os
import sys

import requests

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    apply_limit, benchmark_lang, bytes_to_wav, format_error, load_test_set,
    pick_voice, run_generate_loop, with_retry, write_gen_meta)


# --- model ids ---------------------------------------------------------------
# Keys are version-tagged so each model's output lands in a distinct
# results/cartesia_<key>/ tree (never conflate two models in one dir).
DEFAULT_MODEL = "sonic-3.6"
MODEL_IDS = {"sonic-3.6": "sonic-3.6-2026-08-27"}  # pinned stable snapshot

# --- endpoints ---------------------------------------------------------------
TTS_URL = "https://api.cartesia.ai/tts/bytes"   # returns raw audio bytes
CLONE_URL = "https://api.cartesia.ai/voices/clone"   # POST multipart -> {id,...}
VOICE_URL = "https://api.cartesia.ai/voices/{id}"    # DELETE temp clone (cleanup)
VOICES_LIST_URL = "https://api.cartesia.ai/voices/"  # GET catalogue (stale-clone purge)
CARTESIA_VERSION = "2026-08-14"                 # required Cartesia-Version header

# Request a complete WAV (PCM_16) so soundfile can read the result directly.
SAMPLE_RATE = 44100   # Sonic's native rate
CONTAINER = "wav"
ENCODING = "pcm_s16le"

# --- voice pool (rotated per utterance) --------------------------------------
# A fixed set of Cartesia *public* voices (name -> voice_id), harvested once from
# GET /voices (source = public English voices). Sonic is multilingual, so these
# render any target language via the `language` field -- the same way the
# ElevenLabs engine reuses one English voice pool across all languages.
CARTESIA_VOICES = {
    "Skylar": "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
    "Corey": "630ed21c-2c5c-41cf-9d82-10a7fd668370",
    "Gemma": "62ae83ad-4f6a-430b-af41-a9bede9286ca",
    "Archie": "ef191366-f52f-447a-a398-ed8c0f2943a1",
    "Daniel": "47c38ca4-5f35-497b-b1a3-415245fb35e1",
    "Katie": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
    "Jacqueline": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    "Ronald": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
    "Ella": "2a12b36c-7f9b-4c3a-9f7a-72731b15323a",
    "Caroline": "f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
    "Blake": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
    "Cathy": "e8e5fffb-252c-436d-b842-8879b84445b6",
    "Theo": "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e",
    "Brooke": "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    "Nolan": "65209f8e-6140-4a20-b819-3cc2e21da19b",
    "Lucy": "2f251ac3-89a9-4a77-a452-704b474ccd01",
    "Julia": "273f9ef7-9fc2-4def-88bb-ab108c6249ca",
    "Grace": "a4a16c5e-5902-4732-b9b6-2a48efd2e11b",
    "Jameson": "a5136bf9-224c-4d76-b823-52bd5efcffcc",
    "Austin": "1fcd23d0-bf12-4896-8f60-4f21ef5c9b98",
}
VOICE_NAMES = list(CARTESIA_VOICES)

# Benchmark language -> Sonic 3.6 `language` code. ca and fa are unsupported
# (absent): no `language` is sent and clones fall back to en.
CARTESIA_LANG = {
    **{lg: lg for lg in (
        "en", "fr", "de", "es", "pt", "zh", "ja", "hi", "it", "ko", "nl", "pl",
        "ru", "sv", "tr", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms", "sk",
        "da", "uk", "hu", "vi", "th", "he", "id", "tl")},
    "nb": "no", "fil": "tl",
}


def synth(headers, model_id, text, voice_id, lang, timeout, retries, base_delay):
    """Run one Cartesia synthesis and return the result audio bytes, retried."""
    payload = {
        "model_id": model_id,
        "transcript": text,
        "voice": {"id": voice_id},
        "output_format": {
            "container": CONTAINER,
            "encoding": ENCODING,
            "sample_rate": SAMPLE_RATE,
        },
    }
    if lang in CARTESIA_LANG:
        payload["language"] = CARTESIA_LANG[lang]

    def _call():
        r = requests.post(TTS_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        if not r.content:
            raise RuntimeError("empty response body from cartesia")
        return r.content

    return with_retry(_call, retries, base_delay)


def clone_voice(auth_headers, clip_path, name, lang, timeout, retries, base_delay):
    """Instant-voice-clone one reference clip; return the temporary voice id (retried).

    Uses multipart POST /voices/clone -- note `auth_headers` must NOT set
    Content-Type (requests adds the multipart boundary itself)."""
    clone_lang = CARTESIA_LANG.get(lang, "en")

    def _call():
        with open(clip_path, "rb") as fh:
            r = requests.post(
                CLONE_URL,
                headers=auth_headers,
                data={"name": name, "language": clone_lang},
                files={"clip": (os.path.basename(clip_path), fh, "audio/wav")},
                timeout=timeout,
            )
        r.raise_for_status()
        j = r.json()
        vid = j.get("id")
        if not vid:
            raise RuntimeError(f"clone returned no voice id: {j}")
        return vid

    return with_retry(_call, retries, base_delay)


def delete_voice(auth_headers, vid, timeout):
    """Best-effort cleanup of a temporary cloned voice (DELETE /voices/{id})."""
    try:
        requests.delete(VOICE_URL.format(id=vid), headers=auth_headers, timeout=timeout)
    except Exception:  # noqa: BLE001 - cleanup must never fail the run
        pass


def purge_stale_clones(auth_headers, timeout):
    """Best-effort: delete leftover evtmp-* clone voices from a prior crashed run.

    A clean clone-mode run deletes each temp voice in its finally block, but a
    SIGKILL between clone and delete leaks the voice on the account permanently.
    Mirrors generate_elevenlabs_clone.purge_stale / generate_inworld's
    _sweep_orphans. Needs voices-read scope; silently skipped on any error."""
    try:
        stale = []
        starting_after = None
        for _ in range(100):  # hard page cap so a bad cursor can't loop forever
            params = {"limit": 100}
            if starting_after:
                params["starting_after"] = starting_after
            r = requests.get(VOICES_LIST_URL, headers=auth_headers, params=params,
                             timeout=timeout)
            r.raise_for_status()
            body = r.json()
            voices = body.get("data", []) if isinstance(body, dict) else body
            if not isinstance(voices, list) or not voices:
                break
            for v in voices:
                if isinstance(v, dict) and v.get("id") and \
                        str(v.get("name", "")).startswith("evtmp-"):
                    stale.append(v["id"])
            if not (isinstance(body, dict) and body.get("has_more")):
                break
            last = voices[-1]
            starting_after = last.get("id") if isinstance(last, dict) else None
            if not starting_after:
                break
        if stale:
            print(f"[purge] removing {len(stale)} stale evtmp-* clone voices")
            for vid in stale:
                delete_voice(auth_headers, vid, timeout)
    except Exception as e:  # noqa: BLE001 - purge must never fail the run
        print(f"[purge] skipped (cannot list voices): {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate TTS audio via the Cartesia REST API for a ZTTS2-Eval test set.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_IDS),
                    help=f"Cartesia TTS model (default: {DEFAULT_MODEL})")
    ap.add_argument("--mode", default="preset", choices=["preset", "clone"],
                    help="preset = rotate public voices (default); clone = "
                         "instant-voice-clone each prompt_wav reference (zero-shot)")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--workers", type=int, default=4, help="concurrent cartesia requests")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per cartesia call")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--no-purge", action="store_true",
                    help="clone mode: skip the startup/final purge of leftover "
                         "evtmp-* voices (use when other clone runs are live)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        sys.exit("error: CARTESIA_API_KEY environment variable is not set "
                 "(the token after 'Bearer ' in the curl)")

    model_id = MODEL_IDS[args.model]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
    }
    # Clone/delete are multipart / no body: same auth but let requests set the
    # Content-Type (multipart boundary) itself.
    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": CARTESIA_VERSION,
    }

    # Intersect so the uttid set matches the other engines (each test-set dir
    # has all three files); Cartesia itself only needs the target text.
    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    uids = apply_limit(uids, args.limit)
    lang = benchmark_lang(args.data_dir)
    if lang not in CARTESIA_LANG:
        # Never gate silently: Sonic is not sent a `language` for this split.
        print(f"[warn] '{lang}' is not supported by Sonic; the API request omits "
              f"`language`{' and clones use en' if args.mode == 'clone' else ''}")

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[cartesia:{args.model}:{args.mode}] {args.data_dir} -> {wav_dir} | "
          f"{len(uids)} utterances (text={len(texts)}, prompt_wav={len(prompts)}, "
          f"prompt_text={len(prompt_texts)})")

    # voice manifest -- deterministic, so write it upfront. In preset mode the
    # voice is a rotated public name; in clone mode it is the reference clip.
    voice_for = {}
    clip_for = {}
    missing_clips = 0
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            if args.mode == "clone":
                clip = prompts[uid]
                clip_for[uid] = clip
                if not os.path.exists(clip):
                    missing_clips += 1
                voice_for[uid] = os.path.splitext(os.path.basename(clip))[0]
            else:
                voice_for[uid] = pick_voice(uid, VOICE_NAMES)
            f.write(f"{uid}\t{voice_for[uid]}\t{lang}\n")
    print(f"[cartesia:{args.model}:{args.mode}] voice assignments -> {vpath}")
    if args.mode == "clone" and missing_clips:
        print(f"[warn] {missing_clips}/{len(uids)} prompt_wav clips not found on disk "
              f"(those utterances will fail to clone)")

    # Kept from the FIRST run of a resumed dir unless --overwrite regenerates.
    if args.overwrite or not os.path.exists(os.path.join(args.out_dir, "gen_meta.json")):
        write_gen_meta(args.out_dir, provider="cartesia", model=model_id,
                       endpoint=TTS_URL, voice_mode=args.mode, language=lang,
                       language_sent=CARTESIA_LANG.get(lang),
                       output_format={"container": CONTAINER, "encoding": ENCODING,
                                      "sample_rate": SAMPLE_RATE},
                       api_version=CARTESIA_VERSION)

    if args.mode == "clone" and not args.no_purge:
        purge_stale_clones(auth_headers, args.timeout)  # sweep prior crashes

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            if args.mode == "clone":
                clip = clip_for[uid]
                if not os.path.exists(clip):
                    # A missing local file is permanent -- don't burn retries on it.
                    return uid, f"error: prompt wav missing ({clip})", 0.0
                vid = clone_voice(auth_headers, clip, f"evtmp-{uid}",
                                  lang, args.timeout, args.retries,
                                  args.retry_base_delay)
                try:
                    raw = synth(headers, model_id, texts[uid], vid, lang,
                                args.timeout, args.retries, args.retry_base_delay)
                    dur = bytes_to_wav(raw, out_path)
                finally:
                    delete_voice(auth_headers, vid, args.timeout)
            else:
                voice_id = CARTESIA_VOICES[voice_for[uid]]
                raw = synth(headers, model_id, texts[uid], voice_id, lang,
                            args.timeout, args.retries, args.retry_base_delay)
                dur = bytes_to_wav(raw, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers,
                      desc=f"cartesia:{args.model}", wav_dir=wav_dir)
    if args.mode == "clone" and not args.no_purge:
        purge_stale_clones(auth_headers, args.timeout)  # final sweep


if __name__ == "__main__":
    main()
