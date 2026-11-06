"""Generate TTS audio via Google's native Gemini TTS for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, synthesizes the target text with a
Gemini TTS model (via the official google-genai SDK) and writes
<out-dir>/wavs/<uttid>.wav -- the layout scripts/score/run_score.sh expects.

This harness does not clone with Gemini (voice replication needs a consent
recording from the same speaker, which benchmark prompts can't provide): it uses
the 30 prebuilt voices, and Gemini infers the spoken language from the text
itself (no language code is sent). Like
the other preset-voice engines in this harness, a voice is picked per utterance
deterministically (so re-runs are reproducible) and recorded in voices.txt.

Models (--model):
    3.8-flash  gemini-3.8-flash-tts   (default)

Serving (--batch): by default each utterance is one unary generateContent call.
With --batch, the missing utterances of the split are submitted as Batch API jobs
(JSONL input file, 50% of the unary price, async with a 24 h target turnaround).
Job names are kept in <out-dir>/batch_jobs.json, so an interrupted or
--batch-no-wait run re-attaches to its jobs instead of resubmitting them (job
creation is not idempotent). Finished jobs are downloaded, written as wavs, and
deleted along with their input file (the File API has a 20 GB storage cap);
requests that failed inside a job are resubmitted, for up to --batch-rounds
rounds per run (and again by the next run).
gen_meta.json records `serving` (unary, batch, or unary+batch for a mixed split).

Requires the optional 'gemini' extra (google-genai>=2.25.0) and a Google API key:
    uv sync --inexact --extra gemini
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

Usage:
    python scripts/gen/generate_gemini.py --model 3.8-flash \\
        --data-dir data/clean/en --out-dir results/gemini_3.8-flash/clean/en
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.genclient import (  # noqa: E402
    apply_limit, benchmark_lang, bytes_to_wav, format_error, http_status_of,
    load_test_set, parse_l16_rate, pick_voice, retry_after_of, run_generate_loop,
    save_wav_atomic, with_retry, write_gen_meta)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# --- model ids ---------------------------------------------------------------
# Keys are version-tagged so each model's output lands in a distinct
# results/gemini_<key>/ tree (never conflate two models in one dir).
DEFAULT_MODEL = "3.8-flash"
MODEL_IDS = {"3.8-flash": "gemini-3.8-flash-tts"}

# --- voice pool (rotated per utterance) --------------------------------------
# The 30 prebuilt Gemini voices. They are language-agnostic; Gemini detects the
# spoken language from the text, so the voice pick does not depend on it.
GEMINI_VOICES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe",
    "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir",
    "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Pulcherrima",
    "Puck", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat",
    "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi",
]

# Fallback rate for raw L16 responses without a rate in the mime type (24 kHz,
# 16-bit signed little-endian, mono). 3.8 unary responses are audio/wav instead.
GEMINI_SAMPLE_RATE = 24000


def extract_pcm(resp):
    """Pull the inline audio bytes + mime type out of a generate_content response.

    Returns (data, mime_type); mime_type may be None/'' on SDK paths that omit
    it (see write_audio for how each kind is decoded)."""
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


def write_audio(raw, mime_type, out_path):
    """Write a response's audio as a mono PCM_16 WAV. Returns duration (s).

    audio/wav (RIFF) is decoded as-is; raw L16 (or an unlabelled body) is
    wrapped via pcm_to_wav; anything else raises."""
    mime = (mime_type or "").strip().lower()
    if mime.startswith("audio/wav") or raw[:4] == b"RIFF":
        return bytes_to_wav(raw, out_path)
    if not mime or mime.startswith("audio/l16"):
        return pcm_to_wav(raw, mime_type, out_path)
    raise RuntimeError(f"unexpected audio mime type {mime_type!r}")


def prompt_text(text, style_prompt):
    """The text sent to the model: the target text, optionally style-prefixed."""
    return f"{style_prompt.strip()}: {text}" if style_prompt.strip() else text


def synth(client, model_id, text, voice, style_prompt, retries, base_delay):
    """Run one Gemini TTS synthesis; returns (audio_bytes, mime_type), retried."""
    contents = prompt_text(text, style_prompt)
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


# --- Batch API ------------------------------------------------------------------
BATCH_STATE = "batch_jobs.json"
BATCH_QUOTA_WAITS = 240  # polls to wait out 429 submit refusals (4 h at the default poll)
BATCH_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                  "JOB_STATE_EXPIRED"}


def batch_request(text, voice, style_prompt):
    """One GenerateContentRequest (REST JSON) -- the same request synth() makes."""
    return {
        "contents": [{"parts": [{"text": prompt_text(text, style_prompt)}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }


def _load_jobs(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_jobs(path, jobs):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=1)
    os.replace(tmp, path)


def _delete_file(client, name):
    """Best-effort File API cleanup (input/result files count against 20 GB)."""
    if not name:
        return
    try:
        client.files.delete(name=name)
    except Exception as e:  # noqa: BLE001
        print(f"[batch] could not delete {name}: {format_error(e)}")


def batch_submit(client, model_id, uids, requests_by_uid, out_dir, display, retries,
                 base_delay):
    """Upload one JSONL input file and create a batch job; returns its state entry.

    Creation is not idempotent, so it is retried ONLY on 429 (enqueued-token /
    concurrent-job limits), never on errors that may have created a job."""
    path = os.path.join(out_dir, f".batch_input_{int(time.time() * 1000)}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for uid in uids:
            f.write(json.dumps({"key": uid, "request": requests_by_uid[uid]},
                               ensure_ascii=False) + "\n")
    try:
        up = with_retry(lambda: client.files.upload(
            file=path, config=types.UploadFileConfig(display_name=display,
                                                     mime_type="jsonl")),
            retries, base_delay)
    finally:
        os.unlink(path)
    for attempt in range(retries):
        try:
            job = client.batches.create(
                model=model_id, src=up.name,
                config=types.CreateBatchJobConfig(display_name=display))
            break
        except Exception as e:  # noqa: BLE001
            if http_status_of(e) != 429 or attempt == retries - 1:
                _delete_file(client, up.name)
                raise
            wait = retry_after_of(e) or base_delay * 2 ** (attempt + 2)
            print(f"[batch] create rate-limited (429); retrying in {wait:.0f}s")
            time.sleep(wait)
    return {"name": job.name, "input_file": up.name, "uids": list(uids),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _audio_of(response):
    """(bytes, mime) from a REST-JSON GenerateContentResponse, or raise with the reason."""
    cands = response.get("candidates") or []
    for cand in cands:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return (base64.b64decode(inline["data"]),
                        inline.get("mimeType") or inline.get("mime_type"))
    reason = f" finishReason={cands[0].get('finishReason')}" if cands else ""
    if response.get("promptFeedback"):
        reason += f" promptFeedback={response['promptFeedback']}"
    raise RuntimeError(f"no audio in gemini response{reason}")


def batch_collect(client, entry, out_dir, wav_dir):
    """Write the wavs of one finished job. Returns None while it is still running,
    else (ok, durations, {uid: error}) -- uids absent from the results count as errors."""
    try:
        job = client.batches.get(name=entry["name"])
    except Exception as e:  # noqa: BLE001
        if http_status_of(e) != 404:
            raise
        # Deleted after collection by a run killed before it saved its state:
        # wavs written then are on disk, the rest is resubmitted.
        return 0, [], {u: "error: batch job no longer exists" for u in entry["uids"]
                       if not os.path.exists(os.path.join(wav_dir, u + ".wav"))}
    state = job.state.name if job.state else "JOB_STATE_UNSPECIFIED"
    if state not in BATCH_TERMINAL:
        return None
    errors, durations = {}, []
    result_file = job.dest.file_name if job.dest else None
    if state == "JOB_STATE_SUCCEEDED" and result_file:
        local = os.path.join(out_dir, ".batch_results.jsonl")
        client.files.download(file=result_file, destination=local)
        seen = set()
        with open(local, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = rec.get("key")
                if uid not in entry["uids"]:
                    continue
                seen.add(uid)
                try:
                    if rec.get("response"):
                        raw, mime = _audio_of(rec["response"])
                        durations.append(write_audio(
                            raw, mime, os.path.join(wav_dir, uid + ".wav")))
                    else:
                        raise RuntimeError(f"batch request error: {rec.get('error') or rec.get('status')}")
                except Exception as e:  # noqa: BLE001
                    errors[uid] = format_error(e)
        os.unlink(local)
        for uid in entry["uids"]:
            if uid not in seen:
                errors[uid] = "error: missing from batch results"
    else:
        msg = f"error: batch job {entry['name']} {state}: {job.error}"
        errors = {uid: msg for uid in entry["uids"]}
    _delete_file(client, entry.get("input_file"))
    try:  # also drops the system-owned result file (files.delete rejects its id)
        client.batches.delete(name=entry["name"])
    except Exception as e:  # noqa: BLE001
        print(f"[batch] could not delete {entry['name']}: {format_error(e)}")
    return len(durations), durations, errors


def _mark_serving(out_dir, mode):
    """Record the serving path in gen_meta.json (a split finished by both paths is
    'unary+batch'; meta written before this field existed was unary)."""
    path = os.path.join(out_dir, "gen_meta.json")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    cur = meta.get("serving", "unary")
    if cur in (mode, "unary+batch"):
        return
    meta["serving"] = "unary+batch"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def run_batch(client, model_id, uids, texts, args, wav_dir, label):
    """Batch-API generation of the split's missing wavs (see the module docstring).

    With waiting on, requests that fail inside a job (e.g. the intermittent
    finishReason=OTHER with no audio) are resubmitted for up to --batch-rounds
    rounds, the batch counterpart of the unary path's per-call retries."""
    state_path = os.path.join(args.out_dir, BATCH_STATE)
    jobs = _load_jobs(state_path)

    def has_wav(u):
        return os.path.exists(os.path.join(wav_dir, u + ".wav"))

    wanted = set(uids)
    skipped = {u for u in uids if has_wav(u)} if not args.overwrite else set()
    written, last_error, durations = set(), {}, []

    def sweep():
        for entry in list(jobs):
            res = batch_collect(client, entry, args.out_dir, wav_dir)
            if res is None:
                continue
            n_ok, durs, errs = res
            written.update(u for u in entry["uids"] if u not in errs)
            durations.extend(durs)
            last_error.update(errs)
            jobs.remove(entry)
            _save_jobs(state_path, jobs)
            print(f"[batch] {entry['name']}: {n_ok} ok, {len(errs)} failed")
            if n_ok:
                _mark_serving(args.out_dir, "batch")

    def submit():
        """Submit the missing uids; returns how many were refused with 429."""
        limited = 0
        covered = {u for e in jobs for u in e["uids"]}
        todo = [u for u in uids if u not in covered and u not in written and (
            args.overwrite or not has_wav(u))]
        step = args.batch_max_requests if args.batch_max_requests > 0 else max(1, len(todo))
        out = os.path.normpath(args.out_dir)
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            display = (f"ztts-{os.path.basename(os.path.dirname(out))}-"
                       f"{os.path.basename(out)}-{i // step}")
            try:
                entry = batch_submit(client, model_id, chunk, reqs, args.out_dir,
                                     display, args.retries, args.retry_base_delay)
            except Exception as e:  # noqa: BLE001 -- left missing for the next round/run
                print(f"[batch] submit of {len(chunk)} requests failed: {format_error(e)}")
                last_error.update({u: format_error(e) for u in chunk})
                if http_status_of(e) == 429:
                    limited += len(chunk)
                continue
            jobs.append(entry)
            _save_jobs(state_path, jobs)
            print(f"[batch] submitted {entry['name']} ({len(chunk)} requests)")
        return limited

    reqs = {u: batch_request(texts[u], pick_voice(u, GEMINI_VOICES), args.style_prompt)
            for u in uids}
    sweep()  # collect jobs left by an earlier (interrupted / --batch-no-wait) run
    rounds, quota_waits = max(1, args.batch_rounds), 0
    while rounds > 0:
        limited = submit()
        if not args.batch_wait:
            break
        if limited and not jobs and quota_waits < BATCH_QUOTA_WAITS:
            # Refused on the account's enqueued-token / job limits: wait for other
            # jobs to drain; this does not use up a retry round.
            quota_waits += 1
            time.sleep(args.batch_poll)
            continue
        while jobs:
            time.sleep(args.batch_poll)
            sweep()
        if all(u in written or u in skipped for u in uids):
            break
        rounds -= 1

    pending = {u for e in jobs for u in e["uids"]} & wanted
    failed = sorted(u for u in uids
                    if u not in written and u not in skipped and u not in pending)
    for uid in failed:
        print(f"[gen] {uid}: {last_error.get(uid, 'error: not generated')}")
    ok = len(written & wanted)
    mean_dur = sum(durations) / len(durations) if durations else 0.0
    print(f"done: {ok} ok, {len(skipped)} skipped, {len(failed)} failed"
          f"{f', {len(pending)} pending in batch' if pending else ''} | "
          f"mean audio {mean_dur:.2f}s -> {wav_dir} [{label} batch]")
    if failed or pending:
        print("[warn] re-run the same command to resubmit failures / collect pending jobs "
              "-- finished wavs are skipped.")


def main():
    ap = argparse.ArgumentParser(
        description="Generate TTS audio via native Gemini TTS for a ZTTS2-Eval test set.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_IDS),
                    help=f"Gemini TTS model (default: {DEFAULT_MODEL})")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
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
    ap.add_argument("--batch", action="store_true",
                    help="generate through the Batch API (50%% price, async) instead of "
                         "unary calls")
    ap.add_argument("--batch-no-wait", dest="batch_wait", action="store_false",
                    help="--batch: submit/collect once and exit; pending jobs are "
                         "re-attached by the next run")
    ap.add_argument("--batch-max-requests", type=int, default=2500,
                    help="--batch: requests per job (default 2500 = one job per split)")
    ap.add_argument("--batch-rounds", type=int, default=4,
                    help="--batch: submit rounds per run; requests that failed in a job "
                         "are resubmitted in the next round (default 4)")
    ap.add_argument("--batch-poll", type=float, default=60.0,
                    help="--batch: seconds between job-status polls")
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
    uids = apply_limit(uids, args.limit)
    lang = benchmark_lang(args.data_dir)  # labels voices.txt; never sent

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[gemini:{args.model}] {args.data_dir} -> {wav_dir} | {len(uids)} utterances "
          f"(text={len(texts)}, prompt_wav={len(prompts)}, prompt_text={len(prompt_texts)})")

    # voice manifest -- deterministic, so write it upfront
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            f.write(f"{uid}\t{pick_voice(uid, GEMINI_VOICES)}\t{lang}\n")
    print(f"[gemini:{args.model}] voice assignments -> {vpath}")

    # Kept from the FIRST run of a resumed dir unless --overwrite regenerates.
    if args.overwrite or not os.path.exists(os.path.join(args.out_dir, "gen_meta.json")):
        write_gen_meta(args.out_dir, provider="gemini", model=model_id,
                       endpoint="generateContent", voice_mode="preset", language=lang,
                       language_sent=None, output_format="audio/wav",
                       style_prompt=args.style_prompt or None,
                       serving="batch" if args.batch else "unary")

    if args.batch:
        run_batch(client, model_id, uids, texts, args, wav_dir, f"gemini:{args.model}")
        return

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            raw, mime = synth(client, model_id, texts[uid],
                              pick_voice(uid, GEMINI_VOICES),
                              args.style_prompt, args.retries, args.retry_base_delay)
            dur = write_audio(raw, mime, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    ok, _, _ = run_generate_loop(uids, generate, args.workers,
                                 desc=f"gemini:{args.model}", wav_dir=wav_dir)
    if ok:
        _mark_serving(args.out_dir, "unary")


if __name__ == "__main__":
    main()
