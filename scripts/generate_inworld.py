"""Generate TTS audio via Inworld's REST TTS API for a ZTTS2-Eval test set.

For each utterance in <data-dir>/text, synthesizes the target text with an
Inworld TTS model (via the sync POST /tts/v1/voice endpoint) and writes
<out-dir>/wavs/<uttid>.wav -- the layout run_infer_zonos2_eval.sh expects.

Two voice modes:
  * default (preset): like the Gemini / ElevenLabs engines, uses Inworld's
    built-in (SYSTEM) voices, one per utterance deterministically (recorded in
    voices.txt). Inworld voices are language-specific, so the pool is fetched
    live from GET /voices/v1/voices filtered to source = "SYSTEM" for the target
    language (override with --voices). Speaker similarity is NOT meaningful here.
  * --clone (instant voice cloning): clones each prompt_wav speaker into a
    persistent workspace voice (POST /voices/v1/voices:clone), synthesizes with
    that clone, and records uid->voiceId in clones.txt. This is the real
    zero-shot speaker-similarity benchmark. Clones persist on your account --
    delete them afterward with --delete-clones (reads clones.txt).

Models (--model):
    2        inworld-tts-2         latest generation (default)
    1.5-max  inworld-tts-1.5-max   prior-gen, highest quality
    1.5      inworld-tts-1.5       prior-gen, faster / cheaper
    1        inworld-tts-1         oldest, for comparison

Requires the optional 'inworld' extra and an Inworld Basic auth token:
    uv sync --extra inworld
    export INWORLD_API_KEY=...   # the base64 token after 'Basic ' in the curl

Usage:
    python scripts/generate_inworld.py --model 2 \\
        --data-dir data/zero_shot/en --out-dir results/inworld_2/zero_shot/en \\
        --language en
"""
import argparse
import base64
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import soundfile as sf
from tqdm import tqdm

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    bytes_to_wav, detect_lang, format_error, load_test_set, normalize_lang,
    pick_voice, run_generate_loop, with_retry)


# --- model ids ---------------------------------------------------------------
# Keys are version-tagged so each model's output lands in a distinct
# results/inworld_<key>/ tree (never conflate two models in one dir).
DEFAULT_MODEL = "2"
MODEL_IDS = {
    "2": "inworld-tts-2",
    "1.5-max": "inworld-tts-1.5-max",
    "1.5": "inworld-tts-1.5",
    "1": "inworld-tts-1",
}

# --- endpoints ---------------------------------------------------------------
TTS_URL = "https://api.inworld.ai/tts/v1/voice"          # sync (full audio in one JSON)
VOICES_URL = "https://api.inworld.ai/voices/v1/voices"   # voice catalogue
CLONE_URL = "https://api.inworld.ai/voices/v1/voices:clone"   # instant voice cloning (IVC)

# Inworld langCode (locale) per harness language code, used when cloning a voice.
# The clone API requires a full locale (e.g. EN_US), not the bare lang code.
LOCALE_FOR = {
    "en": "EN_US", "zh": "ZH_CN", "ja": "JA_JP", "ko": "KO_KR", "de": "DE_DE",
    "es": "ES_ES", "fr": "FR_FR", "it": "IT_IT", "ru": "RU_RU", "ar": "AR_SA",
    "hi": "HI_IN", "id": "ID_ID", "pl": "PL_PL", "pt": "PT_BR", "th": "TH_TH",
    "tr": "TR_TR", "nl": "NL_NL",
}

# Inworld trims clone samples to 15 s and caps the upload at 4 MB; we trim here too.
CLONE_MAX_SECONDS = 15.0

# Fallback English SYSTEM voices, used only if the voices API call fails or
# returns nothing (so a run never hard-fails on voice discovery). Verify these
# against your account if you rely on the fallback.
DEFAULT_VOICES = [
    "Ashley", "Dennis", "Edward", "Elizabeth", "Mark",
    "Olivia", "Sarah", "Theodore", "Timothy", "Wendy",
]

# Request MP3 (always available); decoded to PCM_16 WAV for the scorers.
AUDIO_ENCODING = "MP3"


def _voice_name(v):
    """Pull a usable voice_id out of a voices-API record (key order = preference)."""
    for k in ("displayName", "display_name", "voiceId", "voice_id", "name"):
        val = v.get(k)
        if val:
            # 'name' can be a resource path like 'voices/Sarah'; keep the leaf.
            return str(val).rsplit("/", 1)[-1]
    return None


def list_system_voices(headers, lang, timeout):
    """Fetch the sorted SYSTEM voice names for one language from the voices API.

    Pages through GET /voices/v1/voices filtered to source = "SYSTEM" and the
    given lang_code. Returns [] on any error (caller falls back to DEFAULT_VOICES)."""
    voices = []
    page_token = ""
    try:
        while True:
            params = {
                "filter": f'source = "SYSTEM" AND lang_code = "{lang}"',
                "orderBy": "display_name asc",
                "pageSize": 100,
            }
            if page_token:
                params["pageToken"] = page_token
            r = requests.get(VOICES_URL, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            for v in body.get("voices", []):
                name = _voice_name(v)
                if name:
                    voices.append(name)
            page_token = body.get("nextPageToken") or ""
            if not page_token:
                break
    except Exception as e:  # noqa: BLE001
        print(f"[warn] voices API lookup for lang='{lang}' failed: {e}")
        return []
    # De-dup while preserving the API's sort order.
    seen = set()
    return [v for v in voices if not (v in seen or seen.add(v))]


def synth(headers, model_id, text, voice_id, timeout, retries, base_delay):
    """Run one Inworld synthesis and return the result audio bytes, retried."""
    payload = {
        "text": text,
        "voice_id": voice_id,
        "audio_config": {"audio_encoding": AUDIO_ENCODING},
        "model_id": model_id,
    }

    def _call():
        r = requests.post(TTS_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        b64 = body.get("audioContent") or body.get("result", {}).get("audioContent")
        if not b64:
            raise RuntimeError(f"no audioContent in inworld response: {str(body)[:200]}")
        return base64.b64decode(b64)

    return with_retry(_call, retries, base_delay)


# --- instant voice cloning (IVC) ---------------------------------------------
# Unlike the preset-voice path above, these clone the prompt speaker so Inworld
# can be run on the real zero-shot speaker-similarity benchmark. Each prompt_wav
# is cloned once into a persistent workspace voice (a voiceId), reused for that
# utterance's synthesis, and removed afterward with --delete-clones.

def locale_for(lang):
    """Map a harness lang code (en/zh/...) to an Inworld langCode (EN_US/...)."""
    lang = (lang or "").lower()
    if lang in LOCALE_FOR:
        return LOCALE_FOR[lang]
    up = lang.upper()
    return f"{up}_{up}" if up else "EN_US"


def clone_display_name(out_dir, uid):
    """A workspace-unique, valid displayName for a clone of `uid` in `out_dir`.

    The voiceId is workspaces/<ws>/voices/<displayName>, so displayName must be
    unique across the whole workspace -- the same uttid_1 recurs in en/de/fr/...
    splits, so we namespace by a short hash of the output dir."""
    tag = hashlib.md5(out_dir.encode("utf-8")).hexdigest()[:6]
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in uid)
    return f"zse_{tag}_{safe}"


def wav_to_clone_b64(path, max_seconds=CLONE_MAX_SECONDS):
    """Load a reference wav, downmix to mono, trim to max_seconds, return base64 WAV."""
    import io
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if max_seconds and data.size > int(max_seconds * sr):
        data = data[: int(max_seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def clone_voice(headers, display_name, lang_code, audio_b64, transcription,
                timeout, retries, base_delay):
    """Clone one voice from a base64 WAV sample; return its persistent voiceId.

    The clone endpoint rate-limits aggressively (429); this retries more times
    than a normal call, and the shared with_retry honors the server's
    Retry-After on 429 so a resume pass rides out the limit instead of giving up."""
    sample = {"audioData": audio_b64}
    if transcription:
        sample["transcription"] = transcription
    payload = {
        "displayName": display_name,
        "langCode": lang_code,
        "voiceSamples": [sample],
    }
    # 429s need patience, not just more workers -- give clones extra attempts.
    attempts = max(retries, 8)

    def _post(denoise):
        body = dict(payload)
        if denoise:
            body["audioProcessingConfig"] = {"removeBackgroundNoise": True}
        r = requests.post(CLONE_URL, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        b = r.json()
        voice = b.get("voice") or {}
        return voice.get("voiceId") or b.get("voiceId"), b

    def _call():
        vid, b = _post(denoise=False)
        if not vid:
            # 200 but voice:null = sample validation failed (e.g. Inworld's internal
            # ASR "detected no speech"). Denoising the sample often rescues it.
            vid, b2 = _post(denoise=True)
            if not vid:
                raise RuntimeError(f"clone sample rejected (even with denoise): {str(b)[:200]}")
        return vid

    return with_retry(_call, attempts, base_delay)


def delete_voice(headers, voice_id, timeout):
    """Delete a cloned workspace voice. Returns True if gone (200 or already 404)."""
    r = requests.delete(f"{VOICES_URL}/{voice_id}", headers=headers, timeout=timeout)
    if r.status_code in (200, 204, 404):
        return True
    r.raise_for_status()
    return True


def build_preset_voices(headers, uids, texts, forced_lang, args):
    """Assign a preset SYSTEM voice to each uid (deterministic), write voices.txt.

    --voices override wins; otherwise fetch SYSTEM voices for each language we
    actually need (one fetch per distinct lang, cached). Returns {uid: voice}."""
    override_pool = [v.strip() for v in args.voices.split(",") if v.strip()] or None
    pool_cache = {}

    def pool_for(lang):
        if override_pool:
            return override_pool
        if lang not in pool_cache:
            pool = list_system_voices(headers, lang, args.timeout)
            if not pool:
                print(f"[warn] no SYSTEM voices for lang='{lang}'; "
                      f"falling back to {DEFAULT_VOICES}")
                pool = DEFAULT_VOICES
            pool_cache[lang] = pool
        return pool_cache[lang]

    voice_for = {}
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            lg = forced_lang or detect_lang(texts[uid])
            voice = pick_voice(uid, pool_for(lg))
            voice_for[uid] = voice
            f.write(f"{uid}\t{voice}\t{lg}\n")
    print(f"[inworld:{args.model}] voice assignments -> {vpath}")
    return voice_for


class RateLimiter:
    """Thread-safe token bucket: paces clone calls to the server's sustained rate.

    The CloneVoice endpoint is capped at ~5 requests/minute/workspace with a small
    burst bucket (probed empirically; the limit is only reported inside the 429
    body, never in a header). Pacing to that rate up front means nearly every
    request succeeds, so we avoid the 429 churn + resume passes that bursting
    incurs. `capacity` allows the initial burst; `rate` is tokens/second."""

    def __init__(self, rate_per_sec, capacity):
        import threading
        self.rate = rate_per_sec
        self.capacity = max(1.0, capacity)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


def _list_clone_vids(headers, out_dir, timeout):
    """List live IVC voiceIds whose displayName carries this out-dir's clone tag.

    A clean run deletes every clone it makes, so any match is an orphan from a
    crashed/interrupted run -- safe to sweep. Raises on listing/HTTP errors
    (a 401/403 error body would otherwise read as an empty voice list and
    make cleanup report 'no clones to delete' while the slots leak) -- callers
    decide whether that is best-effort (_sweep_orphans) or fatal (--delete-clones)."""
    prefix = clone_display_name(out_dir, "")  # e.g. 'zse_<tag>_'
    vids, page = [], ""
    while True:
        params = {"pageSize": 1000}
        if page:
            params["pageToken"] = page
        r = requests.get(VOICES_URL, headers=headers, params=params,
                         timeout=timeout)
        r.raise_for_status()
        b = r.json()
        for v in b.get("voices", []):
            if v.get("source") == "IVC" and \
                    str(v.get("displayName", "")).startswith(prefix):
                vids.append(v.get("voiceId"))
        page = b.get("nextPageToken") or ""
        if not page:
            break
    return [x for x in vids if x]


def _sweep_orphans(headers, out_dir, timeout):
    """Delete clones left live by a prior crash (matched by this out-dir's tag)."""
    try:
        vids = _list_clone_vids(headers, out_dir, timeout)
    except Exception as e:  # noqa: BLE001 - startup sweep is best-effort
        print(f"[warn] voice listing failed (orphan sweep skipped): {e}")
        return
    if vids:
        print(f"[inworld:clone] sweeping {len(vids)} orphaned clone(s) from a prior run")
        for vid in vids:
            _safe_delete(headers, vid, timeout)


def run_clone_mode(headers, model_id, uids, texts, prompts, prompt_texts,
                   forced_lang, wav_dir, args):
    """Rate-paced streaming instant-voice-cloning: clone -> synth -> delete per utt.

    Each clone is gated through a shared token bucket paced to the workspace's
    CloneVoice rate limit (~5/min), so requests almost never 429. The clone is
    deleted immediately after its synthesis, so only a handful are ever live at
    once -- the 100-voice plan cap is a non-issue and no batching is needed.
    Resumable on wav existence; crash-safe via a tag-prefixed orphan sweep at
    start. A pending ledger records currently-live clones for observability."""
    import threading

    _sweep_orphans(headers, args.out_dir, args.timeout)  # clean prior crash
    pending_path = os.path.join(args.out_dir, "pending_clones.txt")
    open(pending_path, "w").close()  # fresh ledger for this run

    todo = [u for u in uids
            if args.overwrite or not os.path.exists(os.path.join(wav_dir, u + ".wav"))]
    done_already = len(uids) - len(todo)
    limiter = RateLimiter(args.clone_rate_per_min / 60.0, args.clone_burst)
    print(f"[inworld:clone] {len(todo)} to do ({done_already} wavs already present), "
          f"rate={args.clone_rate_per_min}/min, burst={args.clone_burst}, "
          f"workers={args.clone_workers}")

    plock = threading.Lock()

    def process(uid):
        # 1. clone (rate-limited) -------------------------------------------
        lg = forced_lang or detect_lang(texts[uid])
        name = clone_display_name(args.out_dir, uid)
        try:
            limiter.acquire()
            b64 = wav_to_clone_b64(prompts[uid], args.clone_max_seconds)
            vid = clone_voice(headers, name, locale_for(lg), b64,
                              prompt_texts.get(uid, ""), args.timeout,
                              args.retries, args.retry_base_delay)
        except Exception as e:  # noqa: BLE001
            return uid, "clone_fail", f"{type(e).__name__}: {e}"
        with plock:  # record the live clone before synth so a crash can sweep it
            with open(pending_path, "a", encoding="utf-8") as pf:
                pf.write(f"{vid}\t{uid}\n")
        # 2. synth with the clone -------------------------------------------
        try:
            raw = synth(headers, model_id, texts[uid], vid,
                        args.timeout, args.retries, args.retry_base_delay)
            bytes_to_wav(raw, os.path.join(wav_dir, uid + ".wav"))
            status, err = "ok", None
        except Exception as e:  # noqa: BLE001
            status, err = "synth_fail", f"{type(e).__name__}: {e}"
        # 3. delete the clone immediately (frees the cap + keeps account clean)
        _safe_delete(headers, vid, args.timeout)
        return uid, status, err

    ok = fail_clone = fail_synth = 0
    with ThreadPoolExecutor(max_workers=args.clone_workers) as ex:
        for uid, status, err in tqdm(ex.map(process, todo), total=len(todo),
                                     desc="inworld:clone"):
            if status == "ok":
                ok += 1
            elif status == "clone_fail":
                fail_clone += 1
                print(f"[clone] {uid}: {err}")
            else:
                fail_synth += 1
                print(f"[synth] {uid}: {err}")

    if os.path.exists(pending_path):
        os.remove(pending_path)
    print(f"[inworld:clone] done: {ok} ok, {fail_synth} synth-failed, "
          f"{fail_clone} clone-failed (of {len(todo)}) -> {wav_dir}")
    if fail_clone or fail_synth:
        print(f"[warn] re-run the same command to retry the {fail_clone + fail_synth} "
              f"missing utterances (finished wavs are skipped).")


def _safe_delete(headers, vid, timeout):
    try:
        delete_voice(headers, vid, timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[del] {vid}: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate TTS audio via the Inworld REST API for a ZTTS2-Eval test set.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_IDS),
                    help=f"Inworld TTS model (default: {DEFAULT_MODEL})")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_text', 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--language", default="",
                    help="target language code (en/zh/ja/ko/...); selects which "
                         "SYSTEM voice pool is fetched and labels voices.txt")
    ap.add_argument("--voices", default="",
                    help="comma-separated voice override (e.g. 'Sarah,Mark'); "
                         "bypasses the voices-API lookup")
    ap.add_argument("--clone", action="store_true",
                    help="instant voice cloning: clone each prompt_wav speaker and "
                         "synthesize with the clone (real zero-shot spk-sim), instead "
                         "of using preset SYSTEM voices")
    ap.add_argument("--clone-max-seconds", type=float, default=CLONE_MAX_SECONDS,
                    help="trim each clone sample to this many seconds (Inworld caps "
                         "at 15s / 4MB)")
    ap.add_argument("--clone-rate-per-min", type=float, default=5.0,
                    help="clone requests/minute, paced by a token bucket to the "
                         "workspace's CloneVoice rate limit (~5/min) to avoid 429s")
    ap.add_argument("--clone-burst", type=float, default=6.0,
                    help="initial burst the rate bucket allows before throttling "
                         "to --clone-rate-per-min")
    ap.add_argument("--clone-workers", type=int, default=4,
                    help="pipeline concurrency (clone+synth+delete overlap); the "
                         "rate limiter caps the clone rate regardless")
    ap.add_argument("--delete-clones", action="store_true",
                    help="cleanup mode: delete every cloned voice listed in "
                         "<out-dir>/clones.txt, then exit (no synthesis)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--workers", type=int, default=4, help="concurrent inworld requests")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per inworld call")
    ap.add_argument("--retry-base-delay", type=float, default=2.0,
                    help="base backoff delay between retries, in seconds")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    api_key = os.environ.get("INWORLD_API_KEY")
    if not api_key:
        sys.exit("error: INWORLD_API_KEY environment variable is not set "
                 "(the base64 token after 'Basic ' in the curl)")

    model_id = MODEL_IDS[args.model]
    headers = {"Authorization": f"Basic {api_key}", "Content-Type": "application/json"}

    # --- cleanup mode: delete every clone associated with this out-dir --------
    # Batched runs delete their clones automatically; this is a safety net. It
    # sweeps the pending ledger AND any IVC voice whose displayName carries this
    # out-dir's tag (so a crashed/interrupted run leaves nothing on the account).
    if args.delete_clones:
        try:
            vids = set(_list_clone_vids(headers, args.out_dir, args.timeout))
        except Exception as e:  # noqa: BLE001
            # Cleanup mode must not report 'no clones to delete' on a failed
            # listing (e.g. 401/403) -- the account slots would silently leak.
            sys.exit(f"error: voice listing failed ({e}); cannot verify clones "
                     f"for {args.out_dir}")
        # also honor the transient pending ledger + any legacy clones.txt --
        # the two ledgers have opposite column orders, so key on the FILENAME
        # (a string heuristic on the id misclassifies vids without '__' and the
        # 404-tolerant delete would then report the real clone as deleted).
        for fn in ("pending_clones.txt", "clones.txt"):
            p = os.path.join(args.out_dir, fn)
            if not os.path.exists(p):
                continue
            for ln in open(p, encoding="utf-8"):
                parts = ln.strip().split("\t")
                if not parts or not parts[0]:
                    continue
                if fn == "pending_clones.txt":   # vid-first ledger
                    vids.add(parts[0])
                elif len(parts) > 1:             # clones.txt is uid-first
                    vids.add(parts[1])
                else:
                    print(f"[warn] {fn}: single-column line {parts[0]!r} -- "
                          f"cannot tell voiceId from uid, skipping")
        vids = sorted(x for x in vids if x)
        if not vids:
            print(f"[inworld:clone] no clones to delete for {args.out_dir}")
            return
        print(f"[inworld:clone] deleting {len(vids)} cloned voices for {args.out_dir}")
        gone = failed = 0
        for vid in tqdm(vids, desc="delete-clones"):
            try:
                delete_voice(headers, vid, args.timeout)
                gone += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[del] {vid}: {e}")
        for fn in ("pending_clones.txt",):
            p = os.path.join(args.out_dir, fn)
            if os.path.exists(p) and not failed:
                os.remove(p)
        print(f"done: {gone} deleted, {failed} failed")
        return

    # Intersect so the uttid set matches the other engines (each test-set dir
    # has all three files); Inworld itself only needs the target text.
    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    if args.limit > 0:
        uids = uids[:args.limit]

    forced_lang = normalize_lang(args.language)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[inworld:{args.model}] {args.data_dir} -> {wav_dir} | {len(uids)} utterances "
          f"(text={len(texts)}, prompt_wav={len(prompts)}, prompt_text={len(prompt_texts)})")

    if args.clone:
        run_clone_mode(headers, model_id, uids, texts, prompts, prompt_texts,
                       forced_lang, wav_dir, args)
        return

    voice_for = build_preset_voices(headers, uids, texts, forced_lang, args)

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            raw = synth(headers, model_id, texts[uid], voice_for[uid],
                        args.timeout, args.retries, args.retry_base_delay)
            dur = bytes_to_wav(raw, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    run_generate_loop(uids, generate, args.workers,
                      desc=f"inworld:{args.model}", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
