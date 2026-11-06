"""Shared harness for the generation clients (scripts/generate_*.py).

Consolidates the helpers that were previously copy-pasted (and drifting) across
the API/server clients -- language detection, retry/backoff, audio decode +
atomic WAV writes, deterministic voice rotation, the test-set triple loader --
plus the common generate loop (progress bar, ok/skip/fail summary, error
logging, retry hint).

IMPORT RULE: the native in-process clients (kokoro, moss_tts, neutts,
cosyvoice, vibevoice, vibevoice15) run in their OWN venvs, so this module must
import nothing beyond the stdlib at module import time. Third-party imports
(numpy / soundfile / librosa / tqdm) are deferred into the functions that need
them.
"""
import hashlib
import os
import random
import re
import sys
import time

try:
    from .scp import load_scp
except ImportError:  # imported with utils/ itself on sys.path
    from scp import load_scp  # type: ignore


# --- language helpers --------------------------------------------------------

def detect_lang(text):
    """Best-effort target-language detection by Unicode script.

    Hangul -> ko, Kana -> ja, Han -> zh, otherwise en. Kana is checked before
    Han because Japanese text mixes the two."""
    has_hangul = has_kana = has_han = False
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            has_hangul = True
        elif 0x3040 <= o <= 0x30FF:
            has_kana = True
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            has_han = True
    if has_hangul:
        return "ko"
    if has_kana:
        return "ja"
    if has_han:
        return "zh"
    return "en"


def normalize_lang(s):
    """Map a --language value (or a test-set name like 'to_hard_zh') to en/zh/ja/ko."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = s.replace("to_", "").replace("hard_", "")
    if s.startswith("zh") or s in ("cmn", "mandarin", "chinese"):
        return "zh"
    if s.startswith("en") or s == "english":
        return "en"
    if s.startswith("ja") or s in ("jp", "japanese"):
        return "ja"
    if s.startswith("ko") or s == "korean":
        return "ko"
    return s


def benchmark_lang(data_dir):
    """The benchmark language of a test-set dir (its basename); exits if it is not
    in the frozen Clean / ITW baskets."""
    try:
        from .benchmark import CLEAN_LANGS, ITW_LANGS
    except ImportError:  # imported with utils/ itself on sys.path
        from benchmark import CLEAN_LANGS, ITW_LANGS  # type: ignore
    lang = os.path.basename(os.path.normpath(data_dir))
    if lang not in set(CLEAN_LANGS) | set(ITW_LANGS):
        sys.exit(f"error: {lang!r} is not a benchmark language (data dir {data_dir!r})")
    return lang


# --- retry / backoff ---------------------------------------------------------

_RETRYABLE_4XX = (408, 429)  # request timeout / rate limit: retrying CAN help


def http_status_of(exc):
    """Best-effort HTTP status code of an exception, or None.

    Duck-typed across the clients' error shapes: requests.HTTPError
    (.response.status_code), SDK errors (.status_code), urllib.error.HTTPError
    and google-genai APIError (.code), and RuntimeError('HTTP 400: ...') as
    raised by generate_omni's http_post."""
    resp = getattr(exc, "response", None)
    st = getattr(resp, "status_code", None)
    if isinstance(st, int):
        return st
    st = getattr(exc, "status_code", None)
    if isinstance(st, int):
        return st
    st = getattr(exc, "code", None)
    if isinstance(st, int):
        return st
    m = re.match(r"HTTP (\d{3})\b", str(exc))
    if m:
        return int(m.group(1))
    return None


def retry_after_of(exc):
    """The exception's Retry-After header in seconds, or None.

    Only the delta-seconds form is parsed; the HTTP-date form is ignored."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        ra = headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - exotic header containers
        return None
    if not ra:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


def is_permanent(exc):
    """True for HTTP 4xx client errors that retrying cannot fix (except 408/429)."""
    st = http_status_of(exc)
    return st is not None and 400 <= st < 500 and st not in _RETRYABLE_4XX


def with_retry(fn, attempts, base_delay):
    """Call fn(), retrying transient errors with exponential backoff + jitter.

    Permanent HTTP 4xx errors (except 408/429) are re-raised immediately -- a
    bad API key or rejected payload must not burn the whole backoff schedule
    per utterance. On 429 the server's Retry-After is honored (capped at 60 s)
    when it is longer than the backoff. Re-raises the last error."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network / SDK / rate-limit errors are varied
            if is_permanent(e):
                raise
            last = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i) + random.uniform(0, base_delay)
                if http_status_of(e) == 429:
                    ra = retry_after_of(e)
                    if ra is not None:
                        delay = max(delay, min(ra, 60.0))
                time.sleep(delay)
    raise last


def format_error(e):
    """Standard '[gen]' error string: always includes the exception TYPE.

    Several real failures stringify to '' (61/403 of a voxblink/ko cosyvoice
    run once logged only '[gen] uttid_N: error: ' with nothing to diagnose)."""
    return f"error: {type(e).__name__}: {e}"


# --- deterministic voice rotation --------------------------------------------

def pick_index(uttid, n):
    """Deterministic per-utterance index into a pool of size n (uid-hash based,
    so resumes / --limit / --stratified-bins changes never reassign voices)."""
    return int(hashlib.md5(uttid.encode("utf-8")).hexdigest(), 16) % n


def pick_voice(uttid, pool):
    """Deterministic per-utterance voice rotation over the given pool."""
    return pool[pick_index(uttid, len(pool))]


# --- audio decode / atomic WAV writes ----------------------------------------

def decode_audio(raw):
    """Decode audio bytes (wav/mp3/flac/...) to a mono float32 array + sample rate."""
    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tf:
        tf.write(raw)
        tmp = tf.name
    try:
        try:  # soundfile: wav/flac/ogg (and mp3 with libsndfile >= 1.1)
            data, sr = sf.read(tmp, dtype="float32", always_2d=True)
            return data.mean(axis=1), int(sr)
        except Exception:
            pass
        try:  # librosa: audioread / ffmpeg backend -> mp3, m4a, ...
            import librosa
            data, sr = librosa.load(tmp, sr=None, mono=True)
            if data.size:
                return data.astype("float32"), int(sr)
        except Exception:
            pass
        import torchaudio  # last resort: torchcodec / ffmpeg
        wav, sr = torchaudio.load(tmp)
        return wav.mean(dim=0).cpu().numpy().astype("float32"), int(sr)
    finally:
        os.unlink(tmp)


def save_wav_atomic(data, sr, out_path, subtype="PCM_16"):
    """Write mono float32 samples as a WAV atomically (tmp + os.replace).

    Never leaves a half-written .wav behind on SIGKILL/preemption, so the
    skip-existing resume check can trust any file it finds. Returns duration (s)."""
    import numpy as np
    import soundfile as sf

    data = np.asarray(data, dtype="float32").reshape(-1)
    if data.size == 0:
        raise RuntimeError("decoded audio is empty")
    tmp = out_path + ".tmp.wav"
    sf.write(tmp, data, sr, subtype=subtype)
    os.replace(tmp, out_path)  # atomic: never leave a half-written .wav
    return data.size / sr


def bytes_to_wav(raw, out_path):
    """Decode response audio bytes and write a mono PCM_16 WAV. Returns duration (s).

    MP3/other results are re-encoded to true PCM so the scorers' torchaudio.load()
    can read them. Native sample rate is kept (the scorers resample themselves)."""
    data, sr = decode_audio(raw)
    return save_wav_atomic(data, sr, out_path)


def parse_l16_rate(mime_type, default_rate):
    """Validate an inline-audio mime type as raw L16 PCM and return its rate.

    Gemini TTS returns e.g. 'audio/L16;codec=pcm;rate=24000'. An empty/missing
    mime keeps default_rate (some SDK paths omit it); anything that is not
    audio/L16 raises RuntimeError so a silent API format change cannot fill a
    run with valid-looking wrong-speed audio."""
    if not mime_type:
        return int(default_rate)
    parts = [p.strip() for p in str(mime_type).split(";")]
    if parts[0].lower() != "audio/l16":
        raise RuntimeError(f"unexpected audio mime type {mime_type!r} "
                           f"(expected audio/L16 raw PCM)")
    rate = int(default_rate)
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            if k.strip().lower() == "rate":
                try:
                    rate = int(v.strip())
                except ValueError:
                    raise RuntimeError(f"bad rate in audio mime type {mime_type!r}")
    return rate


# --- test-set loading --------------------------------------------------------

def load_test_set(data_dir, need_prompt_wav=True, need_prompt_text=True):
    """Load the text / prompt_wav.scp / prompt_text triple and intersect uids.

    Returns (texts, prompts, prompt_texts, uids). Files a caller does not need
    are still loaded when present (so coverage counts can be printed), but only
    the needed ones gate the uid intersection -- preset-voice engines pass
    need_*=True anyway so their uid set matches the cloning engines. Exits with
    an error when 'text' is missing or the intersection is empty."""
    text_path = os.path.join(data_dir, "text")
    if not os.path.exists(text_path):
        sys.exit(f"error: no 'text' file in {data_dir}")
    texts = load_scp(text_path)
    pw_path = os.path.join(data_dir, "prompt_wav.scp")
    pt_path = os.path.join(data_dir, "prompt_text")
    prompts = load_scp(pw_path) if os.path.exists(pw_path) else {}
    prompt_texts = load_scp(pt_path) if os.path.exists(pt_path) else {}
    uids = [u for u in texts
            if (not need_prompt_wav or u in prompts)
            and (not need_prompt_text or u in prompt_texts)]
    if not uids:
        need = ["text"] + (["prompt_wav"] if need_prompt_wav else []) \
            + (["prompt_text"] if need_prompt_text else [])
        sys.exit(f"error: no utterances with {' + '.join(need)} in {data_dir}")
    return texts, prompts, prompt_texts, uids


def apply_limit(uids, limit=0, stratified_bins=0):
    """--limit / --stratified-bins selection.

    stratified_bins > 0 picks K evenly-spaced utts across the full set (useful
    when the set is sorted by quality bin -- e.g. voxblink is 5 MSR-UTMOS bins
    per lang); otherwise limit > 0 keeps the first N."""
    uids = list(uids)
    if stratified_bins and 0 < stratified_bins < len(uids):
        k, n = stratified_bins, len(uids)
        return [uids[round(i * n / k)] for i in range(k)]
    if limit and limit > 0:
        return uids[:limit]
    return uids


# --- generation metadata ------------------------------------------------------

def _sanitize_argv(argv):
    """Copy of an argv with the VALUES of secret-bearing options redacted.

    Any option whose name contains 'key' or 'token' (--api-key, --hf-token,
    --api_key=..., ...) keeps its name but has its value replaced with '***',
    in both the two-token and the --name=value forms."""
    out = []
    redact_next = False
    for a in argv:
        if redact_next:
            out.append("***")
            redact_next = False
            continue
        if a.startswith("-"):
            name = a.split("=", 1)[0].lower()
            if "key" in name or "token" in name:
                if "=" in a:
                    out.append(a.split("=", 1)[0] + "=***")
                else:
                    out.append(a)
                    redact_next = True
                continue
        out.append(a)
    return out


def _strip_url_credentials(url):
    """Drop the userinfo (user:password@) part of a URL, if any."""
    from urllib.parse import urlsplit, urlunsplit
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if "@" not in parts.netloc:
        return url
    netloc = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query,
                       parts.fragment))


def write_gen_meta(out_dir, **fields):
    """Atomically write <out_dir>/gen_meta.json describing this generation run.

    Always records generated_at (UTC ISO), client (basename of sys.argv[0]),
    and the sanitized command line (values of options whose names contain
    'key'/'token' are redacted); callers add their own fields (model id, seed,
    temperature, base_url, ...). String fields containing a URL get their
    userinfo credentials stripped. Forward-looking provenance for future runs
    -- existing results dirs are not backfilled. Returns the file path."""
    import json
    from datetime import datetime, timezone

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "",
        "argv": _sanitize_argv(sys.argv[1:]),
    }
    for k, v in fields.items():
        if isinstance(v, str) and "://" in v:
            v = _strip_url_credentials(v)
        meta[k] = v
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gen_meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    os.replace(tmp, path)  # atomic: never leave a half-written meta file
    return path


# --- shared generate loop ----------------------------------------------------

def run_generate_loop(uids, generate, workers, desc, wav_dir):
    """The common generation loop: progress bar + ok/skip/fail summary + hint.

    generate(uid) must return (uid, status, duration_s) where status is "ok",
    "skip", or an error string (logged as '[gen] <uid>: <status>' -- build error
    strings with format_error() so the exception type is never lost). workers >
    1 maps utterances over a ThreadPoolExecutor (API/server clients); workers <=
    1 runs sequentially in the caller's thread (in-process GPU engines).
    Also drops a gen_meta.json provenance stub next to wav_dir (in its parent
    when wav_dir is the conventional <out_dir>/wavs) unless one already exists
    -- clients with richer knowledge (model id, sampling params) should call
    write_gen_meta() themselves BEFORE this loop. Returns (ok, skip, fail)."""
    from tqdm import tqdm

    meta_dir = os.path.dirname(os.path.abspath(wav_dir)) \
        if os.path.basename(os.path.normpath(wav_dir)) == "wavs" else wav_dir
    if os.path.isdir(wav_dir) and \
            not os.path.exists(os.path.join(meta_dir, "gen_meta.json")):
        try:
            write_gen_meta(meta_dir)
        except OSError as e:
            print(f"[warn] gen_meta.json not written: {e}")

    ok = fail = skip = 0
    durations = []

    def consume(results):
        nonlocal ok, fail, skip
        for uid, status, dur in results:
            if status == "ok":
                ok += 1
                durations.append(dur)
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[gen] {uid}: {status}")

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            consume(tqdm(ex.map(generate, uids), total=len(uids), desc=desc))
    else:
        consume(tqdm((generate(u) for u in uids), total=len(uids), desc=desc))

    mean_dur = sum(durations) / len(durations) if durations else 0.0
    print(f"done: {ok} ok, {skip} skipped, {fail} failed | "
          f"mean audio {mean_dur:.2f}s -> {wav_dir}")
    if fail:
        print(f"[warn] {fail} failed; scoring counts a missing wav as maximum error "
              f"(utils/run_wer.py), so re-run to retry before scoring -- finished "
              f"wavs are skipped.")
    return ok, skip, fail
