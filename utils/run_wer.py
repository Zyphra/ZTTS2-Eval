import sys, os, json, time, zlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI, BadRequestError

# Text prep is applied identically to REF and HYP, in utils/text_norm.py: apostrophes
# deleted, lowercase, Unicode punctuation -> space; zh/ja/ko/th character-tokenized
# (CER); zh folds traditional -> simplified on both sides. On top of that, abjad
# languages (ar/fa/he, via ORTHO_NORM_LANGS) get a
# pronunciation-neutral orthographic fold -- strip Arabic harakat / Persian ZWNJ /
# Hebrew niqqud, unify hamza-carriers, ta-marbuta, Arabic<->Persian yeh/kaf -- since
# a large share of their GT WER tail is spelling convention, not mispronunciation
# (see docs/edit_nature.md). Apostrophes are deleted (not spaced) because
# contractions are orthography, not pronunciation.
from text_norm import base_lang, prepare, process_one  # noqa: E402,F401

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3]  # zh / en / hard_zh / to_en / ...
base = base_lang(lang)

# ASR is served by vLLM via the OpenAI-compatible transcription endpoint. The
# server(s) are started/stopped by the driver (e.g. run_infer_zonos2_eval.sh); here
# we are only a client. Qwen3-ASR is the default backend.
BASE_URL = os.environ.get("ASR_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
MAX_WORKERS = int(os.environ.get("ASR_CONCURRENCY", "16"))

# Transient-failure policy: a request that times out or errors is retried
# ASR_RETRIES more times (linear backoff) before the utterance is recorded as
# UNSCORED in the `<res_path>.meta` sidecar; average_wer.py folds that sidecar
# into the .wer N_SCORED:/N_EXPECTED: coverage footer and exits non-zero below
# SCORE_MIN_COVERAGE. Timeouts correlate with degenerate TTS audio, so silently
# dropping them biased the survivor mean in favor of the most broken systems.
ASR_RETRIES = int(os.environ.get("ASR_RETRIES", "2"))
ASR_RETRY_DELAY = float(os.environ.get("ASR_RETRY_DELAY", "2.0"))

# --- Unscoreable generations are scored as MAXIMUM ERROR, not excluded -----------
# 2026-09-01. Recording an utterance as UNSCORED still leaves it out of the mean,
# which makes every leaf mean a SURVIVOR mean: the more degenerate a system's
# output, the more of it disappears from its own score. Measured across the whole
# results tree before this change, all 4197 unscored utterances had their generated
# wav present on disk -- none was a missing-generation -- and the failure was always
# the transcriber running away on the audio (733 recorded APITimeoutError vs 9
# BadRequestError in the leaves that carried sidecars). Raising the timeout does not
# recover them: 60 s -> 300 s moved Hindi coverage 0.973 -> 0.971 (docs/controls_2026-08.md).
# So "the reference ASR cannot transcribe this at all" is a property of the
# generation, and the honest score for it is the worst score the metric can express.
#
# An imputed utterance is written into the .raw as a normal row with wer = 1.0
# (WER_CLIP 100) and every reference token deleted, but with the hypothesis column
# replaced by an `<UNSCOREABLE:Reason>` sentinel so that no consumer can mistake it
# for a transcript: average_wer.py counts them into the N_IMPUTED: footer, and
# analyses over hypothesis text (scripts/classify_edits.py) can filter them out.
#
# Only failures attributable to the AUDIO are imputed. Infrastructure failures --
# a dead engine, a refused connection -- are not the system's fault, so they stay
# unscored and keep counting against coverage, where SCORE_MIN_COVERAGE fails the
# leaf. That asymmetry is deliberate: imputing an infrastructure failure would
# silently publish a harness outage as a model result.
#
# WER_IMPUTE=0 restores pure exclusion (used to reproduce a pre-2026-09-01 number).
WER_IMPUTE = os.environ.get("WER_IMPUTE", "1") != "0"
IMPUTE_REASONS = ("missing wav", "APITimeoutError", "BadRequestError")
# A 400 whose detail is a request-parsing failure is the server failing to read
# the upload -- its multipart spool dir on a full disk, for one -- not a verdict
# on the audio. It is a harness fault: retried, never imputed, and never read as
# the backend rejecting the language code. Imputing it once published a node's
# full /tmp as 23% of an eval failing at maximum error.
UPLOAD_PARSE_ERROR = "error parsing the body"


def imputable(reason):
    """Is this failure a property of the generated audio (vs of the harness)?"""
    return (bool(reason) and reason.split(":")[0].strip() in IMPUTE_REASONS
            and UPLOAD_PARSE_ERROR not in reason)


def impute_row(wav_res_path, text_ref, reason):
    """A maximum-error .raw row for an utterance the reference ASR could not read.

    The edit decomposition comes from scoring the EMPTY hypothesis through the
    normal path, so the reference column is normalized exactly as a real row's is
    and ins/del/sub stay self-consistent (every reference token deleted, wer 1.0).
    A reference that normalizes to nothing cannot produce that, so it falls back to
    the same constants explicitly.
    """
    try:
        raw_truth, _, wer, subs, dele, inse = process_one("", text_ref, base)
    except Exception:  # noqa: BLE001 -- never let imputation break a scoring run
        raw_truth, wer, subs, dele, inse = text_ref, 1.0, 0.0, 1.0, 0.0
    if not (isinstance(wer, float) and wer > 0):
        wer, subs, dele, inse = 1.0, 0.0, 1.0, 0.0
    tag = f"<UNSCOREABLE:{reason.split(':')[0].strip().replace(' ', '_')}>"
    return f"{wav_res_path}\t{wer}\t{raw_truth}\t{tag}\t{inse}\t{dele}\t{subs}\n"

# --- Per-language ASR routing (see docs/asr_whisper_vs_qwen.md) ------------------
# Whisper large-v3 with forced language beats Qwen3-ASR on the European/Slavic/
# cross-script langs (the GT WER tail) by 8-72 WER points, while Qwen3-ASR wins CJK,
# Hindi, Thai, Vietnamese, Yoruba and ties the easy Western langs. When a Whisper
# server is configured via WHISPER_BASE_URL, langs NOT in QWEN_LANGS route to it;
# with WHISPER_BASE_URL unset, everything uses Qwen -- i.e. prior behaviour, so GT
# and existing single-server runs are unchanged.
WHISPER_BASE_URL = os.environ.get("WHISPER_BASE_URL", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")

# Langs kept on Qwen3-ASR. zh/ko/th are near-ties (Qwen better on clean read speech,
# Whisper better on noisy VoxBlink) kept on Qwen.
# 2026-08-29 (scorer-swap control, scripts/control_scorer_swap.sh): `en` moved to
# Whisper. Qwen3-ASR measurably favored its own family on en -- both qwen3-tts
# systems got RELATIVELY better under Qwen than every cohort system, DiD +0.51
# [+0.28,+0.74] (1.7b) / +0.70 [+0.46,+0.96] (0.6b) WER_CLIP pts -- while the zh
# DiD was ns, so zh stays. Differential (between-system) bias outranks the
# slightly lower Qwen GT floor, which is common-mode. `yo` dropped from the set:
# the language was removed from the benchmark the same day (LID mislabel).
QWEN_LANGS = set((os.environ.get("QWEN_LANGS")
                  or "zh ko th vi hi fa").split())

# Qwen accepts an ISO hint only for these (its production config); others auto-detect.
QWEN_LANG_HINT = {"en": "en", "zh": "zh", "de": "de", "es": "es", "fr": "fr",
                  "it": "it", "ja": "ja", "ko": "ko", "ru": "ru"}
# Whisper forced-language code per benchmark lang (ISO-639-1 whisper codes; nb->no,
# fil->tl). Listed for every lang so an overridden routing still has a code.
WHISPER_LANG_CODE = {
    "en": "en", "zh": "zh", "de": "de", "es": "es", "fr": "fr", "it": "it",
    "ja": "ja", "ko": "ko", "ru": "ru", "pt": "pt", "ar": "ar", "hi": "hi",
    "tr": "tr", "pl": "pl", "id": "id", "th": "th", "fil": "tl", "tl": "tl",
    "nl": "nl", "vi": "vi", "cs": "cs", "sv": "sv", "da": "da", "fi": "fi",
    "el": "el", "hu": "hu", "ro": "ro", "uk": "uk", "nb": "no", "he": "he",
    "bg": "bg", "hr": "hr", "sk": "sk", "ca": "ca", "ms": "ms", "fa": "fa",
    "yo": "yo",
}

use_whisper = bool(WHISPER_BASE_URL) and base not in QWEN_LANGS
if use_whisper:
    BACKEND, _url, MODEL, iso = "whisper", WHISPER_BASE_URL, WHISPER_MODEL, WHISPER_LANG_CODE.get(base)
else:
    BACKEND, _url, iso = "qwen", BASE_URL, QWEN_LANG_HINT.get(base)
print(f"[wer] {lang}: backend={BACKEND} model={MODEL} lang_force={iso or 'auto'}",
      file=sys.stderr)

# A per-request timeout (no retries) bounds runaway transcriptions: degenerate
# TTS audio (or a Whisper repetition-hallucination) can make the model generate
# until its token cap, clogging the vLLM KV cache. On timeout the request is
# cancelled, freeing the server, and that utterance is skipped (score() catches it).
#
# ASR_TIMEOUT raises the bound for leaves whose audio is legitimately long. The
# default suits normal utterances (<20 s), but a system that OVER-GENERATES makes
# minutes-long files -- qwen3-tts-*-reftext on `hi` has a p95 of 164 s against 17 s
# for the same model without the reference transcript. Those requests time out,
# the utterances drop out of coverage, and the surviving mean would then flatter
# exactly the most broken system (the survivor bias run_wer.py already guards
# against elsewhere). Raise the timeout to SCORE the runaway audio instead.
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "60"))
client = OpenAI(base_url=_url, api_key="EMPTY", timeout=ASR_TIMEOUT, max_retries=0)
_iso_ok = iso is not None  # flips off if the server rejects this code once

# --- Repetition-hallucination guard (surgical, retry-on-loop) --------------------
# On spontaneous / noisy in-the-wild audio the ASR decoder can spiral into a
# repeated n-gram, emitting hundreds of tokens of loop (e.g. a 207-char reference
# transcribed as 1530 chars). Greedy decoding has no defence, so a single clip can
# add 200+ edits and dominate a language's WER tail (see docs/asr_whisper_vs_qwen
# + the reliability plates). A *global* repetition_penalty fixes the loop but
# suppresses legitimate repeated words on every clean clip -- measured net-negative
# (on VoxBlink it, 207 clips worsened vs 26 improved). So the guard is applied
# surgically: every clip is transcribed greedily first (byte-identical to an
# unguarded run), and ONLY a clip whose output trips a loop detector is re-decoded
# with the penalty. The >99% of clips that decode cleanly are untouched.
#
# Detector = Whisper's own hallucination signal: a repeated-ngram loop compresses
# far more than natural language (gzip ratio > ~2.4; a full sweep of the routed
# VoxBlink GT put every clean clip <= 2.26 and every loop at 6-22, a clean margin).
# The guard sampling params pass straight through to vLLM's transcription endpoint.
#
# DEFAULTS (best config, on by default so every run gets it -- override via env):
#   ASR_REPETITION_PENALTY  1.2   swept on the loop clips: 1.2 is optimal/tied on
#                                 all three; 1.5 over-penalizes (fr WER 0.02->0.26),
#                                 1.1 under-breaks a couple. Set 1.0 to disable.
#   ASR_MAX_COMPLETION_TOKENS 256 caps a runaway that still starts (only on the
#                                 retry; the greedy pass is uncapped so long clean
#                                 clips are never truncated).
#   ASR_LOOP_COMPRESSION_RATIO 2.4 loop-detector threshold.
#   ASR_SEED                  0    fixed seed on every request for reproducibility
#                                 (verified: same seed -> identical transcript).
# ASR_GUARD=0 turns the retry off entirely (pure greedy, seed still applied).
def _env_num(name, cast, default=None):
    v = os.environ.get(name, "").strip()
    return cast(v) if v else default


_SEED = _env_num("ASR_SEED", int, 0)
_LOOP_CR = _env_num("ASR_LOOP_COMPRESSION_RATIO", float, 2.4)
_rep = _env_num("ASR_REPETITION_PENALTY", float, 1.2)
_maxtok = _env_num("ASR_MAX_COMPLETION_TOKENS", int, 256)
# guard on by default; ASR_GUARD=0 or repetition_penalty<=1 disables the retry
_GUARD = {}
if os.environ.get("ASR_GUARD", "1") != "0" and _rep and _rep > 1.0:
    _GUARD = {"repetition_penalty": _rep, "max_completion_tokens": _maxtok}
print(f"[wer] {lang}: seed={_SEED}"
      + (f" | repetition guard {_GUARD} on loops (gzip ratio > {_LOOP_CR})"
         if _GUARD else " | guard OFF"), file=sys.stderr)


def _degenerate(text):
    """A repetition-hallucination loop: compresses far more than natural text.
    Short strings compress poorly, so require some length before trusting it."""
    b = text.encode("utf-8")
    if len(b) < 60:
        return False
    return len(b) / max(1, len(zlib.compress(b))) > _LOOP_CR


def transcribe(wav_path):
    global _iso_ok

    def _call(language, guard):
        with open(wav_path, "rb") as f:
            kwargs = dict(model=MODEL, file=f)
            if language:
                kwargs["language"] = language
            # seed on every request (repro); guard params only on the retry
            kwargs["extra_body"] = {"seed": _SEED, **(guard or {})}
            resp = client.audio.transcriptions.create(**kwargs)
        text = (resp if isinstance(resp, str) else resp.text).strip()
        # vLLM's transcription endpoint may hand back a JSON body ({"text": "..."});
        # unwrap it so the hypothesis is the plain transcript, not serialized JSON.
        if text.startswith("{"):
            try:
                text = json.loads(text).get("text", text)
            except (ValueError, AttributeError):
                pass
        return text

    def _greedy_then_guard(language):
        text = _call(language, None)
        if _GUARD and _degenerate(text):
            retry = _call(language, _GUARD)
            # keep the retry only if it actually broke the loop / shortened the spiral
            if not _degenerate(retry) or len(retry) < len(text):
                print(f"[wer] {os.path.basename(wav_path)}: loop guarded "
                      f"({len(text)}->{len(retry)} ch)", file=sys.stderr)
                text = retry
        return text

    if _iso_ok:
        try:
            return _greedy_then_guard(iso)
        except BadRequestError as e:
            if UPLOAD_PARSE_ERROR in str(e):
                raise  # the server could not read the upload: harness, not language
            # Unsupported language code for this backend (e.g. Whisper rejects some
            # ISO codes) -- stop forcing it and auto-detect for the rest of the lang.
            _iso_ok = False
            print(f"[wer] {lang}: language '{iso}' rejected by {BACKEND}; auto-detecting",
                  file=sys.stderr)
    return _greedy_then_guard(None)


def score(item):
    """Return (row, None) on success, or (None, reason) for an utterance that
    could not be scored -- run_asr records the reason in the .meta sidecar."""
    wav_res_path, text_ref = item
    if not os.path.exists(wav_res_path):
        return None, "missing wav"
    transcription = None
    last_err = None
    for attempt in range(ASR_RETRIES + 1):
        try:
            transcription = transcribe(wav_res_path).strip()
            break
        except BadRequestError as e:
            last_err = e
            if UPLOAD_PARSE_ERROR not in str(e):
                break  # permanent (bad audio/params) -- retrying cannot help
            if attempt < ASR_RETRIES:
                print(f"[wer] {wav_res_path}: server could not read the upload "
                      f"-- retry {attempt + 1}/{ASR_RETRIES}")
                time.sleep(ASR_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            last_err = e
            if attempt < ASR_RETRIES:
                print(f"[wer] {wav_res_path}: {type(e).__name__}: {e} "
                      f"-- retry {attempt + 1}/{ASR_RETRIES}")
                time.sleep(ASR_RETRY_DELAY * (attempt + 1))
    if transcription is None:
        print(f"[wer] {wav_res_path}: {type(last_err).__name__}: {last_err}")
        return None, f"{type(last_err).__name__}: {last_err}"

    # NB: the zh traditional->simplified fold happens inside process_one/prepare
    # now, applied to reference AND hypothesis (utils/text_norm.py).
    raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref, base)
    return f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n", None


def run_asr(wav_res_text_path, res_path):
    params = []
    with open(wav_res_text_path) as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) == 2:
                wav_res_path, text_ref = parts
            elif len(parts) == 3:
                wav_res_path, _, text_ref = parts
            elif len(parts) == 4:  # for edit
                wav_res_path, _, text_ref, _ = parts
            else:
                raise ValueError(f"{wav_res_text_path}:{lineno}: expected 2-4 |-separated fields, got {len(parts)}")
            params.append((wav_res_path, text_ref))

    n_scored = 0
    failed = []
    with open(res_path, "w") as fout, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (wav_res_path, text_ref), (row, reason) in zip(
                params, tqdm(ex.map(score, params), total=len(params))):
            if row:
                n_scored += 1
                fout.write(row)
                fout.flush()
            else:
                failed.append((wav_res_path, text_ref, reason))

        # Maximum-error rows for the failures the audio is responsible for. They go
        # after the transcribed rows rather than in manifest order so that a .raw
        # diff shows them as a block; aggregation is order-independent.
        imputed = [f for f in failed if WER_IMPUTE and imputable(f[2])]
        for wav_res_path, text_ref, reason in imputed:
            fout.write(impute_row(wav_res_path, text_ref, reason))
        fout.flush()

    # Explicit coverage record: the sidecar names every utterance that produced no
    # transcript and says whether it was imputed, and average_wer.py folds
    # n_expected into the .wer N_SCORED:/N_EXPECTED: footer (failing the leaf below
    # SCORE_MIN_COVERAGE). The .raw format itself is untouched -- its parsers
    # (per_uid, perlen/perbin builders) see rows only.
    with open(res_path + ".meta", "w") as f:
        json.dump({"n_expected": len(params), "n_transcribed": n_scored,
                   "n_imputed": len(imputed), "n_rows": n_scored + len(imputed),
                   "impute_policy": bool(WER_IMPUTE),
                   "unscored": [{"wav": os.path.basename(w), "reason": r,
                                 "imputed": bool(WER_IMPUTE and imputable(r))}
                                for w, _, r in failed]},
                  f, indent=1)

    # Always surface both numbers: how much of the leaf the ASR could actually read
    # (n_scored) and how much is only in the mean as maximum error (len(imputed)).
    # A model that crashes on 30% of utterances should not look good, and one whose
    # output is untranscribable should not look good *by disappearing*.
    n_dropped = len(failed) - len(imputed)
    print(f"[wer] transcribed {n_scored}/{len(params)} utterances"
          + (f" -- {len(imputed)} scored as maximum error (unscoreable audio)" if imputed else "")
          + (f" -- WARNING: {n_dropped} dropped (harness failure, not imputed)" if n_dropped else ""))


run_asr(wav_res_text_path, res_path)
