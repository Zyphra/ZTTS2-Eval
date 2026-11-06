"""Build the FLEURS-R-derived subset of the ZTTS2-Eval benchmark.

Regenerates the multilingual voice-cloning test set from google/fleurs-r (a
higher-quality restoration of FLEURS):

  - clean/<lang>   mono-lingual cloning; prompt and target are different
                       utterances of the same language.

Only the FLEURS-derived subset is regenerated; the overwrite guard protects any
other pre-existing subset dirs.

google/fleurs-r stores raw FLEURS-style files per language: data/<cfg>/<split>.tsv
(metadata) and data/<cfg>/audio/<split>.tar.gz (wavs). There is no per-language HF
dataset config, so the files are downloaded and parsed directly.

Each <lang> directory is written with the layout the scoring pipeline expects:
  text            '<uttid> <target transcript>'   (what the TTS model synthesizes)
  prompt_text     '<uttid> <prompt transcript>'
  prompt_wav.scp  '<uttid> <path to prompt wav>'
  waveform/       prompt_audio_<K>.wav             (16 kHz mono PCM16)

By default every curated language is rebuilt at the maximum sample size the pooled
train+test splits allow (prompts and targets are disjoint, so the ceiling is
~half the split); pass --count to cap it.

Examples:
  python scripts/prep/prepare_fleurs_r.py --overwrite
  python scripts/prep/prepare_fleurs_r.py --langs nl --count 10 \\
      --splits test --data-root /tmp/scratch --overwrite     # quick smoke test
"""
import argparse
import csv
import os
import random
import shutil
import sys
import tarfile
import tempfile

import librosa
import soundfile as sf

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.text_norm import near_duplicate  # noqa: E402

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("The 'huggingface_hub' package is required. Run `uv sync`.")

# FLEURS-R per-language directory name (data/<cfg>/...) per benchmark language code.
# The first 9 are the original clean langs; the rest are a curated higher-resource
# tier (VoxBlink2-parity langs + widely-supported additions). All configs below are
# present in google/fleurs-r. Note: VoxBlink2's `tl` (Tagalog) maps to FLEURS `fil`
# (Filipino), which is the closest available config, so the two sets align.
FLEURS_R_CONFIG = {
    # original 9
    "en": "en_us", "zh": "cmn_hans_cn", "de": "de_de", "es": "es_419",
    "fr": "fr_fr", "it": "it_it", "ja": "ja_jp", "ko": "ko_kr", "ru": "ru_ru",
    # VoxBlink2-parity additions
    "pt": "pt_br", "ar": "ar_eg", "hi": "hi_in", "tr": "tr_tr", "pl": "pl_pl",
    "id": "id_id", "th": "th_th", "fil": "fil_ph",
    # higher-resource tier
    "nl": "nl_nl", "vi": "vi_vn", "cs": "cs_cz", "sv": "sv_se", "da": "da_dk",
    "fi": "fi_fi", "el": "el_gr", "hu": "hu_hu", "ro": "ro_ro", "uk": "uk_ua",
    "nb": "nb_no", "he": "he_il", "bg": "bg_bg", "hr": "hr_hr", "sk": "sk_sk",
    "ca": "ca_es", "ms": "ms_my", "fa": "fa_ir",
}

HF_DATASET = "google/fleurs-r"
SR = 16000


def clean(text):
    """Collapse all whitespace (incl. newlines) so a transcript fits one line."""
    return " ".join(text.split())


# Target-length cap on transcripts (in whitespace-split words); see the filter
# in parse_tsv. Overridable via --max-words.
MAX_WORDS = 50


def parse_tsv(tsv_path):
    """Parse a FLEURS-R metadata TSV (7 columns, no header):
    id, file_name, raw_transcription, transcription, char_split, num_samples, gender.
    """
    records = []
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 7:
                continue
            text = clean(row[2])
            # Two filters, upstream of select_texts:
            # - '|' rows are malformed (run_wer.py uses '|' as field separator);
            #   FLEURS-R en_us train has a few, incl. one >5000-"word" TSV parse
            #   artifact that merged multiple records into one field.
            # - MAX_WORDS is a deliberate target-length cap: it bounds every
            #   prompt/target transcript, not just the artifact rows. Override
            #   with --max-words (changing it changes which utterances the
            #   seeded sampler sees, i.e. rebuilds a different set).
            if "|" in text or len(text.split()) > MAX_WORDS:
                continue
            records.append({
                "file_name": row[1],
                "text": text,
                "num_samples": int(row[5]) if row[5].isdigit() else 0,
            })
    return records


def _hf_error(e, lang_config, split):
    msg = str(e).lower()
    if any(k in msg for k in ("gated", "401", "403", "authenticate",
                              "restricted", "forbidden")):
        sys.exit(
            f"ERROR: cannot access {HF_DATASET} (authentication required).\n"
            f"  Accept the terms at https://huggingface.co/datasets/{HF_DATASET},\n"
            "  then run `huggingface-cli login` or pass --hf-token <token>.\n"
            f"Original error: {e}")
    if any(k in msg for k in ("not found", "404", "entrynotfound")):
        sys.exit(
            f"ERROR: {HF_DATASET} has no data/{lang_config}/{split}.* file "
            f"(bad language config '{lang_config}' or split '{split}').\n"
            f"Original error: {e}")
    raise e


def fetch_split(lang_config, split, token, need_audio):
    """Download a FLEURS-R language/split.

    Returns (records, audio_dir, tmp_dir); audio_dir/tmp_dir are None when
    need_audio is False. The caller must shutil.rmtree(tmp_dir) when done.
    """
    try:
        tsv_path = hf_hub_download(HF_DATASET, f"data/{lang_config}/{split}.tsv",
                                   repo_type="dataset", token=token)
    except Exception as e:
        _hf_error(e, lang_config, split)
    records = parse_tsv(tsv_path)

    audio_dir = tmp_dir = None
    if need_audio:
        try:
            tar_path = hf_hub_download(
                HF_DATASET, f"data/{lang_config}/audio/{split}.tar.gz",
                repo_type="dataset", token=token)
        except Exception as e:
            _hf_error(e, lang_config, split)
        tmp_dir = tempfile.mkdtemp(prefix="fleurs_r_")
        with tarfile.open(tar_path, "r:gz") as tar:
            try:
                tar.extractall(tmp_dir, filter="data")     # py>=3.10.12
            except TypeError:
                tar.extractall(tmp_dir)
        audio_dir = os.path.join(tmp_dir, split)           # members are <split>/<file>
    return records, audio_dir, tmp_dir


def load_wav(path):
    """Read a wav as a mono 16 kHz float array."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                             # downmix to mono
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def prompt_candidates(records, min_dur, max_dur, rng):
    """Shuffled duration-filtered records whose prompt wav exists (no audio loaded yet).

    Records must carry an 'audio_dir' key (set per split in build_fleurs_r).
    """
    pool = [r for r in records if min_dur <= r["num_samples"] / SR <= max_dur]
    pool = [r for r in pool
            if os.path.exists(os.path.join(r["audio_dir"], r["file_name"]))]
    rng.shuffle(pool)
    return pool


def load_prompts(cands):
    """Load audio for the chosen prompt records.

    Returns (list of {'text', 'audio'}, set of used file_names).
    """
    prompts, used = [], set()
    for r in cands:
        wav_path = os.path.join(r["audio_dir"], r["file_name"])
        prompts.append({"text": r["text"], "audio": load_wav(wav_path)})
        used.add(r["file_name"])
    return prompts, used


def select_texts(records, count, rng, exclude=None):
    """Select `count` transcripts as synthesis targets (text only)."""
    exclude = exclude or set()
    pool = [r["text"] for r in records if r["file_name"] not in exclude]
    rng.shuffle(pool)
    if len(pool) < count:
        sys.exit(f"ERROR: only {len(pool)} candidate target texts available, "
                 f"need {count}. Lower --count or use a larger --split.")
    return pool[:count]


def write_subset(data_root, subset_dir, lang_dir, prompts, targets):
    """Write text / prompt_text / prompt_wav.scp / waveform for one language dir."""
    out_dir = os.path.join(data_root, subset_dir, lang_dir)
    wav_dir = os.path.join(out_dir, "waveform")
    shutil.rmtree(wav_dir, ignore_errors=True)             # drop stale prompt wavs
    os.makedirs(wav_dir, exist_ok=True)

    n = min(len(prompts), len(targets))
    with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as f_text, \
         open(os.path.join(out_dir, "prompt_text"), "w", encoding="utf-8") as f_ptext, \
         open(os.path.join(out_dir, "prompt_wav.scp"), "w", encoding="utf-8") as f_scp:
        for k in range(1, n + 1):
            prompt = prompts[k - 1]
            wav_name = f"prompt_audio_{k}.wav"
            sf.write(os.path.join(wav_dir, wav_name), prompt["audio"], SR,
                     subtype="PCM_16")
            uid = f"uttid_{k}"
            # Double-slash style matches the original .scp files for drop-in use.
            scp_path = f"{data_root}/{subset_dir}//{lang_dir}/waveform/{wav_name}"
            f_text.write(f"{uid} {targets[k - 1]}\n")
            f_ptext.write(f"{uid} {prompt['text']}\n")
            f_scp.write(f"{uid} {scp_path}\n")
    print(f"[ok] {subset_dir}/{lang_dir}: wrote {n} items -> {out_dir}")


def build_fleurs_r(lang, count, args, rng):
    """Build one language dir. count=None means 'max out' the pooled splits.

    Prompts and targets are disjoint utterances (zero-shot cloning), so the
    per-language ceiling is min(#duration-filtered prompts, #records // 2).
    """
    cfg = FLEURS_R_CONFIG[lang]
    print(f"[load] {HF_DATASET} data/{cfg} splits={args.splits}")
    all_records, tmps = [], []
    try:
        for split in args.splits:
            recs, audio_dir, tmp = fetch_split(cfg, split, args.token, need_audio=True)
            for r in recs:
                r["audio_dir"] = audio_dir
            all_records.extend(recs)
            if tmp:
                tmps.append(tmp)

        cands = prompt_candidates(all_records, args.min_dur, args.max_dur, rng)
        total = len(all_records)
        n_max = min(len(cands), total // 2)      # prompts+targets stay disjoint
        n = min(n_max, count) if count else n_max
        if n == 0:
            sys.exit(f"ERROR: {lang}: no usable utterances "
                     f"(records={total}, dur-filtered prompts={len(cands)}). "
                     "Widen --min-dur/--max-dur or add a split.")
        if count and n < count:
            print(f"[warn] {lang}: only {n} available (< requested {count}); "
                  "capping.")
        prompts, used = load_prompts(cands[:n])
        targets = select_texts(all_records, n, rng, exclude=used)  # targets != prompts
        # A prompt whose transcript (near-)equals its target lets a system win by
        # copying the prompt: drop those pairs (utils/text_norm.near_duplicate).
        keep = [k for k in range(min(len(prompts), len(targets)))
                if not near_duplicate(prompts[k]["text"], targets[k])]
        if len(keep) < min(len(prompts), len(targets)):
            print(f"[dedup] {lang}: dropped {min(len(prompts), len(targets)) - len(keep)} "
                  "prompt/target near-duplicates")
        prompts, targets = [prompts[k] for k in keep], [targets[k] for k in keep]
        print(f"[pick] {lang}: N={n} (ceiling {n_max}: prompts_pool={len(cands)}, "
              f"records={total}, splits={args.splits})")
        write_subset(args.data_root, "clean", lang, prompts, targets)
    finally:
        for tmp in tmps:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    global MAX_WORDS
    ap = argparse.ArgumentParser(
        description="Regenerate the FLEURS-R-derived clean subset.")
    ap.add_argument("--langs", nargs="+", default=None,
                    help="language codes (default: all curated langs)")
    ap.add_argument("--data-root", default="data", help="repo data directory")
    ap.add_argument("--count", type=int, default=None,
                    help="utterances per language (default: max the pooled splits allow)")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "dev", "test"],
                    help="FLEURS-R splits to pool from (default: train test)")
    ap.add_argument("--min-dur", type=float, default=3.0,
                    help="minimum prompt-audio duration in seconds")
    ap.add_argument("--max-dur", type=float, default=20.0,
                    help="maximum prompt-audio duration in seconds")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS,
                    help="drop transcripts longer than this many words "
                         "(target-length cap; changing it rebuilds a different set)")
    ap.add_argument("--overwrite", action="store_true",
                    help="required to rewrite a language dir that already exists")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (else uses HF_TOKEN env or cached login)")
    args = ap.parse_args()

    MAX_WORDS = args.max_words
    args.token = args.hf_token or os.environ.get("HF_TOKEN") or None

    langs = args.langs or list(FLEURS_R_CONFIG)
    allowed = set(FLEURS_R_CONFIG)

    bad = [lang for lang in langs if lang not in allowed]
    if bad:
        sys.exit(f"ERROR: unsupported langs: {bad}. Allowed: {sorted(allowed)}")

    # Fail-fast overwrite guard -- protects any pre-existing subset dirs.
    for lang in langs:
        text_path = os.path.join(args.data_root, "clean", lang, "text")
        if os.path.exists(text_path) and not args.overwrite:
            sys.exit(f"ERROR: {text_path} already exists. Pass --overwrite to regenerate.")

    print(f"[start] langs={langs} count={args.count or 'max'} "
          f"splits={args.splits} seed={args.seed}")
    for lang in langs:
        rng = random.Random(args.seed + sum(ord(c) for c in lang))
        build_fleurs_r(lang, args.count, args, rng)
    print("[done]")


if __name__ == "__main__":
    main()
