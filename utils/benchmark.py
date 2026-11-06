"""ZTTS2-Eval benchmark definition: language baskets, scorer pins, version stamps,
score-file footer readers, resume gates and the results/ leaf layout.

Stdlib only, so shell gates and light consumers (explorer/serve.py) can import it.

    python3 -m utils.benchmark version
    python3 -m utils.benchmark langs <task>
    python3 -m utils.benchmark problems <leaf_dir> <lang>     # exit 1 if stale
    python3 -m utils.benchmark stale <mode> <data_dir> <leaf_dir> <lang>
        # the score kinds (wer spk xutt msr) a leaf still needs; mode is
        # system | gt_clean | gt_itw

A score file is "current" when it carries the footer stamps written by the scorers
(average_wer.py, eval_speaker_similarity.py, score_msr_utmos.py). build_results.py
refuses any input whose stamp does not match.
"""
import os
import random
import re
import sys

BENCHMARK_VERSION = "2"

CLEAN_LANGS = ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi",
               "tr", "pl", "id", "th", "fil", "nl", "vi", "cs", "sv", "da", "fi", "el",
               "hu", "ro", "uk", "nb", "he", "bg", "hr", "sk", "ca", "ms", "fa")
ITW_LANGS = ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi",
             "id", "tr", "tl", "pl", "th", "nl", "uk", "vi")

SOURCE_TASK = {
    "clean": "clean",
    "clean_mini": "clean",
    "clean_delta": "clean",
    "in-the-wild": "in-the-wild",
    "in-the-wild_mini": "in-the-wild",
    "in-the-wild_delta": "in-the-wild",
}
FULL_TASKS = ("clean", "in-the-wild")
HF_REPO_ID = "Zyphra/ZTTS2-Eval"


def langs_for(task):
    """The frozen language basket of a task (full set or one of its panels)."""
    src = SOURCE_TASK[task]
    return ITW_LANGS if src == "in-the-wild" else CLEAN_LANGS


def is_panel_alias(system_dir, task):
    """True if results/<system>/<panel task> is a symlink to the system's own full-set
    dir. Such an alias only lets the explorer drawer browse a backfilled panel row
    (the server filters the leaf to the panel's items). It is not a panel run, so
    scoring and the results builders skip it."""
    src = SOURCE_TASK.get(task)
    path = os.path.join(system_dir, task)
    return (src is not None and task != src and os.path.islink(path)
            and os.path.realpath(path) == os.path.realpath(os.path.join(system_dir, src)))


# --- ASR routing -----------------------------------------------------------------
QWEN_LANGS = frozenset({"zh", "ko", "th", "vi", "hi", "fa"})


def asr_backend(lang, qwen_langs=QWEN_LANGS):
    return "qwen" if lang in qwen_langs else "whisper"


# --- Pinned scorers --------------------------------------------------------------
ASR_MODELS = {
    "qwen": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    "whisper": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
}
SPK_HUB_REPO = "PalabraAI/redimnet2"
SPK_HUB_REF = "ffe8f272bfaf6a41efb7814b28d94377b1613ac9"
SPK_VARIANT = ("b6", "lm", "vb2+vox2_v0")
MSR_UTMOS_CKPT = ("https://huggingface.co/sarulab-speech/MSR-UTMOS_w2v2_fold0/resolve/"
                  "2cbd24044e8dabcafc44bf801b19b4d59d557540/model.ckpt")
MHUBERT = ("utter-project/mHuBERT-147", "7ad3fc0bc5106c58c9c13526abccad527150d135")
# Silero VAD v6.2.3 (utils/vad.py): paired ITW pauses (scripts/prosody/paired_itw.py) and the rhythm
# edge cut (scripts/prosody/rhythm_dist.py).
SILERO_VAD = ("snakers4/silero-vad", "5cd7945676eb32225748052e2e6a0580e4686a08")


def asr_stamp(backend, model, revision):
    return f"{backend} {model}@{revision}"


def expected_asr_stamp(lang):
    backend = asr_backend(lang)
    return asr_stamp(backend, *ASR_MODELS[backend])


def spk_stamp(model_name, train_type, dataset):
    return f"{SPK_HUB_REPO}@{SPK_HUB_REF}:{model_name}:{train_type}:{dataset}"


EXPECTED_SPK_STAMP = spk_stamp(*SPK_VARIANT)


# --- Footer readers --------------------------------------------------------------
# The footers grew (CI, N_IMPUTED, WER_TAIL50, BENCHMARK_VERSION, ASR; the speaker
# files carry trailing `#` lines), so every footer reader looks at a 4096-byte tail and
# takes the LAST matching line.
FOOTER_TAIL_BYTES = 4096
WER_RE = re.compile(r"WER:\s*([\-\d\.eN]+(?:nan|inf)?|nan|inf|[\d\.\-eE]+),\s*WER_L50:\s*([\d\.\-eEnan]+|nan),\s*WER_CLIP:\s*([\d\.\-eEnan]+|nan)")
# 95% CI footer line emitted by average_wer.py.
WER_CI_RE = re.compile(r"WER_CI95:\s*\[([\d\.\-eEnan]+),\s*([\d\.\-eEnan]+)\],\s*WER_CLIP_CI95:\s*\[([\d\.\-eEnan]+),\s*([\d\.\-eEnan]+)\]")
MSR_CI_RE = re.compile(r"CI95\s+([\d\.\-eEnan]+)\s+([\d\.\-eEnan]+)")


def read_tail(path, nbytes=FOOTER_TAIL_BYTES):
    """Last `nbytes` of a file as text, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-nbytes, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode(errors="replace")
    except Exception:
        return None


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _last_line(path, prefix):
    """The last tail line starting with `prefix`, or None."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith(prefix):
            return line
    return None


def footer_value(path, key):
    """Value of the LAST line `<key>: <value>` in the file's tail, else None."""
    line = _last_line(path, f"{key}: ")
    return None if line is None else line[len(key) + 2:].strip()


def wer_footer(path):
    """(wer, wer_l50, wer_clip) of the last `WER:` footer line of a .wer file, or None."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("WER:"):
            m = WER_RE.search(line)
            if m:
                return _float(m.group(1)), _float(m.group(2)), _float(m.group(3))
    return None


def wer_ci_footer(path):
    """(wer_lo, wer_hi, clip_lo, clip_hi) from the WER_CI95 footer, or None."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("WER_CI95:"):
            m = WER_CI_RE.search(line)
            if m:
                return tuple(_float(m.group(i)) for i in (1, 2, 3, 4))
    return None


def tail50_footer(path):
    """Share (%) of utterances with WER > 50%, from the `WER_TAIL50:` footer, or None."""
    line = _last_line(path, "WER_TAIL50:")
    if line is None:
        return None
    v = _float(line.split(":", 1)[1].strip())
    return v if v == v else None


def msr_ci_footer(path):
    """(lo, hi) from a msr_utmos_ci.txt `CI95` line, or None."""
    try:
        with open(path) as f:
            m = MSR_CI_RE.search(f.read())
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:
        return None
    return None


def spk_avg_footer(path):
    """Mean hyp score (cosine x100) from the last `avg` line of a speaker-score file."""
    line = _last_line(path, "avg")
    if line is None:
        return None
    try:
        return float(line.split()[2])
    except (IndexError, ValueError):
        return None


def msr_mean(path):
    """The MSR-UTMOS mean of msr_utmos_mean.txt, or None. A 0.0 mean is the scorer's
    historical crash stub (MOS scale is ~1-5, a real mean cannot be 0): unscored."""
    try:
        with open(path) as f:
            v = float(f.read().strip())
    except Exception:
        return None
    return v if v > 0.0 else None


# --- Resume gates ----------------------------------------------------------------
# Score file of each kind, relative to its leaf. "msr" is stamped in its CI file.
SCORE_FILES = {
    "wer": "wav_res_ref_text.wer",
    "spk": "spk_simi_scores.txt",
    "xutt": "spk_simi_scores.xutt.txt",
    "msr": "msr_utmos_ci.txt",
}


def _stamps(kind, lang):
    """[(footer key, expected value)] a current score file of `kind` carries."""
    if kind == "wer":
        return [("BENCHMARK_VERSION", BENCHMARK_VERSION), ("ASR", expected_asr_stamp(lang))]
    if kind in ("spk", "xutt"):
        return [("#BENCHMARK_VERSION", BENCHMARK_VERSION), ("#SPK_MODEL", EXPECTED_SPK_STAMP)]
    if kind == "msr":
        return [("BENCHMARK_VERSION", BENCHMARK_VERSION), ("MSR_UTMOS_CKPT", MSR_UTMOS_CKPT)]
    raise ValueError(f"unknown score kind {kind!r}")


def stamp_problems(kind, path, lang):
    """Stamp mismatches of one score file (`path` is the stamped file of `kind`)."""
    out = []
    for key, want in _stamps(kind, lang):
        got = footer_value(path, key)
        if got != want:
            out.append(f"{path}: {key} is {got!r}, expected {want!r}")
    return out


def _coverage_problem(path, prefix):
    ns, ne = footer_value(path, f"{prefix}N_SCORED"), footer_value(path, f"{prefix}N_EXPECTED")
    if ns is None or ne is None:
        return None
    from utils.average_wer import MIN_COVERAGE  # lazy: average_wer needs numpy
    try:
        n_scored, n_expected = float(ns), float(ne)
    except ValueError:
        return f"{path}: unreadable coverage footer {ns!r}/{ne!r}"
    if n_expected > 0 and n_scored < MIN_COVERAGE * n_expected:
        return f"{path}: coverage {ns}/{ne} below {MIN_COVERAGE}"
    return None


def file_problems(kind, path, lang):
    """Why the score file of `kind` at `path` cannot be resumed from; [] if current.

    For "msr", `path` is the leaf's msr_utmos_ci.txt and msr_utmos_mean.txt beside it
    must be non-empty. A missing file is a problem."""
    if not os.path.isfile(path):
        return [f"{path}: missing"]
    out = stamp_problems(kind, path, lang)
    if kind == "wer":
        line = _last_line(path, "WER:")
        if line is None:
            out.append(f"{path}: no WER: line")
        elif line.split(":", 1)[1].split(",", 1)[0].strip() == "nan":
            out.append(f"{path}: WER is nan")
        cov = _coverage_problem(path, "")
    elif kind in ("spk", "xutt"):
        line = _last_line(path, "avg ")
        if line is None:
            out.append(f"{path}: no avg line")
        elif line.strip() == "avg 0.00 0.00":
            out.append(f"{path}: avg 0.00 0.00 (nothing scored)")
        cov = _coverage_problem(path, "#")
    else:
        mean = os.path.join(os.path.dirname(path), "msr_utmos_mean.txt")
        if not (os.path.isfile(mean) and os.path.getsize(mean) > 0):
            out.append(f"{mean}: missing or empty")
        cov = _coverage_problem(path, "")
    if cov:
        out.append(cov)
    return out


def expected_kinds(mode, data_dir):
    """Score kinds a leaf of `mode` (system | gt_clean | gt_itw) must carry."""
    if mode == "system":
        kinds = ["wer", "spk", "msr"]
        if os.path.isfile(os.path.join(data_dir, "target_wav.scp")):
            kinds.insert(2, "xutt")
        return kinds
    if mode == "gt_clean":
        return ["wer", "msr"]
    if mode == "gt_itw":
        return ["wer", "spk", "msr"]
    raise ValueError(f"unknown scoring mode {mode!r}")


def stale_kinds(mode, data_dir, leaf, lang):
    """The expected score kinds of a leaf whose file fails file_problems()."""
    return [k for k in expected_kinds(mode, data_dir)
            if file_problems(k, os.path.join(leaf, SCORE_FILES[k]), lang)]


def leaf_problems(leaf_dir, lang):
    """Why a scored leaf is not current ZTTS2-Eval <BENCHMARK_VERSION>; [] if current.
    Only the score files present are checked (stamps only)."""
    out = []
    j = os.path.join
    for kind in ("wer", "spk", "xutt"):
        p = j(leaf_dir, SCORE_FILES[kind])
        if os.path.exists(p):
            out.extend(stamp_problems(kind, p, lang))
    if os.path.exists(j(leaf_dir, "msr_utmos_mean.txt")):
        out.extend(stamp_problems("msr", j(leaf_dir, SCORE_FILES["msr"]), lang))
    return out


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- results/ layout -------------------------------------------------------------
# Systems nest the task (results/<system>/<task>/<lang>); the GT and Whisper baselines
# are one system per task with the task folded into the name (results/gt_clean/<lang>).
FLAT_PREFIXES = ("gt_", "whisper_")


def is_flat(system):
    return system.startswith(FLAT_PREFIXES)


def leaf_dir(system, task, lang, root=_ROOT):
    """results/<system>/<lang> for a flat baseline, else results/<system>/<task>/<lang>."""
    if is_flat(system):
        return os.path.join(root, "results", system, lang)
    return os.path.join(root, "results", system, task, lang)


def iter_leaves(task, lang, marker, root=_ROOT):
    """{system: leaf dir} of every results/ system whose (task, lang) leaf holds
    `marker`: nested systems (name not starting with `_`), plus the flat baselines
    named <prefix><task>."""
    out = {}
    res = os.path.join(root, "results")
    try:
        names = sorted(os.listdir(res))
    except OSError:
        return out
    for name in names:
        if name.startswith("_"):
            continue
        if is_flat(name) and name not in {p + task for p in FLAT_PREFIXES}:
            continue
        leaf = leaf_dir(name, task, lang, root)
        if os.path.exists(os.path.join(leaf, marker)):
            out[name] = leaf
    return out


# --- Data helpers ----------------------------------------------------------------
def data_uids(task, lang, root=_ROOT):
    """uids of data/<task>/<lang>/text, in file order."""
    uids = []
    with open(os.path.join(root, "data", task, lang, "text"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uids.append(line.split(None, 1)[0])
    return uids


def sample_uids(task, lang, n=500, seed=0, root=_ROOT):
    """Deterministic n-uid sample (sorted) shared by every sampled prosody metric."""
    uids = sorted(data_uids(task, lang, root))
    return sorted(random.Random(seed).sample(uids, min(n, len(uids))))


def main(argv):
    if len(argv) >= 1 and argv[0] == "version":
        print(BENCHMARK_VERSION)
        return 0
    if len(argv) == 2 and argv[0] == "langs":
        print(" ".join(langs_for(argv[1])))
        return 0
    if len(argv) == 3 and argv[0] == "problems":
        probs = leaf_problems(argv[1], argv[2])
        for p in probs:
            print(p)
        return 1 if probs else 0
    if len(argv) == 5 and argv[0] == "stale":
        print(" ".join(stale_kinds(*argv[1:])))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
