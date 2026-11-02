"""ZTTS2-Eval benchmark definition: language baskets, scorer pins, version stamps.

Stdlib only, so shell gates and light consumers (explorer/serve.py) can import it.

    python3 -m utils.benchmark version
    python3 -m utils.benchmark langs <task>
    python3 -m utils.benchmark problems <leaf_dir> <lang>     # exit 1 if stale

A score file is "current" when it carries the footer stamps written by the scorers
(average_wer.py, eval_speaker_similarity.py, score_msr_utmos.py). build_results.py
refuses any input whose stamp does not match.
"""
import os
import random
import sys

BENCHMARK_VERSION = "2"

CLEAN_LANGS = ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi",
               "tr", "pl", "id", "th", "fil", "nl", "vi", "cs", "sv", "da", "fi", "el",
               "hu", "ro", "uk", "nb", "he", "bg", "hr", "sk", "ca", "ms", "fa")
ITW_LANGS = ("en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru", "pt", "ar", "hi",
             "id", "tr", "tl", "pl", "th", "nl", "uk", "vi")

SOURCE_TASK = {
    "zero_shot": "zero_shot",
    "zero_shot_mini": "zero_shot",
    "zero_shot_delta": "zero_shot",
    "zero_shot_voxblink": "zero_shot_voxblink",
    "zero_shot_voxblink_mini": "zero_shot_voxblink",
    "zero_shot_voxblink_delta": "zero_shot_voxblink",
}


def langs_for(task):
    """The frozen language basket of a task (full set or one of its panels)."""
    src = SOURCE_TASK[task]
    return ITW_LANGS if src == "zero_shot_voxblink" else CLEAN_LANGS


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
# Silero VAD v6.2.3: speech segments for the paired ITW pause count (scripts/paired_itw.py).
SILERO_VAD = ("snakers4/silero-vad", "5cd7945676eb32225748052e2e6a0580e4686a08")


def asr_stamp(backend, model, revision):
    return f"{backend} {model}@{revision}"


def expected_asr_stamp(lang):
    backend = asr_backend(lang)
    return asr_stamp(backend, *ASR_MODELS[backend])


def spk_stamp(model_name, train_type, dataset):
    return f"{SPK_HUB_REPO}@{SPK_HUB_REF}:{model_name}:{train_type}:{dataset}"


EXPECTED_SPK_STAMP = spk_stamp(*SPK_VARIANT)


# --- Footer stamps ---------------------------------------------------------------
def footer_value(path, key, tail_bytes=4096):
    """Value of the LAST line `<key>: <value>` in the file's tail, else None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    prefix = f"{key}: "
    val = None
    for line in tail.splitlines():
        if line.startswith(prefix):
            val = line[len(prefix):].strip()
    return val


def leaf_problems(leaf_dir, lang):
    """Why a scored leaf is not current ZTTS2-Eval <BENCHMARK_VERSION>; [] if current."""
    out = []
    j = os.path.join

    def need(path, key, want):
        got = footer_value(path, key)
        if got != want:
            out.append(f"{path}: {key} is {got!r}, expected {want!r}")

    wer = j(leaf_dir, "wav_res_ref_text.wer")
    if os.path.exists(wer):
        need(wer, "BENCHMARK_VERSION", BENCHMARK_VERSION)
        need(wer, "ASR", expected_asr_stamp(lang))
    for name in ("spk_simi_scores.txt", "spk_simi_scores.xutt.txt"):
        p = j(leaf_dir, name)
        if os.path.exists(p):
            need(p, "#BENCHMARK_VERSION", BENCHMARK_VERSION)
            need(p, "#SPK_MODEL", EXPECTED_SPK_STAMP)
    if os.path.exists(j(leaf_dir, "msr_utmos_mean.txt")):
        ci = j(leaf_dir, "msr_utmos_ci.txt")
        need(ci, "BENCHMARK_VERSION", BENCHMARK_VERSION)
        need(ci, "MSR_UTMOS_CKPT", MSR_UTMOS_CKPT)
    return out


# --- Data helpers ----------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
