#!/usr/bin/env python3
"""Shared constants + helpers for the results_doc builders and plots.

One place for the things that used to drift as per-file copies:

  - the model display-label map (`MODEL_LABEL` / `label()`), previously five
    diverging variants across gen_md.py, build_perbin.py, build_mpm_dist.py,
    build_reliability.py, and the scripts/plot_* files;
  - system discovery from the results tree (`discover_systems()`), mirroring
    the rationale comment in build_perbin.py: hard-coded model lists are how
    caches came to silently hold only stale baselines;
  - the language display order (`ordered_langs()`), derived from the frozen
    baskets in utils/benchmark.py, previously seven diverging hard-coded lists;
  - atomic JSON writes (`write_json_atomic()`), so a crash mid-dump can never
    leave a truncated cache/snapshot file for the explorer to choke on.
"""
import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")

sys.path.insert(0, REPO_ROOT)
from utils.benchmark import CLEAN_LANGS, ITW_LANGS  # noqa: E402

# --- model labels -----------------------------------------------------------
# Optional pretty labels for well-known public models; any other model is shown
# with its results/ directory name minus the omni_ engine prefix. The zonos2_
# prefix is deliberately NOT stripped: the variant name alone ("accurate",
# "expressive", ...) is meaningless out of context. The two ground-truth
# baselines get distinct labels so tables that show both stay unambiguous.
MODEL_LABEL = {
    "omni_fish-s2-pro": "fish-s2-pro",
    "omni_qwen3-tts-1.7b": "qwen3-1.7b",
    "omni_qwen3-tts-0.6b": "qwen3-0.6b",
    "omni_voxcpm2": "voxcpm2",
    "omni_voxtral-tts": "voxtral",
    "omni_moss-tts-nano": "moss-nano",
    "gemini_3.1-flash": "gemini-3.1-flash",
    "cartesia_sonic-3.5": "cartesia-sonic-3.5",
    "elevenlabs_v3": "elevenlabs-v3",
    "inworld_2": "inworld",
    "gt_zero_shot": "gt (fleurs-r)",
    "gt_zero_shot_voxblink": "gt (voxblink)",
    "whisper_zero_shot": "whisper (fleurs-r)",
    "whisper_zero_shot_voxblink": "whisper (voxblink)",
}


def label(model_dir):
    """Display label for a results/ model directory name."""
    if model_dir in MODEL_LABEL:
        return MODEL_LABEL[model_dir]
    for pre in ("omni_", "gt_"):
        if model_dir.startswith(pre):
            return model_dir[len(pre):]
    return model_dir


# --- conditioning classes ---------------------------------------------------
# What each system is actually conditioned on. This is HARNESS POLICY, not a
# model property alone, and it differs per system — derived from the generation
# code, not from model cards:
#   - scripts/omni_models.sh reftext_policy(): which vLLM-Omni clone models get
#     `--no-ref-text` (audio-only) vs keep the reference transcript;
#   - scripts/generate_omni.py: `--x-vector-only` sets x_vector_only_mode=True
#     (the qwen3-tts Base task requires it to accept --no-ref-text);
#   - scripts/generate_zonos2.py: audio-only clone via speaker_audio_base64,
#     fixed --seed 0, vendor sampling temperature 1.15;
#   - the native generate_*.py recipes (kokoro/neutts/vibevoice/moss/cosyvoice).
#
# Classes:
#   with-transcript : clone conditioned on ref wav + its transcript
#   no-transcript   : clone conditioned on ref wav audio only
#   x-vector        : clone via speaker embedding only (x_vector_only_mode)
#   preset          : fixed built-in voice — NOT cloning the prompt speaker
#
# Seeding note: zonos2_* and zonos3-mini_* run with a fixed seed (0); the vLLM-Omni-served
# systems pass no seed/temperature (generate_omni.py defaults are None), i.e.
# unseeded server-default sampling; native recipes note their own defaults.
CONDITIONS = {
    # zonos2 server (scripts/generate_zonos2.py): audio-only clone.
    "zonos2_accurate": ("no-transcript",
                        "audio-only clone; accurate_mode=true; seed 0, temp 1.15 vendor sampling"),
    "zonos2_expressive": ("no-transcript",
                          "audio-only clone; accurate_mode=false; seed 0, temp 1.15 vendor sampling"),
    # ZONOS3 Mini via the release minisgl server (scripts/generate_zonos3.py -> /tts/generate): audio-only clone.
    "zonos3-mini_reference": ("no-transcript",
                              "release server, raw reference prefix (speaker_mode=reference), no transcript; seed 0, "
                              "T 0.75 top-k 80 top-p 0.69, rep 96×2.4, CFG 1, 0.2 s silence prefill, "
                              "server TN for en zh de es fr it pt ja ko"),
    "zonos3-mini_embedding": ("x-vector",
                              "release server, ownmect_s2 speaker embedding only (speaker_mode=embedding), "
                              "no transcript; sampling as zonos3-mini_reference"),
    # As above plus LACI (laci=true; alignment-guarded decoding with rollbacks), served from the z3m-laci
    # runtime tree (d24d592).
    "zonos3-mini-laci_reference": ("no-transcript",
                                   "as zonos3-mini_reference plus LACI (laci=true; ZTTS3-Eval assets/laci heads: "
                                   "zh heads for cmn, en heads for every other server language)"),
    "zonos3-mini-laci_embedding": ("x-vector",
                                   "as zonos3-mini_embedding plus LACI (laci=true; heads as zonos3-mini-laci_reference)"),
    # vLLM-Omni served (scripts/omni_models.sh reftext_policy + generate_omni.py);
    # all unseeded, server-default sampling.
    "omni_voxcpm2": ("no-transcript", "--no-ref-text (reftext_policy); unseeded server defaults"),
    "omni_omnivoice": ("no-transcript", "--no-ref-text (reftext_policy); unseeded server defaults"),
    "omni_qwen3-tts-0.6b": ("x-vector",
                            "--no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults"),
    "omni_qwen3-tts-1.7b": ("x-vector",
                            "--no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults"),
    "omni_omnivoice-reftext": ("with-transcript",
                               "same weights as omnivoice, ref_text kept (separate key); unseeded server defaults"),
    "omni_qwen3-tts-0.6b-reftext": ("with-transcript",
                                    "same weights as qwen3-tts-0.6b, ref_text kept (separate key); unseeded server defaults"),
    "omni_qwen3-tts-1.7b-reftext": ("with-transcript",
                                    "same weights as qwen3-tts-1.7b, ref_text kept (separate key); unseeded server defaults"),
    "omni_voxcpm2-reftext": ("with-transcript",
                             "same weights as voxcpm2, ref_text kept (separate key); unseeded server defaults"),
    "omni_fish-s2-pro": ("with-transcript",
                         "ref_text kept (server rejects no-ref-text); unseeded server defaults"),
    "omni_glm-tts": ("with-transcript",
                     "ref_text kept (default policy, no-ref-text unverified); unseeded server defaults"),
    "omni_higgs-audio-v3": ("with-transcript",
                            "ref_text kept (upstream prescribes ref_audio + ref_text); unseeded server defaults"),
    "omni_voxtral-tts": ("preset", "preset mode (task_type=CustomVoice); unseeded server defaults"),
    # Native in-process recipes.
    # Upstream SGLang-Omni adapter server (scripts/generate_audio8.py -> /v1/audio/speech).
    "audio8-tts-realistic-0.6b": ("with-transcript",
                                  "SGLang-Omni adapter (Audio8_TTS@4eb54a6, sglang-omni@68a5723); prompt wav + "
                                  "transcript; T 0.8 top-p 0.95 top-k 50, ≤1024 frames; unseeded server sampling"),
    "cosyvoice3": ("with-transcript",
                   "native upstream; prompt wav + transcript (endofprompt-terminated system prompt; ja→katakana)"),
    "moss-tts": ("no-transcript",
                 "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "moss-tts-v1.5": ("no-transcript",
                      "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "moss-tts-local": ("no-transcript",
                       "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "moss-tts-local-v1.5": ("no-transcript",
                            "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "vibevoice-1.5b": ("no-transcript",
                       "voice-prompt clone (voice_samples=[ref wav], no transcript); cfg 1.3, 10 DDPM steps"),
    "vibevoice-7b": ("no-transcript",
                     "voice-prompt clone (voice_samples=[ref wav], no transcript); cfg 1.3, 10 DDPM steps"),
    "kokoro": ("preset", "built-in voice per language; prompt wav ignored"),
    "neutts-2e": ("preset", "4 fixed speakers, deterministic per-utt rotation; prompt wav ignored"),
    "vibevoice-rt-0.5b": ("preset", "preset speaker embeddings, per-utt rotation; prompt wav ignored"),
}

# Result-table markers per class (with-transcript is the unmarked default).
COND_MARKER = {"no-transcript": "†", "x-vector": "◇", "preset": "‡"}
COND_LEGEND = ("† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · "
               "‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. "
               "See §1b.")


def condition(model_dir):
    """(class, provenance) for a results/ model dir, or None (gt_*/whisper_*).

    The moss-tts prefix fallback covers moss-tts variants without an explicit entry.
    """
    if model_dir in CONDITIONS:
        return CONDITIONS[model_dir]
    if model_dir.startswith(("gt_", "whisper_")):
        return None
    if model_dir.startswith("moss-tts"):
        return ("no-transcript", "audio-only clone (generate_moss_tts.py, no transcript)")
    return None


def marked_label(model_dir):
    """Display label plus the conditioning-class marker (†/◇/‡) — use this in
    every ranking table so no-transcript / x-vector / preset systems are never
    read against with-transcript ones undisclosed."""
    cond = condition(model_dir)
    if cond is None:
        return label(model_dir)
    return label(model_dir) + COND_MARKER.get(cond[0], "")


# --- system discovery -------------------------------------------------------
def discover_systems(task, root=RESULTS_ROOT):
    """{label: results-dir} for every model with a `<root>/<model>/<task>/` leaf.

    First directory wins on a label collision (sorted order), matching
    build_perbin.py's original behavior.
    """
    out = {}
    if not os.path.isdir(root):
        return out
    for mdir in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, mdir, task)):
            continue
        out.setdefault(label(mdir), mdir)
    return out


# --- language baskets -------------------------------------------------------
# Display order: the ITW basket first (it contains the core-9), then the Clean-only
# languages in basket order. Both baskets are frozen in utils/benchmark.py. Languages
# outside both sort alphabetically after these.
CANON_LANG_ORDER = list(ITW_LANGS) + [l for l in CLEAN_LANGS if l not in ITW_LANGS]


def ordered_langs(langs):
    """Sort a language collection into the canonical display order."""
    order = {l: i for i, l in enumerate(CANON_LANG_ORDER)}
    return sorted(langs, key=lambda l: (order.get(l, len(order)), l))


def data_langs(root_data):
    """Languages present under a data/<task>/ dir (those with a `text` file),
    in canonical display order — the language basket auto-grows with the data."""
    if not os.path.isdir(root_data):
        return []
    return ordered_langs(
        d for d in os.listdir(root_data)
        if os.path.isfile(os.path.join(root_data, d, "text")))

# --- snapshot access + the macro rule ---------------------------------------
# One loader for the snapshot build_results.py writes, shared by gen_md.py and
# plot_tradeoff.py so the published table and the published figure can never
# disagree on what a "macro" is.
SNAPSHOT_PATH = os.path.join(SCRIPT_DIR, "_cache", "results_snapshot.json")


def load_snapshot(path=SNAPSHOT_PATH):
    """The record list written by build_results.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_records(recs):
    """{(model, task, set): record}. gt_* baselines carry task == "" (flat dirs)."""
    return {(r["model"], r["task"], r["set"]): r for r in recs}


def rec_value(rec, field):
    """Scalar of a snapshot field, or None: WER_CLIP for `wer`; NaN counts as missing."""
    if rec is None:
        return None
    v = rec.get(field)
    if field == "wer":
        v = v[2] if v else None
    if v is None or v != v:
        return None
    return v


def lang_values(idx, model, task, field, langs):
    """Per-language values of `field` for a system, None where missing."""
    t = "" if model.startswith("gt_") else task
    return [rec_value(idx.get((model, t, l)), field) for l in langs]


def macro(idx, model, task, field, langs):
    """Unweighted mean over `langs` (the task's frozen basket, utils.benchmark.langs_for),
    defined ONLY when every basket language has the value: a mean over a different
    language subset is a different quantity and must not share a column with the full one."""
    vals = lang_values(idx, model, task, field, langs)
    if not vals or any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def n_covered(idx, model, task, field, langs):
    """How many basket languages have a value for `field`."""
    return sum(v is not None for v in lang_values(idx, model, task, field, langs))



# --- atomic JSON ------------------------------------------------------------
def write_json_atomic(path, obj, **json_kw):
    """json.dump to a temp file in the same directory, then os.replace().

    Readers (the explorer, plot scripts) can never observe a half-written
    file, and a crash leaves the previous version intact.
    """
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **json_kw)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
