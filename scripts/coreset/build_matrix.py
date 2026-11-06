#!/usr/bin/env python3
"""Assemble the system x item result matrix that coreset selection is built on.

Every scored leaf under ``results/`` retains per-utterance WER, speaker similarity and
MSR-UTMOS, and the uid axis is shared across systems for a given (task, lang). That
makes the full matrix reconstructible offline, so candidate panels can be selected and
validated without regenerating a single wav.

Outputs, per set (``clean`` = FLEURS-R ``clean``, ``itw`` = VoxBlink2
``in-the-wild``):

    assets/coreset/matrix_<set>.npz    wer_clip / wer / spk / msr, each float32 [S, N]
                                       with NaN for unscored cells, + ins/del/sub and
                                       the GT reference row
    assets/coreset/items_<set>.csv     one row per item: lang, uid, and the metadata the
                                       coverage quotas are defined over
    assets/coreset/systems_<set>.csv   one row per system: family, preset flag, role,
                                       and per-language leaf coverage

Cells are never silently dropped. A leaf that scored only part of its language stays in
the matrix as partial data, and its coverage is recorded in ``systems_<set>.csv`` so
downstream steps can apply a ``--min-leaf-coverage`` policy of their own. Reproducing
the published RESULTS tables needs coverage 0.0 (they averaged over whatever existed);
selection and validation want ~0.9.

    python scripts/coreset/build_matrix.py --set clean --set itw
    python scripts/coreset/build_matrix.py --set itw --verify   # reproduction gate
"""

import argparse
import csv
import os
import re
import sys
import unicodedata
import wave

import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

from scripts.plots.tufte_common import ref_token_count  # noqa: E402  exact run_wer.py tokenization
from scripts.coreset.common import METRICS  # noqa: E402
from utils.benchmark import leaf_dir  # noqa: E402
from utils.per_uid import load_leaf  # noqa: E402
from utils.scp import load_msr_utmos_cache, load_scp, load_text  # noqa: E402

OUT_DIR = os.path.join(REPO, "assets", "coreset")

SETS = {
    "clean": {"task": "clean", "gt": "gt_clean", "asr_alt": "whisper_clean"},
    "itw": {"task": "in-the-wild", "gt": "gt_in-the-wild",
            "asr_alt": "whisper_in-the-wild"},
}

# (key, results dir, family, preset-voice?, role)
#
# `family` drives the leave-family-out validation split -- variants of one base model
# share a family so a held-out fold is a genuinely novel system, not a sibling. `preset`
# marks engines that synthesize a fixed voice instead of cloning the prompt: their
# speaker-similarity cells are not a meaningful signal and are excluded from the spk
# response space (voxtral scores ~13 vs ~70 for cloning engines).
SYSTEMS = [
    ("zonos2_accurate",        "zonos2_accurate",        "zonos2",     False, "system"),
    ("zonos2_expressive",      "zonos2_expressive",      "zonos2",     False, "system"),
    ("qwen3-tts-1.7b",         "omni_qwen3-tts-1.7b",    "qwen3",      False, "system"),
    ("qwen3-tts-0.6b",         "omni_qwen3-tts-0.6b",    "qwen3",      False, "system"),
    ("voxcpm2",                "omni_voxcpm2",           "voxcpm",     False, "system"),
    ("fish-s2-pro",            "omni_fish-s2-pro",       "fish",       False, "system"),
    ("higgs-audio-v3",         "omni_higgs-audio-v3",    "higgs",      False, "system"),
    ("omnivoice-reftext",      "omni_omnivoice-reftext", "omnivoice",  False, "system"),
    ("moss-tts",               "moss-tts",               "moss",       False, "system"),
    ("cosyvoice3",             "cosyvoice3",             "cosyvoice3", False, "system"),
    ("voxtral-tts",            "omni_voxtral-tts",       "voxtral",    True,  "system"),
    # 2026-08-29 symmetric with-transcript arms (docs/controls_2026-08.md, C5). Same
    # weights as their no-ref siblings, ref_text kept -- they share a family so
    # leave-family-out validation never treats a conditioning variant as a novel
    # system. "partial": they are conditioning controls, not competitors, and the
    # qwen3 pair's `hi` leaves are legitimately under-coverage (runaway generation).
    ("qwen3-tts-1.7b-reftext", "omni_qwen3-tts-1.7b-reftext", "qwen3",  False, "partial"),
    ("qwen3-tts-0.6b-reftext", "omni_qwen3-tts-0.6b-reftext", "qwen3",  False, "partial"),
    ("voxcpm2-reftext",        "omni_voxcpm2-reftext",        "voxcpm", False, "partial"),
    # Partial coverage -- extra held-out rows for validation, on the langs they cover.
    ("omnivoice",              "omni_omnivoice",         "omnivoice",  False, "partial"),
    ("moss-tts-v1.5",          "moss-tts-v1.5",          "moss",       False, "partial"),
    ("moss-tts-local-v1.5",    "moss-tts-local-v1.5",    "moss",       False, "partial"),
    ("glm-tts",                "omni_glm-tts",           "glm",        False, "partial"),
    ("kokoro",                 "kokoro",                 "kokoro",     True,  "partial"),
    ("neutts-2e",              "neutts-2e",              "neutts",     True,  "partial"),
    ("vibevoice-rt-0.5b",      "vibevoice-rt-0.5b",      "vibevoice",  True,  "partial"),
    ("indextts2",              "omni_indextts2",         "indextts",   False, "partial"),
]

EDIT_PARTS = ("ins", "del", "sub")

_ABBREV_RE = re.compile(r"\b[A-Z]{2,}\b")
_DIGIT_RE = re.compile(r"\d")
_QUESTION_CHARS = "?؟？"


# ---------------------------------------------------------------------------
# item metadata (the axes the hard coverage quotas are defined over)
# ---------------------------------------------------------------------------
def latin_fraction(text):
    """Share of alphabetic characters that are Latin script.

    1.0 for a Latin-script language, ~0 for a native-script one; an intermediate value
    in a non-Latin language flags code-switching or embedded Latin tokens, which is a
    phenomenon the panels should keep represented.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return float("nan")
    latin = sum(1 for c in alpha if "LATIN" in unicodedata.name(c, ""))
    return latin / len(alpha)


def wav_duration(path):
    """Duration in seconds from the WAV header alone, or NaN if unreadable."""
    try:
        with wave.open(path, "rb") as fh:
            rate = fh.getframerate()
            return fh.getnframes() / rate if rate else float("nan")
    except (OSError, wave.Error, EOFError):
        return float("nan")


def itw_prompt_msr():
    """``{(lang, uid): prompt MSR-UTMOS}`` for the VoxBlink2 set.

    The materialized prompts do not record their source clip, so recover it by joining
    ``<lang>/prompt_text`` back to
    ``_stage_a/manifest.tsv`` on normalized prompt text, then look the clip up in the
    shipped ``_msr_utmos_cache.tsv``. This is the quality axis the set was stratified on
    (5 uniform bins over 1.0-4.0), so panels must keep it balanced.
    """
    data = os.path.join(REPO, "data", "in-the-wild")
    cache_path = os.path.join(data, "_msr_utmos_cache.tsv")
    man_path = os.path.join(data, "_stage_a", "manifest.tsv")
    if not (os.path.isfile(cache_path) and os.path.isfile(man_path)):
        return {}

    clip_score = load_msr_utmos_cache(cache_path)

    # Normalized prompt text is not a unique key: distinct clips can share a short
    # transcript. Keep EVERY clip per key so a collision can be detected instead of
    # silently attributing the first clip's score.
    man_by_key = {}
    with open(man_path) as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            cid = (row["speaker"], row["prompt_vid"], row["prompt_seg"],
                   float(row["prompt_start"]), float(row["prompt_end"]))
            man_by_key.setdefault((row["lang"], " ".join(row["prompt_text"].split())), set()).add(cid)

    out = {}
    n_match = n_unmatched = n_ambig = 0
    for lang in sorted(os.listdir(data)):
        ptext = os.path.join(data, lang, "prompt_text")
        if not os.path.isfile(ptext):
            continue
        for uid, txt in load_text(ptext, missing_ok=True).items():
            cids = man_by_key.get((lang, " ".join(txt.split())), ())
            scores = {clip_score[c] for c in cids if c in clip_score}
            if not scores:
                n_unmatched += 1
            elif len(scores) > 1:
                # The same (lang, text) maps to clips with DIFFERENT cached scores:
                # ambiguous join. Leave prompt_msr NaN rather than picking one clip
                # by manifest order -- the item then falls outside the quality strata
                # instead of silently landing in the wrong bin.
                n_ambig += 1
            else:
                out[(lang, uid)] = scores.pop()
                n_match += 1
    print(f"[itw] prompt MSR join: {n_match} matched, {n_unmatched} unmatched, "
          f"{n_ambig} ambiguous (left NaN)", flush=True)
    return out


def build_items(task, langs, prompt_msr):
    """Ordered item list + metadata, in the order `data/<task>/<lang>/text` defines."""
    rows = []
    for lang in langs:
        ldir = os.path.join(REPO, "data", task, lang)
        texts = load_text(os.path.join(ldir, "text"), missing_ok=True)
        prompts = load_scp(os.path.join(ldir, "prompt_wav.scp"), resolve=False, missing_ok=True)
        # Preserve file order (load_text keeps insertion order) for a deterministic axis.
        for uid, text in texts.items():
            pw = prompts.get(uid, "")
            pw_abs = pw if os.path.isabs(pw) else os.path.join(REPO, pw)
            rows.append({
                "lang": lang,
                "uid": uid,
                "n_ref_tok": ref_token_count(text, lang),
                "n_chars": len(text),
                "has_digit": int(bool(_DIGIT_RE.search(text))),
                "has_abbrev": int(bool(_ABBREV_RE.search(text))),
                "is_question": int(text.rstrip()[-1:] in _QUESTION_CHARS if text.strip() else 0),
                "latin_frac": latin_fraction(text),
                "prompt_dur_s": wav_duration(pw_abs) if pw else float("nan"),
                "prompt_msr": prompt_msr.get((lang, uid), float("nan")),
            })
    return rows


# ---------------------------------------------------------------------------
# matrix assembly
# ---------------------------------------------------------------------------
def build_set(set_name, verbose=True):
    cfg = SETS[set_name]
    task = cfg["task"]
    data_dir = os.path.join(REPO, "data", task)
    langs = sorted(d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_"))

    prompt_msr = itw_prompt_msr() if set_name == "itw" else {}
    if verbose:
        print(f"[{set_name}] {len(langs)} langs; reading item manifests + prompt headers…",
              flush=True)
    items = build_items(task, langs, prompt_msr)
    index = {(r["lang"], r["uid"]): i for i, r in enumerate(items)}
    n_items = len(items)
    lang_of = np.array([r["lang"] for r in items])

    rows = list(SYSTEMS)
    rows.append(("GT", cfg["gt"], "reference", False, "reference"))
    rows.append(("GT-whisper", cfg["asr_alt"], "reference", False, "reference"))

    mats = {m: np.full((len(rows), n_items), np.nan, dtype=np.float32) for m in METRICS}
    edits = {p: np.full((len(rows), n_items), np.nan, dtype=np.float32) for p in EDIT_PARTS}
    coverage = {}
    imputed = {}

    for si, (key, rdir, _family, _preset, _role) in enumerate(rows):
        n_cells = 0
        for lang in langs:
            d = leaf_dir(rdir, task, lang, root=REPO)
            if not os.path.isdir(d):
                continue
            leaf = load_leaf(d)
            if not leaf["wer"] and not leaf["spk"] and not leaf["msr"]:
                continue
            n_lang_items = int((lang_of == lang).sum())
            n_scored = dict.fromkeys(METRICS, 0)
            for metric in METRICS:
                for uid, val in leaf[metric].items():
                    j = index.get((lang, uid))
                    if j is not None:
                        mats[metric][si, j] = val
                        n_scored[metric] += 1
            for uid, det in leaf["detail"].items():
                j = index.get((lang, uid))
                if j is None:
                    continue
                for p in EDIT_PARTS:
                    if det[p] is not None:
                        edits[p][si, j] = det[p]
            # Coverage = the best-covered metric, not WER alone: a leaf whose spk/msr
            # have backfilled but whose .wer.raw has not is partially scored, not 0%.
            scored = max(n_scored.values())
            coverage[(key, lang)] = scored / n_lang_items if n_lang_items else 0.0
            # Imputed cells ARE covered -- an utterance the reference ASR could not
            # transcribe is scored as maximum error, not dropped (scripts/score/run_wer.py).
            # They are counted separately so systems_<set>.csv can say how much of a
            # row is measurement, which coverage alone can no longer distinguish.
            imputed[(key, lang)] = sum(
                1 for uid, det in leaf["detail"].items()
                if index.get((lang, uid)) is not None
                and (det["hyp_text"] or "").startswith("<UNSCOREABLE"))
            n_cells += scored
        if verbose:
            print(f"  {key:22s} {n_cells:7d} scored items", flush=True)

    return {
        "set": set_name, "task": task, "langs": langs, "items": items,
        "rows": rows, "mats": mats, "edits": edits, "coverage": coverage,
        "imputed": imputed,
    }


def write_outputs(built):
    os.makedirs(OUT_DIR, exist_ok=True)
    s = built["set"]
    items, rows = built["items"], built["rows"]

    npz = {f"m_{k}": v for k, v in built["mats"].items()}
    npz.update({f"e_{k}": v for k, v in built["edits"].items()})
    npz["systems"] = np.array([r[0] for r in rows])
    npz["families"] = np.array([r[2] for r in rows])
    npz["preset"] = np.array([r[3] for r in rows], dtype=bool)
    npz["roles"] = np.array([r[4] for r in rows])
    npz["lang"] = np.array([r["lang"] for r in items])
    npz["uid"] = np.array([r["uid"] for r in items])
    for f in ("n_ref_tok", "n_chars", "has_digit", "has_abbrev", "is_question",
              "latin_frac", "prompt_dur_s", "prompt_msr"):
        npz[f] = np.array([r[f] for r in items], dtype=np.float32)
    npz["task"] = np.array(built["task"])
    np.savez_compressed(os.path.join(OUT_DIR, f"matrix_{s}.npz"), **npz)

    with open(os.path.join(OUT_DIR, f"items_{s}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0].keys()))
        w.writeheader()
        w.writerows(items)

    with open(os.path.join(OUT_DIR, f"systems_{s}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "results_dir", "family", "preset", "role",
                    "n_langs_scored", "mean_leaf_coverage", "min_leaf_coverage",
                    "low_coverage_langs", "n_imputed", "imputed_langs"])
        for key, rdir, family, preset, role in rows:
            cov = {lg: c for (k, lg), c in built["coverage"].items() if k == key}
            imp = {lg: n for (k, lg), n in built["imputed"].items() if k == key and n}
            low = sorted(lg for lg, c in cov.items() if c < 0.9)
            w.writerow([key, rdir, family, int(preset), role, len(cov),
                        f"{np.mean(list(cov.values())):.4f}" if cov else "",
                        f"{min(cov.values()):.4f}" if cov else "",
                        " ".join(low), sum(imp.values()),
                        " ".join(f"{lg}:{n}" for lg, n in sorted(imp.items()))])

    print(f"[{s}] wrote matrix_{s}.npz  ({len(rows)} rows x {len(items)} items), "
          f"items_{s}.csv, systems_{s}.csv -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# reproduction gate: the matrix must regenerate the published tables
# ---------------------------------------------------------------------------
# Expected macro-averages: the published macro-averages recorded below (the clean
# 26-language tail after its 2026-07-30 correction; itw as updated 2026-08-29: the
# `yo` set was removed outright -- its transcripts are Nigerian ENGLISH mislabeled
# by Whisper LID at the enrichment stage -- so every itw basket is now 20 langs).
# Per system: (Δ vs GT, WER_CLIP, spk, MSR, n_langs).
# Update these DELIBERATELY after a rescore that is meant to move published numbers.
#
# 2026-09-01: refreshed after the unscoreable-generation backfill
# (docs/unscoreable_2026-09.md). Every leaf that had unscored utterances was
# re-measured and what still would not transcribe is now scored as maximum error,
# so a handful of macro means move by 0.01-0.07 and the clean GT floor by 0.01. No
# ranking changes; the movements are small here because a macro averages 20-35
# languages, while the per-language cells behind them move by up to ~6 points.
VERIFY_EXPECTED = {
    "clean": {
        "gt_floor": 11.85,
        "systems": {
            "zonos2_accurate":   (-3.9,  7.98, 75.7, 3.09, 26),
            "omnivoice-reftext": (-2.7,  9.16, 87.8, 3.03, 26),
            "higgs-audio-v3":    (-0.6, 11.27, 79.9, 2.97, 26),
            "fish-s2-pro":       (+2.0, 13.80, 81.3, 3.15, 26),
            "voxcpm2":           (+5.3, 17.19, 79.3, 2.88, 26),
            "moss-tts":          (+6.4, 18.29, 83.7, 3.05, 26),
            "zonos2_expressive": (+6.8, 18.69, 66.9, 2.96, 26),
            "qwen3-tts-1.7b":    (+20.3, 32.15, 67.9, 3.52, 26),
            "qwen3-tts-0.6b":    (+28.0, 39.83, 66.8, 3.53, 26),
            "voxtral-tts":       (+49.5, 61.32, 16.0, 3.32, 26),
            "cosyvoice3":        (+60.1, 71.94, 72.9, 3.26, 26),
        },
    },
    "itw": {
        # GT floor 9.38 -> 9.34: `en` is now scored by Whisper (control C1), whose
        # English GT floor is higher (4.45 vs Qwen's 3.78) but unbiased between
        # systems -- see docs/controls_2026-08.md. Every Δ shifts ~0.07 with it.
        "gt_floor": 9.34,
        "systems": {
            "zonos2_accurate":   (-4.6,  4.71, 57.8, 2.79, 20),
            "omnivoice-reftext": (-4.2,  5.11, 75.0, 2.82, 20),
            "voxcpm2":           (-4.0,  5.36, 73.9, 2.31, 20),
            "fish-s2-pro":       (-3.3,  6.06, 70.4, 2.64, 20),
            "moss-tts":          (-0.0,  9.33, 72.9, 2.54, 20),
            "zonos2_expressive": (+0.3,  9.62, 56.9, 2.46, 20),
            "qwen3-tts-1.7b":    (+2.4, 11.75, 67.2, 2.84, 20),
            "higgs-audio-v3":    (+3.4, 12.76, 69.3, 2.34, 20),
            "qwen3-tts-0.6b":    (+5.8, 15.16, 64.3, 2.84, 20),
            "cosyvoice3":        (+33.6, 42.93, 73.0, 2.77, 20),
            "voxtral-tts":       (+34.0, 43.29, 13.8, 3.39, 20),
        },
    },
}
# Absolute tolerances, sized to the precision the expected values were recorded at.
VERIFY_TOL = {"dvsgt": 0.06, "wer_clip": 0.05, "spk": 0.1, "msr": 0.01}


def verify(built):
    """Recompute the published macro-averages straight from the matrix and GATE on them.

    This is the correctness gate for everything downstream: if the matrix cannot
    reproduce the published macro-averages recorded below, no panel selected from it can
    be trusted. Aggregation mirrors scripts/results_doc/gen_md.py -- per-language means,
    then an unweighted macro-average over languages (no utterance weighting). Every
    system is checked against VERIFY_EXPECTED within VERIFY_TOL; returns True when all
    pass, False otherwise (main() exits non-zero on any failure).

    "Δ vs GT" is computed against GT restricted to the languages that system actually
    covers. A system missing a language must not be charged (or credited) with the GT
    floor of a language it never ran -- and the matched form is what the published
    tables used.
    """
    s = built["set"]
    langs, rows = built["langs"], built["rows"]
    lang_of = np.array([r["lang"] for r in built["items"]])
    keys = [r[0] for r in rows]

    if s == "clean":
        base9 = {"en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru"}
        scope = [lg for lg in langs if lg not in base9]  # the 26 tail langs
        title = "clean — 26 tail langs"
    else:
        scope = langs  # all 20, tl included (yo removed 2026-08-29)
        title = "itw — 20 langs"

    masks = {lg: (lang_of == lg) for lg in scope}

    def per_lang(metric, si):
        out = {}
        for lg in scope:
            v = built["mats"][metric][si, masks[lg]]
            v = v[~np.isnan(v)]
            if v.size:
                out[lg] = float(v.mean())
        return out

    expected = VERIFY_EXPECTED[s]
    failures = []

    gt_by_lang = per_lang("wer_clip", keys.index("GT"))
    gt_floor = float(np.mean(list(gt_by_lang.values())))
    print(f"\n=== {s}: {title} ===")
    print(f"GT floor mean WER_CLIP over {len(gt_by_lang)} langs = {gt_floor:.2f}\n")
    if abs(gt_floor - expected["gt_floor"]) > VERIFY_TOL["wer_clip"]:
        failures.append(f"GT floor: got {gt_floor:.2f}, expected {expected['gt_floor']:.2f}")
    print(f"{'system':22s} {'ΔvsGT':>7s} {'WER_CLIP':>9s} {'spk':>6s} {'MSR':>6s} {'langs':>6s}")
    order = []
    for si, key in enumerate(keys):
        if rows[si][4] != "system":
            continue
        w = per_lang("wer_clip", si)
        order.append((np.mean(list(w.values())) if w else np.inf, key, si, w))
    seen = set()
    for wm, key, si, w in sorted(order):
        covered = sorted(w)
        gt_m = np.mean([gt_by_lang[lg] for lg in covered if lg in gt_by_lang])
        spk = per_lang("spk", si)
        msr = per_lang("msr", si)
        spk_m = np.mean(list(spk.values())) if spk else float("nan")
        msr_m = np.mean(list(msr.values())) if msr else float("nan")
        gate = ""
        exp = expected["systems"].get(key)
        seen.add(key)
        if exp is not None:
            got = {"dvsgt": wm - gt_m, "wer_clip": wm, "spk": spk_m, "msr": msr_m}
            want = dict(zip(("dvsgt", "wer_clip", "spk", "msr"), exp))
            bad = [f"{f} {got[f]:.2f}≠{want[f]:.2f}" for f in got
                   if not (np.isfinite(got[f]) and abs(got[f] - want[f]) <= VERIFY_TOL[f])]
            if len(covered) != exp[4]:
                bad.append(f"n_langs {len(covered)}≠{exp[4]}")
            if bad:
                gate = "  FAIL: " + ", ".join(bad)
                failures.append(f"{key}: " + ", ".join(bad))
        else:
            gate = "  (no expected values -- new system?)"
        print(f"{key:22s} {wm - gt_m:+7.1f} {wm:9.2f} {spk_m:6.1f} {msr_m:6.2f} "
              f"{len(covered):6d}{gate}")
    for key in sorted(set(expected["systems"]) - seen):
        failures.append(f"{key}: expected in the matrix but produced no scored row")

    print("\nExpected values are the committed RESULTS_*.md tables (as updated 2026-08-29: "
          "yo removed as mislabeled English -> 20-lang basket for every system).")
    if failures:
        print(f"\n[{s}] REPRODUCTION GATE FAILED — the matrix does not reproduce the "
              f"published tables:")
        for f in failures:
            print(f"  {f}")
        print("If a rescore was MEANT to change the published numbers, update "
              "VERIFY_EXPECTED in scripts/coreset/build_matrix.py alongside the tables.")
    else:
        print(f"\n[{s}] reproduction gate passed "
              f"({len(expected['systems'])} systems within tolerance).")
    return not failures


def load_built(set_name):
    """Rebuild the in-memory view from a previously written matrix_<set>.npz.

    Lets --verify re-run in a second instead of re-reading every leaf on disk.
    """
    d = np.load(os.path.join(OUT_DIR, f"matrix_{set_name}.npz"), allow_pickle=True)
    items = [{"lang": lg, "uid": u} for lg, u in zip(d["lang"], d["uid"])]
    rows = list(zip(d["systems"].tolist(), d["systems"].tolist(), d["families"].tolist(),
                    d["preset"].tolist(), d["roles"].tolist()))
    return {
        "set": set_name, "task": str(d["task"]),
        "langs": sorted(set(d["lang"].tolist())), "items": items, "rows": rows,
        "mats": {m: d[f"m_{m}"] for m in METRICS},
        "edits": {p: d[f"e_{p}"] for p in EDIT_PARTS},
        "coverage": {},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", choices=sorted(SETS),
                    help="which set(s) to build (repeatable; default both)")
    ap.add_argument("--verify", action="store_true",
                    help="print the reproduction-gate table instead of only writing files")
    ap.add_argument("--paired-ci", action="store_true",
                    help="after the verify table, print the adjacent-pair paired ΔCI95 "
                         "table (scripts/analysis/paired_ci.py) from the on-disk matrix_<set>.npz")
    ap.add_argument("--no-write", action="store_true", help="verify only, write nothing")
    ap.add_argument("--from-cache", action="store_true",
                    help="verify from the existing matrix_<set>.npz without re-reading results/")
    args = ap.parse_args()

    failed = []
    for s in (args.sets or sorted(SETS)):
        if args.from_cache:
            if not verify(load_built(s)):
                failed.append(s)
        else:
            built = build_set(s)
            if not args.no_write:
                write_outputs(built)
            if (args.verify or args.no_write) and not verify(built):
                failed.append(s)
        if args.paired_ci:
            # Reads matrix_<set>.npz from disk (freshly written above unless --no-write),
            # so the CI table always describes the same matrix the verify gate ran on.
            from scripts.coreset.common import Matrix  # noqa: PLC0415
            from scripts.analysis.paired_ci import print_adjacent  # noqa: PLC0415
            print_adjacent(Matrix(s))
    if failed:
        sys.exit(f"error: reproduction gate failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
