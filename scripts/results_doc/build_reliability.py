#!/usr/bin/env python3
"""Build the explorer's reliability payload — the interactive twin of the
scripts/tufte_* reliability plates.

The tufte_* plots (p99 CI, survival overlay, small multiples, slopegraph) all
derive from one thing: per-series arrays of integer edit counts reconstructed
from the GT `wav_res_ref_text.wer.raw` leaves (see scripts/tufte_common.py,
which we import so the tokenisation — and thus the edit counts — match exactly).

This collapses those arrays into a compact, portable JSON the browser can render
without numpy: per (dataset, series) a survival curve P(edits >= k), the p99 with
a bootstrap 95% CI, the worst single utterance, n, and the slopegraph threshold
values. Datasets whose .raw inputs are absent are simply skipped (same guard as
run_reliability.sh).

    _cache/reliability.json   ->  published to explorer/snapshot/ (publish_snapshot.py)

Run from the pipeline (run.sh) or standalone:
    python scripts/results_doc/build_reliability.py
"""
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
CACHE_DIR = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(CACHE_DIR, "reliability.json")

# tufte_common lives in scripts/ — reuse its exact loader + tokenisation.
sys.path.insert(0, SCRIPTS_DIR)
import tufte_common as tc  # noqa: E402
sys.path.insert(0, SCRIPT_DIR)
from rd_common import label as model_label, write_json_atomic  # noqa: E402

KS = [1, 3, 5, 10]          # slopegraph severity thresholds (match tufte_slopegraph)
BOOT = 2000                 # bootstrap resamples for the p99 CI (match tufte_p99)

# One entry per plate "set". `globs`/`label`/`rename` mirror run_reliability.sh.
# kind "gt": series are languages / backends of the ground-truth recordings.
DATASETS = [
    {
        "key": "clean", "kind": "gt",
        "label": "FLEURS-R — read speech",
        "globs": ["results/gt_clean/*/wav_res_ref_text.wer.raw"],
        "label_scheme": "lang", "rename": None,
    },
    {
        "key": "in-the-wild", "kind": "gt",
        "label": "VoxBlink2 — in the wild",
        "globs": ["results/gt_in-the-wild/*/wav_res_ref_text.wer.raw"],
        "label_scheme": "lang", "rename": None,
    },
    {
        "key": "backend", "kind": "gt",
        "label": "ASR backend — English GT",
        "globs": ["results/gt_clean/en/wav_res_ref_text.wer.raw",
                  "results/whisper_clean/en/wav_res_ref_text.wer.raw"],
        "label_scheme": "set",
        "rename": "gt_clean=Qwen3-ASR,whisper_clean=Whisper",
    },
]

# Model-wise reliability: one series per TTS system, from the model's own
# re-transcribed WER leaves (results/<engine>_<model>/clean/<lang>/...).
# A single "models" dataset with an all-languages picker:
#   "All languages" (the default `series`) aggregates over the 9-language basket
#     every "complete" run shares — restricted to systems covering all 9, so the
#     aggregate always mixes the identical languages (the coverage confound).
#   each language slice (`bylang`) then holds every system that ran that language,
#     including partial runs (glm-tts, omnivoice-reftext, …) where they have data.
# Every view carries a ground-truth floor row (the ASR-reconstruction limit on
# the same languages) for reference.
MODEL_LANGS_9 = ["de", "en", "es", "fr", "it", "ja", "ko", "ru", "zh"]
MODEL_ENGINE_GLOBS = ["results/*"]
MODEL_MIN_N = 200           # drop smoke-test leaves (n≈3–8) — not measurements


def _has_inputs(globs):
    for g in globs:
        if glob.glob(g) or glob.glob(os.path.join(REPO_ROOT, g)):
            return True
    return False


def series_payload(e, rng):
    """Compact per-series record from an int edit-count array `e`."""
    ks, s = tc.survxy(e)                                  # P(edits >= k) %, tail>0
    p99 = float(np.percentile(e, 99))
    bs = np.array([np.percentile(rng.choice(e, size=len(e), replace=True), 99)
                   for _ in range(BOOT)])
    slope = [round(float((e >= k).mean() * 100), 3) for k in KS]
    return {
        "n": int(e.size),
        "p99": round(p99, 1),
        "ci": [round(float(np.percentile(bs, 2.5)), 1),
               round(float(np.percentile(bs, 97.5)), 1)],
        "worst": int(e.max()) if e.size else 0,
        # survival curve as parallel arrays keeps the JSON small
        "k": [int(x) for x in ks],
        "s": [round(float(v), 3) for v in s],
        "slope": slope,
    }


def _assemble(key, label, kind, E, meta, rng):
    """Turn a {name: edit-array} mapping into a dataset dict, or None if empty."""
    if not E:
        return None
    series = []
    for name, e in E.items():
        rec = series_payload(e, rng)
        rec["name"] = name
        rec["langs"] = meta[name]["langs"]
        series.append(rec)
    # stable order: most reliable (lowest p99) first — the UI re-sorts per view
    series.sort(key=lambda r: r["p99"])
    total = sum(r["n"] for r in series)
    langs = sorted({l for r in series for l in r["langs"]})
    return {
        "key": key, "kind": kind, "label": label,
        "n_series": len(series), "n_utts": total, "n_langs": len(langs),
        "series": series,
    }


def _load_models(langs, require_full, task="clean"):
    """Per-model edit arrays over a fixed `langs` basket, for one generation task
    (`clean` = FLEURS-R read speech, `in-the-wild` = in-the-wild).

    Returns (E, meta) keyed by a display name (engine prefix stripped), plus a
    ground-truth floor series ("ground truth") over the same languages. When
    `require_full`, models missing any basket language are dropped so the
    aggregate always mixes the identical set of languages."""
    m_globs = [f"{eng}/{task}/{l}/wav_res_ref_text.wer.raw"
               for eng in MODEL_ENGINE_GLOBS for l in langs]
    E, meta = ({}, {})
    if _has_inputs(m_globs):
        E, meta = tc.load_series(m_globs, "model", None, None, 0)
    want = set(langs)
    out_E, out_meta = {}, {}
    for name in E:
        have = {tc.base_lang(l) for l in meta[name]["langs"]}
        if require_full and not want <= have:
            print(f"[reliability]   drop {name}: covers {sorted(have)} not {langs}",
                  file=sys.stderr)
            continue
        if E[name].size < MODEL_MIN_N:
            print(f"[reliability]   drop {name}: n={E[name].size} < {MODEL_MIN_N} "
                  "(smoke, not a run)", file=sys.stderr)
            continue
        # shared display-label map (common.py) — the old inline strip tested a
        # "zonos_" prefix that matched no results dir, so zonos2_* kept full names
        disp = model_label(name)
        out_E[disp], out_meta[disp] = E[name], meta[name]
    # ground-truth floor on the same basket (gt_<task>/<lang>, one level shallower)
    gt_globs = [f"results/gt_{task}/{l}/wav_res_ref_text.wer.raw" for l in langs]
    if _has_inputs(gt_globs):
        gE, gmeta = tc.load_series(gt_globs, "set", None,
                                   f"gt_{task}=ground truth", 0)
        out_E.update(gE)
        out_meta.update(gmeta)
    return out_E, out_meta


def _all_model_langs(task="clean"):
    """Every language any model actually produced for `task`, ordered basket-first."""
    files = []
    for eng in MODEL_ENGINE_GLOBS:
        files += glob.glob(os.path.join(REPO_ROOT, eng, task, "*",
                                        "wav_res_ref_text.wer.raw"))
    langs = {tc.base_lang(os.path.basename(os.path.dirname(f))) for f in files}
    return [l for l in MODEL_LANGS_9 if l in langs] + \
        sorted(langs - set(MODEL_LANGS_9))


def _model_basket(task):
    """The fair aggregate basket for `task`: the MODEL_LANGS_9 languages that at
    least two systems produced. On the full FLEURS-R sweep this is all 9; on a
    partial run (e.g. an in-the-wild English-only smoke) it shrinks to what exists, so
    the aggregate never demands languages no system has."""
    per_model = {}
    for eng in MODEL_ENGINE_GLOBS:
        for f in glob.glob(os.path.join(REPO_ROOT, eng, task, "*",
                                        "wav_res_ref_text.wer.raw")):
            # <engine>_<model>/<task>/<lang>/wav_res_ref_text.wer.raw
            model = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
            lang = tc.base_lang(os.path.basename(os.path.dirname(f)))
            per_model.setdefault(model, set()).add(lang)
    return [l for l in MODEL_LANGS_9
            if sum(1 for s in per_model.values() if l in s) >= 2]


def _model_dataset(key, label, rng, task="clean"):
    """Model-wise dataset for one generation task: a fair cross-language aggregate
    plus every language.

    `series` is the "All languages" aggregate over the fair basket (the MODEL_LANGS_9
    languages ≥2 systems produced for this task), restricted to systems covering the
    whole basket so it always mixes the identical set of languages (the coverage
    confound). `bylang` then breaks the result out for *every* language any model
    produced — each slice holding every system that ran that language (smoke leaves
    excluded), not just the systems in the aggregate — so the picker shows
    partial-coverage systems where they have data. `langs_list` is the picker order."""
    agg_langs = _model_basket(task)
    if not agg_langs:
        return None
    E, meta = _load_models(agg_langs, require_full=True, task=task)
    ds = _assemble(key, label, "model", E, meta, rng)
    if ds is None:
        return None
    bylang, order = {}, []
    for l in _all_model_langs(task):
        El, _ = _load_models([l], require_full=False, task=task)
        if not El:
            continue
        recs = []
        for n, a in El.items():
            rec = series_payload(a, rng)
            rec["name"] = n
            rec["langs"] = [l]
            recs.append(rec)
        recs.sort(key=lambda r: r["p99"])
        bylang[l] = recs
        order.append(l)
    if len(order) > 1:
        ds["bylang"] = bylang
        ds["langs_list"] = order
    return ds


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rng = np.random.default_rng(0)                        # seeded → reproducible CIs
    # The TTS-system datasets are listed first (the explorer's chip order, and it opens
    # on the first), but computed after the GT ones: they share one seeded rng, so the
    # computation order fixes every bootstrap CI and must not move.
    out = {"ks": KS, "datasets": []}
    models = []
    for ds in DATASETS:
        if not _has_inputs(ds["globs"]):
            print(f"[reliability] {ds['key']}: no .raw inputs — skipped", file=sys.stderr)
            continue
        E, meta = tc.load_series(ds["globs"], ds["label_scheme"], None, ds["rename"], 0)
        d = _assemble(ds["key"], ds["label"], ds["kind"], E, meta, rng)
        if d is None:
            print(f"[reliability] {ds['key']}: loaded 0 series — skipped", file=sys.stderr)
            continue
        out["datasets"].append(d)
        print(f"[reliability] {ds['key']}: {d['n_series']} series · {d['n_utts']:,} utts")

    # model-wise datasets: one series per TTS system, with an all-languages picker.
    # One per generation task that has model outputs (read speech + in-the-wild).
    for key, label, task in [
        ("models", "TTS systems — read speech", "clean"),
        ("models_in-the-wild", "TTS systems — in the wild", "in-the-wild"),
    ]:
        d = _model_dataset(key, label, rng, task=task)
        if d is None:
            print(f"[reliability] {key}: no model inputs — skipped", file=sys.stderr)
            continue
        models.append(d)
        nl = f" · per-lang {len(d['langs_list'])}" if d.get("langs_list") else ""
        print(f"[reliability] {key}: {d['n_series']} series · {d['n_utts']:,} utts{nl}")

    out["datasets"] = models + out["datasets"]

    if not out["datasets"]:
        print("[reliability] no datasets had inputs — writing empty payload", file=sys.stderr)
    write_json_atomic(OUT, out, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)} "
          f"({os.path.getsize(OUT) / 1024:.0f} KB, {len(out['datasets'])} datasets)")


if __name__ == "__main__":
    main()
