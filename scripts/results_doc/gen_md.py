#!/usr/bin/env python3
"""Generate RESULTS_CURRENT.md from the JSON snapshot built by build_results.py.

Model-agnostic: discovers whichever models are present in the snapshot (no
hardcoded checkpoint list) and emits, for the two ZTTS2-Eval sets -- clean
(Clean, FLEURS-R, 35 languages) and in-the-wild (ITW, VoxBlink2, 20
languages) -- a headline section (accuracy-vs-prosody figure plus one table per
conditioning class), per-language metric tables, and the methodology notes.

Every headline number is a macro over the task's frozen basket
(utils/benchmark.py), defined only when the system covers every basket language
(rd_common.macro); partial systems are listed separately, never ranked.

Run `scripts/results_doc/build_results.py` first to (re)build the snapshot from a
populated `results/` tree, or use `scripts/results_doc/run.sh` for the full chain.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT = os.path.join(REPO_ROOT, "RESULTS_CURRENT.md")
CAP_INDEX = os.path.join(SCRIPT_DIR, "_cache", "capability_index.json")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from utils.benchmark import (  # noqa: E402
    ASR_MODELS, BENCHMARK_VERSION, CLEAN_LANGS, ITW_LANGS, QWEN_LANGS, SPK_HUB_REF,
    SPK_HUB_REPO, SPK_VARIANT, data_uids, langs_for)
# Shared display-label map, conditioning classes, snapshot loader and macro rule.
from rd_common import (  # noqa: E402
    COND_LEGEND, COND_MARKER, condition, index_records, label, load_snapshot,
    macro, marked_label, n_covered, rec_value)

CLEAN, ITW = "clean", "in-the-wild"
GT_NAME = {CLEAN: "gt_clean", ITW: "gt_in-the-wild"}

# Conditioning classes in headline order: title, marker.
CLASS_ORDER = (
    ("with-transcript", "with-transcript clones"),
    ("no-transcript", f"no-transcript clones {COND_MARKER['no-transcript']}"),
    ("x-vector", f"x-vector-only clones {COND_MARKER['x-vector']}"),
    ("preset", f"preset voices {COND_MARKER['preset']} (not cloning the prompt)"),
    ("unclassified", "unclassified (not in `rd_common.CONDITIONS`)"),
)


# --- formatting --------------------------------------------------------------
def fmt(x, decimals=2, default="—"):
    if x is None:
        return default
    try:
        if x != x:
            return default  # NaN
        return f"{x:.{decimals}f}"
    except Exception:
        return default


def fmt_wer(rec, key="wer"):
    if rec is None or rec.get(key) is None:
        return "—"
    v = rec[key][2]
    return "n/a" if v != v else f"{v:.2f}"


def fmt_l50(rec, key="wer"):
    if rec is None or rec.get(key) is None:
        return "—"
    v = rec[key][1]
    return "n/a" if v != v else f"{v:.2f}"


def _sort_key(m):
    # gt_* last (reference rows), omni_* first, everything else in the middle.
    bucket = 0 if m.startswith("omni_") else (2 if m.startswith("gt_") else 1)
    return (bucket, label(m))


def models_for(recs, task):
    """All non-baseline models that have at least one row for `task`, stable order."""
    return sorted({r["model"] for r in recs if r["task"] == task}, key=_sort_key)


def cls_of(model):
    cond = condition(model)
    return cond[0] if cond else "unclassified"


# --- headline columns ---------------------------------------------------------
# (header, value fn(idx, model, task, langs) -> float | None, decimals)
def _m(field):
    return lambda idx, m, t, langs: macro(idx, m, t, field, langs)


def _copy_gap(idx, m, t, langs):
    p, h = macro(idx, m, t, "spk", langs), macro(idx, m, t, "spk_xutt", langs)
    return None if p is None or h is None else p - h


_PROSODY = [("Rhythm pace ×ceil", _m("rhythm_rate_x"), 2),
            ("Rhythm pause ×ceil", _m("rhythm_x"), 2),
            ("Pitch range ×ceil", _m("pitch_range_x"), 2)]
HEADLINE_COLS = {
    CLEAN: [("WER_CLIP", _m("wer"), 2), ("Tail >50%", _m("tail50"), 1),
            ("SPK (prompt)", _m("spk"), 1), ("MSR-UTMOS", _m("msr_utmos"), 2)]
           + _PROSODY + [("MPM span (Clean only)", _m("mpm_span"), 2)],
    ITW: [("WER_CLIP", _m("wer"), 2), ("Tail >50%", _m("tail50"), 1),
          ("SPK held-out", _m("spk_xutt"), 1), ("SPK prompt", _m("spk"), 1),
          ("Copy gap (prompt − held-out)", _copy_gap, 1),
          ("MSR-UTMOS", _m("msr_utmos"), 2)]
         + _PROSODY + [("Dur |log2|", _m("paired_dur"), 2),
                       ("Pause Δ", _m("paired_pause"), 2),
                       ("Pitch-range |log2|", _m("paired_range"), 2)],
}


def _headline_row(idx, m, task, langs, cols, name=None):
    cells = [fmt(fn(idx, m, task, langs), nd) for _h, fn, nd in cols]
    return "| " + " | ".join([name or marked_label(m)] + cells) + " |"


def _headline_header(cols):
    # `|` inside a header cell (e.g. "Dur |log2|") must be escaped or it splits the column.
    heads = ["System"] + [h.replace("|", "\\|") for h, _f, _n in cols]
    return ["| " + " | ".join(heads) + " |", "|---|" + "|".join(["--:"] * len(cols)) + "|"]


def dedup_counts():
    """{task: (n_before, n_after)} from data/near_duplicates_removed.json + data/, or None."""
    try:
        with open(os.path.join(REPO_ROOT, "data", "near_duplicates_removed.json"),
                  encoding="utf-8") as f:
            removed = json.load(f)["removed"]
        out = {}
        for task in (CLEAN, ITW):
            n_rem = sum(len(v) for k, v in removed.items() if k.split("/", 1)[0] == task)
            tdir = os.path.join(REPO_ROOT, "data", task)
            n_now = sum(len(data_uids(task, l)) for l in sorted(os.listdir(tdir))
                        if os.path.isfile(os.path.join(tdir, l, "text")))
            out[task] = (n_now + n_rem, n_now, n_rem)
        return out
    except (OSError, ValueError, KeyError):
        return None


SPK_EMBEDDER_NOTE = (
    "> Speaker-similarity gaps below ~1 point are embedder-dependent: a held-out "
    "embedder control (vox2-only ReDimNet, no VoxBlink2 in training) preserved the "
    "system ordering overall (Kendall tau +0.91...+0.98) but reversed one "
    "published pair on `de`. Read small spk differences as ties -- see "
    "docs/controls_2026-08.md."
)


def render(recs, dedup=None, cap_index=None):
    """RESULTS_CURRENT.md text for a snapshot (list of records) and, when built, the
    capability index payload (build_capability_index.py)."""
    idx = index_records(recs)
    buf = []
    W = buf.append

    def emit_table_header(first_col, cols):
        W("| " + first_col + " | " + " | ".join(marked_label(m) for m in cols) + " |")
        W("|---|" + "|".join(["--:"] * len(cols)) + "|")

    def per_lang_table(models, task_of, langs, cell):
        emit_table_header("Lang", models)
        for lang in langs:
            W("| " + " | ".join([lang] + [cell(idx.get((m, task_of(m), lang))) for m in models]) + " |")

    def avg_row(models, task_of, task, langs):
        row = ["**avg**"]
        for m in models:
            n = n_covered(idx, m, task, "wer", langs)
            a = macro(idx, m, task, "wer", langs)
            row.append("—" if n == 0 else f"**{a:.2f}**" if a is not None
                       else f"partial ({n}/{len(langs)})")
        W("| " + " | ".join(row) + " |")

    # --- header -------------------------------------------------------------
    W("# ZTTS2-Eval — Current Results Snapshot\n")
    W(f"Benchmark version {BENCHMARK_VERSION}\n")
    W("Regenerated from the `results/` directory tree by")
    W("`scripts/results_doc/run.sh`. Numbers are read directly from")
    W("`wav_res_ref_text.wer`, `spk_simi_scores.txt`, `spk_simi_scores.xutt.txt` (ITW),")
    W("`msr_utmos_mean.txt` and the rhythm / pitch-range / paired-ITW CSVs. **WER_CLIP**")
    W("(per-utt WER capped at 100% before averaging) is the robust mean; raw WER, WER_L50")
    W("and the tail rate (share of utterances above 50% WER) are also tracked. Speaker")
    W("similarity is cosine × 100 (ReDimNet); MSR-UTMOS is the MOS predictor.\n")
    W("> Scores are not comparable with ZTTS1-Eval.\n")
    W("What the scores in this document reflect:\n")
    asr_q, asr_w = ASR_MODELS["qwen"], ASR_MODELS["whisper"]
    W(f"- per-language ASR routing, enforced: {', '.join(f'`{l}`' for l in sorted(QWEN_LANGS))} "
      f"on `{asr_q[0]}@{asr_q[1][:7]}`, every other language on `{asr_w[0]}@{asr_w[1][:7]}`; "
      "a leaf scored with any other ASR is refused by the build;")
    W("- digits expanded to words on both reference and hypothesis before WER;")
    if dedup:
        c, i = dedup.get(CLEAN), dedup.get(ITW)
        W("- prompt/target near-duplicates removed "
          f"(Clean {c[2]} removed, {c[0]:,} → {c[1]:,}; "
          f"ITW {i[2]} removed, {i[0]:,} → {i[1]:,}; "
          "list in `data/near_duplicates_removed.json`);")
    else:
        W("- prompt/target near-duplicates removed (list in `data/near_duplicates_removed.json`);")
    W(f"- all {len(ITW_LANGS)} ITW languages in the aggregates, including `tl`;")
    W("- ITW speaker similarity against a held-out recording of the same speaker "
      "(prompt similarity kept as a secondary column);")
    W("- pitch measured as a register-normalized pitch-range distance.\n")

    # --- 0. Headline ----------------------------------------------------------
    W("## 0. Headline\n")
    W("Macro = unweighted mean over the set's language basket "
      f"({len(CLEAN_LANGS)} Clean, {len(ITW_LANGS)} ITW); a system is ranked only when it "
      "covers every language of the basket. Within a class tables are sorted by name, "
      "not by score: pick the axis you care about. **×ceil** columns are "
      "distances divided by the real-speech split-half ceiling (1.0 = indistinguishable "
      "from real speech, lower is better). The figure shows accuracy (x) against rhythm "
      "pause factor (left) and pitch-range distance (right); bottom-left is best.\n")
    if cap_index:
        m = cap_index["method"]
        (hi_s, hi_v), (lo_s, lo_v) = m["anchors"].items()
        name = lambda s: s if s == hi_s else marked_label(s)
        W("### Capability index (both sets)\n")
        W("One score per system, fit like Epoch's Capabilities Index "
          "([ECI](https://epoch.ai/data/eci-documentation/methodology)): item response theory, "
          "`performance = sigmoid(slope_item · (capability_system − difficulty_item))`, over "
          f"{m['n_items']} items — one metric on one set in one language — so a system scored on "
          "only some languages is still placed on the common scale, which no macro can do. "
          "Each metric enters as a performance in [0, 1]: "
          + "; ".join(f"{v['performance']}" + (f" ({'Clean' if v['set'] == CLEAN else 'ITW'})"
                                              if v.get("set") else "")
                      for v in m["metrics"].values())
          + f". Intelligibility, voice, quality and prosody weigh equally. Scale anchors: "
          f"{hi_s} = {hi_v:g}, {label(lo_s)} = {lo_v:g}. Interval = {m['ci_level']:.0%} "
          f"bootstrap over each system's own items ({m['bootstrap']} refits); anchors have none "
          "by definition. Few languages = wide interval.\n")
        W("Over time, each system at its public release date (sources in "
          "`rd_common.RELEASES`; variants of one checkpoint share its date). The line is the "
          "frontier: the best system scored on every language released by that date.\n")
        W("![Capability index over time](assets/plots/capability_index_time.png)\n")

        def index_table(p):
            pm = p["method"]
            W("| # | System | Index | 90% CI | Languages (Clean · ITW) |")
            W("|--:|---|--:|---|---|")
            for rank, s in enumerate(p["systems"], 1):
                band = "anchor" if s["anchor"] else f"{s['ci'][0]:.1f} – {s['ci'][1]:.1f}"
                W(f"| {rank} | {name(s['name'])} | **{s['index']:.1f}** | {band} | "
                  f"{s['langs'][CLEAN]}/{pm['n_langs'][CLEAN]} · {s['langs'][ITW]}/{pm['n_langs'][ITW]} |")
            W("")

        index_table(cap_index)
        pros = cap_index.get("prosody")
        if pros:
            pm = pros["method"]
            aspects = "; ".join(
                f"**{dim}** — " + ", ".join(
                    v["performance"] + (f" ({'Clean' if v['set'] == CLEAN else 'ITW'})" if v.get("set") else "")
                    for v in pm["metrics"].values() if v["dimension"] == dim)
                for dim in pm["dimensions"])
            W("### Prosody index (both sets)\n")
            W(f"The same fit over {pm['n_items']} prosody items only (ECI's domain-specific "
              "index), with every prosody metric rather than the capability index's four: "
              f"{aspects}. The {len(pm['dimensions'])} aspects weigh equally, so a quantity "
              "measured several ways does not outvote one measured once. MPM span is scored "
              "on silence-trimmed audio (`docs/controls_2026-08.md` C6). Same anchors and "
              "interval as the capability index.\n")
            index_table(pros)
    for task, title, extra in (
            (CLEAN, "Clean (FLEURS-R)", ""),
            (ITW, "ITW (VoxBlink2, in the wild)",
             " Speaker similarity is scored against a held-out recording of the same "
             "speaker (**SPK held-out**); **SPK prompt** is similarity to the prompt, and "
             "the **copy gap** is their difference — a large gap means the system "
             "reproduces the prompt's channel rather than the speaker. **Dur**, **Pause "
             "Δ** and **Pitch-range |log2|** are per-utterance paired medians against the "
             "real recording (lower is better).")):
        langs = langs_for(task)
        cols = HEADLINE_COLS[task]
        W(f"### {title}\n")
        if extra:
            W(extra.strip() + "\n")
        W(f"![WER vs prosody, {title}](assets/plots/tradeoff_{task}.png)\n")
        models = models_for(recs, task)
        full = [m for m in models if n_covered(idx, m, task, "wer", langs) == len(langs)]
        partial = [m for m in models if m not in full]

        gt = GT_NAME[task]
        if any(r["model"] == gt for r in recs):
            W("**Real-speech reference**\n")
            W("\n".join(_headline_header(cols)))
            W(_headline_row(idx, gt, task, langs, cols))
            W("")
        for cls, heading in CLASS_ORDER:
            members = sorted((m for m in full if cls_of(m) == cls), key=lambda m: label(m).lower())
            if not members:
                continue
            W(f"**{heading}**\n")
            W("\n".join(_headline_header(cols)))
            for m in members:
                W(_headline_row(idx, m, task, langs, cols))
            W("")
        if partial:
            W("**Partial coverage — not comparable**\n")
            W("Missing WER for at least one basket language; no macro value is shown.\n")
            W("| System | Langs with WER | Missing |")
            W("|---|--:|---|")
            for m in sorted(partial, key=lambda m: label(m).lower()):
                have = {l for l in langs if rec_value(idx.get((m, task, l)), "wer") is not None}
                miss = [l for l in langs if l not in have]
                W(f"| {marked_label(m)} | {len(have)}/{len(langs)} | "
                  f"{', '.join(f'`{l}`' for l in miss) if len(miss) <= 12 else str(len(miss)) + ' languages'} |")
            W("")
        W(COND_LEGEND + "\n")
    W("MPM span is reported for the Clean set only: its anchors collapse on "
      "in-the-wild audio (`docs/mpm_itw_limits.md`).\n")

    # --- 1. Coverage matrix ----------------------------------------------------
    W("## 1. Coverage matrix\n")
    W("| Model | clean | voxblink |")
    W("|---|:-:|:-:|")
    all_models = sorted({r["model"] for r in recs}, key=_sort_key)
    for m in all_models:
        def has(t, mm=m):
            # Flat baselines (gt_*/whisper_*) encode their task in the dir name and
            # carry task="" rows, so match them by name instead of by task field.
            if mm in (f"gt_{t}", f"whisper_{t}"):
                return True
            return any(r["model"] == mm and r["task"] == t for r in recs)
        zs = "✓" if has(CLEAN) else "—"
        vox = "✓" if has(ITW) else "—"
        W(f"| `{label(m)}` | {zs} | {vox} |")
    W("")

    # --- 1b. Conditioning ------------------------------------------------------
    W("## 1b. Conditioning\n")
    W("The systems below are NOT all run under the same conditioning — what each one")
    W("is given alongside the target text is a harness policy that differs per")
    W("system (see `scripts/results_doc/rd_common.py:CONDITIONS` for the code")
    W("provenance). WER is strongly sensitive to it: `omnivoice` scores 34.2 avg WER")
    W("without the reference transcript vs 3.2 with it, on the same weights. Rows in")
    W("the ranking tables carry the class markers († ◇ ‡) so classes are never")
    W("compared undisclosed.\n")
    W("| Model | ref transcript? | cloning class | sampling/config |")
    W("|---|:-:|---|---|")
    cond_classes = {
        "with-transcript": "yes",
        "no-transcript": "no",
        "x-vector": "no",
        "preset": "n/a",
    }
    for m in all_models:
        cond = condition(m)
        if cond is None:
            continue  # gt_*/whisper_* baselines: real speech, nothing is conditioned
        cls, prov = cond
        cls_disp = {"no-transcript": "no-transcript clone †",
                    "x-vector": "x-vector-only clone ◇",
                    "preset": "preset voice ‡",
                    "with-transcript": "with-transcript clone"}[cls]
        W(f"| `{marked_label(m)}` | {cond_classes[cls]} | {cls_disp} | {prov} |")
    W("")
    W("Seeding: `zonos2_*` and `zonos3-mini_*` runs are fixed-seed (0); `elevenlabs_*` requests send")
    W("seed 0 (provider best-effort); vLLM-Omni-served systems and the other hosted APIs pass no")
    W("seed (unseeded server-default sampling); native recipes note their sampling defaults above.\n")

    # --- 2. Clean WER ------------------------------------------------------------
    zs_models = models_for(recs, CLEAN)
    # GT anchor: the gt_clean baseline rows carry task="" (flat dir), so
    # models_for() misses them — append explicitly; _sort_key buckets gt_* last.
    if GT_NAME[CLEAN] not in zs_models and \
            any(r["model"] == GT_NAME[CLEAN] and not r["task"] for r in recs):
        zs_models = sorted(zs_models + [GT_NAME[CLEAN]], key=_sort_key)

    def zs_task(m):
        # The ground-truth reference rows are stored with an empty task string.
        return "" if m == GT_NAME[CLEAN] else CLEAN

    n_clean = len(CLEAN_LANGS)
    W(f"## 2. Zero-shot WER_CLIP % ({n_clean} languages)\n")
    per_lang_table(zs_models, zs_task, CLEAN_LANGS, fmt_wer)
    avg_row(zs_models, zs_task, CLEAN, CLEAN_LANGS)
    W("")
    W(f"The avg row is shown only for systems that cover all {n_clean} languages;")
    W(f"others read `partial (n/{n_clean})` and are not comparable to full-coverage averages.")
    W("The `gt (fleurs-r)` column is real speech scored by the same ASR — the")
    W("achievable WER floor. zh/ja/ko/th are char-tokenized, so their \"WER\" is a")
    W("CER and the `avg` column mixes WER and CER units.\n")
    W(COND_LEGEND + "\n")

    W("### Zero-shot WER_L50 % (mean of utts with WER < 50%, ignores catastrophic tail)\n")
    W("Caveat: when *every* utterance of a leaf is ≥50% WER, `utils/average_wer.py`")
    W("falls back to reporting L50 over the full set — so a cell can exceed 50 for a")
    W("model that collapsed on that language. The catastrophic tail itself is the")
    W("**Tail >50%** column of §0.\n")
    per_lang_table(zs_models, zs_task, CLEAN_LANGS, fmt_l50)
    W("")

    # --- 3. Clean speaker similarity --------------------------------------------
    W("## 3. Zero-shot speaker similarity (×100, higher better)\n")
    per_lang_table(zs_models, zs_task, CLEAN_LANGS, lambda r: fmt(rec_value(r, "spk"), 1))
    W("")
    preset_labels = sorted(f"`{label(m)}`" for m in zs_models
                           if (condition(m) or ("", ""))[0] == "preset")
    if preset_labels:
        W(f"> Preset-voice engines ({', '.join(preset_labels)}) are not cloning the")
        W("> prompt speaker, so their speaker-similarity cells are not a meaningful signal.")
    W(SPK_EMBEDDER_NOTE + "\n")
    W(COND_LEGEND + "\n")

    # --- 4. Clean MSR-UTMOS ------------------------------------------------------
    W("## 4. Zero-shot MSR-UTMOS (higher better)\n")
    W("MOS predictor (`msr_utmos_mean.txt`). Empty cells = MSR-UTMOS not computed for that leaf.\n")
    per_lang_table(zs_models, zs_task, CLEAN_LANGS, lambda r: fmt(rec_value(r, "msr_utmos"), 2))
    W("")
    W("The `gt (fleurs-r)` column is real speech: the MOS predictor's anchor, not a")
    W("ceiling models must beat — read model cells against it.\n")
    W(COND_LEGEND + "\n")

    # --- Prosody --------------------------------------------------------------------
    W("### Prosody\n")
    W("Prosody is read on four axes; all distances are to real speech, normalised by")
    W("the real-speech split-half ceiling (×ceil: 1.0 = as close as two halves of real")
    W("speech are to each other).\n")
    W("- **Rhythm** — distance between the generated and real rhythm distributions in")
    W("  mHuBERT space, reported as two factors: *pace* (speaking-rate distribution) and")
    W("  *pause* (burstiness: pause and stall structure). Method and ceiling:")
    W("  [`docs/rhythm.md`](docs/rhythm.md).")
    W("- **Pitch range** — register-normalized: each utterance's F0 contour is expressed")
    W("  in semitones around its own median and summarised by its p90−p10 range, so the")
    W("  speaker's absolute register drops out and only expressiveness remains. Method:")
    W("  [`docs/pitch_range.md`](docs/pitch_range.md).")
    W("- **Characters/s** — 2-Wasserstein distance to the reference speaking-rate")
    W("  distribution (explorer speaking-rate view), on the utterances with WER ≤ 20%")
    W("  (docs/controls_2026-08.md C7).")
    W("- **MPM span: Clean set only — anchors collapse on in-the-wild audio "
      "(docs/mpm_itw_limits.md); values >1 are worse than contour-scrambled speech.** "
      "0 = real-speech split-half floor, 1 = contour-scrambled floor. Leading/trailing "
      "silence is trimmed before scoring (docs/controls_2026-08.md C6), so spans are not "
      "comparable with releases before 2026-10-31.\n")
    W("On the ITW set the real target recording exists for every item, so prosody is also")
    W("measured *paired* per utterance (duration ratio, pause-count difference, pitch-range")
    W("ratio against the target): the **Dur**, **Pause Δ** and **Pitch-range |log2|**")
    W("columns of §0.\n")

    # --- 5. ITW -----------------------------------------------------------------------
    vox_models = models_for(recs, ITW)
    if GT_NAME[ITW] not in vox_models and any(r["model"] == GT_NAME[ITW] for r in recs):
        vox_models = [GT_NAME[ITW]] + vox_models

    def vox_task(m):
        # The ground-truth reference rows are stored with an empty task string.
        return "" if m == GT_NAME[ITW] else ITW

    def vox_per_lang(title, cell):
        W(f"### Per-language {title}\n")
        W("| lang | " + " | ".join(marked_label(m) for m in vox_models) + " |")
        W("|---|" + "|".join(["--:"] * len(vox_models)) + "|")
        for lang in ITW_LANGS:
            W("| " + " | ".join([lang] + [cell(idx.get((m, vox_task(m), lang)))
                                          for m in vox_models]) + " |")
        W("")

    W(f"## 5. VoxBlink2 multilingual zero-shot ({len(ITW_LANGS)} languages)\n")
    W("Aggregates are in §0. The `gt (voxblink)` row is real in-the-wild speech: its WER")
    W("is the ASR floor, its speaker similarity is the prompt scored against the real")
    W("target recording (the same-speaker ceiling that **SPK held-out** is read against),")
    W("and its MSR-UTMOS shows what the MOS predictor gives genuine in-the-wild audio.")
    W("zh/ja/ko/th are char-tokenized, so their \"WER\" is a CER.\n")
    vox_per_lang("WER_CLIP", fmt_wer)
    vox_per_lang("WER_L50", fmt_l50)
    vox_per_lang("speaker similarity, held-out recording (×100)",
                 lambda r: fmt(rec_value(r, "spk_xutt"), 1))
    vox_per_lang("speaker similarity, prompt (×100)", lambda r: fmt(rec_value(r, "spk"), 1))
    W(SPK_EMBEDDER_NOTE + "\n")
    vox_per_lang("MSR-UTMOS (blank = not computed)", lambda r: fmt(rec_value(r, "msr_utmos"), 2))
    W(COND_LEGEND + "\n")

    # --- 6. Methodology ----------------------------------------------------------------
    qwen_l = ", ".join(sorted(QWEN_LANGS))
    W("## 6. Methodology notes\n")
    W("- **WER_CLIP** caps per-utterance WER at 100% before averaging — a robust mean")
    W("  insensitive to runaway hallucinations. `WER` (raw) and `WER_L50` (mean of")
    W("  utts <50% WER) live in the `.wer` summary line alongside it.")
    W("- **Tail >50%** is the share of utterances with WER above 50% (`WER_TAIL50:`")
    W("  footer): the catastrophic-failure rate that the mean hides.")
    W("- **Confidence intervals** on WER are sentence-clustered bootstraps: repeated")
    W("  FLEURS sentences (the same text read by several speakers) resample together.")
    W("- **Unscoreable generations count as maximum error** (WER_CLIP 100), rather than")
    W("  dropping out of the mean — otherwise a leaf mean is a survivor mean and the")
    W("  most degenerate output is the output that disappears from its own score. Each")
    W("  leaf's `.wer` footer reports `N_TRANSCRIBED:`/`N_IMPUTED:` so the imputed share")
    W("  is always visible; see `docs/unscoreable_2026-09.md`.")
    W("- **WER prep** is applied identically to REF and HYP: apostrophes deleted,")
    W("  lowercase, Unicode punctuation strip, and digits expanded to words with")
    W("  `num2words` (languages without a `num2words` converter keep their digits —")
    W("  `utils/text_norm.py:NUM2WORDS_LANGS`); zh/ja/ko/th scored as CER")
    W("  (`utils/run_wer.py`).")
    W(f"- **ASR** is routed per language and the routing is enforced: {qwen_l} use")
    W(f"  `{asr_q[0]}` (revision `{asr_q[1]}`); every other language uses")
    W(f"  `{asr_w[0]}` (revision `{asr_w[1]}`). Both are served via vLLM")
    W("  (`scripts/serve_asr.sh`). Revisions, language baskets and the version stamp")
    W("  live in `utils/benchmark.py`; every score file carries")
    W(f"  `BENCHMARK_VERSION: {BENCHMARK_VERSION}` and its scorer pin, and `build_results.py`")
    W("  refuses any input whose stamp does not match.")
    W(f"- **Speaker similarity** = cosine × 100, ReDimNet (`{SPK_HUB_REPO}` @")
    W(f"  `{SPK_HUB_REF[:7]}`, `{'/'.join(SPK_VARIANT)}`). On the Clean set the reference is the")
    W("  prompt. On ITW the primary score (**SPK held-out**) is against a held-out")
    W("  recording of the same speaker (`target_wav/`); the prompt similarity is kept as")
    W("  **SPK prompt**, and **copy gap** = prompt − held-out.")
    W("- **MSR-UTMOS** = sarulab-speech MSR-UTMOS (SFI HuBERT) MOS predictor, pinned")
    W("  checkpoint (`MSR_UTMOS_CKPT`).")
    W("- **Rhythm** = distance of the generated rhythm distribution (mHuBERT-147 features)")
    W("  to real speech, as pace and pause factors in units of the real-speech split-half")
    W("  ceiling (`scripts/rhythm_dist.py`, `docs/rhythm.md`); lower is better.")
    W("- **Pitch range** = 2-Wasserstein distance between the per-utterance p90−p10")
    W("  semitone-range distributions of generated and real speech, in units of the")
    W("  split-half ceiling, on a fixed 500-utterance sample per language")
    W("  (`scripts/pitch_range_dist.py`, `docs/pitch_range.md`).")
    W("- **Paired ITW prosody** = per-utterance median |log2| duration ratio (generations")
    W("  with WER ≤ 20% only), mean absolute pause-count difference (pauses = gaps ≥ 0.2 s")
    W("  between Silero VAD speech segments) and median |log2|")
    W("  pitch-range ratio of each generation against the real target recording")
    W("  (`scripts/paired_itw.py`); ITW only.")
    W("- **Characters/s distance** = 2-Wasserstein distance between generated and")
    W("  reference speaking-rate distributions on the utterances the ASR heard with")
    W("  WER ≤ 20% (`scripts/speaking_rate_dist.py`); lower is better.")
    W("- **MPM span** = distance from the real-speech split-half floor (0), normalized")
    W("  by the contour-scrambled floor (1) (`utils/score_mpm_shuffle_floor.py`), on audio")
    W("  with leading/trailing silence trimmed (`docs/controls_2026-08.md` C6). Clean")
    W("  set only — anchors collapse on in-the-wild audio (`docs/mpm_itw_limits.md`);")
    W("  values >1 are worse than contour-scrambled speech.")
    W("- **Reliability gallery** — per-language edit-distance survival/tail plates for")
    W("  the GT baselines: [`assets/reliability_gallery.html`](assets/reliability_gallery.html)")
    W("  (`scripts/run_reliability.sh`).")
    W("- Empty cells (`—`) = test set not run for that model, or score file missing.")
    W("- Refresh by re-running `scripts/results_doc/run.sh` (or just `gen_md.py` if")
    W("  only the markdown layout changed).")
    W("")
    return "\n".join(buf)


def main():
    try:
        recs = load_snapshot()
    except OSError:
        sys.exit("gen_md: results_snapshot.json missing -- run build_results.py first")
    cap_index = None
    if os.path.isfile(CAP_INDEX):
        with open(CAP_INDEX, encoding="utf-8") as f:
            cap_index = json.load(f)
    text = render(recs, dedup_counts(), cap_index)
    with open(OUT, "w") as f:
        f.write(text)
    print(f"Wrote {OUT} ({len(text)} chars, {text.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
