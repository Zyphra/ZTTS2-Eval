#!/usr/bin/env python3
"""Generate RESULTS_CURRENT.md from the JSON snapshot built by build_results.py.

Model-agnostic: discovers whichever models are present in the snapshot (no
hardcoded checkpoint list) and emits per-language metric tables for the two
ZTTS2-Eval sets — zero_shot (FLEURS-R) and zero_shot_voxblink (VoxBlink2) — plus
the retained distributional prosody readouts and DS-WED prosody diversity.

Run `scripts/results_doc/build_results.py` first to (re)build the snapshot from a
populated `results/` tree, or use `scripts/results_doc/run.sh` for the full chain.
"""
import os, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")

ROOT = os.path.join(REPO_ROOT, "results")
OUT = os.path.join(REPO_ROOT, "RESULTS_CURRENT.md")
recs = json.load(open(os.path.join(CACHE, "results_snapshot.json")))

# Index: (model, task, set) -> rec
idx = {}
for r in recs:
    idx[(r["model"], r["task"], r["set"])] = r

# Shared display-label map + conditioning classes — see scripts/results_doc/rd_common.py.
from rd_common import (  # noqa: E402
    label, marked_label, condition, COND_LEGEND, ordered_langs,
    VOX_LANGS_EXCLUDED)


def _sort_key(m):
    # gt_* last (reference rows), omni_* first, everything else in the middle.
    bucket = 0 if m.startswith("omni_") else (2 if m.startswith("gt_") else 1)
    return (bucket, label(m))


def models_for(task):
    """All models that have at least one row for `task`, in a stable order."""
    return sorted({r["model"] for r in recs if r["task"] == task}, key=_sort_key)


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


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


ZS_LANGS = ["en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru"]
# The VoxBlink language basket is discovered from the snapshot (model rows plus
# the flat gt baseline rows) rather than hard-coded, so a newly scored language
# enters the published tables on the next rebuild. `tl` is excluded from
# aggregates (see the excluded-langs callout emitted at the end of section 5).
VOX_LANGS_FULL = ordered_langs(
    {r["set"] for r in recs if r["task"] == "zero_shot_voxblink"}
    | {r["set"] for r in recs
       if r["model"] == "gt_zero_shot_voxblink" and not r["task"]})
VOX_LANGS = [l for l in VOX_LANGS_FULL if l not in VOX_LANGS_EXCLUDED]

buf = []
W = buf.append


def emit_table_header(first_col, cols):
    W("| " + first_col + " | " + " | ".join(marked_label(m) for m in cols) + " |")
    W("|---|" + "|".join(["--:"] * len(cols)) + "|")


W("# ZTTS2-Eval — Current Results Snapshot\n")
W("Regenerated from the `results/` directory tree by")
W("`scripts/results_doc/run.sh`. Numbers are read directly from")
W("`wav_res_ref_text.wer`, `spk_simi_scores.txt`, and `msr_utmos_mean.txt` in")
W("every leaf directory. **WER_CLIP** (per-utt WER capped")
W("at 100% before averaging) is the robust mean; raw WER and WER_L50 are also")
W("tracked. Speaker similarity is cosine × 100 (ReDimNet); MSR-UTMOS is the MOS")
W("predictor.\n")
W("> **Comparability note (2026-08 audit + methodology review):** numbers in this")
W("> snapshot are not directly comparable to earlier snapshots: zh WER shifts on")
W("> rescore (the traditional→simplified fold is now applied to the reference text")
W("> as well as the hypothesis), the VoxBlink aggregate basket grew from 17 to 21")
W("> languages (nl, uk, vi added) and then shrank to 20 on 2026-08-29 when `yo`")
W("> was removed outright — its transcripts are Nigerian English mislabeled by")
W("> Whisper LID, not Yoruba (see RESULTS_VOXBLINK.md) — and the coreset win%")
W("> direction was corrected in RESULTS_CORESET.md.\n")

# --- 1. Coverage matrix -----------------------------------------------------
W("## 1. Coverage matrix\n")
W("| Model | zero_shot | voxblink |")
W("|---|:-:|:-:|")
all_models = sorted({r["model"] for r in recs}, key=_sort_key)
for m in all_models:
    def has(t, mm=m):
        # Flat baselines (gt_*/whisper_*) encode their task in the dir name and
        # carry task="" rows, so match them by name instead of by task field.
        if mm in (f"gt_{t}", f"whisper_{t}"):
            return True
        return any(r["model"] == mm and r["task"] == t for r in recs)
    zs = "✓" if has("zero_shot") else "—"
    vox = "✓" if has("zero_shot_voxblink") else "—"
    W(f"| `{label(m)}` | {zs} | {vox} |")
W("")

# --- 1b. Conditioning --------------------------------------------------------
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
_cond_classes = {
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
    W(f"| `{marked_label(m)}` | {_cond_classes[cls]} | {cls_disp} | {prov} |")
W("")
W("Seeding: `zonos2_*` runs are fixed-seed (0); vLLM-Omni-served systems pass no")
W("seed or temperature (unseeded server-default sampling); native recipes note")
W("their sampling defaults above.\n")

# --- 2. Zero-shot WER -------------------------------------------------------
zs_models = models_for("zero_shot")
# GT anchor: the gt_zero_shot baseline rows carry task="" (flat dir), so
# models_for() misses them — append explicitly; _sort_key buckets gt_* last.
if "gt_zero_shot" not in zs_models and \
        any(r["model"] == "gt_zero_shot" and not r["task"] for r in recs):
    zs_models = sorted(zs_models + ["gt_zero_shot"], key=_sort_key)


def _zs_task(m):
    # The ground-truth reference rows are stored with an empty task string.
    return "" if m == "gt_zero_shot" else "zero_shot"
W("## 2. Zero-shot WER_CLIP %\n")
emit_table_header("Lang", zs_models)
for lang in ZS_LANGS:
    row = [lang] + [fmt_wer(idx.get((m, _zs_task(m), lang))) for m in zs_models]
    W("| " + " | ".join(row) + " |")
avg_row = ["**avg**"]
for m in zs_models:
    vals = [idx[(m, _zs_task(m), l)]["wer"][2] for l in ZS_LANGS
            if (m, _zs_task(m), l) in idx and idx[(m, _zs_task(m), l)]["wer"]]
    a = mean(vals)
    if a is None:
        avg_row.append("—")
    elif len(vals) < len(ZS_LANGS):
        # Partial coverage: an average over fewer languages is a different
        # quantity — annotate it so it is never read against the full-basket ones.
        avg_row.append(f"**{a:.2f}** ({len(vals)}/{len(ZS_LANGS)})")
    else:
        avg_row.append(f"**{a:.2f}**")
W("| " + " | ".join(avg_row) + " |")
W("")
W("Averages are over whichever of the 9 languages each model covers; partial")
W("baskets are marked `(n/9)` and are not comparable to full-coverage averages.")
W("The `gt (fleurs-r)` column is real speech scored by the same ASR — the")
W("achievable WER floor. zh/ja/ko are char-tokenized, so their \"WER\" is a")
W("CER and the `avg` column mixes WER and CER units.\n")
W(COND_LEGEND + "\n")

W("### Zero-shot WER_L50 % (mean of utts with WER < 50%, ignores catastrophic tail)\n")
W("Caveat: when *every* utterance of a leaf is ≥50% WER, `utils/average_wer.py`")
W("falls back to reporting L50 over the full set — so a cell can exceed 50 for a")
W("model that collapsed on that language.\n")
emit_table_header("Lang", zs_models)
for lang in ZS_LANGS:
    row = [lang] + [fmt_l50(idx.get((m, _zs_task(m), lang))) for m in zs_models]
    W("| " + " | ".join(row) + " |")
W("")

SPK_EMBEDDER_NOTE = (
    "> Speaker-similarity gaps below ~1 point are embedder-dependent: a held-out "
    "embedder control (vox2-only ReDimNet, no VoxBlink2 in training) preserved the "
    "system ordering overall (Kendall tau +0.91...+0.98) but reversed one "
    "published pair on `de`. Read small spk differences as ties -- see "
    "docs/controls_2026-08.md."
)

# --- 3. Zero-shot speaker similarity ---------------------------------------
W("## 3. Zero-shot speaker similarity (×100, higher better)\n")
emit_table_header("Lang", zs_models)
for lang in ZS_LANGS:
    row = [lang] + [fmt((idx.get((m, _zs_task(m), lang)) or {}).get("spk"), 1) for m in zs_models]
    W("| " + " | ".join(row) + " |")
W("")
_preset_labels = sorted(f"`{label(m)}`" for m in zs_models
                        if (condition(m) or ("", ""))[0] == "preset")
if _preset_labels:
    W(f"> Preset-voice engines ({', '.join(_preset_labels)}) are not cloning the")
    W("> prompt speaker, so their speaker-similarity cells are not a meaningful signal.")
W(SPK_EMBEDDER_NOTE)
W("> The `gt (fleurs-r)` column is a second real clip of the same speaker — the")
W("> same-speaker ceiling this metric can reach.\n")
W(COND_LEGEND + "\n")

# --- 4. Zero-shot MSR-UTMOS -------------------------------------------------
W("## 4. Zero-shot MSR-UTMOS (higher better)\n")
W("MOS predictor (`msr_utmos_mean.txt`). Empty cells = MSR-UTMOS not computed for that leaf.\n")
emit_table_header("Lang", zs_models)
for lang in ZS_LANGS:
    row = [lang] + [fmt((idx.get((m, _zs_task(m), lang)) or {}).get("msr_utmos"), 2) for m in zs_models]
    W("| " + " | ".join(row) + " |")
W("")
W("The `gt (fleurs-r)` column is real speech: the MOS predictor's anchor, not a")
W("ceiling models must beat — read model cells against it.\n")
W(COND_LEGEND + "\n")

# --- Distributional prosody --------------------------------------------------
W("### Distributional prosody\n")
W("The primary distribution-matching readouts are **characters/s distance**")
W("(2-Wasserstein distance to the reference speaking-rate distribution; lower is")
W("better) and **MPM span** (0 = real-speech split-half floor, 1 =")
W("contour-scrambled floor; lower is better). Both are available in the explorer's")
W("speaking-rate and prosody-overlap views.\n")
W("TTSDS2 **Pitch** is retained as an interpretable auxiliary readout: it compares")
W("the per-utterance mean-F0 distribution against real speech (0..100; higher is")
W("closer). Mean F0 is speaker-dependent, so read Pitch alongside speaker")
W("similarity rather than as an independent quality axis.\n")


def _pitch_models(task):
    pairs = [(m, task) for m in models_for(task)
             if (idx.get((m, task, "en")) or {}).get("ttsds_pitch") is not None]
    baseline = f"gt_{task}"
    if (idx.get((baseline, "", "en")) or {}).get("ttsds_pitch") is not None:
        pairs.insert(0, (baseline, ""))
    return pairs


def _pitch_table(task, heading):
    pitch_models = _pitch_models(task)
    if not pitch_models:
        return
    W(f"**{heading}**\n")
    W("| metric | " + " | ".join(label(model) for model, _ in pitch_models) + " |")
    W("|---|" + "|".join(["--:"] * len(pitch_models)) + "|")
    values = [fmt(idx[(model, disk_task, "en")].get("ttsds_pitch"), 2)
              for model, disk_task in pitch_models]
    W("| Pitch | " + " | ".join(values) + " |")
    W("")


_pitch_table("zero_shot", "TTSDS2 Pitch — Zero-shot/en")
_pitch_table("zero_shot_voxblink", "TTSDS2 Pitch — VoxBlink/en")
W("Inspect the underlying mean-F0 distributions with")
W("`scripts/plot_ttsds_pitch_ridgeline.py`; methodology is documented in")
W("[`docs/ttsds_pitch.md`](docs/ttsds_pitch.md).\n")

# --- DS-WED -----------------------------------------------------------------
def _dswed_rows():
    rows = []
    if not os.path.isdir(ROOT):
        return rows
    for m in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, m, "dswed", "zero_shot", "en", "dswed_mean.txt")
        if not os.path.isfile(p):
            continue
        d = {}
        for line in open(p, errors="replace"):
            if ":" in line:
                k, _, v = line.partition(":")
                d[k.strip()] = v.strip()
        try:
            rows.append({
                "model": m,
                "mean": float(d.get("DS-WED mean", "nan")),
                "std": float(d.get("DS-WED std", "nan")),
                "median": float(d["DS-WED median"]) if "DS-WED median" in d else None,
                "p90": float(d["DS-WED p90"]) if "DS-WED p90" in d else None,
                "n": d.get("n_uids", "—"),
                "k": d.get("k_seeds", "—"),
            })
        except ValueError:
            continue
    rows.sort(key=lambda r: r["mean"], reverse=True)  # most diverse first
    return rows


_dsrows = _dswed_rows()
if _dsrows:
    # Heading + prose gated together with the table: no dswed leaf dirs -> no
    # orphan heading claiming results that don't exist.
    W("### Prosody diversity — DS-WED (multi-seed, zero_shot/en)\n")
    W("Pairwise mean DS-WED (arXiv:2509.19928) across K seeds per (text, prompt):")
    W("weighted edit distance over HuBERT k-means tokens of same-text generations.")
    W("**Higher = more prosodic variation** seed-to-seed. Read from")
    W("`results/<model>/dswed/zero_shot/en/dswed_mean.txt` (`utils/score_dswed.py`).\n")
    W("| Model | DS-WED mean | median | p90 | std | n | K |")
    W("|---|--:|--:|--:|--:|--:|--:|")
    for r in _dsrows:
        W(f"| `{label(r['model'])}` | {r['mean']:.1f} | "
          f"{fmt(r['median'], 1)} | {fmt(r['p90'], 1)} | {r['std']:.1f} | {r['n']} | {r['k']} |")
    W("")
    W("Median/p90 are robust to the long right tail (DS-WED is unbounded).\n")
    W("![DS-WED per-utterance distributions](assets/plots/dswed_distributions.png)\n")

# --- 5. Voxblink ------------------------------------------------------------
vox_models = models_for("zero_shot_voxblink")
if "gt_zero_shot_voxblink" not in vox_models and \
        any(r["model"] == "gt_zero_shot_voxblink" for r in recs):
    vox_models = ["gt_zero_shot_voxblink"] + vox_models


def _vox_task(m):
    # The ground-truth reference rows are stored with an empty task string.
    return "" if m == "gt_zero_shot_voxblink" else "zero_shot_voxblink"


W(f"## 5. VoxBlink2 multilingual zero-shot ({len(VOX_LANGS)} langs)\n")
W("Mean over the aggregate languages (lower WER better). Means are over")
W("whichever of the aggregate languages each model covers — the `langs` column")
W("says how many; partial baskets are not comparable to full-coverage means.\n")


def _fmt_gt_tl_wer():
    r = idx.get(("gt_zero_shot_voxblink", "", "tl"))
    if r and r.get("wer"):
        return f"{r['wer'][2]:.1f}%"
    return "n/a"


W(f"> **Pruned lang**: `{'`, `'.join(VOX_LANGS_EXCLUDED)}` excluded from aggregates —")
W("> when the exclusion was made, GT-on-GT WER exceeded 20% under the pre-routing")
W("> ASR setup, i.e. the ASR pipeline could not reliably transcribe the reference")
W(f"> audio. The routed pipeline now scores gt/tl at {_fmt_gt_tl_wer()} WER_CLIP;")
W("> the exclusion is retained pending revalidation. Raw numbers are in the")
W("> callout at the end of this section.\n")
W("| Model | mean WER_CLIP | mean spk | mean MSR-UTMOS | langs |")
W("|---|--:|--:|--:|--:|")
for m in vox_models:
    task = _vox_task(m)
    wer_v, spk_v, msr_v = [], [], []
    for lang in VOX_LANGS:
        r = idx.get((m, task, lang))
        if not r:
            continue
        if r["wer"]: wer_v.append(r["wer"][2])
        if r.get("spk") is not None: spk_v.append(r["spk"])
        if r.get("msr_utmos") is not None: msr_v.append(r["msr_utmos"])
    n_langs = max(len(wer_v), len(spk_v), len(msr_v))
    W(f"| `{marked_label(m)}` | {fmt(mean(wer_v), 2)} | "
      f"{fmt(mean(spk_v), 1)} | {fmt(mean(msr_v), 2)} | "
      f"{n_langs}/{len(VOX_LANGS)} |")
W("")
W("The `gt (voxblink)` row is real in-the-wild speech: its WER is the ASR floor,")
W("its spk is the same-speaker ceiling, and its MSR-UTMOS shows what the MOS")
W("predictor gives genuine in-the-wild audio. zh/ja/ko/th are char-tokenized, so")
W("their \"WER\" is a CER and the mean column mixes WER and CER units.\n")
W(COND_LEGEND + "\n")


def _vox_per_lang(title, field, key=None, decimals=2):
    W(f"### Per-language {title}\n")
    W("| lang | " + " | ".join(marked_label(m) for m in vox_models) + " |")
    W("|---|" + "|".join(["--:"] * len(vox_models)) + "|")
    for lang in VOX_LANGS:
        row = [lang]
        for m in vox_models:
            r = idx.get((m, _vox_task(m), lang))
            v = None
            if r:
                if field == "wer":
                    if r.get("wer"): v = r["wer"][key]
                else:
                    v = r.get(field)
            row.append(fmt(v, decimals))
        W("| " + " | ".join(row) + " |")
    W("")


_vox_per_lang("WER_CLIP", "wer", key=2)
_vox_per_lang("WER_L50", "wer", key=1)
_vox_per_lang("speaker similarity (×100)", "spk", decimals=1)
W(SPK_EMBEDDER_NOTE + "\n")
_vox_per_lang("MSR-UTMOS (blank = not computed)", "msr_utmos", decimals=2)

W("### Per-bin / per-length analysis\n")
W("Per-MSR-UTMOS-bin WER_CLIP heatmaps (rows = languages, columns = MSR-UTMOS")
W("quality bins, worst→best source audio):\n")
W("![Voxblink per-bin WER_CLIP heatmaps](assets/plots/voxblink_perbin_heatmaps.png)\n")
W("Same WER_CLIP bucketed by REF text-length quantiles per language:\n")
W("![Voxblink per-text-length WER_CLIP heatmaps](assets/plots/voxblink_perlen_heatmaps.png)\n")
W("Per-length on controlled content (`zero_shot`) vs in-the-wild (`voxblink`):\n")
W("![Per-length: voxblink vs zero_shot](assets/plots/perlen_zeroshot_vs_voxblink.png)\n")

# --- Excluded-langs callout ----
W("### Excluded-langs callout: `" + "`, `".join(VOX_LANGS_EXCLUDED) + "` (informational, not in aggregates)\n")
W("Retained so the pruning decision is auditable. The exclusion reflects the ASR")
W("pipeline's historical unreliability on the reference audio (not the TTS")
W("models); compare the gt row here against the model rows before reading them.\n")
W("| Model | " + " | ".join(VOX_LANGS_EXCLUDED) + " |")
W("|---|" + "|".join(["--:"] * len(VOX_LANGS_EXCLUDED)) + "|")
for m in vox_models:
    row = [f"`{marked_label(m)}`"]
    for lang in VOX_LANGS_EXCLUDED:
        r = idx.get((m, _vox_task(m), lang))
        v = None
        if r and r.get("wer"):
            v = r["wer"][2]
        row.append(fmt(v, 2))
    W("| " + " | ".join(row) + " |")
W("")

# --- Methodology ------------------------------------------------------------
W("## 6. Methodology notes\n")
W("- **WER_CLIP** caps per-utterance WER at 100% before averaging — a robust mean")
W("  insensitive to runaway hallucinations. `WER` (raw) and `WER_L50` (mean of")
W("  utts <50% WER) live in the `.wer` summary line alongside it.")
W("- **Unscoreable generations count as maximum error** (WER_CLIP 100) since")
W("  2026-09-01, rather than dropping out of the mean — otherwise a leaf mean is a")
W("  survivor mean and the most degenerate output is the output that disappears from")
W("  its own score. Each leaf's `.wer` footer reports `N_TRANSCRIBED:`/`N_IMPUTED:`")
W("  so the imputed share is always visible, and `assets/coreset/systems_*.csv`")
W("  carries it per system. The unscored utterances were re-measured before the")
W("  switch, not assumed: 43% of them transcribed on the retry. WER published")
W("  before that date is not comparable on an affected leaf —")
W("  see `docs/unscoreable_2026-09.md`.")
W("- **WER prep** is minimal and applied identically to REF and HYP: apostrophes")
W("  deleted, lowercase + Unicode punctuation strip; zh/ja/ko/th scored as CER")
W("  (`utils/run_wer.py`).")
W("- **Speaker similarity** = cosine × 100, ReDimNet (`PalabraAI/redimnet2`).")
W("- **MSR-UTMOS** = sarulab-speech MSR-UTMOS (SFI HuBERT) MOS predictor.")
W("- **Characters/s distance** = 2-Wasserstein distance between generated and")
W("  reference speaking-rate distributions (`scripts/speaking_rate_dist.py`); lower is better.")
W("- **MPM span** = distance from the real-speech split-half floor (0), normalized")
W("  by the contour-scrambled floor (1) (`utils/score_mpm_shuffle_floor.py`).")
W("- **TTSDS2 Pitch** = normalized mean-F0 distribution similarity")
W("  (`utils/score_ttsds_pitch.py`); interpret alongside speaker similarity.")
W("- **DS-WED** = multi-seed prosody diversity over HuBERT k-means tokens")
W("  (`utils/score_dswed.py`); higher = more seed-to-seed variation.")
W("- ASR is Qwen3-ASR served via vLLM (`scripts/serve_asr.sh`).")
W("- **Reliability gallery** — per-language edit-distance survival/tail plates for")
W("  the GT baselines: [`assets/reliability_gallery.html`](assets/reliability_gallery.html)")
W("  (`scripts/run_reliability.sh`).")
W("- Empty cells (`—`) = test set not run for that model, or score file missing.")
W("- Refresh by re-running `scripts/results_doc/run.sh` (or just `gen_md.py` if")
W("  only the markdown layout changed).")
W("")

with open(OUT, "w") as f:
    f.write("\n".join(buf))
print(f"Wrote {OUT} ({sum(len(b) for b in buf)} chars, {len(buf)} lines)")
