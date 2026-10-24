# Coreset panels — running ZTTS2-Eval at 1/10th the cost

Running ZTTS2-Eval in full means 57,388 generations on the Clean set (FLEURS-R, 35
languages) and 17,319 on the In-The-Wild set (VoxBlink2, 20 languages), before scoring.
This document describes four smaller **panels** that estimate the same numbers, how they
were selected, how accurate they actually are, and — importantly — what they cannot do.

| Panel | Task name | Items | vs full set | Purpose |
|---|---|--:|--:|---|
| Clean Mini | `zero_shot_mini` | 3,500 | 16× | estimate the full Clean score |
| ITW Mini | `zero_shot_voxblink_mini` | 2,000 | 8.7× | estimate the full In-The-Wild score |
| Clean Delta | `zero_shot_delta` | 200 | 287× | detect a change between two checkpoints |
| ITW Delta | `zero_shot_voxblink_delta` | 200 | 87× | detect a change between two checkpoints |

> All four panels are selected from **routing-corrected** scores. Building the matrix
> surfaced that the 2026-07-30 ASR-routing fix had covered only 6 of the 28 FLEURS-R
> languages that need Whisper routing; the remaining 22 were re-scored before the Clean
> panels were selected. See [the note below](#the-asr-routing-gap) — it changed the
> benchmark's own published results, not just the panels.

## Using a panel

A panel is an ordinary task directory, so nothing in the pipeline changes:

```bash
bash run_generate_omni.sh --model qwen3-tts-0.6b --task zero_shot_mini --test_set en
DECODE_DIR=results/omni_qwen3-tts-0.6b/zero_shot_mini \
  bash run_infer_zonos2_eval.sh --task zero_shot_mini --test_set en

python scripts/coreset/score_panel.py --decode_dir results/omni_qwen3-tts-0.6b/zero_shot_mini
```

`score_panel.py` applies the published per-item weights and prints the estimate of the
**full-benchmark** score with a bootstrap CI. Comparing two systems on a Delta panel:

```bash
python scripts/coreset/score_panel.py --decode_dir results/A/zero_shot_delta --vs results/B/zero_shot_delta
```

### In the explorer

Panel results also land in the explorer, as their own tasks in the task picker:

```bash
python scripts/results_doc/build_results.py   # re-aggregates panel leaves with the weights
python explorer/serve.py --port 8000
```

`build_results.py` recognises a panel by its `data/<task>/panel.json` and re-aggregates
those leaves from the per-uid score files, because **the leaf's own `.wer` footer is the
wrong number for a panel** — it is an unweighted mean over the sampled items, not the
weighted estimator. The weights sit within about ±10% of uniform on the Mini panels, so
the footer looks plausible while being a different estimator; on `zonos2_accurate` /
`he`, for instance, it reads 10.824 where the panel estimate is 10.808. The per-language
board cells macro-average to exactly what `score_panel.py` prints (7.432 / 74.792 / 3.110
for `zonos2_accurate` on `zero_shot_mini`), which is the check to run if either side is
ever changed.

Three further things the explorer does for panels, all about not misreading them:

- **Every derivable row is backfilled.** A panel is a weighted subset of its source task,
  so for any system already scored on the full task the panel estimate is just that leaf
  restricted to the panel's uids — the same generations of the same texts from the same
  prompts, which is exactly how `validate.py` scored panels offline. Without it a panel
  board carries only the two or three systems that were re-run on the panel task.
  - Rows actually generated on the panel task are tagged `run`; the rest are what the
    panel *would* have reported, not an independent trial of it. Where both exist
    (`zonos2_accurate`, `kokoro`) they agree to 0.02 pt macro WER_CLIP — 0.045 pt mean
    absolute per language — which is the size of the generation noise, not of a bias.
  - A leaf covering only part of a panel renormalises the weights over what it has and
    so estimates a self-selected population; the covered share shows as a `%` tag and
    anything under `MIN_PANEL_COVERAGE` (90%) is dropped, listed on stderr. Today that
    drops 19 rows, all on the known-partial leaves (`cosyvoice3` ar/fa/hi/he/th,
    `omni_voxtral-tts` ko/th/vi, `omni_glm-tts` fil).
- **The GT and Whisper floors are recomputed on the panel's items** by the same
  restriction. Those baselines are only ever scored on the full task, and charging a
  system the full-set floor would misstate every Δ-vs-GT read on the board.
- **Each board carries its scope note**, and Mini cells show their bootstrap CI, since a
  subsample's sampling error is exactly what a reader needs to see. Delta boards say
  outright that their level is biased by construction and only differences may be read.
  They also say that the board is the wrong shape for them: a Delta panel spends k=200
  across 35 languages, so a per-language cell rests on 3–9 items — on `zero_shot_delta`
  / `en` that is 3 items, and all 23 systems tie at 2.564. Delta's power lives in the
  paired aggregate over all 200 items, which is what `score_panel.py` reports.

## The estimator

ZTTS2-Eval's published score is a per-language mean over utterances
(`utils/average_wer.py:63-66`) followed by an **unweighted macro-average over languages**
(`scripts/results_doc/gen_md.py:146-152`):

```
S_m = (1/|L|) · Σ_l (1/N_l) · Σ_{i∈l} y_mi
```

A panel estimates it with the matching weighted form, weights summing to 1 *within each
language*:

```
Ŝ_m = (1/|L|) · Σ_l Σ_{j∈A_l} w_j · y_mj
```

Two consequences shape the whole design. Selection **decomposes per language** — each
language is an independent stratified-sampling problem. And per-language budget is a free
choice, because language weights are equal regardless of `N_l`.

`score_panel.py` also reports a **calibrated** estimate, a difference estimator that adds
back the panel items' historical difficulty. It is offered next to the raw weighted score
rather than replacing it: the raw score depends on nothing but the published weights, so
anyone can recompute it, while the calibrated one depends on this repo's private
historical matrix.

## How the panels were selected

Per language: partition items into **coverage cells**, apportion the language's budget
across cells in proportion to cell mass, and draw at random within each cell. Weights are
`(N_cell / N_lang) / n_drawn`, which makes the estimator design-unbiased.

Cells are built from published item metadata only — length tercile, a numerals/abbreviation
flag, and on the In-The-Wild set the prompt MSR-UTMOS bin the set was originally stratified
on. The grid is **coarsened when the budget is small**: forcing a floor of one item into
each of 30 cells with a 48-item language budget spends everything on the floor and
measurably hurt accuracy at k=1000 before this was added.

Budget is split across languages by Neyman allocation (proportional to each language's
per-utterance spread) with a floor of 12 items so every language stays individually
reportable.

### What did *not* work, and why that matters

Ten selectors were compared under leave-family-out validation (`scripts/coreset/validate.py`).
The headline result is a negative one:

**Response-space clustering does not beat task-balanced stratified random sampling.**
At the shipped sizes every language-stratified unbiased selector ties within noise. This
is not a bug in the implementation — it follows from the structure of the data. Held-out
cell cross-validation (`scripts/coreset/denoise.py`) selects **rank 1** in nearly every
language: items differ along a single axis of overall difficulty, not along multiple axes
of "which system does this favour". With one dimension there is very little for clustering
to exploit that a simple stratification does not already capture.

What *did* separate, in WER_CLIP MAE at the shipped sizes:

- **`highvar` is catastrophic** — 4.44 (ITW) and 6.42 (Clean) against ~0.2–0.3 for the
  shipped selector, a factor of 20. Selecting the most discriminative items preserves
  ordering but badly biases the level, exactly as expected: a panel of pathological items
  is not a panel of representative ones. This is the single clearest result here.
- **Denoising is worth 1.3–2.5× for medoid selection** — `medoid_response` 0.21 vs
  `medoid_raw` 0.52 on In-The-Wild, 0.43 vs 0.55 on Clean. The low-rank shrinkage removes
  real noise; there is just not much structure left underneath it.
- **Ignoring the language axis costs ~1.6× on In-The-Wild** — `random_global` 0.371 vs
  `random_perlang` 0.232 — but **not on Clean**, where the two are indistinguishable
  (0.243 vs 0.250 at k=3,500). With 35 languages and a large budget, a global random
  sample already lands enough items in every language. The In-The-Wild set has languages
  as small as 109 items, where a global sample can miss or barely touch them. So this is a
  property of the *set*, not a universal law — worth stating because it is easy to
  over-generalize from one benchmark.

Medoid selection is biased by construction (a medoid is not its cluster's mean) and was
kept in the comparison to measure that rather than assume it away. It converges to the
unbiased selectors only at large k.

Among the unbiased language-aware selectors, `strat_quantile` and `strat_response` edge
out `strat_meta` on Clean at k=3,500 (0.221 and 0.238 vs 0.290) while `strat_meta` is
level or better on In-The-Wild at every size. Pooled, the gap is about 2 SE and the sign
is inconsistent across sets, so it is not evidence of a real ordering — `strat_meta`
ships, and the full table is in
[`RESULTS_CORESET.md`](../RESULTS_CORESET.md) for anyone who wants to judge for
themselves.

So the coverage quotas in the shipped panels are justified on **robustness** grounds, not
on measured accuracy: they guarantee that rare phenomena stay represented for a future
system that might fail on them specifically — something nine historical families cannot
demonstrate one way or the other.

## Accuracy

Full tables in [`RESULTS_CORESET.md`](../RESULTS_CORESET.md). Mean absolute error of the
Mini panels against the full benchmark, on **held-out model families**:

| metric | Clean (k=3,500) | In-The-Wild (k=2,000) | full-set spread, for scale |
|---|--:|--:|---|
| WER_CLIP | **0.290** (SE 0.035) | **0.201** (SE 0.019) | systems span 4.6 – 72 |
| speaker similarity ×100 | **0.202** (SE 0.019) | **0.131** (SE 0.016) | systems span 13 – 88 |
| MSR-UTMOS | **0.007** (SE 0.001) | **0.007** (SE 0.001) | systems span 2.30 – 3.53 |

Pairwise, pooled over folds, the In-The-Wild panel picks the better of two systems
**100%** of the time when they differ by ≥0.5 points of WER_CLIP, and 87% at 0.25–0.5.
The Clean panel is ≥94% everywhere and 100% in every band except 1–2 points.

Two checks that matter more than the headline:

- **No selection overfitting.** Clean MAE is 0.284 on systems inside the selection pool
  and 0.290 on genuinely held-out families; In-The-Wild is 0.211 vs 0.201. The panels are
  not tuned to the systems that built them.
- **The confidence intervals are honest.** Quoting the shipped panel's bootstrap CI, the
  full-set value falls inside it 93% of the time on Clean (43/46 system × metric
  estimates) and 97% on In-The-Wild (31/32), against a nominal 95%.

For the Delta panel (k=200), the minimum detectable difference at 80% power is **2.13**
points of WER_CLIP, against 3.38 for a random panel of the same size — roughly half the
items for the same power.

An end-to-end run confirms the plumbing as well as the statistics: scoring
`zonos2_accurate` and `kokoro` on an exported panel through the unmodified pipeline put
5 of 6 estimates inside their bootstrap CIs, the sixth being the WER of the set affected
by the routing gap below.

**A published panel is one realization, and that shows.** Measured across the 11 systems
in the selection pool, the shipped panels carry a common-mode offset — `+0.184` WER_CLIP
on Clean, `−0.161` on In-The-Wild — on top of each system's own error. It is tempting to
assume this cancels when comparing two systems on the same panel; it does not. Measured,
the mean error on a *difference* is 0.180 (Clean) and 0.313 (ITW), i.e. no better than the
error on a level. Treat a panel difference with the same caution as a panel level.

The panels use seed 0. No seed search was done: picking the seed with the lowest error on
the selection systems would be fitting the published artifact to the systems it was built
from, and the leave-family-out numbers above would no longer describe it.

## Scope and limits

Read these before quoting a panel number.

- **Mini panels resolve tiers, not the frontier.** The In-The-Wild top three
  (zonos2-accurate −4.6, omnivoice-rt −4.2, voxcpm2 −4.0) sit within 0.6 points. No panel
  will order them — and neither really does the full set, whose own paired standard error
  is ≈0.11.
- **Nine families is a thin basis.** Leave-family-out folds hold out one of ~9 independent
  model families. Per-fold rank correlations over so few systems are underpowered; the
  pooled pairwise sign-accuracy curve is the readout with real sample size.
- **Delta panels have no meaningful level.** They estimate differences only.
  `score_panel.py` refuses to print a calibrated absolute score for them.
- **Item×system interaction and sampling noise are not separately identifiable** from
  single-generation data. Every cell of the matrix is one generation, so the "denoised"
  fingerprint is really "the part a rank-r model can explain". Multi-seed regeneration on
  a small item subset would resolve this and is the natural next step.
- **TTSDS2 prosody, which is replaced by pitch range, is out of scope.** It is not present in the current
  `results/` tree, and TTSDS is distributional — it is not decomposable to per-item scores,
  so it cannot be estimated from a weighted subset. A panel can still be *scored* with it,
  but the result is the panel's own distribution, not an estimate of the full set's.
- **Refresh trigger.** The panels are selected from 11 systems observed up to 2026-08.
  Re-run `scripts/coreset/run.sh` when the system pool grows materially, and re-check the
  error-vs-k curves on the new families before trusting the old sizes.

## Rebuilding

```bash
bash scripts/coreset/run.sh          # matrix → fingerprints → validation → panels
```

Everything is offline and CPU-only; it reads the per-utterance scores already in
`results/` and never regenerates audio. Stages:

| Script | Does |
|---|---|
| `scripts/repair_wer_raw.py` | repairs `.wer.raw` records split by tabs/newlines in the ASR hypothesis |
| `scripts/coreset/build_matrix.py` | assembles the system × item matrix; `--verify` reproduces the published RESULTS tables from it |
| `scripts/coreset/denoise.py` | low-rank shrinkage of item fingerprints, rank by held-out-cell CV |
| `scripts/coreset/panel_select.py` | the ten selectors, coverage quotas, budget allocation |
| `scripts/coreset/validate.py` | leave-family-out sweep for Mini panels |
| `scripts/coreset/validate_delta.py` | matched-pair power analysis for Delta panels |
| `scripts/coreset/export_panel.py` | writes a panel as a runnable `data/<task>/` directory |
| `scripts/coreset/score_panel.py` | weighted estimate + bootstrap CI from a panel run |

## The ASR routing gap

This one was found by the coreset work but is independent of it, and it matters more than
the panels do.

`utils/run_wer.py` sends a language to Whisper-large-v3 with the language forced when it
is **not** in `QWEN_LANGS` (today `zh ko th vi hi fa`; at the time of this analysis `en`
and `yo` were also on Qwen) and `ASR_ROUTING=1`. The ground-truth
baselines were scored that way from the beginning. The synthesized audio was not: the
2026-07-30 fix re-scored only the six languages where misrouting was most visible
(`bg ca hr nb sk uk`), leaving **22 of the 28** FLEURS-R languages that need routing still
scored by Qwen3-ASR auto-detect.

Because GT is routed and synth is not, every "Δ vs GT floor" on those languages is
inflated by the full amount. Re-scoring `zonos2_accurate` measured it:

| | inflation (WER_CLIP points) |
|---|---|
| large | `he` +53.1, `fi` +14.6, `fil` +14.5, `hu` +13.9, `ro` +11.7, `el` +10.1, `cs` +9.5, `sv` +8.9, `da` +5.5, `pl` +5.0, `tr` +4.6, `ms` +3.3 |
| mild (auto-detect mostly copes) | `ar` `id` `fr` `it` `nl` `de` `ja` `ru` `es` `pt`, all ≤1.5 |

`zonos2_accurate`'s 26-tail-language WER_CLIP moves from 13.87 to **7.77**.

The diagnosis is unambiguous rather than inferred:

1. Languages whose score changed by **exactly zero** on re-scoring are precisely
   `QWEN_LANGS` plus the six already fixed — which also proves the scoring is
   deterministic, so the other differences are not noise.
2. Stored `he` hypotheses are 90% Hebrew with Arabic and Cyrillic drift; GT's are 100%
   Hebrew, and re-routed synth is 100% Hebrew.

`scripts/rescore_routed_all.sh` completes the re-score, sharding the 22 languages across
the GPUs of one allocation (WER only — speaker similarity and MSR-UTMOS are
language-agnostic and unaffected). The In-The-Wild sweep was run after the `eval_job.sh`
routing patch and appears unaffected.

## A data-integrity note

Building the matrix surfaced a bug worth knowing about independently of coresets.
`utils/run_wer.py` does not escape the ASR hypothesis, so a transcription containing a
literal TAB or newline splits its own record in `wav_res_ref_text.wer.raw`;
`utils/average_wer.py` then dies on that line and leaves a truncated `.wer` **with no
`WER:` footer**. Anything reading the footer silently loses the entire leaf.

Four leaves were affected, and three of them had silently dropped a language from
`RESULTS_NEWLANGS.md` — enough to reorder the top two systems. See the correction note in
that file. `scripts/repair_wer_raw.py` reassembles the records; it is idempotent and
rewrites nothing that already parses.
