# Coreset panels — running ZTTS2-Eval at 1/10th the cost

Running ZTTS2-Eval in full means 57,366 generations on the Clean set (FLEURS-R, 35
languages) and 17,155 on the In-The-Wild set (VoxBlink2, 20 languages), before scoring.
This document describes four smaller **panels** that estimate the same numbers, how they
were selected, how accurate they actually are, and — importantly — what they cannot do.

| Panel | Task name | Items | vs full set | Purpose |
|---|---|--:|--:|---|
| Clean Mini | `clean_mini` | 3,500 | 16× | estimate the full Clean score |
| ITW Mini | `in-the-wild_mini` | 1,919 | 8.9× | estimate the full In-The-Wild score |
| Clean Delta | `clean_delta` | 200 | 287× | detect a change between two checkpoints |
| ITW Delta | `in-the-wild_delta` | 192 | 89× | detect a change between two checkpoints |

> **After the near-duplicate filter.** The ITW panels were selected at k = 2,000 / 200
> and exported at 1,941 / 194 items (`k_actual`). Removing the prompt/target
> near-duplicates from the full set (`scripts/prep/drop_near_duplicates.py`,
> `data/near_duplicates_removed.json`) dropped 22 items from ITW Mini and 2 from ITW
> Delta, so they now carry **1,919 / 192** items; the per-language weights were
> renormalized to sum to 1 and `k_actual` decremented. No Clean panel item was
> affected. The panels were **not** re-selected. Validation runs on the committed system ×
> item matrix (`assets/coreset/matrix_<set>.npz`, 57,388 Clean / 17,319 In-The-Wild
> items), built before the near-duplicate filter and the final `BENCHMARK_VERSION` 1
> rescore, so its full-set values differ slightly from `RESULTS_CURRENT.md`; it describes
> the panels as selected.
>
> **Speaker similarity.** On the In-The-Wild set a panel estimates the *prompt*
> similarity (`spk_simi_scores.txt`), not the held-out-recording similarity that is the
> primary ITW score of the full benchmark.

> All four panels were selected from routing-corrected scores; see
> [ASR routing](#asr-routing-resolved).

## Using a panel

A panel is an ordinary task directory, so nothing in the pipeline changes:

```bash
bash scripts/gen/run_omni.sh --model qwen3-tts-0.6b --task clean_mini --test-set en
bash scripts/score/run_score.sh --decode-dir results/omni_qwen3-tts-0.6b/clean_mini --test-set en

python scripts/coreset/score_panel.py --decode-dir results/omni_qwen3-tts-0.6b/clean_mini
```

`score_panel.py` applies the published per-item weights and prints the estimate of the
**full-benchmark** score with a bootstrap CI. Comparing two systems on a Delta panel:

```bash
python scripts/coreset/score_panel.py --decode-dir results/A/clean_delta --vs results/B/clean_delta
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
for `zonos2_accurate` on `clean_mini`), which is the check to run if either side is
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
  across 35 languages, so a per-language cell rests on 3–9 items — on `clean_delta`
  / `en` that is 3 items, and all 23 systems tie at 2.564. Delta's power lives in the
  paired aggregate over all 200 items, which is what `score_panel.py` reports.

## The estimator

ZTTS2-Eval's published score is a per-language mean over utterances
(`utils/average_wer.py`) followed by an **unweighted macro-average over languages**
(`scripts/results_doc/gen_md.py`):

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

- **`highvar` is catastrophic** — 4.257 (ITW) and 7.054 (Clean) against 0.325 / 0.287 for
  the shipped selector, a factor of 13–25. Selecting the most discriminative items preserves
  ordering but badly biases the level, exactly as expected: a panel of pathological items
  is not a panel of representative ones. This is the single clearest result here.
- **Denoising is worth 1.8–2.2× for medoid selection** — `medoid_response` 0.326 vs
  `medoid_raw` 0.719 on In-The-Wild, 0.329 vs 0.602 on Clean. The low-rank shrinkage removes
  real noise; there is just not much structure left underneath it.
- **Ignoring the language axis costs little at these sizes** — `random_global` 0.427 vs
  `random_perlang` 0.359 on In-The-Wild, 0.232 vs 0.321 on Clean.

Medoid selection is biased by construction (a medoid is not its cluster's mean) and was
kept in the comparison to measure that rather than assume it away. It converges to the
unbiased selectors only at large k.

Among the unbiased language-aware selectors, `strat_quantile` edges out `strat_meta` on
Clean at k=3,500 (0.262 vs 0.287) and `strat_response` trails it (0.319); on In-The-Wild
at k=2,000 the three read 0.314 / 0.325 / 0.345. The differences are within about 2 SE
(standard errors in `RESULTS_CORESET.md`), so they are not evidence of a real ordering —
`strat_meta` ships, and the full table is in
[`RESULTS_CORESET.md`](../RESULTS_CORESET.md) for anyone who wants to judge for
themselves.

So the coverage quotas in the shipped panels are justified on **robustness** grounds, not
on measured accuracy: they guarantee that rare phenomena stay represented for a future
system that might fail on them specifically — something nine historical families cannot
demonstrate one way or the other.

## Accuracy

Full tables in [`RESULTS_CORESET.md`](../RESULTS_CORESET.md). Mean absolute error of the
Mini panels against the full benchmark, on **held-out model families**:

| metric | Clean (k=3,500) | In-The-Wild (k=2,000 requested) | full-set spread (11 full-coverage systems, both sets) |
|---|--:|--:|---|
| WER_CLIP | **0.287** (SE 0.022) | **0.325** (SE 0.023) | systems span 4.7 – 56.7 |
| speaker similarity ×100 | **0.138** (SE 0.011) | **0.222** (SE 0.017) | systems span 13.8 – 87.7 |
| MSR-UTMOS | **0.007** (SE < 0.001) | **0.008** (SE < 0.001) | systems span 2.31 – 3.55 |

Pairwise, pooled over folds and seeds, the panel picks the better of two systems 99.3%
(In-The-Wild) / 95.3% (Clean) of the time when they differ by ≥ 0.5 WER_CLIP points, and
71.1% / 100.0% at 0.25–0.5. Each of these four figures comes from a bin with fewer than 20
distinct system pairs.

Two checks that matter more than the headline:

- **No selection overfitting.** Clean MAE is 0.289 on systems inside the selection pool
  and 0.287 on genuinely held-out families; In-The-Wild is 0.300 vs 0.325. The panels are
  not tuned to the systems that built them.
- **The confidence intervals are honest.** Quoting the shipped panel's bootstrap CI, the
  full-set value falls inside it 91% of the time on Clean (52/57 system × metric
  estimates) and 93% on In-The-Wild (53/57), against a nominal 95%.

For the Delta panel (k=200), the minimum detectable difference at 80% power is **5.85**
WER_CLIP points on Clean and **5.67** on In-The-Wild, against 7.23 / 6.55 for a
`random_perlang` panel of the same size.

`scripts/coreset/verify_e2e.sh` checks the plumbing end to end: it re-scores a system's
existing full-set wavs restricted to an exported panel through the unmodified pipeline,
and `compare_e2e.py` reports whether each full-set value falls inside the panel's
bootstrap CI.

**A published panel is one realization, and that shows.** Across the 20 systems in the
coverage tables, the shipped Mini panels' mean signed error is small (+0.005 WER_CLIP on
Clean, −0.043 on In-The-Wild), but the mean absolute error is 0.288 / 0.324 on a level
and 0.428 / 0.500 on a *difference* between two systems: panel errors do not cancel when
two systems share a panel. Treat a panel difference with at least the caution of a panel
level.

The panels use seed 0. No seed search was done: picking the seed with the lowest error on
the selection systems would be fitting the published artifact to the systems it was built
from, and the leave-family-out numbers above would no longer describe it.

## Scope and limits

Read these before quoting a panel number.

- **Mini panels resolve tiers, not the frontier.** Behind the In-The-Wild WER leader
  (drama-3, 3.20), the next five cloning systems sit within about 0.6 WER_CLIP points of
  each other (4.68–5.25, `RESULTS_CURRENT.md` §0). No panel will order them — and neither
  really does the full set, whose own paired standard error is ≈0.11.
- **Nine families is a thin basis.** Leave-family-out folds hold out one of ~9 independent
  model families. Per-fold rank correlations over so few systems are underpowered; the
  pooled pairwise sign-accuracy curve is the readout with real sample size.
- **Delta panels have no meaningful level.** They estimate differences only.
  `score_panel.py` refuses to print a calibrated absolute score for them.
- **Item×system interaction and sampling noise are not separately identifiable** from
  single-generation data. Every cell of the matrix is one generation, so the "denoised"
  fingerprint is really "the part a rank-r model can explain". Multi-seed regeneration on
  a small item subset would resolve this and is the natural next step.
- **Distributional prosody is out of scope.** Char/s, rhythm, pitch range and MPM
  span (and the TTSDS2 prosody that pitch range replaced) are distributions, not sums
  of per-item scores, so they cannot be estimated from a weighted subset. A panel can
  still be *scored* with them, but the result is the panel's own distribution, not an
  estimate of the full set's.
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
| `scripts/score/repair_wer_raw.py` | repairs `.wer.raw` records split by tabs/newlines in the ASR hypothesis |
| `scripts/coreset/build_matrix.py` | assembles the system × item matrix; `--verify` reproduces the published RESULTS tables from it |
| `scripts/coreset/denoise.py` | low-rank shrinkage of item fingerprints, rank by held-out-cell CV |
| `scripts/coreset/panel_select.py` | the ten selectors, coverage quotas, budget allocation |
| `scripts/coreset/validate.py` | leave-family-out sweep for Mini panels |
| `scripts/coreset/validate_delta.py` | matched-pair power analysis for Delta panels |
| `scripts/coreset/export_panel.py` | writes a panel as a runnable `data/<task>/` directory |
| `scripts/coreset/score_panel.py` | weighted estimate + bootstrap CI from a panel run |
| `scripts/coreset/common.py` | shared loaders (matrix, fingerprints, validation rows) and the estimator |
| `scripts/coreset/gen_results_md.py` | writes `RESULTS_CORESET.md` from the validation CSVs, matrix and shipped panels |
| `scripts/coreset/verify_e2e.sh` + `compare_e2e.py` | end-to-end panel check through the unmodified scoring pipeline |

## ASR routing (resolved)

Building the coreset matrix exposed a gap in the 2026-07-30 routing fix. It is closed; it
is recorded here because it changed the benchmark's own published results at the time.

`scripts/score/run_wer.py` sends a language to Whisper-large-v3 with the language forced when it
is **not** in `QWEN_LANGS` (today `zh ko th vi hi fa`; at the time of this analysis `en`
and `yo` were also on Qwen, and routing was opt-in via `ASR_ROUTING=1` — it is now the
enforced default). The ground-truth
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

A sharded re-score of the 22 languages across the GPUs of one allocation completed it
(WER only — speaker similarity and MSR-UTMOS are
language-agnostic and unaffected). The In-The-Wild sweep was run after the `eval_job.sh`
routing patch and appears unaffected.

The versioned release supersedes these partial rescores:
`scripts/score/rescore_all.sh` redoes every leaf (WER with routed ASR, both speaker-sim
passes, MSR-UTMOS) so that all score files carry the version stamp.

## A data-integrity note

Building the matrix surfaced a bug worth knowing about independently of coresets.
`scripts/score/run_wer.py` does not escape the ASR hypothesis, so a transcription containing a
literal TAB or newline splits its own record in `wav_res_ref_text.wer.raw`;
`utils/average_wer.py` then dies on that line and leaves a truncated `.wer` **with no
`WER:` footer**. Anything reading the footer silently loses the entire leaf.

Four leaves were affected, and three of them had silently dropped a language from
the published 26-language clean-tail table — enough to reorder the top two systems.
`scripts/score/repair_wer_raw.py` reassembles the records; it is idempotent and
rewrites nothing that already parses.
