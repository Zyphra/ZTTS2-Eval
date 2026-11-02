# Methodology controls — 2026-08-29

A four-pillar methodological review (data construction, metric construct validity,
statistics, experimental fairness) raised several confounds that the benchmark could
not rule out from existing artifacts. This document records the controls run in
response, their measured outcomes, and what was adopted. Each control writes to
alternate filenames first, so published scores are never overwritten by a probe.

Scripts: `scripts/control_scorer_swap.sh`, `scripts/control_spk_embedder.sh`,
`scripts/control_seed_study.sh`, `scripts/adopt_whisper_en.sh`.

---

## C1 — ASR scorer family self-favoritism · **ADOPTED (en routing changed)**

**Question.** Qwen3-ASR-1.7B was the WER scorer for `en zh ko th vi hi fa`, and two
ranked systems are Qwen3-TTS-0.6b/1.7b — same model family, plausibly shared training
data and tokenizer. Does the scorer favor its own family?

**Design.** Rescore the *same audio* for `zero_shot/{en,zh}` with Whisper-large-v3
(forced language) instead of Qwen3-ASR, for the two qwen3 systems plus a four-system
cohort (fish-s2-pro, omnivoice-reftext, voxcpm2, zonos2_accurate) and GT. Readout is
**difference-in-differences**: a scorer swap moves every system's WER a little (a
common-mode effect that cancels in rankings); family favoritism shows up as qwen3
moving *differently from the cohort*. Item-level bootstrap, 5,000 draws.

**Result.** English confirms the confound; Chinese does not.

| system | en DiD (Whisper−Qwen, vs cohort) | zh DiD |
|---|--:|--:|
| qwen3-tts-1.7b | **+0.51** CI95 [+0.28, +0.74] | −0.22 [−0.57, +0.12] (ns) |
| qwen3-tts-0.6b | **+0.70** CI95 [+0.46, +0.96] | +0.01 [−0.34, +0.37] (ns) |

Both qwen3 systems score relatively *worse* once a non-family scorer is used. The
effect (0.5–0.7 WER_CLIP points) is the size of several adjacent rank gaps in the
published base-9 table, where qwen3-tts-1.7b ranked first.

**Adoption.** `en` was removed from `QWEN_LANGS` (`utils/run_wer.py`): English now
routes to Whisper-large-v3 with the language forced, exactly as the non-QWEN_LANGS
tail always has. `zh` stays on Qwen3-ASR — its DiD is null, and Qwen is the better
Chinese scorer on clean read speech (`docs/asr_whisper_vs_qwen.md`). All 55 English
leaves (every system **and** both GT baselines) were rescored symmetrically by
`scripts/adopt_whisper_en.sh`.

**Tradeoff, stated explicitly.** Whisper's English GT floor is *higher* than Qwen's
(4.45 vs 3.78 WER_CLIP), so absolute English numbers rise for everyone. That is a
common-mode shift that cancels in system-vs-system and Δ-vs-GT comparisons, whereas
the family bias is differential and does not. A benchmark that ranks systems should
prefer the unbiased comparison over the lower floor.

**Comparability.** English WER and every aggregate containing it (base-9 `avg`,
VoxBlink 20-lang means, Δ-vs-GT) changed at this boundary. `RESULTS_NEWLANGS.md`
(26-language tail) does not include English and is unaffected.

---

## C2 — Seen-speaker speaker-similarity embedder · *running*

**Question.** The shipped embedder (ReDimNet2-B6 `vb2+vox2_v0`) is trained on
VoxBlink2 — the in-the-wild evaluation corpus itself — so every ITW speaker score is
computed in a seen-speaker regime, including the 76-point same-speaker "ceiling".

**Design.** Rescore ITW `en/zh/de/th`, all systems + GT, with the vox2-only variant
(`--dataset vox2`, VoxBlink2-free training) into `spk_simi_scores.vox2.txt`.
Readout: per-language Kendall τ of the system ordering vs published, plus whether any
adjacent published rank flips beyond its paired CI.

**Adoption rule (pre-registered).** τ < 0.9 on any language, or an adjacent-rank flip
beyond CI → rescore speaker similarity tree-wide with the held-out embedder and
republish. Otherwise → disclosure note only.

**Result.** Rankings are robust to the contamination; the held-out embedder is the
worse-calibrated instrument.

| language | systems | Kendall τ (published vs held-out) | GT same-speaker ceiling: published → held-out |
|---|--:|--:|---|
| en | 27 | +0.909 | 76.4 → 69.2 |
| zh | 19 | +0.977 | 80.7 → 71.0 |
| de | 18 | +0.948 | 78.0 → 71.9 |
| th | 11 | +0.927 | 81.8 → 70.8 |

τ clears the 0.9 threshold in every language, so seen-speaker training is not
reordering the board. Of ~12 adjacent flips, **eleven occur between pairs the
published embedder never resolved** (their paired ΔCI already spanned 0 — e.g. en
`qwen3-0.6b − glm-tts` +0.13 [−0.33, +0.59]); flipping an unresolved pair is noise,
not contradiction.

**One flip does contradict a resolved ordering**: on German, published
`fish-s2-pro − qwen3-tts-1.7b` = +0.97 CI95 [+0.43, +1.49] (resolved), while the
held-out embedder gives −0.20 [−0.79, +0.38] — nominally reversed and *unresolved*.
So the held-out embedder does not assert the opposite ordering; it fails to reproduce
a published one.

**Adoption — disclosure, not a swap, and the reason matters.** A tree-wide switch to
the vox2-only embedder would trade a documented contamination for a demonstrably less
discriminative instrument: its same-speaker GT ceiling collapses by 6–11 points and
the number of systems scoring *above* the human same-speaker anchor rises (en: 1 → 4;
zh: 2 → 3). A metric that puts four synthetic systems above real same-speaker
similarity is measuring identity less well, not more honestly. Combined with τ ≥ 0.91
everywhere, the evidence says the contamination is real but not
conclusion-changing, while the proposed replacement is worse calibrated.

Two concrete consequences, both applied:
1. The published speaker-similarity claim `fish-s2-pro > qwen3-tts-1.7b on de` is
   **demoted to unresolved** — it does not survive an embedder change and should not
   be read as a fact about speaker identity.
2. The seen-speaker caveat stays in README's provenance section, now with the measured
   τ range so a reader can size the risk.

The control's per-utterance outputs are retained as `spk_simi_scores.vox2.txt` beside
the published files for anyone who wants to re-derive this. One leaf
(`cosyvoice3/zero_shot_voxblink/th`) failed the 98 % coverage gate under the held-out
model — 6 of 176 generations are too short for its convolution stack, a known
cosyvoice3 failure mode on unsupported Thai — and was excluded from the τ above rather
than scored on a biased survivor subset.

---

## C3 — Generation sampling variance · **ADOPTED (uncertainty guidance)**

**Question.** Every published score is one generation per prompt (zonos2 at fixed seed
0; vLLM-Omni-served systems on unseeded server defaults). All bootstrap CIs condition
on those realized draws, so run-to-run generation variance is invisible — while the
in-the-wild top-4 are separated by only 0.26–0.32 WER_CLIP points.

**Design.** Regenerate the ITW top-4 twice more under their *published* configs
(en + tr), score identically, into `results/_seeds/run{1,2}/`. Compare the spread of
the three independent generations per (system, language).

**Result** (four systems × two languages, three independent generations each, all on
the post-C1 English routing):

| system | lang | published | run1 | run2 | sd (WER_CLIP) | sd (spk) |
|---|---|--:|--:|--:|--:|--:|
| zonos2_accurate (seeds 0/1/2) | en | 2.07 | 2.01 | 2.14 | 0.065 | 0.342 |
| zonos2_accurate (seeds 0/1/2) | tr | 6.14 | 6.25 | 6.22 | 0.060 | 0.117 |
| omnivoice-reftext | en | 2.02 | 1.93 | 2.22 | 0.147 | 0.042 |
| omnivoice-reftext | tr | 10.66 | 11.17 | 11.14 | 0.284 | 0.021 |
| voxcpm2 | en | 2.22 | 2.19 | 2.23 | 0.025 | 0.035 |
| voxcpm2 | tr | 6.65 | 7.23 | 7.43 | 0.404 | 0.123 |
| fish-s2-pro | en | 1.68 | 1.89 | 1.72 | 0.108 | 0.148 |
| fish-s2-pro | tr | 7.16 | 7.35 | 7.42 | 0.137 | 0.104 |

Three findings.

**(a) Generation variance is real but small at macro scale.** Mean per-language sd is
0.154 WER_CLIP points. Over a 20-language macro basket that contributes ≈ 0.034 per
system, so a *paired* macro difference carries about **±0.10 (95 %)** of generation
noise on top of its item-sampling CI. Added in quadrature to the published half-widths
(±0.26…±0.55) that is a **3–7 % widening** — it does not move any adjacent-pair
verdict in `RESULTS_VOXBLINK.md`.

**(b) It is not negligible per language.** Single-cell comparisons in one language
(voxcpm2 on `tr`: sd 0.40, range 0.78) should not be read below ~0.5 WER points.

**(c) Explicit seeding is not the noisy condition.** zonos2_accurate varied across
*different seeds* (0/1/2) by sd ≈ 0.06 — less than the unseeded omni-served systems
vary between identical-config runs. The published one-generation policy is therefore
not made riskier by zonos2's fixed seed; the exposure is in the unseeded servers.

*(An earlier revision of this document quoted ±0.17 and "~20 % widening" from the `tr`
column alone, before the English runs were rescored onto the post-C1 routing. The
figures above supersede it: `tr` is the noisier of the two languages.)*

**Adoption.** No re-sorting (the ranking-semantics decision for this campaign is
annotate-only). The published tie rule now accounts for generation noise: adjacent
systems are treated as unresolved when their paired ΔCI spans 0, and near-threshold
gaps should be read against a CI inflated by ~20 % for unmeasured generation variance.
`en` results from the seed runs were rescored onto the post-C1 routing before use
(a run scored at 11:54 predates the routing change at 11:57 — mixing them would have
compared a Qwen-scored generation against a Whisper-scored one).

---

## C4 — MPM loudness confound · *running (adoption unconditional)*

**Question.** `docs/mpm_itw_limits.md` measured that MPM span tracks system output
loudness at r = −0.979 on FLEURS-R, because the energy channel is bucketed against an
absolute 0–0.2 window: a pure gain change manufactures span. The p95 per-utterance
normalization fix (`e / p95(e)`, clipped to [0,1]) was designed and validated
(rank preservation τ +0.867) but never applied.

**Design/adoption.** Applied at `utils/score_mpm_shuffle_floor.py` (the repo owns the
tokenization; the upstream ttsds package is untouched, preserving bit-identity for
anyone calling it directly). The anchor cache key does *not* hash the featurization
code, so the version tag was bumped `v3` → `v4` — without that, every anchor would
have been silently reused and the new spans would be meaningless. The full FLEURS-R
MPM array (35 languages × all systems) is being recomputed.

**Result.** All 35 FLEURS-R languages rebuilt (35/35 array tasks succeeded).

- **Rank preservation matches the design prediction almost exactly**: Kendall τ between
  old and new spans is **+0.863 mean** across 35 languages (range +0.545…+0.970),
  against the +0.867 `docs/mpm_itw_limits.md` projected from its validation subset.
- **Inflation removed**: mean span falls 0.569 → 0.495. Per-language mean drops are
  largest where output loudness varied most across systems.
- **The loudness confound explained part, but not all, of the above-ceiling anomaly.**
  Spans above 1.0 mean a system scores worse than *contour-scrambled real speech*, which
  should be impossible for intelligible audio. kokoro — the exemplar quiet system — is
  now back under the ceiling on `es` (1.052 → 0.752) and `pt` (1.216 → 0.751), but stays
  above it on `fr`, `hi`, `it`, `ja` (3.818 → 3.485) and `zh` (1.966 → 1.942). Those are
  precisely the languages kokoro does not support (it ships preset voices for 6 of 9),
  so the residual is a real prosody failure, not a gain artifact. The fix removed the
  measurement error without laundering the finding underneath it.
- **Largest reordering is `th`** (τ +0.545, 12 systems), then `el` (+0.692) and `id`
  (+0.744) — worth knowing before quoting a single Thai MPM cell across the boundary.

Published MPM spans are **not comparable across this change**. The old TSVs are kept at
`results/_mpm_shuffle_floor/pre_p95_20260829/` so any before/after claim can be re-derived.

---

## C5 — Asymmetric with-transcript rows · *running*

**Question.** Conditioning is harness policy, not a model property, and it moves WER
enormously (omnivoice: 34.2 without the reference transcript vs 3.2 with it, same
weights). The published rankings contained exactly one "rescued" with-transcript
variant — `omnivoice-reftext` — while qwen3 and voxcpm2 were held at no-ref-text
(qwen3 additionally forced into `x_vector_only_mode`). That is an asymmetry between
systems introduced by the harness.

**Design.** New registry keys `qwen3-tts-1.7b-reftext`, `qwen3-tts-0.6b-reftext`,
`voxcpm2-reftext` (`scripts/omni_models.sh`, same model, ref_text kept — the
`omnivoice-reftext` idiom), generated and scored over the clean and in-the-wild sets
so every no-ref system has a symmetric with-transcript row.

**Result — the published speaker-similarity deficits of the no-ref systems were
largely a harness artifact.** Macro over languages complete in *both* arms:

| system | set | n langs | WER_CLIP no-ref → +ref | spk no-ref → +ref |
|---|---|--:|--:|--:|
| qwen3-tts-1.7b | clean (35-lang) | 34 | 24.66 → 22.67 (**−1.98**) | 69.41 → 80.79 (**+11.39**) |
| qwen3-tts-0.6b | clean | 34 | 30.72 → 29.36 (−1.36) | 68.94 → 79.94 (**+10.99**) |
| voxcpm2 | clean | 33 | 14.25 → 12.95 (−1.31) | 78.70 → 86.10 (**+7.40**) |
| qwen3-tts-1.7b | in-the-wild | 18 | 9.79 → 10.93 (+1.14) | 67.30 → 76.26 (**+8.95**) |
| qwen3-tts-0.6b | in-the-wild | 19 | 15.16 → 15.72 (+0.57) | 64.25 → 74.68 (**+10.43**) |
| voxcpm2 | in-the-wild | 20 | 5.36 → 5.30 (−0.06) | 73.88 → 77.84 (**+3.96**) |

Speaker similarity gains **+4 to +11 points** purely from a conditioning change the
harness made, not a model change. On the base-9 clean set the effect is starker still
(en: voxcpm2 66.1 → 84.0, qwen3-1.7b 68.5 → 80.0). The published tables ranked
qwen3 and voxcpm2 at ~64–70 spk against with-transcript systems at ~85 and read that
as a cloning-quality gap; **most of that gap was the reftext policy.** For qwen3 the
no-ref path additionally forces `x_vector_only_mode`, which is the weaker speaker
conditioning the model offers.

WER moves the other way and is smaller: with the transcript, qwen3 is ~1.3–2.0 points
*better* on the clean tail but ~0.6–1.1 *worse* in the wild; voxcpm2 is roughly
unchanged. So the conditioning choice is a genuine trade, and the published tables
showed only one side of it.

**A failure mode the with-transcript arm exposes.** qwen3-tts-*-reftext **over-generates
badly on Hindi**: same model, same prompts, only conditioning differs —

| | median duration | p95 | files > 30 s |
|---|--:|--:|--:|
| qwen3-1.7b (published, no-ref) | 7.8 s | 17.4 s | 9 / 1500 |
| qwen3-1.7b-reftext | 19.9 s | 163.8 s | **708 / 1500** |

Those files pile up at the model's 164 s generation cap, and the ASR loops on them:
at a 60 s request timeout ~15 % of Hindi utterances failed to score, and **raising the
timeout to 300 s barely helped** (coverage 0.973 → 0.971) because the transcriber, not
the file length, is what runs away. Five leaves (`hi` for both qwen3-reftext systems on
both sets, plus `vi` for 1.7b) are therefore published as **explicitly partial**, with
their true coverage in the `N_SCORED:`/`N_EXPECTED:` footers (e.g. 1220/1256, 1277/1500)
and Δ-vs-GT restricted to covered languages as usual. They were *not* silently dropped:
averaging only the utterances that survived would flatter precisely the most broken
configuration — the survivor bias `utils/run_wer.py` already guards against elsewhere.

`ASR_TIMEOUT` is now an env knob on `utils/run_wer.py` (default unchanged at 60 s) so a
future leaf with legitimately long audio can be scored without editing the scorer.

**Open recommendation — taken on 2026-09-01, see `docs/unscoreable_2026-09.md`.** The
most defensible treatment of an unscoreable degenerate generation is to score it as
maximum error (WER_CLIP 100) rather than excluding it. That is a change to a core
metric's semantics and has to be applied symmetrically to *every* system's timeouts
to be fair, which is why it was left as a maintainer decision rather than a late
unilateral change.

Two things about it turned out differently from the assumption recorded here. It did
**not** need a full benchmark rescore: only 4197 utterances in 135 of 1557 leaves were
ever unscored, each one identifiable per leaf, so re-measuring just those residuals
under one policy buys the same uniformity for 0.3 % of the compute. And re-measuring
was not a formality — **43 % of them transcribed on the retry**, meaning those leaves
had lost a scoring run rather than produced degenerate audio, and imputing them would
have invented a much worse result than the truth. The `hi` leaves described above are
the other half of the split: they stayed unscoreable and are now maximum error, which
moves them 57.29 → 63.61 and 61.84 → 67.25.

A second defect surfaced during that audit and is fixed with it: `average_wer.py`
assumed `n_expected == n_scored` for any leaf without a `.meta` sidecar — 1110 of
1557 — so those leaves reported full coverage regardless of how much of them had
actually been scored.

---

## C6 — MPM edge-silence confound · **ADOPTED (MPM re-scored)**

**Question.** Building a prosody index over every prosody metric, MPM span disagreed
with all the others: across the 17 full-coverage systems it was uncorrelated with
rhythm pace, char/s, pitch range, MSR-UTMOS, SPK and WER on FLEURS-R, and *negatively*
related to the rhythm pause factor (within-language median Spearman −0.25, positive in
only 14% of languages). It was not noise — its system ranking agreed across languages
(mean pairwise Spearman +0.71, above rhythm's +0.53–0.60) — so it was measuring
something stable, just not what the other metrics measure.

**Diagnosis.** MPM tokenizes pitch, energy and voice-activity contours from *untrimmed*
audio (the first 1000 frames, ~11.6 s). The FLEURS-R recordings carry **2.45 s** of
leading + trailing silence on average; most systems emit tightly trimmed audio (voxtral
0.06 s, ZONOS2 0.09 s, moss-tts 0.10 s, fish-s2-pro 0.15 s). Within each language, span
tracked `|system edge silence − real edge silence|` at Spearman **+0.70** median,
positive in **35/35** languages (+0.78 across systems): tightly trimmed systems were
scored as far from real speech for their missing silence.

**Causal probe** (`en de es fr it ja ru zh`, 8 systems spanning the padding range):
trimming leading/trailing silence from reference *and* system audio with the char/s
metric's own `librosa.effects.trim(top_db=35)` and re-scoring with the unchanged scorer
dropped the span–padding correlation from **+0.80 to +0.27** (median over languages) and
reordered the systems (Spearman before vs after +0.54). Tightly trimmed systems moved
to real-speech-like spans — moss-tts 0.63 → 0.04, higgs-audio-v3 0.37 → 0.06,
fish-s2-pro 0.85 → 0.20, ZONOS2 accurate 0.48 → 0.17 — while voxtral, which speaks in
preset voices rather than the prompt's, stayed near the scrambled floor (1.01 → 0.97).

**Design/adoption.** `utils/score_mpm_shuffle_floor.py` trims leading/trailing silence
(`TRIM_TOP_DB = 35`) in `_load_wav`, the single loader for reference, systems and the
ground-truth split, and the anchor cache tag moves `v4` → `v5` (the content hash cannot
see featurization changes). The full FLEURS-R MPM array (35 languages × all systems) and
the 35 ground-truth split-half floors were recomputed.

**Result.** 35/35 languages re-scored, system rows and ground-truth floors alike. The 64
probe cells reproduce (Spearman +0.994, median |Δ| 0.002).

- **Confound removed.** Over all systems, within-language span vs edge-silence gap falls
  from median **+0.58 to +0.02** (positive in 100% → 57% of languages). The ground-truth
  split-half floor stays at zero (mean span −0.009 → −0.015).
- **MPM now agrees with the other prosody metrics.** Within language, median Spearman
  against char/s distance is **+0.49** (same sign in 100% of languages), rhythm pace
  +0.40 (97%), pitch range +0.30 (91%), rhythm pause +0.24 (83%). Across the 17
  full-coverage systems it correlates **+0.30 to +0.53** with every prosody metric, up
  from −0.36 to +0.17.
- **Rankings move a lot.** Kendall τ between old and new spans averages **+0.13** across
  languages (range −0.25…+1.00; lowest `pl` −0.25, `el` −0.24, `sk` −0.19, `ro` −0.17),
  against +0.86 for the C4 loudness fix. Mean span falls 0.418 → 0.295.
- **Out-of-language failures were being masked.** Cells above 1.0 rise 31 → 47, and they
  fall where the vendor does not support the language. omni_glm-tts (zh/en) holds 0.18 →
  0.17 on its own languages but goes 0.21 → 1.44 on the other 21 (19 above 1).
  cosyvoice3 improves 0.25 → 0.03 on its 9 languages and worsens 0.50 → 1.29 elsewhere.
  Untrimmed, a system whose silence total matched the reference's could pass with any
  contour.
- **Pausing is now judged where it happens.** audio8-tts-realistic-0.6b leaves almost no
  edge silence (0.05 s) but pauses for 20.7% of its trimmed utterances, against 8.8% for
  real speech (`en`, 40 utterances). Untrimmed, its 21.5% total silence matched the
  reference's mostly edge silence (26%), scoring 0.16 on its supported languages. Trimmed
  it scores 0.42.
- **Indices**, refit on one snapshot with only the spans swapped (72 leaves with no
  pre-trim score left out of both fits). The capability index keeps its ranking
  (Spearman +0.96 untrimmed vs trimmed): MPM is one of four metrics in one of its four
  dimensions. Its scale stretches, though. ZONOS2 accurate, the 50 anchor, moves toward
  real speech (span 0.555 → 0.162), so every system's distance from 50 grows; partial-
  coverage systems far from 50 move most. The prosody index changes character: TTS
  systems spread from 47.7–51.3 to 27.9–59.4, and the ranking is new (Spearman +0.26).
  Before the fix, MPM pulled against the other three aspects and the fit cancelled to a
  near tie; trimmed, it agrees with them.

Published MPM spans are **not comparable across this change**. The previous TSVs are kept
at `results/_mpm_shuffle_floor/pre_trim_20261031/` and
`results/_mpm_ground_truth/pre_trim_20261031/`.

---

## C7 — Content errors scored as pace · **ADOPTED (char/s and paired duration re-scored)**

**Question.** After C6, every other prosody metric was audited for the same kind of
pitfall. Char/s divides the *target* text length by the trimmed duration, and paired
duration compares against a real recording of that text, so a content error moves both:
dropped words read as fast speech, a runaway or hallucinated tail as slow speech. The
40 char/s truncation cap only catches the extreme case.

**Measurement.** Char/s ×ceil per system (Clean `en de fr ja`, 300 utterances each) on
every utterance vs only the utterances the ASR heard with WER ≤ 20%. Most systems move
by ≤ 0.3×, but where a few utterances failed, those few dominate the W2 tail:

| System, language | all utterances | WER ≤ 20% only | kept |
|---|--:|--:|--:|
| cosyvoice3, ja | 17.9× | 5.6× | 86% |
| zonos2_accurate, ja | 13.2× | 6.4× | 93% |
| zonos2_expressive, de | 13.1× | 5.2× | 90% |
| moss-tts-v1.5, ja | 9.1× | 3.0× | 92% |
| qwen3-tts-0.6b-reftext, fr | 8.5× | 4.6× | 96% |

**Design/adoption.** `scripts/speaking_rate_dist.py` scores char/s only on utterances
whose WER in the leaf's `wav_res_ref_text.wer.raw` is ≤ `MAX_WER` = 0.2 (`heard_files`);
a leaf with no transcription gets no char/s, and `heard_pct` reports the share kept.
`scripts/paired_itw.py` applies the same rule to the duration columns only (`n_dur`
counts the pairs behind them; below 50 they are empty). Pauses and pitch range keep every
pair. Intelligibility stays in WER and tail rate, where it is already scored. The real
reference speech is not filtered.

**Result.** All 55 char/s panels and all 20 paired ITW languages re-scored.

- **Char/s rankings mostly hold.** Per-language Spearman between old and new ×ceil has a
  median of +0.89 on Clean (lowest +0.61) and +0.95 on In-The-Wild (lowest +0.49); 51% of
  Clean and 71% of ITW leaves move by ≤ 0.3×.
- **The large moves are content failures leaving the pace score.** omnivoice hr
  35.5× → 20.0× (10% of its utterances heard), zonos2_expressive hr 14.1× → 2.3×,
  qwen3-tts-1.7b-reftext hi 18.8× → 6.6×. On In-The-Wild, zonos3-mini_reference zh
  17.0× → 2.4× and ja 10.1× → 1.2×: its char/s distance there was almost all the
  18% of utterances it got wrong.
- **Systems that fail a language outright now have no pace score in it** instead of an
  extreme one: fewer than 100 heard utterances drops the leaf (Clean: glm-tts 21
  languages, cosyvoice3 21, voxtral 19; paired duration likewise, e.g. cosyvoice3 in
  9 ITW languages). Their failures stay in WER and tail rate.
- **Paired duration** keeps its ranking (Spearman 0.88–1.00 per language); the biggest
  change is qwen3-tts-1.7b-reftext hi, 2.25 → 0.31 |log2|. Pause and pitch-range columns
  are unchanged by construction.
- **Prosody index**, refit on one snapshot with only these inputs swapped: Spearman
  +0.999; content-failing systems stop paying twice (glm-tts +5.7, kokoro +5.6, voxtral
  +3.4, zonos2_expressive +2.7). The capability index reads neither metric.

The previous outputs are kept at `results/_prosody_pre_c7_20261101/` (paired CSVs and the
speaking-rate payload).

---

## C8 — Silence detection under background noise · **ADOPTED for paired pauses**

**Question.** Char/s, rhythm, pitch range and the paired ITW metrics find silence with
`librosa.effects.trim` / `.split(top_db=35)`, relative to each file's loudest frame.
Does that fail the way MPM's padding did (C6)?

**Measurement.** Perturbation probes on clean real speech (FLEURS-R `en`, 200 vs 200
utterances, shift as a multiple of the split-half null):

- Edge padding is harmless: 1 s of digital silence, −45 dB hiss or a −20 dB gain all
  probe at ≈1.0× for char/s, rhythm, pauses and pitch range, so C6 does not recur.
- A noise floor within 35 dB of the peak defeats the rule. 1 s noise edges at −30 dB
  are never trimmed: char/s 9.3×, rhythm pace 7.7×, rhythm pause factor 14.9×. Noise
  under the speech alone leaves char/s and rhythm at ≈1× but removes every pause the
  split finds (8.7×), and pitch range (pYIN) reads 0.4–0.6 st flatter even at −45 dB.

How often that happens, against Silero VAD on 40 files of every leaf: the rule misses
≥ 1 pause in 4.8% of Clean system files (concentrated in voxtral, which loses 3.2 of its
7.4 pauses/utt, glm-tts, cosyvoice3, vibevoice-rt) and in 16.9% of In-The-Wild system
files, and it finds only about a third of real VoxBlink2 speech's pauses: 67% of those
recordings have no frame 35 dB below their peak. Edges matter less in practice — the
speech detector finds only 0.07–0.12 s of edge silence in real VoxBlink2 recordings.
Across systems, every In-The-Wild prosody metric correlated with output noise (paired
pause error vs the share of files with no detectable silence: −0.56 median Spearman,
negative in every language).

**Design/adoption.** The paired pause count, where the misses land directly in the
score, now comes from Silero VAD (`utils.benchmark.SILERO_VAD`, v6.2.3 pinned by commit)
at its default thresholds, on the peak-normalized file: a pause is a gap ≥ 0.2 s between
two speech segments, on the generation and the real recording alike
(`scripts/paired_itw.py`). Char/s, rhythm and pitch range keep the energy rule; their
In-The-Wild figures, and voxtral's everywhere, carry this caveat.

**Validation** (same 120 FLEURS-R `en` utterances; pauses/utt):

| | clean | +2 inserted 300 ms pauses | −30 dB noise floor | both |
|---|--:|--:|--:|--:|
| 35 dB rule (before) | 0.97 | 1.39 | 0.00 | 0.00 |
| Silero, as adopted | 0.62 | 2.17 | 0.93 | 1.77 |

The energy rule catches only a fifth of inserted pauses even on clean audio (its 128 ms
frames shave each gap, so its effective minimum is ~0.33 s, not 0.2 s) and none under
noise. Silero catches 78% clean and 42% under noise, and peak normalization makes it
gain-invariant (a −20 dB gain moved the raw count 0.62 → 0.86, the normalized one not
at all). It is not noise-invariant: a noise floor inflates its count on natural speech
(0.62 → 0.84 at −45 dB, 0.97 at −25 dB). A stricter drop-out threshold (0.05) removes
that inflation but finds only 0.09–0.17 pauses per real VoxBlink2 recording; on real
data the two settings rank systems alike (Spearman 0.91–0.97 in `en de es ja`) and leave
the same residual link to output noise (−0.25 to −0.39, vs −0.36 to −0.61 for the energy
rule), so the defaults stay.

**Result.** All 20 VoxBlink2 languages re-scored; the duration and pitch-range columns
are byte-identical. Paired pause error keeps a moderate share of its ranking (median
per-language Spearman +0.79, range +0.65…+0.89), and its link to output noise drops
from −0.56 to −0.32 (median over languages). The largest correction is
omni_voxtral-tts, whose output noise had hidden its pauses: rank 2 → 31 of 31 in `zh`,
4 → 32 of 33 in `en`, 11 → 31 in `ja`, 21 → 30 in `ko`, 22 → 29 in `ru`. The previous
CSVs are kept at `results/_prosody_pre_c8_20261101/`. The prosody index, refit on one
snapshot with only the pause values swapped, barely moves (Spearman +0.999, at most 0.9
points): paired pauses are one In-The-Wild item of the pauses aspect.

---

## Open items that need humans

Two review findings cannot be closed by compute:

1. **The in-the-wild GT floor is not human-verified.** ITW reference transcripts are
   Voxtral-Mini-3B output on segments pre-filtered for cross-ASR agreement ≥ 0.85, so
   "GT WER" is inter-ASR disagreement on ASR-legible speech. A human-verified
   transcript subsample would calibrate what the floor actually means.
2. **No shipped metric is perceptually anchored.** The prosody metrics are validated
   against constructed ground truth (the right choice given that cross-system listener
   correlation over ~12 systems is underpowered — the lesson recorded in
   `docs/rhythm.md` and `scripts/validate_rhythm_metric.py`), but WER, speaker
   similarity and MSR-UTMOS — the three headline columns — have no listener validation
   at all. A small per-language listener probe would give one of them an anchor.
