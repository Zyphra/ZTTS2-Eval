# Unscoreable generations: from silent absence to maximum error (2026-09-01)

Closes the one recommendation the 2026-08 methodology campaign deliberately left
open (`docs/controls_2026-08.md`, end of C5): *the most defensible treatment of an
unscoreable degenerate generation is to score it as maximum error rather than
excluding it, but that changes a core metric's semantics and has to be applied
symmetrically, so it is the maintainer's call.* The call was made; this is the record.

## The defect

`utils/run_wer.py` wrote one `.raw` row per utterance it managed to transcribe.
An utterance it could not transcribe was recorded as UNSCORED in the sidecar and
left out of the average. So every leaf mean was a **survivor mean**: the more of a
system's output was untranscribable, the less of that output counted against the
system. It is the same survivor bias the scorer already had a comment warning about
for missing wavs, applied one level further in.

A second, independent defect sat underneath it. `utils/average_wer.py` recovered
`n_expected` from the `.meta` sidecar, and where no sidecar existed it fell back to
`n_expected = n_scored` — i.e. it *assumed the run had been complete*. Only 447 of
1557 leaves carry a sidecar. The other 1110 reported full coverage no matter how
much of the leaf had actually been scored, which is how
`moss-tts-v1.5/in-the-wild/ru` came to publish a `N_SCORED == N_EXPECTED`
footer over **169 of its 1500 utterances**.

## What the audit found before anything was changed

Across all 1557 published leaves: **4197 unscored utterances in 135 leaves**
(0.290 % of 1,448,263). Two facts shaped the fix.

**Every one of the 4197 had its generated wav present on disk.** Not a single one
was a missing generation. So the failure was always at the scoring step, never at
the synthesis step, and "impute maximum error because the system produced nothing"
would have been the wrong story.

**The ground truth is almost untouched.** GT carries 2 unscored utterances in total
(one `gt_clean/fa`, one `gt_in-the-wild/hi`). Whatever this change does,
it does not move the floor that every Δ is measured against — it moves systems.

## Why the fix re-measures instead of imputing

The tempting version of this change is to mark all 4197 as maximum error and
rebuild. Simulated, that shifts `moss-tts-v1.5`'s VoxBlink macro from 4.35 to
13.55. It would also have been **wrong**, and the reason is the point of this
document.

The unscored utterances turn out to come from two populations that look identical
in the artifacts and are opposite in meaning:

* **Degenerate audio.** The transcriber runs away on the generation and the request
  times out. C5 established this for `qwen3-tts-*-reftext` on Hindi, where raising
  the timeout 5× moved coverage 0.973 → 0.971. Excluding these is survivor bias, and
  maximum error is the honest score.
* **A scoring run that failed.** The audio is ordinary and transcribes on the very
  next attempt; the original pass lost it to a wedged server, a pre-`ASR_RETRIES`
  no-retry policy, or an interrupted job. Imputing maximum error here would invent a
  result, and a much worse one than the truth.

Sidecar reasons cannot separate them — a timeout appears in both — so the backfill
settles it by measurement. `scripts/rescore_residuals.sh` re-runs each leaf's
**residual** (its unscored utterances, and only those) under today's uniform policy:
current retries, the loop guard, and the per-language routing the published scores
use. Whatever transcribes gets a real score. Only what still fails is imputed, by
`run_wer.py` itself. Leaves scored before `ASR_RETRIES` existed therefore stop being
penalised for having had no retries, and the exclusion policy becomes uniform across
leaves without the full-benchmark rescore the open recommendation assumed.

## The new semantics

An utterance whose generated audio the reference ASR cannot transcribe is scored
`wer = 1.0` (WER_CLIP 100), with every reference token counted deleted.

* Imputed rows are ordinary `.raw` rows, so every consumer — the aggregate, the
  per-uid matrices, the coreset panels, the plots — sees them, and there is no
  second code path that can disagree with the first.
* Their hypothesis column carries an `<UNSCOREABLE:Reason>` sentinel instead of a
  transcript, so nothing can mistake one for a transcription.
  `scripts/classify_edits.py` skips them: the edit decomposition is synthetic and
  classifying it would invent an error category.
* `average_wer.py` reports `N_TRANSCRIBED:` and `N_IMPUTED:` beside
  `N_SCORED:`/`N_EXPECTED:`, and warns above a 2 % imputed share.
  `SCORE_MAX_IMPUTED` turns that into a gate for callers that want one.
  `systems_<set>.csv` gains `n_imputed` / `imputed_langs`.
* **Only failures attributable to the audio are imputed** — timeout, rejected
  audio, missing wav. Infrastructure failures (dead engine, refused connection) stay
  unscored and keep failing the coverage gate. Imputing one would publish a harness
  outage as a model result.
* `WER_IMPUTE=0` restores pure exclusion for reproducing a pre-2026-09-01 number.

Note the deliberate consequence: a leaf that is complete but heavily imputed now
*passes* the coverage gate. That is the intended reading — imputation converts an
untrustworthy partial leaf into a trustworthy bad one — which is exactly why the
imputed share is reported separately everywhere the mean is.

## Result

135 leaves backfilled, no failures, and no leaf anywhere in the tree still has an
unscored utterance. Of the 4197:

* **1818 (43 %) transcribed on re-measurement** — those were failed scoring runs,
  and their published numbers were simply computed over the wrong denominator.
* **2379 (57 %) stayed unscoreable** and are now scored as maximum error — those
  were genuine survivor bias.

Both populations are large. Marking all 4197 as maximum error would have been as
wrong as continuing to drop them, which is the whole argument for spending the GPU
time on re-measurement.

They separate cleanly by system, and the split matches what is independently known
about each one:

| leaf | published | corrected | re-transcribed | unscoreable |
|---|--:|--:|--:|--:|
| `qwen3-tts-0.6b-reftext` itw `hi` | 57.29 | **63.61** | 10 | 212 |
| `qwen3-tts-1.7b-reftext` itw `hi` | 61.84 | **67.25** | 9 | 204 |
| `qwen3-tts-1.7b-reftext` clean `hi` | 25.25 | **29.36** | 7 | 62 |
| `voxtral-tts` clean `ko` | 93.94 | **94.86** | 17 | 187 |
| `moss-tts-v1.5` itw `ru` | 6.77 | **6.28** | 1331 | 0 |
| `moss-tts-local` clean `ja` | 10.38 | **10.64** | 246 | 0 |

The imputations concentrate in `voxtral-tts` (1173), the three `-reftext`
conditioning arms (656), `glm-tts` (240) and `moss-tts-local-v1.5` (87) — i.e. in
the systems whose output was already scoring near the ceiling. The two moss leaves
that account for most of the re-transcriptions are ordinary audio that had never
been scored at all.

**Scale of the correction.** 32 leaves move by ≥ 0.5 WER_CLIP, the largest by ~6
points. Published *macro* means move much less, because a macro averages over 20–35
languages: the biggest is `qwen3-tts-0.6b-reftext` on VoxBlink at 17.80 → 18.25
(+0.45), then `qwen3-tts-1.7b-reftext` +0.43, and everything else is ≤ 0.20. No
flagship ranking changes. The correction is real but it lands on per-language cells
and on the honesty of the coverage footers, not on the headline order.

**The floor holds.** GT's two utterances moved `gt_in-the-wild/hi` 12.46 →
12.52 and `gt_clean/fa` 29.20 → 29.24. Every Δ-vs-GT in the tables is therefore
a movement of the system, not of the reference.

## Comparability

WER numbers published before 2026-09-01 are not comparable to those after it on any
affected leaf, for two reasons that pull in different directions: previously-lost
utterances now carry real scores (usually raising a mean, occasionally lowering it),
and genuinely unscoreable ones now carry 100 instead of nothing. Leaves with no gap
are unchanged — re-aggregation is a pure function of the `.raw` and reproduces the
old file byte for byte apart from the two added footer lines.
