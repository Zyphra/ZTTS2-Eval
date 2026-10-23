# ZTTS2 — New-language sweep (26 langs × 11 systems)

Zero-shot FLEURS-R, languages beyond the base 9:
`ar bg ca cs da el fa fi fil he hi hr hu id ms nb nl pl pt ro sk sv th tr uk vi`.
Metric = **WER_CLIP** (per-utt WER capped at 100% before averaging). Because the
per-language ASR error floor varies a lot on genuine human speech (GT WER_CLIP
ranges pt 4.4 … fa 29.2), systems are ranked by **Δ above the GT floor**, not raw WER.
All numbers **fully** routing-corrected as of 2026-08-03 (see notes below — the original
2026-07-30 fix covered only 6 of the 28 languages that needed it). GT floor mean over 26
langs = 11.85.

## Final ranking — all 26 new langs (fully routing-corrected 2026-08-03)

**These numbers superseded the original sweep.** Two corrections landed on 2026-08-03:
a [dropped-language bug](#note--dropped-language-correction-2026-08-03) and, much larger,
[22 languages that were never ASR-routing-corrected](#note--the-routing-fix-was-only-6-of-28-languages-2026-08-03).
Regenerated from the per-utterance data by
`python scripts/coreset/build_matrix.py --set clean --verify`.

| System | Δ vs GT | ±CI95 | mean WER_CLIP | mean spk×100 | mean MSR | Δ before correction |
|---|--:|--:|--:|--:|--:|--:|
| **zonos2-accurate** | **−3.9** | ±0.15 | 7.98 | 76 | 3.09 | +1.9 |
| **omnivoice-rt** | **−2.7** | ±0.15 | 9.16 | 88 | 3.03 | +2.3 |
| **higgs-v3** | **−0.6** | ±0.17 | 11.27 | 80 | 2.97 | +6.0 |
| fish-s2-pro | +2.0 | ±0.16 | 13.80 | 81 | 3.15 | +6.6 |
| voxcpm2 | +5.3 | ±0.18 | 17.19 | 79 | 2.88 | +11.4 |
| moss-tts | +6.4 | ±0.17 | 18.29 | 84 | 3.05 | +10.4 |
| zonos2-expressive | +6.8 | ±0.29 | 18.69 | 67 | 2.96 | +12.4 |
| qwen3-tts-1.7b | +20.3 | ±0.21 | 32.15 | 68 | 3.52 | +28.5 |
| qwen3-tts-0.6b | +28.0 | ±0.22 | 39.83 | 67 | 3.53 | +37.9 |
| voxtral | +49.5 | ±0.23 | 61.32 | 16 | 3.32 | +58.3 |
| cosyvoice3 | +60.1 | ±0.19 | 71.94 | 73 | 3.26 | +61.6 |

GT floor mean over the 26 languages = **11.85** (the 2026-08-03 routing correction left
it unchanged — ground truth was always routed correctly; the 0.01 it moved on 2026-09-01
is one `fa` utterance recovered by the unscoreable-generation backfill).

±CI95 is the half-width of the **paired** bootstrap CI on Δ vs GT (per-item
differences, items resampled within language, languages fixed strata; 10,000 draws,
seeded — `scripts/paired_ci.py --set clean --vs-gt <system>`, also printed by
`build_matrix.py --set clean --verify --paired-ci`). Rank order is resolved by
**adjacent-pair** paired ΔCIs, not by the vs-GT intervals: adjacent systems whose paired
ΔCI spans 0 are statistically unresolved on this basket. On this 26-language basket
**every adjacent gap excludes 0** — the tightest is moss-tts ↔ zonos2-expressive
(Δ −0.40, CI95 [−0.68, −0.13]), so even the closest ranks are resolved (unlike the
in-the-wild set, where three adjacent pairs are not — see RESULTS_VOXBLINK.md).
A negative Δ means the synthesis is *more ASR-legible than real speech* — an
ASR-optimization axis, not extra quality (the WER leaders sit in the rhythm-flattened
cluster, see docs/rhythm.md); the ranking below the floor is kept for continuity but
should not be read as "better".

## Headline findings

1. **The top three now read *more cleanly than human speech* on the tail.**
   zonos2-accurate (−3.9), omnivoice-reftext (−2.7) and higgs-v3 (−0.6) all score below
   the ground-truth WER floor across 26 languages. Before the routing correction all
   three appeared *above* it. As on the in-the-wild set, "below floor" is expected —
   FLEURS-R reference speech is read by real people at natural pace, and the ASR floor is
   not zero.
2. **Base-language leaders still don't generalize.** qwen3-tts-1.7b/0.6b (top on the base
   9 en/zh) fall to +20/+28 on the tail — they emit wrong-language output on many langs.
   The gap narrowed with correct routing but the conclusion holds.
3. **No tail support.** voxtral stops cloning (spk×100 ≈ 16, WER 61); cosyvoice3 emits
   ~100%-WER garbage on unsupported langs (ar/el/he/hi/th/vi).
4. **One ordering flipped:** voxcpm2 (+5.3) now beats moss-tts (+6.4); before the
   correction moss-tts (+10.4) led voxcpm2 (+11.4). Every other rank is unchanged — the
   correction moved the whole field down by ~5–6 points rather than reshuffling it.

## Note — unscoreable generations now score as maximum error (2026-09-01)

A generation the reference ASR cannot transcribe used to be dropped from its leaf
mean, making every mean a survivor mean that flattered the systems with the most
degenerate output. It now scores as maximum error (WER_CLIP 100). The unscored
utterances were **re-measured** before the switch rather than assumed unscoreable:
43 % of them tree-wide transcribed fine on the retry, so only the rest are imputed.
See `docs/unscoreable_2026-09.md`.

On this 26-language basket the movements are small — `qwen3-tts-1.7b` 32.08 → 32.15
(Δ +20.2 → +20.3), `fish-s2-pro` 13.78 → 13.80 (Δ +1.9 → +2.0), `zonos2-expressive`
18.66 → 18.69, `qwen3-tts-0.6b` 39.81 → 39.83, `voxtral` 61.30 → 61.32 — and no rank
changes. The GT floor moves 11.84 → 11.85, from a single recovered `fa` utterance.

## Note — the routing fix was only 6 of 28 languages (2026-08-03)

The [2026-07-30 fix](#note--asr-routing-scoring-fix-2026-07-30) below re-scored six
languages. It should have re-scored twenty-eight.

`utils/run_wer.py` routes a language to Whisper-large-v3 with the language **forced** when
it is not in `QWEN_LANGS` (`en zh ko th vi hi fa yo`) and `ASR_ROUTING=1`. Ground truth was
scored that way from the start. The synthesized audio was not: only the six languages with
the most visible misrouting (`bg ca hr nb sk uk`) were re-scored, leaving **22 languages
where synth went through Qwen3-ASR auto-detect while GT went through forced-language
Whisper**. Every Δ on those languages was inflated by the full amount.

Measured inflation (WER_CLIP points, from re-scoring `zonos2_accurate`):

| | languages |
|---|---|
| large | `he` +53.1, `fi` +14.6, `fil` +14.5, `hu` +13.9, `ro` +11.7, `el` +10.1, `cs` +9.5, `sv` +8.9, `da` +5.5, `pl` +5.0, `tr` +4.6, `ms` +3.3 |
| mild (≤1.5, auto-detect copes) | `ar` `id` `fr` `it` `nl` `de` `ja` `ru` `es` `pt` |

Hebrew shows the mechanism most clearly:

| system | before | after | GT floor |
|---|--:|--:|--:|
| zonos2-accurate | 60.65 | **11.90** | 29.43 |
| higgs-v3 | 71.37 | **21.76** | 29.43 |
| omnivoice-rt | 73.59 | **33.43** | 29.43 |

The diagnosis is not inferred, it is checkable two ways:

1. Languages whose score changed by **exactly zero** on re-scoring are precisely
   `QWEN_LANGS` plus the six already fixed — which also proves the ASR scoring is
   deterministic, so the remaining differences are not run-to-run noise.
2. Stored `he` hypotheses were 90% Hebrew with Arabic and Cyrillic drift; ground truth's
   are 100% Hebrew, and the re-routed synth is 100% Hebrew. The old numbers were largely
   measuring Qwen3-ASR failing to identify the language.

Fixed by `scripts/rescore_routed_all.sh`, which shards the 22 languages across the GPUs of
one allocation (277 leaves re-scored, 0 failures, 1h51m on 8×H100). WER only — speaker
similarity and MSR-UTMOS are language-agnostic and were never affected, which is why their
columns above are unchanged.

The in-the-wild sweep (`RESULTS_VOXBLINK.md`) was run after the `eval_job.sh` routing patch
and is unaffected.

## Note — dropped-language correction (2026-08-03)

`utils/run_wer.py` does not escape the ASR hypothesis, so a transcription containing a
literal TAB or newline splits its own record in `wav_res_ref_text.wer.raw`.
`utils/average_wer.py` then dies on that line and leaves a **truncated `.wer` with no
`WER:` footer** — and anything reading the footer silently loses the whole leaf.

Four leaves repo-wide were affected; three of them are in this sweep, and each lost one
language from its average while the GT floor kept all 26:

| System | Language dropped | was | after this fix |
|---|---|--:|--:|
| omnivoice-rt | `he` (WER_CLIP 73.6) | +0.6 | +2.3 |
| zonos2-expressive | `he` (WER_CLIP 65.4) | +11.5 | +12.4 |
| voxtral | `th` (WER_CLIP 99.7) | +56.9 | +58.3 |

> The "after this fix" column is an **intermediate** state — it still carries the routing
> error described above, which was found later the same day. The published table at the
> top of this file has both corrections applied. This fix alone reordered the top two
> (zonos2-accurate ahead of omnivoice-rt); the routing fix then moved every row.

Records were recovered with `scripts/repair_wer_raw.py` (4 records reassembled, 0
unrecoverable) and the `.wer` files regenerated. Worth noting for its own sake: the
truncation makes a leaf invisible to anything that reads the `WER:` footer, including
`scripts/results_doc/build_results.py`, so it fails silently rather than loudly.

Two related caveats noted at the time of the repair:

- **`RESULTS_VOXBLINK.md` had a separate arithmetic bug** (fixed 2026-08-28):
  qwen3-tts-1.7b/0.6b cover only 19 of the 21 in-the-wild languages (`ar` and `hi` are
  missing), but their spk/MSR sums were divided by 21. The then-reported spk 60/58 and
  MSR 2.58 are now the corrected **66.8/63.8 and 2.85** in that file; the WER_CLIP and
  Δ columns were correct throughout.
- **Δ is computed against the GT floor restricted to the languages each system covers**
  (unchanged, and still the rule). A system missing a language must not be charged the
  floor of a language it never ran.

## Note — ASR-routing scoring fix (2026-07-30)

GT was scored with **per-language forced-ASR routing** (`ASR_ROUTING=1` → Whisper
large-v3 forced-language for langs not in QWEN_LANGS). The sweep's synth scoring in
`slurm/eval_job.sh` originally did **not** set `ASR_ROUTING`, so synth for 6 langs
(`bg ca hr nb sk uk`) was auto-detected by Qwen3-ASR and misrouted to a phonetic
neighbour (uk→Czech, bg→Russian, ca→Spanish, nb→Danish, sk→Czech, hr→Serbian),
inflating WER by 40–60 points. Fixed by re-scoring those leaves with matched routing
(`scripts/rescore_routed.sh`) and patching `eval_job.sh` (now sets `ASR_ROUTING=1`,
`WHISPER_GPU=0`, mem-utils 0.35 so ASR+Whisper co-locate on one GPU). Effect (median
WER over 11 systems):

| Lang | GT | before | after |
|---|--:|--:|--:|
| bg | 14.7 | 60.4 | 19.7 |
| ca | 7.5 | 51.6 | 10.4 |
| hr | 13.3 | 80.9 | 23.8 |
| nb | 12.0 | 58.6 | 12.3 |
| sk | 12.3 | 68.6 | 22.0 |
| uk | 6.6 | 56.9 | 12.4 |

Excluded: glm-tts (cut — timed out at 9/26, over-generating 2–3× too-long audio).
English/limited-lang preset models (neutts, vibevoice, kokoro) not in this sweep.
