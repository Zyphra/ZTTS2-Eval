# ZTTS2 — VoxBlink2 in-the-wild sweep (20 langs × 11 systems)

Zero-shot VoxBlink2 (noisy, in-the-wild reference prompts), 20 langs:
`ar de en es fr hi id it ja ko nl pl pt ru th tl tr uk vi zh`.
Metric = **WER_CLIP** vs **Δ above the GT floor** (GT restricted to the languages each
system covers). Scored with per-language forced-ASR
routing (`ASR_ROUTING=1`), matching the GT. GT floor mean = 9.3 (much higher than the
clean FLEURS-R floor of ~4–12 because the reference prompts are real-world noisy).

## Ranking — 20 langs (Δ WER above GT floor)

| System | Δ vs GT | ±CI95 | mean WER_CLIP | mean spk×100 | mean MSR |
|---|--:|--:|--:|--:|--:|
| **zonos2-accurate** | **−4.6** | ±0.28 | 4.7 | 58 | 2.79 |
| **omnivoice-rt** | **−4.2** | ±0.26 | 5.1 | 75 | 2.82 |
| **voxcpm2** | −4.0 | ±0.26 | 5.4 | 74 | 2.31 |
| **fish-s2-pro** | −3.3 | ±0.39 | 6.1 | 70 | 2.64 |
| moss-tts | −0.0 | ±0.39 | 9.3 | 73 | 2.54 |
| zonos2-expressive | +0.3 | ±0.45 | 9.6 | 57 | 2.46 |
| qwen3-tts-1.7b | +2.4 | ±0.51 | 11.7 | 67.2 | 2.85 |
| higgs-v3 | +3.4 | ±0.37 | 12.8 | 69 | 2.34 |
| qwen3-tts-0.6b | +5.8 | ±0.55 | 15.2 | 64.3 | 2.86 |
| cosyvoice3 | +33.6 | ±0.43 | 42.9 | 73 | 2.77 |
| voxtral | +34.0 | ±0.37 | 43.3 | 14 | 3.39 |

±CI95 is the half-width of the **paired** bootstrap CI on Δ vs GT (per-item differences,
items resampled within language, languages fixed strata; 10,000 draws, seeded —
`scripts/paired_ci.py --set itw --vs-gt <system>`, also printed by
`build_matrix.py --set itw --verify --paired-ci`). Rank order is resolved by
**adjacent-pair** paired ΔCIs, not by the vs-GT intervals: adjacent systems whose paired
ΔCI spans 0 are statistically unresolved on this basket. Three adjacent pairs are
unresolved here — **omnivoice-rt ↔ voxcpm2** (Δ −0.25, CI95 [−0.51, +0.01]),
**moss-tts ↔ zonos2-expressive** (Δ −0.28, CI95 [−0.81, +0.24]) and
**cosyvoice3 ↔ voxtral** (Δ −0.43, CI95 [−0.90, +0.02]); every other adjacent gap
excludes 0. moss-tts (−0.01, CI95 [−0.39, +0.39]) and zonos2-expressive (+0.28, CI95
[−0.18, +0.75]) are also statistically indistinguishable from the GT floor itself.

The first pair is a correction, not a change: neither omnivoice-rt nor voxcpm2 has a
leaf touched by any rescore since, so its interval is computed on identical data and
sits exactly on the boundary (upper bound +0.01). It was previously described here as
resolved. Treat these two as tied.
A negative Δ means the synthesis is *more ASR-legible than real speech* — an
ASR-optimization axis, not extra quality (the WER leaders sit in the rhythm-flattened
cluster, see docs/rhythm.md); the ranking below the floor is kept for continuity but
should not be read as "better".

## Note — unscoreable generations now score as maximum error (2026-09-01)

A generation the reference ASR cannot transcribe used to be left out of its own leaf
mean, so the more of a system's output was untranscribable, the less of it counted
against that system. It is now scored as maximum error (WER_CLIP 100). Before
switching, every leaf's unscored utterances were **re-measured** rather than assumed:
tree-wide, 43 % of them transcribed fine on the retry (those were failed scoring runs,
not degenerate audio) and only the remaining 57 % are imputed. Full record in
`docs/unscoreable_2026-09.md`.

On this basket the effect is small — `voxtral` 43.2 → 43.3 (Δ +33.9 → +34.0),
`zonos2-expressive` Δ +0.2 → +0.3, `qwen3-tts-0.6b` 15.1 → 15.2, everything else
unchanged, and the GT floor still 9.34 — because a macro averages 20 languages. The
per-language cells behind it move much more: the two `qwen3-tts-*-reftext`
conditioning arms shift by 5–6 WER points on `hi`, which is where their runaway
generation actually lives.

## Note — English is now scored by Whisper (2026-08-30)

A scorer-swap control (`docs/controls_2026-08.md`, C1) found that Qwen3-ASR — the WER
scorer for `en` until now — measurably favoured its own model family: rescoring the same
audio with Whisper-large-v3 moved both qwen3-tts systems *relatively worse* than a
four-system cohort (difference-in-differences +0.51 CI95 [+0.28, +0.74] for 1.7b and
+0.70 [+0.46, +0.96] for 0.6b WER_CLIP points). `zh` showed no such effect and stays on
Qwen3-ASR. English now routes to forced-language Whisper for **every** system and both
ground-truth baselines.

The English GT floor moves in *opposite directions* on the two sets, which is consistent
with `docs/asr_whisper_vs_qwen.md`: Qwen is the better scorer on clean read speech, Whisper
on noisy in-the-wild audio. Here, on VoxBlink2, the English GT floor **falls 4.69 → 3.85**,
pulling the 20-language floor from 9.38 to 9.34 and shifting every Δ by about 0.04. (On the
clean FLEURS-R set the same change raises the English floor, 3.78 → 4.45.) Either way the
floor shift is common-mode and cancels between systems; the family bias it replaces was
differential and did not.

## Note — `yo` removed from the benchmark (2026-08-29)

The former 21st language, `yo` (Yoruba, 148 prompts), was **removed outright**: its
transcripts are fluent **Nigerian English**, not Yoruba (135/148 lines carry common
English function words, none carry Yoruba ones). The language labels come from
Whisper LID at prob ≥ 0.85 on noisy YouTube audio, which systematically tags
Nigerian-accented English as `yo`; no gate ever checked the *transcript's* language.
TTS systems were being told `lang=yo` and handed English text, so the row measured
language-tag robustness, not Yoruba synthesis. The enrichment join now carries a
transcript-language gate to prevent recurrence, and the removed set is archived
outside the repo. Every aggregate above is therefore a **20-language basket and not
comparable to previously published 21-language numbers** (the removal of an
easy-English language shifts each system's mean WER_CLIP up by ~0.1–1.9 and the GT
floor from 9.20 to 9.38; the English scorer change above then moved that floor to 9.34).

## Note — qwen3 coverage (updated 2026-08-28)

As of 2026-08-03 qwen3-tts-1.7b/0.6b covered only 19 of the then-21 in-the-wild
languages (`ar` and `hi` had produced no scored output), and their spk/MSR means
were additionally mis-divided by 21 rather than 19. Both `ar` and `hi` have since
been generated and fully scored, so the rows above cover the full basket. These rows
also include the 2026-08 audit's symmetric zh trad→simplified rescore (all systems'
`zh` WER moved by about −0.1).

Δ is computed against the GT floor **restricted to the languages each system covers** —
charging a system the floor of a language it never ran would flatter or penalize it for
coverage rather than quality. The table is reproduced from the per-utterance data by
`scripts/coreset/build_matrix.py --set itw --verify`.

## Findings (and contrast with clean FLEURS-R)

1. **Top TTS beats the noisy human floor.** zonos2-accurate, omnivoice-rt, voxcpm2,
   fish-s2-pro all score *below* the GT WER (negative Δ) — the synthesized speech is
   cleaner/more ASR-legible than the real-world noisy reference clips. Expected for an
   in-the-wild set; "below floor" measures ASR legibility, not extra quality (see the
   caption above the table).
2. **qwen3 looks far better here than on FLEURS-R** (+2.4/+5.8 vs +20.2/+28.0 on the
   26-lang tail). Not a contradiction — VoxBlink's 20 langs are mostly ones qwen3
   supports (major + common langs), whereas the FLEURS tail included many it can't do
   (el/fa/he/fi/hu…). It's a language-set difference, not a quality reversal.
3. **cosyvoice3 and voxtral still collapse** (+34) — voxtral doesn't clone (spk×100 14);
   cosyvoice3 fails on unsupported langs (ar/hi/th/vi + tl).
4. **Speaker-sim is lower across the board** (~57–75 vs ~67–88 on clean) — cloning noisy
   in-the-wild reference prompts is harder.

Notes: reference prompts fold with earlier VoxBlink debugging (fish-s2-pro generated
fine this run, no 0-wavs bug). cosyvoice3/moss-tts each survived one node OOM-kill
mid-run and were resumed (skip-existing); zonos2-expressive's server was killed by the
same node event and was resubmitted on another node.
