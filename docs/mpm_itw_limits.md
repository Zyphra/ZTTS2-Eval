# Why MPM span is FLEURS-R only

MPM span is computed for `clean` (FLEURS-R) and **not** for `in-the-wild`
(VoxBlink2). That exclusion is deliberate and gated in three places:

| file | gate |
|---|---|
| `slurm/submit_metrics.sh` | only `clean` rows enter `mpm.tsv` |
| `slurm/submit_metrics.sh` (`mpm_gt`) | same gate on the GT 50/50 split (`mpm_gt.tsv`) |
| `scripts/results_doc/build_results.py` | `mpm_model` is set only for `clean` |

This page records the measurements behind it, because "we didn't run it" and "it does
not work there" are very different claims and the code alone cannot tell them apart.

Everything below is `en`, and reproducible with the four scripts named at the end.

## The metric depends entirely on two anchors

```
span = (d_ref - d_ref_floor) / (d_noise_floor - d_ref_floor)
```

* `d_ref_floor` — **0** — Fréchet distance between two disjoint halves of the *same*
  real corpus. The finite-sample floor; a perfect system cannot beat it.
* `d_noise_floor` — **1** — the corpus against its own contour-scrambled self
  (`shuffle_joint`: a per-utterance frame permutation applied jointly to the
  pitch/energy/VAD channels, which destroys the temporal contour while holding every
  marginal fixed).

Both are derived from the reference corpus, so they are only as good as that corpus's
internal consistency. Measured:

| corpus | `d_ref_floor` | `d_noise_floor` | usable range |
|---|--:|--:|--:|
| FLEURS-R en | 0.187 | 18.867 | **18.68** |
| VoxBlink2 en | 0.287 | 7.919 | **7.63** |

Two things break at once in the wild. The scramble anchor **collapses 2.4×** — noisy,
spontaneous, variable-channel speech already has disordered contours, so scrambling
moves it much less far. And the real floor **rises 1.5×** — two halves of VoxBlink real
speech are less like each other than two halves of FLEURS-R.

Squeezed from both sides the denominator shrinks **2.45×**, so identical prosody error
reads ~2.5× larger on VoxBlink:

| true `d_ref` | span on FLEURS-R | span on VoxBlink2 |
|--:|--:|--:|
| 1.00 | 0.045 | 0.102 |
| 3.19 | 0.162 | 0.393 |
| 5.00 | 0.259 | 0.634 |

That destroys the property the explorer relies on. MPM span is documented as
comparable *across* languages because each language is normalized to its own anchors,
which is why the prosody-overlap view plots all 35 on one shared axis. A corpus whose
anchors are 2.5× tighter cannot share that axis: a VoxBlink `zh` row at 0.39 next to a
FLEURS `zh` row at 0.16 looks like a regression when it is an anchor artifact.

## The inputs have no robustness mechanism

MPM never sees audio. It sees three token streams from TTSDS2's measures, all
classical DSP with no learned front end:

| channel | source | notes |
|---|---|---|
| pitch | `pw.dio` + `pw.stonemask` (WORLD) | unvoiced → NaN → **linearly interpolated** |
| vad | the *same* dio/stonemask call, thresholded | F0 present = 1. **Not a VAD model** — voicing, not voice activity |
| energy | `librosa.feature.rms`, frame 1024 / hop 256 | absolute `0–0.2` window, **no normalization** |

Degrading clean FLEURS-R one condition at a time (frame-aligned against the clean
control, so errors are measured per frame rather than distributionally) gives the
apparent prosody error each channel condition manufactures **by itself**:

| condition | pitch gross >50¢ | VAD F1 | span produced |
|---|--:|--:|--:|
| mp3 64 kbps | 1.4% | 98.4% | 0.005 |
| clip at 0.3 peak | 0.3% | 99.4% | 0.054 |
| reverb RT60 0.5 s | 47.0% | 75.7% | 0.115 |
| **gain −5.5 dB** | **0.0%** | **100%** | **0.426** |
| noise SNR 10 dB | 5.6% | 93.2% | 0.779 |
| wideband 80–8 k | 4.1% | 77.4% | 1.427 |
| **telephone band** | **97.0%** (median **1180 ¢**) | 55.0% | **8.77** |

For scale, real systems on FLEURS-R span roughly 0.16–1.0. **Channel effects are the
same size as or larger than the prosody signal.**

Three results worth keeping in mind:

* **A pure level change is the third-largest effect.** `gain −5.5 dB` is the *measured*
  median RMS gap between the two corpora (FLEURS-R 0.1182, VoxBlink2 0.0626). Pitch and
  VAD come out bit-identical and the energy *correlation* is r = 1.000, yet 79% of
  energy tokens change and span moves 0.43 — purely because the energy window is
  absolute. MPM cannot distinguish "quieter mic" from "flatter prosody".
* **Band-limiting is catastrophic.** Removing <300 Hz strips the fundamental, DIO locks
  onto a harmonic, and every pitch token saturates at the overflow index 128 (IQR 0.0 —
  the channel collapses to a constant). VAD accuracy 44.6% is *below chance*.
* **Noise-vs-span is non-monotonic.** 20 dB SNR (span 1.31) scores worse than 5 dB
  (0.72), because noise fills the silence floor: clean audio has 19.2% of frames at
  energy token ≤1, every noise condition has 0.0%, with the cliff between 40 and 30 dB
  SNR. MPM is most sensitive at the *mild, realistic* noise levels in-the-wild audio has.

## Four attempted fixes, none sufficient

| intervention | ITW compression | outcome |
|---|--:|---|
| shipped (DIO + absolute energy) | **2.45×** | baseline |
| SwiftF0 F0 backend | 2.07× | −16% of the gap, but −7.2% clean range and a reordered board |
| per-utterance energy norm, linear (`p95`) | 2.66× | no ITW effect |
| per-utterance energy norm, log (`db60`) | 2.10× | −18% clean ceiling |
| per-condition (MSR-UTMOS bin) anchors | **3.01×** | **worse** |

**SwiftF0** ([lars76/swift-f0](https://github.com/lars76/swift-f0), arXiv:2508.18440) is
genuinely the more robust detector — 11× better under mild noise, 1.8× under reverb,
and wideband voicing F1 77% → 97%. But it is **26× less precise on clean audio**
(0.3 vs 7.9 median cents), its headline telephone-band win is *abstention* rather than
tracking (it voices 3.9% of frames, where DIO voices 70%), and on clean speech the two
backends disagree on **one frame in five** with 6.4 points less voicing. Swapping it
would redefine every published MPM number and reorder the board (Kendall τ +0.867,
max shift 0.45 span) for an ITW gain that leaves the metric unusable anyway.

**Per-condition anchors** were the most promising hypothesis: maybe the collapse is
heterogeneity, not something intrinsic — one global anchor pools 2000 speakers over
thousands of channels, inflating the reference variance. Per-*speaker* anchors are
impossible here (VoxBlink2 en has 2000 speakers with a **median of 2 utterances each**),
so the test stratified by the reference prompt's MSR-UTMOS bin
(`data/in-the-wild/_msr_utmos_cache.tsv`), the repo's channel-quality proxy — with FLEURS-R random subsets at matched
`n` as the control:

| bin | n | VB floor | VB range | FL floor | FL range | ratio |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 109 | 4.319 | 3.08 | 2.305 | 16.79 | 5.45× |
| 1 | 389 | 1.930 | 5.58 | 0.465 | 18.63 | 3.34× |
| 2 | 387 | 0.872 | 7.50 | 0.429 | 18.87 | 2.52× |
| 3 | 388 | 1.142 | 7.52 | 1.011 | 17.74 | 2.36× |
| 4 | 227 | 2.706 | 6.20 | 0.935 | 18.01 | 2.90× |

It made things **worse** (2.45× → 3.01×), and the control proves it is not a
sample-size artifact: FLEURS-R holds ~18 range at every matched `n`. The mechanism is
in the floors — the VoxBlink ceiling barely moved (7.92 → 7.40–8.91) while the floor
rose **3–15×** (0.287 → 0.87–4.32). Stratification did not remove channel variance; it
removed the sample size that was averaging it out. Even the cleanest usable stratum is
2.36× worse than its own control.

That is the opposite of the hypothesis: the heterogeneity is not pooled *across* the
corpus, it is *inside each utterance*. VoxBlink prompts are ~3 s segments from single
YouTube videos, each carrying its own compression, room and microphone, so no grouping
recovers a homogeneous reference.

**Conclusion.** MPM span on VoxBlink2 is not fixable by front-end or anchor
engineering — at in-the-wild scale the channel *is* the signal. Keep the exclusion.
Reviving it would need a channel-invariant prosody representation, not a better pitch
tracker or a cleverer normalizer.

## Known open issue on FLEURS-R: a loudness confound

Independent of the above, and live on the current board. Because the energy window is
absolute, span inflation tracks how *quietly* a system renders:

| system | dB vs reference | span (shipped `abs`) | span (`p95`) | inflation |
|---|--:|--:|--:|--:|
| kokoro | **−7.4** | **1.617** | 0.850 | +0.768 |
| zonos2_accurate | −3.7 | 0.834 | 0.577 | +0.257 |
| neutts-2e | −2.0 | 0.616 | 0.426 | +0.190 |
| moss-tts | −1.3 | 0.656 | 0.526 | +0.129 |
| omni_qwen3-tts-1.7b | −0.2 | 0.522 | 0.493 | +0.030 |
| omni_voxcpm2 | +0.7 | 0.405 | 0.414 | −0.008 |

**Pearson r = −0.979**, slope −0.095 span/dB. `kokoro` reads 1.62 — above the
"fully contour-scrambled" ceiling — largely for being 7.4 dB quieter than the
reference. That is a normalization bug, not a prosody reading.

Per-utterance `p95` normalization (`e / p95(e)`, clipped to `[0,1]`; p95 rather than
max so one plosive cannot rescale the contour) is exactly level-invariant — 100% of
energy tokens survive any gain change, versus 21% today — and pulls every tested
system back inside `[0,1]` while preserving the ranking at Kendall τ +0.867.

**APPLIED 2026-08-29** (`score_mpm_shuffle_floor._tokens`, formerly `_tokens_for_wav`; anchor cache tag
`v3` → `v4` — the cache key hashes the reference dir's contents but *not* the
featurization code, so without the bump every anchor would have been silently reused
and the new spans would have been meaningless). Upstream TTSDS2 is untouched: this repo
owns the tokenization, so bit-identity for anyone calling `ttsds` directly is preserved.

The full 35-language rebuild confirmed the projection: rank preservation came out at
τ **+0.863** mean (range +0.545…+0.970) against the +0.867 predicted here, and mean span
fell 0.569 → 0.495. kokoro returned under the scramble ceiling on `es` and `pt` but
stays above it on `fr`/`hi`/`it`/`ja`/`zh` — the languages it has no voice for — so the
residual anomaly is a real prosody failure the gain artifact had been sitting on top of.
Pre-fix TSVs are archived at `results/_mpm_shuffle_floor/pre_p95_20260829/`; see
`docs/controls_2026-08.md`. Published MPM numbers are not comparable across this change.
Since 2026-10-31 leading/trailing silence is also trimmed before featurization (cache tag
`v5`; [controls C6](controls_2026-08.md)), which changes every span again.

Anchor cache tag history:

- v3→v4: per-utterance `p95` energy normalization (above).
- v4→v5: 35 dB energy trim of leading/trailing silence ([controls C6](controls_2026-08.md)).
- v5→v6: Silero VAD span trim
