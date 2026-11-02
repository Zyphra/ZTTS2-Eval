# ZTTS2-Eval — Current Results Snapshot

Benchmark version 2

Regenerated from the `results/` directory tree by
`scripts/results_doc/run.sh`. Numbers are read directly from
`wav_res_ref_text.wer`, `spk_simi_scores.txt`, `spk_simi_scores.xutt.txt` (ITW),
`msr_utmos_mean.txt` and the rhythm / pitch-range / paired-ITW CSVs. **WER_CLIP**
(per-utt WER capped at 100% before averaging) is the robust mean; raw WER, WER_L50
and the tail rate (share of utterances above 50% WER) are also tracked. Speaker
similarity is cosine × 100 (ReDimNet); MSR-UTMOS is the MOS predictor.

> Scores are not comparable with ZTTS1-Eval.

What the scores in this document reflect:

- per-language ASR routing, enforced: `fa`, `hi`, `ko`, `th`, `vi`, `zh` on `Qwen/Qwen3-ASR-1.7B@7278e1e`, every other language on `openai/whisper-large-v3@06f233f`; a leaf scored with any other ASR is refused by the build;
- digits expanded to words on both reference and hypothesis before WER;
- prompt/target near-duplicates removed (Clean 22 removed, 57,388 → 57,366; ITW 164 removed, 17,319 → 17,155; list in `data/near_duplicates_removed.json`);
- all 20 ITW languages in the aggregates, including `tl`;
- ITW speaker similarity against a held-out recording of the same speaker (prompt similarity kept as a secondary column);
- pitch measured as a register-normalized pitch-range distance.

## 0. Headline

Macro = unweighted mean over the set's language basket (35 Clean, 20 ITW); a system is ranked only when it covers every language of the basket. Within a class tables are sorted by name, not by score: pick the axis you care about. **×ceil** columns are distances divided by the real-speech split-half ceiling (1.0 = indistinguishable from real speech, lower is better). The figure shows accuracy (x) against rhythm pause factor (left) and pitch-range distance (right); bottom-left is best.

### Capability index (both sets)

One score per system, fit like Epoch's Capabilities Index ([ECI](https://epoch.ai/data/eci-documentation/methodology)): item response theory, `performance = sigmoid(slope_item · (capability_system − difficulty_item))`, over 420 items — one metric on one set in one language — so a system scored on only some languages is still placed on the common scale, which no macro can do. Each metric enters as a performance in [0, 1]: 1 − WER_CLIP/100; 1 − Tail>50%/100; SPK prompt / 100 (Clean); SPK held-out / 100 (ITW); (MSR-UTMOS − 1) / 4; 1 − MPM span, clipped to [0, 1]; min(1, 1 / rhythm pace ×ceil); min(1, 1 / rhythm pause ×ceil); min(1, 1 / pitch range ×ceil). Intelligibility, voice, quality and prosody weigh equally. Scale anchors: real speech = 100, zonos2_accurate = 50. Interval = 90% bootstrap over each system's own items (500 refits); anchors have none by definition. Few languages = wide interval.

Over time, each system at its public release date (sources in `rd_common.RELEASES`; variants of one checkpoint share its date). The line is the frontier: the best public system scored on every language released by that date. Hollow points are not yet public and are dated by their planned release.

![Capability index over time](assets/plots/capability_index_time.png)

| # | System | Index | 90% CI | Languages (Clean · ITW) |
|--:|---|--:|---|---|
| 1 | real speech | **100.0** | anchor | 35/35 · 20/20 |
| 2 | moss-tts-v1.5† | **69.4** | 64.1 – 71.9 | 35/35 · 20/20 |
| 3 | drama-3 | **62.8** | 58.5 – 65.8 | 35/35 · 20/20 |
| 4 | higgs-audio-v3 | **62.2** | 57.8 – 64.3 | 35/35 · 20/20 |
| 5 | moss-tts† | **61.2** | 55.3 – 63.6 | 35/35 · 20/20 |
| 6 | fish-s2-pro | **58.8** | 52.7 – 63.9 | 35/35 · 20/20 |
| 7 | voxcpm2-reftext | **56.1** | 51.0 – 59.7 | 35/35 · 20/20 |
| 8 | zonos3-rate30-20k_embedding | **55.2** | 51.7 – 58.2 | 35/35 · 20/20 |
| 9 | zonos3-mini-laci_embedding◇ | **51.9** | 48.5 – 54.2 | 35/35 · 20/20 |
| 10 | vibevoice-1.5b† | **51.4** | 37.7 – 70.0 | 2/35 · 2/20 |
| 11 | omnivoice-reftext | **50.9** | 47.4 – 52.8 | 35/35 · 20/20 |
| 12 | vibevoice-7b† | **50.5** | 47.6 – 52.3 | 35/35 · 20/20 |
| 13 | zonos2_accurate† | **50.0** | anchor | 35/35 · 20/20 |
| 14 | zonos3-mini_embedding◇ | **49.8** | 47.2 – 52.4 | 35/35 · 20/20 |
| 15 | zonos3-mini-laci_reference† | **47.6** | 45.1 – 50.8 | 35/35 · 20/20 |
| 16 | voxcpm2† | **47.3** | 44.3 – 50.4 | 35/35 · 20/20 |
| 17 | zonos3-mini_reference† | **46.1** | 42.8 – 49.3 | 35/35 · 20/20 |
| 18 | zonos3-rate30-20k_reference | **45.9** | 42.7 – 48.8 | 35/35 · 20/20 |
| 19 | audio8-tts-realistic-0.6b | **44.3** | 41.9 – 47.9 | 35/35 · 20/20 |
| 20 | moss-tts-local† | **43.7** | 37.7 – 48.1 | 9/35 · 9/20 |
| 21 | qwen3-tts-1.7b-reftext | **40.4** | 37.8 – 49.4 | 35/35 · 20/20 |
| 22 | gemini_3.8-flash | **39.9** | 6.8 – 43.7 | 35/35 · 20/20 |
| 23 | zonos2_expressive† | **39.7** | 24.7 – 42.5 | 35/35 · 20/20 |
| 24 | qwen3-1.7b◇ | **38.6** | 35.9 – 44.5 | 35/35 · 20/20 |
| 25 | qwen3-tts-0.6b-reftext | **37.2** | 34.0 – 42.2 | 35/35 · 20/20 |
| 26 | qwen3-0.6b◇ | **36.2** | 32.9 – 41.8 | 35/35 · 20/20 |
| 27 | omnivoice† | **35.2** | 22.5 – 37.6 | 35/35 · 20/20 |
| 28 | cosyvoice3 | **30.5** | 18.3 – 34.8 | 29/35 · 17/20 |
| 29 | vibevoice-rt-0.5b‡ | **14.8** | -12.9 – 36.4 | 10/35 · 10/20 |
| 30 | kokoro‡ | **9.1** | -18.4 – 15.4 | 8/35 · 8/20 |
| 31 | moss-tts-local-v1.5† | **6.0** | -16.2 – 29.9 | 9/35 · 9/20 |
| 32 | voxtral‡ | **-4.3** | -32.7 – -6.5 | 35/35 · 20/20 |
| 33 | glm-tts | **-13.7** | -43.8 – -13.6 | 21/35 · 9/20 |
| 34 | neutts-2e‡ | **-27.6** | -130.9 – 25.5 | 1/35 · 1/20 |

### Prosody index (both sets)

The same fit over 315 prosody items only (ECI's domain-specific index), with every prosody metric rather than the capability index's four: **pace** — min(1, 1 / rhythm pace ×ceil), min(1, 1 / char/s ×ceil), 1 / (1 + paired Dur |log2|) (ITW); **pauses** — min(1, 1 / rhythm pause ×ceil), 1 / (1 + paired Pause Δ) (ITW); **pitch** — min(1, 1 / pitch range ×ceil), 1 / (1 + paired Range |log2|) (ITW); **contour** — 1 − MPM span, clipped to [0, 1]. The 4 aspects weigh equally, so a quantity measured several ways does not outvote one measured once. MPM span is scored on silence-trimmed audio (`docs/controls_2026-08.md` C6). Same anchors and interval as the capability index.

| # | System | Index | 90% CI | Languages (Clean · ITW) |
|--:|---|--:|---|---|
| 1 | real speech | **100.0** | anchor | 35/35 · 20/20 |
| 2 | moss-tts-v1.5† | **55.9** | 52.6 – 55.9 | 35/35 · 20/20 |
| 3 | higgs-audio-v3 | **53.6** | 51.6 – 53.7 | 35/35 · 20/20 |
| 4 | moss-tts† | **53.5** | 51.2 – 53.7 | 35/35 · 20/20 |
| 5 | drama-3 | **53.3** | 51.5 – 53.4 | 35/35 · 20/20 |
| 6 | voxcpm2-reftext | **51.7** | 50.4 – 52.6 | 35/35 · 20/20 |
| 7 | qwen3-0.6b◇ | **51.3** | 50.2 – 51.7 | 35/35 · 20/20 |
| 8 | zonos3-rate30-20k_embedding | **51.0** | 50.1 – 51.7 | 35/35 · 20/20 |
| 9 | vibevoice-1.5b† | **51.0** | 47.4 – 55.3 | 2/35 · 2/20 |
| 10 | qwen3-tts-1.7b-reftext | **50.8** | 49.8 – 51.4 | 35/35 · 20/20 |
| 11 | vibevoice-7b† | **50.7** | 49.8 – 51.1 | 35/35 · 20/20 |
| 12 | fish-s2-pro | **50.7** | 49.3 – 52.5 | 35/35 · 20/20 |
| 13 | voxcpm2† | **50.4** | 49.7 – 50.8 | 35/35 · 20/20 |
| 14 | zonos3-mini-laci_embedding◇ | **50.1** | 49.5 – 50.6 | 35/35 · 20/20 |
| 15 | zonos2_accurate† | **50.0** | anchor | 35/35 · 20/20 |
| 16 | qwen3-1.7b◇ | **49.8** | 49.1 – 50.8 | 35/35 · 20/20 |
| 17 | zonos3-mini_embedding◇ | **49.7** | 49.2 – 50.2 | 35/35 · 20/20 |
| 18 | qwen3-tts-0.6b-reftext | **49.6** | 48.7 – 50.2 | 35/35 · 20/20 |
| 19 | omnivoice-reftext | **49.4** | 48.9 – 50.1 | 35/35 · 20/20 |
| 20 | zonos3-mini-laci_reference† | **49.3** | 48.8 – 50.1 | 35/35 · 20/20 |
| 21 | zonos3-mini_reference† | **49.0** | 48.6 – 49.8 | 35/35 · 20/20 |
| 22 | zonos3-rate30-20k_reference | **48.9** | 48.4 – 49.8 | 35/35 · 20/20 |
| 23 | cosyvoice3 | **48.7** | 47.9 – 50.7 | 29/35 · 17/20 |
| 24 | moss-tts-local-v1.5† | **48.3** | 47.5 – 49.5 | 9/35 · 9/20 |
| 25 | moss-tts-local† | **48.2** | 47.4 – 49.2 | 9/35 · 9/20 |
| 26 | audio8-tts-realistic-0.6b | **46.5** | 45.9 – 48.2 | 35/35 · 20/20 |
| 27 | omnivoice† | **45.7** | 44.7 – 47.4 | 35/35 · 20/20 |
| 28 | neutts-2e‡ | **45.4** | 40.5 – 48.0 | 1/35 · 1/20 |
| 29 | vibevoice-rt-0.5b‡ | **44.5** | 43.0 – 47.1 | 10/35 · 10/20 |
| 30 | gemini_3.8-flash | **44.5** | 43.2 – 46.7 | 35/35 · 20/20 |
| 31 | zonos2_expressive† | **43.9** | 42.1 – 46.2 | 35/35 · 20/20 |
| 32 | voxtral‡ | **42.3** | 40.5 – 45.2 | 35/35 · 20/20 |
| 33 | glm-tts | **37.5** | 31.0 – 41.1 | 21/35 · 9/20 |
| 34 | kokoro‡ | **33.9** | 26.1 – 40.8 | 8/35 · 8/20 |

### Clean (FLEURS-R)

![WER vs prosody, Clean (FLEURS-R)](assets/plots/tradeoff_zero_shot.png)

**Real-speech reference**

| System | WER_CLIP | Tail >50% | SPK (prompt) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | MPM span (Clean only) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gt (fleurs-r) | 9.66 | 1.4 | — | 3.28 | — | — | — | -0.01 |

**with-transcript clones**

| System | WER_CLIP | Tail >50% | SPK (prompt) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | MPM span (Clean only) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| audio8-tts-realistic-0.6b | 13.45 | 4.1 | 80.3 | 3.12 | 6.96 | 6.15 | 1.94 | 0.44 |
| drama-3 | 5.62 | 0.2 | 82.1 | 3.16 | 2.95 | 2.50 | 1.70 | 0.05 |
| fish-s2-pro | 11.03 | 3.0 | 81.2 | 3.16 | 2.18 | 2.13 | 3.22 | 0.20 |
| higgs-audio-v3 | 9.42 | 1.3 | 79.5 | 2.99 | 2.55 | 2.15 | 2.42 | 0.07 |
| omnivoice-reftext | 7.52 | 1.2 | 87.6 | 3.05 | 9.14 | 8.05 | 2.51 | 0.16 |
| qwen3-tts-0.6b-reftext | 29.56 | 22.1 | 79.8 | 3.20 | 5.03 | 12.83 | 3.74 | 0.14 |
| qwen3-tts-1.7b-reftext | 22.80 | 13.5 | 80.6 | 3.16 | 5.85 | 3.93 | 2.00 | 0.14 |
| voxcpm2-reftext | 12.47 | 4.9 | 86.2 | 2.96 | 5.50 | 6.68 | 2.16 | 0.03 |

**no-transcript clones †**

| System | WER_CLIP | Tail >50% | SPK (prompt) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | MPM span (Clean only) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| moss-tts† | 14.66 | 5.4 | 83.6 | 3.07 | 3.28 | 4.13 | 1.62 | 0.05 |
| moss-tts-v1.5† | 10.95 | 1.9 | 82.9 | 3.04 | 2.78 | 2.23 | 1.73 | 0.06 |
| omnivoice† | 51.10 | 45.9 | 83.1 | 3.08 | 17.28 | 36.23 | 8.41 | 0.41 |
| vibevoice-7b† | 11.50 | 2.9 | 82.2 | 3.14 | 4.11 | 7.76 | 2.75 | 0.03 |
| voxcpm2† | 13.72 | 6.1 | 78.9 | 2.93 | 9.10 | 5.82 | 2.15 | 0.07 |
| zonos2_accurate† | 7.10 | 1.0 | 75.0 | 3.11 | 5.86 | 9.00 | 3.82 | 0.15 |
| zonos2_expressive† | 16.55 | 10.1 | 66.7 | 2.97 | 27.90 | 38.38 | 11.24 | 0.60 |
| zonos3-mini-laci_reference† | 8.35 | 0.9 | 76.5 | 3.03 | 3.93 | 5.31 | 2.37 | 0.17 |
| zonos3-mini_reference† | 8.68 | 1.4 | 76.2 | 3.02 | 4.21 | 5.76 | 2.42 | 0.17 |

**x-vector-only clones ◇**

| System | WER_CLIP | Tail >50% | SPK (prompt) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | MPM span (Clean only) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-0.6b◇ | 30.39 | 24.3 | 69.2 | 3.55 | 5.09 | 3.74 | 2.28 | 0.11 |
| qwen3-1.7b◇ | 24.46 | 15.8 | 69.6 | 3.54 | 5.40 | 3.80 | 2.09 | 0.23 |
| zonos3-mini-laci_embedding◇ | 8.33 | 0.9 | 81.6 | 3.20 | 7.07 | 8.64 | 2.23 | 0.13 |
| zonos3-mini_embedding◇ | 8.38 | 1.1 | 81.5 | 3.20 | 7.71 | 9.69 | 2.60 | 0.13 |

**preset voices ‡ (not cloning the prompt)**

| System | WER_CLIP | Tail >50% | SPK (prompt) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | MPM span (Clean only) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| voxtral‡ | 56.38 | 56.3 | 16.9 | 3.34 | 18.58 | 11.01 | 6.09 | 0.83 |

**unclassified (not in `rd_common.CONDITIONS`)**

| System | WER_CLIP | Tail >50% | SPK (prompt) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | MPM span (Clean only) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gemini_3.8-flash | 5.61 | 0.4 | 29.7 | 3.32 | 6.21 | 3.14 | 8.86 | 0.65 |
| zonos3-rate30-20k_embedding | 8.28 | 0.9 | 83.1 | 3.11 | 4.10 | 2.92 | 2.59 | 0.14 |
| zonos3-rate30-20k_reference | 7.95 | 1.1 | 77.7 | 3.03 | 3.96 | 5.15 | 2.68 | 0.15 |

**Partial coverage — not comparable**

Missing WER for at least one basket language; no macro value is shown.

| System | Langs with WER | Missing |
|---|--:|---|
| cosyvoice3 | 29/35 | `ar`, `hi`, `th`, `vi`, `he`, `fa` |
| glm-tts | 21/35 | 14 languages |
| kokoro‡ | 8/35 | 27 languages |
| moss-tts-local† | 9/35 | 26 languages |
| moss-tts-local-v1.5† | 9/35 | 26 languages |
| neutts-2e‡ | 1/35 | 34 languages |
| vibevoice-1.5b† | 2/35 | 33 languages |
| vibevoice-rt-0.5b‡ | 10/35 | 25 languages |

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### ITW (VoxBlink2, in the wild)

Speaker similarity is scored against a held-out recording of the same speaker (**SPK held-out**); **SPK prompt** is similarity to the prompt, and the **copy gap** is their difference — a large gap means the system reproduces the prompt's channel rather than the speaker. **Dur**, **Pause Δ** and **Pitch-range |log2|** are per-utterance paired medians against the real recording (lower is better).

![WER vs prosody, ITW (VoxBlink2, in the wild)](assets/plots/tradeoff_zero_shot_voxblink.png)

**Real-speech reference**

| System | WER_CLIP | Tail >50% | SPK held-out | SPK prompt | Copy gap (prompt − held-out) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | Dur \|log2\| | Pause Δ | Pitch-range \|log2\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gt (voxblink) | 9.19 | 0.9 | 79.3 | 79.3 | 0.0 | 2.25 | 1.10 | 1.00 | 0.96 | — | — | — |

**with-transcript clones**

| System | WER_CLIP | Tail >50% | SPK held-out | SPK prompt | Copy gap (prompt − held-out) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | Dur \|log2\| | Pause Δ | Pitch-range \|log2\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| audio8-tts-realistic-0.6b | 7.33 | 1.8 | 58.9 | 67.5 | 8.6 | 2.68 | 3.19 | 2.68 | 1.58 | 0.24 | 1.90 | 0.47 |
| drama-3 | 3.20 | 0.4 | 64.1 | 72.5 | 8.4 | 2.65 | 3.34 | 2.56 | 2.83 | 0.22 | 0.80 | 0.48 |
| fish-s2-pro | 5.98 | 1.5 | 61.7 | 70.3 | 8.6 | 2.64 | 2.01 | 2.75 | 2.87 | 0.19 | 0.60 | 0.52 |
| higgs-audio-v3 | 12.66 | 3.6 | 62.4 | 69.2 | 6.9 | 2.34 | 1.85 | 2.22 | 1.38 | 0.19 | 0.59 | 0.44 |
| omnivoice-reftext | 5.03 | 1.0 | 65.0 | 75.0 | 10.0 | 2.82 | 4.46 | 3.19 | 1.50 | 0.22 | 0.64 | 0.45 |
| qwen3-tts-0.6b-reftext | 18.14 | 12.1 | 63.7 | 73.3 | 9.6 | 2.65 | 2.91 | 3.34 | 1.77 | — | 0.74 | 0.46 |
| qwen3-tts-1.7b-reftext | 14.61 | 8.8 | 65.0 | 74.5 | 9.5 | 2.66 | 4.59 | 4.80 | 1.90 | — | 0.67 | 0.44 |
| voxcpm2-reftext | 5.12 | 0.6 | 67.5 | 77.8 | 10.3 | 2.27 | 1.65 | 2.17 | 1.22 | 0.20 | 0.62 | 0.49 |

**no-transcript clones †**

| System | WER_CLIP | Tail >50% | SPK held-out | SPK prompt | Copy gap (prompt − held-out) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | Dur \|log2\| | Pause Δ | Pitch-range \|log2\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| moss-tts† | 9.33 | 2.6 | 63.7 | 72.9 | 9.2 | 2.54 | 2.64 | 1.91 | 1.18 | 0.22 | 0.82 | 0.46 |
| moss-tts-v1.5† | 7.22 | 1.5 | 63.1 | 72.0 | 8.9 | 2.53 | 2.03 | 1.91 | 1.15 | 0.21 | 0.74 | 0.45 |
| omnivoice† | 27.26 | 16.1 | 62.2 | 72.2 | 9.9 | 2.79 | 5.22 | 6.58 | 1.44 | — | 0.82 | 0.50 |
| vibevoice-7b† | 13.51 | 6.5 | 58.1 | 65.2 | 7.2 | 2.43 | 2.85 | 2.33 | 2.14 | 0.24 | 0.89 | 0.50 |
| voxcpm2† | 5.25 | 0.7 | 64.2 | 73.9 | 9.7 | 2.31 | 4.11 | 3.98 | 1.18 | 0.23 | 0.56 | 0.48 |
| zonos2_accurate† | 4.68 | 0.9 | 52.6 | 57.8 | 5.3 | 2.79 | 2.61 | 3.03 | 1.64 | 0.22 | 0.85 | 0.47 |
| zonos2_expressive† | 9.59 | 4.5 | 51.4 | 57.0 | 5.6 | 2.46 | 7.73 | 9.57 | 3.75 | 0.24 | 1.07 | 0.61 |
| zonos3-mini-laci_reference† | 8.66 | 3.6 | 51.5 | 56.8 | 5.3 | 2.58 | 6.08 | 5.13 | 1.48 | 0.29 | 1.59 | 0.52 |
| zonos3-mini_reference† | 11.37 | 6.8 | 50.0 | 55.2 | 5.2 | 2.56 | 8.10 | 6.94 | 1.62 | 0.29 | 1.61 | 0.54 |

**x-vector-only clones ◇**

| System | WER_CLIP | Tail >50% | SPK held-out | SPK prompt | Copy gap (prompt − held-out) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | Dur \|log2\| | Pause Δ | Pitch-range \|log2\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3-0.6b◇ | 15.07 | 9.8 | 55.9 | 64.3 | 8.3 | 2.85 | 3.18 | 3.35 | 1.45 | — | 0.81 | 0.45 |
| qwen3-1.7b◇ | 11.71 | 6.7 | 58.1 | 67.1 | 9.0 | 2.84 | 3.41 | 3.56 | 1.21 | — | 0.78 | 0.46 |
| zonos3-mini-laci_embedding◇ | 8.75 | 3.4 | 63.1 | 72.1 | 9.0 | 2.69 | 2.54 | 2.40 | 1.42 | 0.21 | 0.72 | 0.51 |
| zonos3-mini_embedding◇ | 9.64 | 4.5 | 62.5 | 71.3 | 8.8 | 2.68 | 2.91 | 2.76 | 1.35 | 0.21 | 0.73 | 0.51 |

**preset voices ‡ (not cloning the prompt)**

| System | WER_CLIP | Tail >50% | SPK held-out | SPK prompt | Copy gap (prompt − held-out) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | Dur \|log2\| | Pause Δ | Pitch-range \|log2\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| voxtral‡ | 43.29 | 43.1 | 13.9 | 13.8 | -0.1 | 3.38 | 6.98 | 4.15 | 2.83 | — | 4.48 | 0.68 |

**unclassified (not in `rd_common.CONDITIONS`)**

| System | WER_CLIP | Tail >50% | SPK held-out | SPK prompt | Copy gap (prompt − held-out) | MSR-UTMOS | Rhythm pace ×ceil | Rhythm pause ×ceil | Pitch range ×ceil | Dur \|log2\| | Pause Δ | Pitch-range \|log2\| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemini_3.8-flash | 4.77 | 0.3 | 24.7 | 23.9 | -0.8 | 3.29 | 3.03 | 2.55 | 2.88 | 0.21 | 0.71 | 0.56 |
| zonos3-rate30-20k_embedding | 7.52 | 2.8 | 64.6 | 74.0 | 9.4 | 2.68 | 3.36 | 3.71 | 1.33 | 0.22 | 0.74 | 0.51 |
| zonos3-rate30-20k_reference | 12.57 | 8.6 | 49.9 | 55.1 | 5.3 | 2.51 | 11.96 | 11.19 | 2.01 | 0.32 | 1.56 | 0.56 |

**Partial coverage — not comparable**

Missing WER for at least one basket language; no macro value is shown.

| System | Langs with WER | Missing |
|---|--:|---|
| cosyvoice3 | 17/20 | `ko`, `ar`, `th` |
| glm-tts | 9/20 | `pt`, `ar`, `hi`, `id`, `tr`, `tl`, `pl`, `th`, `nl`, `uk`, `vi` |
| kokoro‡ | 8/20 | `de`, `ko`, `ru`, `ar`, `id`, `tr`, `tl`, `pl`, `th`, `nl`, `uk`, `vi` |
| moss-tts-local† | 9/20 | `pt`, `ar`, `hi`, `id`, `tr`, `tl`, `pl`, `th`, `nl`, `uk`, `vi` |
| moss-tts-local-v1.5† | 9/20 | `pt`, `ar`, `hi`, `id`, `tr`, `tl`, `pl`, `th`, `nl`, `uk`, `vi` |
| neutts-2e‡ | 1/20 | 19 languages |
| vibevoice-1.5b† | 2/20 | 18 languages |
| vibevoice-rt-0.5b‡ | 10/20 | `zh`, `ru`, `ar`, `hi`, `id`, `tr`, `tl`, `th`, `uk`, `vi` |

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

MPM span is reported for the Clean set only: its anchors collapse on in-the-wild audio (`docs/mpm_itw_limits.md`).

## 1. Coverage matrix

| Model | zero_shot | voxblink |
|---|:-:|:-:|
| `drama-3` | ✓ | ✓ |
| `fish-s2-pro` | ✓ | ✓ |
| `glm-tts` | ✓ | ✓ |
| `higgs-audio-v3` | ✓ | ✓ |
| `omnivoice` | ✓ | ✓ |
| `omnivoice-reftext` | ✓ | ✓ |
| `qwen3-0.6b` | ✓ | ✓ |
| `qwen3-1.7b` | ✓ | ✓ |
| `qwen3-tts-0.6b-reftext` | ✓ | ✓ |
| `qwen3-tts-1.7b-reftext` | ✓ | ✓ |
| `voxcpm2` | ✓ | ✓ |
| `voxcpm2-reftext` | ✓ | ✓ |
| `voxtral` | ✓ | ✓ |
| `audio8-tts-realistic-0.6b` | ✓ | ✓ |
| `cosyvoice3` | ✓ | ✓ |
| `gemini_3.8-flash` | ✓ | ✓ |
| `kokoro` | ✓ | ✓ |
| `moss-tts` | ✓ | ✓ |
| `moss-tts-local` | ✓ | ✓ |
| `moss-tts-local-v1.5` | ✓ | ✓ |
| `moss-tts-v1.5` | ✓ | ✓ |
| `neutts-2e` | ✓ | ✓ |
| `vibevoice-1.5b` | ✓ | ✓ |
| `vibevoice-7b` | ✓ | ✓ |
| `vibevoice-rt-0.5b` | ✓ | ✓ |
| `zonos2_accurate` | ✓ | ✓ |
| `zonos2_expressive` | ✓ | ✓ |
| `zonos3-mini-laci_embedding` | ✓ | ✓ |
| `zonos3-mini-laci_reference` | ✓ | ✓ |
| `zonos3-mini_embedding` | ✓ | ✓ |
| `zonos3-mini_reference` | ✓ | ✓ |
| `zonos3-rate30-20k_embedding` | ✓ | ✓ |
| `zonos3-rate30-20k_reference` | ✓ | ✓ |
| `gt (fleurs-r)` | ✓ | — |
| `gt (voxblink)` | — | ✓ |

## 1b. Conditioning

The systems below are NOT all run under the same conditioning — what each one
is given alongside the target text is a harness policy that differs per
system (see `scripts/results_doc/rd_common.py:CONDITIONS` for the code
provenance). WER is strongly sensitive to it: `omnivoice` scores 34.2 avg WER
without the reference transcript vs 3.2 with it, on the same weights. Rows in
the ranking tables carry the class markers († ◇ ‡) so classes are never
compared undisclosed.

| Model | ref transcript? | cloning class | sampling/config |
|---|:-:|---|---|
| `drama-3` | yes | with-transcript clone | ref_text kept (server requires ref_audio + ref_text); vllm-omni 0.32.0 on vllm 0.32.0; unseeded server defaults (T 1.0 top-k 30 top-p 0.9) |
| `fish-s2-pro` | yes | with-transcript clone | ref_text kept (server rejects no-ref-text); unseeded server defaults |
| `glm-tts` | yes | with-transcript clone | ref_text kept (default policy, no-ref-text unverified); unseeded server defaults |
| `higgs-audio-v3` | yes | with-transcript clone | ref_text kept (upstream prescribes ref_audio + ref_text); unseeded server defaults |
| `omnivoice†` | no | no-transcript clone † | --no-ref-text (reftext_policy); unseeded server defaults |
| `omnivoice-reftext` | yes | with-transcript clone | same weights as omnivoice, ref_text kept (separate key); unseeded server defaults |
| `qwen3-0.6b◇` | no | x-vector-only clone ◇ | --no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults |
| `qwen3-1.7b◇` | no | x-vector-only clone ◇ | --no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults |
| `qwen3-tts-0.6b-reftext` | yes | with-transcript clone | same weights as qwen3-tts-0.6b, ref_text kept (separate key); unseeded server defaults |
| `qwen3-tts-1.7b-reftext` | yes | with-transcript clone | same weights as qwen3-tts-1.7b, ref_text kept (separate key); unseeded server defaults |
| `voxcpm2†` | no | no-transcript clone † | --no-ref-text (reftext_policy); unseeded server defaults |
| `voxcpm2-reftext` | yes | with-transcript clone | same weights as voxcpm2, ref_text kept (separate key); unseeded server defaults |
| `voxtral‡` | n/a | preset voice ‡ | preset mode (task_type=CustomVoice); unseeded server defaults |
| `audio8-tts-realistic-0.6b` | yes | with-transcript clone | SGLang-Omni adapter (Audio8_TTS@4eb54a6, sglang-omni@68a5723); prompt wav + transcript; T 0.8 top-p 0.95 top-k 50, ≤1024 frames; unseeded server sampling |
| `cosyvoice3` | yes | with-transcript clone | native upstream; prompt wav + transcript (endofprompt-terminated system prompt; ja→katakana) |
| `kokoro‡` | n/a | preset voice ‡ | built-in voice per language; prompt wav ignored |
| `moss-tts†` | no | no-transcript clone † | audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling |
| `moss-tts-local†` | no | no-transcript clone † | audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling |
| `moss-tts-local-v1.5†` | no | no-transcript clone † | audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling |
| `moss-tts-v1.5†` | no | no-transcript clone † | audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling |
| `neutts-2e‡` | n/a | preset voice ‡ | 4 fixed speakers, deterministic per-utt rotation; prompt wav ignored |
| `vibevoice-1.5b†` | no | no-transcript clone † | voice-prompt clone (voice_samples=[ref wav], no transcript); cfg 1.3, 10 DDPM steps |
| `vibevoice-7b†` | no | no-transcript clone † | voice-prompt clone (voice_samples=[ref wav], no transcript); cfg 1.3, 10 DDPM steps |
| `vibevoice-rt-0.5b‡` | n/a | preset voice ‡ | preset speaker embeddings, per-utt rotation; prompt wav ignored |
| `zonos2_accurate†` | no | no-transcript clone † | audio-only clone; accurate_mode=true; seed 0, temp 1.15 vendor sampling |
| `zonos2_expressive†` | no | no-transcript clone † | audio-only clone; accurate_mode=false; seed 0, temp 1.15 vendor sampling |
| `zonos3-mini-laci_embedding◇` | no | x-vector-only clone ◇ | as zonos3-mini_embedding plus LACI (laci=true; heads as zonos3-mini-laci_reference) |
| `zonos3-mini-laci_reference†` | no | no-transcript clone † | as zonos3-mini_reference plus LACI (laci=true; ZTTS3-Eval assets/laci heads: zh heads for cmn, en heads for every other server language) |
| `zonos3-mini_embedding◇` | no | x-vector-only clone ◇ | release server, ownmect_s2 speaker embedding only (speaker_mode=embedding), no transcript; sampling as zonos3-mini_reference |
| `zonos3-mini_reference†` | no | no-transcript clone † | release server, raw reference prefix (speaker_mode=reference), no transcript; seed 0, T 0.75 top-k 80 top-p 0.69, rep 96×2.4, CFG 1, 0.2 s silence prefill, server TN for en zh de es fr it pt ja ko |

Seeding: `zonos2_*` and `zonos3-mini_*` runs are fixed-seed (0); vLLM-Omni-served systems pass no
seed or temperature (unseeded server-default sampling); native recipes note
their sampling defaults above.

## 2. Zero-shot WER_CLIP % (35 languages)

| Lang | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 1.92 | 2.14 | 4.54 | 3.56 | 28.30 | 2.60 | 2.20 | 1.89 | 3.38 | 2.37 | 3.59 | 3.93 | 2.64 | 2.10 | 3.14 | 2.10 | 2.10 | 2.59 | 2.62 | 25.06 | 2.83 | 11.96 | 3.87 | 2.81 | 2.91 | 2.45 | 4.33 | 2.51 | 2.26 | 2.40 | 2.21 | 2.87 | 2.11 | 4.30 |
| zh | 3.52 | 4.02 | 4.23 | 6.29 | 33.05 | 3.34 | 3.48 | 3.07 | 5.19 | 4.64 | 5.06 | 4.63 | 91.49 | 4.32 | 3.35 | 4.01 | 4.65 | 3.72 | 3.75 | 99.67 | 3.89 | — | 4.72 | 3.68 | — | 11.82 | 16.25 | 3.63 | 3.64 | 3.65 | 7.32 | 4.09 | 4.08 | 3.02 |
| de | 2.54 | 3.10 | 87.63 | 5.20 | 38.33 | 3.40 | 3.18 | 2.66 | 4.71 | 3.17 | 4.50 | 4.06 | 3.40 | 3.45 | 8.75 | 2.90 | — | 4.05 | 4.84 | 19.68 | 4.17 | — | — | 5.81 | 13.21 | 3.75 | 11.96 | 3.84 | 3.89 | 3.90 | 3.90 | 3.74 | 3.88 | 4.64 |
| es | 1.75 | 1.99 | 73.33 | 3.18 | 40.95 | 2.13 | 2.17 | 1.77 | 3.15 | 2.26 | 3.70 | 3.21 | 2.00 | 3.00 | 4.15 | 1.99 | 1.96 | 2.52 | 2.56 | 28.63 | 2.52 | — | — | 3.02 | 4.20 | 2.09 | 7.65 | 2.62 | 2.22 | 2.78 | 2.34 | 2.15 | 2.22 | 2.89 |
| fr | 2.56 | 3.03 | 87.00 | 4.74 | 26.15 | 2.81 | 3.07 | 2.84 | 4.81 | 3.07 | 5.10 | 4.33 | 3.03 | 3.45 | 6.62 | 2.92 | 3.85 | 4.58 | 4.60 | 16.67 | 4.21 | — | — | 6.73 | 15.03 | 3.37 | 12.42 | 3.68 | 3.96 | 3.65 | 4.11 | 3.54 | 3.73 | 4.08 |
| it | 1.50 | 2.48 | 78.83 | 4.27 | 42.56 | 2.57 | 2.71 | 2.08 | 3.75 | 2.36 | 3.94 | 3.52 | 2.17 | 3.36 | 6.66 | 1.80 | 2.07 | 3.63 | 4.05 | 23.00 | 3.14 | — | — | 4.45 | 6.14 | 1.94 | 7.53 | 2.61 | 2.55 | 2.66 | 2.60 | 2.66 | 2.29 | 4.05 |
| ja | 2.61 | 3.16 | 97.97 | 5.08 | 17.84 | 3.10 | 5.46 | 3.56 | 7.30 | 3.71 | 5.06 | 4.44 | 95.06 | 4.86 | 10.74 | 2.96 | 3.69 | 8.69 | 11.69 | 35.71 | 6.74 | — | — | 10.59 | 26.94 | 7.20 | 9.37 | 3.58 | 4.68 | 3.60 | 5.11 | 3.62 | 4.57 | 4.16 |
| ko | 3.23 | 3.64 | 97.96 | 5.69 | 42.71 | 3.44 | 3.58 | 3.22 | 4.86 | 4.32 | 4.00 | 4.20 | 94.97 | 3.96 | 6.26 | 3.39 | — | 4.67 | 5.37 | 90.96 | 5.39 | — | — | 5.58 | 14.87 | 5.90 | 16.93 | 5.23 | 5.04 | 5.22 | 6.37 | 5.25 | 5.05 | 3.20 |
| ru | 4.07 | 4.77 | 99.65 | 6.48 | 37.96 | 4.48 | 5.92 | 4.75 | 7.44 | 5.21 | 5.84 | 5.43 | 92.76 | 7.07 | 6.59 | 4.35 | — | 6.80 | 6.99 | 17.39 | 6.60 | — | — | 6.84 | — | 4.80 | 9.42 | 5.92 | 5.83 | 6.04 | 6.31 | 5.92 | 5.63 | 5.67 |
| pt | 1.95 | 2.59 | — | 4.75 | 50.20 | 4.08 | 4.23 | 3.31 | 5.14 | 3.18 | 4.44 | 3.38 | 5.34 | 4.12 | 35.09 | 2.39 | 2.48 | 4.58 | — | — | 4.59 | — | — | 4.96 | 6.20 | 2.87 | 8.17 | 2.77 | 2.81 | 2.89 | 2.83 | 2.63 | 2.55 | 4.18 |
| ar | 4.27 | 6.86 | 99.66 | 11.65 | 60.29 | 9.19 | 21.34 | 33.76 | 27.31 | 31.66 | 9.28 | 8.00 | 29.09 | 11.65 | — | 5.52 | — | 17.75 | — | — | 14.81 | — | — | 15.78 | — | 5.68 | 13.04 | 10.50 | 9.08 | 10.89 | 9.27 | 13.07 | 10.18 | 13.69 |
| hi | 9.21 | 13.04 | 99.97 | 13.78 | 81.63 | 12.36 | 22.54 | 22.21 | 37.26 | 29.32 | 10.17 | 10.27 | 60.83 | 12.69 | — | 11.38 | 9.66 | 17.03 | — | — | 14.34 | — | — | 13.26 | — | 9.97 | 14.25 | 12.69 | 12.69 | 12.88 | 14.62 | 12.46 | 13.06 | 14.61 |
| tr | 4.10 | 7.04 | — | 8.30 | 69.76 | 7.10 | 60.13 | 43.24 | 52.73 | 37.49 | 5.89 | 4.67 | 68.00 | 7.47 | 85.80 | 4.74 | — | 9.84 | — | — | 9.87 | — | — | 10.16 | — | 5.32 | 16.48 | 6.76 | 7.24 | 7.03 | 7.58 | 7.26 | 7.74 | 9.60 |
| pl | 2.20 | 5.42 | — | 5.81 | 43.20 | 3.49 | 67.00 | 53.63 | 61.81 | 51.17 | 4.86 | 4.03 | 81.61 | 5.52 | 88.76 | 2.44 | — | 6.90 | — | — | 6.38 | — | — | 8.64 | 9.38 | 3.77 | 9.87 | 4.43 | 4.65 | 4.49 | 4.80 | 4.02 | 3.90 | 4.48 |
| id | 6.90 | 9.41 | — | 5.91 | 60.81 | 8.32 | 19.13 | 6.93 | 13.75 | 8.18 | 5.82 | 5.76 | 49.48 | 8.59 | 46.56 | 3.62 | — | 11.39 | — | — | 9.66 | — | — | 7.26 | — | 3.89 | 12.55 | 5.00 | 5.01 | 5.15 | 4.90 | 4.73 | 4.95 | 7.31 |
| th | 3.27 | 9.73 | — | 6.65 | 47.81 | 5.90 | 6.54 | 7.62 | 9.19 | 5.64 | 5.35 | 4.40 | 99.60 | 9.83 | — | 3.99 | — | 35.55 | — | — | 6.29 | — | — | 7.52 | — | 4.93 | 13.84 | 8.09 | 7.92 | 8.03 | 8.47 | 7.04 | 7.34 | 5.44 |
| fil | 7.35 | 16.24 | — | 10.78 | 77.49 | 10.96 | 31.90 | 19.55 | 27.78 | 16.18 | 8.92 | 9.25 | 20.37 | 23.20 | 44.50 | 8.42 | — | 15.78 | — | — | 13.59 | — | — | 12.45 | — | 7.72 | 11.13 | 10.26 | 10.39 | 10.15 | 10.52 | 11.46 | 10.69 | 13.43 |
| nl | 3.25 | 5.46 | — | 8.05 | 38.44 | 4.08 | 17.09 | 12.79 | 14.68 | 7.39 | 6.53 | 5.28 | 15.47 | 5.84 | 53.84 | 3.85 | — | 5.95 | — | — | 6.67 | — | — | 9.63 | 20.43 | 8.71 | 19.44 | 4.98 | 5.57 | 4.95 | 5.68 | 5.41 | 5.14 | 6.83 |
| vi | 2.30 | 26.96 | — | 5.58 | 28.75 | 3.43 | 63.44 | 68.55 | 36.31 | 26.48 | 4.39 | 5.09 | 96.86 | 24.77 | — | 3.21 | — | 36.73 | — | — | 10.91 | — | — | 16.76 | — | 4.75 | 10.78 | 4.20 | 4.50 | 4.29 | 4.58 | 5.14 | 4.97 | 4.35 |
| cs | 6.30 | 12.53 | 97.42 | 12.12 | 60.11 | 10.75 | 59.94 | 45.03 | 54.22 | 40.73 | 37.26 | 32.04 | 78.23 | 14.92 | 83.31 | 6.18 | — | 13.19 | — | — | 13.47 | — | — | 14.84 | — | 6.85 | 19.50 | 9.65 | 10.07 | 9.66 | 10.34 | 9.66 | 9.38 | 12.31 |
| sv | 3.79 | 6.34 | — | 7.86 | 53.79 | 10.42 | 28.68 | 19.07 | 23.79 | 16.15 | 7.66 | 6.34 | 39.35 | 6.99 | 55.14 | 4.79 | — | 8.46 | — | — | 8.97 | — | — | 11.48 | — | 5.66 | 20.57 | 7.47 | 7.89 | 7.52 | 7.95 | 7.95 | 7.54 | 7.99 |
| da | 5.82 | 13.12 | 87.39 | 12.54 | 57.27 | 8.50 | 32.71 | 27.16 | 38.17 | 29.93 | 9.93 | 8.39 | 42.73 | 13.89 | 64.59 | 6.85 | — | 13.15 | — | — | 12.61 | — | — | 21.09 | — | 10.83 | 32.34 | 10.05 | 11.09 | 10.00 | 11.19 | 11.44 | 12.20 | 11.71 |
| fi | 6.50 | 12.13 | 94.92 | 13.24 | 57.07 | 7.95 | 66.15 | 50.62 | 50.45 | 37.75 | 10.48 | 8.63 | 76.09 | 11.55 | 66.03 | 6.24 | — | 15.21 | — | — | 15.28 | — | — | 20.27 | — | 6.63 | 22.06 | 14.66 | 13.00 | 14.88 | 13.60 | 11.78 | 11.30 | 12.87 |
| el | 13.33 | 19.09 | 99.60 | 14.07 | 64.26 | 8.11 | 77.97 | 60.71 | 93.84 | 76.66 | 9.25 | 8.90 | 98.45 | 19.82 | 99.84 | 7.66 | — | 12.75 | — | — | 12.51 | — | — | 11.83 | — | 11.10 | 17.47 | 12.93 | 14.21 | 11.40 | 11.66 | 11.58 | 10.47 | 16.69 |
| hu | 7.24 | 11.71 | 95.24 | 14.27 | 48.49 | 8.93 | 66.49 | 51.70 | 65.61 | 49.35 | 65.38 | 64.19 | 79.41 | 23.04 | 81.70 | 8.81 | — | 17.71 | — | — | 16.65 | — | — | 19.29 | — | 11.87 | 21.77 | 15.12 | 14.25 | 15.32 | 14.25 | 12.36 | 12.67 | 18.73 |
| ro | 6.54 | 19.97 | — | 7.36 | 50.97 | 5.23 | 36.91 | 27.63 | 31.86 | 24.83 | 31.74 | 29.92 | 44.49 | 21.49 | 61.41 | 3.52 | — | 11.04 | — | — | 8.31 | — | — | 11.02 | — | 5.56 | 17.06 | 7.78 | 8.08 | 7.82 | 8.41 | 7.61 | 7.20 | 6.85 |
| uk | 4.91 | 11.75 | — | 7.26 | 48.04 | 5.57 | 22.98 | 14.29 | 23.61 | 13.83 | 15.23 | 12.98 | 89.72 | 22.59 | 91.11 | 4.27 | — | 12.24 | — | — | 12.33 | — | — | 10.63 | — | 6.14 | 11.41 | 7.88 | 7.66 | 7.99 | 8.16 | 6.90 | 6.98 | 6.31 |
| nb | 5.27 | 10.02 | — | 11.95 | 56.76 | 7.18 | 32.69 | 22.89 | 30.80 | 20.80 | 10.82 | 8.26 | 42.51 | 10.68 | 54.44 | 6.84 | — | 10.59 | — | — | 12.34 | — | — | 17.84 | — | 8.81 | 34.98 | 9.64 | 10.06 | 9.80 | 11.19 | 11.18 | 10.56 | 11.70 |
| he | 17.27 | 55.66 | 99.96 | 21.46 | 79.30 | 33.41 | 87.34 | 88.29 | 90.60 | 91.64 | 13.98 | 14.55 | 95.58 | 53.75 | — | 10.48 | — | 91.01 | — | — | 31.99 | — | — | 15.44 | — | 11.75 | 18.43 | 15.38 | 14.70 | 15.95 | 15.62 | 15.97 | 14.59 | 28.97 |
| bg | 7.49 | 12.67 | 99.35 | 12.12 | 60.73 | 7.88 | 31.46 | 28.31 | 34.63 | 29.28 | 35.44 | 32.26 | 85.54 | 32.70 | 40.74 | 6.52 | — | 19.71 | — | — | 17.54 | — | — | 13.94 | — | 8.59 | 18.41 | 11.26 | 11.83 | 11.48 | 12.09 | 11.01 | 10.01 | 14.70 |
| hr | 6.99 | 12.08 | 90.75 | 12.68 | 55.92 | 7.98 | 47.36 | 25.55 | 43.98 | 24.10 | 28.00 | 24.18 | 59.95 | 13.40 | 69.30 | 6.43 | — | 18.07 | — | — | 20.59 | — | — | 16.22 | — | 8.20 | 23.78 | 11.51 | 12.59 | 11.39 | 13.05 | 11.66 | 10.96 | 13.35 |
| sk | 7.08 | 11.42 | — | 13.21 | 50.31 | 8.94 | 56.42 | 37.07 | 49.99 | 34.40 | 36.48 | 30.90 | 70.61 | 14.69 | 78.90 | 7.03 | — | 17.59 | — | — | 16.41 | — | — | 16.42 | — | 8.56 | 21.91 | 11.88 | 11.97 | 11.90 | 11.94 | 12.17 | 11.56 | 12.20 |
| ca | 3.71 | 6.61 | 70.41 | 7.22 | 47.18 | 3.90 | 9.86 | 10.83 | 10.58 | 8.91 | 16.26 | 14.64 | 17.35 | 10.61 | 19.81 | 5.20 | — | 10.20 | — | — | 10.20 | — | — | 9.94 | — | 7.20 | 23.33 | 6.87 | 6.43 | 7.13 | 6.33 | 5.81 | 5.41 | 7.26 |
| ms | 6.14 | 9.06 | — | 7.93 | 63.45 | 9.88 | 18.93 | 7.67 | 12.29 | 8.05 | 7.63 | 6.82 | 53.48 | 8.80 | 45.35 | 4.66 | — | 11.18 | — | — | 8.98 | — | — | 8.99 | — | 4.87 | 14.60 | 6.41 | 6.18 | 6.38 | 6.18 | 7.06 | 6.42 | 8.15 |
| fa | 19.75 | 30.97 | 99.85 | 28.57 | 68.70 | 21.91 | 42.60 | 41.97 | 49.69 | 43.65 | 38.30 | 35.75 | 85.56 | 42.58 | — | 25.02 | — | 28.38 | — | — | 28.54 | — | — | 27.50 | — | 24.96 | 36.30 | 29.65 | 28.28 | 30.01 | 28.34 | 28.59 | 26.95 | 28.50 |
| **avg** | **5.62** | **11.03** | partial (21/35) | **9.42** | **51.10** | **7.52** | **30.39** | **24.46** | **29.56** | **22.80** | **13.72** | **12.47** | **56.38** | **13.45** | partial (29/35) | **5.61** | partial (8/35) | **14.66** | partial (9/35) | partial (9/35) | **10.95** | partial (1/35) | partial (2/35) | **11.50** | partial (10/35) | **7.10** | **16.55** | **8.33** | **8.35** | **8.38** | **8.68** | **8.28** | **7.95** | **9.66** |

The avg row is shown only for systems that cover all 35 languages;
others read `partial (n/35)` and are not comparable to full-coverage averages.
The `gt (fleurs-r)` column is real speech scored by the same ASR — the
achievable WER floor. zh/ja/ko/th are char-tokenized, so their "WER" is a
CER and the `avg` column mixes WER and CER units.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### Zero-shot WER_L50 % (mean of utts with WER < 50%, ignores catastrophic tail)

Caveat: when *every* utterance of a leaf is ≥50% WER, `utils/average_wer.py`
falls back to reporting L50 over the full set — so a cell can exceed 50 for a
model that collapsed on that language. The catastrophic tail itself is the
**Tail >50%** column of §0.

| Lang | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 1.92 | 2.14 | 3.95 | 3.26 | 21.67 | 2.28 | 2.16 | 1.89 | 2.54 | 2.22 | 2.81 | 3.02 | 2.37 | 2.10 | 3.06 | 2.03 | 2.10 | 2.44 | 2.41 | 2.44 | 2.78 | 5.61 | 2.90 | 2.71 | 2.78 | 2.31 | 2.63 | 2.51 | 2.26 | 2.40 | 2.21 | 2.34 | 2.11 | 4.21 |
| zh | 3.37 | 3.75 | 3.86 | 5.92 | 26.19 | 3.28 | 3.06 | 2.85 | 2.97 | 3.20 | 3.99 | 3.89 | 42.66 | 4.19 | 3.15 | 3.84 | 4.46 | 3.48 | 3.62 | 11.71 | 3.65 | — | 3.77 | 2.85 | — | 4.40 | 4.94 | 3.50 | 3.44 | 3.51 | 3.40 | 3.90 | 3.51 | 3.02 |
| de | 2.51 | 3.01 | 40.94 | 4.75 | 23.07 | 2.94 | 3.13 | 2.63 | 3.59 | 3.00 | 4.06 | 3.79 | 3.28 | 3.40 | 7.43 | 2.83 | — | 3.97 | 3.76 | 4.28 | 3.91 | — | — | 5.13 | 7.94 | 3.34 | 4.23 | 3.69 | 3.75 | 3.63 | 3.68 | 3.58 | 3.54 | 4.58 |
| es | 1.70 | 1.95 | 39.62 | 3.02 | 23.92 | 2.04 | 2.07 | 1.72 | 2.64 | 2.03 | 2.92 | 2.92 | 1.88 | 2.93 | 3.95 | 1.94 | 1.96 | 2.47 | 2.40 | 2.65 | 2.47 | — | — | 2.55 | 3.32 | 1.95 | 2.31 | 2.51 | 2.16 | 2.47 | 2.23 | 2.09 | 2.01 | 2.82 |
| fr | 2.56 | 3.00 | 42.05 | 4.29 | 15.94 | 2.74 | 3.04 | 2.81 | 3.47 | 2.98 | 4.13 | 3.82 | 3.03 | 3.35 | 6.27 | 2.92 | 3.85 | 4.52 | 4.21 | 4.32 | 4.09 | — | — | 5.28 | 8.28 | 3.24 | 4.08 | 3.65 | 3.87 | 3.55 | 3.75 | 3.42 | 3.54 | 4.05 |
| it | 1.50 | 2.44 | 41.02 | 3.89 | 23.50 | 2.15 | 2.59 | 2.00 | 3.15 | 2.36 | 3.28 | 3.00 | 2.17 | 3.25 | 5.25 | 1.80 | 2.07 | 3.44 | 3.49 | 3.60 | 3.14 | — | — | 3.82 | 5.05 | 1.91 | 2.68 | 2.61 | 2.52 | 2.62 | 2.56 | 2.63 | 2.25 | 4.05 |
| ja | 2.52 | 3.13 | 124.77 | 4.79 | 16.33 | 3.06 | 4.97 | 3.40 | 5.61 | 3.54 | 4.79 | 3.93 | 42.86 | 4.61 | 8.51 | 2.81 | 3.69 | 8.49 | 10.65 | 9.17 | 6.35 | — | — | 7.57 | 20.38 | 3.88 | 4.15 | 3.53 | 4.19 | 3.47 | 4.19 | 3.44 | 4.19 | 4.12 |
| ko | 3.10 | 3.44 | 107.21 | 5.51 | 25.23 | 3.30 | 3.43 | 3.18 | 3.69 | 3.46 | 3.65 | 3.79 | 44.62 | 3.78 | 4.49 | 3.35 | — | 4.54 | 4.79 | 6.21 | 4.60 | — | — | 4.46 | 9.91 | 4.91 | 5.22 | 5.08 | 4.63 | 5.10 | 4.81 | 5.06 | 4.66 | 3.15 |
| ru | 4.07 | 4.74 | 115.60 | 6.19 | 23.50 | 4.42 | 5.65 | 4.66 | 6.33 | 5.09 | 5.54 | 5.40 | 40.32 | 6.95 | 6.13 | 4.32 | — | 6.31 | 6.71 | 7.30 | 6.14 | — | — | 5.97 | — | 4.65 | 5.67 | 5.82 | 5.74 | 5.79 | 5.67 | 5.76 | 5.46 | 5.56 |
| pt | 1.92 | 2.52 | — | 3.23 | 27.16 | 3.06 | 3.94 | 3.20 | 3.94 | 3.02 | 3.76 | 3.12 | 4.63 | 3.95 | 25.96 | 2.25 | 2.45 | 4.17 | — | — | 4.13 | — | — | 4.11 | 4.99 | 2.64 | 3.50 | 2.67 | 2.68 | 2.64 | 2.67 | 2.59 | 2.40 | 4.05 |
| ar | 4.17 | 6.58 | 111.92 | 11.09 | 30.18 | 8.53 | 19.19 | 26.38 | 23.25 | 26.18 | 7.78 | 7.41 | 17.61 | 11.05 | — | 5.40 | — | 16.36 | — | — | 13.62 | — | — | 12.88 | — | 5.40 | 8.28 | 9.93 | 8.85 | 10.02 | 8.85 | 11.76 | 9.40 | 12.63 |
| hi | 9.08 | 12.62 | 136.16 | 12.84 | 39.70 | 12.03 | 14.92 | 12.39 | 18.76 | 13.48 | 9.87 | 10.00 | 13.81 | 12.07 | — | 11.21 | 9.55 | 15.76 | — | — | 13.60 | — | — | 11.54 | — | 9.40 | 10.98 | 12.20 | 12.02 | 12.02 | 12.16 | 12.01 | 11.61 | 14.27 |
| tr | 3.86 | 6.72 | — | 7.55 | 33.59 | 6.82 | 38.84 | 31.61 | 35.30 | 29.12 | 5.53 | 4.53 | 37.73 | 7.16 | 43.76 | 4.65 | — | 9.05 | — | — | 9.03 | — | — | 8.63 | — | 5.08 | 7.08 | 6.51 | 6.80 | 6.55 | 6.80 | 6.61 | 6.36 | 9.18 |
| pl | 2.20 | 5.33 | — | 5.46 | 21.61 | 3.24 | 39.73 | 36.46 | 37.74 | 35.55 | 4.30 | 3.51 | 40.46 | 5.35 | 43.48 | 2.44 | — | 6.48 | — | — | 6.12 | — | — | 6.93 | 6.05 | 3.72 | 4.34 | 4.37 | 4.58 | 4.23 | 4.41 | 3.82 | 3.71 | 4.30 |
| id | 6.90 | 8.72 | — | 5.66 | 29.87 | 7.18 | 17.09 | 6.66 | 10.86 | 6.51 | 4.67 | 5.03 | 34.34 | 8.10 | 34.58 | 3.47 | — | 10.27 | — | — | 8.93 | — | — | 6.12 | — | 3.77 | 5.35 | 4.87 | 4.83 | 4.94 | 4.71 | 4.61 | 4.63 | 6.17 |
| th | 3.14 | 9.13 | — | 6.22 | 25.08 | 5.68 | 4.25 | 5.68 | 5.95 | 5.02 | 4.30 | 3.77 | 110.12 | 9.52 | — | 3.80 | — | 31.19 | — | — | 5.82 | — | — | 5.70 | — | 4.55 | 6.54 | 7.88 | 7.52 | 7.49 | 7.42 | 6.74 | 6.67 | 5.17 |
| fil | 7.32 | 15.90 | — | 10.27 | 38.72 | 10.82 | 29.50 | 19.10 | 24.71 | 15.95 | 8.29 | 8.75 | 18.52 | 22.26 | 35.21 | 8.26 | — | 15.26 | — | — | 13.36 | — | — | 11.37 | — | 7.66 | 9.47 | 10.22 | 10.29 | 10.07 | 10.36 | 11.00 | 10.43 | 13.26 |
| nl | 3.25 | 5.33 | — | 7.04 | 19.90 | 4.02 | 15.59 | 12.05 | 13.36 | 7.13 | 5.77 | 4.77 | 12.74 | 5.66 | 36.91 | 3.69 | — | 5.80 | — | — | 6.27 | — | — | 8.21 | 14.07 | 7.12 | 7.95 | 4.90 | 5.47 | 4.72 | 5.32 | 5.18 | 4.95 | 6.74 |
| vi | 2.30 | 20.58 | — | 5.28 | 21.00 | 3.17 | 15.26 | 13.74 | 21.15 | 15.15 | 3.79 | 3.82 | 40.96 | 19.17 | — | 3.14 | — | 26.28 | — | — | 9.50 | — | — | 12.48 | — | 4.50 | 6.14 | 4.10 | 4.28 | 3.96 | 4.25 | 4.95 | 4.50 | 4.32 |
| cs | 6.27 | 12.14 | 42.02 | 11.25 | 31.33 | 9.77 | 38.80 | 35.27 | 38.01 | 33.38 | 30.29 | 27.23 | 42.12 | 14.42 | 43.41 | 6.05 | — | 12.17 | — | — | 12.59 | — | — | 12.89 | — | 6.79 | 8.74 | 9.39 | 9.68 | 9.28 | 9.79 | 9.48 | 8.80 | 11.23 |
| sv | 3.75 | 6.16 | — | 6.68 | 27.02 | 5.92 | 23.73 | 16.44 | 18.82 | 14.44 | 6.74 | 5.78 | 28.48 | 6.77 | 35.23 | 4.66 | — | 7.94 | — | — | 8.22 | — | — | 9.02 | — | 5.43 | 7.59 | 7.04 | 7.27 | 6.84 | 7.07 | 7.27 | 6.70 | 7.95 |
| da | 5.71 | 12.21 | 41.77 | 10.27 | 27.41 | 8.07 | 27.44 | 23.84 | 27.20 | 24.38 | 9.03 | 7.66 | 31.22 | 12.78 | 38.68 | 6.50 | — | 12.45 | — | — | 11.61 | — | — | 17.37 | — | 10.30 | 13.19 | 9.43 | 10.31 | 9.41 | 9.91 | 10.35 | 10.82 | 10.86 |
| fi | 6.34 | 11.33 | 48.72 | 11.95 | 28.57 | 7.67 | 39.36 | 34.02 | 35.71 | 30.11 | 8.44 | 7.08 | 42.04 | 10.98 | 39.90 | 6.16 | — | 13.82 | — | — | 14.05 | — | — | 16.60 | — | 6.38 | 8.39 | 13.13 | 11.73 | 12.93 | 11.79 | 10.57 | 9.89 | 12.18 |
| el | 12.84 | 17.59 | 123.63 | 13.08 | 29.36 | 8.03 | 40.66 | 39.68 | 42.50 | 41.13 | 8.54 | 8.47 | 36.84 | 17.78 | 225.57 | 7.60 | — | 12.21 | — | — | 11.74 | — | — | 9.81 | — | 10.64 | 11.75 | 12.32 | 13.32 | 10.74 | 10.50 | 11.22 | 9.79 | 14.20 |
| hu | 7.09 | 11.05 | 106.33 | 13.55 | 28.84 | 8.58 | 41.91 | 37.36 | 40.68 | 35.66 | 40.47 | 40.49 | 43.36 | 20.18 | 43.91 | 8.68 | — | 15.90 | — | — | 14.75 | — | — | 15.91 | — | 11.34 | 13.83 | 13.28 | 12.64 | 13.27 | 12.82 | 11.33 | 11.31 | 15.13 |
| ro | 6.38 | 18.85 | — | 6.97 | 27.06 | 5.11 | 30.33 | 24.98 | 27.05 | 23.26 | 27.12 | 25.70 | 33.93 | 20.26 | 39.76 | 3.52 | — | 10.57 | — | — | 8.10 | — | — | 9.10 | — | 5.34 | 7.04 | 7.49 | 7.81 | 7.51 | 7.92 | 7.34 | 6.69 | 6.68 |
| uk | 4.91 | 11.60 | — | 6.97 | 28.51 | 5.43 | 20.71 | 13.50 | 20.91 | 13.19 | 14.13 | 12.50 | 44.30 | 20.55 | 38.75 | 4.23 | — | 11.53 | — | — | 11.21 | — | — | 9.65 | — | 5.93 | 7.70 | 7.73 | 7.50 | 7.63 | 7.58 | 6.77 | 6.79 | 6.31 |
| nb | 5.27 | 9.66 | — | 9.96 | 26.39 | 6.83 | 27.04 | 20.28 | 23.08 | 19.26 | 9.59 | 7.40 | 30.92 | 10.15 | 36.79 | 6.49 | — | 10.19 | — | — | 11.31 | — | — | 14.23 | — | 8.47 | 12.22 | 9.29 | 9.60 | 9.32 | 10.15 | 10.20 | 9.85 | 11.21 |
| he | 16.70 | 38.49 | 150.21 | 19.36 | 37.77 | 27.79 | 47.70 | 44.67 | 47.50 | 48.08 | 12.88 | 12.70 | 44.52 | 38.23 | — | 10.22 | — | 45.65 | — | — | 26.39 | — | — | 13.98 | — | 10.90 | 14.11 | 14.70 | 13.97 | 14.71 | 13.99 | 14.72 | 13.26 | 22.30 |
| bg | 7.34 | 12.16 | 115.65 | 11.68 | 29.88 | 7.37 | 27.56 | 25.48 | 28.24 | 25.78 | 29.88 | 27.80 | 44.63 | 28.38 | 31.37 | 6.36 | — | 17.22 | — | — | 15.60 | — | — | 11.94 | — | 8.20 | 10.67 | 10.69 | 11.00 | 10.72 | 10.47 | 10.34 | 9.38 | 13.68 |
| hr | 6.76 | 11.67 | 44.57 | 11.69 | 31.72 | 7.77 | 34.68 | 23.16 | 33.35 | 22.21 | 23.24 | 20.64 | 39.64 | 12.90 | 41.36 | 6.34 | — | 16.04 | — | — | 17.68 | — | — | 12.96 | — | 7.82 | 10.59 | 10.93 | 11.79 | 10.72 | 11.60 | 11.19 | 10.22 | 13.02 |
| sk | 7.00 | 10.86 | — | 12.34 | 27.61 | 8.61 | 37.76 | 31.17 | 35.65 | 28.94 | 29.41 | 26.31 | 41.05 | 13.83 | 43.27 | 6.95 | — | 15.75 | — | — | 15.14 | — | — | 14.06 | — | 8.29 | 10.78 | 11.29 | 11.15 | 11.25 | 11.22 | 11.55 | 10.61 | 11.78 |
| ca | 3.61 | 6.33 | 39.55 | 6.79 | 26.34 | 3.81 | 9.30 | 10.28 | 9.11 | 8.68 | 14.08 | 12.97 | 14.74 | 10.06 | 17.30 | 4.51 | — | 9.14 | — | — | 9.37 | — | — | 8.48 | — | 6.85 | 9.58 | 6.59 | 6.11 | 6.54 | 6.06 | 5.58 | 5.14 | 6.90 |
| ms | 6.14 | 8.60 | — | 7.49 | 31.49 | 6.94 | 16.52 | 7.37 | 11.41 | 7.05 | 6.18 | 6.04 | 35.64 | 8.30 | 33.93 | 4.63 | — | 10.31 | — | — | 8.48 | — | — | 7.78 | — | 4.75 | 7.11 | 6.32 | 6.03 | 6.06 | 5.82 | 6.82 | 6.12 | 7.99 |
| fa | 18.99 | 26.61 | 106.17 | 25.55 | 34.30 | 20.72 | 32.50 | 32.07 | 35.29 | 34.09 | 30.93 | 29.80 | 41.50 | 33.58 | — | 23.03 | — | 25.51 | — | — | 25.41 | — | — | 23.89 | — | 22.84 | 27.15 | 26.10 | 25.26 | 26.40 | 25.14 | 25.95 | 24.29 | 25.59 |

## 3. Zero-shot speaker similarity (×100, higher better)

| Lang | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 74.8 | 77.2 | 83.4 | 72.4 | 82.7 | 84.6 | 70.1 | 68.5 | 80.7 | 80.0 | 66.1 | 84.0 | 13.0 | 77.3 | 85.2 | 12.8 | 10.5 | 81.4 | 83.0 | 77.0 | 80.7 | 9.0 | 77.8 | 79.9 | 14.2 | 67.0 | 68.4 | 79.0 | 70.4 | 79.1 | 70.1 | 79.8 | 72.0 | — |
| zh | 84.1 | 83.3 | 89.4 | 81.0 | 87.0 | 88.9 | 81.2 | 79.8 | 85.3 | 85.3 | 77.0 | 88.2 | 23.9 | 83.3 | 90.5 | 37.0 | 33.2 | 85.7 | 86.8 | 82.7 | 85.9 | — | 83.9 | 84.5 | — | 69.8 | 65.5 | 84.3 | 77.4 | 84.3 | 74.8 | 85.1 | 78.5 | — |
| de | 79.6 | 79.2 | 79.7 | 77.1 | 84.9 | 87.3 | 73.8 | 71.5 | 81.3 | 80.4 | 80.6 | 86.0 | 17.6 | 79.2 | 86.5 | 17.8 | — | 81.8 | 83.5 | 77.2 | 80.8 | — | — | 78.8 | 27.3 | 72.8 | 64.9 | 78.9 | 73.3 | 78.8 | 73.2 | 80.5 | 74.9 | — |
| es | 83.1 | 82.5 | 79.2 | 79.6 | 80.8 | 85.8 | 71.8 | 68.2 | 85.1 | 84.6 | 74.8 | 87.0 | 20.0 | 80.7 | 88.6 | 18.1 | 19.2 | 84.2 | 85.0 | 79.9 | 83.4 | — | — | 82.3 | 21.8 | 74.7 | 66.0 | 81.3 | 78.2 | 81.3 | 78.2 | 83.4 | 79.6 | — |
| fr | 78.6 | 77.5 | 78.8 | 75.0 | 82.2 | 84.1 | 73.1 | 71.2 | 81.8 | 81.6 | 76.4 | 84.7 | 20.8 | 77.9 | 86.5 | 19.8 | 20.7 | 81.5 | 83.5 | 77.8 | 80.5 | — | — | 77.4 | 16.1 | 70.6 | 61.1 | 79.9 | 72.5 | 79.9 | 72.2 | 80.6 | 73.9 | — |
| it | 80.8 | 79.3 | 81.7 | 77.1 | 84.6 | 87.3 | 76.0 | 75.2 | 83.2 | 82.3 | 80.6 | 86.3 | 21.0 | 78.8 | 85.9 | 21.9 | 29.8 | 82.2 | 83.2 | 78.2 | 81.2 | — | — | 80.2 | 26.8 | 73.6 | 65.9 | 78.9 | 74.2 | 78.9 | 74.3 | 80.8 | 75.5 | — |
| ja | 85.9 | 85.1 | 71.1 | 81.9 | 88.6 | 89.5 | 81.8 | 80.6 | 86.6 | 86.8 | 82.9 | 88.8 | 22.6 | 84.2 | 89.1 | 36.4 | 36.6 | 85.9 | 86.6 | 83.2 | 85.0 | — | — | 84.8 | 39.6 | 74.9 | 71.4 | 85.6 | 79.1 | 85.6 | 78.7 | 86.2 | 81.1 | — |
| ko | 84.9 | 84.1 | 70.0 | 81.8 | 86.0 | 89.4 | 80.2 | 79.1 | 85.7 | 85.3 | 83.2 | 88.2 | 20.5 | 83.8 | 88.7 | 37.1 | — | 85.2 | 85.9 | 82.4 | 84.6 | — | — | 85.4 | 33.1 | 78.3 | 64.7 | 85.0 | 80.0 | 85.0 | 79.1 | 86.2 | 80.6 | — |
| ru | 80.8 | 80.5 | 64.0 | 79.5 | 85.3 | 87.2 | 76.8 | 73.9 | 83.5 | 82.3 | 80.5 | 86.7 | 15.7 | 80.2 | 87.4 | 26.5 | — | 82.7 | 84.2 | 78.6 | 81.9 | — | — | 82.7 | — | 75.6 | 67.1 | 81.1 | 75.7 | 81.2 | 75.4 | 82.4 | 77.0 | — |
| pt | 82.0 | 81.9 | — | 80.8 | 84.8 | 88.2 | 68.5 | 68.7 | 83.4 | 83.8 | 80.1 | 86.9 | 18.5 | 80.9 | 78.5 | 25.4 | 27.4 | 82.2 | — | — | 82.0 | — | — | 85.0 | 31.8 | 73.4 | 65.1 | 81.1 | 78.6 | 81.0 | 78.5 | 82.9 | 79.4 | — |
| ar | 84.3 | 84.0 | 62.4 | 79.7 | 81.4 | 87.4 | 77.2 | 70.6 | 84.3 | 82.4 | 80.5 | 87.7 | 24.3 | 81.9 | — | 30.8 | — | 83.8 | — | — | 83.2 | — | — | 80.5 | — | 74.3 | 70.4 | 80.2 | 76.6 | 80.2 | 76.5 | 83.5 | 77.7 | — |
| hi | 83.7 | 82.3 | 64.4 | 79.3 | 84.4 | 88.8 | 77.1 | 74.5 | 75.4 | 75.5 | 81.8 | 86.7 | 17.6 | 82.4 | — | 23.8 | 29.5 | 84.0 | — | — | 82.7 | — | — | 83.8 | — | 75.0 | 72.1 | 81.0 | 77.5 | 80.8 | 76.3 | 83.8 | 78.5 | — |
| tr | 82.7 | 82.1 | — | 79.4 | 76.7 | 87.7 | 61.9 | 66.0 | 76.3 | 80.1 | 81.5 | 86.4 | 17.3 | 82.2 | 78.5 | 33.5 | — | 83.3 | — | — | 82.5 | — | — | 83.3 | — | 78.0 | 68.8 | 82.8 | 78.5 | 82.8 | 78.1 | 84.5 | 78.6 | — |
| pl | 82.3 | 80.8 | — | 78.3 | 85.0 | 87.8 | 63.7 | 67.1 | 74.4 | 76.5 | 80.6 | 86.7 | 14.6 | 81.3 | 80.7 | 29.1 | — | 83.0 | — | — | 82.3 | — | — | 80.6 | 35.7 | 76.3 | 67.8 | 81.1 | 75.7 | 81.2 | 75.5 | 82.0 | 76.7 | — |
| id | 83.3 | 81.7 | — | 82.1 | 83.2 | 89.1 | 62.8 | 68.9 | 81.6 | 84.1 | 84.0 | 88.6 | 13.6 | 82.0 | 85.1 | 34.3 | — | 84.7 | — | — | 83.9 | — | — | 83.8 | — | 78.2 | 72.1 | 82.5 | 79.5 | 82.5 | 79.5 | 84.8 | 80.5 | — |
| th | 83.3 | 85.2 | — | 83.7 | 86.5 | 90.0 | 81.4 | 79.4 | 86.3 | 86.9 | 83.8 | 88.7 | 14.5 | 84.4 | — | 41.4 | — | 86.5 | — | — | 85.2 | — | — | 85.5 | — | 78.5 | 72.2 | 83.8 | 80.0 | 83.8 | 80.1 | 85.4 | 81.0 | — |
| fil | 86.7 | 85.4 | — | 84.2 | 80.9 | 90.3 | 65.5 | 71.1 | 85.5 | 87.6 | 85.2 | 89.4 | 20.4 | 83.8 | 87.2 | 36.1 | — | 88.0 | — | — | 86.8 | — | — | 87.4 | — | 78.6 | 74.3 | 84.7 | 82.7 | 84.7 | 82.7 | 86.4 | 83.4 | — |
| nl | 79.2 | 78.6 | — | 75.4 | 80.3 | 86.2 | 63.4 | 61.6 | 80.5 | 80.9 | 78.5 | 84.9 | 16.6 | 78.1 | 78.2 | 25.6 | — | 81.4 | — | — | 80.7 | — | — | 78.7 | 25.7 | 73.2 | 64.0 | 80.8 | 73.1 | 80.6 | 73.2 | 81.9 | 74.6 | — |
| vi | 85.3 | 84.1 | — | 83.7 | 86.5 | 90.1 | 70.6 | 68.4 | 81.5 | 83.5 | 83.5 | 89.3 | 15.5 | 83.7 | — | 44.6 | — | 84.3 | — | — | 83.4 | — | — | 84.6 | — | 77.6 | 71.5 | 85.2 | 79.5 | 85.2 | 79.5 | 86.7 | 80.7 | — |
| cs | 80.8 | 79.5 | 75.0 | 79.4 | 83.3 | 87.1 | 60.8 | 63.5 | 75.2 | 74.9 | 69.7 | 80.2 | 13.8 | 78.5 | 78.7 | 32.5 | — | 82.7 | — | — | 82.0 | — | — | 80.0 | — | 75.0 | 64.9 | 80.7 | 75.2 | 80.6 | 75.2 | 82.0 | 76.4 | — |
| sv | 82.8 | 82.1 | — | 80.8 | 83.2 | 87.0 | 63.1 | 64.8 | 77.7 | 79.4 | 82.4 | 88.2 | 13.8 | 82.0 | 82.6 | 28.9 | — | 85.0 | — | — | 84.3 | — | — | 83.0 | — | 76.8 | 64.2 | 82.3 | 77.5 | 82.3 | 77.0 | 82.9 | 78.5 | — |
| da | 81.2 | 80.0 | 75.5 | 79.9 | 81.2 | 87.2 | 63.1 | 63.0 | 73.6 | 74.4 | 81.8 | 87.3 | 14.3 | 80.0 | 79.8 | 33.7 | — | 84.1 | — | — | 82.8 | — | — | 81.0 | — | 75.8 | 60.1 | 82.1 | 76.2 | 82.1 | 75.9 | 82.7 | 77.3 | — |
| fi | 84.0 | 82.5 | 76.0 | 80.7 | 83.5 | 89.2 | 60.2 | 62.5 | 79.4 | 81.3 | 83.4 | 88.0 | 11.8 | 82.8 | 85.2 | 35.0 | — | 85.2 | — | — | 84.4 | — | — | 83.4 | — | 78.1 | 64.0 | 83.2 | 77.6 | 83.1 | 77.5 | 84.5 | 78.8 | — |
| el | 80.8 | 76.7 | 68.4 | 80.0 | 82.3 | 88.3 | 67.2 | 65.5 | 64.6 | 68.2 | 81.8 | 86.5 | 17.4 | 76.3 | 71.2 | 35.7 | — | 83.0 | — | — | 82.5 | — | — | 81.0 | — | 76.2 | 70.1 | 81.5 | 75.5 | 81.4 | 75.1 | 82.9 | 76.9 | — |
| hu | 82.0 | 80.6 | 72.8 | 78.9 | 83.9 | 86.9 | 62.4 | 65.8 | 75.5 | 76.9 | 70.6 | 80.2 | 13.9 | 79.5 | 80.8 | 30.0 | — | 82.3 | — | — | 81.0 | — | — | 81.3 | — | 74.8 | 66.5 | 80.0 | 75.2 | 79.8 | 75.1 | 82.3 | 76.3 | — |
| ro | 83.3 | 80.6 | — | 81.5 | 87.2 | 89.4 | 65.5 | 69.8 | 82.3 | 82.9 | 78.2 | 86.2 | 18.1 | 80.3 | 86.0 | 38.0 | — | 84.4 | — | — | 84.0 | — | — | 83.7 | — | 78.3 | 68.1 | 82.4 | 77.8 | 82.3 | 77.7 | 84.1 | 79.2 | — |
| uk | 81.8 | 81.3 | — | 79.9 | 82.5 | 84.8 | 75.7 | 75.8 | 82.9 | 83.0 | 77.8 | 86.7 | 17.2 | 79.5 | 83.0 | 26.4 | — | 82.8 | — | — | 82.5 | — | — | 82.2 | — | 75.3 | 68.3 | 81.2 | 76.6 | 81.2 | 76.6 | 82.8 | 78.0 | — |
| nb | 80.9 | 80.6 | — | 78.9 | 73.2 | 86.2 | 57.4 | 57.6 | 74.2 | 77.7 | 81.3 | 87.0 | 9.4 | 80.2 | 78.9 | 29.4 | — | 83.7 | — | — | 82.8 | — | — | 80.7 | — | 75.4 | 54.5 | 80.7 | 74.3 | 80.8 | 73.8 | 81.7 | 75.8 | — |
| he | 79.6 | 78.5 | 71.8 | 75.7 | 83.1 | 85.9 | 70.4 | 70.6 | 71.2 | 71.1 | 79.2 | 85.4 | 19.4 | 77.9 | — | 21.5 | — | 80.5 | — | — | 80.1 | — | — | 79.0 | — | 69.7 | 67.3 | 79.1 | 73.1 | 78.9 | 72.3 | 80.5 | 74.0 | — |
| bg | 80.3 | 78.7 | 66.0 | 76.6 | 79.4 | 86.4 | 64.9 | 66.0 | 73.5 | 77.0 | 71.3 | 80.1 | 16.7 | 70.7 | 81.0 | 26.8 | — | 81.8 | — | — | 81.5 | — | — | 80.5 | — | 74.1 | 64.9 | 78.7 | 72.5 | 78.7 | 72.3 | 80.8 | 74.3 | — |
| hr | 81.2 | 81.2 | 78.4 | 79.5 | 82.8 | 87.2 | 66.1 | 71.6 | 81.5 | 81.9 | 72.2 | 83.7 | 15.4 | 81.3 | 84.1 | 25.6 | — | 83.2 | — | — | 82.5 | — | — | 81.4 | — | 72.7 | 61.5 | 80.1 | 74.7 | 80.2 | 74.6 | 81.0 | 75.8 | — |
| sk | 82.0 | 79.0 | — | 77.5 | 83.4 | 86.9 | 63.6 | 69.7 | 77.0 | 77.7 | 72.5 | 81.8 | 14.6 | 79.1 | 82.2 | 26.1 | — | 82.0 | — | — | 81.3 | — | — | 80.9 | — | 74.0 | 63.3 | 79.8 | 75.0 | 79.8 | 75.0 | 80.8 | 76.0 | — |
| ca | 79.9 | 78.5 | 81.4 | 75.8 | 80.2 | 84.7 | 69.7 | 65.5 | 80.1 | 80.3 | 73.7 | 83.1 | 14.5 | 74.8 | 83.7 | 24.0 | — | 81.1 | — | — | 80.5 | — | — | 79.4 | — | 72.1 | 57.4 | 78.8 | 73.3 | 78.8 | 73.3 | 81.2 | 74.7 | — |
| ms | 83.2 | 83.6 | — | 81.7 | 82.7 | 89.2 | 65.5 | 71.9 | 85.0 | 85.7 | 84.2 | 89.1 | 16.2 | 83.5 | 87.4 | 34.3 | — | 86.0 | — | — | 85.8 | — | — | 84.9 | — | 76.4 | 71.5 | 82.4 | 79.2 | 82.4 | 79.0 | 85.9 | 80.4 | — |
| fa | 83.7 | 84.8 | 67.2 | 82.9 | 86.3 | 91.3 | 68.7 | 66.5 | 77.1 | 80.2 | 80.8 | 88.2 | 17.9 | 76.5 | — | 38.6 | — | 87.1 | — | — | 86.6 | — | — | 86.8 | — | 79.2 | 73.8 | 83.9 | 80.8 | 83.8 | 80.8 | 85.9 | 82.0 | — |

> Preset-voice engines (`kokoro`, `neutts-2e`, `vibevoice-rt-0.5b`, `voxtral`) are not cloning the
> prompt speaker, so their speaker-similarity cells are not a meaningful signal.
> Speaker-similarity gaps below ~1 point are embedder-dependent: a held-out embedder control (vox2-only ReDimNet, no VoxBlink2 in training) preserved the system ordering overall (Kendall tau +0.91...+0.98) but reversed one published pair on `de`. Read small spk differences as ties -- see docs/controls_2026-08.md.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

## 4. Zero-shot MSR-UTMOS (higher better)

MOS predictor (`msr_utmos_mean.txt`). Empty cells = MSR-UTMOS not computed for that leaf.

| Lang | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 3.63 | 3.48 | 3.56 | 3.41 | 3.53 | 3.53 | 3.89 | 3.87 | 3.64 | 3.64 | 3.48 | 3.39 | 3.83 | 3.48 | 3.70 | 3.69 | 3.89 | 3.53 | 3.55 | 3.41 | 3.51 | 3.42 | 3.56 | 3.55 | 3.80 | 3.51 | 3.36 | 3.55 | 3.47 | 3.55 | 3.47 | 3.50 | 3.46 | 3.65 |
| zh | 3.18 | 3.24 | 3.26 | 2.95 | 3.11 | 3.12 | 3.71 | 3.70 | 3.29 | 3.30 | 3.11 | 2.99 | 3.38 | 3.24 | 3.42 | 3.36 | 3.48 | 3.15 | 3.16 | 3.01 | 3.12 | — | 3.16 | 3.22 | — | 3.10 | 3.05 | 3.28 | 3.07 | 3.27 | 3.06 | 3.21 | 3.10 | 3.29 |
| de | 3.24 | 3.25 | 3.20 | 2.98 | 3.09 | 3.08 | 3.67 | 3.68 | 3.28 | 3.22 | 3.04 | 3.03 | 3.65 | 3.19 | 3.46 | 3.55 | — | 3.09 | 3.07 | 3.01 | 3.05 | — | — | 3.11 | 3.24 | 3.33 | 3.04 | 3.38 | 3.12 | 3.38 | 3.12 | 3.19 | 3.13 | 3.32 |
| es | 3.02 | 2.99 | 3.14 | 2.87 | 2.98 | 2.91 | 3.40 | 3.39 | 3.16 | 3.13 | 2.81 | 2.84 | 3.14 | 3.00 | 3.24 | 3.27 | 2.80 | 2.93 | 2.94 | 2.85 | 2.90 | — | — | 3.05 | 3.57 | 2.93 | 2.76 | 3.01 | 2.87 | 3.02 | 2.86 | 2.96 | 2.87 | 3.14 |
| fr | 3.39 | 3.25 | 3.38 | 3.17 | 3.28 | 3.29 | 3.66 | 3.65 | 3.37 | 3.39 | 3.21 | 3.21 | 3.66 | 3.32 | 3.60 | 3.34 | 3.14 | 3.25 | 3.31 | 3.19 | 3.20 | — | — | 3.34 | 2.88 | 3.32 | 3.05 | 3.41 | 3.23 | 3.41 | 3.23 | 3.34 | 3.24 | 3.48 |
| it | 2.97 | 3.01 | 3.05 | 2.83 | 2.93 | 2.89 | 3.37 | 3.38 | 3.07 | 3.02 | 2.71 | 2.74 | 3.32 | 2.93 | 3.19 | 3.19 | 3.40 | 2.90 | 2.89 | 2.80 | 2.86 | — | — | 2.97 | 2.86 | 2.96 | 2.74 | 3.00 | 2.82 | 3.00 | 2.82 | 2.94 | 2.83 | 3.07 |
| ja | 3.33 | 3.35 | 3.05 | 3.16 | 3.32 | 3.29 | 3.66 | 3.69 | 3.40 | 3.40 | 3.21 | 3.18 | 3.31 | 3.32 | 3.45 | 3.41 | 3.75 | 3.30 | 3.26 | 3.16 | 3.28 | — | — | 3.29 | 3.21 | 3.27 | 3.25 | 3.42 | 3.22 | 3.42 | 3.21 | 3.37 | 3.25 | 3.44 |
| ko | 3.13 | 3.19 | 3.05 | 2.93 | 3.02 | 2.97 | 3.52 | 3.54 | 3.16 | 3.16 | 3.03 | 3.01 | 3.32 | 3.17 | 3.44 | 3.25 | — | 3.13 | 3.13 | 3.02 | 3.08 | — | — | 3.12 | 3.04 | 3.16 | 2.90 | 3.24 | 3.08 | 3.24 | 3.08 | 3.19 | 3.09 | 3.32 |
| ru | 3.18 | 3.09 | 2.89 | 3.01 | 3.06 | 3.02 | 3.47 | 3.51 | 3.23 | 3.15 | 2.85 | 2.94 | 3.52 | 3.11 | 3.32 | 3.23 | — | 3.02 | 3.04 | 2.93 | 2.97 | — | — | 3.13 | — | 3.06 | 2.79 | 3.15 | 2.97 | 3.14 | 2.97 | 3.05 | 2.99 | 3.33 |
| pt | 3.07 | 3.09 | — | 2.89 | 2.95 | 2.90 | 3.45 | 3.51 | 3.12 | 3.07 | 2.84 | 2.84 | 3.42 | 2.99 | 3.35 | 3.35 | 2.94 | 3.00 | — | — | 2.98 | — | — | 3.02 | 3.14 | 3.08 | 2.92 | 3.05 | 2.92 | 3.06 | 2.92 | 2.98 | 2.93 | 3.20 |
| ar | 3.30 | 3.21 | 2.99 | 3.10 | 3.20 | 3.20 | 3.55 | 3.52 | 3.37 | 3.28 | 3.03 | 3.08 | 3.41 | 3.26 | — | 3.24 | — | 3.18 | — | — | 3.13 | — | — | 3.22 | — | 3.19 | 3.14 | 3.26 | 3.13 | 3.26 | 3.12 | 3.21 | 3.13 | 3.41 |
| hi | 3.05 | 3.07 | 2.80 | 2.88 | 2.94 | 2.86 | 3.35 | 3.37 | 2.86 | 2.78 | 2.86 | 2.86 | 3.45 | 3.00 | — | 3.43 | 3.46 | 3.00 | — | — | 2.98 | — | — | 3.06 | — | 3.01 | 2.96 | 3.15 | 2.93 | 3.15 | 2.93 | 3.00 | 2.92 | 3.14 |
| tr | 3.06 | 3.05 | — | 2.95 | 2.95 | 2.94 | 3.41 | 3.43 | 3.05 | 2.97 | 2.86 | 2.90 | 3.29 | 2.99 | 3.45 | 3.37 | — | 2.98 | — | — | 2.95 | — | — | 3.04 | — | 3.06 | 2.96 | 3.11 | 2.90 | 3.11 | 2.90 | 2.94 | 2.91 | 3.23 |
| pl | 3.05 | 3.02 | — | 2.84 | 2.94 | 2.89 | 3.47 | 3.42 | 3.05 | 3.02 | 2.81 | 2.85 | 3.31 | 2.96 | 3.43 | 3.32 | — | 2.88 | — | — | 2.86 | — | — | 3.00 | 3.54 | 3.04 | 2.75 | 3.13 | 2.91 | 3.14 | 2.90 | 2.98 | 2.90 | 3.17 |
| id | 2.97 | 3.10 | — | 2.81 | 2.87 | 2.85 | 3.42 | 3.43 | 3.03 | 3.06 | 2.65 | 2.75 | 3.33 | 2.98 | 3.31 | 3.08 | — | 2.85 | — | — | 2.85 | — | — | 2.95 | — | 2.85 | 2.84 | 2.99 | 2.77 | 2.99 | 2.77 | 2.85 | 2.78 | 3.02 |
| th | 3.17 | 3.24 | — | 2.98 | 3.08 | 3.10 | 3.64 | 3.62 | 3.21 | 3.21 | 2.91 | 3.00 | 2.77 | 3.21 | — | 3.05 | — | 3.12 | — | — | 2.98 | — | — | 3.05 | — | 3.21 | 3.16 | 3.29 | 3.00 | 3.29 | 2.99 | 3.17 | 3.07 | 3.35 |
| fil | 2.97 | 3.05 | — | 2.83 | 2.83 | 2.83 | 3.44 | 3.43 | 3.01 | 3.00 | 2.72 | 2.76 | 3.32 | 2.97 | 3.30 | 3.26 | — | 2.93 | — | — | 2.94 | — | — | 2.96 | — | 2.98 | 2.90 | 3.11 | 2.83 | 3.11 | 2.84 | 2.89 | 2.82 | 3.09 |
| nl | 3.33 | 3.32 | — | 3.10 | 3.27 | 3.27 | 3.78 | 3.75 | 3.44 | 3.34 | 3.09 | 3.13 | 3.43 | 3.29 | 3.62 | 3.29 | — | 3.19 | — | — | 3.17 | — | — | 3.27 | 2.71 | 3.28 | 3.12 | 3.35 | 3.22 | 3.35 | 3.22 | 3.27 | 3.22 | 3.36 |
| vi | 3.08 | 3.14 | — | 2.80 | 3.01 | 2.99 | 3.62 | 3.57 | 3.19 | 3.14 | 2.84 | 2.88 | 3.25 | 3.09 | — | 3.09 | — | 3.07 | — | — | 3.02 | — | — | 3.00 | — | 3.14 | 2.94 | 3.22 | 2.91 | 3.22 | 2.92 | 3.08 | 2.94 | 3.21 |
| cs | 3.01 | 3.11 | 3.16 | 2.90 | 3.03 | 2.96 | 3.54 | 3.53 | 3.21 | 3.11 | 2.86 | 2.91 | 3.30 | 3.06 | 3.50 | 3.21 | — | 2.97 | — | — | 2.93 | — | — | 3.07 | — | 3.05 | 2.92 | 3.10 | 2.93 | 3.10 | 2.93 | 3.01 | 2.91 | 3.32 |
| sv | 3.33 | 3.28 | — | 3.21 | 3.23 | 3.25 | 3.67 | 3.67 | 3.38 | 3.37 | 3.09 | 3.11 | 3.36 | 3.25 | 3.43 | 3.40 | — | 3.25 | — | — | 3.21 | — | — | 3.30 | — | 3.16 | 2.97 | 3.25 | 3.23 | 3.25 | 3.23 | 3.24 | 3.20 | 3.43 |
| da | 3.42 | 3.31 | 3.37 | 3.19 | 3.27 | 3.31 | 3.78 | 3.73 | 3.45 | 3.42 | 3.20 | 3.22 | 3.35 | 3.33 | 3.61 | 3.41 | — | 3.30 | — | — | 3.26 | — | — | 3.34 | — | 3.29 | 3.05 | 3.37 | 3.28 | 3.36 | 3.27 | 3.31 | 3.29 | 3.50 |
| fi | 3.32 | 3.35 | 3.17 | 3.16 | 3.20 | 3.22 | 3.68 | 3.70 | 3.40 | 3.35 | 3.08 | 3.13 | 3.32 | 3.30 | 3.50 | 3.41 | — | 3.23 | — | — | 3.19 | — | — | 3.31 | — | 3.22 | 3.06 | 3.31 | 3.23 | 3.30 | 3.22 | 3.24 | 3.20 | 3.34 |
| el | 3.07 | 3.06 | 2.74 | 2.88 | 2.96 | 2.93 | 3.40 | 3.43 | 2.65 | 2.71 | 2.83 | 2.85 | 3.19 | 3.02 | 3.24 | 3.32 | — | 2.97 | — | — | 2.93 | — | — | 3.09 | — | 3.05 | 3.00 | 3.16 | 2.96 | 3.15 | 2.95 | 3.04 | 2.94 | 3.18 |
| hu | 3.26 | 3.26 | 3.08 | 3.12 | 3.21 | 3.21 | 3.63 | 3.63 | 3.37 | 3.36 | 2.99 | 3.06 | 3.34 | 3.18 | 3.48 | 3.57 | — | 3.23 | — | — | 3.23 | — | — | 3.31 | — | 3.18 | 3.01 | 3.26 | 3.16 | 3.26 | 3.17 | 3.20 | 3.18 | 3.41 |
| ro | 3.15 | 3.13 | — | 3.00 | 3.07 | 3.02 | 3.59 | 3.62 | 3.24 | 3.17 | 2.83 | 2.86 | 3.34 | 3.12 | 3.44 | 3.27 | — | 3.05 | — | — | 2.99 | — | — | 3.17 | — | 3.12 | 3.00 | 3.21 | 3.03 | 3.21 | 3.02 | 3.10 | 3.04 | 3.35 |
| uk | 3.25 | 3.10 | — | 3.07 | 3.14 | 3.05 | 3.50 | 3.52 | 3.27 | 3.20 | 2.98 | 3.04 | 3.26 | 3.09 | 3.46 | 3.27 | — | 3.06 | — | — | 3.03 | — | — | 3.17 | — | 3.16 | 2.90 | 3.18 | 3.07 | 3.18 | 3.07 | 3.14 | 3.07 | 3.38 |
| nb | 3.11 | 3.03 | — | 2.93 | 3.01 | 3.04 | 3.49 | 3.46 | 3.11 | 3.14 | 2.80 | 2.84 | 3.36 | 3.02 | 3.34 | 3.28 | — | 2.98 | — | — | 2.96 | — | — | 3.05 | — | 2.97 | 2.81 | 3.09 | 3.00 | 3.09 | 3.00 | 3.02 | 2.98 | 3.14 |
| he | 3.21 | 3.24 | 2.92 | 3.06 | 3.12 | 3.11 | 3.42 | 3.21 | 3.00 | 2.68 | 3.03 | 3.04 | 3.24 | 3.21 | — | 3.36 | — | 3.15 | — | — | 3.11 | — | — | 3.18 | — | 3.18 | 3.09 | 3.32 | 3.12 | 3.33 | 3.12 | 3.21 | 3.11 | 3.31 |
| bg | 3.03 | 3.02 | 2.82 | 2.90 | 2.97 | 2.89 | 3.45 | 3.44 | 3.10 | 3.04 | 2.72 | 2.82 | 3.30 | 2.97 | 3.24 | 3.21 | — | 2.88 | — | — | 2.87 | — | — | 3.00 | — | 3.03 | 2.91 | 3.08 | 2.88 | 3.09 | 2.88 | 2.97 | 2.89 | 3.14 |
| hr | 3.14 | 3.14 | 3.22 | 3.00 | 3.09 | 3.03 | 3.56 | 3.58 | 3.19 | 3.18 | 2.78 | 2.89 | 3.30 | 3.08 | 3.42 | 3.40 | — | 3.06 | — | — | 3.02 | — | — | 3.13 | — | 3.13 | 2.99 | 3.23 | 3.02 | 3.24 | 3.03 | 3.13 | 3.05 | 3.32 |
| sk | 3.06 | 3.11 | — | 2.93 | 2.99 | 2.96 | 3.50 | 3.49 | 3.18 | 3.15 | 2.78 | 2.86 | 3.32 | 3.07 | 3.34 | 3.26 | — | 2.96 | — | — | 2.93 | — | — | 3.07 | — | 3.03 | 2.87 | 3.06 | 2.94 | 3.06 | 2.94 | 2.99 | 2.94 | 3.21 |
| ca | 3.11 | 3.11 | 3.22 | 2.98 | 3.09 | 3.04 | 3.52 | 3.52 | 3.26 | 3.23 | 2.81 | 2.85 | 3.34 | 3.09 | 3.41 | 3.31 | — | 3.03 | — | — | 2.98 | — | — | 3.12 | — | 3.01 | 2.77 | 3.11 | 2.94 | 3.11 | 2.93 | 3.06 | 2.93 | 3.23 |
| ms | 2.97 | 3.05 | — | 2.80 | 2.89 | 2.86 | 3.48 | 3.49 | 3.04 | 2.99 | 2.71 | 2.78 | 3.32 | 2.94 | 3.33 | 3.23 | — | 2.86 | — | — | 2.86 | — | — | 2.95 | — | 2.95 | 2.90 | 3.02 | 2.73 | 3.02 | 2.74 | 2.87 | 2.74 | 3.12 |
| fa | 3.06 | 3.17 | 2.88 | 3.00 | 3.05 | 3.05 | 3.46 | 3.49 | 3.09 | 3.15 | 2.85 | 2.90 | 3.34 | 3.05 | — | 3.39 | — | 3.08 | — | — | 3.07 | — | — | 3.14 | — | 2.99 | 2.93 | 3.13 | 3.00 | 3.12 | 2.99 | 3.04 | 3.01 | 3.25 |

The `gt (fleurs-r)` column is real speech: the MOS predictor's anchor, not a
ceiling models must beat — read model cells against it.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### Prosody

Prosody is read on four axes; all distances are to real speech, normalised by
the real-speech split-half ceiling (×ceil: 1.0 = as close as two halves of real
speech are to each other).

- **Rhythm** — distance between the generated and real rhythm distributions in
  mHuBERT space, reported as two factors: *pace* (speaking-rate distribution) and
  *pause* (burstiness: pause and stall structure). Method and ceiling:
  [`docs/rhythm.md`](docs/rhythm.md).
- **Pitch range** — register-normalized: each utterance's F0 contour is expressed
  in semitones around its own median and summarised by its p90−p10 range, so the
  speaker's absolute register drops out and only expressiveness remains. Method:
  [`docs/pitch_range.md`](docs/pitch_range.md).
- **Characters/s** — 2-Wasserstein distance to the reference speaking-rate
  distribution (explorer speaking-rate view), on the utterances with WER ≤ 20%
  (docs/controls_2026-08.md C7).
- **MPM span: Clean set only — anchors collapse on in-the-wild audio (docs/mpm_itw_limits.md); values >1 are worse than contour-scrambled speech.** 0 = real-speech split-half floor, 1 = contour-scrambled floor. Leading/trailing silence is trimmed before scoring (docs/controls_2026-08.md C6), so spans are not comparable with releases before 2026-10-31.

On the ITW set the real target recording exists for every item, so prosody is also
measured *paired* per utterance (duration ratio, pause-count difference, pitch-range
ratio against the target): the **Dur**, **Pause Δ** and **Pitch-range |log2|**
columns of §0.

## 5. VoxBlink2 multilingual zero-shot (20 languages)

Aggregates are in §0. The `gt (voxblink)` row is real in-the-wild speech: its WER
is the ASR floor, its speaker similarity is the prompt scored against the real
target recording (the same-speaker ceiling that **SPK held-out** is read against),
and its MSR-UTMOS shows what the MOS predictor gives genuine in-the-wild audio.
zh/ja/ko/th are char-tokenized, so their "WER" is a CER.

### Per-language WER_CLIP

| lang | gt (voxblink) | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 3.82 | 1.46 | 1.66 | 5.40 | 6.57 | 16.80 | 2.01 | 2.94 | 2.52 | 2.96 | 2.63 | 2.17 | 2.68 | 1.85 | 1.49 | 2.68 | 2.05 | 0.96 | 2.43 | 3.18 | 78.51 | 3.06 | 6.61 | 7.58 | 5.55 | 2.72 | 2.09 | 3.66 | 4.09 | 3.59 | 4.52 | 3.38 | 4.05 | 6.74 |
| zh | 8.78 | 2.44 | 2.36 | 3.98 | 8.68 | 18.30 | 3.18 | 2.47 | 2.19 | 4.58 | 5.79 | 2.94 | 3.40 | 83.15 | 2.38 | 3.64 | 2.80 | 3.54 | 3.22 | 3.23 | 99.97 | 3.50 | — | 8.18 | 5.70 | — | 6.53 | 8.34 | 4.54 | 8.32 | 5.87 | 16.60 | 3.88 | 10.52 |
| de | 6.25 | 2.03 | 2.24 | 86.40 | 9.18 | 22.81 | 3.22 | 2.18 | 2.05 | 3.46 | 2.92 | 3.73 | 3.26 | 2.69 | 2.48 | 4.12 | 3.09 | — | 3.70 | 4.58 | 50.09 | 4.08 | — | — | 8.63 | 13.98 | 2.34 | 6.26 | 5.38 | 3.75 | 7.00 | 5.44 | 4.26 | 7.33 |
| es | 5.08 | 1.40 | 1.76 | 80.73 | 8.73 | 23.48 | 2.39 | 1.62 | 1.35 | 3.33 | 2.25 | 2.41 | 2.66 | 1.59 | 2.47 | 2.81 | 1.89 | 1.32 | 2.73 | 3.36 | 60.94 | 3.00 | — | — | 7.18 | 4.72 | 1.28 | 5.27 | 6.03 | 4.24 | 6.16 | 5.20 | 4.06 | 7.14 |
| fr | 5.57 | 1.85 | 2.39 | 91.45 | 8.69 | 19.29 | 2.98 | 2.17 | 1.70 | 3.64 | 2.69 | 3.39 | 3.55 | 2.28 | 2.38 | 4.41 | 2.77 | 2.31 | 4.28 | 4.50 | 49.91 | 4.88 | — | — | 9.94 | 18.82 | 2.53 | 6.91 | 6.58 | 5.54 | 7.93 | 6.59 | 4.92 | 7.77 |
| it | 6.25 | 1.55 | 2.45 | 82.26 | 7.85 | 27.95 | 3.61 | 2.11 | 1.66 | 3.95 | 2.99 | 2.65 | 2.83 | 1.72 | 3.05 | 3.93 | 2.45 | 1.50 | 4.45 | 4.97 | 47.83 | 4.23 | — | — | 7.23 | 8.32 | 1.86 | 5.53 | 4.89 | 3.99 | 5.43 | 3.75 | 5.13 | 7.00 |
| ja | 6.46 | 1.59 | 1.85 | 98.00 | 11.79 | 14.55 | 2.66 | 2.32 | 1.75 | 4.84 | 4.46 | 4.65 | 3.30 | 93.35 | 2.26 | 7.49 | 1.60 | 2.85 | 7.32 | 9.13 | 57.75 | 4.86 | — | — | 18.87 | 15.84 | 4.17 | 4.42 | 7.09 | 11.99 | 6.99 | 16.13 | 4.39 | 12.36 |
| ko | 9.02 | 2.28 | 2.58 | 98.32 | 12.87 | 14.05 | 2.92 | 2.55 | 2.39 | 6.92 | 4.78 | 3.74 | 3.88 | 88.95 | 3.03 | — | 3.09 | — | 4.71 | 5.23 | 98.41 | 4.69 | — | — | 14.58 | 13.37 | 5.05 | 7.73 | 7.05 | 11.90 | 7.10 | 18.81 | 6.25 | 20.77 |
| ru | 9.53 | 3.38 | 3.53 | 99.58 | 9.96 | 27.35 | 3.93 | 3.66 | 3.15 | 6.68 | 5.05 | 4.66 | 4.68 | 90.59 | 4.74 | 5.52 | 4.19 | — | 5.78 | 6.74 | 27.61 | 6.22 | — | — | 8.66 | — | 3.64 | 6.70 | 6.86 | 7.55 | 7.71 | 9.87 | 6.42 | 11.79 |
| pt | 5.58 | 2.00 | 2.42 | — | 8.47 | 27.55 | 3.44 | 2.90 | 2.12 | 4.84 | 4.23 | 3.98 | 3.12 | 2.46 | 3.83 | 44.05 | 2.96 | 1.50 | 4.52 | — | — | 4.82 | — | — | 8.16 | 7.57 | 2.34 | 5.06 | 5.98 | 4.94 | 6.59 | 6.07 | 5.16 | 6.71 |
| ar | 11.62 | 5.19 | 7.14 | — | 22.36 | 44.41 | 7.95 | 20.28 | 13.91 | 34.50 | 24.76 | 9.89 | 7.85 | 5.82 | 14.93 | — | 9.05 | — | 18.53 | — | — | 19.15 | — | — | 23.34 | — | 7.12 | 17.96 | 15.15 | 13.24 | 16.62 | 17.61 | 14.19 | 21.12 |
| hi | 13.34 | 7.48 | 10.28 | — | 20.57 | 47.53 | 8.84 | 15.28 | 9.89 | 63.52 | 67.34 | 8.83 | 9.24 | 7.27 | 10.97 | 99.10 | 10.84 | 7.56 | 12.55 | — | — | 11.59 | — | — | 15.04 | — | 8.87 | 16.20 | 15.12 | 13.49 | 18.11 | 16.56 | 15.11 | 19.36 |
| id | 15.18 | 7.55 | 11.04 | — | 16.74 | 32.38 | 9.47 | 15.57 | 10.12 | 16.46 | 11.17 | 9.33 | 9.32 | 39.23 | 12.87 | 51.16 | 8.61 | — | 13.41 | — | — | 14.30 | — | — | 21.23 | — | 8.52 | 15.46 | 15.79 | 13.08 | 17.62 | 14.86 | 12.03 | 19.27 |
| tr | 17.24 | 4.74 | 7.19 | — | 18.66 | 40.90 | 10.63 | 52.09 | 35.95 | 48.42 | 35.52 | 6.55 | 6.75 | 56.46 | 8.29 | 85.72 | 7.64 | — | 10.84 | — | — | 10.18 | — | — | 15.57 | — | 6.12 | 11.93 | 11.51 | 11.19 | 12.35 | 12.00 | 10.60 | 13.78 |
| tl | 12.54 | 5.21 | 10.73 | — | 16.04 | 40.38 | 10.64 | 17.89 | 11.96 | 16.36 | 10.69 | 7.29 | 7.39 | 15.82 | 12.63 | 22.43 | 9.00 | — | 11.93 | — | — | 10.43 | — | — | 18.46 | — | 6.84 | 12.11 | 15.72 | 13.09 | 18.20 | 15.89 | 14.24 | 17.62 |
| pl | 9.90 | 2.28 | 3.74 | — | 11.29 | 27.43 | 4.01 | 61.61 | 51.42 | 54.41 | 49.40 | 4.37 | 3.80 | 76.80 | 4.38 | 88.59 | 4.53 | — | 5.27 | — | — | 6.18 | — | — | 11.63 | 7.51 | 3.68 | 7.43 | 6.24 | 6.21 | 7.10 | 7.14 | 4.91 | 9.56 |
| th | 8.74 | 1.78 | 5.70 | — | 14.68 | 23.40 | 4.03 | 5.71 | 8.18 | 6.39 | 7.19 | 4.54 | 4.46 | 99.81 | 6.00 | — | 2.72 | — | 26.69 | — | — | 3.60 | — | — | 14.76 | — | 5.13 | 14.84 | 9.64 | 11.36 | 10.06 | 16.07 | 4.91 | 15.45 |
| nl | 6.51 | 2.28 | 4.74 | — | 11.30 | 22.69 | 4.01 | 16.03 | 10.23 | 13.98 | 7.38 | 4.45 | 4.53 | 13.06 | 4.59 | 55.44 | 4.28 | — | 5.35 | — | — | 5.26 | — | — | 11.99 | 19.48 | 4.30 | 14.41 | 9.51 | 8.25 | 10.75 | 11.04 | 10.96 | 13.60 |
| uk | 12.20 | 5.14 | 8.92 | — | 13.21 | 30.99 | 7.95 | 22.40 | 12.85 | 20.91 | 13.26 | 12.36 | 11.14 | 84.89 | 20.36 | 77.16 | 7.60 | — | 14.13 | — | — | 13.91 | — | — | 14.62 | — | 6.29 | 10.05 | 9.01 | 9.33 | 8.00 | 15.02 | 8.00 | 12.57 |
| vi | 10.28 | 2.40 | 26.95 | — | 15.50 | 23.02 | 2.72 | 49.68 | 48.82 | 42.63 | 27.78 | 3.08 | 4.55 | 98.05 | 23.48 | 99.36 | 4.30 | — | 24.80 | — | — | 6.54 | — | — | 29.14 | — | 4.93 | 11.62 | 8.74 | 8.25 | 8.72 | 9.31 | 6.90 | 11.00 |

### Per-language WER_L50

| lang | gt (voxblink) | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 3.77 | 1.19 | 1.56 | 3.65 | 5.43 | 12.77 | 1.77 | 2.67 | 2.37 | 2.57 | 2.47 | 1.98 | 2.62 | 1.55 | 1.49 | 2.68 | 2.05 | 0.89 | 2.31 | 2.41 | 2.45 | 2.67 | 4.31 | 4.89 | 4.25 | 2.18 | 1.97 | 2.83 | 2.88 | 2.75 | 2.58 | 1.87 | 2.16 | 1.99 |
| zh | 8.07 | 2.21 | 2.15 | 3.01 | 8.13 | 14.52 | 2.88 | 2.00 | 1.98 | 2.82 | 2.87 | 2.75 | 3.03 | 39.95 | 2.17 | 3.33 | 2.64 | 3.31 | 2.91 | 2.68 | 103.84 | 3.06 | — | 5.41 | 3.60 | — | 3.36 | 3.68 | 3.44 | 3.51 | 3.25 | 3.45 | 2.58 | 2.85 |
| de | 5.94 | 1.84 | 2.17 | 41.93 | 7.91 | 15.57 | 2.58 | 2.12 | 1.99 | 3.05 | 2.54 | 3.67 | 3.26 | 2.50 | 2.30 | 3.69 | 2.82 | — | 3.65 | 3.49 | 3.96 | 3.85 | — | — | 6.17 | 7.67 | 2.34 | 3.93 | 4.52 | 3.55 | 4.06 | 3.58 | 3.55 | 3.42 |
| es | 4.16 | 1.29 | 1.70 | 36.77 | 5.89 | 15.42 | 1.88 | 1.56 | 1.30 | 2.54 | 2.05 | 2.36 | 2.40 | 1.53 | 2.47 | 2.61 | 1.78 | 1.32 | 2.48 | 2.46 | 3.02 | 2.83 | — | — | 4.40 | 3.03 | 1.28 | 2.77 | 3.19 | 2.36 | 2.84 | 2.18 | 2.31 | 2.26 |
| fr | 5.13 | 1.85 | 2.25 | 41.65 | 6.79 | 13.22 | 2.65 | 1.85 | 1.70 | 2.86 | 2.40 | 3.13 | 3.13 | 2.03 | 2.38 | 4.04 | 2.77 | 2.31 | 4.04 | 3.79 | 4.70 | 4.25 | — | — | 6.47 | 9.41 | 2.32 | 4.10 | 4.18 | 3.87 | 3.98 | 3.62 | 3.30 | 3.43 |
| it | 5.61 | 1.42 | 2.14 | 36.77 | 6.64 | 16.80 | 2.23 | 1.98 | 1.57 | 3.29 | 2.86 | 2.65 | 2.55 | 1.72 | 2.79 | 3.75 | 2.45 | 1.50 | 3.60 | 3.91 | 4.28 | 3.83 | — | — | 5.58 | 5.41 | 1.73 | 3.29 | 3.40 | 3.23 | 3.12 | 2.77 | 2.93 | 2.56 |
| ja | 6.26 | 1.44 | 1.78 | 45.45 | 9.32 | 9.89 | 2.41 | 2.17 | 1.61 | 3.41 | 2.68 | 3.84 | 2.83 | 43.16 | 1.99 | 6.42 | 1.60 | 2.64 | 5.78 | 7.25 | 6.46 | 4.31 | — | — | 10.24 | 10.79 | 2.44 | 2.61 | 4.85 | 4.31 | 4.62 | 3.65 | 3.09 | 3.40 |
| ko | 8.51 | 1.59 | 1.80 | 111.35 | 11.10 | 10.25 | 2.49 | 2.12 | 1.62 | 3.85 | 3.61 | 3.32 | 3.46 | 39.00 | 2.48 | — | 2.53 | — | 3.40 | 3.68 | 2.88 | 3.09 | — | — | 6.75 | 8.52 | 3.95 | 4.28 | 4.96 | 4.81 | 5.03 | 4.05 | 4.30 | 3.47 |
| ru | 9.11 | 3.10 | 3.40 | 115.91 | 8.81 | 18.07 | 3.65 | 3.46 | 2.98 | 5.49 | 4.25 | 4.55 | 4.28 | 42.35 | 4.54 | 5.13 | 4.13 | — | 5.48 | 5.57 | 5.82 | 5.47 | — | — | 6.57 | — | 3.50 | 4.54 | 5.24 | 4.65 | 5.04 | 4.49 | 4.79 | 4.55 |
| pt | 5.30 | 1.78 | 2.31 | — | 6.45 | 17.97 | 2.77 | 2.53 | 1.95 | 3.87 | 3.40 | 3.65 | 3.00 | 2.19 | 3.69 | 28.08 | 2.92 | 1.47 | 3.99 | — | — | 4.50 | — | — | 5.88 | 5.09 | 2.12 | 3.57 | 3.99 | 3.75 | 3.85 | 3.46 | 3.38 | 3.17 |
| ar | 10.65 | 4.86 | 6.52 | — | 17.80 | 23.46 | 6.73 | 16.95 | 11.34 | 23.06 | 17.17 | 8.42 | 7.33 | 5.36 | 12.63 | — | 8.61 | — | 14.96 | — | — | 15.10 | — | — | 15.98 | — | 6.50 | 11.60 | 11.32 | 10.01 | 11.32 | 9.62 | 10.43 | 9.51 |
| hi | 12.11 | 6.80 | 9.26 | — | 17.49 | 27.22 | 7.87 | 11.76 | 8.18 | 21.61 | 15.92 | 7.85 | 8.37 | 6.72 | 9.82 | 35.48 | 9.94 | 7.03 | 11.47 | — | — | 10.45 | — | — | 11.56 | — | 7.99 | 10.99 | 10.63 | 9.78 | 10.46 | 9.79 | 10.43 | 9.49 |
| id | 13.54 | 6.91 | 9.78 | — | 12.95 | 20.64 | 7.74 | 12.42 | 8.47 | 12.76 | 8.50 | 7.43 | 7.86 | 25.70 | 11.00 | 31.77 | 8.16 | — | 11.35 | — | — | 12.74 | — | — | 12.28 | — | 7.75 | 10.36 | 10.64 | 9.74 | 11.01 | 8.80 | 8.65 | 8.37 |
| tr | 16.69 | 4.29 | 6.57 | — | 14.44 | 27.02 | 9.42 | 33.01 | 24.66 | 31.39 | 24.58 | 6.16 | 6.14 | 32.41 | 7.62 | 40.81 | 7.38 | — | 9.45 | — | — | 8.80 | — | — | 11.48 | — | 5.57 | 8.33 | 8.85 | 8.21 | 8.49 | 8.29 | 7.82 | 7.37 |
| tl | 11.29 | 4.79 | 9.24 | — | 12.81 | 23.77 | 9.18 | 14.54 | 10.81 | 12.70 | 9.41 | 6.41 | 6.40 | 12.63 | 11.34 | 17.01 | 8.53 | — | 9.92 | — | — | 8.87 | — | — | 12.61 | — | 6.21 | 8.95 | 9.87 | 9.25 | 9.32 | 8.69 | 8.52 | 8.06 |
| pl | 9.90 | 2.10 | 3.74 | — | 9.05 | 17.73 | 3.40 | 35.74 | 32.82 | 33.56 | 31.45 | 4.14 | 3.57 | 38.99 | 4.00 | 41.51 | 4.53 | — | 4.99 | — | — | 4.99 | — | — | 8.95 | 5.36 | 3.68 | 5.05 | 4.61 | 4.40 | 4.44 | 4.51 | 3.52 | 4.60 |
| th | 8.74 | 1.78 | 4.79 | — | 11.12 | 15.94 | 3.28 | 2.75 | 5.19 | 3.62 | 4.32 | 3.16 | 3.91 | 119.90 | 5.45 | — | 2.72 | — | 21.13 | — | — | 3.29 | — | — | 7.39 | — | 2.89 | 6.63 | 5.52 | 6.54 | 4.80 | 6.41 | 4.36 | 4.43 |
| nl | 6.51 | 2.28 | 4.11 | — | 10.36 | 15.39 | 3.56 | 13.47 | 9.65 | 11.95 | 6.57 | 4.22 | 4.53 | 10.01 | 4.28 | 34.97 | 4.28 | — | 4.79 | — | — | 4.60 | — | — | 9.07 | 14.04 | 4.01 | 7.07 | 4.90 | 5.40 | 5.07 | 4.52 | 5.70 | 4.36 |
| uk | 11.07 | 5.14 | 8.47 | — | 10.37 | 21.07 | 6.49 | 16.52 | 10.87 | 17.18 | 11.10 | 11.95 | 10.04 | 40.34 | 16.97 | 28.87 | 7.14 | — | 10.88 | — | — | 10.80 | — | — | 10.69 | — | 6.29 | 7.66 | 8.56 | 8.72 | 7.54 | 7.47 | 7.37 | 6.38 |
| vi | 10.28 | 2.40 | 14.41 | — | 14.41 | 16.05 | 2.72 | 13.29 | 14.05 | 17.52 | 13.08 | 3.08 | 4.55 | 108.90 | 13.90 | 105.61 | 4.30 | — | 18.53 | — | — | 6.54 | — | — | 15.31 | — | 3.73 | 6.95 | 5.88 | 4.24 | 6.40 | 4.91 | 4.69 | 4.31 |

### Per-language speaker similarity, held-out recording (×100)

| lang | gt (voxblink) | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 76.3 | 55.2 | 54.9 | 60.1 | 56.0 | 54.3 | 57.8 | 61.6 | 63.0 | 61.7 | 63.0 | 57.5 | 61.4 | 8.8 | 52.7 | 63.9 | 10.1 | 8.8 | 59.0 | 61.8 | 55.8 | 58.4 | 7.3 | 51.0 | 55.6 | 8.2 | 42.3 | 48.9 | 58.8 | 45.3 | 58.4 | 44.6 | 60.1 | 44.7 |
| zh | 80.6 | 68.5 | 67.3 | 71.6 | 65.5 | 67.4 | 69.2 | 64.5 | 66.0 | 71.4 | 71.9 | 69.4 | 70.9 | 19.1 | 65.4 | 74.7 | 31.1 | 27.7 | 69.0 | 70.6 | 66.6 | 68.7 | — | 61.2 | 63.2 | — | 57.1 | 57.8 | 67.4 | 53.3 | 66.6 | 49.1 | 68.7 | 52.4 |
| de | 77.9 | 61.2 | 58.7 | 54.3 | 59.7 | 61.3 | 62.9 | 56.5 | 58.1 | 64.8 | 65.4 | 61.9 | 65.0 | 14.9 | 56.7 | 67.8 | 16.6 | — | 61.3 | 63.6 | 57.9 | 61.0 | — | — | 56.0 | 17.2 | 48.3 | 46.6 | 61.1 | 48.9 | 60.4 | 48.4 | 63.1 | 48.0 |
| es | 78.9 | 62.5 | 59.5 | 55.7 | 61.1 | 58.9 | 61.7 | 54.7 | 56.5 | 64.7 | 66.0 | 59.9 | 65.5 | 13.8 | 55.1 | 68.0 | 18.5 | 16.5 | 61.8 | 63.6 | 58.9 | 61.1 | — | — | 57.9 | 19.0 | 49.7 | 48.8 | 60.9 | 50.1 | 60.7 | 49.8 | 63.1 | 49.6 |
| fr | 77.7 | 60.7 | 58.5 | 52.9 | 58.7 | 57.8 | 60.3 | 55.1 | 57.1 | 64.3 | 65.3 | 60.6 | 64.5 | 16.3 | 56.7 | 67.0 | 17.0 | 10.8 | 60.8 | 63.2 | 57.1 | 60.1 | — | — | 54.4 | 14.2 | 47.4 | 46.6 | 60.3 | 47.8 | 59.4 | 47.0 | 61.7 | 47.0 |
| it | 79.1 | 62.3 | 60.4 | 55.6 | 61.6 | 62.1 | 64.3 | 56.6 | 59.2 | 65.5 | 66.2 | 62.9 | 66.7 | 17.9 | 56.4 | 68.4 | 20.0 | 18.3 | 62.8 | 65.0 | 60.3 | 62.5 | — | — | 58.6 | 21.5 | 49.5 | 47.4 | 62.8 | 50.6 | 62.3 | 50.3 | 63.1 | 49.8 |
| ja | 80.4 | 67.1 | 65.8 | 51.0 | 64.4 | 65.0 | 66.5 | 63.1 | 65.4 | 69.0 | 70.4 | 66.5 | 69.3 | 14.5 | 63.6 | 72.1 | 31.3 | 26.1 | 65.8 | 67.4 | 62.8 | 65.1 | — | — | 57.2 | 31.7 | 54.8 | 54.5 | 66.2 | 51.6 | 66.2 | 48.9 | 68.2 | 52.1 |
| ko | 79.0 | 67.3 | 65.9 | 49.3 | 63.3 | 67.3 | 68.5 | 63.5 | 65.5 | 68.2 | 70.7 | 66.6 | 69.0 | 13.1 | 64.4 | — | 30.6 | — | 66.2 | 67.7 | 62.5 | 65.6 | — | — | 57.3 | 26.8 | 55.2 | 53.5 | 65.4 | 51.8 | 65.5 | 48.0 | 67.3 | 47.2 |
| ru | 79.2 | 64.4 | 62.6 | 42.9 | 62.6 | 62.6 | 65.3 | 58.6 | 61.2 | 66.5 | 67.9 | 64.2 | 67.2 | 10.7 | 59.2 | 69.6 | 22.5 | — | 63.8 | 66.0 | 60.7 | 63.4 | — | — | 60.1 | — | 52.5 | 49.1 | 63.9 | 51.4 | 63.2 | 50.4 | 64.8 | 49.9 |
| pt | 80.1 | 64.6 | 62.5 | — | 64.3 | 61.9 | 64.8 | 57.4 | 59.5 | 66.6 | 67.0 | 63.6 | 67.8 | 19.4 | 58.4 | 58.1 | 24.2 | 22.7 | 63.2 | — | — | 62.9 | — | — | 61.0 | 26.0 | 52.4 | 52.5 | 64.2 | 53.5 | 63.7 | 53.1 | 65.6 | 53.0 |
| ar | 79.8 | 66.1 | 64.0 | — | 62.6 | 64.6 | 67.3 | 57.8 | 58.9 | 63.2 | 66.2 | 65.3 | 69.0 | 22.8 | 59.6 | — | 27.2 | — | 63.6 | — | — | 62.9 | — | — | 55.4 | — | 53.5 | 53.2 | 63.7 | 52.1 | 62.7 | 49.7 | 64.2 | 48.3 |
| hi | 78.0 | 61.7 | 59.7 | — | 58.3 | 61.1 | 64.4 | 56.0 | 58.4 | 43.7 | 39.2 | 62.3 | 65.4 | 23.1 | 56.5 | 35.5 | 20.0 | 21.0 | 60.9 | — | — | 60.0 | — | — | 56.3 | — | 48.8 | 49.1 | 57.2 | 47.3 | 55.2 | 45.8 | 59.5 | 45.3 |
| id | 78.8 | 62.9 | 58.4 | — | 62.5 | 60.0 | 63.6 | 46.2 | 53.6 | 61.0 | 66.1 | 64.5 | 67.1 | 7.9 | 55.5 | 61.8 | 30.4 | — | 63.6 | — | — | 61.6 | — | — | 55.8 | — | 54.9 | 54.5 | 61.4 | 52.3 | 60.8 | 51.1 | 63.8 | 49.2 |
| tr | 80.2 | 65.9 | 64.7 | — | 64.3 | 64.2 | 67.6 | 46.9 | 51.6 | 59.9 | 63.3 | 66.7 | 69.4 | 11.5 | 62.2 | 59.9 | 27.9 | — | 65.4 | — | — | 64.9 | — | — | 60.8 | — | 56.1 | 55.1 | 65.3 | 54.5 | 64.7 | 54.1 | 66.3 | 53.3 |
| tl | 78.3 | 61.7 | 54.5 | — | 59.8 | 53.0 | 60.0 | 44.8 | 48.8 | 59.3 | 63.9 | 61.8 | 65.5 | 11.2 | 51.0 | 61.0 | 22.0 | — | 60.0 | — | — | 59.8 | — | — | 54.1 | — | 51.4 | 51.2 | 58.5 | 48.0 | 56.4 | 46.4 | 60.3 | 46.4 |
| pl | 80.2 | 65.8 | 63.5 | — | 65.5 | 65.7 | 68.5 | 48.2 | 52.0 | 59.2 | 59.9 | 66.5 | 69.3 | 9.0 | 61.5 | 61.5 | 25.7 | — | 66.3 | — | — | 65.3 | — | — | 59.9 | 30.3 | 56.8 | 51.0 | 65.9 | 55.4 | 65.3 | 54.4 | 67.1 | 53.8 |
| th | 81.5 | 69.3 | 67.7 | — | 66.9 | 68.9 | 70.2 | 66.4 | 65.1 | 71.3 | 71.8 | 70.1 | 71.4 | 10.9 | 66.4 | — | 36.7 | — | 68.1 | — | — | 68.1 | — | — | 63.0 | — | 58.4 | 54.5 | 66.8 | 55.5 | 66.2 | 53.4 | 68.4 | 53.1 |
| nl | 76.1 | 59.3 | 57.8 | — | 59.4 | 58.7 | 61.5 | 46.4 | 46.7 | 59.6 | 61.8 | 60.5 | 63.5 | 11.2 | 55.0 | 54.8 | 24.1 | — | 60.2 | — | — | 59.3 | — | — | 55.1 | 21.3 | 48.6 | 47.9 | 60.2 | 48.2 | 59.6 | 46.6 | 60.6 | 44.5 |
| uk | 80.8 | 65.8 | 63.5 | — | 63.4 | 63.7 | 65.7 | 59.3 | 59.8 | 67.9 | 67.8 | 65.8 | 68.9 | 11.5 | 58.4 | 68.1 | 22.0 | — | 64.8 | — | — | 64.4 | — | — | 59.3 | — | 54.6 | 50.8 | 64.9 | 54.3 | 65.2 | 51.2 | 66.8 | 52.6 |
| vi | 82.2 | 70.3 | 64.8 | — | 67.5 | 66.2 | 70.8 | 55.0 | 56.2 | 65.3 | 66.1 | 68.0 | 72.6 | 10.5 | 63.5 | 58.4 | 36.0 | — | 67.6 | — | — | 67.0 | — | — | 60.1 | — | 59.0 | 54.6 | 67.5 | 58.4 | 67.2 | 57.9 | 68.6 | 57.3 |

### Per-language speaker similarity, prompt (×100)

| lang | gt (voxblink) | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 76.3 | 65.1 | 65.3 | 73.3 | 64.8 | 66.3 | 69.9 | 73.5 | 74.7 | 73.3 | 74.7 | 69.2 | 73.7 | 8.6 | 63.3 | 78.0 | 9.8 | 8.7 | 70.4 | 73.8 | 66.4 | 69.7 | 7.2 | 58.9 | 64.7 | 8.1 | 48.7 | 57.1 | 69.6 | 52.2 | 69.2 | 51.7 | 71.7 | 51.5 |
| zh | 80.6 | 75.0 | 73.8 | 81.7 | 70.9 | 75.2 | 76.7 | 71.3 | 73.2 | 79.2 | 79.4 | 77.4 | 79.1 | 18.9 | 72.3 | 85.3 | 30.2 | 27.1 | 76.0 | 78.5 | 73.5 | 76.2 | — | 66.0 | 68.7 | — | 61.8 | 62.6 | 75.4 | 57.0 | 74.6 | 52.2 | 77.0 | 56.0 |
| de | 77.9 | 71.0 | 68.9 | 66.3 | 68.0 | 73.2 | 74.8 | 65.7 | 68.0 | 75.8 | 76.2 | 73.1 | 77.0 | 14.6 | 67.0 | 81.3 | 15.7 | — | 72.0 | 75.1 | 67.5 | 71.4 | — | — | 64.6 | 17.0 | 54.8 | 53.2 | 72.2 | 55.8 | 71.1 | 55.1 | 74.1 | 54.7 |
| es | 78.9 | 72.1 | 68.8 | 66.9 | 68.8 | 69.4 | 72.4 | 63.2 | 65.9 | 75.2 | 76.4 | 70.4 | 76.8 | 13.7 | 64.5 | 80.6 | 17.8 | 16.4 | 72.0 | 74.4 | 68.2 | 71.2 | — | — | 66.0 | 18.9 | 55.4 | 55.2 | 70.7 | 56.2 | 70.3 | 56.0 | 73.3 | 55.9 |
| fr | 77.7 | 70.0 | 67.6 | 64.8 | 66.6 | 68.8 | 71.3 | 64.2 | 66.7 | 74.9 | 75.5 | 71.2 | 76.0 | 16.0 | 66.2 | 80.6 | 15.9 | 10.7 | 71.3 | 74.2 | 66.4 | 70.3 | — | — | 62.3 | 13.8 | 53.3 | 53.0 | 70.6 | 54.1 | 69.5 | 53.2 | 72.5 | 53.2 |
| it | 79.1 | 71.6 | 69.8 | 66.8 | 69.4 | 73.0 | 75.2 | 65.3 | 68.6 | 75.9 | 76.5 | 73.3 | 78.0 | 17.7 | 65.8 | 81.3 | 19.7 | 18.1 | 73.1 | 75.6 | 69.5 | 72.5 | — | — | 66.7 | 21.3 | 55.7 | 53.8 | 73.1 | 57.2 | 72.6 | 56.8 | 73.3 | 56.0 |
| ja | 80.4 | 74.5 | 72.4 | 60.0 | 69.8 | 73.5 | 75.3 | 70.4 | 73.4 | 77.5 | 78.7 | 74.5 | 78.0 | 14.3 | 70.6 | 83.3 | 31.0 | 26.2 | 73.4 | 75.6 | 69.8 | 72.3 | — | — | 62.5 | 31.3 | 58.6 | 58.8 | 74.2 | 55.1 | 74.1 | 52.4 | 77.0 | 55.7 |
| ko | 79.0 | 73.9 | 72.6 | 58.6 | 68.9 | 75.9 | 77.1 | 70.6 | 73.1 | 76.0 | 78.8 | 74.7 | 77.8 | 13.7 | 71.4 | — | 29.0 | — | 73.5 | 75.7 | 69.3 | 72.7 | — | — | 62.6 | 25.7 | 59.6 | 57.7 | 73.5 | 55.5 | 73.5 | 51.3 | 76.0 | 51.1 |
| ru | 79.2 | 73.8 | 72.0 | 51.8 | 70.5 | 73.4 | 76.2 | 67.3 | 70.8 | 77.1 | 78.7 | 75.0 | 78.8 | 10.7 | 68.6 | 82.8 | 22.0 | — | 74.1 | 76.9 | 70.1 | 73.4 | — | — | 68.9 | — | 58.6 | 55.3 | 73.9 | 57.9 | 73.1 | 56.5 | 75.3 | 56.1 |
| pt | 80.1 | 72.9 | 70.7 | — | 71.0 | 71.2 | 74.4 | 64.9 | 67.7 | 76.0 | 76.3 | 72.8 | 77.8 | 19.2 | 66.5 | 70.1 | 23.4 | 22.4 | 72.1 | — | — | 71.6 | — | — | 68.3 | 25.4 | 57.3 | 58.1 | 73.1 | 59.0 | 72.5 | 58.5 | 74.8 | 58.3 |
| ar | 79.8 | 74.2 | 71.9 | — | 69.0 | 74.1 | 76.7 | 65.8 | 67.2 | 72.1 | 75.3 | 74.7 | 79.0 | 22.3 | 67.7 | — | 26.4 | — | 72.3 | — | — | 71.3 | — | — | 61.5 | — | 58.4 | 58.9 | 72.3 | 57.1 | 71.1 | 54.6 | 73.3 | 53.3 |
| hi | 78.0 | 71.1 | 69.1 | — | 66.1 | 71.6 | 74.9 | 64.4 | 67.5 | 50.6 | 45.5 | 72.7 | 76.6 | 22.8 | 65.4 | 43.5 | 19.5 | 21.1 | 70.7 | — | — | 69.3 | — | — | 64.1 | — | 54.2 | 55.1 | 66.0 | 52.6 | 63.5 | 51.1 | 68.9 | 50.2 |
| id | 78.8 | 71.3 | 67.5 | — | 69.0 | 69.7 | 73.5 | 54.0 | 62.6 | 71.3 | 76.3 | 73.7 | 77.2 | 7.4 | 63.8 | 74.2 | 29.4 | — | 72.1 | — | — | 70.2 | — | — | 62.2 | — | 59.2 | 59.9 | 69.0 | 57.2 | 68.2 | 55.5 | 73.0 | 53.6 |
| tr | 80.2 | 74.0 | 72.9 | — | 71.2 | 73.8 | 77.2 | 55.5 | 61.0 | 70.4 | 73.5 | 75.9 | 79.3 | 11.1 | 70.4 | 72.4 | 26.9 | — | 74.1 | — | — | 73.5 | — | — | 68.1 | — | 61.3 | 60.8 | 74.0 | 59.8 | 73.3 | 59.2 | 75.3 | 58.6 |
| tl | 78.3 | 69.1 | 63.0 | — | 66.1 | 62.8 | 70.0 | 52.6 | 57.8 | 69.0 | 73.5 | 70.6 | 75.1 | 11.2 | 59.3 | 73.3 | 21.2 | — | 68.5 | — | — | 68.1 | — | — | 60.8 | — | 56.1 | 56.6 | 66.7 | 52.6 | 64.0 | 51.0 | 68.7 | 51.0 |
| pl | 80.2 | 74.9 | 72.9 | — | 72.6 | 76.2 | 79.0 | 57.4 | 61.8 | 70.0 | 70.4 | 76.7 | 80.3 | 9.0 | 70.6 | 74.5 | 25.0 | — | 76.1 | — | — | 74.7 | — | — | 67.8 | 29.6 | 62.4 | 56.7 | 75.2 | 61.2 | 74.4 | 60.3 | 76.7 | 59.8 |
| th | 81.5 | 76.3 | 75.1 | — | 72.2 | 76.9 | 78.3 | 73.0 | 72.1 | 78.8 | 79.3 | 78.1 | 80.0 | 11.0 | 73.3 | — | 35.8 | — | 75.9 | — | — | 74.9 | — | — | 68.5 | — | 63.2 | 59.3 | 73.5 | 59.3 | 73.1 | 56.9 | 76.4 | 56.8 |
| nl | 76.1 | 68.5 | 67.2 | — | 66.8 | 70.1 | 73.1 | 56.7 | 58.2 | 71.9 | 73.5 | 71.3 | 75.0 | 11.5 | 64.5 | 69.9 | 22.7 | — | 70.3 | — | — | 68.6 | — | — | 63.6 | 20.7 | 54.6 | 53.5 | 70.2 | 53.9 | 69.3 | 52.2 | 70.3 | 50.7 |
| uk | 80.8 | 73.7 | 71.6 | — | 70.9 | 74.0 | 75.6 | 66.9 | 68.6 | 76.7 | 77.7 | 75.9 | 79.6 | 11.8 | 67.1 | 79.7 | 21.4 | — | 74.5 | — | — | 73.7 | — | — | 67.1 | — | 59.9 | 55.3 | 74.4 | 59.6 | 74.4 | 56.6 | 76.2 | 58.8 |
| vi | 82.2 | 77.0 | 72.7 | — | 72.1 | 74.3 | 78.8 | 62.4 | 64.0 | 73.7 | 73.6 | 76.4 | 80.9 | 9.9 | 71.2 | 67.9 | 36.0 | — | 74.8 | — | — | 74.1 | — | — | 65.9 | — | 63.4 | 58.4 | 74.0 | 62.9 | 73.4 | 62.3 | 75.7 | 61.4 |

> Speaker-similarity gaps below ~1 point are embedder-dependent: a held-out embedder control (vox2-only ReDimNet, no VoxBlink2 in training) preserved the system ordering overall (Kendall tau +0.91...+0.98) but reversed one published pair on `de`. Read small spk differences as ties -- see docs/controls_2026-08.md.

### Per-language MSR-UTMOS (blank = not computed)

| lang | gt (voxblink) | drama-3 | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | audio8-tts-realistic-0.6b | cosyvoice3 | gemini_3.8-flash | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | zonos3-mini-laci_embedding◇ | zonos3-mini-laci_reference† | zonos3-mini_embedding◇ | zonos3-mini_reference† | zonos3-rate30-20k_embedding | zonos3-rate30-20k_reference |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 2.47 | 3.01 | 2.93 | 2.71 | 2.69 | 3.08 | 3.11 | 3.03 | 3.07 | 3.03 | 3.07 | 2.61 | 2.53 | 3.79 | 3.00 | 3.08 | 3.59 | 3.91 | 2.88 | 2.83 | 2.75 | 2.87 | 3.42 | 2.61 | 2.64 | 3.72 | 3.11 | 2.59 | 2.95 | 2.85 | 2.95 | 2.84 | 2.92 | 2.76 |
| zh | 2.18 | 2.62 | 2.67 | 2.41 | 2.27 | 2.82 | 2.87 | 2.84 | 2.85 | 2.67 | 2.66 | 2.31 | 2.24 | 3.43 | 2.78 | 2.71 | 3.34 | 3.55 | 2.51 | 2.50 | 2.49 | 2.49 | — | 2.35 | 2.41 | — | 2.52 | 2.44 | 2.79 | 2.59 | 2.79 | 2.56 | 2.75 | 2.63 |
| de | 2.39 | 2.81 | 2.78 | 2.54 | 2.52 | 2.93 | 2.98 | 3.03 | 3.01 | 2.83 | 2.85 | 2.49 | 2.44 | 3.66 | 2.82 | 2.98 | 3.50 | — | 2.68 | 2.67 | 2.60 | 2.70 | — | — | 2.58 | 3.17 | 3.03 | 2.57 | 2.83 | 2.77 | 2.81 | 2.75 | 2.81 | 2.71 |
| es | 2.18 | 2.54 | 2.54 | 2.39 | 2.34 | 2.75 | 2.77 | 2.82 | 2.77 | 2.64 | 2.63 | 2.24 | 2.19 | 3.19 | 2.59 | 2.75 | 3.26 | 2.86 | 2.48 | 2.45 | 2.42 | 2.48 | — | — | 2.37 | 3.58 | 2.73 | 2.35 | 2.54 | 2.47 | 2.54 | 2.46 | 2.53 | 2.39 |
| fr | 2.25 | 2.66 | 2.65 | 2.43 | 2.39 | 2.82 | 2.83 | 2.86 | 2.83 | 2.69 | 2.71 | 2.30 | 2.29 | 3.65 | 2.73 | 2.83 | 3.29 | 3.19 | 2.56 | 2.50 | 2.52 | 2.56 | — | — | 2.47 | 2.83 | 2.85 | 2.49 | 2.66 | 2.62 | 2.66 | 2.62 | 2.69 | 2.57 |
| it | 2.11 | 2.50 | 2.47 | 2.28 | 2.27 | 2.71 | 2.74 | 2.76 | 2.70 | 2.55 | 2.59 | 2.15 | 2.13 | 3.36 | 2.48 | 2.68 | 3.17 | 3.44 | 2.40 | 2.34 | 2.34 | 2.40 | — | — | 2.33 | 2.85 | 2.69 | 2.37 | 2.47 | 2.42 | 2.48 | 2.40 | 2.46 | 2.37 |
| ja | 2.13 | 2.58 | 2.58 | 2.18 | 2.18 | 2.94 | 2.95 | 2.80 | 2.79 | 2.62 | 2.66 | 2.23 | 2.18 | 3.47 | 2.73 | 2.78 | 3.38 | 3.76 | 2.49 | 2.47 | 2.48 | 2.47 | — | — | 2.31 | 3.27 | 2.84 | 2.69 | 2.66 | 2.58 | 2.65 | 2.58 | 2.65 | 2.51 |
| ko | 2.38 | 2.77 | 2.73 | 2.33 | 2.32 | 2.83 | 2.86 | 2.91 | 2.90 | 2.72 | 2.78 | 2.42 | 2.41 | 3.49 | 2.82 | — | 3.27 | — | 2.66 | 2.63 | 2.63 | 2.60 | — | — | 2.46 | 3.00 | 2.79 | 2.56 | 2.85 | 2.71 | 2.83 | 2.66 | 2.87 | 2.62 |
| ru | 2.29 | 2.57 | 2.56 | 2.27 | 2.34 | 2.73 | 2.74 | 2.83 | 2.77 | 2.65 | 2.64 | 2.28 | 2.26 | 3.54 | 2.56 | 2.79 | 3.20 | — | 2.49 | 2.43 | 2.40 | 2.45 | — | — | 2.44 | — | 2.74 | 2.40 | 2.59 | 2.54 | 2.58 | 2.53 | 2.58 | 2.46 |
| pt | 2.19 | 2.58 | 2.57 | — | 2.35 | 2.77 | 2.83 | 2.84 | 2.86 | 2.64 | 2.71 | 2.26 | 2.24 | 3.22 | 2.58 | 2.80 | 3.31 | 2.97 | 2.52 | — | — | 2.52 | — | — | 2.41 | 3.13 | 2.81 | 2.45 | 2.61 | 2.51 | 2.61 | 2.51 | 2.59 | 2.48 |
| ar | 2.22 | 2.51 | 2.51 | — | 2.27 | 2.58 | 2.63 | 2.72 | 2.71 | 2.54 | 2.52 | 2.23 | 2.21 | 3.34 | 2.53 | — | 3.23 | — | 2.41 | — | — | 2.39 | — | — | 2.41 | — | 2.61 | 2.32 | 2.58 | 2.45 | 2.59 | 2.41 | 2.52 | 2.36 |
| hi | 2.23 | 2.64 | 2.63 | — | 2.33 | 2.68 | 2.74 | 2.75 | 2.80 | 2.16 | 2.02 | 2.34 | 2.26 | 2.92 | 2.65 | 2.39 | 3.37 | 3.54 | 2.57 | — | — | 2.55 | — | — | 2.50 | — | 2.82 | 2.50 | 2.75 | 2.53 | 2.75 | 2.52 | 2.63 | 2.45 |
| id | 2.04 | 2.56 | 2.56 | — | 2.23 | 2.67 | 2.72 | 2.76 | 2.80 | 2.56 | 2.63 | 2.11 | 2.07 | 3.36 | 2.63 | 2.82 | 3.11 | — | 2.43 | — | — | 2.44 | — | — | 2.28 | — | 2.62 | 2.27 | 2.51 | 2.40 | 2.51 | 2.38 | 2.54 | 2.32 |
| tr | 2.21 | 2.56 | 2.52 | — | 2.30 | 2.66 | 2.70 | 2.72 | 2.71 | 2.60 | 2.57 | 2.27 | 2.24 | 3.31 | 2.52 | 2.76 | 3.37 | — | 2.46 | — | — | 2.44 | — | — | 2.39 | — | 2.78 | 2.48 | 2.64 | 2.52 | 2.64 | 2.52 | 2.58 | 2.48 |
| tl | 2.12 | 2.74 | 2.71 | — | 2.38 | 2.82 | 2.84 | 2.88 | 2.91 | 2.66 | 2.75 | 2.33 | 2.22 | 3.44 | 2.80 | 2.84 | 3.42 | — | 2.60 | — | — | 2.56 | — | — | 2.34 | — | 2.87 | 2.48 | 2.72 | 2.61 | 2.72 | 2.58 | 2.70 | 2.53 |
| pl | 2.35 | 2.69 | 2.64 | — | 2.33 | 2.78 | 2.79 | 2.87 | 2.85 | 2.69 | 2.63 | 2.37 | 2.33 | 3.32 | 2.63 | 2.90 | 3.33 | — | 2.52 | — | — | 2.54 | — | — | 2.50 | 3.56 | 2.81 | 2.42 | 2.68 | 2.60 | 2.68 | 2.60 | 2.70 | 2.49 |
| th | 2.36 | 2.72 | 2.75 | — | 2.31 | 2.84 | 2.84 | 2.84 | 2.89 | 2.68 | 2.75 | 2.34 | 2.33 | 3.11 | 2.81 | — | 3.02 | — | 2.56 | — | — | 2.57 | — | — | 2.50 | — | 2.78 | 2.48 | 2.84 | 2.62 | 2.86 | 2.58 | 2.93 | 2.58 |
| nl | 2.23 | 2.81 | 2.74 | — | 2.40 | 2.93 | 3.00 | 3.00 | 3.05 | 2.81 | 2.82 | 2.38 | 2.31 | 3.45 | 2.75 | 2.95 | 3.31 | — | 2.67 | — | — | 2.66 | — | — | 2.48 | 2.68 | 3.03 | 2.51 | 2.73 | 2.63 | 2.73 | 2.62 | 2.71 | 2.53 |
| uk | 2.47 | 2.65 | 2.59 | — | 2.44 | 2.78 | 2.75 | 2.84 | 2.83 | 2.70 | 2.72 | 2.32 | 2.34 | 3.28 | 2.58 | 2.88 | 3.27 | — | 2.54 | — | — | 2.47 | — | — | 2.52 | — | 2.74 | 2.52 | 2.71 | 2.61 | 2.68 | 2.60 | 2.71 | 2.52 |
| vi | 2.23 | 2.52 | 2.63 | — | 2.20 | 2.67 | 2.76 | 2.81 | 2.73 | 2.63 | 2.59 | 2.23 | 2.15 | 3.29 | 2.65 | 2.57 | 3.12 | — | 2.40 | — | — | 2.42 | — | — | 2.29 | — | 2.68 | 2.35 | 2.65 | 2.51 | 2.66 | 2.53 | 2.64 | 2.50 |

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### Per-bin / per-length analysis

Per-MSR-UTMOS-bin WER_CLIP heatmaps (rows = languages, columns = MSR-UTMOS
quality bins, worst→best source audio):

![Voxblink per-bin WER_CLIP heatmaps](assets/plots/voxblink_perbin_heatmaps.png)

Same WER_CLIP bucketed by REF text-length quantiles per language:

![Voxblink per-text-length WER_CLIP heatmaps](assets/plots/voxblink_perlen_heatmaps.png)

Per-length on controlled content (`zero_shot`) vs in-the-wild (`voxblink`):

![Per-length: voxblink vs zero_shot](assets/plots/perlen_zeroshot_vs_voxblink.png)

## 6. Methodology notes

- **WER_CLIP** caps per-utterance WER at 100% before averaging — a robust mean
  insensitive to runaway hallucinations. `WER` (raw) and `WER_L50` (mean of
  utts <50% WER) live in the `.wer` summary line alongside it.
- **Tail >50%** is the share of utterances with WER above 50% (`WER_TAIL50:`
  footer): the catastrophic-failure rate that the mean hides.
- **Confidence intervals** on WER are sentence-clustered bootstraps: repeated
  FLEURS sentences (the same text read by several speakers) resample together.
- **Unscoreable generations count as maximum error** (WER_CLIP 100), rather than
  dropping out of the mean — otherwise a leaf mean is a survivor mean and the
  most degenerate output is the output that disappears from its own score. Each
  leaf's `.wer` footer reports `N_TRANSCRIBED:`/`N_IMPUTED:` so the imputed share
  is always visible; see `docs/unscoreable_2026-09.md`.
- **WER prep** is applied identically to REF and HYP: apostrophes deleted,
  lowercase, Unicode punctuation strip, and digits expanded to words with
  `num2words` (languages without a `num2words` converter keep their digits —
  `utils/text_norm.py:NUM2WORDS_LANGS`); zh/ja/ko/th scored as CER
  (`utils/run_wer.py`).
- **ASR** is routed per language and the routing is enforced: fa, hi, ko, th, vi, zh use
  `Qwen/Qwen3-ASR-1.7B` (revision `7278e1e70fe206f11671096ffdd38061171dd6e5`); every other language uses
  `openai/whisper-large-v3` (revision `06f233fe06e710322aca913c1bc4249a0d71fce1`). Both are served via vLLM
  (`scripts/serve_asr.sh`). Revisions, language baskets and the version stamp
  live in `utils/benchmark.py`; every score file carries
  `BENCHMARK_VERSION: 2` and its scorer pin, and `build_results.py`
  refuses any input whose stamp does not match.
- **Speaker similarity** = cosine × 100, ReDimNet (`PalabraAI/redimnet2` @
  `ffe8f27`, `b6/lm/vb2+vox2_v0`). On the Clean set the reference is the
  prompt. On ITW the primary score (**SPK held-out**) is against a held-out
  recording of the same speaker (`target_wav/`); the prompt similarity is kept as
  **SPK prompt**, and **copy gap** = prompt − held-out.
- **MSR-UTMOS** = sarulab-speech MSR-UTMOS (SFI HuBERT) MOS predictor, pinned
  checkpoint (`MSR_UTMOS_CKPT`).
- **Rhythm** = distance of the generated rhythm distribution (mHuBERT-147 features)
  to real speech, as pace and pause factors in units of the real-speech split-half
  ceiling (`scripts/rhythm_dist.py`, `docs/rhythm.md`); lower is better.
- **Pitch range** = 2-Wasserstein distance between the per-utterance p90−p10
  semitone-range distributions of generated and real speech, in units of the
  split-half ceiling, on a fixed 500-utterance sample per language
  (`scripts/pitch_range_dist.py`, `docs/pitch_range.md`).
- **Paired ITW prosody** = per-utterance median |log2| duration ratio (generations
  with WER ≤ 20% only), mean absolute pause-count difference (pauses = gaps ≥ 0.2 s
  between Silero VAD speech segments) and median |log2|
  pitch-range ratio of each generation against the real target recording
  (`scripts/paired_itw.py`); ITW only.
- **Characters/s distance** = 2-Wasserstein distance between generated and
  reference speaking-rate distributions on the utterances the ASR heard with
  WER ≤ 20% (`scripts/speaking_rate_dist.py`); lower is better.
- **MPM span** = distance from the real-speech split-half floor (0), normalized
  by the contour-scrambled floor (1) (`utils/score_mpm_shuffle_floor.py`), on audio
  with leading/trailing silence trimmed (`docs/controls_2026-08.md` C6). Clean
  set only — anchors collapse on in-the-wild audio (`docs/mpm_itw_limits.md`);
  values >1 are worse than contour-scrambled speech.
- **Reliability gallery** — per-language edit-distance survival/tail plates for
  the GT baselines: [`assets/reliability_gallery.html`](assets/reliability_gallery.html)
  (`scripts/run_reliability.sh`).
- Empty cells (`—`) = test set not run for that model, or score file missing.
- Refresh by re-running `scripts/results_doc/run.sh` (or just `gen_md.py` if
  only the markdown layout changed).
