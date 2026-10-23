# ZTTS2-Eval — Current Results Snapshot

Regenerated from the `results/` directory tree by
`scripts/results_doc/run.sh`. Numbers are read directly from
`wav_res_ref_text.wer`, `spk_simi_scores.txt`, and `msr_utmos_mean.txt` in
every leaf directory. **WER_CLIP** (per-utt WER capped
at 100% before averaging) is the robust mean; raw WER and WER_L50 are also
tracked. Speaker similarity is cosine × 100 (ReDimNet); MSR-UTMOS is the MOS
predictor.

> **Comparability note (2026-08 audit + methodology review):** numbers in this
> snapshot are not directly comparable to earlier snapshots: zh WER shifts on
> rescore (the traditional→simplified fold is now applied to the reference text
> as well as the hypothesis), the VoxBlink aggregate basket grew from 17 to 21
> languages (nl, uk, vi added) and then shrank to 20 on 2026-08-29 when `yo`
> was removed outright — its transcripts are Nigerian English mislabeled by
> Whisper LID, not Yoruba (see RESULTS_VOXBLINK.md) — and the coreset win%
> direction was corrected in RESULTS_CORESET.md.

## 1. Coverage matrix

| Model | zero_shot | voxblink |
|---|:-:|:-:|
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
| `cosyvoice3` | ✓ | ✓ |
| `kokoro` | ✓ | ✓ |
| `moss-tts` | ✓ | ✓ |
| `moss-tts-local` | ✓ | ✓ |
| `moss-tts-local-v1.5` | ✓ | ✓ |
| `moss-tts-v1.5` | ✓ | ✓ |
| `neutts-2e` | ✓ | ✓ |
| `vibevoice-1.5b` | ✓ | ✓ |
| `vibevoice-7b` | ✓ | ✓ |
| `vibevoice-rt-0.5b` | ✓ | ✓ |
| `whisper (fleurs-r)` | ✓ | — |
| `whisper (voxblink)` | — | ✓ |
| `zonos2_accurate` | ✓ | ✓ |
| `zonos2_expressive` | ✓ | ✓ |
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
| `fish-s2-pro` | yes | with-transcript clone | ref_text kept (server rejects no-ref-text); unseeded server defaults |
| `glm-tts` | yes | with-transcript clone | ref_text kept (default policy, no-ref-text unverified); unseeded server defaults |
| `higgs-audio-v3` | yes | with-transcript clone | ref_text kept (upstream prescribes ref_audio + ref_text); unseeded server defaults |
| `omnivoice†` | no | no-transcript clone † | --no-ref-text (reftext_policy); unseeded server defaults |
| `omnivoice-reftext` | yes | with-transcript clone | same weights as omnivoice, ref_text kept (separate key); unseeded server defaults |
| `qwen3-0.6b◇` | no | x-vector-only clone ◇ | --no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults |
| `qwen3-1.7b◇` | no | x-vector-only clone ◇ | --no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults |
| `voxcpm2†` | no | no-transcript clone † | --no-ref-text (reftext_policy); unseeded server defaults |
| `voxtral‡` | n/a | preset voice ‡ | preset mode (task_type=CustomVoice); unseeded server defaults |
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

Seeding: `zonos2_*` runs are fixed-seed (0); vLLM-Omni-served systems pass no
seed or temperature (unseeded server-default sampling); native recipes note
their sampling defaults above.

## 2. Zero-shot WER_CLIP %

| Lang | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 2.29 | 4.69 | 3.75 | 28.35 | 2.81 | 2.35 | 2.06 | 3.57 | 2.50 | 3.70 | 4.04 | 2.84 | 3.29 | 2.24 | 2.78 | 2.83 | 25.20 | 3.00 | 11.67 | 4.09 | 3.11 | 3.23 | 2.65 | 4.91 | 4.45 |
| zh | 4.01 | 4.21 | 6.26 | 33.16 | 3.33 | 3.40 | 3.07 | 5.19 | 4.62 | 5.07 | 4.71 | 91.97 | 3.40 | 4.65 | 3.71 | 3.79 | 99.75 | 3.89 | — | 4.77 | 3.72 | — | 11.84 | 16.21 | 2.99 |
| de | 3.20 | 87.58 | 5.31 | 38.42 | 3.48 | 3.18 | 2.73 | 4.72 | 3.21 | 4.58 | 4.13 | 3.53 | 8.71 | — | 4.15 | 4.93 | 19.76 | 4.23 | — | — | 5.92 | 13.31 | 3.79 | 12.09 | 4.79 |
| es | 2.22 | 73.67 | 3.36 | 41.12 | 2.37 | 2.28 | 2.04 | 3.29 | 2.45 | 3.92 | 3.40 | 2.15 | 4.38 | 2.14 | 2.76 | 2.76 | 28.72 | 2.74 | — | — | 3.25 | 4.39 | 2.34 | 7.93 | 3.11 |
| fr | 3.22 | 87.50 | 4.96 | 26.24 | 3.07 | 3.24 | 3.05 | 5.03 | 3.28 | 5.37 | 4.57 | 3.30 | 6.87 | 4.06 | 4.83 | 4.85 | 16.91 | 4.47 | — | — | 6.98 | 15.27 | 3.59 | 12.74 | 4.32 |
| it | 2.68 | 78.80 | 4.50 | 42.69 | 2.79 | 2.89 | 2.22 | 3.93 | 2.52 | 4.12 | 3.74 | 2.38 | 6.80 | 2.12 | 3.79 | 4.22 | 23.17 | 3.34 | — | — | 4.68 | 6.35 | 2.10 | 7.68 | 4.25 |
| ja | 3.10 | 97.92 | 4.98 | 17.86 | 3.02 | 5.40 | 3.46 | 7.26 | 3.63 | 4.97 | 4.35 | 95.00 | 10.65 | 3.60 | 8.62 | 10.64 | 35.74 | 6.67 | — | — | 10.52 | 26.88 | 7.13 | 9.30 | 4.06 |
| ko | 3.55 | 97.93 | 5.59 | 42.67 | 3.31 | 3.42 | 3.08 | 4.73 | 4.23 | 3.96 | 4.13 | 94.86 | 5.95 | — | 4.61 | 5.27 | 90.98 | 5.30 | — | — | 5.43 | 14.70 | 6.01 | 16.79 | 3.03 |
| ru | 4.72 | 99.73 | 6.43 | 37.91 | 4.47 | 5.80 | 4.69 | 7.33 | 5.19 | 5.83 | 5.45 | 93.28 | 6.50 | — | 6.76 | 6.96 | 17.50 | 6.63 | — | — | 6.91 | — | 4.73 | 9.37 | 5.86 |
| **avg** | **3.22** | **70.22** | **5.02** | **34.27** | **3.18** | **3.55** | **2.93** | **5.00** | **3.52** | **4.61** | **4.28** | **43.26** | **6.28** | **3.13** (6/9) | **4.67** | **5.14** | **39.75** | **4.48** | **11.67** (1/9) | **4.43** (2/9) | **5.61** | **12.02** (7/9) | **4.91** | **10.78** | **4.10** |

Averages are over whichever of the 9 languages each model covers; partial
baskets are marked `(n/9)` and are not comparable to full-coverage averages.
The `gt (fleurs-r)` column is real speech scored by the same ASR — the
achievable WER floor. zh/ja/ko are char-tokenized, so their "WER" is a
CER and the `avg` column mixes WER and CER units.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### Zero-shot WER_L50 % (mean of utts with WER < 50%, ignores catastrophic tail)

Caveat: when *every* utterance of a leaf is ≥50% WER, `utils/average_wer.py`
falls back to reporting L50 over the full set — so a cell can exceed 50 for a
model that collapsed on that language.

| Lang | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 2.29 | 4.06 | 3.45 | 21.66 | 2.47 | 2.35 | 2.06 | 2.73 | 2.35 | 2.92 | 3.10 | 2.57 | 3.21 | 2.24 | 2.63 | 2.61 | 2.68 | 2.92 | 5.65 | 3.08 | 3.02 | 3.03 | 2.47 | 3.13 | 4.41 |
| zh | 3.70 | 3.85 | 5.90 | 26.18 | 3.29 | 3.07 | 2.85 | 3.00 | 3.18 | 4.00 | 3.90 | 41.57 | 3.20 | 4.46 | 3.48 | 3.65 | 14.69 | 3.65 | — | 3.75 | 2.89 | — | 4.40 | 4.92 | 2.99 |
| de | 3.14 | 40.82 | 4.85 | 23.11 | 3.03 | 3.13 | 2.70 | 3.60 | 3.08 | 4.11 | 3.86 | 3.41 | 7.41 | — | 4.07 | 3.88 | 4.37 | 3.96 | — | — | 5.25 | 8.08 | 3.45 | 4.32 | 4.73 |
| es | 2.19 | 39.80 | 3.21 | 24.19 | 2.31 | 2.22 | 1.99 | 2.78 | 2.27 | 3.18 | 3.15 | 2.04 | 4.23 | 2.14 | 2.72 | 2.65 | 2.87 | 2.74 | — | — | 2.78 | 3.54 | 2.25 | 2.62 | 3.05 |
| fr | 3.19 | 42.44 | 4.51 | 15.92 | 3.03 | 3.24 | 3.05 | 3.67 | 3.16 | 4.35 | 4.05 | 3.23 | 6.45 | 4.06 | 4.74 | 4.31 | 4.60 | 4.34 | — | — | 5.47 | 8.58 | 3.49 | 4.29 | 4.29 |
| it | 2.63 | 40.91 | 4.12 | 23.57 | 2.38 | 2.74 | 2.15 | 3.34 | 2.52 | 3.50 | 3.23 | 2.38 | 5.40 | 2.12 | 3.60 | 3.68 | 3.76 | 3.34 | — | — | 4.05 | 5.24 | 2.10 | 2.84 | 4.25 |
| ja | 3.07 | 123.96 | 4.72 | 16.29 | 2.98 | 4.92 | 3.30 | 5.57 | 3.46 | 4.74 | 3.84 | 43.16 | 8.45 | 3.60 | 8.43 | 9.79 | 9.09 | 6.28 | — | — | 7.50 | 20.33 | 3.82 | 4.08 | 4.03 |
| ko | 3.47 | 107.52 | 5.41 | 25.15 | 3.11 | 3.26 | 3.03 | 3.63 | 3.36 | 3.60 | 3.73 | 44.62 | 4.28 | — | 4.49 | 4.80 | 6.06 | 4.52 | — | — | 4.32 | 9.74 | 5.02 | 5.05 | 2.98 |
| ru | 4.69 | 116.11 | 6.15 | 23.30 | 4.40 | 5.62 | 4.60 | 6.24 | 5.04 | 5.50 | 5.35 | 40.50 | 6.05 | — | 6.25 | 6.74 | 7.30 | 6.13 | — | — | 5.98 | — | 4.58 | 5.64 | 5.75 |

## 3. Zero-shot speaker similarity (×100, higher better)

| Lang | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 77.2 | 83.4 | 72.4 | 82.7 | 84.6 | 70.1 | 68.5 | 80.7 | 80.0 | 66.1 | 84.0 | 13.0 | 85.2 | 10.5 | 81.4 | 83.0 | 77.0 | 80.7 | 9.0 | 77.8 | 79.9 | 14.2 | 67.0 | 68.4 | 79.5 |
| zh | 83.3 | 89.4 | 81.0 | 86.9 | 88.9 | 81.2 | 79.8 | 85.3 | 85.3 | 77.0 | 88.2 | 23.9 | 90.5 | 33.2 | 85.7 | 86.8 | 82.7 | 85.9 | — | 83.9 | 84.5 | — | 69.8 | 65.5 | 68.1 |
| de | 79.2 | 79.7 | 77.1 | 84.8 | 87.3 | 73.8 | 71.5 | 81.3 | 80.4 | 80.6 | 86.0 | 17.6 | 86.5 | — | 81.8 | 83.5 | 77.2 | 80.8 | — | — | 78.8 | 27.3 | 72.8 | 64.9 | 79.5 |
| es | 82.5 | 79.2 | 79.6 | 80.7 | 85.8 | 71.8 | 68.2 | 85.1 | 84.5 | 74.8 | 87.0 | 20.0 | 88.6 | 19.2 | 84.2 | 85.0 | 79.9 | 83.4 | — | — | 82.3 | 21.8 | 74.7 | 65.9 | 68.7 |
| fr | 77.5 | 78.8 | 75.0 | 82.2 | 84.2 | 73.1 | 71.2 | 81.8 | 81.6 | 76.4 | 84.7 | 20.8 | 86.5 | 20.6 | 81.5 | 83.5 | 77.8 | 80.5 | — | — | 77.4 | 16.1 | 70.6 | 61.1 | 77.0 |
| it | 79.3 | 81.7 | 77.1 | 84.6 | 87.3 | 76.0 | 75.2 | 83.2 | 82.3 | 80.6 | 86.3 | 21.0 | 85.9 | 29.8 | 82.2 | 83.2 | 78.2 | 81.2 | — | — | 80.2 | 26.8 | 73.6 | 65.9 | 74.9 |
| ja | 85.1 | 71.1 | 81.9 | 88.6 | 89.5 | 81.8 | 80.6 | 86.6 | 86.8 | 82.9 | 88.8 | 22.6 | 89.1 | 36.6 | 85.9 | 86.6 | 83.2 | 85.0 | — | — | 84.8 | 39.6 | 74.9 | 71.4 | 69.0 |
| ko | 84.1 | 70.0 | 81.8 | 86.0 | 89.4 | 80.2 | 79.1 | 85.7 | 85.3 | 83.2 | 88.2 | 20.5 | 88.7 | — | 85.2 | 85.9 | 82.4 | 84.6 | — | — | 85.4 | 33.1 | 78.3 | 64.7 | 66.2 |
| ru | 80.5 | 64.0 | 79.5 | 85.3 | 87.2 | 76.8 | 73.9 | 83.5 | 82.3 | 80.5 | 86.7 | 15.7 | 87.4 | — | 82.7 | 84.2 | 78.6 | 81.9 | — | — | 82.7 | — | 75.6 | 67.1 | 65.7 |

> Preset-voice engines (`kokoro`, `neutts-2e`, `vibevoice-rt-0.5b`, `voxtral`) are not cloning the
> prompt speaker, so their speaker-similarity cells are not a meaningful signal.
> Speaker-similarity gaps below ~1 point are embedder-dependent: a held-out embedder control (vox2-only ReDimNet, no VoxBlink2 in training) preserved the system ordering overall (Kendall tau +0.91...+0.98) but reversed one published pair on `de`. Read small spk differences as ties -- see docs/controls_2026-08.md.
> The `gt (fleurs-r)` column is a second real clip of the same speaker — the
> same-speaker ceiling this metric can reach.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

## 4. Zero-shot MSR-UTMOS (higher better)

MOS predictor (`msr_utmos_mean.txt`). Empty cells = MSR-UTMOS not computed for that leaf.

| Lang | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† | gt (fleurs-r) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 3.48 | 3.56 | 3.41 | 3.53 | 3.53 | 3.89 | 3.87 | 3.64 | 3.64 | 3.48 | 3.39 | 3.83 | 3.70 | 3.89 | 3.53 | 3.55 | 3.41 | 3.51 | 3.42 | 3.56 | 3.55 | 3.80 | 3.51 | 3.36 | 3.65 |
| zh | 3.24 | 3.26 | 2.95 | 3.11 | 3.12 | 3.71 | 3.69 | 3.29 | 3.32 | 3.11 | 2.99 | 3.37 | 3.42 | 3.48 | 3.15 | 3.16 | 3.01 | 3.12 | — | 3.16 | 3.22 | — | 3.10 | 3.05 | 3.29 |
| de | 3.25 | 3.20 | 2.98 | 3.09 | 3.08 | 3.67 | 3.68 | 3.28 | 3.22 | 3.04 | 3.03 | 3.65 | 3.46 | — | 3.09 | 3.07 | 3.01 | 3.05 | — | — | 3.11 | 3.24 | 3.33 | 3.04 | 3.32 |
| es | 2.99 | 3.14 | 2.87 | 2.98 | 2.91 | 3.40 | 3.39 | 3.16 | 3.13 | 2.81 | 2.84 | 3.14 | 3.24 | 2.80 | 2.93 | 2.94 | 2.85 | 2.90 | — | — | 3.05 | 3.57 | 2.93 | 2.76 | 3.14 |
| fr | 3.25 | 3.38 | 3.17 | 3.28 | 3.29 | 3.66 | 3.65 | 3.37 | 3.39 | 3.21 | 3.21 | 3.66 | 3.60 | 3.14 | 3.25 | 3.31 | 3.19 | 3.20 | — | — | 3.34 | 2.88 | 3.32 | 3.05 | 3.48 |
| it | 3.01 | 3.05 | 2.83 | 2.93 | 2.89 | 3.36 | 3.38 | 3.07 | 3.02 | 2.71 | 2.74 | 3.32 | 3.19 | 3.40 | 2.90 | 2.89 | 2.80 | 2.86 | — | — | 2.97 | 2.86 | 2.96 | 2.74 | 3.07 |
| ja | 3.35 | 3.05 | 3.16 | 3.32 | 3.29 | 3.66 | 3.69 | 3.40 | 3.40 | 3.21 | 3.18 | 3.31 | 3.48 | 3.75 | 3.30 | 3.26 | 3.16 | 3.28 | — | — | 3.29 | 3.21 | 3.27 | 3.25 | 3.44 |
| ko | 3.19 | 3.05 | 2.93 | 3.02 | 2.97 | 3.52 | 3.54 | 3.19 | 3.16 | 3.03 | 3.01 | 3.40 | 3.44 | — | 3.13 | 3.13 | 3.02 | 3.08 | — | — | 3.12 | 3.04 | 3.16 | 2.90 | 3.32 |
| ru | 3.09 | 2.89 | 3.01 | 3.06 | 3.02 | 3.47 | 3.51 | 3.23 | 3.15 | 2.85 | 2.94 | 3.52 | 3.32 | — | 3.02 | 3.04 | 2.93 | 2.97 | — | — | 3.13 | — | 3.06 | 2.79 | 3.33 |

The `gt (fleurs-r)` column is real speech: the MOS predictor's anchor, not a
ceiling models must beat — read model cells against it.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### Distributional prosody

The primary distribution-matching readouts are **characters/s distance**
(2-Wasserstein distance to the reference speaking-rate distribution; lower is
better) and **MPM span** (0 = real-speech split-half floor, 1 =
contour-scrambled floor; lower is better). Both are available in the explorer's
speaking-rate and prosody-overlap views.

TTSDS2 **Pitch** is retained as an interpretable auxiliary readout: it compares
the per-utterance mean-F0 distribution against real speech (0..100; higher is
closer). Mean F0 is speaker-dependent, so read Pitch alongside speaker
similarity rather than as an independent quality axis.

**TTSDS2 Pitch — Zero-shot/en**

| metric | gt (fleurs-r) | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice | omnivoice-reftext | qwen3-0.6b | qwen3-1.7b | voxcpm2 | voxtral | cosyvoice3 | kokoro | moss-tts | moss-tts-local | moss-tts-local-v1.5 | moss-tts-v1.5 | neutts-2e | vibevoice-1.5b | vibevoice-7b | vibevoice-rt-0.5b | zonos2_accurate | zonos2_expressive |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Pitch | 97.43 | 91.13 | 94.80 | 95.67 | 94.42 | 95.48 | 93.02 | 89.81 | 88.95 | 87.62 | 94.97 | 66.53 | 94.08 | 95.12 | 93.38 | 94.60 | 78.15 | 93.94 | 95.87 | 67.44 | 89.81 | 76.96 |

**TTSDS2 Pitch — VoxBlink/en**

| metric | gt (voxblink) | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice | omnivoice-reftext | qwen3-0.6b | qwen3-1.7b | voxcpm2 | voxtral | cosyvoice3 | kokoro | moss-tts | moss-tts-local | moss-tts-local-v1.5 | moss-tts-v1.5 | neutts-2e | vibevoice-1.5b | vibevoice-7b | vibevoice-rt-0.5b | zonos2_accurate | zonos2_expressive |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Pitch | 93.84 | 79.65 | 96.39 | 86.20 | 87.00 | 85.97 | 83.13 | 78.94 | 94.24 | 60.84 | 92.28 | 62.31 | 88.11 | 86.95 | 91.10 | 88.01 | 59.71 | 93.31 | 91.19 | 54.45 | 81.82 | 88.34 |

Inspect the underlying mean-F0 distributions with
`scripts/plot_ttsds_pitch_ridgeline.py`; methodology is documented in
[`docs/ttsds_pitch.md`](docs/ttsds_pitch.md).

## 5. VoxBlink2 multilingual zero-shot (19 langs)

Mean over the aggregate languages (lower WER better). Means are over
whichever of the aggregate languages each model covers — the `langs` column
says how many; partial baskets are not comparable to full-coverage means.

> **Pruned lang**: `tl` excluded from aggregates —
> when the exclusion was made, GT-on-GT WER exceeded 20% under the pre-routing
> ASR setup, i.e. the ASR pipeline could not reliably transcribe the reference
> audio. The routed pipeline now scores gt/tl at 12.4% WER_CLIP;
> the exclusion is retained pending revalidation. Raw numbers are in the
> callout at the end of this section.

| Model | mean WER_CLIP | mean spk | mean MSR-UTMOS | langs |
|---|--:|--:|--:|--:|
| `gt (voxblink)` | 9.18 | 79.4 | 2.26 | 19/19 |
| `fish-s2-pro` | 5.81 | 70.7 | 2.63 | 19/19 |
| `glm-tts` | 71.78 | 65.6 | 2.39 | 9/19 |
| `higgs-audio-v3` | 12.59 | 69.4 | 2.34 | 19/19 |
| `omnivoice†` | 21.14 | 71.9 | 2.84 | 9/19 |
| `omnivoice-reftext` | 4.83 | 75.3 | 2.82 | 19/19 |
| `qwen3-0.6b◇` | 15.03 | 64.9 | 2.84 | 19/19 |
| `qwen3-1.7b◇` | 11.75 | 67.6 | 2.84 | 19/19 |
| `qwen3-tts-0.6b-reftext` | 18.35 | 73.7 | 2.66 | 19/19 |
| `qwen3-tts-1.7b-reftext` | 14.89 | 74.6 | 2.66 | 19/19 |
| `voxcpm2†` | 5.26 | 74.1 | 2.31 | 19/19 |
| `voxcpm2-reftext` | 5.20 | 78.0 | 2.27 | 19/19 |
| `voxtral‡` | 44.75 | 13.9 | 3.39 | 19/19 |
| `cosyvoice3` | 44.02 | 73.0 | 2.76 | 19/19 |
| `kokoro‡` | 2.73 | 18.8 | 3.40 | 8/19 |
| `moss-tts†` | 9.21 | 73.1 | 2.54 | 19/19 |
| `moss-tts-local†` | 5.03 | 75.6 | 2.54 | 9/19 |
| `moss-tts-local-v1.5†` | 63.50 | 69.0 | 2.51 | 9/19 |
| `moss-tts-v1.5†` | 4.31 | 72.2 | 2.56 | 9/19 |
| `neutts-2e‡` | 6.49 | 7.2 | 3.42 | 1/19 |
| `vibevoice-1.5b†` | 7.88 | 62.5 | 2.48 | 2/19 |
| `vibevoice-7b†` | 13.35 | 65.6 | 2.43 | 19/19 |
| `vibevoice-rt-0.5b‡` | 11.30 | 21.2 | 3.18 | 10/19 |
| `zonos2_accurate†` | 4.60 | 57.9 | 2.79 | 19/19 |
| `zonos2_expressive†` | 9.49 | 57.0 | 2.46 | 19/19 |

The `gt (voxblink)` row is real in-the-wild speech: its WER is the ASR floor,
its spk is the same-speaker ceiling, and its MSR-UTMOS shows what the MOS
predictor gives genuine in-the-wild audio. zh/ja/ko/th are char-tokenized, so
their "WER" is a CER and the mean column mixes WER and CER units.

† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · ‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. See §1b.

### Per-language WER_CLIP

| lang | gt (voxblink) | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 3.85 | 1.68 | 5.45 | 6.57 | 17.24 | 2.02 | 2.97 | 2.55 | 3.01 | 2.65 | 2.22 | 2.71 | 1.88 | 2.73 | 1.01 | 2.49 | 3.27 | 78.50 | 3.12 | 6.49 | 7.58 | 5.61 | 2.73 | 2.07 | 3.73 |
| zh | 8.77 | 2.37 | 3.96 | 8.70 | 18.72 | 3.13 | 2.48 | 2.18 | 4.54 | 5.77 | 2.92 | 3.39 | 83.11 | 3.63 | 3.48 | 3.20 | 3.24 | 99.97 | 3.48 | — | 8.18 | 5.68 | — | 6.66 | 8.40 |
| de | 6.26 | 2.24 | 86.31 | 9.20 | 23.01 | 3.24 | 2.21 | 2.11 | 3.52 | 2.95 | 3.74 | 3.31 | 2.75 | 4.11 | — | 3.74 | 4.60 | 49.97 | 4.12 | — | — | 8.64 | 14.00 | 2.34 | 6.28 |
| es | 5.12 | 1.83 | 80.77 | 8.84 | 24.02 | 2.42 | 1.66 | 1.41 | 3.40 | 2.29 | 2.51 | 2.72 | 1.64 | 2.80 | 1.36 | 2.80 | 3.38 | 61.03 | 3.07 | — | — | 7.23 | 4.80 | 1.36 | 5.35 |
| fr | 5.63 | 2.50 | 91.55 | 8.79 | 19.93 | 3.11 | 2.26 | 1.78 | 3.76 | 2.78 | 3.46 | 3.79 | 2.38 | 4.51 | 2.42 | 4.37 | 4.61 | 50.01 | 4.93 | — | — | 10.06 | 18.88 | 2.64 | 6.98 |
| it | 6.26 | 2.48 | 82.26 | 7.95 | 28.34 | 3.68 | 2.15 | 1.70 | 4.16 | 3.03 | 2.73 | 2.95 | 1.78 | 3.93 | 1.56 | 4.56 | 5.03 | 47.80 | 4.28 | — | — | 7.32 | 8.38 | 1.92 | 5.60 |
| ja | 6.44 | 1.92 | 97.96 | 11.69 | 16.44 | 2.69 | 2.33 | 1.81 | 4.81 | 4.43 | 4.64 | 3.47 | 93.11 | 7.54 | 2.87 | 7.21 | 9.14 | 58.17 | 4.79 | — | — | 18.79 | 16.08 | 4.12 | 4.43 |
| ko | 9.40 | 2.65 | 98.21 | 12.69 | 14.78 | 2.90 | 2.56 | 2.42 | 7.10 | 4.73 | 3.56 | 3.88 | 88.64 | 4.00 | — | 4.58 | 5.29 | 98.48 | 4.68 | — | — | 14.35 | 13.23 | 4.94 | 7.69 |
| ru | 9.54 | 3.58 | 99.57 | 10.06 | 27.77 | 3.96 | 3.67 | 3.15 | 6.66 | 4.98 | 4.68 | 4.68 | 90.58 | 5.51 | — | 5.78 | 6.70 | 27.61 | 6.28 | — | — | 8.77 | — | 3.62 | 6.68 |
| pt | 5.68 | 2.53 | — | 8.56 | — | 3.54 | 2.96 | 2.21 | 4.91 | 4.27 | 4.08 | 3.26 | 2.56 | 44.13 | 1.56 | 4.58 | — | — | — | — | — | 8.27 | 7.69 | 2.42 | 5.15 |
| ar | 12.27 | 7.15 | — | 22.25 | — | 7.88 | 20.10 | 13.63 | 34.16 | 24.57 | 9.85 | 7.86 | 5.74 | 99.58 | — | 18.30 | — | — | — | — | — | 23.14 | — | 7.08 | 17.88 |
| hi | 12.52 | 10.28 | — | 20.57 | — | 8.81 | 15.25 | 9.92 | 63.61 | 67.25 | 8.79 | 9.25 | 7.24 | 99.11 | 7.60 | 12.57 | — | — | — | — | — | 15.10 | — | 8.80 | 16.21 |
| id | 15.38 | 10.91 | — | 17.03 | — | 9.79 | 15.51 | 10.05 | 16.55 | 11.31 | 9.43 | 9.75 | 39.61 | 51.38 | — | 13.53 | — | — | — | — | — | 21.35 | — | 8.69 | 15.42 |
| tr | 17.27 | 7.16 | — | 18.72 | — | 10.66 | 52.00 | 36.00 | 48.43 | 35.53 | 6.65 | 6.87 | 56.47 | 85.64 | — | 10.91 | — | — | — | — | — | 15.61 | — | 6.14 | 11.93 |
| pl | 10.00 | 3.85 | — | 11.35 | — | 4.11 | 61.82 | 51.42 | 54.31 | 49.39 | 4.55 | 4.11 | 76.65 | 88.61 | — | 5.35 | — | — | — | — | — | 11.68 | 7.78 | 3.76 | 7.49 |
| th | 10.06 | 6.41 | — | 15.43 | — | 4.28 | 6.65 | 8.87 | 7.46 | 8.03 | 5.51 | 5.72 | 99.79 | 96.97 | — | 26.75 | — | — | — | — | — | 15.60 | — | 5.62 | 15.07 |
| nl | 6.51 | 4.76 | — | 11.26 | — | 4.00 | 16.34 | 10.34 | 13.91 | 7.33 | 4.43 | 4.50 | 13.01 | 55.22 | — | 5.31 | — | — | — | — | — | 12.35 | 19.41 | 4.21 | 14.27 |
| uk | 12.22 | 8.97 | — | 13.18 | — | 7.95 | 22.54 | 12.84 | 20.92 | 13.35 | 12.42 | 11.15 | 85.03 | 77.16 | — | 14.17 | — | — | — | — | — | 14.62 | — | 6.29 | 10.06 |
| vi | 11.23 | 27.18 | — | 16.37 | — | 3.58 | 50.06 | 48.78 | 43.46 | 28.19 | 3.78 | 5.35 | 98.28 | 99.71 | — | 24.70 | — | — | — | — | — | 29.43 | — | 4.72 | 11.70 |

### Per-language WER_L50

| lang | gt (voxblink) | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 3.79 | 1.58 | 3.70 | 5.43 | 12.77 | 1.78 | 2.71 | 2.40 | 2.62 | 2.50 | 2.03 | 2.62 | 1.58 | 2.73 | 0.94 | 2.37 | 2.50 | 2.57 | 2.73 | 4.24 | 4.91 | 4.28 | 2.20 | 1.96 | 2.90 |
| zh | 8.06 | 2.16 | 3.00 | 8.09 | 14.52 | 2.83 | 2.01 | 1.97 | 2.78 | 2.88 | 2.73 | 3.02 | 39.87 | 3.32 | 3.25 | 2.89 | 2.69 | 103.88 | 3.05 | — | 5.33 | 3.60 | — | 3.29 | 3.67 |
| de | 5.96 | 2.18 | 41.76 | 7.94 | 15.61 | 2.61 | 2.15 | 2.05 | 3.05 | 2.56 | 3.69 | 3.31 | 2.56 | 3.69 | — | 3.68 | 3.52 | 4.08 | 3.89 | — | — | 6.18 | 7.70 | 2.34 | 3.94 |
| es | 4.21 | 1.77 | 36.64 | 5.97 | 15.47 | 1.91 | 1.59 | 1.35 | 2.62 | 2.09 | 2.43 | 2.46 | 1.59 | 2.61 | 1.36 | 2.56 | 2.48 | 3.10 | 2.90 | — | — | 4.47 | 3.07 | 1.36 | 2.83 |
| fr | 5.20 | 2.33 | 41.47 | 6.87 | 13.29 | 2.75 | 1.92 | 1.74 | 3.00 | 2.44 | 3.12 | 3.22 | 2.13 | 4.06 | 2.38 | 4.09 | 3.87 | 4.75 | 4.24 | — | — | 6.55 | 9.51 | 2.40 | 4.15 |
| it | 5.62 | 2.17 | 36.77 | 6.75 | 16.83 | 2.30 | 2.02 | 1.60 | 3.35 | 2.90 | 2.73 | 2.67 | 1.78 | 3.75 | 1.56 | 3.72 | 3.98 | 4.32 | 3.89 | — | — | 5.68 | 5.49 | 1.79 | 3.37 |
| ja | 6.24 | 1.85 | 45.45 | 9.29 | 9.93 | 2.45 | 2.19 | 1.67 | 3.42 | 2.70 | 3.85 | 2.88 | 43.16 | 6.42 | 2.66 | 5.71 | 7.32 | 6.46 | 4.29 | — | — | 10.28 | 11.01 | 2.44 | 2.67 |
| ko | 8.44 | 1.74 | 111.18 | 10.96 | 10.09 | 2.35 | 2.13 | 1.66 | 3.54 | 3.31 | 2.99 | 3.45 | 39.80 | 3.52 | — | 3.42 | 3.76 | 2.88 | 3.10 | — | — | 6.51 | 8.35 | 3.71 | 4.11 |
| ru | 9.13 | 3.42 | 115.93 | 8.90 | 18.09 | 3.64 | 3.46 | 2.94 | 5.48 | 4.19 | 4.57 | 4.29 | 41.48 | 5.13 | — | 5.45 | 5.51 | 5.79 | 5.64 | — | — | 6.61 | — | 3.47 | 4.54 |
| pt | 5.37 | 2.43 | — | 6.55 | — | 2.87 | 2.58 | 2.04 | 3.94 | 3.45 | 3.76 | 3.14 | 2.28 | 28.14 | 1.52 | 4.02 | — | — | — | — | — | 6.00 | 5.18 | 2.21 | 3.67 |
| ar | 10.58 | 6.52 | — | 17.78 | — | 6.70 | 16.70 | 11.12 | 22.84 | 16.93 | 8.39 | 7.30 | 5.30 | 104.55 | — | 14.72 | — | — | — | — | — | 16.02 | — | 6.48 | 11.54 |
| hi | 11.71 | 9.28 | — | 17.55 | — | 7.91 | 11.72 | 8.21 | 21.66 | 15.99 | 7.85 | 8.38 | 6.72 | 35.48 | 7.07 | 11.49 | — | — | — | — | — | 11.63 | — | 7.86 | 11.03 |
| id | 13.79 | 9.70 | — | 13.10 | — | 7.69 | 12.53 | 8.55 | 12.75 | 8.59 | 7.56 | 8.09 | 25.71 | 31.79 | — | 11.49 | — | — | — | — | — | 12.50 | — | 7.94 | 10.53 |
| tr | 16.73 | 6.54 | — | 14.47 | — | 9.40 | 33.00 | 24.55 | 31.37 | 24.61 | 6.17 | 6.16 | 32.32 | 40.88 | — | 9.47 | — | — | — | — | — | 11.53 | — | 5.58 | 8.34 |
| pl | 10.00 | 3.85 | — | 9.12 | — | 3.51 | 35.71 | 32.92 | 33.60 | 31.43 | 4.32 | 3.69 | 38.58 | 41.51 | — | 5.07 | — | — | — | — | — | 9.02 | 5.42 | 3.76 | 5.10 |
| th | 9.19 | 4.99 | — | 11.29 | — | 3.40 | 3.06 | 5.44 | 3.83 | 4.83 | 3.61 | 4.16 | 119.47 | 98.56 | — | 20.73 | — | — | — | — | — | 7.87 | — | 2.56 | 6.24 |
| nl | 6.51 | 4.14 | — | 10.32 | — | 3.56 | 13.57 | 9.77 | 11.90 | 6.53 | 4.21 | 4.50 | 9.99 | 34.68 | — | 4.76 | — | — | — | — | — | 9.24 | 14.02 | 3.92 | 6.99 |
| uk | 11.10 | 8.53 | — | 10.34 | — | 6.49 | 16.68 | 10.86 | 17.19 | 11.20 | 12.00 | 10.04 | 40.34 | 28.87 | — | 10.92 | — | — | — | — | — | 10.70 | — | 6.29 | 7.67 |
| vi | 11.23 | 13.88 | — | 15.31 | — | 3.58 | 13.28 | 14.36 | 16.90 | 12.76 | 3.78 | 5.35 | 109.18 | 110.46 | — | 18.88 | — | — | — | — | — | 15.31 | — | 3.52 | 7.09 |

### Per-language speaker similarity (×100)

| lang | gt (voxblink) | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 76.4 | 65.4 | 73.3 | 64.8 | 66.1 | 70.0 | 73.5 | 74.6 | 73.3 | 74.7 | 69.2 | 73.7 | 8.6 | 78.0 | 8.7 | 70.4 | 73.8 | 66.4 | 69.7 | 7.2 | 58.9 | 64.7 | 8.1 | 48.7 | 57.0 |
| zh | 80.7 | 73.8 | 81.7 | 70.9 | 75.1 | 76.7 | 71.3 | 73.2 | 78.3 | 79.5 | 77.4 | 79.1 | 18.9 | 85.3 | 27.0 | 76.1 | 78.6 | 73.5 | 76.2 | — | 66.0 | 68.8 | — | 61.7 | 62.5 |
| de | 78.0 | 68.9 | 66.3 | 68.0 | 73.1 | 74.8 | 65.7 | 68.0 | 75.8 | 76.2 | 73.1 | 77.0 | 14.6 | 81.3 | — | 72.0 | 75.1 | 67.5 | 71.4 | — | — | 64.6 | 17.0 | 54.8 | 53.2 |
| es | 79.0 | 68.8 | 66.9 | 68.8 | 69.2 | 72.4 | 63.2 | 66.0 | 75.2 | 76.4 | 70.4 | 76.8 | 13.8 | 80.6 | 16.4 | 72.0 | 74.5 | 68.3 | 71.2 | — | — | 66.1 | 18.9 | 55.4 | 55.2 |
| fr | 77.8 | 67.6 | 64.8 | 66.6 | 68.6 | 71.4 | 64.1 | 66.6 | 74.9 | 75.5 | 71.2 | 75.9 | 15.9 | 80.6 | 10.7 | 71.3 | 74.3 | 66.5 | 70.4 | — | — | 62.4 | 13.8 | 53.3 | 53.0 |
| it | 79.1 | 69.8 | 66.8 | 69.3 | 72.7 | 75.2 | 65.3 | 68.6 | 75.8 | 76.5 | 73.3 | 78.0 | 17.6 | 81.3 | 18.1 | 73.0 | 75.6 | 69.5 | 72.4 | — | — | 66.7 | 21.4 | 55.7 | 53.8 |
| ja | 80.7 | 72.6 | 60.0 | 69.9 | 73.1 | 75.4 | 70.5 | 73.5 | 77.6 | 78.8 | 74.6 | 78.2 | 14.4 | 83.4 | 26.1 | 73.5 | 75.7 | 70.0 | 72.4 | — | — | 62.7 | 31.2 | 58.7 | 58.9 |
| ko | 79.1 | 72.7 | 58.5 | 68.9 | 75.6 | 77.2 | 70.6 | 73.1 | 75.8 | 78.9 | 74.7 | 77.9 | 13.7 | 83.5 | — | 73.7 | 75.8 | 69.4 | 72.8 | — | — | 62.7 | 25.8 | 59.6 | 57.7 |
| ru | 79.3 | 72.1 | 51.8 | 70.5 | 73.2 | 76.2 | 67.3 | 70.8 | 77.1 | 78.5 | 75.0 | 78.8 | 10.7 | 82.8 | — | 74.1 | 76.9 | 70.2 | 73.4 | — | — | 68.9 | — | 58.6 | 55.3 |
| pt | 80.2 | 70.8 | — | 71.0 | — | 74.4 | 64.9 | 67.8 | 76.4 | 76.5 | 72.8 | 77.8 | 19.3 | 70.1 | 22.4 | 72.1 | — | — | — | — | — | 68.3 | 25.3 | 57.3 | 58.1 |
| ar | 79.5 | 72.1 | — | 69.2 | — | 76.8 | 65.8 | 67.2 | 72.3 | 75.5 | 74.8 | 79.2 | 22.3 | 45.1 | — | 72.4 | — | — | — | — | — | 61.8 | — | 58.5 | 59.0 |
| hi | 78.1 | 69.1 | — | 66.1 | — | 74.9 | 64.4 | 67.5 | 50.6 | 45.6 | 72.7 | 76.6 | 22.8 | 43.4 | 21.1 | 70.7 | — | — | — | — | — | 64.2 | — | 54.2 | 55.1 |
| id | 79.0 | 67.7 | — | 69.1 | — | 73.6 | 53.9 | 62.7 | 72.3 | 76.5 | 73.7 | 77.3 | 7.5 | 74.2 | — | 72.1 | — | — | — | — | — | 62.5 | — | 59.2 | 60.0 |
| tr | 80.2 | 72.9 | — | 71.2 | — | 77.2 | 55.5 | 61.0 | 71.0 | 73.6 | 75.9 | 79.3 | 11.1 | 72.4 | — | 74.1 | — | — | — | — | — | 68.2 | — | 61.3 | 60.8 |
| pl | 80.3 | 72.9 | — | 72.6 | — | 79.0 | 57.4 | 61.8 | 70.7 | 70.4 | 76.7 | 80.3 | 8.9 | 74.5 | — | 76.1 | — | — | — | — | — | 67.9 | 29.5 | 62.4 | 56.6 |
| th | 81.8 | 75.2 | — | 72.3 | — | 78.3 | 73.0 | 72.1 | 79.9 | 79.5 | 78.0 | 80.0 | 11.0 | 52.5 | — | 75.9 | — | — | — | — | — | 68.7 | — | 63.1 | 58.7 |
| nl | 76.2 | 67.3 | — | 66.9 | — | 73.1 | 56.6 | 58.1 | 72.3 | 73.6 | 71.3 | 75.0 | 11.5 | 70.0 | — | 70.3 | — | — | — | — | — | 63.6 | 20.7 | 54.6 | 53.4 |
| uk | 80.8 | 71.6 | — | 70.9 | — | 75.6 | 66.9 | 68.6 | 76.7 | 78.6 | 75.9 | 79.6 | 11.8 | 79.7 | — | 74.5 | — | — | — | — | — | 67.1 | — | 59.9 | 55.3 |
| vi | 82.3 | 72.8 | — | 72.2 | — | 78.8 | 62.4 | 64.0 | 73.8 | 73.5 | 76.5 | 81.0 | 10.0 | 67.9 | — | 75.0 | — | — | — | — | — | 66.0 | — | 63.5 | 58.5 |

> Speaker-similarity gaps below ~1 point are embedder-dependent: a held-out embedder control (vox2-only ReDimNet, no VoxBlink2 in training) preserved the system ordering overall (Kendall tau +0.91...+0.98) but reversed one published pair on `de`. Read small spk differences as ties -- see docs/controls_2026-08.md.

### Per-language MSR-UTMOS (blank = not computed)

| lang | gt (voxblink) | fish-s2-pro | glm-tts | higgs-audio-v3 | omnivoice† | omnivoice-reftext | qwen3-0.6b◇ | qwen3-1.7b◇ | qwen3-tts-0.6b-reftext | qwen3-tts-1.7b-reftext | voxcpm2† | voxcpm2-reftext | voxtral‡ | cosyvoice3 | kokoro‡ | moss-tts† | moss-tts-local† | moss-tts-local-v1.5† | moss-tts-v1.5† | neutts-2e‡ | vibevoice-1.5b† | vibevoice-7b† | vibevoice-rt-0.5b‡ | zonos2_accurate† | zonos2_expressive† |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| en | 2.47 | 2.93 | 2.71 | 2.69 | 3.07 | 3.11 | 3.03 | 3.07 | 3.03 | 3.07 | 2.61 | 2.53 | 3.79 | 3.08 | 3.91 | 2.88 | 2.83 | 2.75 | 2.87 | 3.42 | 2.61 | 2.64 | 3.72 | 3.11 | 2.60 |
| zh | 2.18 | 2.67 | 2.41 | 2.27 | 2.81 | 2.87 | 2.84 | 2.85 | 2.68 | 2.66 | 2.31 | 2.24 | 3.44 | 2.72 | 3.55 | 2.51 | 2.50 | 2.49 | 2.49 | — | 2.35 | 2.41 | — | 2.52 | 2.44 |
| de | 2.39 | 2.78 | 2.54 | 2.52 | 2.93 | 2.98 | 3.03 | 3.01 | 2.83 | 2.85 | 2.49 | 2.43 | 3.66 | 2.98 | — | 2.68 | 2.67 | 2.60 | 2.70 | — | — | 2.58 | 3.17 | 3.03 | 2.57 |
| es | 2.18 | 2.54 | 2.39 | 2.34 | 2.74 | 2.78 | 2.82 | 2.77 | 2.64 | 2.64 | 2.24 | 2.19 | 3.19 | 2.75 | 2.86 | 2.48 | 2.45 | 2.42 | 2.48 | — | — | 2.38 | 3.58 | 2.73 | 2.35 |
| fr | 2.24 | 2.65 | 2.43 | 2.39 | 2.82 | 2.83 | 2.86 | 2.83 | 2.69 | 2.71 | 2.30 | 2.29 | 3.65 | 2.83 | 3.19 | 2.56 | 2.50 | 2.52 | 2.56 | — | — | 2.46 | 2.83 | 2.85 | 2.49 |
| it | 2.11 | 2.47 | 2.28 | 2.27 | 2.71 | 2.73 | 2.76 | 2.70 | 2.55 | 2.59 | 2.15 | 2.13 | 3.36 | 2.68 | 3.44 | 2.40 | 2.34 | 2.34 | 2.40 | — | — | 2.33 | 2.85 | 2.69 | 2.37 |
| ja | 2.13 | 2.57 | 2.17 | 2.17 | 2.93 | 2.95 | 2.79 | 2.78 | 2.61 | 2.65 | 2.22 | 2.17 | 3.45 | 2.77 | 3.76 | 2.48 | 2.47 | 2.47 | 2.46 | — | — | 2.31 | 3.27 | 2.84 | 2.69 |
| ko | 2.38 | 2.73 | 2.33 | 2.33 | 2.84 | 2.86 | 2.92 | 2.90 | 2.71 | 2.78 | 2.42 | 2.42 | 3.49 | 2.97 | — | 2.66 | 2.63 | 2.63 | 2.60 | — | — | 2.46 | 3.00 | 2.79 | 2.56 |
| ru | 2.29 | 2.56 | 2.27 | 2.34 | 2.72 | 2.74 | 2.82 | 2.77 | 2.66 | 2.64 | 2.28 | 2.26 | 3.59 | 2.79 | — | 2.49 | 2.43 | 2.40 | 2.45 | — | — | 2.43 | — | 2.74 | 2.40 |
| pt | 2.19 | 2.57 | — | 2.35 | — | 2.83 | 2.83 | 2.85 | 2.65 | 2.71 | 2.26 | 2.24 | 3.22 | 2.80 | 2.97 | 2.52 | — | — | — | — | — | 2.41 | 3.13 | 2.81 | 2.45 |
| ar | 2.22 | 2.51 | — | 2.27 | — | 2.62 | 2.72 | 2.71 | 2.54 | 2.52 | 2.22 | 2.21 | 3.34 | 2.40 | — | 2.41 | — | — | — | — | — | 2.40 | — | 2.60 | 2.32 |
| hi | 2.23 | 2.63 | — | 2.33 | — | 2.74 | 2.76 | 2.80 | 2.16 | 2.02 | 2.34 | 2.26 | 2.92 | 2.39 | 3.54 | 2.57 | — | — | — | — | — | 2.50 | — | 2.82 | 2.50 |
| id | 2.04 | 2.56 | — | 2.22 | — | 2.71 | 2.76 | 2.80 | 2.58 | 2.63 | 2.11 | 2.07 | 3.35 | 2.81 | — | 2.43 | — | — | — | — | — | 2.28 | — | 2.62 | 2.26 |
| tr | 2.21 | 2.52 | — | 2.30 | — | 2.70 | 2.72 | 2.71 | 2.61 | 2.57 | 2.27 | 2.24 | 3.31 | 2.76 | — | 2.46 | — | — | — | — | — | 2.39 | — | 2.78 | 2.48 |
| pl | 2.35 | 2.64 | — | 2.34 | — | 2.79 | 2.87 | 2.86 | 2.71 | 2.63 | 2.37 | 2.33 | 3.32 | 2.90 | — | 2.53 | — | — | — | — | — | 2.50 | 3.56 | 2.81 | 2.41 |
| th | 2.36 | 2.75 | — | 2.30 | — | 2.84 | 2.84 | 2.89 | 2.69 | 2.74 | 2.34 | 2.33 | 3.16 | 2.47 | — | 2.56 | — | — | — | — | — | 2.49 | — | 2.79 | 2.48 |
| nl | 2.23 | 2.74 | — | 2.40 | — | 2.99 | 3.00 | 3.05 | 2.81 | 2.82 | 2.37 | 2.31 | 3.45 | 2.95 | — | 2.66 | — | — | — | — | — | 2.48 | 2.68 | 3.03 | 2.51 |
| uk | 2.47 | 2.59 | — | 2.44 | — | 2.75 | 2.84 | 2.83 | 2.70 | 2.72 | 2.32 | 2.34 | 3.30 | 2.88 | — | 2.54 | — | — | — | — | — | 2.52 | — | 2.74 | 2.52 |
| vi | 2.22 | 2.62 | — | 2.20 | — | 2.75 | 2.80 | 2.73 | 2.63 | 2.58 | 2.23 | 2.14 | 3.32 | 2.57 | — | 2.39 | — | — | — | — | — | 2.29 | — | 2.67 | 2.35 |

### Per-bin / per-length analysis

Per-MSR-UTMOS-bin WER_CLIP heatmaps (rows = languages, columns = MSR-UTMOS
quality bins, worst→best source audio):

![Voxblink per-bin WER_CLIP heatmaps](assets/plots/voxblink_perbin_heatmaps.png)

Same WER_CLIP bucketed by REF text-length quantiles per language:

![Voxblink per-text-length WER_CLIP heatmaps](assets/plots/voxblink_perlen_heatmaps.png)

Per-length on controlled content (`zero_shot`) vs in-the-wild (`voxblink`):

![Per-length: voxblink vs zero_shot](assets/plots/perlen_zeroshot_vs_voxblink.png)

### Excluded-langs callout: `tl` (informational, not in aggregates)

Retained so the pruning decision is auditable. The exclusion reflects the ASR
pipeline's historical unreliability on the reference audio (not the TTS
models); compare the gt row here against the model rows before reading them.

| Model | tl |
|---|--:|
| `gt (voxblink)` | 12.44 |
| `fish-s2-pro` | 10.72 |
| `glm-tts` | — |
| `higgs-audio-v3` | 16.09 |
| `omnivoice†` | — |
| `omnivoice-reftext` | 10.40 |
| `qwen3-0.6b◇` | 17.72 |
| `qwen3-1.7b◇` | 11.81 |
| `qwen3-tts-0.6b-reftext` | 16.30 |
| `qwen3-tts-1.7b-reftext` | 10.64 |
| `voxcpm2†` | 7.28 |
| `voxcpm2-reftext` | 7.40 |
| `voxtral‡` | 15.62 |
| `cosyvoice3` | 22.35 |
| `kokoro‡` | — |
| `moss-tts†` | 11.76 |
| `moss-tts-local†` | — |
| `moss-tts-local-v1.5†` | — |
| `moss-tts-v1.5†` | — |
| `neutts-2e‡` | — |
| `vibevoice-1.5b†` | — |
| `vibevoice-7b†` | 18.50 |
| `vibevoice-rt-0.5b‡` | — |
| `zonos2_accurate†` | 6.74 |
| `zonos2_expressive†` | 12.03 |

## 6. Methodology notes

- **WER_CLIP** caps per-utterance WER at 100% before averaging — a robust mean
  insensitive to runaway hallucinations. `WER` (raw) and `WER_L50` (mean of
  utts <50% WER) live in the `.wer` summary line alongside it.
- **Unscoreable generations count as maximum error** (WER_CLIP 100) since
  2026-09-01, rather than dropping out of the mean — otherwise a leaf mean is a
  survivor mean and the most degenerate output is the output that disappears from
  its own score. Each leaf's `.wer` footer reports `N_TRANSCRIBED:`/`N_IMPUTED:`
  so the imputed share is always visible, and `assets/coreset/systems_*.csv`
  carries it per system. The unscored utterances were re-measured before the
  switch, not assumed: 43% of them transcribed on the retry. WER published
  before that date is not comparable on an affected leaf —
  see `docs/unscoreable_2026-09.md`.
- **WER prep** is minimal and applied identically to REF and HYP: apostrophes
  deleted, lowercase + Unicode punctuation strip; zh/ja/ko/th scored as CER
  (`utils/run_wer.py`).
- **Speaker similarity** = cosine × 100, ReDimNet (`PalabraAI/redimnet2`).
- **MSR-UTMOS** = sarulab-speech MSR-UTMOS (SFI HuBERT) MOS predictor.
- **Characters/s distance** = 2-Wasserstein distance between generated and
  reference speaking-rate distributions (`scripts/speaking_rate_dist.py`); lower is better.
- **MPM span** = distance from the real-speech split-half floor (0), normalized
  by the contour-scrambled floor (1) (`utils/score_mpm_shuffle_floor.py`).
- **TTSDS2 Pitch** = normalized mean-F0 distribution similarity
  (`utils/score_ttsds_pitch.py`); interpret alongside speaker similarity.
- **DS-WED** = multi-seed prosody diversity over HuBERT k-means tokens
  (`utils/score_dswed.py`); higher = more seed-to-seed variation.
- ASR is Qwen3-ASR served via vLLM (`scripts/serve_asr.sh`).
- **Reliability gallery** — per-language edit-distance survival/tail plates for
  the GT baselines: [`assets/reliability_gallery.html`](assets/reliability_gallery.html)
  (`scripts/run_reliability.sh`).
- Empty cells (`—`) = test set not run for that model, or score file missing.
- Refresh by re-running `scripts/results_doc/run.sh` (or just `gen_md.py` if
  only the markdown layout changed).
