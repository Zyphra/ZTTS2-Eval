# Whisper large-v3 vs Qwen3-ASR — full-set GT WER comparison

Follow-up to [wer_tail_error_modes.md](wer_tail_error_modes.md), which found the GT
WER tail is a Qwen3-ASR language-routing artifact and that Whisper large-v3 with
forced language fixes it on a 40-clip probe. Here we score the **entire** GT set with
Whisper and compare, to decide which ASR to trust per language.

## Method

- Both ASRs served by vLLM on the OpenAI `/v1/audio/transcriptions` endpoint.
- Whisper scored the **same audio and references** as the committed Qwen GT
  (reused each `results/gt_*/<lang>/wav_res_ref_text`), through the same
  `average_wer.py` aggregation and the same text-prep/CER path as `run_wer.py`, so
  numbers are directly comparable.
- Whisper decoded with the **correct language forced** for every lang (ISO codes;
  `nb`→`no`, `fil`→`tl`). 4 vLLM servers, ~74.5k utts, 100% scored (no skips).
- Qwen numbers are the committed GT (auto-detect, hint only for 9 langs — its
  production config).

## Clean set (FLEURS-R), macro-mean WER over 35 langs

**Qwen3-ASR 24.9 → Whisper 10.8 → routed best-of-both 10.0.** Whisper wins 24/35.

| lang | mode | Qwen | Whisper | Δ | | lang | mode | Qwen | Whisper | Δ |
|------|:--:|--:|--:|--:|-|------|:--:|--:|--:|--:|
| hr | A | 84.9 | **13.3** | −71.6 | | ro | C | 18.2 | **7.2** | −11.0 |
| he | A | 81.0 | **29.5** | −51.6 | | hi | – | 14.4 | 29.2 | **+14.8** |
| uk | A | 69.8 | **6.6** | −63.2 | | tr | C | 14.2 | **9.9** | −4.3 |
| sk | B | 65.0 | **12.3** | −52.7 | | pl | C | 12.5 | **4.6** | −7.9 |
| ca | B | 64.2 | **7.5** | −56.7 | | ms | – | 11.2 | **8.2** | −3.1 |
| bg | B | 61.9 | **14.7** | −47.2 | | nl | – | 8.5 | **7.1** | −1.5 |
| nb | B | 56.2 | **12.0** | −44.2 | | id | C | 7.6 | 7.8 | +0.2 |
| hu | C | 37.0 | **19.1** | −17.9 | | ru | – | 7.2 | **5.9** | −1.4 |
| el | C | 35.5 | **16.7** | −18.8 | | th | – | 6.1 | 8.7 | +2.7 |
| fi | C | 34.0 | **13.1** | −20.9 | | pt | – | 5.0 | **4.4** | −0.6 |
| fa | B | 31.2 | 33.2 | +2.1 | | vi | – | 4.8 | 6.5 | +1.8 |
| fil | B | 27.9 | **13.4** | −14.4 | | it | – | 4.7 | **4.2** | −0.4 |
| cs | C | 25.5 | **12.6** | −12.9 | | de | – | 4.6 | 4.8 | +0.2 |
| da | C | 20.2 | **11.9** | −8.2 | | ja | – | 4.5 | **4.1** | −0.4 |
| ar | C | 19.5 | 20.5 | +1.0 | | en | – | 3.8 | 4.5 | +0.7 |
| sv | C | 18.7 | **8.2** | −10.5 | | fr | – | 3.6 | 4.3 | +0.7 |
|    |    |      |         |       | | es | – | 3.5 | **3.1** | −0.4 |
|    |    |      |         |       | | ko | – | 3.0 | 4.9 | +1.9 |
|    |    |      |         |       | | zh | – | 3.0 | 5.3 | +2.3 |

## VoxBlink2 (in-the-wild), macro-mean WER over 21 langs

**Qwen3-ASR 14.6 → Whisper 13.9 → routed best-of-both 9.0.** Whisper wins 18/21, but
the mean is dragged by one catastrophic Whisper failure (`yo`).

| lang | Qwen | Whisper | Δ | | lang | Qwen | Whisper | Δ |
|------|--:|--:|--:|-|------|--:|--:|--:|
| uk | 74.2 | **12.2** | −62.0 | | ru | 11.3 | **9.5** | −1.8 |
| tl | 23.8 | **13.6** | −10.3 | | th | 10.8 | **8.9** | −1.9 |
| tr | 21.4 | **17.3** | −4.2 | | ko | 9.4 | **6.3** | −3.1 |
| ar | 20.3 | **14.2** | −6.1 | | ja | 9.2 | **6.4** | −2.7 |
| id | 16.8 | **15.4** | −1.4 | | zh | 9.0 | **6.1** | −2.9 |
| pl | 16.3 | **10.0** | −6.3 | | pt | 8.9 | **5.7** | −3.2 |
| hi | 12.5 | 13.7 | +1.2 | | de | 8.1 | **6.3** | −1.8 |
| nl | 11.6 | **6.5** | −5.1 | | es | 6.5 | **5.1** | −1.4 |
| vi | 11.2 | 13.1 | +1.9 | | en | 4.7 | **3.9** | −0.8 |
| it | 8.1 | **7.2** | −0.9 | | **yo** | **5.7** | **103.7** | **+98.0** |
| fr | 7.3 | **6.0** | −1.3 | |    |      |         |       |

## What this shows

- **No single ASR wins everywhere.** Whisper crushes the European / Slavic /
  cross-script tail (Mode A/B: −44 to −72 pts), the whole reason the GT tail existed.
- **Qwen3-ASR wins where Whisper is weaker:** CJK (`zh` +2.3, `ko` +1.9 on clean),
  `hi` (+14.8 — Whisper's Devanagari is noticeably worse), `th`, `vi`, and it is
  ~tied/marginally better on the easy Western langs (`en`, `fr`, `de`). On VoxBlink's
  noisier in-the-wild audio Whisper flips to winning CJK too.
- **`yo` is a landmine:** the VoxBlink "Yoruba" clips are actually code-switched
  English; forcing `language=yo` triggers Whisper's repetition-hallucination (output
  floods with a repeated Bengali glyph), WER 103.7 (WER_CLIP still 58.7). Qwen handled
  them at 5.7. Forcing a wrong language label can *cause* a Whisper meltdown.
  `yo` was removed from the benchmark on 2026-08-29: the set was wholesale Nigerian
  English mislabeled by Whisper LID, not code-switched speech. Its row above is kept
  as the record of the comparison.

## Recommendation (adopted)

Route ASR **per language** by whichever gives the lower GT WER, and score TTS models with the same
per-language ASR. Best-of-both macro-WER at the time of this comparison: clean **24.9 → 10.0**, VoxBlink
**14.6 → 9.0**. The routing in force (`QWEN_LANGS` in `utils/benchmark.py`):

- **Qwen3-ASR** for `zh ko th vi hi fa`.
- **Whisper large-v3, language forced,** for every other language, English included (`en` moved after
  the scorer-swap control, [controls C1](controls_2026-08.md)).

Both run as vLLM servers on the same OpenAI endpoint; `scripts/score/run_wer.py` picks the base URL
per language.

## Applied to the GT baselines

The first routed GT rescore kept 8 languages on Qwen (`en zh ko th vi hi fa yo`) and moved the 42 other
sets (28 clean + 14 VoxBlink) to Whisper. Macro GT WER fell **clean 24.9 → 10.1, VoxBlink 14.6 → 9.5**,
a touch above the per-language argmin 10.0 / 9.0. One global routing is shared by GT and models, so
zh/ko/th/vi stay on Qwen where Whisper is marginally better on VoxBlink, and de/fr/ar/id go to Whisper
where Qwen is marginally better by < 1 pt. Since then `en` has moved to Whisper and `yo` has left the
benchmark; current GT values are in `RESULTS_CURRENT.md`. MSR-UTMOS and speaker similarity are
ASR-independent.

**Leaderboard consistency.** Every model leaf and both GT baselines are scored with the same enforced
routing (`scripts/score/rescore_all.sh`), and `build_results.py` refuses a leaf whose `ASR:` footer stamp
does not match the language's routed backend.

### Using it

Routing is the default. The scoring drivers start a fresh Whisper server themselves
(`maybe_start_whisper` in `scripts/score/serve_asr.sh`; it co-locates on the Qwen server's
GPU, 0.35 memory utilisation each, when no other GPU has ≥50000 MiB free), or use the
external one given by a preset `WHISPER_BASE_URL`. A Whisper-routed language with no Whisper server is an
error, never a silent fall-back to Qwen. `ASR_ROUTING=0` is a diagnostic all-Qwen
mode whose results are not benchmark-comparable and are refused by `build_results.py`.

```bash
bash scripts/score/run_score.sh --decode-dir results/<system>/clean --test-set en
# or reuse a running Whisper server:
export WHISPER_BASE_URL=http://127.0.0.1:8001/v1
```

Knobs: `WHISPER_GPU`, `WHISPER_BASE_URL`, `WHISPER_ASR_MODEL` (served model; `serve_asr.sh` passes it to the
client as `WHISPER_MODEL`), `ASR_REVISION` /
`WHISPER_REVISION` (default: the revisions pinned in `utils/benchmark.py`), and
`QWEN_LANGS` (space-separated override of the keep-on-Qwen set; default
`zh ko th vi hi fa` — `en` moved to Whisper on 2026-08-29 after the scorer-swap
control in [controls_2026-08.md](controls_2026-08.md); `yo` was removed from the
benchmark the same day). Whisper codes are forced automatically (`nb`→`no`,
`fil`→`tl`) with an auto-detect fallback if a code is rejected.

Whisper's repetition/hallucination failure mode is guarded in `scripts/score/run_wer.py`: a transcript
whose gzip compression ratio exceeds `ASR_LOOP_COMPRESSION_RATIO` (2.4) is retried with
`ASR_REPETITION_PENALTY` 1.2 and `ASR_MAX_COMPLETION_TOKENS` 256 (`ASR_GUARD=0` disables the retry).
Never force a language the audio may not actually be in; the `yo` meltdown above is what that does.
