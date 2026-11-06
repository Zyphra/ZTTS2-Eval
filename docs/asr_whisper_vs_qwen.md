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
  *(Update 2026-08-29: this observation was the tell — the set was not partially
  code-switched but wholesale Nigerian English mislabeled by Whisper LID, and `yo`
  has been removed from the benchmark entirely; see RESULTS_VOXBLINK.md. The rows
  below are kept as the historical record of the comparison.)*

## Recommendation

Route ASR **per language** by whichever gives the lower GT WER, then score TTS models
with the same per-lang ASR. Best-of-both macro-WER: clean **24.9 → 10.0**, VoxBlink
**14.6 → 9.0**. Concretely:

- **Whisper large-v3 (forced lang)** for: bg ca cs da el es fi fil he hr hu it ja ms
  nb nl pl pt ro ru sk sv tr uk (+ most of VoxBlink incl. the CJK there).
- **Keep Qwen3-ASR** for: zh ko hi th vi yo, and the easy Western langs where it ties
  (en de fr, ar/fa where both are hard).

Both run as drop-in vLLM servers on the same endpoint, so routing is a per-lang
base-URL switch — now wired into `run_wer.py`.

## Applied to the committed GT baselines

The committed `results/gt_{clean,in-the-wild}/<lang>/wav_res_ref_text.wer` now use
this routing: the 42 non-`QWEN_LANGS` sets (28 clean + 14 VoxBlink) carry the
Whisper-forced-lang score, the 8 `QWEN_LANGS` (`en zh ko th vi hi fa yo`) stay on
Qwen. Macro-WER dropped **clean 24.9 → 10.1, VoxBlink 14.6 → 9.5** (a touch above the
per-lang-argmin 10.0/9.0 because the single global `QWEN_LANGS` keeps zh/ko/th/vi on
Qwen where Whisper is marginally better on VoxBlink, and keeps 4 clean langs — de/fr/ar/id
— on Whisper where Qwen is marginally better by <1 pt; the global policy is the price of
one routing shared by GT and models). MSR-UTMOS and speaker-sim are ASR-independent and
unchanged.

**Caveat for the leaderboard:** when this comparison was made the model results in
`RESULTS_CURRENT.md` had been scored all-Qwen, so comparing them to the routed GT
numbers was apples-to-oranges on the tail langs. Benchmark version 2 closes this:
`scripts/rescore_all.sh` rescores every model leaf and the GT baselines with the same
enforced routing, and `build_results.py` refuses a leaf whose `ASR:` footer stamp does
not match the language's routed backend.

### Using it

Routing is the default. The scoring drivers start a Whisper server themselves
(`maybe_start_whisper` in `scripts/serve_asr.sh`; it co-locates on the Qwen server's
GPU, 0.35 memory utilisation each, when no other GPU has ≥50 GB free), or reuse one
given by `WHISPER_BASE_URL`. A Whisper-routed language with no Whisper server is an
error, never a silent fall-back to Qwen. `ASR_ROUTING=0` is a diagnostic all-Qwen
mode whose results are not benchmark-comparable and are refused by `build_results.py`.

```bash
bash run_score.sh --decode_dir results/<system>/clean --test_set en
# or reuse a running Whisper server:
export WHISPER_BASE_URL=http://127.0.0.1:8001/v1
```

Knobs: `WHISPER_GPU`, `WHISPER_BASE_URL`, `WHISPER_MODEL`, `ASR_REVISION` /
`WHISPER_REVISION` (default: the revisions pinned in `utils/benchmark.py`), and
`QWEN_LANGS` (space-separated override of the keep-on-Qwen set; default
`zh ko th vi hi fa` — `en` moved to Whisper on 2026-08-29 after the scorer-swap
control in [controls_2026-08.md](controls_2026-08.md); `yo` was removed from the
benchmark the same day). Whisper codes are forced automatically (`nb`→`no`,
`fil`→`tl`) with an auto-detect fallback if a code is rejected.

Caveat: guard Whisper against its repetition/hallucination failure mode (a
max-new-tokens cap and a repeat-ngram check) before adopting it at scale, and never
force a language the audio may not actually be in — `yo` was kept on Qwen for
exactly this reason until the set was removed (its clips were Nigerian English).
