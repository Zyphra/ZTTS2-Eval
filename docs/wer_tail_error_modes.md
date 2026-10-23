# WER-tail error modes (ground-truth diagnosis)

The v2 GT baselines score **real human speech** (FLEURS-R prompt clips; VoxBlink2
target clips) through the same Qwen3-ASR → WER path as the TTS systems. A chunk of
languages come back with alarming GT WER (up to ~85%). That number is **not** a data
defect or a speech-quality problem — it is the ASR being scored, and the audio is
genuine. This note classifies *why* the tail is high, because the mechanism decides
whether a TTS model's WER in that language is interpretable at all.

Source: `results/gt_zero_shot/<lang>/wav_res_ref_text.wer` (and the voxblink twin).
Per-utterance ref/hyp live in the un-committed `.wer.raw` (regenerable via
`scripts/run_gt_zero_shot.sh` / `extract_voxblink_gt.py`).

## The signal that separates the modes

For each language we look at three things beyond the headline WER:

- **WER_L50** — WER over the best-scoring 50% of clips.
- **%w=0** — fraction of clips transcribed exactly (a real "head").
- **%w≥1** — fraction of clips at ≥100% WER (a catastrophic tail).

`WER_CLIP ≈ WER` everywhere, so per-utterance WER rarely exceeds 100%; clipping is
not the lever. The shape of the distribution is.

## Mode A — cross-script routing (catastrophic, bimodal)

**Langs:** `he`, `hr`, `uk` (clean); `uk` (voxblink).
**Signature:** huge WER, `%w≥1` = 30–68%, but `WER_L50` far lower (a real correct
head coexists with a wall of 100%-wrong clips).

The ASR mis-identifies the language and transcribes into a **sister language in a
different script**, so every word is "wrong" even when the transcription is
phonetically faithful:

| lang | actual | ASR routes to | example hyp fragment |
|------|--------|---------------|----------------------|
| `hr` Croatian (Latin) | Serbo-Croatian | Serbian/Macedonian **Cyrillic** | `Гома е туристички град…` for *Goma je turistički grad…* |
| `uk` Ukrainian (Cyrillic) | Ukrainian | **Russian / Czech** | `Maršrutní taxis je švédský…` for a Ukrainian sentence |
| `he` Hebrew | Hebrew | Arabic / romanized | `الصفحة tropic دانيين…` |

`hr` is the purest case: 68.5% of clips at exactly 100% WER because the transcript is
correct **Cyrillic** for Latin-script reference text. Which clips flip is essentially
luck of the ASR's language guess → the metric is **bimodal**, so *neither* WER nor
WER_L50 is a stable floor.

**Verdict:** WER is **uninterpretable** for these langs. A TTS model here cannot be
told apart from ASR routing noise. Report MSR-UTMOS + speaker-similarity; treat WER
as unusable (not merely "high floor").

## Mode B — same-script sister-language / orthography drift (systematic offset)

**Langs:** `bg`, `ca`, `fa`, `fil`, `nb`, `sk` (clean).
**Signature:** WER 30–65% but **`%w=0` ≈ 0–2%** and **`%w≥1` ≈ 0–4%** — almost no
clip is perfect, almost none is catastrophic. The whole distribution is shifted into
the 0.3–0.6 band.

The ASR understands the audio but emits a **neighboring language's spelling/lexis in
the same script**, so nearly every clip takes a moderate, roughly constant hit:

- `nb` Norwegian → Danish/Swedish orthography (*virkelighet*→`virkelighed`, *Canada*→`Kanada`)
- `ca` Catalan → Spanish (*Les plantes presenten*→`Las plantas presentan`)
- `sk` Slovak → Czech (*V 60. rokoch 20. storočia*→`V šedesátých rocích dvacátého století`)
- `bg` Bulgarian → Macedonian (`Тоа`, `милиони`, `се преместива`)
- `fil` Filipino → correct Tagalog but word-boundary splits (`pamamangkasa`, `magin hawang`)

**Verdict:** WER carries a large **fixed floor**. Model WER is only meaningful
*relative to the GT floor* (model − GT), never against 0. Because the offset is
roughly constant across clips (unimodal), the relative reading is more trustworthy
than in Mode A.

## Mode C — genuinely harder / mixed

**Langs:** `ar`, `cs`, `da`, `el`, `fi`, `hu`, `ro`, `sv` (clean); `id`, `pl`, `tl`,
`tr` (voxblink). WER 15–37% with a real correct head (`%w=0` 6–17%). A blend of ASR
quality limits and normalization; usable with the GT floor subtracted, with caution.

## Mode OK — trustworthy

WER < 15% and a large exact-match head: `en es fr de it ja ko ru zh pt id th nl tr pl
ms vi hi` (clean) and most of the voxblink set. Read WER directly.

## Practical guidance for the leaderboard

1. **Never read absolute WER against 0** outside Mode-OK langs. Subtract the
   committed GT floor.
2. **Exclude Mode-A langs (`he`, `hr`, `uk`) from WER entirely** — the metric is
   bimodal by ASR routing luck. Speaker similarity and the prosody metrics remain
   valid there; MSR-UTMOS is **not** a naturalness fallback (on the in-the-wild
   set it ranks real speech below every TTS system — it is a channel/TTS-ness
   proxy). So Mode-A langs have speaker + prosody readouts and no naturalness
   readout at all — say so rather than substituting one.
3. WER macro-averages should either drop Mode-A/B langs or report model−GT deltas;
   raw cross-lang WER means are dominated by ASR-artifact floors, not TTS quality.
4. This is an **ASR-choice** limitation (Qwen3-ASR language routing), not a set
   defect.

## Can the ASR be told the language? (tested — no)

The scoring endpoint (vLLM `/v1/audio/transcriptions`, OpenAI-compatible) *does*
accept a `language` param — validated against the 58-language Whisper set — and the
client (`utils/run_wer.py`) already sends it, but only for 9 langs (`LANG_HINT`: en
zh de es fr it ja ko ru); every v2-added lang sends nothing.

Extending the hint does **not** fix the tail. A/B on ~40 clips/lang, correct ISO
hint vs none:

| lang | WER no-hint | WER +hint | Δ | still routes to |
|------|-------------|-----------|---|-----------------|
| hr | 84.0 | 83.7 | −0.3 | Cyrillic (unchanged) |
| uk | 64.0 | 64.8 | +0.8 | Russian/Czech (unchanged) |
| he | 84.3 | 83.7 | −0.6 | romanized (unchanged) |
| bg | 61.5 | 61.6 | +0.1 | Macedonian |
| ca | 65.5 | 65.8 | +0.3 | Spanish |
| sk | 63.0 | 63.7 | +0.7 | Czech |

The param reaches the model (output text shifts slightly) but is a **soft hint** the
model overrides with its own audio-based language ID — the script/language routing is
unchanged. The `prompt` context-bias param is likewise inert here (hr/uk/he outputs
byte-identical with and without it). So forced-language decoding is **not** available
with this ASR; do not extend `LANG_HINT` expecting a fix (it risks marginal
regressions like uk with no upside). ISO caveats: `nb`→`no`, `fil`→`tl`, Hebrew is
`he` not `iw`, and Yoruba (`yo`) is unsupported entirely.

Real remedies, in order of effort: (a) **transliteration-normalize** ref+hyp before
WER (romanize both — collapses Mode-A script mismatches like hr-Latin vs hr-Cyrillic,
which are phonetically correct); (b) a different, script/language-locked ASR; or
(c) exclude Mode-A langs from WER. Until then, treat the tail per the guidance above.

## Can a script-locked ASR fix it? (tested — yes)

Remedy (b) works. Whisper large-v3, whose `<|lang|>` decoder token is a **hard**
forced-language constraint (unlike Qwen3-ASR's soft hint), was run on 40 clips/lang
with the correct language forced, same text-prep:

| lang | mode | Qwen3-ASR (auto) | Whisper-lv3 (forced) | Δ |
|------|------|-----------------:|---------------------:|----:|
| hr | A | 84.0 | 13.1 | −71 |
| uk | A | 64.0 | 5.9  | −58 |
| ca | B | 65.5 | 7.9  | −58 |
| he | A | 84.3 | 28.6 | −56 |
| sk | B | 63.0 | 12.1 | −51 |
| bg | B | 61.5 | 11.7 | −50 |
| nb | B | 56.2\* | 12.0 | −44 |
| el | C | 35.1 | 15.1 | −20 |
| fi | C | 27.8 | 9.1  | −19 |
| fil | C | 26.1 | 11.4 | −15 |
| ro | C | 22.1 | 11.6 | −11 |
| fa | C | 34.1 | 35.3 | +1.2 |

\*nb: full-set GT WER (Qwen3 rejected the `nb` code; no 40-clip A/B).

Mode-A/B collapse from 50–84% to 6–29% — direct proof the tail was ASR language
routing, not audio. (The **full-set** Whisper-vs-Qwen comparison across all 56
lang-sets, with a per-lang routing recommendation, is in
[asr_whisper_vs_qwen.md](asr_whisper_vs_qwen.md).) Whisper forces the right language/script (uk→correct Cyrillic,
bg/nb→exact match). Exceptions: `fa` is unchanged (it was never mis-routed — genuine
difficulty/normalization, Mode C), and `he` improves greatly but stays ~29% (Hebrew
is hard for Whisper too, just no longer routed to Arabic/romanized).

Whisper runs as a **drop-in vLLM server** on the same OpenAI endpoint the scorer
already uses — `ASR_MODEL=openai/whisper-large-v3 bash scripts/serve_asr.sh` (vLLM
0.21 auto-detects it and exposes `/v1/audio/transcriptions`, no extra flags). The
server-path numbers above match the in-process run exactly, so `run_wer.py` needs
only a base-URL switch to use it.

**Actionable:** score the tail langs (or all non-Mode-OK langs) with Whisper
large-v3 forced-language instead of Qwen3-ASR — a per-lang ASR routing (Qwen3 where
it is strong, Whisper for the rest, each its own server) would make WER comparable
across the set. Caveat: 40-clip greedy probe; Whisper has its own failure modes
(silence hallucination, repetition) to guard against at full scale.
