# ZTTS2-Eval

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)

ZTTS2-Eval is a multilingual, in-the-wild benchmark for zero-shot (voice-cloning)
speech synthesis, introduced alongside ZONOS3 Mini, Zyphra's compact zero-shot
voice-cloning TTS model. It is the second release of the benchmark and supersedes
[ZTTS1-Eval](https://github.com/Zyphra/ZTTS1-Eval), which was introduced alongside
[ZONOS2](https://github.com/Zyphra/ZONOS2).

It pairs a controlled **Clean set** (FLEURS-R read speech, 35 languages, 57,366
prompts) with an **In-The-Wild set** (VoxBlink2 YouTube speech, 20 languages,
17,155 prompts), then scores generated speech for content accuracy (Qwen3-ASR or
Whisper-large-v3, routed per language), speaker similarity (ReDimNet2; on the
In-The-Wild set against a held-out real recording of the same speaker), audio
quality (MSR-UTMOS), and distributional prosody. The prosody layer asks not just
whether a system says the right words in the right voice, but whether it preserves
speaking-rate, rhythm, pitch-range, and contour distributions — whether the output
sounds like *real speech*. Every score is stamped with the benchmark version
(`BENCHMARK_VERSION`, currently `1`, a prerelease) and with the pinned scorer
revisions it was computed with ([`utils/benchmark.py`](utils/benchmark.py)).

What sets it apart:

* **Breadth and depth** — 35 Clean + 20 In-The-Wild languages; 1,196–2,089
  prompts per Clean language and up to 1,500 per In-The-Wild language; a
  quality-stratified in-the-wild set.
* **Scoring that holds at that breadth** — per-language ASR routing validated on
  real speech, untranscribable outputs counted as errors, ground-truth floors and
  confidence intervals, measured scorer-bias controls, pinned scorer revisions and
  version-stamped scores.
* **Cheap to iterate** — coreset panels estimate full-set scores with 8.9–16×
  fewer generations.

On clean read-aloud speech the cloning models bunch together; in the wild the
picture diverges sharply. Per-language metrics, Clean-set numbers, and the
distributional-prosody readouts are in
[RESULTS_CURRENT.md](RESULTS_CURRENT.md) — per-system conditioning class and
sampling config are in its "§1b. Conditioning" table; the per-language heatmaps
live in [`assets/plots/`](assets/plots/) and the interactive explorer (below).

How ZTTS2 relates to other benchmarks: [Why ZTTS2](#why-ztts2); what changed since
ZTTS1-Eval: [What's new](#whats-new-since-ztts1-eval).

## Why ZTTS2

Zero-shot TTS results are mostly reported on English and Chinese read speech: the
LibriSpeech test-clean protocol ([VALL-E](https://arxiv.org/abs/2301.02111),
[F5-TTS](https://arxiv.org/abs/2410.06885)) and
[Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval)
([Seed-TTS](https://arxiv.org/abs/2406.02430)). The multilingual sets are small
and read-speech-heavy: CV3-Eval ([CosyVoice 3](https://arxiv.org/abs/2505.17589);
9 languages × 500 prompts) and the
[MiniMax-Speech](https://arxiv.org/abs/2505.07916) multilingual set
([24 languages](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set),
100 sentences and two Common Voice speakers each).
[TTSDS2](https://arxiv.org/abs/2506.19441) is the distributional,
listener-validated exception. ZTTS2-Eval targets four gaps:

1. **Clean read speech no longer separates systems.** Cloning models bunch
   together on it. ZTTS2 adds a separate, quality-stratified In-The-Wild set with
   real same-speaker, cross-video prompt/target pairs.
2. **Error metrics are tail-driven.** A handful of catastrophic generations
   dominate WER. ZTTS2 uses thousands of prompts per language and scores
   untranscribable outputs as maximum error.
3. **One ASR cannot score 35 languages.** Auto-detect misidentifies languages
   across scripts, and a scorer can favor its own model family. ZTTS2 routes ASR
   per language, which more than halves the ground-truth WER across the 35 Clean
   languages ([asr_whisper_vs_qwen.md](docs/asr_whisper_vs_qwen.md)), and checks
   scorer bias with a scorer-swap control
   ([controls_2026-08.md](docs/controls_2026-08.md)).
4. **MOS predictors and LLM judges are not ground truth.** They are unreliable on
   clean pairs and on linguistically structured errors
   ([RefFreeLimits](https://arxiv.org/abs/2609.13150),
   [BeyondNaturalness](https://arxiv.org/abs/2608.09930)). ZTTS2 keeps MSR-UTMOS
   as a channel readout and measures prosody as distances to real speech with
   split-half floors.

|Benchmark|Languages|Prompt audio|Size|Scorers (ASR · speaker · quality)|Distribution / prosody|Reliability reporting|
|---|---|---|---|---|---|---|
|LibriSpeech test-clean (VALL-E, F5-TTS)|en|read audiobooks|LibriSpeech-PC subset: 1,127|HuBERT-L CTC (VALL-E) · WavLM-TDNN · —|—|—|
|Seed-TTS-Eval|en, zh|Common Voice, DiDiSpeech-2|1,000 en + 2,000 zh (+ zh hard)|Whisper-large-v3 / Paraformer-zh · WavLM-large SV · —|—|—|
|CV3-Eval|9|Common Voice, FLEURS, EmoBox, web-crawled|500 per language (+ hard, cross-lingual, emotion)|Whisper-large-v3 / Paraformer · ERes2Net · DNSMOS|emotion accuracy|—|
|MiniMax multilingual|24|Common Voice, 2 speakers per language|100 sentences per language|WER · SIM|—|—|
|TTSDS2|14|clean, noisy, in-the-wild, children's|20 systems, 11,000+ listener ratings|distribution factors|factor-wise distances to real speech|correlation with listeners|
|ZTTS1-Eval|9 clean / 17 wild|read speech / VoxBlink2|4,500 + 1,618|Qwen3-ASR · ReDimNet · MSR-UTMOS|TTSDS2 composite, DS-WED|GT WER floor|
|**ZTTS2-Eval**|**35 clean / 20 wild**|**FLEURS-R / VoxBlink2 in 5 quality bins**|**57,366 + 17,155**|**Qwen3-ASR or Whisper-large-v3, routed per language · ReDimNet2 · MSR-UTMOS**|**char/s distance, mHuBERT rhythm, register-normalized pitch range, MPM span (Clean only); per-utterance paired prosody vs the real recording (ITW)**|**GT and split-half floors, failed outputs = max error, bootstrap CIs, scorer controls, version-stamped, pinned scorers**|

## What's new since ZTTS1-Eval

ZTTS2-Eval is the second release of [ZTTS1-Eval](https://github.com/Zyphra/ZTTS1-Eval),
the June 2026 benchmark introduced alongside ZONOS2.

|Area|ZTTS1-Eval|ZTTS2-Eval|Details|
|---|---|---|---|
|Clean set|9 languages × 500 prompts (4,500) + `hard_en`/`hard_zh` text sets (124)|35 languages, each sampled to the maximum its disjoint FLEURS-R train+test pool allows: 1,196–2,089 per language, 57,366 total after dropping 22 prompt/target near-duplicates; hard sets not carried over|[Data](#data)|
|In-the-wild set|17 languages, 84–120 prompts each (1,618), 10 UTMOS bins|20 languages (+ nl, uk, vi), up to 1,500 prompts sampled per language (17,155 after dropping 164 prompt/target near-duplicates), 5 round-robin MSR-UTMOS bins, plus a held-out real recording of the same speaker for every item|[Data](#data), [rebuild recipe](docs/voxblink2_rebuild.md)|
|ASR|Qwen3-ASR for every language; language hint for 9, auto-detect elsewhere|Qwen3-ASR for zh ko th vi hi fa; Whisper-large-v3 with the language forced for all others, English included; routing enforced, model revisions pinned|[ASR comparison](docs/asr_whisper_vs_qwen.md), [tail modes](docs/wer_tail_error_modes.md)|
|Text prep|traditional→simplified fold on zh hypotheses only|identical on reference and hypothesis: case, punctuation, apostrophes, digits → words (num2words); CER for zh ja ko th; orthographic fold for ar fa he|[edit nature](docs/edit_nature.md)|
|Failed transcriptions|skipped, so the mean covers survivors only|scored as maximum error (WER_CLIP 100); `.wer` footers report `N_TRANSCRIBED`/`N_IMPUTED`|[unscoreable](docs/unscoreable_2026-09.md)|
|Prosody|TTSDS2 composite (Allosaurus rate, HuBERT rate, Pitch, MPM) + DS-WED|char/s distance, mHuBERT rhythm, register-normalized pitch range (each as a multiple of the real-speech split-half ceiling), MPM span (Clean only); per-utterance paired prosody vs the real recording (ITW)|[rhythm](docs/rhythm.md), [pitch range](docs/pitch_range.md), [MPM limits](docs/mpm_itw_limits.md)|
|Ground truth|GT WER floor|GT WER floor + disjoint 50/50 split-half floor for every distributional metric|[Metrics](#metrics)|
|Uncertainty and controls|none published|bootstrap CIs, generation-variance study, seen-speaker and scorer self-favoritism controls, documented confounds|[controls](docs/controls_2026-08.md), [Provenance](#provenance--known-confounds)|
|Cost|full sets only|coreset panels: Mini (16× / 8.9× fewer generations), Delta (287× / 89×), validated on held-out model families|[Coreset](docs/coreset.md)|
|Diagnostics|explorer built from your own `results/`|+ reliability gallery of per-language error tails; explorer ships a committed snapshot|[gallery](assets/reliability_gallery.html)|
|Data distribution|prompt audio in git|metadata in git, prompt audio on Hugging Face via `scripts/prep/download_data.py`|[Data](#data)|

ZTTS1-Eval and ZTTS2-Eval scores are not comparable — prompts, ASR routing, text
prep, failure handling, the speaker-similarity reference and the prosody metrics all
changed; compare systems within one release.

## Data

> **Licensing:** the code is Apache-2.0, but the data carries its own terms — the
> FLEURS-R set is CC-BY-4.0 and the **VoxBlink2 set is CC-BY-NC-SA 4.0
> (non-commercial)**. See [DATA_LICENSE.md](DATA_LICENSE.md) before using the data.

We release two evaluation sets:

* **Clean set (FLEURS-R)** — 35 languages, drawn from
  [FLEURS-R](https://huggingface.co/datasets/google/fleurs-r), a restored version
  of the original FLEURS dataset. The original 9 (en, zh, de, es, fr, it, ja, ko,
  ru) plus a higher-resource tier (pt, ar, hi, tr, pl, id, th, fil, nl, vi, cs,
  sv, da, fi, el, hu, ro, uk, nb, he, bg, hr, sk, ca, ms, fa). Each language is
  sampled to the maximum its FLEURS-R train+test pool allows with prompt and
  target kept disjoint (1,196–2,089 utterances per language, 57,366 total). Lives at
  `data/clean/<lang>/`. The prompt is a duration-filtered clip; the target
  is the *text* of a random other utterance of the same language — FLEURS-R
  metadata carries no speaker IDs, so no speaker or session pairing is implied.
* **In-The-Wild set (VoxBlink2)** — a subsample of
  [VoxBlink2](https://voxblink2.github.io/) across 20 languages (en, zh, de, es,
  fr, it, ja, ko, ru, pt, ar, hi, tr, tl, pl, id, th, nl, uk, vi) for acoustic,
  content, and prosodic diversity. VoxBlink2 is composed of in-the-wild YouTube
  clips with highly variable quality, so prompts are selected round-robin across 5
  uniform bins spanning a 1–4 MSR-UTMOS range rather than letting clean prompts
  dominate the distribution. Each language is sampled up to 1,500 prompts (the 7
  largest languages keep 1,463–1,496 after the near-duplicate filter below; smaller
  languages take their available ceiling, down to 108 — 17,155 prompts total). Lives
  at `data/in-the-wild/<lang>/`. The large
  per-language N is deliberate: TTS error metrics are tail-driven — dominated by
  rare catastrophic generations — so more samples give a materially more stable
  estimate.
  Prompt and target text come from two different videos of the same speaker —
  the only set with real same-speaker pairing (the Clean set's target is
  text-only, with no speaker linkage to the prompt), so cross-recording
  speaker-similarity claims rest on this set. The real recording behind each
  target text ships too (`target_wav/`): it is the reference for the held-out
  speaker-similarity score.

**Near-duplicate filter.** Items whose prompt transcript and target text are the
same sentence (equal after case/punctuation folding, or ≥ 0.8 `difflib` ratio;
`utils/text_norm.near_duplicate`) are removed from both sets and from the coreset
panels, because a system can score well on them by repeating its prompt: 22 Clean and
164 In-The-Wild items (Clean 57,388 → 57,366; In-The-Wild 17,319 → 17,155). They
are listed in [`data/near_duplicates_removed.json`](data/near_duplicates_removed.json);
the prep scripts skip such pairs, and `python scripts/prep/drop_near_duplicates.py`
re-applies the filter idempotently.

Each language directory uses the layout the scoring pipeline expects:

```
text            # "<uttid> <target_text>"      (what the TTS model synthesizes)
prompt_text     # "<uttid> <reference_text>"
prompt_wav.scp  # "<uttid> <path_to_prompt.wav>"
waveform/       # prompt_audio_<N>.wav          (reference speaker prompts)
target_wav.scp  # In-The-Wild only: "<uttid> <path_to_real_target.wav>"
target_wav/     # In-The-Wild only: <uttid>.wav (held-out real recording of the same speaker)
```

The audio (`waveform/`, 24 GB, plus the In-The-Wild `target_wav/` recordings) is not in git: it is the Hugging Face dataset [`Zyphra/ZTTS2-Eval`](https://huggingface.co/datasets/Zyphra/ZTTS2-Eval). Fetch it into place with `python scripts/prep/download_data.py` (both sets), or a subset with `--tasks in-the-wild --langs en zh`; for the In-The-Wild set the script also fetches `target_wav/` and verifies that every wav listed in `prompt_wav.scp` and `target_wav.scp` resolves. The `target_wav.scp` index and `target_provenance.json` (pairing rule, counts, exclusions) are in git. The coreset panel directories symlink into these trees.

**What is in git.** The `data/` metadata is tracked: `text`, `prompt_text`,
`prompt_wav.scp`, the In-The-Wild `target_wav.scp` / `target_provenance.json`,
`data/near_duplicates_removed.json` (the prompt/target near-duplicates dropped from
both sets and the panels), the VoxBlink2 selection provenance
(`data/in-the-wild/_stage_a/manifest.tsv`) and `_msr_utmos_cache.tsv`, the shipped
MSR-UTMOS binning cache the eval scripts read. The prompt wavs (`waveform/`) and the
target recordings (`target_wav/`) come from the Hugging Face dataset; only the
`_stage_a/*_to_chmod.txt` lists are untracked prep byproducts (regenerated by the
stage A/B scripts). `results/` is not tracked except for the ground-truth score
baselines (`results/gt_clean/`, `results/gt_in-the-wild/`).

The coreset panels `clean_{mini,delta}/` and `in-the-wild_{mini,delta}/` are task
directories of the same layout: `panel.json` holds the item weights, their
`waveform/` entries symlink into the full-set trees, and the In-The-Wild panels'
`target_wav.scp` lists the panel's subset of the full set's `target_wav/` files.

## Metrics

ZTTS2-Eval reports the standard intelligibility, speaker, and quality scores with
pinned scorers, then adds a prosody layer for speaking rate, rhythm, pitch range,
and contour. Scorer revisions, language baskets, and the ASR routing table are
defined once in [`utils/benchmark.py`](utils/benchmark.py).

* **Content accuracy** — Character / Word Error Rate of the ASR transcription
  against the target text (CER for zh, ja, ko, th), after identical text prep on
  both sides. ASR is routed per language: **Qwen3-ASR** (`Qwen/Qwen3-ASR-1.7B`)
  for `zh ko th vi hi fa`, **Whisper-large-v3** with the language forced for every
  other language including English ([docs/asr_whisper_vs_qwen.md](docs/asr_whisper_vs_qwen.md)).
  Routing is the default and is enforced: scoring a Whisper-routed language
  without a Whisper server is an error, not a silent fallback to Qwen. Setting
  `ASR_ROUTING=0` scores everything with Qwen3-ASR as a diagnostic; its results are
  not comparable and `build_results.py` refuses them. Both models are pinned to a
  Hugging Face revision.
  **Digits are expanded to words** (num2words) on both reference and hypothesis
  before alignment, so `20` and `twenty` match. Languages num2words does not cover
  stay in digits on both sides: bg el fil hi hr ms tl zh.
  Ground truth is scored through the same pipeline as a
  reference floor. Many systems score *below* human-speech WER on the
  In-The-Wild set, i.e. read more cleanly than real people speak. Read WER
  alongside the prosody metrics, not in isolation.
  A generation the reference ASR cannot transcribe at all counts as **maximum
  error** (WER_CLIP 100), not as a missing utterance — otherwise a leaf mean is a
  survivor mean and the most degenerate output is the output that disappears from
  its own score. Each `.wer` footer reports `N_TRANSCRIBED:`/`N_IMPUTED:` so the
  imputed share is always visible. See `docs/unscoreable_2026-09.md`.
  The footer also reports `WER_TAIL50` — the **tail rate**, the percentage of
  utterances with WER above 50%. Confidence intervals on WER are bootstrapped over
  reference *sentences*: utterances that share a target sentence (FLEURS repeats
  sentences across speakers) resample together.
* **Speaker similarity** — cosine similarity between ReDimNet2 speaker embeddings
  ([ReDimNet2](https://github.com/PalabraAI/redimnet2), loaded via torch.hub at a
  pinned commit; `b6`, `vb2+vox2_v0`).
  On the **In-The-Wild set the primary score compares the generation with a
  held-out real recording of the same speaker** (the item's `target_wav`, from a
  different video than the prompt), so a system cannot score well merely by
  reproducing the prompt's channel and session. Similarity to the **prompt** is kept
  as a secondary column, and the **copy gap** (prompt − held-out) shows how much a
  system leans on the prompt recording. The ceiling for the held-out score is the
  real prompt against the real target of the same item. The Clean set has no
  same-speaker target, so it reports prompt similarity only. Coreset panels estimate
  the prompt similarity.
* **Audio quality** — **MSR-UTMOS** ([msr-utmos](https://github.com/sarulab-speech/msr-utmos),
  a parallel MOS predictor; checkpoint pinned to a commit).
* **Speaking-rate distance** — 2-Wasserstein distance between generated and
  reference characters-per-second distributions, computed at the same half-vs-half
  geometry as the ceiling. The duration is that of the file cut to its Silero VAD
  speech span (computed at 16 kHz, 30 ms edge pad), so lead-in silence, tones or
  noise do not count as speech time. Only utterances the ASR heard with WER ≤ 20% are
  scored, so a dropped or runaway sentence counts against intelligibility, not pace
  ([controls C7](docs/controls_2026-08.md)). Lower is better; the reference split-half
  distance is the finite-sample ceiling.
* **Rhythm** — per-utterance timing statistics of the run-length distribution of
  deduplicated mHuBERT-147 k-means tokens, on each file cut to its Silero VAD speech
  span (so generated lead-in tones or noise do not read as stalls; [controls
  C9](docs/controls_2026-08.md)), scored as the 2-Wasserstein distance to
  the reference at split-half geometry and reported as a multiple of the real-speech
  split-half ceiling (`× ceiling`; 1 = as close to real speech as two halves of real
  speech are to each other). Two dimensions are reported: the **pause/stall factor**
  (burstiness of the run lengths) and the **pace factor** (token rate). As for
  speaking rate, only generations the ASR heard with WER ≤ 20% are scored; the CSV's
  `heard_pct` column is the share of a system's utterances that pass (the real-speech
  reference is not filtered). Lower is better. See [docs/rhythm.md](docs/rhythm.md).
* **Pitch range** — per-utterance register-normalized pitch range: p90 − p10 of the
  voiced F0 (pYIN) in semitones around the utterance's own median, so it measures
  how much the melody moves and ignores whether the voice is high or low. Each file is
  cut to its Silero VAD speech span, and frames pYIN calls voiced without a voice
  (near-silent pauses, background hum) are removed
  ([controls C10](docs/controls_2026-08.md)). Scored as
  the 2-Wasserstein distance to the prompt recordings at split-half geometry, in
  `× ceiling`, on a fixed 500-utterance sample per (set, language), keeping only the
  generations heard with WER ≤ 20% (`heard_pct` as for rhythm). Lower is better.
  See [docs/pitch_range.md](docs/pitch_range.md).
* **Paired prosody (In-The-Wild only)** — every In-The-Wild item has a real
  recording of the same speaker saying the target sentence, so prosody is also
  compared utterance by utterance against it (same 500-utterance sample): median
  |log2 duration ratio| (durations of the Silero VAD speech spans, on the generations
  with WER ≤ 20%, as for speaking rate), mean |pause-count difference| (a pause is a
  gap ≥ 0.2 s between Silero VAD speech segments, which still finds pauses under
  background noise), and median |log2 pitch-range ratio|. 0 means the generation
  matches the real recording; lower is better.
* **MPM span (Clean set only)** — Masked Prosody Model distance normalized per
  language between the real-speech split-half floor (0) and a contour-scrambled
  floor (1). Lower is better, and **values above 1 are worse than the same speech
  with its contour scrambled**. Every utterance is first cut to its Silero VAD speech
  span (found at 16 kHz and mapped onto the 22.05 kHz MPM input; a file with no
  speech is dropped), since the FLEURS-R recordings carry ~2.45 s of leading/trailing
  silence and most systems almost none ([controls C6](docs/controls_2026-08.md)). It
  is not computed on the In-The-Wild set: the anchors collapse on in-the-wild audio
  ([docs/mpm_itw_limits.md](docs/mpm_itw_limits.md)).

The `gt_clean` and `gt_in-the-wild` rows use a reproducible, disjoint random 50/50 split of each
reference corpus rather than comparing ground truth with itself. Seed and
language determine the split; `METRICS_KINDS="speaking_gt mpm_gt" bash slurm/submit_metrics.sh`
refreshes the char/s and MPM halves. For rhythm and pitch range the split-half ceiling is the
1× line that systems are measured against.

## Provenance & known confounds

Facts to keep next to the numbers when reading results:

* **FLEURS-R prompts are restored audio, not raw recordings.** FLEURS-R is the
  output of a speech-restoration model (Miipher) over FLEURS, so Clean-set
  prompts — and the "real-audio" MSR-UTMOS ceiling on that set — are enhanced
  speech.
* **Possible ASR-scorer contamination on the Clean set.** The set pools FLEURS
  train+test splits, and the ASR scorers (Whisper-large-v3, Qwen3-ASR)
  plausibly trained on FLEURS train. The effect is unbounded; a train-vs-test
  GT-floor comparison is an open control.
* **In-the-wild reference transcripts are ASR output.** VoxBlink2 target texts
  are Voxtral-Mini-3B transcriptions of segments pre-filtered for cross-ASR
  agreement ≥ 0.85. The GT WER floor is inter-ASR disagreement on ASR-legible
  speech, not human intelligibility — read "beats the GT floor" as an
  ASR-legibility margin.
* **Seen-speaker speaker similarity on the In-The-Wild set.** The speaker
  embedder (ReDimNet2-B6, `vb2+vox2_v0`) is trained on VoxBlink2 — the ITW eval
  corpus itself — so ITW similarity is a seen-speaker regime. A held-out
  (`vox2`-only) embedder control found system rankings robust to this
  (Kendall τ +0.91…+0.98 over en/zh/de/th) but the held-out model is worse
  calibrated, so the shipped embedder was kept; see
  [docs/controls_2026-08.md](docs/controls_2026-08.md).
* **Scorer/system family overlap.** Qwen3-ASR scores `zh ko th vi hi fa` and the
  evaluated qwen3-tts systems are the same model family. A scorer-swap control
  measured real self-favoritism on English (+0.5…+0.7 WER points vs a
  four-system cohort), so **English is now scored by Whisper-large-v3**; the zh
  effect was null and zh stays on Qwen. See
  [docs/controls_2026-08.md](docs/controls_2026-08.md).
* **One generation per prompt.** Every published score is a single generation
  (zonos2 at seed 0; served models on unseeded server defaults), so the reported
  CIs cover prompt sampling, not generation sampling variance. A multi-seed
  study measured that variance over four systems × two languages × three
  generations: mean per-language sd is 0.15 WER_CLIP points (0.03–0.40 across
  cells) and ≤ 0.35 for speaker similarity, contributing ≈ ±0.10 to a paired
  macro difference — a 3–7 % widening of the published CIs that changes no
  published verdict. Per-*language* cells are noisier: do not read a single-
  language gap below ~0.5 WER points. See
  [docs/controls_2026-08.md](docs/controls_2026-08.md).
* **MSR-UTMOS reads as clean-studio-TTS-ness, not naturalness.** On the
  In-The-Wild set it ranks real speech below every TTS system (GT ≈ 2.25). Use
  it for channel comparisons, not quality rankings.
* **Open items that need humans:** a transcript-verification sample for the ITW
  floor, and a listener probe to perceptually anchor any headline metric.

## Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# install uv first: https://docs.astral.sh/uv/getting-started/installation/

uv sync                       # core deps, incl. the vLLM Qwen3-ASR server
source .venv/bin/activate     # or prefix the commands below with `uv run`

# recommended quick-start env: local vLLM-Omni generation + full scoring
#   uv sync --extra omni --extra scorers
#
# optional extras (install only what you need):
#   uv sync --extra scorers   # MSR-UTMOS scorer
#   uv sync --extra viz       # matplotlib/scipy for the plots
#   uv sync --extra omni      # vllm-omni local multi-model TTS serving
#   uv sync --extra gemini    # (or --extra elevenlabs / cartesia / inworld)
```

`torch`, `torchaudio`, `torchvision`, `torchcodec`, and `vllm` are pinned to CUDA
12.9 builds — the PyTorch `cu129` wheel index plus vLLM's `+cu129` release wheel,
configured in `pyproject.toml`. CUDA 12.9 builds run on a 12.8 driver via CUDA
minor-version compatibility. To target a different CUDA, change the
`pytorch-cu129` index URL and the vLLM wheel URL in `pyproject.toml` before
running `uv sync`. The benchmark needs a CUDA GPU — vLLM, speaker similarity, and
MOS scoring all run on GPU — and `torchcodec` requires FFmpeg shared libraries on
the system.

The repo supports Python 3.10–3.12 in `.venv`. `transformers` is pinned below 5.9
for vLLM-Omni compatibility. Shell entry points select their interpreter with
`repo_py` (`utils/lib_common.sh`): `$PYTHON` if set, else `.venv/bin/python` if it
exists, else `python3`. Jobs submitted with `sbatch` always use `.venv/bin/python`.
Shell scripts parse Kaldi-style `--name value` options (`utils/parse_options.sh`),
and every long flag, in shell and in Python, is spelled with dashes (`--test-set`,
`--out-root`, `--decode-dir`). Run every command from the repo root. Optional model
venvs (`<MODEL>_VENV`) and the other environment variables the scripts read are
listed in [`.env.example`](.env.example).

## Quick start

A **CUDA GPU is required** for generation and scoring (vLLM, speaker similarity,
and MOS all run on GPU). With no GPU you can still validate the shared utilities
and browse the shipped benchmark inputs.

```bash
# 0. (no GPU needed) sanity-check the shared utils — no downloads, no network
python3 tests/smoke_test.py

# 1. install deps for the local vLLM-Omni path + full scoring (GPU host)
uv sync --extra omni --extra scorers && source .venv/bin/activate

# 2. fetch the audio for the set you'll run: prompts + held-out target recordings
#    (drop the flags for everything, 24 GB+)
python scripts/prep/download_data.py --tasks in-the-wild --langs en

# 3. generate the English slice of the In-The-Wild Mini coreset panel (50 items)
#    with one local model
bash scripts/gen/run_omni.sh --model qwen3-tts-0.6b --task in-the-wild_mini --test-set en

# 4. score it. scripts/score/run_score.sh auto-starts/stops the ASR servers and routes each
#    language to Qwen3-ASR or Whisper-large-v3 as the published numbers do; both
#    co-locate on GPU 0 by default (same settings as slurm/eval_job.sh). It also
#    scores speaker similarity against the held-out real recordings.
bash scripts/score/run_score.sh --decode-dir results/omni_qwen3-tts-0.6b/in-the-wild_mini --test-set en

# 5. weighted estimate of the FULL In-The-Wild score, with a bootstrap CI
python scripts/coreset/score_panel.py --decode-dir results/omni_qwen3-tts-0.6b/in-the-wild_mini

# 6. browse results in the explorer
python scripts/results_doc/build_results.py    # builds the explorer snapshot from results/
python explorer/serve.py --port 8000           # open http://localhost:8000
```

> **Explorer note:** the explorer prefers the live build cache
> (`scripts/results_doc/_cache/results_snapshot.json`, generated from your
> `results/` tree in step 6) and falls back to the **committed** snapshot in
> `explorer/snapshot/`, so a fresh clone serves a populated explorer. The snapshot,
> its allowlist and the API are documented in [explorer/README.md](explorer/README.md).
> `build_results.py` refuses score files that are not stamped with the current
> benchmark version, so score every leaf with the scripts above rather than
> copying score files in.

The bundled explorer (`python explorer/serve.py`) provides a leaderboard
sortable by any metric, a model × language heatmap, VoxBlink2 quality-bin
curves, and a per-utterance drill-down with audio playback and ASR diffs.
[Coreset panels](docs/coreset.md) show up as their own tasks, scored with their
published item weights and captioned with what they can and cannot resolve.

Data-prep and evaluation details follow. The shipped data directories are
already prepared (metadata in git, audio via `scripts/prep/download_data.py`); the raw-corpus rebuild recipe is in
[docs/voxblink2_rebuild.md](docs/voxblink2_rebuild.md).

## Data preparation

The FLEURS-R set can be (re)built from
[`google/fleurs-r`](https://huggingface.co/datasets/google/fleurs-r):

```bash
uv run python scripts/prep/prepare_fleurs_r.py --overwrite
```

This downloads FLEURS-R for all 35 curated languages and writes `text`,
`prompt_text`, `prompt_wav.scp`, and `waveform/` into each language directory. By
default each language is sampled to the maximum its pooled train+test splits allow
(prompts and targets stay disjoint, and near-duplicate prompt/target pairs are
skipped); pass `--count N` to cap it, or `--langs` to
restrict the set. It requires HuggingFace access to `google/fleurs-r`
(`huggingface-cli login` or `--hf-token`).

### VoxBlink2 set

The released `data/in-the-wild/` directories are already prepared: they
include the target/reference text, manifest, the `target_wav.scp` /
`target_provenance.json` index of the held-out recordings, and the MSR-UTMOS binning
cache used by the evaluation scripts; the prompt audio and the held-out target
recordings (`target_wav/`) come from `scripts/prep/download_data.py`.
Rebuilding this set from raw VoxBlink2 requires a
local VoxBlink2 audio tree and a pre-computed enriched candidate pool, so the full
recipe lives in [docs/voxblink2_rebuild.md](docs/voxblink2_rebuild.md).

The prep scripts live in `scripts/prep/`: `download_data.py` (fetches the released
prompt audio and target recordings from the Hub), `prepare_fleurs_r.py` (builds
`clean/`), `build_voxblink2_enriched.py` (rebuilds the enriched candidate pool),
`prepare_voxblink2_eval_stage_a.py` (candidate selection),
`prepare_voxblink2_eval_stage_b.py` (MSR-UTMOS stratified sample, round-robin across
5 bins up to ~1500/lang), `extract_voxblink_gt.py` (extracts the real target
recordings to `target_wav/` from a local raw VoxBlink2 tree),
`build_voxblink_targets.py` (validates them and writes `target_wav.scp` /
`target_provenance.json`), `drop_near_duplicates.py` (the idempotent near-duplicate
filter over both sets and the panels) and `upload_release_data.py` (publishes the
target recordings and prompt deletions to the dataset). `voxblink_common.py` holds
the shared VoxBlink2 paths: the local audio tree comes from `VOXBLINK2_AUDIO_ROOT`,
the enriched candidate pool from `VOXBLINK2_ENRICHED`.

## Generation

Every generation driver is `scripts/gen/run_<name>.sh` and writes
`<out_root>/<task>/<lang>/wavs/` plus a `gen_meta.json` (model, seed and sampling
fields, written once per leaf unless `--overwrite true`). All drivers share one
interface (`scripts/gen/lib_gen.sh`):

* `--task` (`clean`, `in-the-wild` or a panel) and `--test-set "<langs>"` (default:
  the task's frozen basket, `python3 -m utils.benchmark langs <task>`). Languages the
  model does not support are skipped with a `[skip]` line; `--list-langs true` prints
  the languages the driver would run and exits, before any venv or server check.
* `--out-root` (default `results/<model dir>`, from `gen_default_out_root`),
  `--overwrite true` to regenerate existing wavs (otherwise they are skipped), and
  `--seed` where the model accepts one (left empty, the generator's own default or
  an unseeded run is used, and `gen_meta.json` records it).
* At the end each driver prints the `scripts/score/run_score.sh` command for its output.

Models with their own venv (`cosyvoice`, `kokoro`, `ming3`, `moss_tts`, `neutts`,
`phonon`, `pocket_tts`, `vibevoice`, `vibevoice15`, `audio8`) are installed with
`bash scripts/setup/setup_<model>_env.sh` (`--force` rebuilds); the venv path is
`<MODEL>_VENV`, and the upstream checkouts are pinned by `<MODEL>_COMMIT`.

**Servers.** Server-backed drivers launch a **fresh** server per run and stop it on
exit; a server already listening on the port is never adopted. Startup tries the
preferred `--port`, then OS-chosen free ports (three attempts), and the server's
process group is killed on exit (`utils/lib_serve.sh`). Pass `--server URL` to use an
external server instead (no launch/stop). Server logs go to `/tmp/<role>_<port>.log`,
or to `${SERVE_LOG_DIR}/<role>.log` when `SERVE_LOG_DIR` is set (the SLURM jobs set it
to `logs/slurm/servers/<tag>/`).

* **`scripts/gen/run_omni.sh`** is self-contained: it starts/stops a local vLLM-Omni
  server (`scripts/gen/serve_omni.sh`, `uv sync --extra omni`) per model of its
  registry (`scripts/gen/omni_models.sh`: qwen3-tts, cosyvoice3, voxcpm2, voxtral-tts,
  fish-s2-pro, …) and is the recommended starting point. `OMNI_VENV` names a venv
  with the omni extra other than `.venv`.
* **`scripts/gen/run_zonos2.sh`** starts/stops a Zonos2 Mini-SGLang server
  (`scripts/gen/serve_zonos2.sh` → `python -m zonos2`; set `ZONOS2_HOME` to a
  [ZONOS2](https://github.com/Zyphra/ZONOS2) checkout with its own venv; health check
  `GET /tts/capabilities`), optionally `--gpu N --port N`.
  Sampling defaults match the released Zonos2 model's reference eval config
  (temperature 1.15, topk 106, min-p 0.18; speaking-rate and server-side text
  normalization off — re-enable with `--speaking-rate-bucket N` /
  `--text-norm true`). Zonos2 has two modes: **stable/accurate** (`--accurate-mode
  true`, closer voice match, → `results/zonos2_accurate`) and **expressive**
  (`--accurate-mode false`, → `results/zonos2_expressive`). Under SLURM,
  `slurm/submit_eval.sh --engine zonos2 --accurate-mode {true|false}` runs each mode
  as its own job.
* **`scripts/gen/run_zonos3.sh`** generates ZONOS3 Mini through the release minisgl server
  (`scripts/gen/serve_zonos3.sh` → `python -m zonos2` from the `ZONOS3_HOME` checkout,
  optional site env `ZONOS3_ENV`; client `scripts/gen/generate_zonos3.py` → `POST
  /tts/generate`), one conditioning mode per call: `--speaker-mode reference` (raw
  reference prefix) or `--speaker-mode embedding` (speaker embedding only) →
  `results/zonos3-mini_<mode>`. It sends the release server's sampling defaults
  explicitly, seed 0, server TN for en zh de es fr it pt ja ko (every other language
  raw, TN off); `--laci true` adds LACI decoding on a server started with
  `--laci-heads` (via `ZONOS3_SERVE_ARGS`). `slurm/zonos3_gen_job.sh` runs both modes
  over whole 8-GPU nodes (one server per GPU).
* **`scripts/gen/run_audio8.sh`** does the same for the Audio8 SGLang-Omni server
  (`scripts/gen/serve_audio8.sh`).

### Hosted providers

Each hosted-provider script installs via an extra (`uv sync --inexact --extra <name>`,
so the scorer extras already in `.venv` stay installed) and reads its API key from an
environment variable; install only what you need. Only one instance per provider
account may run at a time (the clone purges are account-wide).

| Script | Extra | Env var |
| --- | --- | --- |
| `scripts/gen/run_gemini.sh` | `--extra gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `scripts/gen/run_elevenlabs.sh` (+ `_clone`) | `--extra elevenlabs` | `ELEVENLABS_API_KEY` |
| `scripts/gen/run_cartesia.sh` | `--extra cartesia` | `CARTESIA_API_KEY` |
| `scripts/gen/run_inworld.sh` | `--extra inworld` | `INWORLD_API_KEY` |

`scripts/gen/run_elevenlabs_clone.sh`, `scripts/gen/run_cartesia.sh --mode clone` and
`scripts/gen/run_inworld.sh --clone true` clone each prompt voice per utterance and
delete the clone after use; Gemini runs on preset voices. `scripts/gen/run_gemini.sh`
uses the Gemini Batch API by default (`--batch true`: half the unary price, async;
jobs are tracked in `<out_dir>/batch_jobs.json` so an interrupted run re-attaches
instead of resubmitting). ElevenLabs and Cartesia have no batch TTS endpoint;
Inworld's `voice:synthesizeBatch` is not used because each item names a voice, so
every per-utterance clone would have to stay live for the batch's hours-long
turnaround, against the workspace's live-voice cap.

## Evaluation

Evaluation has two complementary layers: the standard intelligibility, speaker
match, and MOS pass, plus a prosody pass that measures whether the generated
corpus keeps the reference distribution's speaking-rate, rhythm, pitch-range, and
contour behavior.

```bash
# WER, speaker similarity, and MSR-UTMOS for one system's task directory; the task
# is the --decode-dir basename and --test-set defaults to its basket.
# (auto-starts and stops the ASR servers — Qwen3-ASR plus Whisper-large-v3 for the
#  routed languages; the run fails if the Whisper server cannot start)
bash scripts/score/run_score.sh --decode-dir results/<system>/clean

# On the In-The-Wild set (VoxBlink2) speaker similarity is also scored against the
# held-out real recordings (spk_simi_scores.xutt.txt)
bash scripts/score/run_score.sh --decode-dir results/<system>/in-the-wild

# ground-truth baselines: real prompt clips (clean) or real target recordings
# (in-the-wild) through the same pipeline -> results/gt_<task>/<lang>/
bash scripts/score/run_gt.sh --task clean
bash scripts/score/run_gt.sh --task in-the-wild --test-set "en zh"
```

`scripts/score/run_score.sh` and `scripts/score/run_gt.sh` share one loop,
`score_lang_dir` in `scripts/score/lib_infer_scoring.sh`: WER
(`scripts/score/get_wav_res_ref_text.py` → `scripts/score/run_wer.py` →
`utils/average_wer.py`), speaker similarity (`scripts/score/eval_speaker_similarity.py`,
ReDimNet2) and MSR-UTMOS (`scripts/score/score_msr_utmos.py`). The ASR servers come
from `scripts/score/serve_asr.sh`: a fresh Qwen3-ASR and Whisper-large-v3 server per
run, logged like the TTS servers (`/tmp/asr_<port>.log` or `${SERVE_LOG_DIR}/asr.log`).
A preset `ASR_BASE_URL` / `WHISPER_BASE_URL` points scoring at an external server
instead. Routing is on by default; `ASR_ROUTING=0` is a diagnostic all-Qwen mode
that `build_results.py` refuses. `scripts/score/rescore_all.sh` (an `sbatch` array)
rescores every stale leaf under `results/`; `RESCORE_SYSTEMS="a b"` restricts it.
`scripts/score/repair_wer_raw.py` repairs `.wer.raw` records split by tabs/newlines
in an ASR hypothesis.

**Outputs.** Model generations nest as `results/<model>/<task>/<lang>/`
(`omni_<model>/`, `zonos2_<mode>/`, hosted-provider dirs); the GT and Whisper
baselines fold the task into the dir name and are flat — `results/gt_clean/<lang>/`,
`results/whisper_in-the-wild/<lang>/`. Each leaf holds `wavs/`, `wav.scp`,
`wav_res_ref_text.wer[.raw]`, `spk_simi_scores.txt` (vs the prompt),
`spk_simi_scores.xutt.txt` (In-The-Wild only: vs the held-out target recording),
`msr_utmos_scores.txt` + `msr_utmos_mean.txt` + `msr_utmos_ci.txt`, and the metric
cache `pitch_range.tsv`. Distribution metrics write
`results/{rhythm,pitch_range}_<set>_<lang>.csv`, `results/paired_itw_<lang>.csv` (each
row with `benchmark_version`) and the MPM span aggregates in their own results
subtrees.

**Version stamps and rescoring.** Every score file records what produced it: the
`.wer` footer carries `WER_TAIL50` (% of utterances with WER > 50%),
`BENCHMARK_VERSION` and `ASR` (`<backend> <model>@<revision>`), the
`spk_simi_scores*.txt` files start with `#BENCHMARK_VERSION` and `#SPK_MODEL`, and
`msr_utmos_ci.txt` carries `BENCHMARK_VERSION` and `MSR_UTMOS_CKPT`. Each score kind
of a leaf is redone only when it is stale — a missing or outdated stamp, a `nan`
WER, an all-zero speaker average, an empty MSR mean, or coverage below
`SCORE_MIN_COVERAGE` (0.98). `python3 -m utils.benchmark stale <mode> <data_dir>
<leaf> <lang>` prints the stale kinds of a leaf, and `python3 -m utils.benchmark
problems <leaf_dir> <lang>` lists why its stamps are stale.

```bash
# Characters/s distributional distance (defaults to clean/en)
python scripts/prosody/speaking_rate_dist.py
python scripts/prosody/speaking_rate_dist.py --test-set in-the-wild --lang en

# mHuBERT rhythm (GPU): pause/stall and pace factors vs the real-speech ceiling
python scripts/prosody/rhythm_dist.py --test-set in-the-wild --lang en

# register-normalized pitch range (CPU; fixed 500-utterance sample)
python scripts/prosody/pitch_range_dist.py --test-set in-the-wild --lang en

# per-utterance paired prosody vs the real recording (In-The-Wild only; run after
# pitch_range_dist.py, it shares the pitch caches)
python scripts/prosody/paired_itw.py --lang en

# UTMOS shift: signed mean shift (dmean, with CI and the reference split-half band)
# and spread ratio of a system's per-utterance MSR-UTMOS distribution vs. the
# reference's (needs the msr_utmos_scores.txt leaves); W2 kept as a secondary column
python scripts/analysis/utmos_shift.py --test-set in-the-wild --lang en

# MSR-UTMOS level-sensitivity probe (gain sweep); writes assets/probes/utmos_level.csv
python scripts/analysis/probe_utmos_level.py

# MPM span against the real/scrambled anchors (Clean set only)
python scripts/prosody/score_mpm_shuffle_floor.py \
    --ref-dir data/clean/en/waveform \
    --synth-dir results/omni_qwen3-tts-1.7b/clean/en/wavs \
    --out results/_mpm_shuffle_floor/clean_en.tsv
```

The prosody scripts share `scripts/prosody/common.py` (split-half W2 ceiling, the
WER ≤ 0.2 `heard_uids` filter, system discovery) and the Silero VAD helpers in
`utils/vad.py`; their CLIs take `--models`, `--out` and `--workers` (default 48).
Speech-span trims are computed at 16 kHz. `scripts/plots/plot_rhythm_*.py` and
`scripts/plots/plot_pitch_carpet.py` draw the rhythm and pitch figures in
`assets/plots/`. The explorer exposes characters/s distance and MPM span as
dedicated distribution views.

Because TTS error metrics are tail-driven, the ground-truth baselines ship with a
**reliability gallery** — edit-distance survival/tail plates per language (and per
ASR backend) that show *how often* an utterance goes badly wrong and *by how much*.
`scripts/plots/tufte_common.py` rebuilds per-utterance edit counts from
`wav_res_ref_text.wer.raw` with `run_wer.py`'s tokenization and feeds the plotters
`tufte_p99.py`, `tufte_survival_overlay.py`, `tufte_survival_smallmultiples.py` and
`tufte_slopegraph.py`. Regenerate the plates and the self-contained page with `bash
scripts/plots/run_reliability.sh` (also run late in `scripts/results_doc/run.sh`);
`scripts/plots/build_reliability_gallery.py` writes
[`assets/reliability_gallery.html`](assets/reliability_gallery.html). Context
in [docs/edit_nature.md](docs/edit_nature.md).

The report pipeline (`RESULTS_CURRENT.md`, the explorer snapshot, the plots) is
`bash scripts/results_doc/run.sh`; see
[scripts/results_doc/README.md](scripts/results_doc/README.md). SLURM launchers for
generation, scoring, metrics and the release DAG are documented in
[slurm/README.md](slurm/README.md).

### Coreset panels — the same numbers for 1/10th the compute

Running the full benchmark is 57k generations on the Clean set and 17k on the
In-The-Wild set. Four smaller **panels** estimate the same scores from a weighted
subset, selected from the historical system × item result matrix and validated by
holding out entire model families:

| Panel | Task name | Items | vs full | Purpose |
|---|---|--:|--:|---|
| Clean Mini | `clean_mini` | 3,500 | 16× | estimate the full Clean score |
| ITW Mini | `in-the-wild_mini` | 1,919 | 8.9× | estimate the full In-The-Wild score |
| Clean Delta | `clean_delta` | 200 | 287× | detect a change between checkpoints |
| ITW Delta | `in-the-wild_delta` | 192 | 89× | detect a change between checkpoints |

The In-The-Wild panels were requested at 2,000 / 200 items; they export 1,919 / 192
after the near-duplicate filter dropped 22 / 2 of their selected items (weights
renormalized per language; no Clean panel item was affected). On the In-The-Wild set
a panel estimates the prompt similarity, not the held-out one.

On **held-out model families**, the Mini panels estimate the full benchmark to within
**0.29 / 0.20 WER_CLIP** (Clean / In-The-Wild), **0.20 / 0.13** speaker similarity, and
**0.007** MSR-UTMOS. The In-The-Wild panel picks the better of two systems 100% of the
time when they differ by ≥0.5 WER points. Quoting a panel's bootstrap CI, the full-set
value falls inside it 93–97% of the time against a nominal 95%. These validation
figures predate the near-duplicate filter and the enforced-routing rescore.

They are ordinary task directories, so nothing in the pipeline changes:

```bash
bash scripts/gen/run_omni.sh --model qwen3-tts-0.6b --task clean_mini --test-set en
bash scripts/score/run_score.sh --decode-dir results/omni_qwen3-tts-0.6b/clean_mini --test-set en

# weighted estimate of the FULL benchmark score, with a bootstrap CI
python scripts/coreset/score_panel.py --decode-dir results/omni_qwen3-tts-0.6b/clean_mini
```

Method, accuracy, and — importantly — the limits (Mini panels resolve tiers, not the
frontier; Delta panels estimate differences only) are in
[docs/coreset.md](docs/coreset.md), with the validation tables in
[RESULTS_CORESET.md](RESULTS_CORESET.md). Rebuild with `bash scripts/coreset/run.sh`
(offline, CPU-only — it reads the per-utterance scores already in `results/`):
`build_matrix.py` assembles the system × item matrix
(`assets/coreset/matrix_<set>.npz`; `--verify` reproduces the published RESULTS
tables from it), `denoise.py` low-rank-shrinks item fingerprints, `panel_select.py`
holds the selectors and coverage quotas, `validate.py` / `validate_delta.py` run
leave-family-out and matched-pair power validation, `export_panel.py` writes a panel
as a runnable `data/<task>/` directory, and `score_panel.py` turns a panel run into a
weighted estimate of the full score.

Current numbers are in [RESULTS_CURRENT.md](RESULTS_CURRENT.md).

> **Reproducibility:** the benchmark is defined in one place,
> [`utils/benchmark.py`](utils/benchmark.py): `BENCHMARK_VERSION`, the language baskets, the ASR routing
> table, and the pinned revisions of the *scoring* models (Qwen3-ASR and
> Whisper-large-v3 weights, ReDimNet2 via torch.hub, the MSR-UTMOS checkpoint,
> mHuBERT-147, Silero VAD) — plus the `sfi-utmos` git commit in `pyproject.toml`. Every score
> file is stamped with the version and scorer it came from, and
> `scripts/results_doc/build_results.py` refuses unstamped or stale inputs (leaves
> may be listed in `scripts/results_doc/excluded_leaves.txt`). The
> *generation* model ids in `scripts/gen/omni_models.sh` are best-effort
> and unpinned — pin them to a Hugging Face revision (`<id>@<commit>`) and record
> the revision if you need to reproduce a specific row of `RESULTS_CURRENT.md`. The
> CUDA 12.9 wheel pins in `pyproject.toml` can be retargeted to another CUDA build
> (see the comments there).

## Citation

ZTTS2-Eval was introduced alongside ZONOS3 Mini and builds on
[ZTTS1-Eval](https://github.com/Zyphra/ZTTS1-Eval) (introduced with ZONOS2), the
CV3-Eval / CosyVoice 3 subset design, and FLEURS-R / VoxBlink2 data. If this
benchmark is useful, please cite:

```
@misc{ztts2eval,
  title={{ZTTS2-Eval}: A Multilingual, In-the-Wild Zero-Shot TTS Benchmark},
  author={Zyphra},
  year={2026},
  howpublished={\url{https://github.com/Zyphra/ZTTS2-Eval}},
}

@misc{zyphra2026zonos3mini,
  title     = {ZONOS3 Mini Technical Report},
  author    = {Gabriel Clark and Sofian Mejjoute and Mohamed Osman and George Close and Beren Millidge},
  year      = {2026},
}

@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark and Sofian Mejjoute and Mohamed Osman and George Close and Beren Millidge},
  year      = {2026},
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@article{ma2024fleursr,
  title={{FLEURS-R}: A Restored Multilingual Speech Corpus for Generation Tasks},
  author={Ma, Min and Koizumi, Yuma and Karita, Shigeki and Zen, Heiga and Riesa, Jason and Bapna, Ankur and Rosenberg, Andrew},
  journal={arXiv preprint arXiv:2408.06227},
  year={2024}
}

@inproceedings{lin2024voxblink2,
  title={{VoxBlink2}: A 100K+ Speaker Recognition Corpus and the Open-Set Speaker-Identification Benchmark},
  author={Lin, Yuke and others},
  booktitle={Interspeech},
  year={2024}
}
```

## License

Code is licensed under the [Apache License 2.0](LICENSE) (Copyright 2025 Zyphra).
Third-party code and scoring models are acknowledged in [NOTICE](NOTICE). The
benchmark **data** has separate, dataset-specific terms — see
[DATA_LICENSE.md](DATA_LICENSE.md) (notably, the VoxBlink2 subset is
non-commercial).