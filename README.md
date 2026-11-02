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
sounds like *real speech*. This is **benchmark version 2**: every score is stamped
with it and with the pinned scorer revisions it was computed with
([`utils/benchmark.py`](utils/benchmark.py)).

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
|Data distribution|prompt audio in git|metadata in git, prompt audio on Hugging Face via `scripts/download_data.py`|[Data](#data)|

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
  `data/zero_shot/<lang>/`. The prompt is a duration-filtered clip; the target
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
  at `data/zero_shot_voxblink/<lang>/`. The large
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
the prep scripts skip such pairs, and `python scripts/drop_near_duplicates.py`
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

The audio (`waveform/`, 24 GB, plus the In-The-Wild `target_wav/` recordings) is not in git: it is the Hugging Face dataset [`Zyphra/ZTTS2-Eval`](https://huggingface.co/datasets/Zyphra/ZTTS2-Eval). Fetch it into place with `python scripts/download_data.py` (both sets), or a subset with `--tasks zero_shot_voxblink --langs en zh`; for the In-The-Wild set the script also fetches `target_wav/` and verifies that every wav listed in `prompt_wav.scp` and `target_wav.scp` resolves. The `target_wav.scp` index and `target_provenance.json` (pairing rule, counts, exclusions) are in git. The coreset panel directories symlink into these trees.

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
  geometry as the ceiling. Only utterances the ASR heard with WER ≤ 20% are scored, so
  a dropped or runaway sentence counts against intelligibility, not pace
  ([controls C7](docs/controls_2026-08.md)). Lower is better; the reference split-half
  distance is the finite-sample ceiling.
* **Rhythm** — per-utterance timing statistics of the run-length distribution of
  deduplicated mHuBERT-147 k-means tokens, scored as the 2-Wasserstein distance to
  the reference at split-half geometry and reported as a multiple of the real-speech
  split-half ceiling (`× ceiling`; 1 = as close to real speech as two halves of real
  speech are to each other). Two dimensions are reported: the **pause/stall factor**
  (burstiness of the run lengths) and the **pace factor** (token rate). Lower is
  better. See [docs/rhythm.md](docs/rhythm.md).
* **Pitch range** — per-utterance register-normalized pitch range: p90 − p10 of the
  voiced F0 (pYIN) in semitones around the utterance's own median, so it measures
  how much the melody moves and ignores whether the voice is high or low. Scored as
  the 2-Wasserstein distance to the prompt recordings at split-half geometry, in
  `× ceiling`, on a fixed 500-utterance sample per (set, language). Lower is better.
  See [docs/pitch_range.md](docs/pitch_range.md).
* **Paired prosody (In-The-Wild only)** — every In-The-Wild item has a real
  recording of the same speaker saying the target sentence, so prosody is also
  compared utterance by utterance against it (same 500-utterance sample): median
  |log2 duration ratio| (on the generations with WER ≤ 20%, as for speaking rate),
  mean |pause-count difference| (a pause is a gap ≥ 0.2 s between Silero VAD speech
  segments, which still finds pauses under background noise), and median
  |log2 pitch-range ratio|. 0 means the generation matches the real recording; lower
  is better.
* **MPM span (Clean set only)** — Masked Prosody Model distance normalized per
  language between the real-speech split-half floor (0) and a contour-scrambled
  floor (1). Lower is better, and **values above 1 are worse than the same speech
  with its contour scrambled**. Leading/trailing silence is trimmed from every
  utterance first, since the FLEURS-R recordings carry ~2.45 s of it and most
  systems almost none ([controls C6](docs/controls_2026-08.md)). It is not computed
  on the In-The-Wild set: the anchors collapse on in-the-wild audio
  ([docs/mpm_itw_limits.md](docs/mpm_itw_limits.md)).

The `gt_zero_shot*` rows use a reproducible, disjoint random 50/50 split of each
reference corpus rather than comparing ground truth with itself. Seed and
language determine the split; `slurm/submit_ground_truth_metrics.sh` refreshes
the char/s and MPM halves. For rhythm and pitch range the split-half ceiling is the
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
python scripts/download_data.py --tasks zero_shot_voxblink --langs en

# 3. generate the English slice of the In-The-Wild Mini coreset panel (50 items)
#    with one local model
bash run_generate_omni.sh --model qwen3-tts-0.6b --task zero_shot_voxblink_mini --test_set en

# 4. score that DECODE_DIR (script name is historical and model-agnostic). It
#    auto-starts/stops the ASR servers and routes each language to Qwen3-ASR or
#    Whisper-large-v3 as the published numbers do; both co-locate on GPU 0 by
#    default (same settings as slurm/eval_job.sh). It also scores speaker
#    similarity against the held-out real recordings.
DECODE_DIR=results/omni_qwen3-tts-0.6b/zero_shot_voxblink_mini \
    bash run_infer_voxblink_eval.sh --task zero_shot_voxblink_mini --test_set en

# 5. weighted estimate of the FULL In-The-Wild score, with a bootstrap CI
python scripts/coreset/score_panel.py --decode_dir results/omni_qwen3-tts-0.6b/zero_shot_voxblink_mini

# 6. browse results in the explorer
python scripts/results_doc/build_results.py    # builds the explorer snapshot from results/
python explorer/serve.py --port 8000           # open http://localhost:8000
```

> **Explorer note:** the explorer prefers the live build cache
> (`scripts/results_doc/_cache/results_snapshot.json`, generated from your
> `results/` tree in step 6) and falls back to the **committed** snapshot in
> `explorer/snapshot/` — so a fresh clone serves a populated explorer out of the
> box. Data outside the published allowlist (see
> `scripts/results_doc/publish_snapshot.py`) appears only after a local build.
> `build_results.py` refuses score files that are not stamped with the current
> benchmark version, so score every leaf with the scripts above rather than
> copying score files in.

The bundled explorer (`python explorer/serve.py`) provides a leaderboard
sortable by any metric, a model × language heatmap, VoxBlink2 quality-bin
curves, and a per-utterance drill-down with audio playback and ASR diffs.
[Coreset panels](docs/coreset.md) show up as their own tasks, scored with their
published item weights and captioned with what they can and cannot resolve.

Data-prep and evaluation details follow. The shipped data directories are
already prepared (metadata in git, audio via `scripts/download_data.py`); the raw-corpus rebuild recipe is in
[docs/voxblink2_rebuild.md](docs/voxblink2_rebuild.md).

## Data preparation

The FLEURS-R set can be (re)built from
[`google/fleurs-r`](https://huggingface.co/datasets/google/fleurs-r):

```bash
uv run python scripts/prepare_fleurs_r.py --overwrite
```

This downloads FLEURS-R for all 35 curated languages and writes `text`,
`prompt_text`, `prompt_wav.scp`, and `waveform/` into each language directory. By
default each language is sampled to the maximum its pooled train+test splits allow
(prompts and targets stay disjoint, and near-duplicate prompt/target pairs are
skipped); pass `--count N` to cap it, or `--langs` to
restrict the set. It requires HuggingFace access to `google/fleurs-r`
(`huggingface-cli login` or `--hf-token`).

### VoxBlink2 set

The released `data/zero_shot_voxblink/` directories are already prepared: they
include the target/reference text, manifest, the `target_wav.scp` /
`target_provenance.json` index of the held-out recordings, and the MSR-UTMOS binning
cache used by the evaluation scripts; the prompt audio and the held-out target
recordings (`target_wav/`) come from `scripts/download_data.py`.
Rebuilding this set from raw VoxBlink2 requires a
local VoxBlink2 audio tree and a pre-computed enriched candidate pool, so the full
recipe lives in [docs/voxblink2_rebuild.md](docs/voxblink2_rebuild.md).

## Evaluation

Evaluation has two complementary layers: the standard intelligibility, speaker
match, and MOS pass, plus a prosody pass that measures whether the generated
corpus keeps the reference distribution's speaking-rate, rhythm, pitch-range, and
contour behavior.

```bash
# remember to point DECODE_DIR at your generated audio in the run scripts

# WER, speaker similarity, and MSR-UTMOS — Clean set (FLEURS-R).
# The script name is historical; it scores any compatible DECODE_DIR.
# (auto-starts and stops the ASR servers — Qwen3-ASR plus Whisper-large-v3 for the
#  routed languages; the run fails if the Whisper server cannot start)
bash run_infer_zonos2_eval.sh

# Same metrics on the In-The-Wild set (VoxBlink2); speaker similarity is also
# scored against the held-out real recordings (spk_simi_scores.xutt.txt)
bash run_infer_voxblink_eval.sh
```

```bash
# Characters/s distributional distance (defaults to zero_shot/en)
python scripts/speaking_rate_dist.py
python scripts/speaking_rate_dist.py --test-set zero_shot_voxblink --lang en

# mHuBERT rhythm (GPU): pause/stall and pace factors vs the real-speech ceiling
python scripts/rhythm_dist.py --test-set zero_shot_voxblink --lang en

# register-normalized pitch range (CPU; fixed 500-utterance sample)
python scripts/pitch_range_dist.py --test-set zero_shot_voxblink --lang en

# per-utterance paired prosody vs the real recording (In-The-Wild only; run after
# pitch_range_dist.py, it shares the pitch caches)
python scripts/paired_itw.py --lang en

# UTMOS shift: signed mean shift (dmean, with CI and the reference split-half band)
# and spread ratio of a system's per-utterance MSR-UTMOS distribution vs. the
# reference's (needs the msr_utmos_scores.txt leaves); W2 kept as a secondary column
python scripts/utmos_shift.py --test-set zero_shot_voxblink --lang en

# MSR-UTMOS level-sensitivity probe (gain sweep); writes assets/probes/utmos_level.csv
python scripts/probe_utmos_level.py

# MPM span against the real/scrambled anchors (Clean set only)
python utils/score_mpm_shuffle_floor.py \
    --ref-dir data/zero_shot/en/waveform \
    --synth-dir results/omni_qwen3-tts-1.7b/zero_shot/en/wavs \
    --out results/_mpm_shuffle_floor/zero_shot_en.tsv
```

The explorer exposes characters/s distance and MPM span as dedicated
distribution views.

**Version stamps and rescoring.** Every score file records what produced it: the
`.wer` footer carries `BENCHMARK_VERSION` and `ASR` (model@revision), the
`spk_simi_scores*.txt` files `#BENCHMARK_VERSION` and `#SPK_MODEL`, and
`msr_utmos_ci.txt` `BENCHMARK_VERSION` and `MSR_UTMOS_CKPT`. The scoring drivers
treat a leaf with a missing or stale stamp as unscored and redo it;
`python3 -m utils.benchmark problems <leaf_dir> <lang>` lists why a leaf is stale.
`scripts/rescore_all.sh` rescores every leaf under `results/`, and
`slurm/submit_release_run.sh` is the whole release DAG on SLURM (see
[slurm/README.md](slurm/README.md)).

Because TTS error metrics are tail-driven, the ground-truth baselines ship with a
**reliability gallery** — edit-distance survival/tail plates per language (and per
ASR backend) that show *how often* an utterance goes badly wrong and *by how much*.
Regenerate the plates and the self-contained page with `bash
scripts/run_reliability.sh` (also run late in `scripts/results_doc/run.sh`,
before the explorer payload builders and `publish_snapshot.py`);
open [`assets/reliability_gallery.html`](assets/reliability_gallery.html). Context
in [docs/edit_nature.md](docs/edit_nature.md).

### Coreset panels — the same numbers for 1/10th the compute

Running the full benchmark is 57k generations on the Clean set and 17k on the
In-The-Wild set. Four smaller **panels** estimate the same scores from a weighted
subset, selected from the historical system × item result matrix and validated by
holding out entire model families:

| Panel | Task name | Items | vs full | Purpose |
|---|---|--:|--:|---|
| Clean Mini | `zero_shot_mini` | 3,500 | 16× | estimate the full Clean score |
| ITW Mini | `zero_shot_voxblink_mini` | 1,919 | 8.9× | estimate the full In-The-Wild score |
| Clean Delta | `zero_shot_delta` | 200 | 287× | detect a change between checkpoints |
| ITW Delta | `zero_shot_voxblink_delta` | 192 | 89× | detect a change between checkpoints |

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
bash run_generate_omni.sh --model qwen3-tts-0.6b --task zero_shot_mini --test_set en
DECODE_DIR=results/omni_qwen3-tts-0.6b/zero_shot_mini \
    bash run_infer_zonos2_eval.sh --task zero_shot_mini --test_set en

# weighted estimate of the FULL benchmark score, with a bootstrap CI
python scripts/coreset/score_panel.py --decode_dir results/omni_qwen3-tts-0.6b/zero_shot_mini
```

Method, accuracy, and — importantly — the limits (Mini panels resolve tiers, not the
frontier; Delta panels estimate differences only) are in
[docs/coreset.md](docs/coreset.md), with the validation tables in
[RESULTS_CORESET.md](RESULTS_CORESET.md). Rebuild with `bash scripts/coreset/run.sh`
(offline, CPU-only — it reads the per-utterance scores already in `results/`).

Generation scripts are provided for local/open models (`run_generate_zonos2.sh`,
`run_generate_zonos3.sh`, `run_generate_omni.sh`) and for hosted providers. For the In-The-Wild set,
generation scripts are provided as siblings of the FLEURS-R ones
(`run_generate_*_voxblink.sh`); they pin `task=zero_shot_voxblink`.

* **`run_generate_omni.sh`** is self-contained — it starts/stops a local
  vLLM-Omni server (`uv sync --extra omni`) and is the recommended starting point.
* **`run_generate_zonos2.sh`** now **starts/stops its own** Zonos2 Mini-SGLang
  server too (`scripts/serve_zonos2.sh` → `python -m zonos2`), the same shape as
  the omni path — just run it (optionally `--gpu N --port N`). The server code
  lives in a separate checkout (set `ZONOS2_HOME` to a [ZONOS2](https://github.com/Zyphra/ZONOS2) checkout
  with its own venv); health check `GET /tts/capabilities`. Pass
  `--server http://HOST:PORT` to reuse an already-running external server instead
  of launching one.
  Sampling defaults match the released Zonos2 model's reference eval config
  (temperature 1.15, topk 106, min-p 0.18; speaking-rate and server-side text
  normalization off — re-enable with `--speaking_rate_bucket N` /
  `--text_norm true`). Zonos2 has two modes: **stable/accurate** (`--accurate_mode
  true`, closer voice match) and **expressive** (`--accurate_mode false`); the
  server defaults to accurate. Under SLURM, `slurm/submit_eval.sh --engine zonos2
  --accurate_mode {true|false}` runs each mode as its own job, writing to
  `results/zonos2_accurate` / `results/zonos2_expressive`.
* **`run_generate_zonos3.sh`** generates ZONOS3 Mini through the release minisgl server
  (`scripts/serve_zonos3.sh` → `python -m zonos2` from the `ZONOS3_HOME` checkout,
  optional site env `ZONOS3_ENV`; client `scripts/generate_zonos3.py` → `POST
  /tts/generate`), one conditioning mode per call: `--speaker_mode reference` (raw
  reference prefix) or `--speaker_mode embedding` (speaker embedding only) →
  `results/zonos3-mini_<mode>`. It sends the release server's sampling defaults
  explicitly, seed 0, server TN for en zh de es fr it pt ja ko (every other language
  raw, TN off); `--laci true` adds LACI decoding on a server started with
  `--laci-heads` (via `ZONOS3_SERVE_ARGS`). Self-managed by default, `--server URL` for
  an external one. `slurm/zonos3_gen_job.sh` runs both modes over whole 8-GPU nodes
  (one server per GPU; see [slurm/README.md](slurm/README.md)).

### Hosted providers

Each hosted-provider script installs via an extra and reads its API key from an
environment variable; install only what you need.

| Script | Extra | Env var |
| --- | --- | --- |
| `run_generate_gemini.sh` | `--extra gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `run_generate_elevenlabs.sh` (+ `_clone`) | `--extra elevenlabs` | `ELEVENLABS_API_KEY` |
| `run_generate_cartesia.sh` | `--extra cartesia` | `CARTESIA_API_KEY` |
| `run_generate_inworld.sh` | `--extra inworld` | `INWORLD_API_KEY` |

`run_generate_elevenlabs_clone.sh` is the instant-voice-cloning variant (clones
the prompt voice before synthesis) of `run_generate_elevenlabs.sh`. See
[`.env.example`](.env.example) for the full set of environment variables the
scripts read.

Current numbers are in [RESULTS_CURRENT.md](RESULTS_CURRENT.md); the repository
structure is mapped in [ARCHITECTURE.md](ARCHITECTURE.md).

> **Reproducibility:** this is benchmark version 2, defined in one place,
> [`utils/benchmark.py`](utils/benchmark.py): the language baskets, the ASR routing
> table, and the pinned revisions of the *scoring* models (Qwen3-ASR and
> Whisper-large-v3 weights, ReDimNet2 via torch.hub, the MSR-UTMOS checkpoint,
> mHuBERT-147) — plus the `sfi-utmos` git commit in `pyproject.toml`. Every score
> file is stamped with the version and scorer it came from, and
> `scripts/results_doc/build_results.py` refuses unstamped or stale inputs (leaves
> may be listed in `scripts/results_doc/excluded_leaves.txt`). The
> *generation* model ids in `run_generate_omni.sh` (`model_spec`) are best-effort
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