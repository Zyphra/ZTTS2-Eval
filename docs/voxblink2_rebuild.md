# Rebuild the VoxBlink2 Evaluation Set

The released `data/in-the-wild/` directories are already prepared. Most
users do not need this flow; use it only if you want to reconstruct the
in-the-wild set from a local copy of the raw VoxBlink2 audio.

> **Sizing.** The pipeline below (Stage A at 2000 speakers/lang → Stage B
> round-robin across 5 MSR-UTMOS bins) samples up to ~1500 prompts/language across
> the 20 entries of `DEFAULT_LANGS` in `scripts/prep/prepare_voxblink2_eval_stage_a.py`.
> `yo` is not in the set (removed 2026-08-29: its transcripts were Nigerian
> English mislabeled by Whisper LID). The released set has 20 languages and 17,155
> prompts after the prompt/target near-duplicate filter (Stage B skips such pairs;
> `scripts/prep/drop_near_duplicates.py` removed 164 from the earlier 17,319); the 7
> largest languages keep 1,463–1,496 of the sampled 1,500. The large per-language N is deliberate:
> TTS error metrics are tail-driven (dominated by rare catastrophic generations),
> so more samples materially stabilize the estimate. Reading the audio requires the
> `chmod` step below (run as the audio-tree owner) on the remaining VoxBlink2
> speaker dirs/wavs.

`data/in-the-wild/` is built from a pre-computed candidate pool over a
local VoxBlink2 audio tree. Point `VOXBLINK2_AUDIO_ROOT` at the audio tree
(`<spk>/<vid>/<seg>.wav`). There are three steps: build the enriched candidate
pool, Stage A (select speaker cross-video pairs per language), and Stage B (score
the chosen slices with MSR-UTMOS and stratified-sample). Stage B selects prompts
**round-robin across 5 MSR-UTMOS bins spanning 1.0–4.0, up to ~1,500 prompts/language**
(one per speaker), for the 20 `DEFAULT_LANGS` entries.

```bash
export VOXBLINK2_AUDIO_ROOT=/path/to/VoxBlink2/audio
export VOXBLINK2_ENRICHED=/path/to/voxblink2_enriched.pkl

# 0. Build the enriched candidate pool (see below). Skip if you already have it.
python scripts/prep/build_voxblink2_enriched.py --xtx-json /path/to/segment_xtx_scores.json --transcripts-root /path/to/collapsed_transcripts --out "$VOXBLINK2_ENRICHED"

# Stage A: pick candidates, emit manifest + lists of dirs/files needing read access
python scripts/prep/prepare_voxblink2_eval_stage_a.py

# Grant read+exec on the speaker/video dirs and read on the wav files the manifest
# touches. Run as the audio-tree owner (or via sudo) if you lack chmod rights.
xargs -d'\n' -n1 -P16 chmod o+rX < data/in-the-wild/_stage_a/dirs_to_chmod.txt
xargs -d'\n' -n1 -P16 chmod o+r  < data/in-the-wild/_stage_a/files_to_chmod.txt

# Stage B: MSR-UTMOS-score, stratified-sample, write text/prompt_text/prompt_wav.scp/waveform/
python scripts/prep/prepare_voxblink2_eval_stage_b.py

# Extract the real target recordings behind each `text` line (the held-out
# speaker-similarity reference and the ground-truth baseline); writes
# data/in-the-wild/<lang>/target_wav/<uid>.wav
python scripts/prep/extract_voxblink_gt.py

# Validate them against the manifest and write target_wav.scp + target_provenance.json
# (same-video pairs, ambiguous prompt texts, missing wavs and duration mismatches are excluded)
python scripts/prep/build_voxblink_targets.py
```

## The enriched candidate pool

`scripts/prep/build_voxblink2_enriched.py` builds `voxblink2_enriched.pkl` —
`(rows, lang_counts)` where each row is
`(spk, vid, seg, start, end, dur, xtx, lang, lang_prob, text)`. It joins two
pre-computed artifacts, filtered to the `voxblink2/` source:

- **Cross-ASR-agreement (xtx) scores** — a flat JSON dict keyed by
  `voxblink2/<...>/audio/<spk>/<vid>/<seg>###<start>_<end>` → agreement score
  (`--xtx-json`, ~34 GB; only the
  `voxblink2/` keys are read).
- **Collapsed transcripts** — one JSON per segment under
  `<transcripts-root>/voxblink2/…/<seg>.json`, whose `pipeline_v1` entries carry,
  per VAD span, multi-ASR transcripts plus `whisper_language` /
  `whisper_language_prob` (`--transcripts-root`).

`text` is the Voxtral transcript, `lang`/`lang_prob` are the Whisper language
id/probability, and `xtx` is the agreement score. The join pre-filters on xtx and
duration (looser than Stage A's gates) so only the surviving segments' small
transcript JSONs are read. The build takes ~15 minutes and depends on filesystem
layout outside this repo; the released `data/in-the-wild/` directories are
the prepared output of this pipeline.
