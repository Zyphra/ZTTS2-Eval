# Data licensing

The **code** in this repository is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)).
The **audio and text data** under `data/` (and the prompt audio and held-out target recordings published as the Hugging Face dataset [`Zyphra/ZTTS2-Eval`](https://huggingface.co/datasets/Zyphra/ZTTS2-Eval)) are *not* covered by that license — each source dataset carries its own terms, summarized below. By using these data you agree to the upstream licenses.

> **At a glance:** the FLEURS-R–derived set is CC-BY-4.0; the VoxBlink2-derived
> set is **CC-BY-NC-SA 4.0 (non-commercial)** and is sourced from YouTube. If you
> have a commercial use case, the VoxBlink2 subset is not licensed for it.

## `data/zero_shot/` — FLEURS-R

Built from [`google/fleurs-r`](https://huggingface.co/datasets/google/fleurs-r)
(a restored version of Google FLEURS).

- **License:** Creative Commons Attribution 4.0 ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Attribution:** Ma et al., *FLEURS-R: A Restored Multilingual Speech Corpus for
  Generation Tasks*, [arXiv:2408.06227](https://arxiv.org/abs/2408.06227); built on
  FLEURS (Conneau et al.). See the citation block in the [README](README.md#citation).

## `data/zero_shot_voxblink/` — VoxBlink2

A stratified subsample of [VoxBlink2](https://voxblink2.github.io/), an in-the-wild
speaker-recognition corpus of YouTube-derived clips.

- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  ([CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) — **non-commercial
  use only**, attribution required, derivatives must be shared alike.
- **Source / privacy:** clips originate from public YouTube videos. Upstream
  distributes *annotations only* (YouTube IDs, timestamps, speaker labels) and
  leaves audio retrieval to the user. The selection manifest for this subset lives
  at `data/zero_shot_voxblink/_stage_a/manifest.tsv`; the prepared audio can be
  rebuilt from your own VoxBlink2 copy via `scripts/prepare_voxblink2_eval_stage_a.py`
  and `scripts/prepare_voxblink2_eval_stage_b.py` (see the README). For takedown or
  privacy requests, contact the VoxBlink2 maintainers (see their site).
- **Target recordings:** `data/zero_shot_voxblink/<lang>/target_wav/` holds, for each
  item, the real recording behind the target text (same speaker as the prompt, a
  different video). They are cut from the same VoxBlink2 clips and ship on
  `Zyphra/ZTTS2-Eval` as the reference for the held-out speaker-similarity score;
  they carry the same **CC-BY-NC-SA 4.0** terms, non-commercial use only. The
  `target_wav.scp` / `target_provenance.json` index files are metadata only.
- **Attribution:** Lin et al., *VoxBlink2: A 100K+ Speaker Recognition Corpus and
  the Open-Set Speaker-Identification Benchmark*, Interspeech 2024.

## Scoring models

The models used to *score* generations have their own licenses; they are not
redistributed here (they are fetched at runtime). See [`NOTICE`](NOTICE) for the
full list with source URLs and licenses.
