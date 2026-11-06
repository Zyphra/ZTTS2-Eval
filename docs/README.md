# docs/

Documentation and result reports for ZTTS2-Eval.

## Live (root of repo)
- [`../RESULTS_CURRENT.md`](../RESULTS_CURRENT.md) — results snapshot, regenerated from `results/`.
- [../README.md](../README.md) — overview, data, metrics, usage
- [`../DATA_LICENSE.md`](../DATA_LICENSE.md) — dataset licenses (FLEURS-R, VoxBlink2).
- [`../NOTICE`](../NOTICE) — third-party code / scoring-model attribution.

## Here
- [`pitch_range.md`](pitch_range.md) — register-normalized pitch range (split-half ceiling, 500-uid sample), per-utterance paired prosody on ITW, and the melody-carpet contour view.
- [`mpm_itw_limits.md`](mpm_itw_limits.md) — why MPM span is FLEURS-R only: measured anchor collapse on VoxBlink2, four rejected fixes, and a known loudness confound.
- [`voxblink2_rebuild.md`](voxblink2_rebuild.md) — advanced raw-corpus rebuild recipe for `data/in-the-wild/`.
- [`asr_whisper_vs_qwen.md`](asr_whisper_vs_qwen.md) — ASR-backend comparison (Qwen3-ASR vs Whisper) behind the routing rule.
- [`coreset.md`](coreset.md) — coreset panel method, accuracy, and limits (see also `RESULTS_CORESET.md`).
- [`edit_nature.md`](edit_nature.md) — what the GT edit-distance tail is made of (context for the reliability plates).
- [`rhythm.md`](rhythm.md) — mHuBERT-token rhythm metric (pause/stall and pace factors) and its perturbation validation.
- [`wer_tail_error_modes.md`](wer_tail_error_modes.md) — WER-tail error modes per language and the cross-script routing fix.
- [`controls_2026-08.md`](controls_2026-08.md) — methodology controls: ASR scorer self-favoritism (why English moved to Whisper), seen-speaker embedder, generation sampling variance, MPM loudness, with-transcript rows.
- [`unscoreable_2026-09.md`](unscoreable_2026-09.md) — why an untranscribable generation scores as maximum error (WER_CLIP 100) instead of dropping out of the mean.
- [`../scripts/results_doc/README.md`](../scripts/results_doc/README.md) — results-report pipeline.
