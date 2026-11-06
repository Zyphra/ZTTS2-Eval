# `scripts/results_doc/`

Pipeline that regenerates `RESULTS_CURRENT.md` (and the heatmap PNGs it
embeds), the reliability gallery, and the committed explorer snapshot
(`explorer/snapshot/`) from the on-disk evaluation artifacts under `results/`.

## Run

```bash
bash scripts/results_doc/run.sh
```

All paths are computed relative to this directory, so the scripts run from
anywhere. Override the interpreter with `PYTHON=...` if needed.

## Pipeline

```
results/<model>/<task>/<set>/{wav_res_ref_text.wer,spk_simi_scores.txt,spk_simi_scores.xutt.txt,msr_utmos_mean.txt}
results/{rhythm_*,pitch_range_*,paired_itw_*}.csv  results/_mpm_*/*.tsv
        │   every input must carry the current BENCHMARK_VERSION stamp (utils/benchmark.py);
        │   build_results.py refuses to build otherwise, unless the leaf is listed in
        │   excluded_leaves.txt (`<model> <task|-> <lang> <reason>`)
        ▼
build_results.py   ──►  _cache/results_snapshot.json     (one record per leaf, plus one ALL row per
                                system and panel / full set: the macro over its language basket)
build_capability_index.py ──►  _cache/capability_index.json  (ECI-style composites: capability and
                                prosody indices, one IRT index per system + 90% bootstrap CI, over
                                metric × set × language items; prosody also reads speaking_rate.json)
plot_capability_time.py   ──►  assets/plots/capability_index_time.png  (index vs public release date
                                + frontier, ECI's headline chart; dates in rd_common.RELEASES)

plot_tradeoff.py           ──►  assets/plots/tradeoff_{clean,in-the-wild}.png  (WER vs rhythm / pitch range)

gen_md.py                  ──►  RESULTS_CURRENT.md  (consumes results_snapshot.json + capability_index.json + emo_score.txt files)

scripts/plots/run_reliability.sh ──►  assets/plots/reliability_*.png + assets/reliability_gallery.html
                                (tufte plates for the GT baselines, via scripts/plots/build_reliability_gallery.py)
build_reliability.py       ──►  _cache/reliability.json     (interactive reliability payload for the explorer)
build_speaking_rate.py     ──►  _cache/speaking_rate.json   (char/s d_ref + ECDF per system)
build_mpm_dist.py          ──►  _cache/mpm_dist.json        (MPM GT-vs-system span per language)
publish_snapshot.py        ──►  explorer/snapshot/          (allowlisted aggregate JSONs — COMMITTED;
                                the shipped explorer payload on a fresh clone)
```

Shared constants (model labels, language baskets, system discovery, atomic JSON
writes) live in `rd_common.py` — builders and plots import it instead of keeping
per-file copies.

`_cache/` is intermediate and gitignored.

## Editing

- **Numbers wrong** → rebuild only what changed. e.g. after a new model run:
  `build_results.py && build_capability_index.py && gen_md.py`.
- **Markdown layout only** → just `gen_md.py`.

## Notes

- Char/s decodes and trims every wav, which used to make `build_speaking_rate.py` the
  slowest step of `run.sh` (hours). Each wav's trimmed and raw duration is now cached per
  directory (`<leaf>/speaking_rate_durations.tsv`; the references' in
  `results/_speaking_rate_ref/<set>_<lang>.tsv`), keyed on file name, size, mtime and the
  trim threshold, so a rebuild only probes wavs that changed (VoxBlink2 `en`: 219 s cold,
  22 s warm, identical output). The WER content filter is applied after the cache, so it
  never goes stale with a rescore.
- `build_capability_index.py` fits the Epoch Capabilities Index model
  ([methodology](https://epoch.ai/data/eci-documentation/methodology)):
  `performance = sigmoid(slope_item · (capability_system − difficulty_item))`, one item per
  metric × set × language, ridge-penalised least squares. Every metric is mapped to a
  performance in [0, 1] (`METRICS`); the four dimensions (intelligibility, voice, quality,
  prosody) weigh equally whatever their item counts. The `gt_*` baselines enter as one system,
  **real speech**, and the scale is anchored at real speech = 100, `zonos2_accurate` = 50
  (`ANCHORS`). CIs: 500 refits resampling each system's own items (`--bootstrap`), ~40 s on
  32 workers. Adding a system refits every item, so all indices can shift slightly.
- The prosody index is the same fit over prosody items only (`PROSODY_METRICS`): rhythm
  pace, char/s and paired duration (pace); rhythm pause and paired pause (pauses); pitch
  range and paired pitch range (pitch); MPM span (contour). The four aspects weigh equally.
  The char/s items come from `speaking_rate.json`, so build it first. A system's char/s
  item counts only for a leaf that `results_snapshot.json` scored (`scored_leaves`): the
  speaking-rate payload also keeps partial generations that the coverage gate drops. Real
  speech's char/s items are the ground-truth split-half rows `build_speaking_rate.py`
  computes for every panel. System char/s is scored only on utterances with WER ≤ 20%
  (`speaking_rate_dist.MAX_WER`; `docs/controls_2026-08.md` C7).
- Release dates for the over-time chart live in `rd_common.RELEASES`, one entry per results
  dir with its source. They are taken from each project's announcement or dated news
  log, not Hugging Face `createdAt` (repos are often created privately weeks before
  launch). A new system needs an entry there to appear on the time axis.
- Model sizes for the explorer's parameters chart live in `rd_common.PARAMS`: the published
  size of each system's main generative model (LM / backbone), with its source, plus the
  active count for mixture-of-experts models. It is the size a model is known by, not every
  weight it loads: Hugging Face repo totals mix in tokenizers and codecs inconsistently
  (VibeVoice-1.5B's repo holds 2.7B). ZONOS3 Mini's count comes from its release checkpoint.
