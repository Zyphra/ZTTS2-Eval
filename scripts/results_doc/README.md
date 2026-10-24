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
build_results.py   ──►  _cache/results_snapshot.json     (one record per leaf)

data/zero_shot_voxblink/_stage_a/manifest.tsv + _msr_utmos_cache.tsv
        │
        ▼
build_perbin.py    ──►  _cache/voxblink_uid_bin.json
                        _cache/voxblink_perbin.json       (per-(model,lang,MSR-UTMOS-bin) WER)
                        _cache/voxblink_bin_counts.json

build_perlen.py            ──►  _cache/voxblink_perlen.json           (per-text-length bucket, simple mean)
build_perlen_weighted.py   ──►  _cache/voxblink_perlen_weighted.json  (∑edits/∑ref_len — the right cross-length aggregate)
build_perlen_zeroshot.py   ──►  _cache/zeroshot_perlen_{simple,weighted}.json   (same bucketing on the controlled-content zero_shot set)

plot_perbin.py             ──►  assets/plots/voxblink_perbin_heatmaps.png
plot_perlen.py             ──►  assets/plots/voxblink_perlen_heatmaps.png
plot_perlen_mean.py        ──►  assets/plots/voxblink_perlen_crosslang.png
plot_perlen_zeroshot.py    ──►  assets/plots/perlen_zeroshot_vs_voxblink.png
plot_tradeoff.py           ──►  assets/plots/tradeoff_{zero_shot,zero_shot_voxblink}.png  (WER vs rhythm / pitch range)

gen_md.py                  ──►  RESULTS_CURRENT.md  (consumes results_snapshot.json + voxblink_perbin.json + voxblink_bin_counts.json + emo_score.txt files)

scripts/run_reliability.sh ──►  assets/plots/reliability_*.png + assets/reliability_gallery.html
                                (tufte plates for the GT baselines, via scripts/build_reliability_gallery.py)
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
  `build_results.py && gen_md.py`.
- **Voxblink bin/length analysis changed** → rerun the matching `build_*` and
  `plot_*` pair, then `gen_md.py`.
- **Markdown layout only** → just `gen_md.py`.

## Notes

- `build_perbin.py` reconstructs the MSR-UTMOS-bin assignment by mirroring the
  selection algorithm in `scripts/prepare_voxblink2_eval_stage_b.py` against the
  cached manifest. Defaults: 5 bins, MSR-UTMOS range [1.0, 4.0] (bin width 0.6).
- The per-length voxblink trend (WER falling monotonically with REF length) is a
  **content-selection confound** in voxblink2 — see the memory note
  `voxblink-length-confound.md`. The real signal lives on `zero_shot` and is a
  U-shape, plotted by `plot_perlen_zeroshot.py`.
