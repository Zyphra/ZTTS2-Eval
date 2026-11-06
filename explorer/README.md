# ZTTS2 · Eval — reliability explorer

A zero-dependency web UI for browsing the ZTTS2-Eval TTS benchmark, styled after
[zyphra.com](https://www.zyphra.com/) (warm off-white ground, Inter throughout with
light-weight headings, JetBrains Mono for the small uppercase labels, black active
pills, one orange accent, faint horizontal gridlines). A sortable model × language leaderboard (and
heatmap) built from the existing aggregate snapshot, with drill-down into
per-utterance rows where you can **listen** to the generated audio against the
reference prompt and read the WER ref→hyp diff.

## Run

```bash
python explorer/serve.py            # serves http://localhost:8000 (loopback only)
# options:
python explorer/serve.py --port 8080 --root .
python explorer/serve.py --host 0.0.0.0   # expose on the network (serves results/ + data/)
# results/<system> symlinked into a shared store: also allow files resolving there
python explorer/serve.py --allow-root /path/to/shared/results
```

Then open <http://localhost:8000>. Stdlib only — no `pip install`, no `uv` needed.
Run it from the repo root (so `results/` and `data/` resolve), or pass `--root`.

## What it reads (nothing is recomputed)

- **Leaderboard / heatmap / reliability / speaking rate / MPM / indices:**
  the data snapshot, resolved live-first then published (see [Sharing](#sharing)) —
  `results_snapshot.json`, `reliability.json`, `speaking_rate.json`, `mpm_dist.json`,
  `capability_index.json`, and `panels.json`
  (coreset panel provenance). The board columns are WER, tail rate (`Tail>50%`,
  from `WER_TAIL50`), speaker similarity — **SPK held-out** (ITW, against the real
  recording; `spk_simi_scores.xutt.txt`) and **SPK prompt** (`spk_simi_scores.txt`) —
  MSR-UTMOS, **MPM (Clean only)**, the rhythm **pace** and **pause** factors and the
  **pitch range** factor (each as a multiple of the real-speech split-half ceiling,
  from `rhythm_<set>_<lang>.csv` / `pitch_range_<set>_<lang>.csv`), and, on the
  In-The-Wild set, the paired prosody columns (`Dur |log2|`, `Pause Δ`,
  `Range |log2|`, from `paired_itw_<lang>.csv`).
- **Reliability views** are the interactive twins of the `scripts/tufte_*` plates
  (p99 + bootstrap CI, survival overlay, small multiples, slopegraph), rendered as
  native SVG from `reliability.json` — built by
  `scripts/results_doc/build_reliability.py`, which reuses `scripts/tufte_common.py`
  so the edit counts match the plates exactly.
- **Speaking-rate views** are the interactive twin of `scripts/speaking_rate_dist.py`
  (d ref · bootstrap CI ranking, and per-system pace distributions drawn from the char/s
  quantile functions), rendered from
  `speaking_rate.json` — built by `scripts/results_doc/build_speaking_rate.py`, which
  reuses the metric itself so the char/s, W2 distances and split-half ceiling match.
  One panel per (test-set, language): char/s is only comparable within a language.
- **Per-utterance (joined by `uid`):** the leaf result dir —
  `results/<model>/<task>/<lang>/` for model generations, or the flat
  `results/<model>/<lang>/` for the GT/whisper baselines (e.g. `gt_clean/en`) —
  reading `wav.scp`, `wav_res_ref_text.wer`, `msr_utmos_scores.txt`,
  `spk_simi_scores.txt`, and optionally `pause_metadata.tsv`.
- **Targets / reference voice:** `data/<task>/<lang>/` — `text`, `prompt_text`, `prompt_wav.scp`
- **Audio:** streamed from `results/.../wavs/*.wav` (generated) and
  `data/.../waveform/*.wav` (reference) via `/api/audio` with HTTP Range support.

## Sharing

The explorer's data snapshot lives in two directories with distinct roles, so a
fresh clone renders without any of the heavy private inputs:

- **`scripts/results_doc/_cache/`** — gitignored *build scratch*. The `build_*`
  scripts write here; it mixes large derived intermediates and host-specific paths
  and is rebuilt from the full `results/` tree + the 13 MB VoxBlink manifest + the
  MSR-UTMOS cache. Never committed.
- **`explorer/snapshot/`** — the **committed**, portable payload: only the small
  aggregate JSONs the board/grid/reliability/speaking-rate/MPM/index views need
  (`results_snapshot.json`, `voxblink_perlen.json`, `reliability.json`, `speaking_rate.json`,
  `mpm_dist.json`, `capability_index.json`, `panels.json`; no host paths). This is what ships with the
  repo — the full allowlist is `PUBLISHED` in
  `scripts/results_doc/publish_snapshot.py`.

`scripts/results_doc/publish_snapshot.py` copies the curated allowlist from scratch
into the committed snapshot — an explicit, reviewable step (the last stage of
`run.sh`), so scratch is never committed by accident and the shipped data only
changes when someone publishes. `publish_snapshot.py --check` verifies they're in
sync (CI / pre-commit). `serve.py` resolves each file to the **newer** of the
scratch copy and the published copy (so an old or aborted build's scratch file
can't shadow a freshly pulled snapshot) — the same server serves fresh data
locally and the shipped snapshot on a clone, with no `.gitignore` special-casing.
`/api/meta` reports each file's chosen source + mtime under `snapshot_files`, and
the server warns on stderr when the set mixes live and published vintages (e.g.
after a partial rebuild).

Per-utterance **drill-down + audio** read the local `results/` and `data/` trees
live (mostly un-shipped: multi-GB wavs, absolute-path `.scp`), so playback works
only where those exist — the origin host or a tunnel to it, not a bare clone. The
aggregate views work everywhere.

## API

| route | returns |
|---|---|
| `GET /api/snapshot` | aggregate rows + sorted `models`/`tasks`/`sets` |
| `GET /api/meta` | row count + the option lists + `panels` (coreset provenance) + `snapshot_files` (per-file data vintage) |
| `GET /api/rows?offset=&limit=` | paginated snapshot rows — what the client actually loads the board from |
| `GET /api/utterances?model=&task=&lang=[&panel=]` | per-utterance rows joined by uid; `panel` restricts to that panel's items and attaches each weight `w` |
| `GET /api/audio?path=<repo-relative .wav>` | wav stream (206 on `Range:`, 416 past EOF) |
| `GET /api/voxblink/perlen` | passthrough of the cache JSON |
| `GET /api/reliability` | survival / p99 / slopegraph payload (tufte plates) |
| `GET /api/speaking_rate` | per-(set,lang) char/s d_ref + CI + quantile function per system |
| `GET /api/mpm_dist` | per-language MPM spans + per-system means (prosody-overlap view) |
| `GET /api/capability_index` | per-system ECI-style capability index + 90% bootstrap CI + language coverage; the prosody index is its `prosody` member, same shape (indices view) |

`/api/audio` only serves `.wav` and refuses any path that escapes `--root` (after
resolving symlinks) unless it resolves under an `--allow-root` dir; the
`model`/`task`/`lang`/`panel` query parameters are each constrained to a single
plain path component. `/api/voxblink/perlen` (the per-REF-length trend built by
`build_perlen.py`) currently has **no explorer view** — it is served and published
for the offline `plot_perlen*.py` tooling and for a future per-length view.

Any API route also answers as JSONP (`?cb=<name>`) for content-inspection proxies
that mangle large `application/json` bodies — gated on a per-process random token
(`?t=`) minted into `index.html` at serve time, so a foreign origin can't read the
data via `<script src>`.

## UI notes

- The console puts the **view** picker on its own first row, then task, language,
  **metric** and filter on one line. The board has no metric dropdown (every metric is a
  column; click a header to sort) and the grid has no language dropdown (every language
  is a column; clicking a model name opens its per-language breakdown). The metric
  dropdown drives the grid's heatmap and sets the board's sort. Gauges are normalized
  across the visible real models (GT rows excluded from scaling and tagged `GT`).
- **ALL langs** (board + grid, both full sets; the default scope): the unweighted macro over the set's
  language basket, built by `build_results.py` from the same rows as
  `RESULTS_CURRENT.md`'s headline macros, so the two never disagree. A metric gets a
  macro only where the system has it in every basket language; a system scored on fewer
  stays on the board with a `k/N langs` tag and no numbers. Clicking an ALL row lists
  the per-language pieces, worst first. The panel tasks' ALL is their weighted panel
  estimate, as before.
- **board** and **grid** use an app layout: the masthead and console stay put, the page
  does not scroll, and the table fills the rest of the window and scrolls on its own
  with its column headings pinned (the grid also pins the model column and corner
  cell). The chart views scroll as ordinary pages.
- **grid** view: model × language heatmap for the active metric (orange = worse, blue =
  better). The colour range defaults to the min–max of the per-language cells of real
  models (baselines and the ALL column excluded); the **colour range** knob beside the legend, under the grid,
  clips that range to percentiles (`clip` p5–p95 etc., against outliers) or pins an
  exact `min`/`max` per metric (`auto` resets). Cells outside the range take the end
  colour; the legend prints the range's end values.
- **indices** view: one score per system across both sets, fit like Epoch's
  Capabilities Index (item response theory over every metric × set × language item;
  `scripts/results_doc/build_capability_index.py`), ranked with its 90% bootstrap
  interval. **capability** fits every metric; **prosody** fits the prosody metrics only
  (pace, pauses, pitch and contour weighing equally). Both anchor real speech = 100 and
  `zonos2_accurate` = 50, and the ranking, over-time and parameters modes all work on
  either. Grey rows are not
  scored on every language; their index is still on the common scale, with a wider
  interval. The axis spans the point estimates, so an interval running past it is cut at
  the edge with an arrowhead (the tooltip has the full range). **over time** plots the
  same index against each system's release date (`rd_common.RELEASES`, source in
  the tooltip), with the frontier: the best system scored on every language
  released by then. Zyphra systems are orange; real speech is the dashed level at 100. **parameters** plots
  the index against each system's published model size (`rd_common.PARAMS`: the LM /
  backbone the developer names, codecs excluded; log scale) with the size frontier, the
  best system scored on every language at or below that size. Mixture-of-experts models
  sit at their total parameters; the active count is in the tooltip. Clicking a system (any
  mode) pins it and opens its **breakdown** under the chart: per dimension (with its
  metrics) and per test set, the level that group's scores alone would place it at, the
  group's weight (its share of the information the fit has about the system, which is not
  the equal per-dimension residual weighting), and points = weight × level, which add up
  to the index.
- **coreset panels** (`clean_mini`, `clean_delta`, and the VoxBlink twins —
  see [../docs/coreset.md](../docs/coreset.md)) appear as ordinary tasks, captioned
  in the picker with their kind and size. They are *not* ordinary boards:
  - Cells are the **weighted** panel estimator, `Σ w·y` over the language's selected
    items — not the leaf's `.wer` footer, which is an unweighted mean over the same
    items and so a different (wrong) estimator. `build_results.py` re-aggregates
    panel leaves from the per-uid score files to get this right; the per-language
    cells macro-average to exactly what `scripts/coreset/score_panel.py` reports.
  - WER and MSR cells carry their **95% bootstrap CI** underneath, because a
    subsample's sampling error is the point.
  - **Every row that can be derived is**. A panel is a weighted subset of its source
    task, so for any system already scored on the full task the panel estimate is that
    leaf restricted to the panel's uids — same generations, same texts, same prompts,
    which is how `scripts/coreset/validate.py` computed them offline. Without this
    backfill a panel board shows only the two or three systems someone re-ran on the
    panel task. This also covers the **GT / Whisper floors**, which are only ever
    scored on the full task: the full-set floor belongs to a different item set and
    would misstate every "vs GT" read.
    - Rows generated on the panel task itself are tagged `run`; everything else is
      derived, i.e. what the panel *would* have said, not an independent trial of it.
      On the two systems that have both, the two agree to 0.02 pt macro WER_CLIP.
    - A leaf missing part of the panel renormalises the weights over what it has, so
      it estimates a slightly different population; the covered share is shown as an
      orange `%` tag, and anything under `MIN_PANEL_COVERAGE` (90%) is dropped and
      listed on stderr rather than quietly averaged.
  - A scope note sits above the board: `mini` estimates the full-set level, `delta`
    resolves *differences only* and its level is biased by construction. Note the
    delta panels put k=200 across 35 languages, so a per-language delta cell rests on
    3–9 items and systems routinely tie on it — the delta panel's power is in the
    paired aggregate (`score_panel.py`), not in these cells.
  - The drawer follows suit — it shows only the panel's utterances (for a baseline,
    its full leaf filtered down), weights its summary, and prints each item's `w`.
    A derived mini row needs a leaf on disk: `results/<model>/<task>_mini` is then a
    symlink to the system's own `<task>` dir. That alias is for browsing only;
    `utils.benchmark.is_panel_alias` makes scoring and `build_results.py` skip it (the
    row stays derived), so delete the symlink before generating a real panel run.
- **reliability** view: the `tufte_*` plates live — pick a dataset (TTS systems, read
  speech or in the wild, listed first and opened by default; then the GT sets FLEURS-R /
  VoxBlink2 / ASR-backend) and a chart type (p99 · CI, survival overlay, small
  multiples, slopegraph). In the survival overlay, click a legend chip to drop or
  restore a language; best / least-reliable / most-catastrophic are auto-highlighted.
- **speaking rate** view: `scripts/speaking_rate_dist.py` live — pick the test set,
  then the language within it (the reliability view's two-row idiom; there are ~50
  panels, too many for one chip row), then a chart type. Switching test set keeps the
  current language where that set has it. **d · CI** ranks systems by how
  far their char/s distribution sits from the reference voice (green rule = the
  split-half ceiling, a perfect system at finite n). **distribution** gives each system
  one row on a shared char/s axis: a density ridge (estimated from its quantiles) for
  the shape, a p25–p75 bar with p10/p90 whiskers and a median tick, the reference
  voice's middle half and median as a green band and line through every row (its shape
  outlined faintly in each), and median char/s, % vs the natural median and d ×ceil on
  the right. Rows sort fastest-first or closest-first (by d ref). The axis spans every
  row's p10–p90, so one extreme system cannot squash the rest; ridge tails beyond it are
  clipped. The reference's split-half twin (`gt_* · reference half`) is labelled apart
  from the GT system. Hover a row for its quantiles, click to pin. NB char/s is
  never comparable across languages, so each panel is a single language. The **all**
  chip of each set (the view opens on Clean's) ranks systems by d ref averaged over the set's languages (only
  systems scored on every one), with no interval. The chart-type switch stays, with
  **distribution** greyed out (tooltip says why) until a language is picked; the last
  choice comes back with it.
- **prosody overlap** view: MPM span ranks systems against per-language anchors;
  0 is the real-speech split-half floor and 1 is the contour-scrambled floor. The
  chart is one row per system, so it sizes its own SVG (`newSvg`, inline width/height)
  — a bare `<svg>` inside `.scope-chart` is pinned to 440px by CSS, which a viewBox
  honours by scaling the whole drawing, text and all, down to fit.
- Click any row/cell to open the drill-down drawer; expand an utterance for the
  ASR diff, prompt transcript, extra measures, and the two waveform players.

Files: `serve.py` (server) · `index.html` · `app.css` (Zyphra theme) ·
`app.js` (board/grid/drawer) · `chart_common.js` (shared chart kit:
tooltip/focus/hit/loader/resize + SVG helpers) · `reliability.js` (reliability
views) · `speaking_rate.js` (speaking-rate views) · `mpm.js` (prosody-overlap
view) · `capability.js` (capability-index views).
