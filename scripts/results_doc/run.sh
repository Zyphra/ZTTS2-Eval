#!/usr/bin/env bash
# Regenerate RESULTS_CURRENT.md end-to-end.
#
# Build steps fill scripts/results_doc/_cache/ with JSON intermediates;
# plot steps emit PNGs into assets/plots/; gen_md.py writes RESULTS_CURRENT.md
# at the repo root. Each step is idempotent — safe to re-run.

set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

# Default to the project venv if present; override with PYTHON=...
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY="python3"
fi
echo "Using interpreter: $PY"

echo "[build_results]     snapshot results/ → _cache/results_snapshot.json"
$PY build_results.py
echo "[build_perbin]      voxblink MSR-UTMOS-bin reconstruction"
$PY build_perbin.py
echo "[build_perlen]      voxblink per-length-bucket WER (simple)"
$PY build_perlen.py
echo "[build_perlen_weighted] voxblink per-length WER (weighted ∑edits/∑ref)"
$PY build_perlen_weighted.py
echo "[build_perlen_zeroshot] same bucketing on zero_shot (controlled content)"
$PY build_perlen_zeroshot.py
echo "[plot_perbin]       voxblink per-MSR-UTMOS-bin heatmaps"
$PY plot_perbin.py
echo "[plot_perlen]       voxblink per-text-length heatmaps"
$PY plot_perlen.py
echo "[plot_perlen_mean]  voxblink cross-lang mean curves (simple+weighted)"
$PY plot_perlen_mean.py
echo "[plot_perlen_zeroshot] voxblink vs zero_shot side-by-side"
$PY plot_perlen_zeroshot.py
echo "[plot_dswed_dists]  DS-WED per-utt distribution (KDE + boxplot)"
$PY "$REPO_ROOT/scripts/plot_dswed_dists.py"
echo "[gen_md]            write RESULTS_CURRENT.md"
$PY gen_md.py
echo "[run_reliability]   reliability plates + gallery from GT .wer.raw"
PYTHON="$PY" bash "$REPO_ROOT/scripts/run_reliability.sh"
echo "[build_reliability] interactive reliability payload for the explorer"
$PY build_reliability.py
echo "[build_speaking_rate] speaking-rate (char/s) payload for the explorer"
$PY build_speaking_rate.py
echo "[build_mpm_dist]    MPM GT-vs-system distribution overlap for the explorer"
$PY build_mpm_dist.py
echo "[publish_snapshot]  copy shareable aggregates → explorer/snapshot/ (committed)"
$PY publish_snapshot.py
