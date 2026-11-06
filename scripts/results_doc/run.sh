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
echo "[build_speaking_rate] speaking-rate (char/s) payload (explorer view + prosody index items)"
$PY build_speaking_rate.py
echo "[build_capability_index] ECI-style capability + prosody indices (IRT fit + bootstrap CIs)"
$PY build_capability_index.py
echo "[plot_capability_time] capability index vs public release date"
$PY plot_capability_time.py
echo "[plot_tradeoff]     WER vs rhythm / pitch-range tradeoff plots"
$PY plot_tradeoff.py
echo "[gen_md]            write RESULTS_CURRENT.md"
$PY gen_md.py
echo "[run_reliability]   reliability plates + gallery from GT .wer.raw"
PYTHON="$PY" bash "$REPO_ROOT/scripts/plots/run_reliability.sh"
echo "[build_reliability] interactive reliability payload for the explorer"
$PY build_reliability.py
echo "[build_mpm_dist]    MPM GT-vs-system distribution overlap for the explorer"
$PY build_mpm_dist.py
echo "[publish_snapshot]  copy shareable aggregates → explorer/snapshot/ (committed)"
$PY publish_snapshot.py
