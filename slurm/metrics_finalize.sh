#!/usr/bin/env bash
# Merge successful metric arrays and publish the explorer/report payloads.
# METRICS_FINALIZE_MODE=gt (set by slurm/submit_metrics.sh when every requested kind is a
# ground-truth kind) overlays the GT speaking-rate shards onto the published payload
# instead of rebuilding it from the system shards alone.
set -euo pipefail

: "${METRICS_REPO:?METRICS_REPO is required}"
: "${METRICS_SHARD_DIR:?METRICS_SHARD_DIR is required}"
cd "${METRICS_REPO}"

merge_args=(--merge-shards "${METRICS_SHARD_DIR}")
if [ "${METRICS_FINALIZE_MODE:-system}" = gt ]; then
    merge_args+=(--merge-base explorer/snapshot/speaking_rate.json)
fi
.venv/bin/python scripts/results_doc/build_speaking_rate.py "${merge_args[@]}"
.venv/bin/python scripts/results_doc/build_mpm_dist.py
.venv/bin/python scripts/results_doc/build_results.py
.venv/bin/python scripts/results_doc/plot_tradeoff.py
.venv/bin/python scripts/results_doc/gen_md.py
.venv/bin/python scripts/results_doc/publish_snapshot.py
.venv/bin/python scripts/results_doc/publish_snapshot.py --check
