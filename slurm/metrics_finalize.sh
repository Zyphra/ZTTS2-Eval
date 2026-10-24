#!/usr/bin/env bash
# Merge successful metric arrays and publish the explorer/report payloads.
set -euo pipefail

: "${METRICS_REPO:?METRICS_REPO is required}"
: "${METRICS_SHARD_DIR:?METRICS_SHARD_DIR is required}"
cd "${METRICS_REPO}"

.venv/bin/python scripts/results_doc/build_speaking_rate.py \
    --merge-shards "${METRICS_SHARD_DIR}"
.venv/bin/python scripts/results_doc/build_mpm_dist.py
.venv/bin/python scripts/results_doc/build_results.py
.venv/bin/python scripts/results_doc/plot_tradeoff.py
.venv/bin/python scripts/results_doc/gen_md.py
.venv/bin/python scripts/results_doc/publish_snapshot.py
.venv/bin/python scripts/results_doc/publish_snapshot.py --check
