#!/usr/bin/env bash
# Re-run utils/average_wer.py over every published leaf's .raw (CPU only).
#
# Needed once after 2026-09-01, for two reasons that both live in average_wer.py:
#   * N_EXPECTED no longer defaults to N_SCORED when a leaf has no .meta sidecar,
#     so leaves scored before the sidecar existed stop claiming full coverage;
#   * the footer gained N_TRANSCRIBED:/N_IMPUTED:, which every leaf needs for the
#     imputed share to be readable anywhere it matters.
#
# Aggregation is a pure function of the .raw, so this rewrites .wer without
# touching a single measurement: on a leaf with no gap the output is byte-identical
# apart from the two added footer lines.
#
# A leaf below SCORE_MIN_COVERAGE still exits non-zero by design; that is reported
# at the end rather than aborting the sweep.
#
#   bash scripts/reaggregate_wer.sh [results_root]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
repo="$(pwd -P)"
PY="${repo}/.venv/bin/python"
root="${1:-results}"

# average_wer's bootstrap is already numpy-threaded; keep the outer loop serial so
# the two do not oversubscribe the node.
mapfile -t raws < <(find "${root}" -name 'wav_res_ref_text.wer.raw' -not -path '*/_*' | sort)
echo "[reagg] ${#raws[@]} leaves under ${root}"

n_ok=0; n_low=0
low_list=()
for raw in "${raws[@]}"; do
    wer="${raw%.raw}"
    if (cd utils && "${PY}" average_wer.py "${repo}/${raw}" "${repo}/${wer}" >/dev/null 2>&1); then
        n_ok=$((n_ok+1))
    else
        n_low=$((n_low+1)); low_list+=("$(dirname "${raw}")")
    fi
done

echo "[reagg] ${n_ok} ok, ${n_low} below the coverage gate:"
printf '  %s\n' "${low_list[@]}"
echo "REAGG-DONE"
