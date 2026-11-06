#!/usr/bin/env bash
# Score one language on a reproducible random 50/50 ground-truth split.
set -euo pipefail

: "${METRICS_REPO:?METRICS_REPO is required}"
: "${METRICS_KIND:?METRICS_KIND is required}"
: "${METRICS_MANIFEST:?METRICS_MANIFEST is required}"

cd "${METRICS_REPO}"
mapfile -t rows < "${METRICS_MANIFEST}"
row="${rows[${SLURM_ARRAY_TASK_ID}]:-}"
if [ -z "${row}" ]; then
    echo "missing manifest row ${SLURM_ARRAY_TASK_ID} in ${METRICS_MANIFEST}" >&2
    exit 1
fi
IFS=$'\t' read -r task lang <<< "${row}"
ref_dir="data/${task}/${lang}/waveform"
seed="${GT_SPLIT_SEED:-0}"
if [ ! -d "${ref_dir}" ]; then
    echo "missing reference directory ${ref_dir}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "${METRICS_KIND}" in
    speaking_gt)
        : "${METRICS_SHARD_DIR:?METRICS_SHARD_DIR is required for speaking_gt}"
        mkdir -p "${METRICS_SHARD_DIR}"
        .venv/bin/python scripts/results_doc/build_speaking_rate.py \
            --sets "${task}" --langs "${lang}" --ground-truth-only --seed "${seed}" \
            --workers "${SLURM_CPUS_PER_TASK:-28}" --n-boot 400 \
            --out "${METRICS_SHARD_DIR}/${task}_${lang}.json"
        ;;

    mpm_gt)
        py="${TTSDS_VENV:-$HOME/.venvs/ttsds}/bin/python"
        if [ ! -x "${py}" ]; then
            echo "missing TTSDS interpreter ${py}" >&2
            exit 1
        fi
        work=$(mktemp -d "${TMPDIR:-/tmp}/ztts-gt-split.XXXXXX")
        trap 'rm -rf "${work}"' EXIT
        "${py}" utils/ground_truth_split.py \
            --src "${ref_dir}" --left "${work}/left" --right "${work}/right" \
            --seed "${seed}" --key "${task}/${lang}"
        out="results/_mpm_ground_truth/${task}_${lang}.tsv"
        mkdir -p "${out%/*}"
        "${py}" scripts/prosody/score_mpm_shuffle_floor.py \
            --ref-dir "${work}/right" \
            --synth-dir "${work}/left" --label "gt_${task}" \
            --out "${out}" --seed "${seed}"
        ;;

    *)
        echo "unknown METRICS_KIND=${METRICS_KIND}" >&2
        exit 1
        ;;
esac
