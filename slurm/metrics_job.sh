#!/usr/bin/env bash
# Array worker for the retained distributional metrics.
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
if [ ! -d "${ref_dir}" ]; then
    echo "missing reference directory ${ref_dir}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

has_audio() {
    local wav size
    shopt -s nullglob
    for wav in "$1"/*.wav; do
        size=$(stat -Lc '%s' "${wav}" 2>/dev/null) || continue
        if [ "${size}" -gt 44 ]; then
            return 0
        fi
    done
    return 1
}

case "${METRICS_KIND}" in
    speaking)
        : "${METRICS_SHARD_DIR:?METRICS_SHARD_DIR is required for speaking}"
        mkdir -p "${METRICS_SHARD_DIR}"
        out="${METRICS_SHARD_DIR}/${task}_${lang}.json"
        .venv/bin/python scripts/results_doc/build_speaking_rate.py \
            --sets "${task}" --langs "${lang}" \
            --workers "${SLURM_CPUS_PER_TASK:-28}" --n-boot 400 \
            --out "${out}"
        ;;

    pitch)
        py="${TTSDS_VENV:-$HOME/.venvs/ttsds}/bin/python"
        if [ ! -x "${py}" ]; then
            echo "missing TTSDS interpreter ${py}" >&2
            exit 1
        fi
        shopt -s nullglob
        wav_dirs=(results/*/"${task}"/"${lang}"/wavs)
        if [ "${#wav_dirs[@]}" -eq 0 ]; then
            echo "no systems for ${task}/${lang}" >&2
            exit 1
        fi
        for wav_dir in "${wav_dirs[@]}"; do
            if ! has_audio "${wav_dir}"; then
                echo "[skip] empty ${wav_dir}"
                continue
            fi
            rel="${wav_dir#results/}"
            model="${rel%%/*}"
            leaf="${wav_dir%/wavs}"
            out="${leaf}/ttsds_pitch.csv"
            if [ -s "${out}" ]; then
                echo "[skip] ${model} ${task}/${lang}: ${out}"
                continue
            fi
            "${py}" utils/score_ttsds_pitch.py \
                --synth-dir "${wav_dir}" \
                --ref-dir "${ref_dir}" \
                --label "${model}" \
                --ref-label "ref_${task}_${lang}" \
                --out "${out}"
        done
        ;;

    mpm)
        py="${TTSDS_VENV:-$HOME/.venvs/ttsds}/bin/python"
        if [ ! -x "${py}" ]; then
            echo "missing TTSDS interpreter ${py}" >&2
            exit 1
        fi
        shopt -s nullglob
        wav_dirs=(results/*/"${task}"/"${lang}"/wavs)
        if [ "${#wav_dirs[@]}" -eq 0 ]; then
            echo "no systems for ${task}/${lang}" >&2
            exit 1
        fi
        args=()
        for wav_dir in "${wav_dirs[@]}"; do
            if ! has_audio "${wav_dir}"; then
                echo "[skip] empty ${wav_dir}"
                continue
            fi
            rel="${wav_dir#results/}"
            model="${rel%%/*}"
            args+=(--synth-dir "${wav_dir}" --label "${model}")
        done
        if [ "${#args[@]}" -eq 0 ]; then
            echo "no non-empty systems for ${task}/${lang}" >&2
            exit 1
        fi
        mkdir -p results/_mpm_shuffle_floor
        "${py}" utils/score_mpm_shuffle_floor.py \
            --ref-dir "${ref_dir}" \
            "${args[@]}" \
            --out "results/_mpm_shuffle_floor/${task}_${lang}.tsv" \
            --dist-out "results/_mpm_shuffle_floor/${lang}_dist.json" \
            --lang "${lang}"
        ;;

    *)
        echo "unknown METRICS_KIND=${METRICS_KIND}" >&2
        exit 1
        ;;
esac
