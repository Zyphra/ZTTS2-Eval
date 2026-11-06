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

    mpm)
        py="${TTSDS_VENV:-$HOME/.venvs/ttsds}/bin/python"
        if [ ! -x "${py}" ]; then
            echo "missing TTSDS interpreter ${py}" >&2
            exit 1
        fi
        shopt -s nullglob
        # Score only the uids still in data/<task>/<lang>/text: systems' wav dirs also
        # hold generations for items removed by the near-duplicate filter. Symlink the
        # kept ones into a scratch dir per system.
        work=$(mktemp -d "${TMPDIR:-/tmp}/ztts-mpm.XXXXXX")
        trap 'rm -rf "${work}"' EXIT
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
            filt="${work}/${model}/wavs"
            mkdir -p "${filt}"
            .venv/bin/python - "${wav_dir}" "data/${task}/${lang}/text" "${filt}" <<'PYEOF'
import os, sys
src, text, dst = sys.argv[1:4]
with open(text, encoding="utf-8") as f:
    uids = [ln.split(None, 1)[0] for ln in f if ln.strip()]
for u in uids:
    p = os.path.join(src, u + ".wav")
    if os.path.exists(p):
        os.symlink(os.path.realpath(p), os.path.join(dst, u + ".wav"))
PYEOF
            args+=(--synth-dir "${filt}" --label "${model}")
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

    rhythm)
        .venv/bin/python scripts/rhythm_dist.py \
            --test-set "${task}" --lang "${lang}" \
            --workers "${SLURM_CPUS_PER_TASK:-16}"
        ;;

    pitch_range)
        .venv/bin/python scripts/pitch_range_dist.py \
            --test-set "${task}" --lang "${lang}" \
            --workers "${SLURM_CPUS_PER_TASK:-28}"
        if [ "${task}" = in-the-wild ]; then
            # Same job as pitch_range_dist so the shared per-leaf pitch caches never race.
            .venv/bin/python scripts/paired_itw.py --lang "${lang}" \
                --workers "${SLURM_CPUS_PER_TASK:-28}"
        fi
        ;;

    *)
        echo "unknown METRICS_KIND=${METRICS_KIND}" >&2
        exit 1
        ;;
esac
