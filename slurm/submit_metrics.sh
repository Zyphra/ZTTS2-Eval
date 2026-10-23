#!/usr/bin/env bash
# Submit all retained distributional metrics for every populated system leaf.
# Packs efficiently: CPU arrays share the node while MPM uses
# up to six H100s. A dependent finalizer merges and publishes all outputs.
set -euo pipefail

if [ ! -f slurm/metrics_job.sh ]; then
    echo "run from the repository root" >&2
    exit 1
fi

# SLURM partition/account: unset = the cluster's defaults (METRICS_PARTITION / METRICS_ACCOUNT override).
partition="${METRICS_PARTITION:-}"
account="${METRICS_ACCOUNT:-}"
site_args=()
if [ -n "${partition}" ]; then site_args+=(--partition="${partition}"); fi
if [ -n "${account}" ]; then site_args+=(--account="${account}"); fi
run_id="$(date +%Y%m%d-%H%M%S)"
run_dir="$(pwd -P)/logs/slurm/metrics-${run_id}"
shard_dir="${run_dir}/speaking_shards"
mkdir -p "${run_dir}" "${shard_dir}"

speaking_manifest="${run_dir}/speaking.tsv"
pitch_manifest="${run_dir}/pitch.tsv"
mpm_manifest="${run_dir}/mpm.tsv"
: > "${speaking_manifest}"
: > "${pitch_manifest}"
: > "${mpm_manifest}"

shopt -s nullglob
for task in zero_shot zero_shot_voxblink; do
    for lang_dir in data/"${task}"/*; do
        [ -d "${lang_dir}/waveform" ] || continue
        lang="${lang_dir##*/}"
        wav_dirs=(results/*/"${task}"/"${lang}"/wavs)
        [ "${#wav_dirs[@]}" -gt 0 ] || continue
        printf '%s\t%s\n' "${task}" "${lang}" >> "${speaking_manifest}"
        printf '%s\t%s\n' "${task}" "${lang}" >> "${pitch_manifest}"
        # MPM span is FLEURS-R only, deliberately. Its two anchors (real split-half
        # floor, contour-scrambled ceiling) collapse on in-the-wild audio: the usable
        # range is 18.68 on FLEURS-R en vs 7.63 on VoxBlink2 en, so identical prosody
        # error reads ~2.5x larger there and the cross-language shared span axis stops
        # being valid. A better F0 backend, per-utterance energy normalisation and
        # per-condition anchors were all measured and none fixes it (the last makes it
        # worse). See docs/mpm_itw_limits.md before removing this gate.
        if [ "${task}" = zero_shot ]; then
            printf '%s\t%s\n' "${task}" "${lang}" >> "${mpm_manifest}"
        fi
    done
done

export METRICS_REPO="$(pwd -P)"
export METRICS_SHARD_DIR="${shard_dir}"

submit_array() {
    local kind="$1" manifest="$2" throttle="$3" cpus="$4" mem="$5" time="$6" gpus="$7"
    local rows last job_id
    mapfile -t rows < "${manifest}"
    if [ "${#rows[@]}" -eq 0 ]; then
        echo "empty ${kind} manifest" >&2
        exit 1
    fi
    last=$((${#rows[@]} - 1))
    export METRICS_KIND="${kind}"
    export METRICS_MANIFEST="${manifest}"
    cmd=(sbatch --parsable
        --job-name="ztts-${kind}"
        ${site_args[@]+"${site_args[@]}"}
        --array="0-${last}%${throttle}"
        --cpus-per-task="${cpus}"
        --mem="${mem}"
        --time="${time}"
        --chdir="${METRICS_REPO}"
        --output="${run_dir}/${kind}-%A_%a.out"
        --export=ALL)
    if [ "${gpus}" -gt 0 ]; then
        cmd+=(--gpus="${gpus}")
    fi
    cmd+=(slurm/metrics_job.sh)
    job_id=$("${cmd[@]}")
    printf '%s' "${job_id}"
}

speaking_job=$(submit_array speaking "${speaking_manifest}" 4 28 32G 10:00:00 0)
pitch_job=$(submit_array pitch "${pitch_manifest}" 8 4 8G 10:00:00 0)
mpm_job=$(submit_array mpm "${mpm_manifest}" 6 28 64G 24:00:00 1)

export METRICS_KIND=finalize
finalize_job=$(sbatch --parsable \
    --job-name=ztts-metrics-finalize \
    ${site_args[@]+"${site_args[@]}"} \
    --dependency="afterok:${speaking_job}:${pitch_job}:${mpm_job}" \
    --cpus-per-task=4 --mem=16G --time=01:00:00 \
    --chdir="${METRICS_REPO}" \
    --output="${run_dir}/finalize-%j.out" \
    --export=ALL \
    slurm/metrics_finalize.sh)

cat > "${run_dir}/jobs.env" <<EOF
METRICS_RUN_DIR=${run_dir}
SPEAKING_JOB=${speaking_job}
PITCH_JOB=${pitch_job}
MPM_JOB=${mpm_job}
FINALIZE_JOB=${finalize_job}
EOF

printf 'run directory: %s\n' "${run_dir}"
printf 'jobs: speaking=%s pitch=%s mpm=%s finalize=%s\n' \
    "${speaking_job}" "${pitch_job}" "${mpm_job}" "${finalize_job}"
printf 'watch: squeue -j %s,%s,%s,%s\n' \
    "${speaking_job}" "${pitch_job}" "${mpm_job}" "${finalize_job}"
