#!/usr/bin/env bash
# Submit retained metrics for random 50/50 splits of every ground-truth language.
set -euo pipefail

if [ ! -f slurm/ground_truth_metrics_job.sh ]; then
    echo "run from the repository root" >&2
    exit 1
fi

# SLURM partition/account: unset = the cluster's defaults (METRICS_PARTITION / METRICS_ACCOUNT override).
partition="${METRICS_PARTITION:-}"
account="${METRICS_ACCOUNT:-}"
site_args=()
if [ -n "${partition}" ]; then site_args+=(--partition="${partition}"); fi
if [ -n "${account}" ]; then site_args+=(--account="${account}"); fi
speaking_throttle="${GT_SPEAKING_THROTTLE:-4}"
pitch_throttle="${GT_PITCH_THROTTLE:-8}"
mpm_throttle="${GT_MPM_THROTTLE:-8}"
mpm_time="${GT_MPM_TIME:-10:00:00}"
run_id="$(date +%Y%m%d-%H%M%S)"
run_dir="$(pwd -P)/logs/slurm/ground-truth-metrics-${run_id}"
shard_dir="${run_dir}/speaking_shards"
mkdir -p "${run_dir}" "${shard_dir}"

speaking_manifest="${run_dir}/speaking_gt.tsv"
pitch_manifest="${run_dir}/pitch_gt.tsv"
mpm_manifest="${run_dir}/mpm_gt.tsv"
: > "${speaking_manifest}"
: > "${pitch_manifest}"
: > "${mpm_manifest}"

for task in zero_shot zero_shot_voxblink; do
    for lang_dir in data/"${task}"/*; do
        [ -d "${lang_dir}/waveform" ] || continue
        lang="${lang_dir##*/}"
        printf '%s\t%s\n' "${task}" "${lang}" >> "${speaking_manifest}"
        printf '%s\t%s\n' "${task}" "${lang}" >> "${pitch_manifest}"
        # FLEURS-R only, matching the systems' own gate in submit_metrics.sh: the GT
        # row has to be scored on the same item set as the systems it is the floor for.
        # See docs/mpm_itw_limits.md for why VoxBlink2 is excluded.
        if [ "${task}" = zero_shot ]; then
            printf '%s\t%s\n' "${task}" "${lang}" >> "${mpm_manifest}"
        fi
    done
done

export METRICS_REPO="$(pwd -P)"
export METRICS_SHARD_DIR="${shard_dir}"
export GT_SPLIT_SEED="${GT_SPLIT_SEED:-0}"

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
    cmd+=(slurm/ground_truth_metrics_job.sh)
    job_id=$("${cmd[@]}")
    printf '%s' "${job_id}"
}

speaking_job=$(submit_array speaking_gt "${speaking_manifest}" "${speaking_throttle}" 28 32G 10:00:00 0)
pitch_job=$(submit_array pitch_gt "${pitch_manifest}" "${pitch_throttle}" 4 8G 10:00:00 0)
mpm_job=$(submit_array mpm_gt "${mpm_manifest}" "${mpm_throttle}" 24 48G "${mpm_time}" 1)

export METRICS_KIND=finalize_gt
finalize_job=$(sbatch --parsable \
    --job-name=ztts-gt-metrics-finalize \
    ${site_args[@]+"${site_args[@]}"} \
    --dependency="afterok:${speaking_job}:${pitch_job}:${mpm_job}" \
    --cpus-per-task=4 --mem=16G --time=01:00:00 \
    --chdir="${METRICS_REPO}" \
    --output="${run_dir}/finalize-%j.out" \
    --export=ALL \
    slurm/ground_truth_metrics_finalize.sh)

cat > "${run_dir}/jobs.env" <<EOF
METRICS_RUN_DIR=${run_dir}
GT_SPLIT_SEED=${GT_SPLIT_SEED}
SPEAKING_GT_JOB=${speaking_job}
PITCH_GT_JOB=${pitch_job}
MPM_GT_JOB=${mpm_job}
FINALIZE_JOB=${finalize_job}
EOF

printf 'run directory: %s\n' "${run_dir}"
printf 'jobs: speaking_gt=%s pitch_gt=%s mpm_gt=%s finalize=%s\n' \
    "${speaking_job}" "${pitch_job}" "${mpm_job}" "${finalize_job}"
printf 'watch: squeue -j %s,%s,%s,%s\n' \
    "${speaking_job}" "${pitch_job}" "${mpm_job}" "${finalize_job}"
