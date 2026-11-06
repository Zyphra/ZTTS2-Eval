#!/usr/bin/env bash
# Submit retained metrics for random 50/50 splits of every ground-truth language.
# Optional environment: METRICS_PARTITION / METRICS_ACCOUNT / METRICS_QOS,
# METRICS_KINDS (default "speaking_gt mpm_gt"), METRICS_DEPENDENCY (sbatch --dependency
# for every array), METRICS_FINALIZE (1 default; 0 skips the finalizer). The last stdout
# line is METRICS_JOBS=<id>[:<id>...] (arrays, plus the finalizer's when submitted).
set -euo pipefail

if [ ! -f slurm/ground_truth_metrics_job.sh ]; then
    echo "run from the repository root" >&2
    exit 1
fi

# SLURM site: unset = the cluster's defaults.
partition="${METRICS_PARTITION:-}"
account="${METRICS_ACCOUNT:-}"
qos="${METRICS_QOS:-}"
site_args=()
if [ -n "${partition}" ]; then site_args+=(--partition="${partition}"); fi
if [ -n "${account}" ]; then site_args+=(--account="${account}"); fi
if [ -n "${qos}" ]; then site_args+=(--qos="${qos}"); fi
kinds="${METRICS_KINDS:-speaking_gt mpm_gt}"
dependency="${METRICS_DEPENDENCY:-}"
finalize="${METRICS_FINALIZE:-1}"
speaking_throttle="${GT_SPEAKING_THROTTLE:-4}"
mpm_throttle="${GT_MPM_THROTTLE:-8}"
mpm_time="${GT_MPM_TIME:-10:00:00}"
run_id="$(date +%Y%m%d-%H%M%S)"
run_dir="$(pwd -P)/logs/slurm/ground-truth-metrics-${run_id}"
shard_dir="${run_dir}/speaking_shards"
mkdir -p "${run_dir}" "${shard_dir}"

speaking_manifest="${run_dir}/speaking_gt.tsv"
mpm_manifest="${run_dir}/mpm_gt.tsv"
: > "${speaking_manifest}"
: > "${mpm_manifest}"

for task in clean in-the-wild; do
    for lang_dir in data/"${task}"/*; do
        [ -d "${lang_dir}/waveform" ] || continue
        lang="${lang_dir##*/}"
        printf '%s\t%s\n' "${task}" "${lang}" >> "${speaking_manifest}"
        # FLEURS-R only, matching the systems' own gate in submit_metrics.sh: the GT
        # row has to be scored on the same item set as the systems it is the floor for.
        # See docs/mpm_itw_limits.md for why VoxBlink2 is excluded.
        if [ "${task}" = clean ]; then
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
    if [ -n "${dependency}" ]; then
        cmd+=(--dependency="${dependency}")
    fi
    if [ "${gpus}" -gt 0 ]; then
        cmd+=(--gpus="${gpus}")
    fi
    cmd+=(slurm/ground_truth_metrics_job.sh)
    job_id=$("${cmd[@]}")
    printf '%s' "${job_id}"
}

jobs=()
speaking_job=""; mpm_job=""
for kind in ${kinds}; do
    case "${kind}" in
        speaking_gt) speaking_job=$(submit_array speaking_gt "${speaking_manifest}" "${speaking_throttle}" 28 32G 10:00:00 0)
                     jobs+=("${speaking_job}") ;;
        mpm_gt)      mpm_job=$(submit_array mpm_gt "${mpm_manifest}" "${mpm_throttle}" 24 48G "${mpm_time}" 1)
                     jobs+=("${mpm_job}") ;;
        *) echo "unknown METRICS_KINDS entry '${kind}'" >&2; exit 1 ;;
    esac
done
if [ "${#jobs[@]}" -eq 0 ]; then
    echo "METRICS_KINDS selects nothing" >&2
    exit 1
fi

finalize_job=""
if [ "${finalize}" != "0" ]; then
    dep_ids=$(IFS=:; printf '%s' "${jobs[*]}")
    export METRICS_KIND=finalize_gt
    finalize_job=$(sbatch --parsable \
        --job-name=ztts-gt-metrics-finalize \
        ${site_args[@]+"${site_args[@]}"} \
        --dependency="afterok:${dep_ids}" \
        --cpus-per-task=4 --mem=16G --time=01:00:00 \
        --chdir="${METRICS_REPO}" \
        --output="${run_dir}/finalize-%j.out" \
        --export=ALL \
        slurm/ground_truth_metrics_finalize.sh)
    jobs+=("${finalize_job}")
fi

cat > "${run_dir}/jobs.env" <<EOF
METRICS_RUN_DIR=${run_dir}
GT_SPLIT_SEED=${GT_SPLIT_SEED}
SPEAKING_GT_JOB=${speaking_job}
MPM_GT_JOB=${mpm_job}
FINALIZE_JOB=${finalize_job}
EOF

printf 'run directory: %s\n' "${run_dir}"
printf 'watch: squeue -j %s\n' "$(IFS=,; printf '%s' "${jobs[*]}")"
printf 'METRICS_JOBS=%s\n' "$(IFS=:; printf '%s' "${jobs[*]}")"
