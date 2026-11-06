#!/usr/bin/env bash
# Submit the retained distributional metrics for every populated system leaf:
#   speaking     char/s distribution (both sets)
#   mpm          MPM span (Clean set ONLY, see below)
#   rhythm       mHuBERT rhythm distribution (both sets, GPU)
#   pitch_range  register-normalized pitch range (both sets, CPU); on the ITW set the
#                same job also runs scripts/prosody/paired_itw.py so the shared pitch caches
#                never race
# Walltimes stay <= 10 h: single-node jobs with < 5 GPUs are capped at that on the dev
# partition at the site this was run on (job_submit policy), and 10 h also fits every
# array element measured so far (MPM ~1-5 h). Packs efficiently: CPU arrays share the node while MPM/rhythm use one GPU each. A
# dependent finalizer merges and publishes all outputs.
#
# Environment (all optional; unset = today's standalone behaviour):
#   METRICS_PARTITION / METRICS_ACCOUNT / METRICS_QOS   sbatch --partition/--account/--qos
#   METRICS_KINDS       arrays to submit (default "speaking mpm rhythm pitch_range")
#   METRICS_DEPENDENCY  sbatch --dependency string applied to every array
#   METRICS_FINALIZE    1 (default) submits the finalizer; 0 skips it
#   METRICS_NODES       N > 0: instead of one array per kind, ONE array of N whole-node tasks
#                       (slurm/metrics_node_job.sh) runs every selected kind's rows, GPU rows
#                       one per GPU; also takes METRICS_CPU_LANES, METRICS_WAIT_JOBS and
#                       METRICS_WAIT_SYSTEMS (see that script). Default 0 = per-kind arrays
#   METRICS_RESERVATION sbatch --reservation
#   METRICS_EXCLUDE     sbatch --exclude node list for the GPU/CPU arrays (e.g. a node whose
#                       GPUs fail CUDA init)
#   METRICS_LANGS       only these languages (space-separated), in every task; empty = all
# The last stdout line is always METRICS_JOBS=<id>[:<id>...]: the array job ids, plus the
# finalizer's when one is submitted.
set -euo pipefail

if [ ! -f slurm/metrics_job.sh ]; then
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
# The 0-GPU finalizer gets no --reservation: job_submit reroutes jobs with < 5 GPUs to dev,
# where reserved nodes are unavailable and sbatch rejects the job.
finalize_site_args=(${site_args[@]+"${site_args[@]}"})
if [ -n "${METRICS_RESERVATION:-}" ]; then site_args+=(--reservation="${METRICS_RESERVATION}"); fi
if [ -n "${METRICS_EXCLUDE:-}" ]; then site_args+=(--exclude="${METRICS_EXCLUDE}"); fi
nodes="${METRICS_NODES:-0}"
kinds="${METRICS_KINDS:-speaking mpm rhythm pitch_range}"
dependency="${METRICS_DEPENDENCY:-}"
finalize="${METRICS_FINALIZE:-1}"
run_id="$(date +%Y%m%d-%H%M%S)"
run_dir="$(pwd -P)/logs/slurm/metrics-${run_id}"
shard_dir="${run_dir}/speaking_shards"
mkdir -p "${run_dir}" "${shard_dir}"

speaking_manifest="${run_dir}/speaking.tsv"
mpm_manifest="${run_dir}/mpm.tsv"
rhythm_manifest="${run_dir}/rhythm.tsv"
pitch_range_manifest="${run_dir}/pitch_range.tsv"
: > "${speaking_manifest}"
: > "${mpm_manifest}"
: > "${rhythm_manifest}"
: > "${pitch_range_manifest}"

shopt -s nullglob
for task in clean in-the-wild; do
    for lang_dir in data/"${task}"/*; do
        [ -d "${lang_dir}/waveform" ] || continue
        lang="${lang_dir##*/}"
        if [ -n "${METRICS_LANGS:-}" ]; then
            case " ${METRICS_LANGS} " in *" ${lang} "*) ;; *) continue ;; esac
        fi
        wav_dirs=(results/*/"${task}"/"${lang}"/wavs)
        [ "${#wav_dirs[@]}" -gt 0 ] || continue
        printf '%s\t%s\n' "${task}" "${lang}" >> "${speaking_manifest}"
        printf '%s\t%s\n' "${task}" "${lang}" >> "${rhythm_manifest}"
        printf '%s\t%s\n' "${task}" "${lang}" >> "${pitch_range_manifest}"
        # MPM span is FLEURS-R only, deliberately. Its two anchors (real split-half
        # floor, contour-scrambled ceiling) collapse on in-the-wild audio: the usable
        # range is 18.68 on FLEURS-R en vs 7.63 on VoxBlink2 en, so identical prosody
        # error reads ~2.5x larger there and the cross-language shared span axis stops
        # being valid. A better F0 backend, per-utterance energy normalisation and
        # per-condition anchors were all measured and none fixes it (the last makes it
        # worse). See docs/mpm_itw_limits.md before removing this gate.
        if [ "${task}" = clean ]; then
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
    if [ -n "${dependency}" ]; then
        cmd+=(--dependency="${dependency}")
    fi
    if [ "${gpus}" -gt 0 ]; then
        cmd+=(--gpus="${gpus}")
    fi
    cmd+=(slurm/metrics_job.sh)
    job_id=$("${cmd[@]}")
    printf '%s' "${job_id}"
}

jobs=()
speaking_job=""; mpm_job=""; rhythm_job=""; pitch_range_job=""; node_job=""
for kind in ${kinds}; do
    case "${kind}" in
        speaking|mpm|rhythm|pitch_range) ;;
        *) echo "unknown METRICS_KINDS entry '${kind}'" >&2; exit 1 ;;
    esac
done
if [ "${nodes}" -gt 0 ]; then
    export METRICS_RUN_DIR="${run_dir}" METRICS_NODE_KINDS="${kinds}"
    cmd=(sbatch --parsable ${site_args[@]+"${site_args[@]}"}
        --array="0-$((nodes - 1))"
        --chdir="${METRICS_REPO}"
        --output="${run_dir}/node-%A_%a.out"
        --export=ALL)
    if [ -n "${dependency}" ]; then
        cmd+=(--dependency="${dependency}")
    fi
    node_job=$("${cmd[@]}" slurm/metrics_node_job.sh)
    jobs+=("${node_job}")
else
    for kind in ${kinds}; do
        case "${kind}" in
            speaking)    speaking_job=$(submit_array speaking "${speaking_manifest}" 4 28 32G 10:00:00 0)
                         jobs+=("${speaking_job}") ;;
            mpm)         mpm_job=$(submit_array mpm "${mpm_manifest}" 6 28 64G 10:00:00 1)
                         jobs+=("${mpm_job}") ;;
            rhythm)      rhythm_job=$(submit_array rhythm "${rhythm_manifest}" 8 16 64G 10:00:00 1)
                         jobs+=("${rhythm_job}") ;;
            pitch_range) pitch_range_job=$(submit_array pitch_range "${pitch_range_manifest}" 8 28 32G 10:00:00 0)
                         jobs+=("${pitch_range_job}") ;;
        esac
    done
fi
if [ "${#jobs[@]}" -eq 0 ]; then
    echo "METRICS_KINDS selects nothing" >&2
    exit 1
fi

finalize_job=""
if [ "${finalize}" != "0" ]; then
    dep_ids=$(IFS=:; printf '%s' "${jobs[*]}")
    export METRICS_KIND=finalize
    finalize_job=$(sbatch --parsable \
        --job-name=ztts-metrics-finalize \
        ${finalize_site_args[@]+"${finalize_site_args[@]}"} \
        --dependency="afterok:${dep_ids}" \
        --cpus-per-task=4 --mem=16G --time=01:00:00 \
        --chdir="${METRICS_REPO}" \
        --output="${run_dir}/finalize-%j.out" \
        --export=ALL \
        slurm/metrics_finalize.sh)
    jobs+=("${finalize_job}")
fi

cat > "${run_dir}/jobs.env" <<EOF
METRICS_RUN_DIR=${run_dir}
SPEAKING_JOB=${speaking_job}
MPM_JOB=${mpm_job}
RHYTHM_JOB=${rhythm_job}
PITCH_RANGE_JOB=${pitch_range_job}
NODE_JOB=${node_job}
FINALIZE_JOB=${finalize_job}
EOF

all_ids=$(IFS=,; printf '%s' "${jobs[*]}")
printf 'run directory: %s\n' "${run_dir}"
printf 'watch: squeue -j %s\n' "${all_ids}"
printf 'METRICS_JOBS=%s\n' "$(IFS=:; printf '%s' "${jobs[*]}")"
