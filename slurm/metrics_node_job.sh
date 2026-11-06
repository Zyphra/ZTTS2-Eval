#!/usr/bin/env bash
#SBATCH --job-name=ztts-metrics-node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --mem=1500G
#SBATCH --time=10:00:00
# Whole-node runner for submit_metrics.sh with METRICS_NODES > 0: every array task runs
# slurm/metrics_job.sh rows of the run's manifests locally instead of one sbatch array
# element per row. Jobs asking for < 5 GPUs are rerouted to the small dev partition at
# the site this was run on, where the per-row GPU arrays starve.
#
# GPU kinds (mpm, rhythm) run one row per GPU; CPU kinds (speaking, pitch_range) run
# METRICS_CPU_LANES (default 3) rows at a time. Rows are queued mpm, rhythm, then CPU
# kinds, each largest set first, and CLAIMED (atomic mkdir under <run dir>/claims/), so
# every array task works through the same list. A requeued task releases its previous
# incarnation's unfinished claims.
#
# Pipelining (optional): with METRICS_WAIT_JOBS="<id> ..." and METRICS_WAIT_SYSTEMS="a b",
# a row starts only once each listed system's leaf for its task/lang has a wav per text
# line, or once none of those jobs is pending or running any more.
#
# Environment (set by submit_metrics.sh): METRICS_REPO, METRICS_RUN_DIR, METRICS_NODE_KINDS,
# METRICS_SHARD_DIR (speaking). A failed row is listed in <run dir>/failed-node<N>.txt and
# makes this task exit 1.
set -uo pipefail

: "${METRICS_REPO:?METRICS_REPO is required}"
: "${METRICS_RUN_DIR:?METRICS_RUN_DIR is required}"
: "${METRICS_NODE_KINDS:?METRICS_NODE_KINDS is required}"
cd "${METRICS_REPO}" || exit 1
. slurm/lib_slurm.sh || exit 1

node=${SLURM_ARRAY_TASK_ID:-0}
n_gpu=$(slurm_gpu_count)
cpu_lanes=${METRICS_CPU_LANES:-3}
claims="${METRICS_RUN_DIR}/claims"
failed="${METRICS_RUN_DIR}/failed-node${node}.txt"
slurm_claims_init "${claims}" "${node}"
rm -f "${failed}"

# Queue lines "<kind> <row> <task> <lang>": one file per lane type, built identically by
# every task (row order is the manifest's; queue order is by kind, then set size).
queue() {
    local kind row task lang
    for kind in "$@"; do
        case " ${METRICS_NODE_KINDS} " in *" ${kind} "*) ;; *) continue ;; esac
        row=0
        while IFS=$'\t' read -r task lang; do
            echo "$(grep -c . "data/${task}/${lang}/text") ${kind} ${row} ${task} ${lang}"
            row=$(( row + 1 ))
        done < "${METRICS_RUN_DIR}/${kind}.tsv" | LC_ALL=C sort -k1,1nr | cut -d' ' -f2-
    done
}
gpu_q="${METRICS_RUN_DIR}/queue-gpu-node${node}.txt"
cpu_q="${METRICS_RUN_DIR}/queue-cpu-node${node}.txt"
queue mpm rhythm > "${gpu_q}"
queue pitch_range speaking > "${cpu_q}"

waiting() {  # 0 while any of METRICS_WAIT_JOBS is pending/running, or sacct is unreadable
    local ids out
    ids=$(printf '%s' "${METRICS_WAIT_JOBS:-}" | tr ' ' ',')
    [ -n "${ids}" ] || return 1
    out=$(sacct -n -X -j "${ids}" -o State 2>/dev/null) || return 0
    printf '%s\n' "${out}" | grep -qE "PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING|COMPLETING"
}
complete() {  # TASK LANG: every METRICS_WAIT_SYSTEMS leaf has a wav per text line
    local want got s
    want=$(grep -c . "data/$1/$2/text")
    for s in ${METRICS_WAIT_SYSTEMS:-}; do
        got=$(find "results/${s}/$1/$2/wavs" -name '*.wav' 2>/dev/null | wc -l)
        [ "${got}" -ge "${want}" ] || return 1
    done
}

run_row() {  # KIND ROW TASK LANG CLAIM_KEY GPU|''
    local kind=$1 row=$2 task=$3 lang=$4 key=$5 gpu=$6 cpus where=cpu
    case "${kind}" in rhythm) cpus=16 ;; *) cpus=28 ;; esac
    [ -n "${gpu}" ] && where="gpu ${gpu}"
    echo "=== [$(date +%T) node ${node} ${where}] ${kind} ${task}/${lang} ==="
    if CUDA_VISIBLE_DEVICES="${gpu}" METRICS_KIND="${kind}" \
            METRICS_MANIFEST="${METRICS_RUN_DIR}/${kind}.tsv" SLURM_ARRAY_TASK_ID="${row}" \
            SLURM_CPUS_PER_TASK="${cpus}" bash slurm/metrics_job.sh \
            > "${METRICS_RUN_DIR}/${kind}-${task}_${lang}.out" 2>&1; then
        slurm_claim_done "${claims}" "${key}"
    else
        echo "${kind} ${task} ${lang}" >> "${failed}"
        echo "[metrics-node] FAILED ${kind} ${task}/${lang} (${METRICS_RUN_DIR}/${kind}-${task}_${lang}.out)"
    fi
}

lane() {  # QUEUE GPU|''
    local q=$1 gpu=$2 kind row task lang key open picked gate
    while :; do
        open=0; picked=""
        # One sacct per scan; rows are gated only while the generation jobs still run.
        gate=0
        if [ -n "${METRICS_WAIT_SYSTEMS:-}" ] && waiting; then gate=1; fi
        while read -r kind row task lang; do
            key="${kind}_${row}"
            [ -e "${claims}/${key}" ] && continue
            open=1
            if [ "${gate}" = 1 ] && ! complete "${task}" "${lang}"; then continue; fi
            slurm_claim "${claims}" "${key}" "${node}" || continue
            picked="${kind} ${row} ${task} ${lang}"
            break
        done < "${q}"
        if [ -n "${picked}" ]; then
            # shellcheck disable=SC2086
            run_row ${picked} "${key}" "${gpu}"
        elif [ "${open}" -eq 1 ]; then
            sleep 30
        else
            return 0
        fi
    done
}

echo "[metrics-node] task ${node}: $(grep -c . "${gpu_q}") GPU rows on ${n_gpu} GPU(s)," \
     "$(grep -c . "${cpu_q}") CPU rows on ${cpu_lanes} lane(s); kinds: ${METRICS_NODE_KINDS}"
pids=""
if [ -s "${gpu_q}" ]; then
    for ((g = 0; g < n_gpu; g++)); do
        lane "${gpu_q}" "${g}" > "${METRICS_RUN_DIR}/lane-node${node}-gpu${g}.log" 2>&1 &
        pids="${pids} $!"
    done
fi
if [ -s "${cpu_q}" ]; then
    for ((l = 0; l < cpu_lanes; l++)); do
        lane "${cpu_q}" "" > "${METRICS_RUN_DIR}/lane-node${node}-cpu${l}.log" 2>&1 &
        pids="${pids} $!"
    done
fi
for p in ${pids}; do wait "${p}"; done
echo "[metrics-node] task ${node}: done in ${SECONDS}s"
if [ -s "${failed}" ]; then
    echo "[metrics-node] $(grep -c . "${failed}") failed rows listed in ${failed}"
    exit 1
fi
