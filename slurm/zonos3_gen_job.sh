#!/usr/bin/env bash
#SBATCH --job-name=ztts-zonos3-gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
# Generate ZONOS3 Mini for both conditioning modes (results/<Z3_SYSTEM>_<mode>/<task>/<lang>)
# on one 8-GPU node: one release minisgl server per GPU serving both modes and every
# language it is handed. Always submit as an ARRAY job (the %A_%a output name and the
# ${array_job}_${node} file tags assume one), from the repo root:
#
#   Z3_TASKS="clean_mini in-the-wild_mini" sbatch --parsable --chdir="$(pwd -P)" \
#       --export=ALL --array=0-0 slurm/zonos3_gen_job.sh
#
# Environment (--export=ALL):
#   ZONOS3_HOME     minisgl-zonos3 checkout                          (required)
#   ZONOS3_ENV      site env script, forwarded to serve_zonos3.sh    (optional)
#   Z3_TASKS        space-separated tasks                            (required)
#   Z3_MODES        speaker modes            (default "reference embedding")
#   Z3_LIMIT        only the first N utterances per leaf (default 0 = all)
#   Z3_MAX_RUNNING  server --max-running-requests AND client --workers (default 16)
#   Z3_SYSTEM       system-dir prefix: writes results/<Z3_SYSTEM>_<mode> (default zonos3-mini)
#   Z3_LACI         true = LACI decoding (laci=true); the server needs --laci-heads via
#                   ZONOS3_SERVE_ARGS                                  (default false)
#   Z3_REVERSE      1 = walk the leaf list smallest-first. A second job submitted with the
#                   same tasks/modes meets the running job in the middle (default 0)
#
# Pending leaves (mode x task x lang with a wanted text uid lacking a wav) are listed once
# per array job (first task to start, under a mkdir lock), largest first, and frozen. Each
# GPU takes the next leaf whose claim (atomic mkdir under
# logs/slurm/ztts-zonos3-gen-<array job>-claims/) it wins; a requeued task releases its
# previous incarnation's unfinished claims. Wavs are written atomically and existing ones
# skipped, so a requeue or resubmit resumes. One failed leaf does not stop the other GPUs;
# the job exits 1 iff a GPU's server never started or any leaf is left in
# logs/slurm/ztts-zonos3-gen-<array job>_<task>-failed.txt. Prints
# "COVERAGE <system> <task> <lang> <got>/<want>" per leaf this task claimed; coverage
# below 98% is reported, not failed.
set -uo pipefail

. slurm/lib_slurm.sh || { echo "[zonos3-gen] ERROR: run from the repo root (sbatch --chdir)" >&2; exit 1; }
[ -f scripts/gen/run_zonos3.sh ] || { echo "[zonos3-gen] ERROR: run from the repo root (sbatch --chdir)" >&2; exit 1; }
[ -d "${ZONOS3_HOME:-}" ] || { echo "[zonos3-gen] ERROR: ZONOS3_HOME is not set to a directory" >&2; exit 1; }
[ -n "${Z3_TASKS:-}" ] || { echo "[zonos3-gen] ERROR: Z3_TASKS is empty" >&2; exit 1; }
Z3_MODES=${Z3_MODES:-"reference embedding"}
Z3_LIMIT=${Z3_LIMIT:-0}
Z3_MAX_RUNNING=${Z3_MAX_RUNNING:-16}
Z3_SYSTEM=${Z3_SYSTEM:-zonos3-mini}
Z3_LACI=${Z3_LACI:-false}

job=${SLURM_JOB_ID:-$$}
array_job=${SLURM_ARRAY_JOB_ID:-$job}
node=${SLURM_ARRAY_TASK_ID:-0}
n_gpu=$(slurm_gpu_count)
tag=${array_job}_${node}

mkdir -p logs/slurm
leaves=logs/slurm/ztts-zonos3-gen-${array_job}-leaves.txt
claims=logs/slurm/ztts-zonos3-gen-${array_job}-claims
failed=logs/slurm/ztts-zonos3-gen-${tag}-failed.txt

leaf_text() { echo "data/$2/$3/text"; }                      # <mode> <task> <lang>
leaf_wavs() { echo "results/${Z3_SYSTEM}_$1/$2/$3/wavs"; }   # <mode> <task> <lang>
want_of() { slurm_want "$(leaf_text "$@")" "${Z3_LIMIT}"; }
missing_of() { slurm_missing "$(leaf_text "$@")" "$(leaf_wavs "$@")" "${Z3_LIMIT}"; }

build_leaves() {  # "<missing> <mode> <task> <lang>" per pending leaf, largest first
    local mode task lang m
    for mode in ${Z3_MODES}; do
        for task in ${Z3_TASKS}; do
            for lang in $(python3 -m utils.benchmark langs "${task}"); do
                [ -f "data/${task}/${lang}/text" ] || continue
                m=$(missing_of "${mode}" "${task}" "${lang}")
                [ "${m}" -gt 0 ] && echo "${m} ${mode} ${task} ${lang}"
            done
        done
    done | LC_ALL=C sort -k1,1nr -k2
}
slurm_list_once "${leaves}" build_leaves
slurm_claims_init "${claims}" "${node}"
rm -f "${failed}"
echo "[zonos3-gen] task ${node}: $(grep -c . "${leaves}") pending leaves (${leaves}), ${n_gpu} GPU(s), tag ${tag}"

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export ZONOS3_GPU=$s
        export ZONOS3_PORT=$(slurm_role_port "$(slurm_port_base "${s}")" tts)
        export ZONOS3_MAX_RUNNING=${Z3_MAX_RUNNING}
        export SERVE_LOG_DIR=$(slurm_server_log_dir "ztts-zonos3-gen-${tag}-g${s}")
        export CUDA_MPS_PIPE_DIRECTORY=$(slurm_mps_dir "${job}-${s}")
        . scripts/gen/serve_zonos3.sh
        trap stop_zonos3_server EXIT
        trap 'stop_zonos3_server; exit 143' INT TERM

        restarts=0
        # Healthy, else restart (at most 3 times per GPU).
        ensure_up() {
            zonos3_healthy && return 0
            [ "${restarts}" -lt 3 ] || return 1
            restarts=$(( restarts + 1 ))
            echo "[zonos3-gen] gpu ${s}: server unhealthy, restart ${restarts}/3"
            stop_zonos3_server
            start_zonos3_server
        }
        run_leaf() {
            bash scripts/gen/run_zonos3.sh --server "${ZONOS3_BASE_URL}" --speaker-mode "$1" \
                --task "$2" --test-set "$3" --out-root "results/${Z3_SYSTEM}_$1" \
                --workers "${Z3_MAX_RUNNING}" --limit "${Z3_LIMIT}" --laci "${Z3_LACI}" < /dev/null
        }

        if ! start_zonos3_server; then
            echo "[zonos3-gen] gpu ${s}: server failed to start; leaving the list to the other GPUs"
            exit 1
        fi
        # Process substitution, not a pipe: a restart must update ZONOS3_PID in THIS
        # shell, the one the EXIT trap runs in.
        while read -r _n mode task lang; do
            key="${mode}__${task}__${lang}"
            slurm_claim "${claims}" "${key}" "${node}" || continue
            echo "=== [$(date +%T) gpu ${s}] ${mode} ${task} ${lang} ==="
            rc=1
            if ensure_up; then
                run_leaf "${mode}" "${task}" "${lang}"; rc=$?
                if ! { [ "${rc}" -eq 0 ] && zonos3_healthy; }; then
                    # Rerun once after a restart; it only fills the missing wavs.
                    rc=1
                    if ! zonos3_healthy && ensure_up; then
                        run_leaf "${mode}" "${task}" "${lang}"; rc=$?
                        zonos3_healthy || rc=1
                    fi
                fi
            fi
            if [ "${rc}" -eq 0 ]; then
                slurm_claim_done "${claims}" "${key}"
            else
                echo "${mode} ${task} ${lang}" >> "${failed}"
            fi
        done < <(if [ "${Z3_REVERSE:-0}" = 1 ]; then tac "${leaves}"; else cat "${leaves}"; fi)
    ) > "logs/slurm/ztts-zonos3-gen-${tag}-g${s}.log" 2>&1 &
    pids="${pids} $!"
done
server_fails=0
for p in ${pids}; do wait "${p}" || server_fails=$(( server_fails + 1 )); done

# Coverage of the leaves this array task claimed (uid-exact).
low=0
while read -r _n mode task lang; do
    [ "$(cat "${claims}/${mode}__${task}__${lang}/owner" 2>/dev/null)" = "${node}" ] || continue
    want=$(want_of "${mode}" "${task}" "${lang}")
    got=$(( want - $(missing_of "${mode}" "${task}" "${lang}") ))
    echo "COVERAGE ${Z3_SYSTEM}_${mode} ${task} ${lang} ${got}/${want}"
    [ $(( got * 100 )) -ge $(( want * 98 )) ] || low=$(( low + 1 ))
done < "${leaves}"
echo "[zonos3-gen] ${low} leaf/leaves below 98% generation coverage"
echo "[zonos3-gen] wall ${SECONDS}s"

if [ "${server_fails}" -gt 0 ]; then
    echo "[zonos3-gen] ${server_fails} GPU(s) ended non-zero (server never started)"
fi
if [ -s "${failed}" ]; then
    echo "[zonos3-gen] $(grep -c . "${failed}") failed leaves listed in ${failed}"
    exit 1
fi
[ "${server_fails}" -eq 0 ] || exit 1
