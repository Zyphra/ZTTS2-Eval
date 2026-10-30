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
#   Z3_TASKS="zero_shot_mini zero_shot_voxblink_mini" sbatch --parsable --chdir="$(pwd -P)" \
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
#   Z3_REVERSE      1 = walk each GPU's leaves smallest-first. A second job submitted with
#                   the same tasks/modes and node count owns the same leaves, so a reversed
#                   one is a helper that meets the running job in the middle (default 0)
#
# Leaves (mode x task x lang) are dealt largest-first round-robin over every array task x
# GPU. Nothing is claimed: wavs are written atomically and existing ones skipped, so a
# requeue or resubmit resumes. One failed leaf does not stop the other GPUs; the job exits
# 1 iff any leaf is left in logs/slurm/ztts-zonos3-gen-<array job>_<task>-failed.txt.
set -uo pipefail

[ -f run_generate_zonos3.sh ] || { echo "[zonos3-gen] ERROR: run from the repo root (sbatch --chdir)" >&2; exit 1; }
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
n_nodes=${SLURM_ARRAY_TASK_COUNT:-1}
n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${n_gpu}" -ge 1 ] || n_gpu=1
tag=${array_job}_${node}

mkdir -p logs/slurm
srv_dir=$(pwd -P)/logs/slurm/servers/ztts-zonos3-gen-${tag}
mkdir -p "${srv_dir}"
leaves=logs/slurm/ztts-zonos3-gen-${tag}-leaves.txt
failed=logs/slurm/ztts-zonos3-gen-${tag}-failed.txt

# "<n_items> <mode> <task> <lang>", largest first; deterministic, so a requeue rebuilds
# the identical list.
for mode in ${Z3_MODES}; do
    for task in ${Z3_TASKS}; do
        for lang in $(python3 -m utils.benchmark langs "${task}"); do
            [ -f "data/${task}/${lang}/text" ] || continue
            echo "$(grep -c . "data/${task}/${lang}/text") ${mode} ${task} ${lang}"
        done
    done
done | LC_ALL=C sort -k1,1nr -k2 > "${leaves}"
rm -f "${failed}"
echo "[zonos3-gen] task ${node}/${n_nodes}: $(grep -c . "${leaves}") leaves, ${n_gpu} GPU(s), tag ${tag}"

# Lines owned by slot K of M (GPU s of array task `node` is slot node*n_gpu + s).
owned() { awk -v k="$1" -v m="$2" '(NR-1) % m == k' "${leaves}"; }
slots=$(( n_nodes * n_gpu ))

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export ZONOS3_GPU=$s
        export ZONOS3_PORT=$(( 20000 + (job % 1000) * 40 + s * 4 + 2 ))
        export ZONOS3_MAX_RUNNING=${Z3_MAX_RUNNING}
        export ZONOS3_LOG=${srv_dir}/gpu${s}.log
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nompss-${USER:-u}-${job}-${s}
        . scripts/serve_zonos3.sh
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
            bash run_generate_zonos3.sh --server "${ZONOS3_BASE_URL}" --speaker_mode "$1" \
                --task "$2" --test_set "$3" --out_root "results/${Z3_SYSTEM}_$1" \
                --workers "${Z3_MAX_RUNNING}" --limit "${Z3_LIMIT}" --laci "${Z3_LACI}" < /dev/null
        }

        if ! start_zonos3_server; then
            owned $(( node * n_gpu + s )) "${slots}" | while read -r _n mode task lang; do
                echo "${mode} ${task} ${lang}" >> "${failed}"
            done
            exit 1
        fi
        # Process substitution, not a pipe: a restart must update ZONOS3_PID in THIS
        # shell, the one the EXIT trap runs in.
        while read -r _n mode task lang; do
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
            [ "${rc}" -eq 0 ] || echo "${mode} ${task} ${lang}" >> "${failed}"
        done < <(owned $(( node * n_gpu + s )) "${slots}" | if [ "${Z3_REVERSE:-0}" = 1 ]; then tac; else cat; fi)
    ) > "logs/slurm/ztts-zonos3-gen-${tag}-g${s}.log" 2>&1 &
    pids="${pids} $!"
done
for p in ${pids}; do wait "${p}"; done

# Coverage of this array task's leaves.
low=0
for ((s = 0; s < n_gpu; s++)); do
    while read -r n mode task lang; do
        want=${n}
        [ "${Z3_LIMIT}" -gt 0 ] && [ "${Z3_LIMIT}" -lt "${n}" ] && want=${Z3_LIMIT}
        got=$(find "results/${Z3_SYSTEM}_${mode}/${task}/${lang}/wavs" -name '*.wav' 2>/dev/null | wc -l)
        echo "COVERAGE ${Z3_SYSTEM}_${mode} ${task} ${lang} ${got}/${want}"
        [ $(( got * 100 )) -ge $(( want * 98 )) ] || low=$(( low + 1 ))
    done < <(owned $(( node * n_gpu + s )) "${slots}")
done
echo "[zonos3-gen] ${low} leaf/leaves below 98% generation coverage"
echo "[zonos3-gen] wall ${SECONDS}s"

if [ -s "${failed}" ]; then
    echo "[zonos3-gen] $(grep -c . "${failed}") failed leaves listed in ${failed}"
    exit 1
fi
