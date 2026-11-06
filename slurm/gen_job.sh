#!/usr/bin/env bash
#SBATCH --job-name=ztts-gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
# Generate one system on 8-GPU nodes, one generator per GPU, one leaf (task x lang) per
# driver call. The driver must take `--model M --task T --test-set LANG --out-root R
# --limit N --list-langs true`:
#   GEN_SERVER=0  in-process drivers (no server): scripts/gen/run_moss_tts.sh (MOSS-TTS
#                 family), scripts/gen/run_ming3.sh (Ming-3.0-tts), scripts/gen/run_pocket_tts.sh,
#                 scripts/gen/run_zonos3_reftext.sh, ... Each GPU's process sees only its GPU.
#   GEN_SERVER=1  drivers that start and stop their own server per call and also take
#                 `--gpu N --port P --workers W`: scripts/gen/run_omni.sh (vLLM-Omni). Every
#                 leaf starts its own server on its GPU (tts port of slurm/lib_slurm.sh's
#                 per-GPU window, server logs under logs/slurm/servers/<job>-g<gpu>/). The
#                 servers resolve the model from the local HF cache only (HF_HUB_OFFLINE=1):
#                 pre-download it on a networked host first.
# Always submit as an ARRAY job (the %A_%a output name and the <array job>_<task> file
# tags assume one), from the repo root:
#
#   GEN_DRIVER=scripts/gen/run_moss_tts.sh GEN_MODEL=moss-tts-v1.5 sbatch --parsable \
#       --chdir="$(pwd -P)" --export=ALL --array=0-3 slurm/gen_job.sh
#   GEN_DRIVER=scripts/gen/run_omni.sh GEN_MODEL=omnivoice GEN_SERVER=1 sbatch --parsable \
#       --chdir="$(pwd -P)" --export=ALL --array=0-1 slurm/gen_job.sh
#
# Environment (--export=ALL):
#   GEN_DRIVER     driver script, scripts/gen/run_<name>.sh                   (required)
#   GEN_MODEL      model key of the driver                                    (required)
#   GEN_TASKS      space-separated tasks                 (default "clean in-the-wild")
#   GEN_LANGS      only these langs; empty = every language the driver lists for the task
#                  (`<driver> --model M --task T --list-langs true`)           (default '')
#   GEN_LIMIT      only the first N utterances per leaf                (default 0 = all)
#   GEN_SERVER     0 = in-process driver, 1 = server-launching driver        (default 0)
#   GEN_WORKERS    concurrent requests per server (GEN_SERVER=1 only)        (default 16)
#   GEN_OUT_ROOT   leaves land in <root>/<task>/<lang>
#                  (default: gen_default_out_root <driver> GEN_MODEL, scripts/gen/lib_gen.sh)
#   OMNI_VENV      venv with vllm-omni installed (read by scripts/gen/run_omni.sh)
#
# Pending leaves (task x lang with a text uid lacking a wav) are listed once per array job
# (first task to start, under a mkdir lock), largest first, and frozen. Each GPU takes the
# next leaf whose claim (atomic mkdir under logs/slurm/ztts-gen-<array job>-claims/) it
# wins, so a task that starts late never leaves a fixed share idle. A leaf still missing
# wavs after a clean run is rerun once (only the missing utterances are retried). Wavs are
# written atomically and existing ones skipped, so a requeue or resubmit resumes; a
# requeued task releases its previous incarnation's unfinished claims. A leaf is FAILED
# when its driver exits non-zero or it ends below 98% coverage (a server driver exits 0
# when its server fails to start or an utterance fails); its claim stays unfinished so a
# requeue retries it. Prints "COVERAGE <system> <task> <lang> <got>/<want>" per leaf this
# task ran (<system> = basename of GEN_OUT_ROOT); exits 1 iff a leaf is listed in
# logs/slurm/ztts-gen-<array job>_<task>-failed.txt.
set -uo pipefail

. slurm/lib_slurm.sh || { echo "[gen] ERROR: run from the repo root (sbatch --chdir)" >&2; exit 1; }

[ -n "${GEN_DRIVER:-}" ] || { echo "[gen] ERROR: GEN_DRIVER is empty" >&2; exit 1; }
[ -f "${GEN_DRIVER}" ] || { echo "[gen] ERROR: no ${GEN_DRIVER}: run from the repo root (sbatch --chdir)" >&2; exit 1; }
[ -n "${GEN_MODEL:-}" ] || { echo "[gen] ERROR: GEN_MODEL is empty" >&2; exit 1; }
GEN_TASKS=${GEN_TASKS:-"clean in-the-wild"}
GEN_LANGS=${GEN_LANGS:-}
GEN_LIMIT=${GEN_LIMIT:-0}
GEN_SERVER=${GEN_SERVER:-0}
GEN_WORKERS=${GEN_WORKERS:-16}
case "${GEN_SERVER}" in
    0|1) ;;
    *) echo "[gen] ERROR: GEN_SERVER must be 0 or 1 (got '${GEN_SERVER}')" >&2; exit 1 ;;
esac
driver_name=$(basename "${GEN_DRIVER}" .sh | sed 's/^run_//')
if [ -z "${GEN_OUT_ROOT:-}" ]; then
    GEN_OUT_ROOT=$(. scripts/gen/lib_gen.sh; gen_default_out_root "${driver_name}" "${GEN_MODEL}") || exit 1
fi
system=$(basename "${GEN_OUT_ROOT}")

job=${SLURM_JOB_ID:-$$}
array_job=${SLURM_ARRAY_JOB_ID:-$job}
node=${SLURM_ARRAY_TASK_ID:-0}
n_gpu=$(slurm_gpu_count)
if [ "${GEN_SERVER}" = 1 ]; then
    export HF_HUB_OFFLINE=1
    per_gpu="${GEN_WORKERS} workers each, venv ${OMNI_VENV:-.venv}"
else
    cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
    threads=$(( cpus / n_gpu )); [ "${threads}" -ge 1 ] || threads=1
    export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads}
    per_gpu="${threads} threads each"
fi

mkdir -p logs/slurm
base=logs/slurm/ztts-gen-${array_job}
list_file=${base}-leaves.txt
claims=${base}-claims
failed=${base}_${node}-failed.txt

leaf_text() { echo "data/$1/$2/text"; }
leaf_wavs() { echo "${GEN_OUT_ROOT}/$1/$2/wavs"; }
want_of() { slurm_want "$(leaf_text "$1" "$2")" "${GEN_LIMIT}"; }
missing_of() { slurm_missing "$(leaf_text "$1" "$2")" "$(leaf_wavs "$1" "$2")" "${GEN_LIMIT}"; }

build_leaves() {  # "<missing> <task> <lang>" per pending leaf, largest first
    local task lang m langs
    for task in ${GEN_TASKS}; do
        # The driver's own language policy (basket ∩ what the model supports).
        langs=$(bash "${GEN_DRIVER}" --model "${GEN_MODEL}" --task "${task}" --list-langs true 2>/dev/null) || {
            echo "[gen] ERROR: ${GEN_DRIVER} --list-langs failed for task ${task}" >&2; continue; }
        for lang in ${langs}; do
            [ -f "data/${task}/${lang}/text" ] || continue
            if [ -n "${GEN_LANGS}" ]; then
                case " ${GEN_LANGS} " in *" ${lang} "*) ;; *) continue ;; esac
            fi
            m=$(missing_of "${task}" "${lang}")
            [ "${m}" -gt 0 ] && echo "${m} ${task} ${lang}"
        done
    done | LC_ALL=C sort -k1,1nr -k2
}
slurm_list_once "${list_file}" build_leaves
slurm_claims_init "${claims}" "${node}"
rm -f "${failed}"
echo "[gen] task ${node}: ${system} via ${GEN_DRIVER}, $(grep -c . "${list_file}") pending leaves" \
     "(${list_file}), ${n_gpu} GPU(s), ${per_gpu}"

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export CUDA_MPS_PIPE_DIRECTORY=$(slurm_mps_dir "${job}-${s}")
        if [ "${GEN_SERVER}" = 1 ]; then
            # No CUDA_VISIBLE_DEVICES here: the driver's server script sets it from --gpu
            # (scripts/gen/serve_omni.sh), so exporting a device id as well would remap --gpu.
            export SERVE_LOG_DIR=$(slurm_server_log_dir "${job}-g${s}")
            port=$(slurm_role_port "$(slurm_port_base "${s}")" tts)
            srv_args=(--gpu "${s}" --port "${port}" --workers "${GEN_WORKERS}")
        else
            export CUDA_VISIBLE_DEVICES=${s}
            srv_args=()
        fi
        run_leaf() {
            bash "${GEN_DRIVER}" --model "${GEN_MODEL}" --task "$1" --test-set "$2" \
                --out-root "${GEN_OUT_ROOT}" --limit "${GEN_LIMIT}" "${srv_args[@]}" < /dev/null
        }
        while read -r _n task lang <&3; do
            key="${task}__${lang}"
            slurm_claim "${claims}" "${key}" "${node}" || continue
            echo "=== [$(date +%T) task ${node} gpu ${s}] ${system} ${task} ${lang} ==="
            run_leaf "${task}" "${lang}"; rc=$?
            if [ "${rc}" -eq 0 ] && [ "$(missing_of "${task}" "${lang}")" -gt 0 ]; then
                echo "[gen] ${task}/${lang}: wavs still missing, rerunning once"
                run_leaf "${task}" "${lang}"; rc=$?
            fi
            want=$(want_of "${task}" "${lang}")
            got=$(( want - $(missing_of "${task}" "${lang}") ))
            if [ "${rc}" -eq 0 ] && [ $(( got * 100 )) -ge $(( want * 98 )) ]; then
                slurm_claim_done "${claims}" "${key}"
            else
                echo "[gen] ${task}/${lang}: FAILED (rc ${rc}, ${got}/${want} wavs)"
                echo "${task} ${lang}" >> "${failed}"
            fi
        done 3< "${list_file}"
    ) > "${base}_${node}-g${s}.log" 2>&1 &
    pids="${pids} $!"
done
for p in ${pids}; do wait "${p}"; done

while read -r _n task lang; do
    [ "$(cat "${claims}/${task}__${lang}/owner" 2>/dev/null)" = "${node}" ] || continue
    want=$(want_of "${task}" "${lang}")
    echo "COVERAGE ${system} ${task} ${lang} $(( want - $(missing_of "${task}" "${lang}") ))/${want}"
done < "${list_file}"
echo "[gen] wall ${SECONDS}s"

if [ -s "${failed}" ]; then
    echo "[gen] $(grep -c . "${failed}") failed leaves listed in ${failed}"
    exit 1
fi
