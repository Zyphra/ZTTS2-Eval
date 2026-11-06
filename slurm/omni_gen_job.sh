#!/usr/bin/env bash
#SBATCH --job-name=ztts-omni-gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
# Generate a vLLM-Omni model (run_generate_omni.sh / run_generate_omni_voxblink.sh, which
# start and stop their own server) on 8-GPU nodes, one server + client per GPU. Always
# submit as an ARRAY job (the %A_%a output name and the <array job>_<task> file tags
# assume one), from the repo root:
#
#   OMNI_GEN_MODEL=omnivoice sbatch --parsable --chdir="$(pwd -P)" --export=ALL \
#       --array=0-1 slurm/omni_gen_job.sh
#
# Environment (--export=ALL):
#   OMNI_GEN_MODEL    model key of scripts/omni_models.sh                       (required)
#   OMNI_GEN_TASKS    space-separated tasks      (default "clean in-the-wild")
#   OMNI_GEN_LANGS    only these langs; empty = each task's whole basket      (default '')
#   OMNI_GEN_LIMIT    only the first N utterances per leaf               (default 0 = all)
#   OMNI_GEN_WORKERS  concurrent requests per server               (default 16, as eval_job)
#   OMNI_VENV         venv with vllm-omni installed (read by the drivers)  (default .venv)
#   OMNI_OUT_ROOT     results root (read by the drivers); leaves land in
#                     <root>/omni_<model>/<task>/<lang>                   (default results)
#
# Pending leaves (task x lang with a text uid lacking a wav) are listed once per array job
# (first task to start, under a mkdir lock), largest first, and frozen. Each GPU takes the
# next leaf whose claim (atomic mkdir under logs/slurm/ztts-omni-gen-<array job>-claims/)
# it wins and runs the driver for that one leaf, so every leaf starts its own server on
# that GPU (port window per GPU, server log under logs/slurm/servers/<job>-g<gpu>/). The
# servers resolve the model from the local HF cache only (HF_HUB_OFFLINE=1): pre-download
# it on a networked host first. A leaf still missing wavs after a clean run is rerun once
# (only the missing utterances are retried). Wavs are written atomically and existing ones
# skipped, so a requeue or resubmit resumes; a requeued task releases its previous
# incarnation's unfinished claims. A leaf is FAILED when its driver exits non-zero or it
# ends below 98% coverage (the drivers exit 0 when the server fails to start or an
# utterance fails); its claim stays unfinished so a requeue retries it. Prints
# "COVERAGE omni_<model> <task> <lang> <got>/<want>" per leaf this task ran; exits 1 iff a
# leaf is listed in logs/slurm/ztts-omni-gen-<array job>_<task>-failed.txt.
set -uo pipefail

[ -f run_generate_omni.sh ] || { echo "[omni-gen] ERROR: run from the repo root (sbatch --chdir)" >&2; exit 1; }
[ -n "${OMNI_GEN_MODEL:-}" ] || { echo "[omni-gen] ERROR: OMNI_GEN_MODEL is empty" >&2; exit 1; }
OMNI_GEN_TASKS=${OMNI_GEN_TASKS:-"clean in-the-wild"}
OMNI_GEN_LANGS=${OMNI_GEN_LANGS:-}
OMNI_GEN_LIMIT=${OMNI_GEN_LIMIT:-0}
OMNI_GEN_WORKERS=${OMNI_GEN_WORKERS:-16}
system=omni_${OMNI_GEN_MODEL}
leaf_root=${OMNI_OUT_ROOT:-results}/${system}
export HF_HUB_OFFLINE=1

job=${SLURM_JOB_ID:-$$}
array_job=${SLURM_ARRAY_JOB_ID:-$job}
node=${SLURM_ARRAY_TASK_ID:-0}
n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${n_gpu}" -ge 1 ] || n_gpu=1

mkdir -p logs/slurm
base=logs/slurm/ztts-omni-gen-${array_job}
list_file=${base}-leaves.txt
claims=${base}-claims
failed=${base}_${node}-failed.txt

want_of() {  # TASK LANG -> utterances the leaf should have
    local n; n=$(grep -c . "data/$1/$2/text")
    if [ "${OMNI_GEN_LIMIT}" -gt 0 ] && [ "${OMNI_GEN_LIMIT}" -lt "${n}" ]; then n=${OMNI_GEN_LIMIT}; fi
    echo "${n}"
}
missing_of() {  # TASK LANG -> wanted text uids without a wav
    LC_ALL=C comm -23 \
        <(awk 'NF {print $1}' "data/$1/$2/text" | head -n "$(want_of "$1" "$2")" | LC_ALL=C sort -u) \
        <(find "${leaf_root}/$1/$2/wavs" -maxdepth 1 -name '*.wav' -printf '%f\n' 2>/dev/null \
            | sed 's/\.wav$//' | LC_ALL=C sort -u) \
        | grep -c .
}

if [ ! -f "${list_file}" ]; then
    if mkdir "${list_file}.lock" 2>/dev/null; then
        for task in ${OMNI_GEN_TASKS}; do
            for lang in $(python3 -m utils.benchmark langs "${task}"); do
                [ -f "data/${task}/${lang}/text" ] || continue
                if [ -n "${OMNI_GEN_LANGS}" ]; then
                    case " ${OMNI_GEN_LANGS} " in *" ${lang} "*) ;; *) continue ;; esac
                fi
                m=$(missing_of "${task}" "${lang}")
                [ "${m}" -gt 0 ] && echo "${m} ${task} ${lang}"
            done
        done | LC_ALL=C sort -k1,1nr -k2 > "${list_file}.tmp"
        mv "${list_file}.tmp" "${list_file}"
    else
        until [ -f "${list_file}" ]; do sleep 5; done
    fi
fi
mkdir -p "${claims}"
for c in "${claims}"/*/; do
    [ -d "${c}" ] || continue
    [ -f "${c}done" ] && continue
    [ "$(cat "${c}owner" 2>/dev/null)" = "${node}" ] && rm -rf "${c}"
done
rm -f "${failed}"
echo "[omni-gen] task ${node}: ${system}, $(grep -c . "${list_file}") pending leaves" \
     "(${list_file}), ${n_gpu} GPU(s), ${OMNI_GEN_WORKERS} workers each," \
     "venv ${OMNI_VENV:-.venv}"

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nompss-${USER:-u}-${job}-${s}
        export OMNI_LOG_DIR=logs/slurm/servers/${job}-g${s}
        mkdir -p "${OMNI_LOG_DIR}"
        port=$(( 20000 + (job % 1000) * 40 + s * 4 ))
        run_leaf() {
            if [ "$1" = in-the-wild ]; then
                bash run_generate_omni_voxblink.sh --model "${OMNI_GEN_MODEL}" --test_set "$2" \
                    --gpu "${s}" --port "${port}" --workers "${OMNI_GEN_WORKERS}" \
                    --limit "${OMNI_GEN_LIMIT}" < /dev/null
            else
                bash run_generate_omni.sh --model "${OMNI_GEN_MODEL}" --task "$1" --test_set "$2" \
                    --gpu "${s}" --port "${port}" --workers "${OMNI_GEN_WORKERS}" \
                    --limit "${OMNI_GEN_LIMIT}" < /dev/null
            fi
        }
        while read -r _n task lang <&3; do
            c="${claims}/${task}__${lang}"
            mkdir "${c}" 2>/dev/null || continue
            echo "${node}" > "${c}/owner"
            echo "=== [$(date +%T) task ${node} gpu ${s}] ${system} ${task} ${lang} ==="
            run_leaf "${task}" "${lang}"; rc=$?
            if [ "${rc}" -eq 0 ] && [ "$(missing_of "${task}" "${lang}")" -gt 0 ]; then
                echo "[omni-gen] ${task}/${lang}: wavs still missing, rerunning once"
                run_leaf "${task}" "${lang}"; rc=$?
            fi
            want=$(want_of "${task}" "${lang}")
            got=$(( want - $(missing_of "${task}" "${lang}") ))
            if [ "${rc}" -eq 0 ] && [ $(( got * 100 )) -ge $(( want * 98 )) ]; then
                touch "${c}/done"
            else
                echo "[omni-gen] ${task}/${lang}: FAILED (rc ${rc}, ${got}/${want} wavs)"
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
echo "[omni-gen] wall ${SECONDS}s"

if [ -s "${failed}" ]; then
    echo "[omni-gen] $(grep -c . "${failed}") failed leaves listed in ${failed}"
    exit 1
fi
