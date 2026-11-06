#!/usr/bin/env bash
#SBATCH --job-name=ztts-native-gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
# Generate a system whose driver runs the model in-process (no server) on 8-GPU nodes, one
# generator process per GPU. The driver must take `--model M --task T --test_set LANG
# --out_root R --limit N` and generate one leaf per call: run_generate_moss_tts.sh (MOSS-TTS
# family), run_generate_ming3.sh (Ming-3.0-tts). Always submit as an ARRAY job (the %A_%a
# output name and the <array job>_<task> file tags assume one), from the repo root:
#
#   GEN_DRIVER=run_generate_moss_tts.sh GEN_MODEL=moss-tts-v1.5 sbatch --parsable \
#       --chdir="$(pwd -P)" --export=ALL --array=0-3 slurm/native_gen_job.sh
#
# Environment (--export=ALL):
#   GEN_DRIVER     driver script in the repo root                             (required)
#   GEN_MODEL      model key of the driver; results/<GEN_MODEL> by default    (required)
#   GEN_TASKS      space-separated tasks       (default "clean in-the-wild")
#   GEN_LANGS      only these langs; empty = each task's whole basket       (default '')
#   GEN_LIMIT      only the first N utterances per leaf                (default 0 = all)
#   GEN_OUT_ROOT   results root                           (default results/<GEN_MODEL>)
#
# Pending leaves (task x lang with a text uid lacking a wav) are listed once per array job
# (first task to start, under a mkdir lock), largest first, and frozen. Each GPU takes the
# next leaf whose claim (atomic mkdir under logs/slurm/ztts-native-gen-<array job>-claims/)
# it wins, so a task that starts late never leaves a fixed share idle. A leaf still missing
# wavs after a clean run is rerun once (only the missing utterances are retried). Wavs are
# written atomically and existing ones skipped, so a requeue or resubmit resumes; a
# requeued task releases its previous incarnation's unfinished claims, and a leaf whose
# generator exited non-zero keeps an unfinished claim so a requeue retries it. Prints
# "COVERAGE <model> <task> <lang> <got>/<want>" per leaf this task ran; exits 1 iff a leaf
# is listed in logs/slurm/ztts-native-gen-<array job>_<task>-failed.txt.
set -uo pipefail

[ -n "${GEN_DRIVER:-}" ] || { echo "[native-gen] ERROR: GEN_DRIVER is empty" >&2; exit 1; }
[ -f "${GEN_DRIVER}" ] || { echo "[native-gen] ERROR: no ${GEN_DRIVER}: run from the repo root (sbatch --chdir)" >&2; exit 1; }
[ -n "${GEN_MODEL:-}" ] || { echo "[native-gen] ERROR: GEN_MODEL is empty" >&2; exit 1; }
GEN_TASKS=${GEN_TASKS:-"clean in-the-wild"}
GEN_LANGS=${GEN_LANGS:-}
GEN_LIMIT=${GEN_LIMIT:-0}
GEN_OUT_ROOT=${GEN_OUT_ROOT:-results/${GEN_MODEL}}

job=${SLURM_JOB_ID:-$$}
array_job=${SLURM_ARRAY_JOB_ID:-$job}
node=${SLURM_ARRAY_TASK_ID:-0}
n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${n_gpu}" -ge 1 ] || n_gpu=1
cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
threads=$(( cpus / n_gpu )); [ "${threads}" -ge 1 ] || threads=1
export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads}

mkdir -p logs/slurm
base=logs/slurm/ztts-native-gen-${array_job}
list_file=${base}-leaves.txt
claims=${base}-claims
failed=${base}_${node}-failed.txt

want_of() {  # TASK LANG -> utterances the leaf should have
    local n; n=$(grep -c . "data/$1/$2/text")
    if [ "${GEN_LIMIT}" -gt 0 ] && [ "${GEN_LIMIT}" -lt "${n}" ]; then n=${GEN_LIMIT}; fi
    echo "${n}"
}
missing_of() {  # TASK LANG -> wanted text uids without a wav
    LC_ALL=C comm -23 \
        <(awk 'NF {print $1}' "data/$1/$2/text" | head -n "$(want_of "$1" "$2")" | LC_ALL=C sort -u) \
        <(find "${GEN_OUT_ROOT}/$1/$2/wavs" -maxdepth 1 -name '*.wav' -printf '%f\n' 2>/dev/null \
            | sed 's/\.wav$//' | LC_ALL=C sort -u) \
        | grep -c .
}

if [ ! -f "${list_file}" ]; then
    if mkdir "${list_file}.lock" 2>/dev/null; then
        for task in ${GEN_TASKS}; do
            for lang in $(python3 -m utils.benchmark langs "${task}"); do
                [ -f "data/${task}/${lang}/text" ] || continue
                if [ -n "${GEN_LANGS}" ]; then
                    case " ${GEN_LANGS} " in *" ${lang} "*) ;; *) continue ;; esac
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
echo "[native-gen] task ${node}: ${GEN_MODEL} via ${GEN_DRIVER}, $(grep -c . "${list_file}") pending leaves" \
     "(${list_file}), ${n_gpu} GPU(s), ${threads} threads each"

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export CUDA_VISIBLE_DEVICES=${s}
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nompss-${USER:-u}-${job}-${s}
        run_leaf() {
            bash "${GEN_DRIVER}" --model "${GEN_MODEL}" --task "$1" --test_set "$2" \
                --out_root "${GEN_OUT_ROOT}" --limit "${GEN_LIMIT}" < /dev/null
        }
        while read -r _n task lang <&3; do
            c="${claims}/${task}__${lang}"
            mkdir "${c}" 2>/dev/null || continue
            echo "${node}" > "${c}/owner"
            echo "=== [$(date +%T) task ${node} gpu ${s}] ${GEN_MODEL} ${task} ${lang} ==="
            run_leaf "${task}" "${lang}"; rc=$?
            if [ "${rc}" -eq 0 ] && [ "$(missing_of "${task}" "${lang}")" -gt 0 ]; then
                echo "[native-gen] ${task}/${lang}: wavs still missing, rerunning once"
                run_leaf "${task}" "${lang}"; rc=$?
            fi
            if [ "${rc}" -eq 0 ]; then
                touch "${c}/done"
            else
                echo "${task} ${lang}" >> "${failed}"
            fi
        done 3< "${list_file}"
    ) > "${base}_${node}-g${s}.log" 2>&1 &
    pids="${pids} $!"
done
for p in ${pids}; do wait "${p}"; done

low=0
while read -r _n task lang; do
    [ "$(cat "${claims}/${task}__${lang}/owner" 2>/dev/null)" = "${node}" ] || continue
    want=$(want_of "${task}" "${lang}")
    got=$(( want - $(missing_of "${task}" "${lang}") ))
    echo "COVERAGE ${GEN_MODEL} ${task} ${lang} ${got}/${want}"
    [ $(( got * 100 )) -ge $(( want * 98 )) ] || low=$(( low + 1 ))
done < "${list_file}"
echo "[native-gen] task ${node}: ${low} leaf/leaves below 98% generation coverage"
echo "[native-gen] wall ${SECONDS}s"

if [ -s "${failed}" ]; then
    echo "[native-gen] $(grep -c . "${failed}") failed leaves listed in ${failed}"
    exit 1
fi
