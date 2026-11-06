#!/usr/bin/env bash
#SBATCH --job-name=ztts-rescore-all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1500G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
# Rescore EVERY system leaf under the current benchmark definition (full rescore: WER with
# routed ASR, speaker similarity incl. the ITW held-out pass, MSR-UTMOS), so that every
# score file carries the BENCHMARK_VERSION / scorer stamps build_results.py requires.
#
# Leaves: results/<sys>/<task>/<lang>/wavs for every nested system (not gt_* / whisper_*
# / _*), every task in utils.benchmark.SOURCE_TASK (full sets and coreset panels), every
# lang in that task's frozen basket that has data/<task>/<lang>/text. Only leaves with
# a stale score kind (utils.benchmark.stale_kinds) are queued, largest first.
#
# Work is CLAIMED, not pre-assigned: array tasks start whenever the scheduler finds a node,
# so a fixed split would leave the first node idle while the rest of the list waited for
# nodes that had not started. Each GPU runs one ASR + one Whisper server and
# RESCORE_WORKERS_PER_GPU workers (default 2) that walk the list and take the next leaf
# whose claim (an atomic mkdir under logs/slurm/rescore-all-<array job>-claims/) they win.
# A task that starts after the list is fully claimed exits without starting servers.
#
#   sbatch --array=0-3 --chdir="$(pwd -P)" --export=ALL scripts/score/rescore_all.sh
# RESCORE_SYSTEMS="a b" restricts the leaves to those results/ dirs (default: every system).
#
# Requeue (preemption): claims record their array task; a restarted task first releases
# its previous incarnation's unfinished claims, and the version-aware gates make the
# rescored leaves resume where they stopped.
#
# Leaf failures are appended to logs/slurm/rescore-all-<array job>-failed.txt and do NOT
# fail the job (dependents still run; build_results.py surfaces any leaf left stale). The
# job exits 1 only if a GPU's servers failed to start.
set +u
# sbatch copies the script to spool, so $0 is not the repo path: rely on --chdir, fall
# back to the $0-relative root when run directly.
[ -f scripts/score/serve_asr.sh ] || cd "$(dirname "$0")/../.." || exit 1
[ -f scripts/score/serve_asr.sh ] || { echo "[rescore-all] ERROR: not at repo root" >&2; exit 1; }
mkdir -p logs/slurm

PY=.venv/bin/python
array_job="${SLURM_ARRAY_JOB_ID:-$$}"
node=${SLURM_ARRAY_TASK_ID:-0}
failed_file="logs/slurm/rescore-all-${array_job}-failed.txt"
list_file="logs/slurm/rescore-all-${array_job}-leaves.txt"
claims="logs/slurm/rescore-all-${array_job}-claims"

. slurm/lib_slurm.sh
n_gpu=$(slurm_gpu_count)
# Workers per GPU share that GPU's ASR + Whisper servers, overlapping one leaf's ASR
# with another's speaker-sim / MSR-UTMOS (measured ~88% GPU utilization at 2).
wpg=${RESCORE_WORKERS_PER_GPU:-2}
# Cap torch/BLAS threads per worker: uncapped scorers (~240 threads each) on 112 cores
# spent their time contending, not scoring.
cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
threads=$(( cpus / (n_gpu * wpg) )); [ "${threads}" -ge 1 ] || threads=1
export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads}

. scripts/score/lib_infer_scoring.sh

# The work list is computed ONCE per array job (first task to start, under a mkdir lock)
# and frozen: every task and every requeue reads the same file.
build_list() {
    "${PY}" - <<'PYEOF' | LC_ALL=C sort -k1,1nr -k2 | cut -d' ' -f2-
import os
import sys
from utils.benchmark import (SOURCE_TASK, is_flat, is_panel_alias, iter_leaves, langs_for,
                             stale_kinds)
want = set(os.environ.get("RESCORE_SYSTEMS", "").split())
for name in sorted(want):
    if not os.path.isdir(f"results/{name}"):
        print(f"[rescore-all] RESCORE_SYSTEMS: no results/{name}", file=sys.stderr)
n_leaves = 0
for task in SOURCE_TASK:
    for lang in langs_for(task):
        data = f"data/{task}/{lang}"
        if not os.path.isfile(f"{data}/text"):
            continue
        for sys_name, leaf in iter_leaves(task, lang, "wavs").items():
            if is_flat(sys_name) or (want and sys_name not in want):
                continue
            if is_panel_alias(f"results/{sys_name}", task):
                continue   # browse-only symlink to the full set: scoring it would overwrite that leaf
            n_leaves += 1
            if stale_kinds("system", data, leaf, lang):
                n_wavs = sum(1 for f in os.listdir(f"{leaf}/wavs") if f.endswith(".wav"))
                print(n_wavs, sys_name, task, lang)
print(f"[rescore-all] {n_leaves} leaves in total", file=sys.stderr)
PYEOF
}
slurm_list_once "${list_file}" build_list

# Release this task's claims left unfinished by a previous incarnation (requeue): the
# task id runs once at a time, so any such claim belongs to a dead process.
slurm_claims_init "${claims}" "${node}"

n_list=$(grep -c . "${list_file}")
n_open=0
while read -r sys_name task lang; do
    [ -n "${sys_name}" ] || continue
    [ -d "${claims}/${sys_name}__${task}__${lang}" ] || n_open=$(( n_open + 1 ))
done < "${list_file}"
echo "[rescore-all] task ${node}: ${n_open}/${n_list} leaves unclaimed (${list_file});" \
     "${n_gpu} GPU(s) x ${wpg} worker(s), ${threads} threads each"
[ "${n_open}" -gt 0 ] || { echo "[rescore-all] nothing left to claim"; exit 0; }

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export ASR_GPU=$s WHISPER_GPU=$s SCORE_GPU=$s
        port_base=$(slurm_port_base "${s}")
        export ASR_PORT=$(slurm_role_port "${port_base}" asr) WHISPER_PORT=$(slurm_role_port "${port_base}" whisper)
        # Leave room for the workers' torch scorers next to both servers: 0.35 each
        # left ~24 GB and long utterances OOMed in speaker-sim / MSR-UTMOS.
        export ASR_GPU_MEM_UTIL=${ASR_GPU_MEM_UTIL:-0.22} WHISPER_GPU_MEM_UTIL=${WHISPER_GPU_MEM_UTIL:-0.22}
        export SERVE_LOG_DIR=$(slurm_server_log_dir "rescore-all-${array_job}_${node}-s${s}")

        . scripts/score/serve_asr.sh
        trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
        start_asr_server || exit 1
        maybe_start_whisper || exit 1

        wpids=""
        for ((w = 0; w < wpg; w++)); do
            (
                SCORE_FAILS=0
                n_done=0
                while read -r sys_name task lang <&3; do
                    [ -n "${sys_name}" ] || continue
                    key="${sys_name}__${task}__${lang}"
                    slurm_claim "${claims}" "${key}" "${node}" || continue   # someone else has it
                    before=${SCORE_FAILS}
                    echo "=== [$(date +%T) task ${node} gpu ${s} worker ${w}] ${sys_name}/${task}/${lang} ==="
                    score_lang_dir "data/${task}/${lang}" "results/${sys_name}/${task}/${lang}" "${lang}"
                    if [ "${SCORE_FAILS}" -gt "${before}" ]; then
                        echo "${sys_name} ${task} ${lang}" >> "${failed_file}"
                    fi
                    slurm_claim_done "${claims}" "${key}"
                    n_done=$(( n_done + 1 ))
                done 3< "${list_file}"
                echo "[rescore-all] task ${node} gpu ${s} worker ${w} done:" \
                     "${n_done} leaves, ${SCORE_FAILS} failed"
            ) > "logs/slurm/rescore-all-${array_job}_${node}-s${s}w${w}.log" 2>&1 &
            wpids="${wpids} $!"
        done
        for p in ${wpids}; do wait "${p}"; done
        exit 0
    ) > "logs/slurm/rescore-all-${array_job}_${node}-s${s}.log" 2>&1 &
    pids="${pids} $!"
done

dead=0
for p in ${pids}; do
    wait "${p}" || dead=$(( dead + 1 ))
done
echo "[rescore-all] task ${node}: all GPUs finished (${dead} with failed servers)"
[ -f "${failed_file}" ] && echo "[rescore-all] failed leaves so far: $(wc -l < "${failed_file}") (${failed_file})"
[ "${dead}" -eq 0 ]
