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
# Leaves: results/<sys>/<task>/<lang>/wavs for every sys not starting with gt_, whisper_
# or _, every task in utils.benchmark.SOURCE_TASK (full sets and coreset panels), every
# lang in that task's frozen basket that has data/<task>/<lang>/text. The sorted leaf list
# is dealt round-robin over node x GPU shards; each GPU shard is one background subshell
# with its own ASR + Whisper servers and port window, scoring leaves through
# scripts/lib_infer_scoring.sh:score_lang_dir. Its version-aware gates make a requeue
# (preemption) resume at the first stale leaf.
#
#   sbatch --array=0-3 --chdir="$(pwd -P)" --export=ALL scripts/rescore_all.sh
#
# Leaf failures are appended to logs/slurm/rescore-all-<array job>-failed.txt and do NOT
# fail the job (dependents still run; build_results.py surfaces any leaf left stale). The
# job exits 1 only if a shard died, i.e. a server failed to start.
set +u
# sbatch copies the script to spool, so $0 is not the repo path: rely on --chdir, fall
# back to the $0-relative root when run directly.
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[rescore-all] ERROR: not at repo root" >&2; exit 1; }
mkdir -p logs/slurm

PY=.venv/bin/python
array_job="${SLURM_ARRAY_JOB_ID:-$$}"
failed_file="logs/slurm/rescore-all-${array_job}-failed.txt"

leaves=$(python3 - <<'PYEOF' | LC_ALL=C sort
import os
from utils.benchmark import SOURCE_TASK, langs_for
for sys_name in sorted(os.listdir("results")):
    if sys_name.startswith(("gt_", "whisper_", "_")) or not os.path.isdir(f"results/{sys_name}"):
        continue
    for task in SOURCE_TASK:
        for lang in langs_for(task):
            if (os.path.isdir(f"results/{sys_name}/{task}/{lang}/wavs")
                    and os.path.isfile(f"data/{task}/{lang}/text")):
                print(sys_name, task, lang)
PYEOF
)
n_nodes=${SLURM_ARRAY_TASK_COUNT:-1}
node=${SLURM_ARRAY_TASK_ID:-0}
n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${n_gpu}" -ge 1 ] || n_gpu=1
# Workers per GPU share that GPU's ASR + Whisper servers. The scorers are CPU-bound
# (decode/resample) and the servers are idle while they run, so two workers per GPU
# overlap one leaf's ASR with another's speaker-sim / MSR-UTMOS.
wpg=${RESCORE_WORKERS_PER_GPU:-2}
total=$(( n_nodes * n_gpu * wpg ))
# Cap torch/BLAS threads per worker: 16 uncapped scorers (~240 threads each) on 112
# cores spent their time contending, not scoring.
cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
threads=$(( cpus / (n_gpu * wpg) )); [ "${threads}" -ge 1 ] || threads=1
export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads}

# Keep only the leaves that still have work (same gates as score_lang_dir), largest
# first, so the static round-robin below balances the remaining cost rather than the
# leaf count; on a resume most leaves are already current. Array tasks start at
# different times, so the list is computed ONCE per array job (first task to start,
# under a mkdir lock) and frozen in a file every task -- and every requeue -- reads:
# recomputing it per task would give the tasks different assignments.
. scripts/lib_infer_scoring.sh
list_file="logs/slurm/rescore-all-${array_job}-leaves.txt"
if [ ! -f "${list_file}" ]; then
    if mkdir "${list_file}.lock" 2>/dev/null; then
        stale=""
        while read -r sys_name task lang; do
            [ -n "${sys_name}" ] || continue
            leaf_current "data/${task}/${lang}" "results/${sys_name}/${task}/${lang}" && continue
            n=$(find "results/${sys_name}/${task}/${lang}/wavs/" -name '*.wav' | wc -l)
            stale+="${n} ${sys_name} ${task} ${lang}"$'\n'
        done <<< "${leaves}"
        printf '%s' "${stale}" | LC_ALL=C sort -k1,1nr -k2 | cut -d' ' -f2- > "${list_file}.tmp"
        mv "${list_file}.tmp" "${list_file}"
    else
        until [ -f "${list_file}" ]; do sleep 5; done
    fi
fi
stale=$(cat "${list_file}")
n_stale=$(printf '%s\n' "${stale}" | grep -c .)
echo "[rescore-all] ${n_leaves} leaves, ${n_stale} with work (${list_file}); node ${node}/${n_nodes}," \
     "${n_gpu} GPU(s) x ${wpg} worker(s), ${threads} threads each"

pids=""
for ((s = 0; s < n_gpu; s++)); do
    (
        export ASR_GPU=$s WHISPER_GPU=$s SCORE_GPU=$s
        port_base=$(( 20000 + (${SLURM_JOB_ID:-$$} % 1000) * 40 + s * 4 ))
        export ASR_PORT=$(( port_base + 1 )) WHISPER_PORT=$(( port_base + 2 ))
        # Leave room for two workers' torch scorers next to both servers: 0.35 each
        # left ~24 GB and long utterances OOMed in speaker-sim / MSR-UTMOS.
        export ASR_GPU_MEM_UTIL=${ASR_GPU_MEM_UTIL:-0.22} WHISPER_GPU_MEM_UTIL=${WHISPER_GPU_MEM_UTIL:-0.22}
        srv="logs/slurm/servers/rescore-all-${array_job}_${node}-s${s}"
        mkdir -p "${srv}"
        export ASR_LOG="${srv}/asr.log" WHISPER_LOG="${srv}/whisper.log"

        . scripts/serve_asr.sh
        trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
        start_asr_server || exit 1
        maybe_start_whisper || exit 1

        wpids=""
        for ((w = 0; w < wpg; w++)); do
            g=$(( (node * n_gpu + s) * wpg + w ))
            (
                SCORE_FAILS=0
                i=0
                while read -r sys_name task lang <&3; do
                    [ -n "${sys_name}" ] || continue
                    if [ $(( i % total )) -eq "${g}" ]; then
                        before=${SCORE_FAILS}
                        echo "=== [worker ${g}] ${sys_name}/${task}/${lang} ==="
                        score_lang_dir "data/${task}/${lang}" "results/${sys_name}/${task}/${lang}" "${lang}"
                        if [ "${SCORE_FAILS}" -gt "${before}" ]; then
                            echo "${sys_name} ${task} ${lang}" >> "${failed_file}"
                        fi
                    fi
                    i=$(( i + 1 ))
                done 3<<< "${stale}"
                echo "[rescore-all] worker ${g} done (${SCORE_FAILS} failed leaf/leaves)"
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
echo "[rescore-all] node ${node}: all shards finished (${dead} shard(s) died)"
[ -f "${failed_file}" ] && echo "[rescore-all] failed leaves so far: $(wc -l < "${failed_file}") (${failed_file})"
[ "${dead}" -eq 0 ]
