#!/usr/bin/env bash
# slurm/eval_job.sh -- per-model SLURM job body: generate then score ONE local
# model in a single 1-GPU allocation. Submitted (one per model) by
# slurm/submit_eval.sh; not meant to be run by hand -- but it can be, if you set
# the EJ_* vars below and run it inside an salloc/srun shell.
#
# All configuration arrives as EJ_* environment variables (the submitter exports
# them and calls `sbatch --export=ALL`, which sidesteps the comma/space escaping
# problems of `sbatch --export=VAR=value` for multi-word test-set lists):
#   EJ_REPO       repo root (also passed as sbatch --chdir)          (required)
#   EJ_ENGINE     omni | zonos2 | kokoro | cosyvoice3 | moss-tts | neutts |
#                 vibevoice | vibevoice15 | audio8                    (required)
#   EJ_MODEL      the engine's model key: omni model key; moss-tts variant
#                 (moss-tts | moss-tts-v1.5 | moss-tts-local | moss-tts-local-v1.5);
#                 vibevoice15 1.5b | 7b; any label for the single-model engines
#                 (zonos2, kokoro, cosyvoice3, neutts, vibevoice, audio8)  (required)
#   EJ_TASK       clean | in-the-wild                     (default clean)
#   EJ_TEST_SET   space-separated langs; empty = the driver default  (default empty)
#   EJ_WORKERS    concurrent generation requests (omni, zonos2)      (default 16)
#   EJ_LIMIT      first N utts per set, 0 = all                      (default 0)
#   EJ_SCORE      true | false -- run WER/spk/MSR-UTMOS after gen     (default true)
#   EJ_OVERWRITE  true = regenerate existing wavs                    (default false)
#   EJ_SERVER     external zonos2 URL; EMPTY = self-manage the server (default empty)
#   EJ_ACCURATE_MODE  zonos2 --accurate-mode: true|false|empty        (default empty)
#                 true = accurate ("stability"), false = expressive; routes output to
#                 results/zonos2_accurate | results/zonos2_expressive respectively.
#   EJ_MODEL_PATH zonos2 model path/id for the launched server        (default serve default)
#
# Output lands in $(gen_default_out_root <driver> EJ_MODEL <mode>)/<task>
# (scripts/gen/lib_gen.sh), the driver's own default.
#
# GPU: this repo's server scripts default to CUDA index 0, which -- under this
# cluster's `ConstrainDevices=yes` cgroup isolation -- IS the single GPU SLURM
# allocated to the job. So no GPU juggling is needed; we only pin ASR_GPU /
# SCORE_GPU=0 to skip their nvidia-smi free-memory scans (which see just the one
# visible GPU anyway).
#
# The omni TTS server is stopped before the ASR server starts (serve_omni.sh waits
# for the port to free), so a single GPU serves both the generation and scoring
# phases with no contention.
#SBATCH --job-name=ztts-eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail

cd "${EJ_REPO:?EJ_REPO not set (repo root)}"

engine="${EJ_ENGINE:?EJ_ENGINE not set (omni|zonos2|kokoro|cosyvoice3|moss-tts|neutts|vibevoice|vibevoice15|audio8)}"
model="${EJ_MODEL:?EJ_MODEL not set}"
task="${EJ_TASK:-clean}"
test_set="${EJ_TEST_SET:-}"
workers="${EJ_WORKERS:-16}"
limit="${EJ_LIMIT:-0}"
do_score="${EJ_SCORE:-true}"
overwrite="${EJ_OVERWRITE:-false}"
server="${EJ_SERVER:-}"                 # empty = self-managed zonos2 server
accurate_mode="${EJ_ACCURATE_MODE:-}"  # zonos2: true|false|empty
model_path="${EJ_MODEL_PATH:-}"

case "${task}" in
    clean|in-the-wild) ;;
    *) echo "eval_job: unsupported EJ_TASK '${task}'" 1>&2; exit 1 ;;
esac

# Unique localhost ports per job so 1-GPU jobs packed onto the same node never
# collide on 127.0.0.1. slurm/lib_slurm.sh gives each job a port window keyed on
# SLURM_JOB_ID; this job uses slot 0's roles: tts (the omni / zonos2 / audio8 HTTP
# port; zonos2's Mini-SGLang IPC socket takes tts+1 = tts_aux), asr and whisper.
. slurm/lib_slurm.sh
job_id="${SLURM_JOB_ID:-$$}"
port_base=$(slurm_port_base 0)
tts_port=$(slurm_role_port "${port_base}" tts)
export ASR_PORT=$(slurm_role_port "${port_base}" asr)
export WHISPER_PORT=$(slurm_role_port "${port_base}" whisper)
# The single GPU visible inside the cgroup is index 0 for both the ASR server and
# the torch speaker-sim / MSR-UTMOS models; pin it so their scans don't run.
export ASR_GPU=0
export SCORE_GPU=0
export WHISPER_GPU=0
# ASR + Whisper co-locate on the one cgroup GPU during scoring (the TTS server is
# already stopped): 0.35 mem-util each (28GB of 80GB) fits both comfortably.
export ASR_GPU_MEM_UTIL=0.35
export WHISPER_GPU_MEM_UTIL=0.35

# Server logs on SHARED storage. The repo lives on /data (cluster-wide), whereas
# /tmp is node-local -- a server log written to /tmp on the compute node can't be
# read from the login node once the job moves on. Every server script writes
# ${SERVE_LOG_DIR}/<role>.log (utils/lib_serve.sh).
export SERVE_LOG_DIR=$(slurm_server_log_dir "${SLURM_JOB_ID:-local}")

# Stale foreign NVIDIA MPS control daemons on these nodes (another user's default
# /tmp/nvidia-mps) make CUDA fail at init with "Error 805: MPS client failed to
# connect to the MPS control daemon". Point CUDA at a private, daemon-less pipe dir
# so it ignores the foreign daemon and drives the GPU directly.
export CUDA_MPS_PIPE_DIRECTORY=$(slurm_mps_dir "${SLURM_JOB_ID:-$$}")

echo "=== eval_job: engine=${engine} model=${model} task=${task} test_set='${test_set:-<default>}' ==="
echo "=== node=$(hostname) job=${job_id} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ports tts=${tts_port} asr=${ASR_PORT} ==="
echo "=== server logs -> ${SERVE_LOG_DIR}/ (shared) ==="

# ---------------------------------------------------------------- generate ----
# engine -> driver (scripts/gen/run_<driver>.sh) and its arguments.
mode=""
case "${engine}" in
    omni)
        driver=omni
        gen_args=(--model "${model}" --task "${task}" --port "${tts_port}"
                  --workers "${workers}" --limit "${limit}")
        ;;
    zonos2)
        # Self-manage the server unless EJ_SERVER gives an external one. --gpu 0 is the
        # single cgroup-visible GPU; --port is the job's tts port (the server also binds
        # tts+1 for its IPC socket).
        driver=zonos2
        mode="${accurate_mode}"
        gen_args=(--gpu 0 --port "${tts_port}")
        [ -n "${server}" ] && gen_args=(--server "${server}")
        [ -n "${model_path}" ] && gen_args+=(--model-path "${model_path}")
        [ -n "${accurate_mode}" ] && gen_args+=(--accurate-mode "${accurate_mode}")
        gen_args+=(--task "${task}" --workers "${workers}" --limit "${limit}")
        ;;
    kokoro|cosyvoice3|neutts|vibevoice)
        # In-process engines with their own venvs, no server.
        case "${engine}" in cosyvoice3) driver=cosyvoice ;; *) driver=${engine} ;; esac
        gen_args=(--task "${task}" --limit "${limit}")
        ;;
    moss-tts)
        # MOSS-TTS family, in-process; EJ_MODEL is the variant, written to results/<variant>.
        driver=moss_tts
        gen_args=(--model "${model}" --task "${task}" --limit "${limit}")
        ;;
    vibevoice15)
        # VibeVoice-1.5B (zero-shot cloner) or its 7B backbone; same code and venv.
        case "${model}" in
            vibevoice15|1.5b) model=1.5b ;;
            7b|vibevoice7b)   model=7b ;;
            *) echo "eval_job: EJ_MODEL for vibevoice15 must be 1.5b or 7b (got '${model}')" 1>&2; exit 1 ;;
        esac
        driver=vibevoice15
        gen_args=(--model "${model}" --task "${task}" --limit "${limit}")
        ;;
    audio8)
        # Audio8-TTS-Realistic-0.6B via its SGLang-Omni adapter server, self-managed on the
        # cgroup GPU and stopped before scoring. No --workers: the client default (32)
        # matches the server's max running requests.
        driver=audio8
        gen_args=(--gpu 0 --port "${tts_port}" --task "${task}" --limit "${limit}")
        ;;
    *)
        echo "eval_job: unknown EJ_ENGINE '${engine}' (omni|zonos2|kokoro|cosyvoice3|moss-tts|neutts|vibevoice|vibevoice15|audio8)" 1>&2; exit 1 ;;
esac
gen_script="scripts/gen/run_${driver}.sh"
out_base=$(. scripts/gen/lib_gen.sh; gen_default_out_root "${driver}" "${model}" "${mode}")
# Append --test-set as ONE argument so multi-word lists ("en zh de") aren't split.
[ -n "${test_set}" ] && gen_args+=(--test-set "${test_set}")
[ "${overwrite}" = true ] && gen_args+=(--overwrite true)

echo "+ bash ${gen_script} ${gen_args[*]}"
bash "${gen_script}" "${gen_args[@]}"

# ------------------------------------------------------------------- score ----
if [ "${do_score}" = true ]; then
    decode_dir="${out_base}/${task}"
    # Score exactly what was generated: a driver may cover only part of the task's
    # basket. When no EJ_TEST_SET was given, derive the language list from the
    # generated subdirs.
    if [ -n "${test_set}" ]; then
        # Keep only languages the generator actually produced wavs for. A driver may
        # legitimately skip some of EJ_TEST_SET (unsupported language, missing data),
        # and scoring a language with no wavs is not a soft failure -- the scorers die
        # on the missing wav.scp and take the whole job down with them.
        score_ts=""
        for _l in ${test_set}; do
            [ -d "${decode_dir}/${_l}/wavs" ] && score_ts="${score_ts} ${_l}"
        done
        score_ts="$(echo ${score_ts})"
    else
        # `|| score_ts=""` keeps a missing/empty decode_dir from killing the job
        # via set -e; the empty list is handled by the skip branch below.
        score_ts="$(cd "${decode_dir}" 2>/dev/null && ls -d */ 2>/dev/null | tr -d / | tr '\n' ' ')" || score_ts=""
    fi
    # An empty score list must actually SKIP (otherwise scripts/score/run_score.sh would
    # score the task's whole basket).
    if [ -z "${score_ts// }" ]; then
        echo "=== nothing was generated for test_set='${test_set:-<derived>}' under ${decode_dir}; skipping scoring ==="
    else
        echo "+ bash scripts/score/run_score.sh --decode-dir ${decode_dir} --test-set \"${score_ts}\""
        bash scripts/score/run_score.sh --decode-dir "${decode_dir}" --test-set "${score_ts}"
    fi
else
    echo "=== EJ_SCORE=${do_score}; skipping scoring. Generated audio under ${out_base}/${task}/ ==="
fi

echo "=== eval_job done: ${engine}/${model}/${task} ==="
