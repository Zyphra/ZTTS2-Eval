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
#   EJ_ENGINE     omni|zonos2|kokoro|cosyvoice3|moss-tts|neutts|vibevoice|
#                 vibevoice15|audio8                                 (required)
#   EJ_MODEL      omni model key, or "zonos2" for the zonos2 engine  (required)
#   EJ_TASK       zero_shot | zero_shot_voxblink                     (default zero_shot)
#   EJ_TEST_SET   space-separated langs; empty = the script default  (default empty)
#   EJ_WORKERS    concurrent generation requests                     (default 16)
#   EJ_LIMIT      first N utts per set, 0 = all                      (default 0)
#   EJ_SCORE      true | false -- run WER/spk/MSR-UTMOS after gen     (default true)
#   EJ_SERVER     external zonos2 URL; EMPTY = self-manage the server (default empty)
#   EJ_ACCURATE_MODE  zonos2 accurate_mode: true|false|empty          (default empty)
#                 true = accurate ("stability"), false = expressive; routes output to
#                 results/zonos2_accurate | results/zonos2_expressive respectively.
#   EJ_MODEL_PATH zonos2 model path/id for the launched server        (default serve default)
#
# GPU: this repo's server scripts default to CUDA index 0, which -- under this
# cluster's `ConstrainDevices=yes` cgroup isolation -- IS the single GPU SLURM
# allocated to the job. So no GPU juggling is needed; we only pin ASR_GPU /
# SCORE_GPU=0 to skip their nvidia-smi free-memory scans (which see just the one
# visible GPU anyway).
#
# The omni TTS server is stopped before the ASR server starts (serve_omni.sh waits
# for the port to free), so a single GPU serves both the generation and scoring
# phases with no contention -- the same shape as scripts/run_full_zonos2_generic.sh.
#SBATCH --job-name=ztts-eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail

cd "${EJ_REPO:?EJ_REPO not set (repo root)}"

engine="${EJ_ENGINE:?EJ_ENGINE not set (omni|zonos2)}"
model="${EJ_MODEL:?EJ_MODEL not set}"
task="${EJ_TASK:-zero_shot}"
test_set="${EJ_TEST_SET:-}"
workers="${EJ_WORKERS:-16}"
limit="${EJ_LIMIT:-0}"
do_score="${EJ_SCORE:-true}"
overwrite="${EJ_OVERWRITE:-false}"
server="${EJ_SERVER:-}"                 # empty = self-managed zonos2 server
accurate_mode="${EJ_ACCURATE_MODE:-}"  # zonos2: true|false|empty
model_path="${EJ_MODEL_PATH:-}"

case "${task}" in
    zero_shot|zero_shot_voxblink) ;;
    *) echo "eval_job: unsupported EJ_TASK '${task}'" 1>&2; exit 1 ;;
esac

# Unique localhost ports per job so 1-GPU jobs packed onto the same node never
# collide on 127.0.0.1 (which would make one job silently reuse another's
# wrong-model server -- see slurm/README.md). SLURM_JOB_ID is distinct per job.
#
# STRIDE of 4: each job owns a 4-port window [base .. base+3]. A job uses base (omni
# or zonos2 HTTP), base+1 (ASR / zonos2's Mini-SGLang IPC socket) and base+2 (whisper);
# the next job's base is +4 away, so consecutive job IDs never overlap on a sub-port.
# (Stride 1 collided: zonos2 job N's IPC on base+1 == zonos2 job N+1's HTTP port.)
job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 20000 + (job_id % 10000) * 4 ))
export OMNI_PORT=$(( port_base ))
export ASR_PORT=$(( port_base + 1 ))
export WHISPER_PORT=$(( port_base + 2 ))
# The single GPU visible inside the cgroup is index 0 for both the ASR server and
# the torch speaker-sim / MSR-UTMOS models; pin it so their scans don't run.
export ASR_GPU=0
export SCORE_GPU=0
# Per-language ASR routing, matching how the GT baselines were scored: langs NOT in
# QWEN_LANGS (utils/benchmark.py: zh ko th vi hi fa) are transcribed by Whisper
# large-v3 with a
# FORCED language instead of Qwen3-ASR auto-detect. Without this, synth for langs
# like uk/bg/hr/ca/nb/sk is auto-detected and misrouted to a phonetic neighbour
# (uk->Czech, bg->Russian, ca->Spanish), inflating WER to 50-99% while GT stays low.
# Whisper co-locates on the same cgroup GPU at 0.35 mem-util (WHISPER_GPU=0).
export ASR_ROUTING=1
export WHISPER_GPU=0
# ASR + Whisper co-locate on the one cgroup GPU during scoring (the TTS server is
# already stopped). Each defaults to 0.5 mem-util -> 100% together -> Whisper OOMs at
# init; 0.35 each (28GB of 80GB) fits both comfortably.
export ASR_GPU_MEM_UTIL=0.35
export WHISPER_GPU_MEM_UTIL=0.35

# Server logs on SHARED storage. The repo lives on /data (cluster-wide), whereas
# /tmp is node-local -- a server log written to /tmp on the compute node can't be
# read from the login node once the job moves on. Keep them per-job under the repo.
srv_log_dir="logs/slurm/servers/${SLURM_JOB_ID:-local}"
mkdir -p "${srv_log_dir}"
export OMNI_LOG_DIR="${srv_log_dir}"                       # run_generate_omni*.sh reads this
export ZONOS2_LOG="${srv_log_dir}/zonos2_${OMNI_PORT}.log" # serve_zonos2.sh reads this
export AUDIO8_LOG="${srv_log_dir}/audio8_${OMNI_PORT}.log" # serve_audio8.sh reads this
export ASR_LOG="${srv_log_dir}/vllm_asr_${ASR_PORT}.log"   # serve_asr.sh reads this
export WHISPER_LOG="${srv_log_dir}/vllm_whisper_${WHISPER_PORT}.log"

# Stale foreign NVIDIA MPS control daemons on these nodes (another user's default
# /tmp/nvidia-mps) make CUDA fail at init with "Error 805: MPS client failed to
# connect to the MPS control daemon". Point CUDA at a private, daemon-less pipe dir
# so it ignores the foreign daemon and drives the GPU directly.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nompss-${USER:-u}-${SLURM_JOB_ID:-$$}"

echo "=== eval_job: engine=${engine} model=${model} task=${task} test_set='${test_set:-<default>}' ==="
echo "=== node=$(hostname) job=${job_id} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ports omni=${OMNI_PORT} asr=${ASR_PORT} ==="
echo "=== server logs -> ${srv_log_dir}/ (shared) ==="

# ---------------------------------------------------------------- generate ----
gen_args=()
case "${engine}" in
    omni)
        out_base="results/omni_${model}"
        if [ "${task}" = zero_shot_voxblink ]; then
            gen_script="run_generate_omni_voxblink.sh"   # task is pinned; no --task
            gen_args=(--model "${model}")
        else
            gen_script="run_generate_omni.sh"
            gen_args=(--model "${model}" --task "${task}")
        fi
        gen_args+=(--port "${OMNI_PORT}" --workers "${workers}" --limit "${limit}")
        ;;
    zonos2)
        # Mode-aware output root so the two accurate_mode variants never collide:
        #   accurate ("stability") -> results/zonos2_accurate
        #   expressive             -> results/zonos2_expressive
        #   unset                  -> results/zonos2 (server default)
        case "${accurate_mode}" in
            true)  out_base="results/zonos2_accurate" ;;
            false) out_base="results/zonos2_expressive" ;;
            *)     out_base="results/zonos2" ;;
        esac
        # Self-manage the server unless EJ_SERVER gives an external one. --gpu 0 is the
        # single cgroup-visible GPU; --port reuses the per-job base (the server also
        # binds base+1 for its IPC socket, free until the ASR scorer starts later).
        zonos2_srv=(--gpu 0 --port "${OMNI_PORT}")
        [ -n "${server}" ] && zonos2_srv=(--server "${server}")
        [ -n "${model_path}" ] && zonos2_srv+=(--model_path "${model_path}")
        [ -n "${accurate_mode}" ] && zonos2_srv+=(--accurate_mode "${accurate_mode}")
        if [ "${task}" = zero_shot_voxblink ]; then
            gen_script="run_generate_zonos2_voxblink.sh"  # task is pinned; no --task
            gen_args=("${zonos2_srv[@]}" --out_root "${out_base}/zero_shot_voxblink")
        else
            gen_script="run_generate_zonos2.sh"
            gen_args=("${zonos2_srv[@]}" --task "${task}" --out_root "${out_base}")
        fi
        gen_args+=(--workers "${workers}" --limit "${limit}")
        ;;
    kokoro)
        # Kokoro is an in-process preset-voice model (its own venv ~/.venvs/kokoro,
        # no server, no ports). One script handles both tasks via --task; it maps
        # langs to Kokoro voices and skips unsupported ones (de/ko/ru). Preset voice
        # -> speaker-sim is not meaningful, but WER/MSR-UTMOS scoring below still runs.
        out_base="results/kokoro"
        gen_script="run_generate_kokoro.sh"
        gen_args=(--task "${task}" --limit "${limit}")
        ;;
    cosyvoice3)
        # CosyVoice3 native (upstream) -- in-process, its own venv ~/.venvs/cosyvoice,
        # no server. WITH-ref-text cloner, multilingual (all 9 langs). One script
        # handles both tasks via --task.
        out_base="results/cosyvoice3"
        gen_script="run_generate_cosyvoice.sh"
        gen_args=(--task "${task}" --limit "${limit}")
        ;;
    moss-tts)
        # MOSS-TTS family native (upstream) -- in-process, own venv ~/.venvs/moss-tts,
        # no server (vllm-omni's moss_tts codec is weight-incompatible; upstream works).
        # EJ_MODEL is the variant key (moss-tts | moss-tts-v1.5 | moss-tts-local |
        # moss-tts-local-v1.5 | moss-tts-nano); each writes results/<variant>.
        out_base="results/${model}"
        gen_script="run_generate_moss_tts.sh"
        gen_args=(--model "${model}" --task "${task}" --limit "${limit}")
        ;;
    neutts)
        # NeuTTS-2E native -- in-process, own venv ~/.venvs/neutts, no server. Preset
        # voice, ENGLISH ONLY (only the en set is generated + scored).
        out_base="results/neutts-2e"
        gen_script="run_generate_neutts.sh"
        gen_args=(--task "${task}" --limit "${limit}")
        ;;
    vibevoice)
        # VibeVoice-Realtime-0.5B native -- in-process, own venv ~/.venvs/vibevoice, no
        # server. Preset-voice streaming TTS. An empty EJ_TEST_SET covers the 10 langs
        # upstream ships preset voices for (en + 9 vendor-flagged exploratory); the
        # driver filters any explicit list down to those. See run_generate_vibevoice.sh.
        out_base="results/vibevoice-rt-0.5b"
        gen_script="run_generate_vibevoice.sh"
        gen_args=(--task "${task}" --limit "${limit}")
        ;;
    vibevoice15)
        # VibeVoice-1.5B native -- in-process, own venv ~/.venvs/vibevoice15, no server.
        # A real zero-shot CLONER (takes prompt_wav.scp), so speaker-sim is meaningful
        # here, unlike the preset-voice vibevoice engine above. English + Chinese only;
        # the driver filters any other language out. Inference code is VENDORED --
        # see third_party/VibeVoice-1.5B/PROVENANCE.md.
        # EJ_MODEL selects the variant: 1.5b (microsoft) or 7b (third-party backup of
        # the withdrawn VibeVoice-Large). Same code, same venv, different backbone.
        case "${model}" in
            vibevoice15|1.5b) vv_model=1.5b; out_base="results/vibevoice-1.5b" ;;
            7b|vibevoice7b)   vv_model=7b;   out_base="results/vibevoice-7b" ;;
            *) echo "eval_job: EJ_MODEL for vibevoice15 must be 1.5b or 7b (got '${model}')" 1>&2; exit 1 ;;
        esac
        gen_script="run_generate_vibevoice15.sh"
        gen_args=(--model "${vv_model}" --task "${task}" --limit "${limit}")
        ;;
    audio8)
        # Audio8-TTS-Realistic-0.6B via the upstream SGLang-Omni adapter server
        # (scripts/serve_audio8.sh, venv ~/.venvs/audio8-sglang), self-managed on the cgroup
        # GPU, stopped before scoring. WITH-ref-text cloner (prompt wav + prompt_text).
        # No --workers: the client default (32) matches the server's max running requests.
        out_base="results/audio8-tts-realistic-0.6b"
        gen_script="run_generate_audio8.sh"
        gen_args=(--gpu 0 --port "${OMNI_PORT}" --task "${task}" --limit "${limit}")
        ;;
    *)
        echo "eval_job: unknown EJ_ENGINE '${engine}' (omni|zonos2|kokoro|cosyvoice3|moss-tts|neutts|vibevoice|vibevoice15|audio8)" 1>&2; exit 1 ;;
esac
# Append --test_set as ONE argument so multi-word lists ("en zh de") aren't split.
[ -n "${test_set}" ] && gen_args+=(--test_set "${test_set}")
[ "${overwrite}" = true ] && gen_args+=(--overwrite true)

echo "+ bash ${gen_script} ${gen_args[*]}"
bash "${gen_script}" "${gen_args[@]}"

# ------------------------------------------------------------------- score ----
if [ "${do_score}" = true ]; then
    decode_dir="${out_base}/${task}"
    score_args=()
    if [ "${task}" = zero_shot_voxblink ]; then
        score_script="run_infer_voxblink_eval.sh"        # task pinned inside
    else
        score_script="run_infer_zonos2_eval.sh"
        score_args=(--task "${task}")
    fi
    # Score exactly what was generated. The run_infer_* scripts default to test_set="en"
    # only, but generation with an empty --test_set covers ALL of the task's languages
    # (sets_for), so without this the other 8 zero_shot langs would go unscored. When no
    # EJ_TEST_SET was given, derive the language list from the generated subdirs.
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
    # An empty score list must actually SKIP (this used to fall through and let
    # the run_infer script score its own unrelated default test_set).
    if [ -z "${score_ts// }" ]; then
        echo "=== nothing was generated for test_set='${test_set:-<derived>}' under ${decode_dir}; skipping scoring ==="
    else
        score_args+=(--test_set "${score_ts}")
        echo "+ DECODE_DIR=${decode_dir} bash ${score_script} ${score_args[*]}"
        DECODE_DIR="${decode_dir}" bash "${score_script}" "${score_args[@]}"
    fi
else
    echo "=== EJ_SCORE=${do_score}; skipping scoring. Generated audio under ${out_base}/${task}/ ==="
fi

echo "=== eval_job done: ${engine}/${model}/${task} ==="
