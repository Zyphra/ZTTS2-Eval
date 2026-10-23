# scripts/serve_asr.sh -- vLLM Qwen3-ASR server lifecycle.
#
# SOURCE this file (`. scripts/serve_asr.sh`); do not execute it. Sourcing lets the
# caller own the server PID so its `trap` can clean up on exit/error/interrupt.
#
#   start_asr_server   launch (or reuse) the server, block until /health responds
#   stop_asr_server    terminate a server we started (no-op for a reused one)
#
# Configure via environment:
#   ASR_MODEL             HuggingFace model id           (default Qwen/Qwen3-ASR-1.7B)
#   ASR_HOST ASR_PORT     bind address                   (default 127.0.0.1:8000)
#   ASR_GPU               GPU index for the server       (default 0)
#   ASR_GPU_MEM_UTIL      vLLM --gpu-memory-utilization   (default 0.5; vLLM default
#                         0.9 would starve the speaker-sim / MSR-UTMOS steps on the
#                         same GPU -- raise it only if the server has a GPU to itself)
#   ASR_STARTUP_TIMEOUT   seconds to wait for readiness   (default 600)
#   ASR_LOG               server log file                (default /tmp/vllm_asr_<port>.log)
#   ASR_EXTRA_ARGS        extra flags appended to `vllm serve` (default empty),
#                         e.g. tuning flags for a different ASR. Note: swapping in
#                         openai/whisper-large-v3 needs only ASR_MODEL -- vLLM 0.21
#                         auto-detects it and exposes /v1/audio/transcriptions with
#                         no extra flags (an older `--task transcription` is gone).

ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}
ASR_HOST=${ASR_HOST:-127.0.0.1}
ASR_PORT=${ASR_PORT:-8000}
# GPU placement: if ASR_GPU is unset, auto-pick the first GPU with at least
# ASR_GPU_MIN_FREE_MIB free (default 20 GB -- weights ~3.4 GB + KV cache headroom
# at gpu-mem-util 0.5). Without this, the default GPU 0 may be reclaimed by a
# co-resident TTS server between runs and vLLM fails to start with an OOM.
if [ -z "${ASR_GPU}" ]; then
    # Require more than the gpu-memory-utilization target (0.5*81GB=40.5GB) so
    # vLLM has headroom to actually allocate the KV cache it asks for. 20 GB
    # passed nominal startup but OOMed once the KV cache grew.
    _free=${ASR_GPU_MIN_FREE_MIB:-50000}
    ASR_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
              | awk -F', ' -v n="${_free}" '$2>=n{print $1; exit}')
    if [ -z "${ASR_GPU}" ]; then
        echo "[asr] WARN: no GPU with >= ${_free} MiB free; falling back to GPU 0" 1>&2
        ASR_GPU=0
    else
        echo "[asr] auto-selected GPU ${ASR_GPU} (>= ${_free} MiB free)"
    fi
fi
ASR_GPU_MEM_UTIL=${ASR_GPU_MEM_UTIL:-0.5}
ASR_STARTUP_TIMEOUT=${ASR_STARTUP_TIMEOUT:-600}
ASR_BASE_URL="http://${ASR_HOST}:${ASR_PORT}/v1"
export ASR_BASE_URL ASR_MODEL

# vLLM defaults to FlashInfer for sampling; on this host FlashInfer's JIT CUDA
# build (via ninja) fails, so force vLLM's native sampler. Export
# VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable it where the JIT works.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

_ASR_HEALTH_URL="http://${ASR_HOST}:${ASR_PORT}/health"
VLLM_PID=""

# Prepend the project venv to PATH if it's not already there. Backgrounded
# callers (nohup'd drivers) inherit only the user's base PATH, so bare `vllm`,
# `python`, and `python3` all miss the project venv -- and scoring then dies
# on `tqdm` etc. This is a no-op for callers that already activated .venv.
if [ -d .venv/bin ] && [[ ":${PATH}:" != *":$(pwd -P)/.venv/bin:"* ]]; then
    export PATH="$(pwd -P)/.venv/bin:${PATH}"
fi

# Resolve the vllm CLI. After the PATH fix above this normally just finds the
# venv copy; keep an explicit fallback in case .venv is absent.
if command -v vllm >/dev/null 2>&1; then
    ASR_VLLM_BIN=${ASR_VLLM_BIN:-vllm}
elif [ -x .venv/bin/vllm ]; then
    ASR_VLLM_BIN=.venv/bin/vllm
else
    ASR_VLLM_BIN=${ASR_VLLM_BIN:-vllm}    # let it fail loudly in start_asr_server
fi

_asr_healthy() {
    curl -s -o /dev/null -m 2 "${_ASR_HEALTH_URL}"
}

# The servers spool any upload larger than 1 MB -- a wav over ~12 s at 44.1 kHz --
# to a temp file before parsing it. On a node whose /tmp shares a full root disk
# those uploads fail with HTTP 400 "There was an error parsing the body" while
# shorter clips still transcribe, so a third of a language fails silently
# (2147 of 9322 utterances in one observed run). Give each server its own
# temp dir on node-local tmpfs instead of trusting the node's /tmp.
_server_tmpdir() {  # <name> -> creates and prints a private temp dir
    local base="${ASR_TMPDIR_BASE:-/dev/shm}"
    { [ -d "${base}" ] && [ -w "${base}" ]; } || base="${TMPDIR:-/tmp}"
    local d="${base}/${USER:-u}-${1}-${SLURM_JOB_ID:-$$}"
    mkdir -p "${d}" && echo "${d}"
}

start_asr_server() {
    local logf="${ASR_LOG:-/tmp/vllm_asr_${ASR_PORT}.log}"

    # Reuse a server someone already started; leave VLLM_PID empty so we never
    # kill a process we do not own.
    if _asr_healthy; then
        echo "[asr] reusing healthy server at ${ASR_BASE_URL}"
        VLLM_PID=""
        return 0
    fi

    echo "[asr] starting: ${ASR_VLLM_BIN} serve ${ASR_MODEL} on ${ASR_HOST}:${ASR_PORT}" \
         "(GPU ${ASR_GPU}, mem-util ${ASR_GPU_MEM_UTIL}, args '${ASR_EXTRA_ARGS}', log ${logf})"
    # Both Qwen3-ASR and whisper-large-v3 expose the OpenAI-compatible
    # /v1/audio/transcriptions endpoint with no extra flags (vLLM auto-detects the
    # runner). ASR_EXTRA_ARGS is intentionally unquoted so multiple flags word-split.
    ASR_TMPDIR=$(_server_tmpdir "asr-${ASR_PORT}") || return 1
    TMPDIR="${ASR_TMPDIR}" CUDA_VISIBLE_DEVICES="${ASR_GPU}" "${ASR_VLLM_BIN}" serve "${ASR_MODEL}" \
        --host "${ASR_HOST}" --port "${ASR_PORT}" \
        --gpu-memory-utilization "${ASR_GPU_MEM_UTIL}" \
        ${ASR_EXTRA_ARGS} \
        > "${logf}" 2>&1 &
    VLLM_PID=$!

    local waited=0
    until _asr_healthy; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[asr] ERROR: vLLM exited during startup. Last log lines:"
            tail -n 40 "${logf}"
            VLLM_PID=""
            return 1
        fi
        if [ "${waited}" -ge "${ASR_STARTUP_TIMEOUT}" ]; then
            echo "[asr] ERROR: startup timed out after ${ASR_STARTUP_TIMEOUT}s. Last log lines:"
            tail -n 40 "${logf}"
            stop_asr_server
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[asr] server ready after ~${waited}s (pid ${VLLM_PID})"
}

stop_asr_server() {
    [ -z "${VLLM_PID}" ] && return 0
    echo "[asr] stopping vLLM (pid ${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    local i=0
    while kill -0 "${VLLM_PID}" 2>/dev/null; do
        i=$((i + 1))
        [ "${i}" -ge 20 ] && break
        sleep 1
    done
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
    [ -n "${ASR_TMPDIR:-}" ] && rm -rf "${ASR_TMPDIR}"
    ASR_TMPDIR=""
}

# --- Optional second ASR: Whisper large-v3 for per-language routing --------------
# run_wer.py routes the non-CJK "tail" langs to Whisper when WHISPER_BASE_URL is set
# (see docs/asr_whisper_vs_qwen.md). These functions bring up that second server on
# its own GPU/port/PID so it never clobbers the Qwen server above.
#   WHISPER_ASR_MODEL   model to serve            (default openai/whisper-large-v3)
#   WHISPER_PORT        bind port                 (default 8001)
#   WHISPER_GPU         GPU index                 (default: auto-pick, != ASR_GPU)
#   WHISPER_GPU_MEM_UTIL vLLM --gpu-memory-utilization (default 0.5)
#   WHISPER_LOG         server log file           (default /tmp/vllm_whisper_<port>.log)
WHISPER_ASR_MODEL=${WHISPER_ASR_MODEL:-openai/whisper-large-v3}
WHISPER_PORT=${WHISPER_PORT:-8001}
WHISPER_GPU=${WHISPER_GPU:-}
WHISPER_GPU_MEM_UTIL=${WHISPER_GPU_MEM_UTIL:-0.5}
WHISPER_VLLM_PID=""

start_whisper_server() {
    local logf="${WHISPER_LOG:-/tmp/vllm_whisper_${WHISPER_PORT}.log}"
    local url="http://${ASR_HOST}:${WHISPER_PORT}"

    if curl -s -o /dev/null -m 2 "${url}/health"; then
        echo "[whisper] reusing healthy server at ${url}/v1"
        WHISPER_VLLM_PID=""
        export WHISPER_BASE_URL="${url}/v1" WHISPER_MODEL="${WHISPER_ASR_MODEL}"
        return 0
    fi

    # Pick a GPU distinct from the Qwen server's, with headroom for the KV cache.
    local gpu="${WHISPER_GPU}"
    if [ -z "${gpu}" ]; then
        gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
              | awk -F', ' -v skip="${ASR_GPU:-99}" '$1!=skip && $2>=50000{print $1; exit}')
        [ -z "${gpu}" ] && { echo "[whisper] ERROR: no free GPU (!= ASR_GPU) for Whisper" >&2; return 1; }
    fi

    echo "[whisper] starting ${WHISPER_ASR_MODEL} on ${ASR_HOST}:${WHISPER_PORT} (GPU ${gpu}, log ${logf})"
    WHISPER_TMPDIR=$(_server_tmpdir "whisper-${WHISPER_PORT}") || return 1
    TMPDIR="${WHISPER_TMPDIR}" CUDA_VISIBLE_DEVICES="${gpu}" "${ASR_VLLM_BIN}" serve "${WHISPER_ASR_MODEL}" \
        --host "${ASR_HOST}" --port "${WHISPER_PORT}" \
        --gpu-memory-utilization "${WHISPER_GPU_MEM_UTIL}" \
        ${ASR_EXTRA_ARGS} \
        > "${logf}" 2>&1 &
    WHISPER_VLLM_PID=$!

    local waited=0
    until curl -s -o /dev/null -m 2 "${url}/health"; do
        if ! kill -0 "${WHISPER_VLLM_PID}" 2>/dev/null; then
            echo "[whisper] ERROR: vLLM exited during startup. Last log lines:"
            tail -n 40 "${logf}"; WHISPER_VLLM_PID=""; return 1
        fi
        if [ "${waited}" -ge "${ASR_STARTUP_TIMEOUT}" ]; then
            echo "[whisper] ERROR: startup timed out after ${ASR_STARTUP_TIMEOUT}s."
            tail -n 40 "${logf}"; stop_whisper_server; return 1
        fi
        sleep 5; waited=$((waited + 5))
    done
    echo "[whisper] server ready after ~${waited}s (pid ${WHISPER_VLLM_PID})"
    export WHISPER_BASE_URL="${url}/v1" WHISPER_MODEL="${WHISPER_ASR_MODEL}"
}

stop_whisper_server() {
    [ -z "${WHISPER_VLLM_PID}" ] && return 0
    echo "[whisper] stopping vLLM (pid ${WHISPER_VLLM_PID})"
    kill "${WHISPER_VLLM_PID}" 2>/dev/null || true
    local i=0
    while kill -0 "${WHISPER_VLLM_PID}" 2>/dev/null; do
        i=$((i + 1))
        [ "${i}" -ge 20 ] && break
        sleep 1
    done
    kill -9 "${WHISPER_VLLM_PID}" 2>/dev/null || true
    WHISPER_VLLM_PID=""
    [ -n "${WHISPER_TMPDIR:-}" ] && rm -rf "${WHISPER_TMPDIR}"
    WHISPER_TMPDIR=""
}

# Convenience for the scoring drivers: enable per-lang routing only when asked --
# ASR_ROUTING=1, or WHISPER_GPU set, or an external WHISPER_BASE_URL already given.
# Off by default, so GT and existing single-server runs are byte-for-byte unchanged.
maybe_start_whisper() {
    if [ -n "${WHISPER_BASE_URL:-}" ]; then
        echo "[whisper] routing on: using external server ${WHISPER_BASE_URL}"
        return 0
    fi
    if [ "${ASR_ROUTING:-0}" != "1" ] && [ -z "${WHISPER_GPU:-}" ]; then
        return 0
    fi
    start_whisper_server || echo "[whisper] WARN: start failed; continuing all-Qwen" >&2
}
