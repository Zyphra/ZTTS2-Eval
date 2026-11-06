# scripts/score/serve_asr.sh -- vLLM Qwen3-ASR (+ Whisper) server lifecycle.
#
# SOURCE this file from the repo root (`. scripts/score/serve_asr.sh`); do not execute
# it. Sourcing lets the caller own the server PIDs so its `trap` can clean up on
# exit/error/interrupt. The launch/retry/stop mechanics are utils/lib_serve.sh's.
#
#   start_asr_server   launch a fresh server, block until /health responds; a preset
#                      ASR_BASE_URL names an external server and nothing is launched
#   stop_asr_server    terminate the server we started (no-op for an external one)
#
# Configure via environment:
#   ASR_MODEL             HuggingFace model id           (default Qwen/Qwen3-ASR-1.7B)
#   ASR_HOST ASR_PORT     bind address                   (default 127.0.0.1:8000)
#   ASR_GPU               GPU index for the server       (default 0)
#   ASR_GPU_MEM_UTIL      vLLM --gpu-memory-utilization   (default 0.35, the published
#                         co-location value: Qwen + Whisper + speaker-sim/MSR-UTMOS share
#                         one GPU; vLLM default 0.9 would starve the scorers)
#   ASR_STARTUP_TIMEOUT   seconds to wait for readiness   (default 600)
#   SERVE_LOG_DIR         server log dir (utils/lib_serve.sh; default /tmp/asr_<port>.log)
#   ASR_EXTRA_ARGS        extra flags appended to `vllm serve` (default empty),
#                         e.g. tuning flags for a different ASR. Note: swapping in
#                         openai/whisper-large-v3 needs only ASR_MODEL -- vLLM 0.21
#                         auto-detects it and exposes /v1/audio/transcriptions with
#                         no extra flags (an older `--task transcription` is gone).

. utils/lib_serve.sh

ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}
ASR_HOST=${ASR_HOST:-127.0.0.1}
ASR_PORT=${ASR_PORT:-8000}
# GPU placement: if ASR_GPU is unset, auto-pick the first GPU with at least
# ASR_GPU_MIN_FREE_MIB free (default 50000 MiB). Without this, the default GPU 0 may be
# reclaimed by a co-resident TTS server between runs and vLLM fails to start with an OOM.
if [ -z "${ASR_GPU}" ]; then
    # 50000 MiB leaves vLLM headroom to actually allocate the KV cache it asks for;
    # 20 GB passed nominal startup but OOMed once the KV cache grew.
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
ASR_GPU_MEM_UTIL=${ASR_GPU_MEM_UTIL:-0.35}
# Per-language ASR routing is the benchmark default (scripts/score/run_wer.py enforces it).
ASR_ROUTING=${ASR_ROUTING:-1}
# Pinned model revisions (utils/benchmark.py). Only meaningful for the pinned model id;
# a swapped-in model is served unpinned and stamped "unpinned" by run_wer.py.
_pin() {  # <backend> <model> -> pinned revision when <model> is the pinned id, else ""
    python3 - "$1" "$2" <<'PYEOF' 2>/dev/null
import sys
from utils.benchmark import ASR_MODELS
mid, rev = ASR_MODELS[sys.argv[1]]
print(rev if sys.argv[2] == mid else "")
PYEOF
}
ASR_REVISION=${ASR_REVISION-$(_pin qwen "${ASR_MODEL}")}
ASR_STARTUP_TIMEOUT=${ASR_STARTUP_TIMEOUT:-600}
# A preset ASR_BASE_URL is an external server: start_asr_server launches nothing.
ASR_EXTERNAL_URL=${ASR_BASE_URL:-}
ASR_BASE_URL=${ASR_EXTERNAL_URL:-http://${ASR_HOST}:${ASR_PORT}/v1}
export ASR_BASE_URL ASR_MODEL ASR_ROUTING ASR_REVISION

# vLLM defaults to FlashInfer for sampling; on this host FlashInfer's JIT CUDA
# build (via ninja) fails, so force vLLM's native sampler. Export
# VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable it where the JIT works.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

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

_asr_health() {  # <port>
    serve_healthy "http://${ASR_HOST}:$1/health"
}

_asr_launch() {  # <port> <logf>
    echo "[asr] ${ASR_VLLM_BIN} serve ${ASR_MODEL} on ${ASR_HOST}:$1" \
         "(GPU ${ASR_GPU}, mem-util ${ASR_GPU_MEM_UTIL}, args '${ASR_EXTRA_ARGS}')"
    [ -n "${ASR_TMPDIR:-}" ] && rm -rf "${ASR_TMPDIR}"   # a failed earlier attempt's
    ASR_TMPDIR=$(_server_tmpdir "asr-$1") || return 1
    # Both Qwen3-ASR and whisper-large-v3 expose the OpenAI-compatible
    # /v1/audio/transcriptions endpoint with no extra flags (vLLM auto-detects the
    # runner). ASR_EXTRA_ARGS is intentionally unquoted so multiple flags word-split.
    TMPDIR="${ASR_TMPDIR}" CUDA_VISIBLE_DEVICES="${ASR_GPU}" setsid "${ASR_VLLM_BIN}" serve "${ASR_MODEL}" \
        --host "${ASR_HOST}" --port "$1" \
        --gpu-memory-utilization "${ASR_GPU_MEM_UTIL}" \
        ${ASR_REVISION:+--revision ${ASR_REVISION} --tokenizer-revision ${ASR_REVISION}} \
        ${ASR_EXTRA_ARGS} \
        > "$2" 2>&1 &
    SERVE_PID=$!
}

start_asr_server() {
    if [ -n "${ASR_EXTERNAL_URL}" ]; then
        echo "[asr] using external server ${ASR_EXTERNAL_URL}"
        return 0
    fi
    serve_start asr "${ASR_PORT}" 1 "${ASR_STARTUP_TIMEOUT}" _asr_launch _asr_health asr || return 1
    ASR_PORT=${SERVE_PORT}
    VLLM_PID=${SERVE_PID}
    ASR_BASE_URL="http://${ASR_HOST}:${ASR_PORT}/v1"
    export ASR_PORT ASR_BASE_URL
}

stop_asr_server() {
    [ -n "${VLLM_PID}" ] && echo "[asr] stopping vLLM (pid ${VLLM_PID})"
    serve_stop "${VLLM_PID}"
    VLLM_PID=
    [ -n "${ASR_TMPDIR:-}" ] && rm -rf "${ASR_TMPDIR}"
    ASR_TMPDIR=""
}

# --- Optional second ASR: Whisper large-v3 for per-language routing --------------
# run_wer.py routes the non-CJK "tail" langs to Whisper when WHISPER_BASE_URL is set
# (see docs/asr_whisper_vs_qwen.md). These functions bring up that second server on
# its own GPU/port/PID so it never clobbers the Qwen server above.
#   WHISPER_ASR_MODEL   model to serve            (default openai/whisper-large-v3)
#   WHISPER_PORT        preferred port            (default 8001)
#   WHISPER_GPU         GPU index                 (default: auto-pick != ASR_GPU with at least
#                       ASR_GPU_MIN_FREE_MIB free; when no other GPU has room, co-locate
#                       on ASR_GPU)
#   WHISPER_GPU_MEM_UTIL vLLM --gpu-memory-utilization (default 0.35)
# A preset WHISPER_BASE_URL is an external server (maybe_start_whisper launches nothing).
WHISPER_ASR_MODEL=${WHISPER_ASR_MODEL:-openai/whisper-large-v3}
WHISPER_PORT=${WHISPER_PORT:-8001}
WHISPER_GPU=${WHISPER_GPU:-}
WHISPER_GPU_MEM_UTIL=${WHISPER_GPU_MEM_UTIL:-0.35}
WHISPER_REVISION=${WHISPER_REVISION-$(_pin whisper "${WHISPER_ASR_MODEL}")}
export WHISPER_REVISION
WHISPER_VLLM_PID=""

_whisper_health() {  # <port>
    serve_healthy "http://${ASR_HOST}:$1/health"
}

_whisper_launch() {  # <port> <logf>
    echo "[whisper] ${WHISPER_ASR_MODEL} on ${ASR_HOST}:$1 (GPU ${_WHISPER_GPU})"
    [ -n "${WHISPER_TMPDIR:-}" ] && rm -rf "${WHISPER_TMPDIR}"   # a failed earlier attempt's
    WHISPER_TMPDIR=$(_server_tmpdir "whisper-$1") || return 1
    TMPDIR="${WHISPER_TMPDIR}" CUDA_VISIBLE_DEVICES="${_WHISPER_GPU}" setsid "${ASR_VLLM_BIN}" serve "${WHISPER_ASR_MODEL}" \
        --host "${ASR_HOST}" --port "$1" \
        --gpu-memory-utilization "${WHISPER_GPU_MEM_UTIL}" \
        ${WHISPER_REVISION:+--revision ${WHISPER_REVISION} --tokenizer-revision ${WHISPER_REVISION}} \
        ${ASR_EXTRA_ARGS} \
        > "$2" 2>&1 &
    SERVE_PID=$!
}

start_whisper_server() {
    # Pick a GPU distinct from the Qwen server's, with headroom for the KV cache.
    _WHISPER_GPU="${WHISPER_GPU}"
    if [ -z "${_WHISPER_GPU}" ]; then
        _WHISPER_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
              | awk -F', ' -v skip="${ASR_GPU:-99}" -v n="${ASR_GPU_MIN_FREE_MIB:-50000}" \
                    '$1!=skip && $2>=n{print $1; exit}')
        if [ -z "${_WHISPER_GPU}" ]; then
            _WHISPER_GPU="${ASR_GPU}"
            echo "[whisper] co-locating on GPU ${ASR_GPU}"
        fi
    fi
    serve_start whisper "${WHISPER_PORT}" 1 "${ASR_STARTUP_TIMEOUT}" \
        _whisper_launch _whisper_health whisper || return 1
    WHISPER_PORT=${SERVE_PORT}
    WHISPER_VLLM_PID=${SERVE_PID}
    export WHISPER_BASE_URL="http://${ASR_HOST}:${WHISPER_PORT}/v1" WHISPER_MODEL="${WHISPER_ASR_MODEL}"
}

stop_whisper_server() {
    [ -n "${WHISPER_VLLM_PID}" ] && echo "[whisper] stopping vLLM (pid ${WHISPER_VLLM_PID})"
    serve_stop "${WHISPER_VLLM_PID}"
    WHISPER_VLLM_PID=
    [ -n "${WHISPER_TMPDIR:-}" ] && rm -rf "${WHISPER_TMPDIR}"
    WHISPER_TMPDIR=""
}

# Scoring-driver entry point. Routing is the benchmark default: the Whisper server is
# started unless ASR_ROUTING=0 (diagnostic all-Qwen mode, not benchmark-comparable).
# A failed start is an error -- callers do `maybe_start_whisper || exit 1`.
maybe_start_whisper() {
    if [ -n "${WHISPER_BASE_URL:-}" ]; then
        echo "[whisper] routing on: using external server ${WHISPER_BASE_URL}"
        return 0
    fi
    if [ "${ASR_ROUTING:-1}" = "0" ]; then
        echo "[whisper] ASR_ROUTING=0: all languages on Qwen3-ASR; results are not benchmark-comparable and build_results.py will refuse them"
        return 0
    fi
    start_whisper_server || { echo "[whisper] ERROR: Whisper server failed to start" >&2; return 1; }
}
