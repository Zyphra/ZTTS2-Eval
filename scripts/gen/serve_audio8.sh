# scripts/gen/serve_audio8.sh -- Audio8-TTS (upstream SGLang-Omni adapter server) lifecycle.
#
# SOURCE this file from the repo root (`. scripts/gen/serve_audio8.sh`); do not execute
# it, so the caller owns AUDIO8_PID and its `trap` can clean up. The launch/retry/stop
# mechanics are utils/lib_serve.sh's.
#
#   start_audio8_server   launch a fresh server, block until ready (3 attempts)
#   stop_audio8_server    terminate the server we started (whole process group)
#   audio8_healthy        0 once ${AUDIO8_BASE_URL}/health answers 200
#
# The server is the Edge0-AI/Audio8_TTS sglang_omni/scripts/run_server.sh, run from
# the pinned sglang-omni checkout with the adapter installed and its own venv (all
# set up by scripts/setup/setup_audio8_env.sh). Pipeline IPC lives in
# a per-run mkdtemp and the NCCL port comes from MASTER_PORT, so several servers on
# one node need only distinct HTTP and MASTER ports.
#
# Environment (exported; the launch reads them):
#   AUDIO8_VENV               venv with sglang-omni   (default ~/.venvs/audio8-sglang)
#   AUDIO8_SRC                Audio8_TTS checkout     (default third_party/Audio8_TTS)
#   SGLANG_OMNI_ROOT          sglang-omni checkout    (default third_party/sglang-omni)
#   AUDIO8_MODEL_DIR          local model dir; empty -> the pinned HF snapshot (cache only)
#   AUDIO8_HOST               bind address            (default 127.0.0.1)
#   AUDIO8_PORT               preferred HTTP port     (default 8010)
#   AUDIO8_GPU                CUDA_VISIBLE_DEVICES    (default 0)
#   AUDIO8_MAX_RUNNING        AUDIO8_TTS_MAX_RUNNING_REQUESTS (default 32)
#   AUDIO8_DISABLE_CUDA_GRAPH AUDIO8_TTS_DISABLE_CUDA_GRAPH   (default 0 = graphs on)
#   AUDIO8_STARTUP_TIMEOUT    seconds to wait per attempt     (default 1800)
#   SERVE_LOG_DIR             server log dir (utils/lib_serve.sh; default /tmp/audio8_<port>.log)
#
# After a successful start: AUDIO8_PORT / AUDIO8_BASE_URL name the port actually
# bound (a retry may move it) and AUDIO8_RUNTIME_REV names both checkouts' git revs.

. utils/lib_serve.sh

_audio8_repo="$(pwd -P)"
AUDIO8_VENV=${AUDIO8_VENV:-$HOME/.venvs/audio8-sglang}
AUDIO8_SRC=${AUDIO8_SRC:-${_audio8_repo}/third_party/Audio8_TTS}
SGLANG_OMNI_ROOT=${SGLANG_OMNI_ROOT:-${_audio8_repo}/third_party/sglang-omni}
AUDIO8_MODEL_DIR=${AUDIO8_MODEL_DIR:-}
AUDIO8_HOST=${AUDIO8_HOST:-127.0.0.1}
AUDIO8_PORT=${AUDIO8_PORT:-8010}
AUDIO8_GPU=${AUDIO8_GPU:-0}
AUDIO8_MAX_RUNNING=${AUDIO8_MAX_RUNNING:-32}
AUDIO8_DISABLE_CUDA_GRAPH=${AUDIO8_DISABLE_CUDA_GRAPH:-0}
AUDIO8_STARTUP_TIMEOUT=${AUDIO8_STARTUP_TIMEOUT:-1800}
AUDIO8_BASE_URL="http://${AUDIO8_HOST}:${AUDIO8_PORT}"
AUDIO8_PID=""
export AUDIO8_VENV AUDIO8_SRC SGLANG_OMNI_ROOT AUDIO8_MODEL_DIR AUDIO8_HOST AUDIO8_PORT AUDIO8_GPU \
       AUDIO8_MAX_RUNNING AUDIO8_DISABLE_CUDA_GRAPH AUDIO8_BASE_URL

_AUDIO8_SETUP_HINT="run 'bash scripts/setup/setup_audio8_env.sh'"

# /health is 200 only once the pipeline is running.
audio8_healthy() {
    serve_healthy "${AUDIO8_BASE_URL}/health"
}

_audio8_health() {  # <port>
    serve_healthy "http://${AUDIO8_HOST}:$1/health"
}

_audio8_launch() {  # <port> <logf>
    local master omni_root
    master=$(serve_free_port 0 1)
    omni_root=$(readlink -f "${SGLANG_OMNI_ROOT}")
    echo "[audio8] GPU ${AUDIO8_GPU}, port $1, MASTER_PORT ${master}, cuda graphs off=${AUDIO8_DISABLE_CUDA_GRAPH}"
    # setsid -> new process-group leader, so serve_stop can kill the pipeline stage
    # children that hold the GPU.
    setsid env CUDA_VISIBLE_DEVICES="${AUDIO8_GPU}" MASTER_PORT="${master}" \
        PYTHON_BIN="${AUDIO8_VENV}/bin/python" SGLANG_OMNI_ROOT="${omni_root}" \
        MODEL="${AUDIO8_MODEL_DIR}" HOST="${AUDIO8_HOST}" PORT="$1" \
        AUDIO8_TTS_MAX_RUNNING_REQUESTS="${AUDIO8_MAX_RUNNING}" \
        AUDIO8_TTS_DISABLE_CUDA_GRAPH="${AUDIO8_DISABLE_CUDA_GRAPH}" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        bash "${AUDIO8_SRC}/sglang_omni/scripts/run_server.sh" > "$2" 2>&1 &
    SERVE_PID=$!
}

start_audio8_server() {
    if [ ! -x "${AUDIO8_VENV}/bin/python" ] || [ ! -f "${AUDIO8_SRC}/sglang_omni/scripts/run_server.sh" ]; then
        echo "[audio8] ERROR: no venv at ${AUDIO8_VENV} or no Audio8_TTS checkout at ${AUDIO8_SRC} -- ${_AUDIO8_SETUP_HINT}" 1>&2
        return 1
    fi
    if [ -z "${AUDIO8_MODEL_DIR}" ]; then
        AUDIO8_MODEL_DIR=$("${AUDIO8_VENV}/bin/python" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Edge0/Audio8-TTS-Realistic-0.6b", revision="1e857568bfd4f02931aca3988ee773a44d2d26d3", local_files_only=True))' 2>/dev/null) || {
            AUDIO8_MODEL_DIR=""
            echo "[audio8] ERROR: model snapshot not cached -- ${_AUDIO8_SETUP_HINT}" 1>&2
            return 1
        }
        export AUDIO8_MODEL_DIR
    fi
    serve_start audio8 "${AUDIO8_PORT}" 1 "${AUDIO8_STARTUP_TIMEOUT}" \
        _audio8_launch _audio8_health audio8 || return 1
    AUDIO8_PORT=${SERVE_PORT}
    AUDIO8_PID=${SERVE_PID}
    AUDIO8_BASE_URL="http://${AUDIO8_HOST}:${AUDIO8_PORT}"
    AUDIO8_RUNTIME_REV="audio8_tts@$(git -C "${AUDIO8_SRC}" rev-parse --short HEAD 2>/dev/null) sglang-omni@$(git -C "${SGLANG_OMNI_ROOT}" rev-parse --short HEAD 2>/dev/null)"
    export AUDIO8_PORT AUDIO8_BASE_URL AUDIO8_RUNTIME_REV
}

stop_audio8_server() {
    [ -n "${AUDIO8_PID}" ] && echo "[audio8] stopping server (pid ${AUDIO8_PID})"
    serve_stop "${AUDIO8_PID}"
    AUDIO8_PID=
}
