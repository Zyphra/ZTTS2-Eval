# scripts/gen/serve_omni.sh -- vLLM-Omni TTS server lifecycle (one model at a time).
#
# SOURCE this file from the repo root (`. scripts/gen/serve_omni.sh`); do not execute
# it. Sourcing lets the caller own the server PID so its `trap` can clean up. The
# launch/retry/stop mechanics are utils/lib_serve.sh's.
#
#   start_omni_server   launch a fresh server for $OMNI_MODEL, block until /health responds
#   stop_omni_server    terminate the server we started (whole process group)
#
# The server is launched with the vLLM-Omni CLI:
#   <vllm-omni> serve <model> --omni <OMNI_SERVE_ARGS> --host <h> --port <p>
# The `--omni` flag is what switches the `vllm-omni` binary out of plain-vLLM mode.
# Everything model-specific -- notably `--deploy-config <yaml>`, which selects the
# model's multi-stage pipeline config -- is passed via OMNI_SERVE_ARGS, filled in
# per model by scripts/gen/run_omni.sh. The deploy YAML owns gpu-memory / eager / etc.,
# so this script intentionally does not pass those.
#
# Configure via environment:
#   OMNI_MODEL            HF model id to serve            (required)
#   OMNI_SERVE_ARGS       per-model `serve` flags         (e.g. --deploy-config ...)
#   OMNI_HOST OMNI_PORT   bind address / preferred port   (default 127.0.0.1:8091)
#   OMNI_GPU              GPU index (CUDA_VISIBLE_DEVICES) (default 0)
#   OMNI_STARTUP_TIMEOUT  seconds to wait for readiness   (default 1800 -- the first
#                                                          run downloads the model)
#   OMNI_VLLM_BIN         the vllm-omni executable         (default: vllm-omni on PATH)
#   OMNI_LOG_ROLE         server log name                 (default tts; the log is
#                         ${SERVE_LOG_DIR}/<role>.log, else /tmp/<role>_<port>.log)
#
# After a successful start OMNI_PORT / OMNI_BASE_URL name the port actually bound.

. utils/lib_serve.sh

OMNI_HOST=${OMNI_HOST:-127.0.0.1}
OMNI_PORT=${OMNI_PORT:-8091}
OMNI_GPU=${OMNI_GPU:-0}
OMNI_STARTUP_TIMEOUT=${OMNI_STARTUP_TIMEOUT:-1800}
OMNI_VLLM_BIN=${OMNI_VLLM_BIN:-vllm-omni}
OMNI_BASE_URL="http://${OMNI_HOST}:${OMNI_PORT}/v1"
OMNI_PID=""
export OMNI_BASE_URL

# vLLM defaults to FlashInfer for sampling; on this host FlashInfer's JIT CUDA
# build (via ninja) fails, so force vLLM's native sampler. Export
# VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable it where the JIT works.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

_omni_health() {  # <port>
    serve_healthy "http://${OMNI_HOST}:$1/health"
}

_omni_launch() {  # <port> <logf>
    echo "[omni] ${OMNI_VLLM_BIN} serve ${OMNI_MODEL} --omni ${OMNI_SERVE_ARGS}" \
         "on ${OMNI_HOST}:$1 (GPU ${OMNI_GPU})"
    # setsid: vLLM-Omni's multi-stage serving forks workers; serve_stop kills the group.
    CUDA_VISIBLE_DEVICES="${OMNI_GPU}" setsid "${OMNI_VLLM_BIN}" serve "${OMNI_MODEL}" \
        --omni \
        ${OMNI_SERVE_ARGS} \
        --host "${OMNI_HOST}" --port "$1" \
        > "$2" 2>&1 &
    SERVE_PID=$!
}

start_omni_server() {
    [ -z "${OMNI_MODEL}" ] && { echo "[omni] ERROR: OMNI_MODEL is not set" 1>&2; return 1; }
    serve_start omni "${OMNI_PORT}" 1 "${OMNI_STARTUP_TIMEOUT}" \
        _omni_launch _omni_health "${OMNI_LOG_ROLE:-tts}" || return 1
    OMNI_PORT=${SERVE_PORT}
    OMNI_PID=${SERVE_PID}
    OMNI_BASE_URL="http://${OMNI_HOST}:${OMNI_PORT}/v1"
    export OMNI_PORT OMNI_BASE_URL
}

stop_omni_server() {
    [ -n "${OMNI_PID}" ] && echo "[omni] stopping vllm-omni (pid ${OMNI_PID})"
    serve_stop "${OMNI_PID}"
    OMNI_PID=
}
