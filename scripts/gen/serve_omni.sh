# scripts/gen/serve_omni.sh -- vLLM-Omni TTS server lifecycle (one model at a time).
#
# SOURCE this file (`. scripts/gen/serve_omni.sh`); do not execute it. Sourcing lets the
# caller own the server PID so its `trap` can clean up -- as in scripts/score/serve_asr.sh.
#
#   start_omni_server   launch the server for $OMNI_MODEL, block until /health responds
#   stop_omni_server    terminate a server we started (no-op for a reused one)
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
#   OMNI_HOST OMNI_PORT   bind address                    (default 127.0.0.1:8091)
#   OMNI_GPU              GPU index (CUDA_VISIBLE_DEVICES) (default 0)
#   OMNI_STARTUP_TIMEOUT  seconds to wait for readiness   (default 1800 -- the first
#                                                          run downloads the model)
#   OMNI_VLLM_BIN         the vllm-omni executable         (default: vllm-omni on PATH)
#   OMNI_LOG              server log file                 (default /tmp/vllm_omni_<port>.log)
#
# Set OMNI_HOST/OMNI_PORT before sourcing this file (the health URL is built here).

OMNI_HOST=${OMNI_HOST:-127.0.0.1}
OMNI_PORT=${OMNI_PORT:-8091}
OMNI_GPU=${OMNI_GPU:-0}
OMNI_STARTUP_TIMEOUT=${OMNI_STARTUP_TIMEOUT:-1800}
OMNI_VLLM_BIN=${OMNI_VLLM_BIN:-vllm-omni}
OMNI_BASE_URL="http://${OMNI_HOST}:${OMNI_PORT}/v1"
export OMNI_BASE_URL

# vLLM defaults to FlashInfer for sampling; on this host FlashInfer's JIT CUDA
# build (via ninja) fails, so force vLLM's native sampler. Export
# VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable it where the JIT works.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

_OMNI_HEALTH_URL="http://${OMNI_HOST}:${OMNI_PORT}/health"
_OMNI_MODELS_URL="http://${OMNI_HOST}:${OMNI_PORT}/v1/models"
OMNI_PID=""

_omni_healthy() {
    curl -s -o /dev/null -m 2 "${_OMNI_HEALTH_URL}"
}

# Is a server already on the port serving exactly the model we want?
_omni_model_matches() {
    curl -s -m 2 "${_OMNI_MODELS_URL}" 2>/dev/null | grep -qF -- "${OMNI_MODEL}"
}

start_omni_server() {
    [ -z "${OMNI_MODEL}" ] && { echo "[omni] ERROR: OMNI_MODEL is not set" 1>&2; return 1; }
    local logf="${OMNI_LOG:-/tmp/vllm_omni_${OMNI_PORT}.log}"

    # Reuse a server already serving this exact model; if a different model holds the
    # port, refuse rather than collide (cycling models would otherwise mismatch).
    if _omni_healthy; then
        if _omni_model_matches; then
            echo "[omni] reusing healthy server for ${OMNI_MODEL} at ${OMNI_BASE_URL}"
            OMNI_PID=""
            return 0
        fi
        echo "[omni] ERROR: another server already holds ${OMNI_HOST}:${OMNI_PORT};" \
             "stop it or set OMNI_PORT to a free port." 1>&2
        return 1
    fi

    echo "[omni] starting: ${OMNI_VLLM_BIN} serve ${OMNI_MODEL} --omni ${OMNI_SERVE_ARGS}" \
         "on ${OMNI_HOST}:${OMNI_PORT} (GPU ${OMNI_GPU}, log ${logf})"
    CUDA_VISIBLE_DEVICES="${OMNI_GPU}" "${OMNI_VLLM_BIN}" serve "${OMNI_MODEL}" \
        --omni \
        ${OMNI_SERVE_ARGS} \
        --host "${OMNI_HOST}" --port "${OMNI_PORT}" \
        > "${logf}" 2>&1 &
    OMNI_PID=$!

    local waited=0
    until _omni_healthy; do
        if ! kill -0 "${OMNI_PID}" 2>/dev/null; then
            echo "[omni] ERROR: vllm-omni exited during startup. Last log lines:" 1>&2
            tail -n 40 "${logf}" 1>&2
            OMNI_PID=""
            return 1
        fi
        if [ "${waited}" -ge "${OMNI_STARTUP_TIMEOUT}" ]; then
            echo "[omni] ERROR: startup timed out after ${OMNI_STARTUP_TIMEOUT}s." \
                 "Last log lines:" 1>&2
            tail -n 40 "${logf}" 1>&2
            stop_omni_server
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[omni] server ready after ~${waited}s (pid ${OMNI_PID})"
}

stop_omni_server() {
    [ -z "${OMNI_PID}" ] && return 0
    echo "[omni] stopping vllm-omni (pid ${OMNI_PID})"
    kill "${OMNI_PID}" 2>/dev/null || true
    local i=0
    while kill -0 "${OMNI_PID}" 2>/dev/null; do
        i=$((i + 1))
        [ "${i}" -ge 30 ] && break
        sleep 1
    done
    kill -9 "${OMNI_PID}" 2>/dev/null || true
    OMNI_PID=""
    # vLLM-Omni's multi-stage serving can leave worker processes; wait for the port
    # to actually free up before the caller starts the next model.
    local w=0
    while _omni_healthy && [ "${w}" -lt 30 ]; do
        sleep 1
        w=$((w + 1))
    done
}
