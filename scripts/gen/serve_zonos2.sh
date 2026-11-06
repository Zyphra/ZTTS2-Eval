# scripts/gen/serve_zonos2.sh -- Zonos2 (Mini-SGLang) TTS server lifecycle.
#
# SOURCE this file from the repo root (`. scripts/gen/serve_zonos2.sh`); do not execute
# it. Sourcing lets the caller own the server PID so its `trap` can clean up. The
# launch/retry/stop mechanics are utils/lib_serve.sh's.
#
#   start_zonos2_server   launch a fresh server for $ZONOS2_MODEL, block until ready
#   stop_zonos2_server    terminate the server we started (whole process group)
#
# The server is the upstream Zonos2 Mini-SGLang inference server, launched exactly
# as its README prescribes:
#   <py> -m zonos2 --model-path <model> --tts-default-voices-dir <dir> --host <h> --port <p>
# It is NOT a pip package in this repo's .venv -- it lives in a separate checkout
# (point ZONOS2_HOME at it) with its own uv venv. We call that venv's
# python directly so no `uv run` / activation is needed. GPU selection is via
# CUDA_VISIBLE_DEVICES (the server calls torch.cuda.set_device(rank); rank 0 is the
# single visible GPU), the same contract serve_omni.sh uses.
#
# Configure via environment (all but ZONOS2_HOME have defaults):
#   ZONOS2_MODEL          model path/id to serve      (default Zyphra/ZONOS2, HF-cached)
#   ZONOS2_HOME           the Zonos2 checkout root     (required unless ZONOS2_PY is set)
#   ZONOS2_PY             python that has zonos2        (default $ZONOS2_HOME/.venv/bin/python)
#   ZONOS2_VOICES_DIR     --tts-default-voices-dir      (default $ZONOS2_HOME/default_voices)
#   ZONOS2_HOST ZONOS2_PORT  bind address / preferred port (default 127.0.0.1:1919)
#   ZONOS2_GPU            GPU index (CUDA_VISIBLE_DEVICES) (default 0)
#   ZONOS2_SERVE_ARGS     extra `-m zonos2` flags        (e.g. --dtype, --attn, --memory-ratio)
#   ZONOS2_STARTUP_TIMEOUT  seconds to wait for readiness (default 1800 -- loads a 15G model)
#   SERVE_LOG_DIR         server log dir (utils/lib_serve.sh; default /tmp/zonos2_<port>.log)
#
# The Mini-SGLang scheduler also binds an IPC socket on port+1, so the server takes a
# window of two free ports. After a successful start ZONOS2_PORT / ZONOS2_BASE_URL name
# the port actually bound (a retry may move it).

. utils/lib_serve.sh

ZONOS2_MODEL=${ZONOS2_MODEL:-Zyphra/ZONOS2}
ZONOS2_HOME=${ZONOS2_HOME:-}
ZONOS2_PY=${ZONOS2_PY:-${ZONOS2_HOME}/.venv/bin/python}
ZONOS2_VOICES_DIR=${ZONOS2_VOICES_DIR:-${ZONOS2_HOME}/default_voices}
ZONOS2_HOST=${ZONOS2_HOST:-127.0.0.1}
ZONOS2_PORT=${ZONOS2_PORT:-1919}
ZONOS2_GPU=${ZONOS2_GPU:-0}
ZONOS2_STARTUP_TIMEOUT=${ZONOS2_STARTUP_TIMEOUT:-1800}
ZONOS2_BASE_URL="http://${ZONOS2_HOST}:${ZONOS2_PORT}"
ZONOS2_PID=""
export ZONOS2_BASE_URL

# /tts/capabilities returns 200 only once the model is fully loaded and the request
# state is live -- it is the cheapest readiness probe (the generation client already
# uses it to gate speaker/accurate-mode support).
_zonos2_health() {  # <port>
    serve_healthy "http://${ZONOS2_HOST}:$1/tts/capabilities"
}

_zonos2_launch() {  # <port> <logf>
    local voices_opt=""
    [ -n "${ZONOS2_VOICES_DIR}" ] && [ -d "${ZONOS2_VOICES_DIR}" ] && \
        voices_opt="--tts-default-voices-dir ${ZONOS2_VOICES_DIR}"
    echo "[zonos2] ${ZONOS2_PY} -m zonos2 --model-path ${ZONOS2_MODEL} ${voices_opt} ${ZONOS2_SERVE_ARGS}" \
         "on ${ZONOS2_HOST}:$1 (GPU ${ZONOS2_GPU})"
    # setsid makes the server a new process-group leader (pgid == pid) so serve_stop
    # can kill the WHOLE group -- Mini-SGLang forks scheduler/worker children that hold
    # the GPU and the port+1 IPC socket. Run from the checkout root so relative asset
    # paths resolve; the package is importable from that venv regardless.
    setsid bash -c '
        cd "'"${ZONOS2_HOME}"'" || exit 127
        export CUDA_VISIBLE_DEVICES="'"${ZONOS2_GPU}"'"
        exec "'"${ZONOS2_PY}"'" -m zonos2 \
            --model-path "'"${ZONOS2_MODEL}"'" \
            '"${voices_opt}"' \
            --host "'"${ZONOS2_HOST}"'" --port "'"$1"'" \
            '"${ZONOS2_SERVE_ARGS}"'
    ' > "$2" 2>&1 &
    SERVE_PID=$!
}

start_zonos2_server() {
    [ -z "${ZONOS2_MODEL}" ] && { echo "[zonos2] ERROR: ZONOS2_MODEL is not set" 1>&2; return 1; }
    if [ ! -x "${ZONOS2_PY}" ]; then
        echo "[zonos2] ERROR: python '${ZONOS2_PY}' not found/executable." \
             "Set ZONOS2_PY or ZONOS2_HOME to a Zonos2 checkout with a built venv." 1>&2
        return 1
    fi
    serve_start zonos2 "${ZONOS2_PORT}" 2 "${ZONOS2_STARTUP_TIMEOUT}" \
        _zonos2_launch _zonos2_health zonos2 || return 1
    ZONOS2_PORT=${SERVE_PORT}
    ZONOS2_PID=${SERVE_PID}
    ZONOS2_BASE_URL="http://${ZONOS2_HOST}:${ZONOS2_PORT}"
    export ZONOS2_PORT ZONOS2_BASE_URL
}

stop_zonos2_server() {
    [ -n "${ZONOS2_PID}" ] && echo "[zonos2] stopping server (pid ${ZONOS2_PID})"
    serve_stop "${ZONOS2_PID}"
    ZONOS2_PID=
}
