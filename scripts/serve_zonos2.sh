# scripts/serve_zonos2.sh -- Zonos2 (Mini-SGLang) TTS server lifecycle.
#
# SOURCE this file (`. scripts/serve_zonos2.sh`); do not execute it. Sourcing lets
# the caller own the server PID so its `trap` can clean up -- exactly like
# scripts/serve_omni.sh (vLLM-Omni) and scripts/serve_asr.sh.
#
#   start_zonos2_server   launch the server for $ZONOS2_MODEL, block until ready
#   stop_zonos2_server    terminate a server we started (no-op for a reused one)
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
#   ZONOS2_HOST ZONOS2_PORT  bind address              (default 127.0.0.1:1919)
#   ZONOS2_GPU            GPU index (CUDA_VISIBLE_DEVICES) (default 0)
#   ZONOS2_SERVE_ARGS     extra `-m zonos2` flags        (e.g. --dtype, --attn, --memory-ratio)
#   ZONOS2_STARTUP_TIMEOUT  seconds to wait for readiness (default 1800 -- loads a 15G model)
#   ZONOS2_LOG            server log file               (default /tmp/zonos2_<port>.log)
#
# NOTE: the Mini-SGLang scheduler also binds an IPC socket on ZONOS2_PORT+1, so keep
# that port free too (the per-job port scheme in slurm/eval_job.sh accounts for it).
#
# Set ZONOS2_HOST/ZONOS2_PORT before sourcing this file (the health URL is built here).

ZONOS2_MODEL=${ZONOS2_MODEL:-Zyphra/ZONOS2}
ZONOS2_HOME=${ZONOS2_HOME:-}
ZONOS2_PY=${ZONOS2_PY:-${ZONOS2_HOME}/.venv/bin/python}
ZONOS2_VOICES_DIR=${ZONOS2_VOICES_DIR:-${ZONOS2_HOME}/default_voices}
ZONOS2_HOST=${ZONOS2_HOST:-127.0.0.1}
ZONOS2_PORT=${ZONOS2_PORT:-1919}
ZONOS2_GPU=${ZONOS2_GPU:-0}
ZONOS2_STARTUP_TIMEOUT=${ZONOS2_STARTUP_TIMEOUT:-1800}
ZONOS2_BASE_URL="http://${ZONOS2_HOST}:${ZONOS2_PORT}"
export ZONOS2_BASE_URL

# /tts/capabilities returns 200 only once the model is fully loaded and the request
# state is live -- it is the cheapest readiness probe (the generation client already
# uses it to gate speaker/accurate-mode support). -f makes a 4xx/5xx a failure so we
# wait for a real 200, not just a bound-but-not-ready port.
_ZONOS2_HEALTH_URL="http://${ZONOS2_HOST}:${ZONOS2_PORT}/tts/capabilities"
ZONOS2_PID=""

_zonos2_healthy() {
    curl -fs -o /dev/null -m 3 "${_ZONOS2_HEALTH_URL}"
}

start_zonos2_server() {
    [ -z "${ZONOS2_MODEL}" ] && { echo "[zonos2] ERROR: ZONOS2_MODEL is not set" 1>&2; return 1; }
    if [ ! -x "${ZONOS2_PY}" ]; then
        echo "[zonos2] ERROR: python '${ZONOS2_PY}' not found/executable." \
             "Set ZONOS2_PY or ZONOS2_HOME to a Zonos2 checkout with a built venv." 1>&2
        return 1
    fi
    local logf="${ZONOS2_LOG:-/tmp/zonos2_${ZONOS2_PORT}.log}"

    # Reuse a server already up on this port; if something else holds it, refuse
    # rather than collide (mirrors serve_omni.sh -- a foreign server would silently
    # serve the wrong config).
    if _zonos2_healthy; then
        echo "[zonos2] reusing healthy server at ${ZONOS2_BASE_URL}"
        ZONOS2_PID=""
        return 0
    fi

    local voices_opt=""
    [ -n "${ZONOS2_VOICES_DIR}" ] && [ -d "${ZONOS2_VOICES_DIR}" ] && \
        voices_opt="--tts-default-voices-dir ${ZONOS2_VOICES_DIR}"

    echo "[zonos2] starting: ${ZONOS2_PY} -m zonos2 --model-path ${ZONOS2_MODEL} ${voices_opt} ${ZONOS2_SERVE_ARGS}" \
         "on ${ZONOS2_HOST}:${ZONOS2_PORT} (GPU ${ZONOS2_GPU}, log ${logf})"
    # setsid makes the server a new process-group leader (pgid == pid) so
    # stop_zonos2_server can kill the WHOLE group -- Mini-SGLang forks scheduler/worker
    # children that hold the GPU and the port+1 IPC socket, and a bare parent-kill
    # would orphan them. Run from the checkout root so relative asset paths resolve;
    # the package is importable from that venv regardless.
    setsid bash -c '
        cd "'"${ZONOS2_HOME}"'" || exit 127
        export CUDA_VISIBLE_DEVICES="'"${ZONOS2_GPU}"'"
        exec "'"${ZONOS2_PY}"'" -m zonos2 \
            --model-path "'"${ZONOS2_MODEL}"'" \
            '"${voices_opt}"' \
            --host "'"${ZONOS2_HOST}"'" --port "'"${ZONOS2_PORT}"'" \
            '"${ZONOS2_SERVE_ARGS}"'
    ' > "${logf}" 2>&1 &
    ZONOS2_PID=$!

    local waited=0
    until _zonos2_healthy; do
        if ! kill -0 "${ZONOS2_PID}" 2>/dev/null; then
            echo "[zonos2] ERROR: server exited during startup. Last log lines:" 1>&2
            tail -n 40 "${logf}" 1>&2
            ZONOS2_PID=""
            return 1
        fi
        if [ "${waited}" -ge "${ZONOS2_STARTUP_TIMEOUT}" ]; then
            echo "[zonos2] ERROR: startup timed out after ${ZONOS2_STARTUP_TIMEOUT}s. Last log lines:" 1>&2
            tail -n 40 "${logf}" 1>&2
            stop_zonos2_server
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[zonos2] server ready after ~${waited}s (pid ${ZONOS2_PID})"
}

stop_zonos2_server() {
    [ -z "${ZONOS2_PID}" ] && return 0
    echo "[zonos2] stopping server (pid ${ZONOS2_PID})"
    # Kill the whole process group -- Mini-SGLang forks scheduler/worker children that
    # outlive the parent otherwise (they hold the GPU and the port+1 IPC socket).
    kill -- -"${ZONOS2_PID}" 2>/dev/null || kill "${ZONOS2_PID}" 2>/dev/null || true
    local i=0
    while kill -0 "${ZONOS2_PID}" 2>/dev/null; do
        i=$((i + 1))
        [ "${i}" -ge 30 ] && break
        sleep 1
    done
    kill -9 -- -"${ZONOS2_PID}" 2>/dev/null || kill -9 "${ZONOS2_PID}" 2>/dev/null || true
    ZONOS2_PID=""
    # Wait for the port to actually free before the caller starts the next phase.
    local w=0
    while _zonos2_healthy && [ "${w}" -lt 30 ]; do
        sleep 1
        w=$((w + 1))
    done
}
