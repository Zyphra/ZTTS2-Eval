# scripts/gen/serve_zonos3.sh -- ZONOS3 Mini (release minisgl server) lifecycle.
#
# SOURCE this file (`. scripts/gen/serve_zonos3.sh`); do not execute it, so the caller
# owns ZONOS3_PID and its `trap` can clean up (same contract as serve_zonos2.sh).
#
#   start_zonos3_server   launch a fresh server, block until ready (3 attempts)
#   stop_zonos3_server    terminate the server we started (whole process group)
#   zonos3_healthy        0 once ${ZONOS3_BASE_URL}/tts/capabilities answers 200
#
# The server is `python -m zonos2` from a minisgl-zonos3 checkout (a separate tree
# with its own venv, not part of this repo's .venv), launched with the release
# flags. Unlike serve_zonos2.sh it NEVER reuses a server already on the port: a
# foreign server would silently serve another config.
#
# Environment (exported; the launch shell reads them):
#   ZONOS3_HOME             checkout to cd into                          (required)
#   ZONOS3_ENV              site env script, sourced in the launch shell (optional)
#   ZONOS3_PY               python; empty -> ${PYTHON:-$ZONOS3_HOME/.venv/bin/python}
#   ZONOS3_MODEL            --model-path; empty -> ${ZONOS3_RELEASE_LM:-hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k}
#   ZONOS3_HOST             bind address                   (default 127.0.0.1)
#   ZONOS3_PORT             preferred port                 (default 1920; PORT+1 is the
#                           torch-distributed rendezvous and must be free too)
#   ZONOS3_GPU              CUDA_VISIBLE_DEVICES           (default 0)
#   ZONOS3_MAX_RUNNING      --max-running-requests and --cuda-graph-max-bs (default 16)
#   ZONOS3_MEMORY_RATIO     --memory-ratio                 (default 0.6; 0.75 OOMs speaker_mode=embedding)
#   ZONOS3_EOA_POLICY       --eoa-policy                   (default any; "none" omits the flag, for
#                           runtimes without it, whose only behaviour is any-codebook EOA)
#   ZONOS3_SERVE_ARGS       extra `-m zonos2` flags
#   ZONOS3_STARTUP_TIMEOUT  seconds to wait per attempt    (default 1800)
#   ZONOS3_LOG              server log                     (default /tmp/zonos3_<port>.log)
#
# After a successful start: ZONOS3_PORT / ZONOS3_BASE_URL name the port actually
# bound (a retry may move it) and ZONOS3_RUNTIME_REV is the checkout's git rev.

ZONOS3_HOME=${ZONOS3_HOME:-}
ZONOS3_ENV=${ZONOS3_ENV:-}
ZONOS3_PY=${ZONOS3_PY:-}
ZONOS3_MODEL=${ZONOS3_MODEL:-}
ZONOS3_HOST=${ZONOS3_HOST:-127.0.0.1}
ZONOS3_PORT=${ZONOS3_PORT:-1920}
ZONOS3_GPU=${ZONOS3_GPU:-0}
ZONOS3_MAX_RUNNING=${ZONOS3_MAX_RUNNING:-16}
ZONOS3_MEMORY_RATIO=${ZONOS3_MEMORY_RATIO:-0.6}
ZONOS3_EOA_POLICY=${ZONOS3_EOA_POLICY:-any}
ZONOS3_SERVE_ARGS=${ZONOS3_SERVE_ARGS:-}
ZONOS3_STARTUP_TIMEOUT=${ZONOS3_STARTUP_TIMEOUT:-1800}
ZONOS3_BASE_URL="http://${ZONOS3_HOST}:${ZONOS3_PORT}"
ZONOS3_PID=""
export ZONOS3_HOME ZONOS3_ENV ZONOS3_PY ZONOS3_MODEL ZONOS3_HOST ZONOS3_PORT ZONOS3_GPU \
       ZONOS3_MAX_RUNNING ZONOS3_MEMORY_RATIO ZONOS3_EOA_POLICY ZONOS3_SERVE_ARGS ZONOS3_BASE_URL

# /tts/capabilities is 200 only once the model is loaded; -f turns 4xx/5xx into failure.
zonos3_healthy() {
    curl -fs -o /dev/null -m 3 "${ZONOS3_BASE_URL}/tts/capabilities"
}

# _zonos3_free_pair PREFERRED: PREFERRED if it and PREFERRED+1 are bindable on
# 127.0.0.1, else an OS-chosen free pair.
_zonos3_free_pair() {
    python3 - "$1" <<'PY'
import socket, sys

def free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True

port = int(sys.argv[1])
while not (port < 65535 and free(port) and free(port + 1)):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
print(port)
PY
}

start_zonos3_server() {
    if [ ! -d "${ZONOS3_HOME}" ]; then
        echo "[zonos3] ERROR: ZONOS3_HOME is not set (the minisgl-zonos3 checkout to serve from)" 1>&2
        return 1
    fi
    local pref=${ZONOS3_PORT} attempt port logf waited failed why
    for attempt in 1 2 3; do
        port=$(_zonos3_free_pair $(( pref + 1000 * (attempt - 1) )))
        ZONOS3_PORT=${port}
        ZONOS3_BASE_URL="http://${ZONOS3_HOST}:${port}"
        export ZONOS3_PORT ZONOS3_BASE_URL
        logf=${ZONOS3_LOG:-/tmp/zonos3_${port}.log}
        echo "[zonos3] attempt ${attempt}: starting on ${ZONOS3_BASE_URL} (GPU ${ZONOS3_GPU}, log ${logf})"
        # setsid -> new process-group leader, so stop_zonos3_server can kill the
        # scheduler/worker children that hold the GPU and the rendezvous port.
        setsid env -u ZONOS2_ACCESS_TOKEN CUDA_VISIBLE_DEVICES="${ZONOS3_GPU}" bash -c '
            cd "$ZONOS3_HOME" || exit 127
            if [ -n "${ZONOS3_ENV:-}" ]; then . "$ZONOS3_ENV" || exit 126; fi
            py="${ZONOS3_PY:-${PYTHON:-$ZONOS3_HOME/.venv/bin/python}}"
            model="${ZONOS3_MODEL:-${ZONOS3_RELEASE_LM:-hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k}}"
            eoa="--eoa-policy $ZONOS3_EOA_POLICY"
            [ "$ZONOS3_EOA_POLICY" = none ] && eoa=""
            echo "[zonos3] runtime $(git -C "$ZONOS3_HOME" rev-parse --short HEAD 2>/dev/null) python $py model $model"
            exec "$py" -m zonos2 --model-path "$model" --attention-backend fa --cache-type naive \
                --host "$ZONOS3_HOST" --port "$1" \
                --max-running-requests "$ZONOS3_MAX_RUNNING" --cuda-graph-max-bs "$ZONOS3_MAX_RUNNING" \
                --max-seq-len-override 4096 --num-tokenizer 1 --memory-ratio "$ZONOS3_MEMORY_RATIO" \
                $eoa --tts-emotion-directions-dir "" --log-level INFO $ZONOS3_SERVE_ARGS
        ' zonos3-serve "${port}" > "${logf}" 2>&1 &
        ZONOS3_PID=$!

        # The frontend binds its port only after the model loads and can lose it to
        # another process; the engine can die on a rendezvous timeout while the
        # frontend lives on. Both leave one of these signatures in the log.
        waited=0
        failed=""
        while :; do
            if zonos3_healthy; then
                break
            fi
            if ! kill -0 "${ZONOS3_PID}" 2>/dev/null; then
                failed="server process exited"
                break
            fi
            if grep -qE "address already in use|DistNetworkError|^Traceback" "${logf}"; then
                failed="startup error in log"
                break
            fi
            if [ "${waited}" -ge "${ZONOS3_STARTUP_TIMEOUT}" ]; then
                failed="not healthy after ${ZONOS3_STARTUP_TIMEOUT}s"
                break
            fi
            sleep 2
            waited=$((waited + 2))
        done
        if [ -z "${failed}" ]; then
            ZONOS3_RUNTIME_REV=$(git -C "${ZONOS3_HOME}" rev-parse --short HEAD 2>/dev/null)
            export ZONOS3_RUNTIME_REV
            echo "[zonos3] server ready after ~${waited}s on ${ZONOS3_BASE_URL} (pid ${ZONOS3_PID})"
            return 0
        fi
        why=$(grep -m1 -E "address already in use|DistNetworkError|^Traceback|Error" "${logf}")
        echo "[zonos3] attempt ${attempt} failed on port ${port}: ${why:-${failed}}" 1>&2
        stop_zonos3_server
    done
    echo "[zonos3] ERROR: server failed to start in 3 attempts. Last log lines:" 1>&2
    tail -n 40 "${logf}" 1>&2
    return 1
}

stop_zonos3_server() {
    [ -z "${ZONOS3_PID}" ] && return 0
    echo "[zonos3] stopping server (pid ${ZONOS3_PID})"
    kill -- -"${ZONOS3_PID}" 2>/dev/null || kill "${ZONOS3_PID}" 2>/dev/null || true
    local i=0
    while kill -0 "${ZONOS3_PID}" 2>/dev/null; do
        i=$((i + 1))
        [ "${i}" -ge 30 ] && break
        sleep 1
    done
    kill -9 -- -"${ZONOS3_PID}" 2>/dev/null || kill -9 "${ZONOS3_PID}" 2>/dev/null || true
    ZONOS3_PID=""
    # Wait for the port to actually free before the caller starts the next server.
    local w=0
    while zonos3_healthy && [ "${w}" -lt 30 ]; do
        sleep 1
        w=$((w + 1))
    done
}
