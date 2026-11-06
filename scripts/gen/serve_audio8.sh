# scripts/gen/serve_audio8.sh -- Audio8-TTS (upstream SGLang-Omni adapter server) lifecycle.
#
# SOURCE this file (`. scripts/gen/serve_audio8.sh`); do not execute it, so the caller
# owns AUDIO8_PID and its `trap` can clean up (same contract as serve_zonos3.sh).
#
#   start_audio8_server   launch a fresh server, block until ready (3 attempts)
#   stop_audio8_server    terminate the server we started (whole process group)
#   audio8_healthy        0 once ${AUDIO8_BASE_URL}/health answers 200
#
# The server is the Edge0-AI/Audio8_TTS sglang_omni/scripts/run_server.sh, run from
# the pinned sglang-omni checkout with the adapter installed and its own venv (all
# set up by scripts/setup/setup_audio8_env.sh). It NEVER reuses a server already on the
# port: a foreign server would silently serve another config. Pipeline IPC lives in
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
#   AUDIO8_LOG                server log              (default /tmp/audio8_<port>.log)
#
# After a successful start: AUDIO8_PORT / AUDIO8_BASE_URL name the port actually
# bound (a retry may move it) and AUDIO8_RUNTIME_REV names both checkouts' git revs.

_audio8_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

# /health is 200 only once the pipeline is running; -f turns 4xx/5xx into failure.
audio8_healthy() {
    curl -fs -o /dev/null -m 3 "${AUDIO8_BASE_URL}/health"
}

# _audio8_free_port PREFERRED: PREFERRED if bindable on 127.0.0.1, else an OS-chosen
# free port. PREFERRED=0 always means OS-chosen.
_audio8_free_port() {
    python3 - "$1" <<'PY'
import socket, sys

port = int(sys.argv[1])
if port:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            port = 0
if not port:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
print(port)
PY
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
    local pref=${AUDIO8_PORT} attempt port master logf waited failed why omni_root
    omni_root=$(readlink -f "${SGLANG_OMNI_ROOT}")
    for attempt in 1 2 3; do
        port=$(_audio8_free_port $(( pref + 1000 * (attempt - 1) )))
        master=$(_audio8_free_port 0)
        AUDIO8_PORT=${port}
        AUDIO8_BASE_URL="http://${AUDIO8_HOST}:${port}"
        export AUDIO8_PORT AUDIO8_BASE_URL
        logf=${AUDIO8_LOG:-/tmp/audio8_${port}.log}
        echo "[audio8] attempt ${attempt}: starting on ${AUDIO8_BASE_URL} (GPU ${AUDIO8_GPU}, MASTER_PORT ${master}, cuda graphs off=${AUDIO8_DISABLE_CUDA_GRAPH}, log ${logf})"
        # setsid -> new process-group leader, so stop_audio8_server can kill the
        # pipeline stage children that hold the GPU.
        setsid env CUDA_VISIBLE_DEVICES="${AUDIO8_GPU}" MASTER_PORT="${master}" \
            PYTHON_BIN="${AUDIO8_VENV}/bin/python" SGLANG_OMNI_ROOT="${omni_root}" \
            MODEL="${AUDIO8_MODEL_DIR}" HOST="${AUDIO8_HOST}" PORT="${port}" \
            AUDIO8_TTS_MAX_RUNNING_REQUESTS="${AUDIO8_MAX_RUNNING}" \
            AUDIO8_TTS_DISABLE_CUDA_GRAPH="${AUDIO8_DISABLE_CUDA_GRAPH}" \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            bash "${AUDIO8_SRC}/sglang_omni/scripts/run_server.sh" > "${logf}" 2>&1 &
        AUDIO8_PID=$!

        waited=0
        failed=""
        while :; do
            if audio8_healthy; then
                break
            fi
            if ! kill -0 "${AUDIO8_PID}" 2>/dev/null; then
                failed="server process exited"
                break
            fi
            if grep -qE "address already in use|^Traceback" "${logf}"; then
                failed="startup error in log"
                break
            fi
            if [ "${waited}" -ge "${AUDIO8_STARTUP_TIMEOUT}" ]; then
                failed="not healthy after ${AUDIO8_STARTUP_TIMEOUT}s"
                break
            fi
            sleep 2
            waited=$((waited + 2))
        done
        if [ -z "${failed}" ]; then
            AUDIO8_RUNTIME_REV="audio8_tts@$(git -C "${AUDIO8_SRC}" rev-parse --short HEAD 2>/dev/null) sglang-omni@$(git -C "${SGLANG_OMNI_ROOT}" rev-parse --short HEAD 2>/dev/null)"
            export AUDIO8_RUNTIME_REV
            echo "[audio8] server ready after ~${waited}s on ${AUDIO8_BASE_URL} (pid ${AUDIO8_PID})"
            return 0
        fi
        why=$(grep -m1 -E "address already in use|^Traceback|Error" "${logf}")
        echo "[audio8] attempt ${attempt} failed on port ${port}: ${why:-${failed}}" 1>&2
        stop_audio8_server
    done
    echo "[audio8] ERROR: server failed to start in 3 attempts. Last log lines:" 1>&2
    tail -n 40 "${logf}" 1>&2
    return 1
}

stop_audio8_server() {
    [ -z "${AUDIO8_PID}" ] && return 0
    echo "[audio8] stopping server (pid ${AUDIO8_PID})"
    kill -- -"${AUDIO8_PID}" 2>/dev/null || kill "${AUDIO8_PID}" 2>/dev/null || true
    local i=0
    while kill -0 "${AUDIO8_PID}" 2>/dev/null; do
        i=$((i + 1))
        [ "${i}" -ge 30 ] && break
        sleep 1
    done
    kill -9 -- -"${AUDIO8_PID}" 2>/dev/null || kill -9 "${AUDIO8_PID}" 2>/dev/null || true
    AUDIO8_PID=""
    # Wait for the port to actually free before the caller starts the next server.
    local w=0
    while audio8_healthy && [ "${w}" -lt 30 ]; do
        sleep 1
        w=$((w + 1))
    done
}
