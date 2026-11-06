# utils/lib_serve.sh -- shared lifecycle for the locally launched model servers.
#
# SOURCE this from the repo root (`. utils/lib_serve.sh`); do not execute it. Sourcing
# lets the caller own the server PID so its `trap` can stop it on exit/error/interrupt.
# Used by scripts/gen/serve_{zonos2,zonos3,omni,audio8}.sh and scripts/score/serve_asr.sh.
#
#   serve_free_port PREF [N]   PREF when 127.0.0.1:PREF..PREF+N-1 all bind; PREF=0 or any
#                              of them busy -> an OS-chosen P with P..P+N-1 free
#   serve_log_path ROLE PORT   ${SERVE_LOG_DIR}/<ROLE>.log when SERVE_LOG_DIR is set,
#                              else /tmp/<ROLE>_<PORT>.log
#   serve_healthy URL          0 once URL answers 2xx
#   serve_stop PID             kill PID's process group (TERM, <=30 s, then KILL)
#   serve_start NAME PREF N TIMEOUT LAUNCH_FN HEALTH_FN ROLE
#       Launch a fresh server and block until HEALTH_FN PORT succeeds. Attempt 1 binds
#       $(serve_free_port PREF N), attempts 2-3 an OS-chosen window of N ports.
#       LAUNCH_FN PORT LOGF must start the server as `setsid <cmd> > LOGF 2>&1 &` and set
#       SERVE_PID=$!. A port race (address already in use / DistNetworkError in the log)
#       stops the server and retries; any other exit, or no health after TIMEOUT
#       seconds, prints the log tail and returns 1. On success SERVE_PORT and SERVE_PID
#       name the running server. A server already listening is never adopted.
#
# Environment: SERVE_LOG_DIR (optional) collects every server log of one job.

. utils/lib_common.sh

serve_free_port() {  # <pref> [n]
    python3 - "$1" "${2:-1}" <<'PY'
import socket, sys

def window_free(port, n):
    socks = []
    try:
        for p in range(port, port + n):
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # as the servers bind
            socks.append(s)
            s.bind(("127.0.0.1", p))
        return True
    except OSError:
        return False
    finally:
        for s in socks:
            s.close()

pref, n = int(sys.argv[1]), int(sys.argv[2])
if pref > 0 and pref + n <= 65536 and window_free(pref, n):
    print(pref)
    sys.exit(0)
while True:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    if port + n <= 65536 and window_free(port, n):
        print(port)
        break
PY
}

serve_log_path() {  # <role> <port>
    if [ -n "${SERVE_LOG_DIR:-}" ]; then
        mkdir -p "${SERVE_LOG_DIR}"
        echo "${SERVE_LOG_DIR}/$1.log"
    else
        echo "/tmp/$1_$2.log"
    fi
}

serve_healthy() {  # <url>
    curl -fs -o /dev/null -m 3 "$1"
}

serve_stop() {  # <pid>
    local pid=$1 i=0
    [ -n "${pid}" ] || return 0
    kill -- -"${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    while kill -0 "${pid}" 2>/dev/null && [ "${i}" -lt 30 ]; do
        sleep 1
        i=$((i + 1))
    done
    kill -9 -- -"${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    return 0
}

_SERVE_RETRY_RE='address already in use|Address already in use|DistNetworkError'

serve_start() {  # <name> <pref> <n> <timeout> <launch_fn> <health_fn> <role>
    # Locals are _ss_-prefixed: bash scoping is dynamic, so plain names would shadow
    # globals the launch/health functions read.
    local _ss_name=$1 _ss_pref=$2 _ss_n=$3 _ss_timeout=$4 _ss_launch=$5 _ss_health=$6 _ss_role=$7
    local _ss_attempt _ss_port _ss_logf _ss_waited
    SERVE_PORT=""
    SERVE_PID=""
    for _ss_attempt in 1 2 3; do
        if [ "${_ss_attempt}" -eq 1 ]; then
            _ss_port=$(serve_free_port "${_ss_pref}" "${_ss_n}")
        else
            _ss_port=$(serve_free_port 0 "${_ss_n}")
        fi
        _ss_logf=$(serve_log_path "${_ss_role}" "${_ss_port}")
        echo "[${_ss_name}] attempt ${_ss_attempt}: starting on port ${_ss_port} (log ${_ss_logf})"
        SERVE_PID=""
        "${_ss_launch}" "${_ss_port}" "${_ss_logf}"
        if [ -z "${SERVE_PID}" ]; then
            echo "[${_ss_name}] ERROR: launch function set no SERVE_PID" 1>&2
            return 1
        fi
        _ss_waited=0
        while :; do
            if "${_ss_health}" "${_ss_port}"; then
                SERVE_PORT=${_ss_port}
                echo "[${_ss_name}] server ready after ~${_ss_waited}s on port ${_ss_port} (pid ${SERVE_PID})"
                return 0
            fi
            # A lost port race leaves its signature in the log; the process may or may
            # not have exited yet (a frontend can outlive its dead engine).
            if grep -qE "${_SERVE_RETRY_RE}" "${_ss_logf}" 2>/dev/null; then
                echo "[${_ss_name}] attempt ${_ss_attempt} lost port ${_ss_port}:" \
                     "$(grep -m1 -E "${_SERVE_RETRY_RE}" "${_ss_logf}")" 1>&2
                serve_stop "${SERVE_PID}"
                SERVE_PID=""
                continue 2
            fi
            if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
                echo "[${_ss_name}] ERROR: server exited during startup. Last log lines:" 1>&2
                serve_stop "${SERVE_PID}"   # reap any surviving group members
                SERVE_PID=""
                tail -n 40 "${_ss_logf}" 1>&2
                return 1
            fi
            if [ "${_ss_waited}" -ge "${_ss_timeout}" ]; then
                echo "[${_ss_name}] ERROR: not healthy after ${_ss_timeout}s. Last log lines:" 1>&2
                serve_stop "${SERVE_PID}"
                SERVE_PID=""
                tail -n 40 "${_ss_logf}" 1>&2
                return 1
            fi
            sleep 2
            _ss_waited=$((_ss_waited + 2))
        done
    done
    echo "[${_ss_name}] ERROR: server failed to start in 3 attempts" 1>&2
    return 1
}
