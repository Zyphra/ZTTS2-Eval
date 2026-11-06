# slurm/lib_slurm.sh -- plumbing shared by the sbatch job bodies.
#
# SOURCE this from the repo root (`. slurm/lib_slurm.sh`); do not execute it.
#
# Ports: each job owns a 40-port window keyed on its job id; a GPU slot owns 4 ports
# of it, one per role.
#   slurm_port_base SLOT         20000 + (${SLURM_JOB_ID:-$$} % 1000) * 40 + SLOT * 4
#   slurm_role_port BASE ROLE    tts=+0  tts_aux=+1  asr=+2  whisper=+3
# Node resources:
#   slurm_gpu_count              GPUs visible on the node (at least 1)
#   slurm_mps_dir TAG            a private CUDA_MPS_PIPE_DIRECTORY (keeps jobs off MPS)
#   slurm_server_log_dir TAG     logs/slurm/servers/TAG, created; use as SERVE_LOG_DIR
# Work distribution across array tasks (atomic mkdir claims):
#   slurm_claims_init DIR OWNER  create DIR; release OWNER's unfinished claims (requeue)
#   slurm_claim DIR KEY OWNER    0 when this caller won KEY
#   slurm_claim_done DIR KEY     mark KEY finished
#   slurm_list_once LIST BUILDER the first caller runs BUILDER > LIST (under a mkdir
#                                lock); every other caller waits until LIST exists
# Leaf coverage:
#   slurm_want TEXT LIMIT        uids in a text file, capped at LIMIT when 0 < LIMIT < n
#   slurm_missing TEXT WAVDIR LIMIT  wanted uids (first slurm_want of TEXT) lacking WAVDIR/<uid>.wav

slurm_port_base() {  # <slot>
    echo $(( 20000 + (${SLURM_JOB_ID:-$$} % 1000) * 40 + $1 * 4 ))
}

slurm_role_port() {  # <base> <role>
    case "$2" in
        tts) echo $(( $1 + 0 )) ;;
        tts_aux) echo $(( $1 + 1 )) ;;
        asr) echo $(( $1 + 2 )) ;;
        whisper) echo $(( $1 + 3 )) ;;
        *) echo "slurm_role_port: unknown role '$2'" 1>&2; return 1 ;;
    esac
}

slurm_gpu_count() {
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "${n}" -ge 1 ] || n=1
    echo "${n}"
}

slurm_mps_dir() {  # <tag>
    echo "/tmp/nompss-${USER:-u}-$1"
}

slurm_server_log_dir() {  # <tag>
    mkdir -p "logs/slurm/servers/$1"
    echo "logs/slurm/servers/$1"
}

slurm_claims_init() {  # <dir> <owner>
    local c
    mkdir -p "$1"
    for c in "$1"/*/; do
        [ -d "${c}" ] || continue
        [ -f "${c}done" ] && continue
        [ "$(cat "${c}owner" 2>/dev/null)" = "$2" ] && rm -rf "${c}"
    done
    return 0
}

slurm_claim() {  # <dir> <key> <owner>
    mkdir "$1/$2" 2>/dev/null || return 1
    echo "$3" > "$1/$2/owner"
}

slurm_claim_done() {  # <dir> <key>
    touch "$1/$2/done"
}

slurm_list_once() {  # <list> <builder>
    if [ ! -f "$1" ]; then
        if mkdir "$1.lock" 2>/dev/null; then
            "$2" > "$1.tmp"
            mv "$1.tmp" "$1"
        else
            until [ -f "$1" ]; do sleep 5; done
        fi
    fi
}

slurm_want() {  # <text> <limit>
    local n
    n=$(awk 'NF' "$1" | grep -c .)
    if [ "$2" -gt 0 ] && [ "$2" -lt "${n}" ]; then n=$2; fi
    echo "${n}"
}

slurm_missing() {  # <text> <wavdir> <limit>
    LC_ALL=C comm -23 \
        <(awk 'NF {print $1}' "$1" | head -n "$(slurm_want "$1" "$3")" | LC_ALL=C sort -u) \
        <(find "$2" -maxdepth 1 -name '*.wav' -printf '%f\n' 2>/dev/null \
            | sed 's/\.wav$//' | LC_ALL=C sort -u) \
        | grep -c . || true
}
