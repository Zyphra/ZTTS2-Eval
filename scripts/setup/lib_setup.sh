# Shared helpers for scripts/setup/setup_*_env.sh. Source it; do not execute it.
#
#   setup_step MSG                         print "\n=== MSG ===\n"
#   setup_force_from_args "$@"             SETUP_FORCE=1 when $1 = --force
#   setup_venv DIR PY_VER                  rm -rf DIR when SETUP_FORCE=1; then
#                                          `uv venv --python PY_VER DIR` if DIR/bin/python is missing
#   setup_checkout URL DIR COMMIT [--recursive]
#                                          clone URL into DIR if missing, fetch origin,
#                                          check out COMMIT (+ recursive submodule update
#                                          with --recursive); echo "DIR @ <short sha>"

SETUP_FORCE="${SETUP_FORCE:-0}"

setup_step() { printf "\n=== %s ===\n" "$*"; }

setup_force_from_args() {
    if [ "${1:-}" = "--force" ]; then SETUP_FORCE=1; fi
}

setup_venv() {
    local dir=$1 py_ver=$2
    if [ "${SETUP_FORCE}" = 1 ]; then rm -rf "${dir}"; fi
    if [ ! -x "${dir}/bin/python" ]; then
        uv venv --python "${py_ver}" "${dir}"
    fi
}

setup_checkout() {
    local url=$1 dir=$2 commit=$3 recursive=${4:-}
    if [ ! -d "${dir}/.git" ]; then
        mkdir -p "$(dirname "${dir}")"
        git clone "${url}" "${dir}"
    fi
    git -C "${dir}" fetch -q origin
    git -C "${dir}" checkout -q "${commit}"
    if [ "${recursive}" = "--recursive" ]; then
        git -C "${dir}" submodule update --init --recursive
    fi
    echo "${dir} @ $(git -C "${dir}" rev-parse --short HEAD)"
}
