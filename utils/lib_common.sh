# utils/lib_common.sh -- interpreter selection shared by every non-sbatch shell entry point.
#
# SOURCE this from the repo root (`. utils/lib_common.sh`); do not execute it. A failed
# source is the caller's "not run from the repo root" check.
#
#   repo_py    PY = $PYTHON if set, else .venv/bin/python if executable, else python3
#
# Jobs submitted with sbatch hard-code .venv/bin/python instead; stdlib-only helpers
# (`python3 -m utils.benchmark langs|version`) stay on python3.

repo_py() {
    if [ -n "${PYTHON:-}" ]; then PY=${PYTHON}
    elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
    else PY=python3; fi
}
