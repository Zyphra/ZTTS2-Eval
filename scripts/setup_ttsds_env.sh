#!/usr/bin/env bash
# Reproducible environment for the TTSDS2 MPM (Masked Prosody Model) metric.
#
# TTSDS2's dependency tree needs a few compatibility pins on Python 3.12:
# pyworld must build against NumPy 2, pyannote.audio must be new enough for
# current torchaudio, and transformers 5 requires a newer torch than this host.
# No vendored source or GPU patches are needed.
#
# Pass --force to recreate the venv.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/ttsds}"
PY_VER="${PY_VER:-3.12}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH="${TORCH:-2.5.1}"
TORCHAUDIO="${TORCHAUDIO:-2.5.1}"
TORCHVISION="${TORCHVISION:-0.20.1}"
TRANSFORMERS="${TRANSFORMERS:-4.46.3}"
PYANNOTE_MIN="${PYANNOTE_MIN:-3.3.0}"
TTSDS_VERSION="${TTSDS_VERSION:-2.1.1}"

if [ "${1:-}" = "--force" ]; then
    rm -rf "${VENV}"
fi

UV_PIP=(uv pip install --python "${VENV}/bin/python" --no-build-isolation)
step() { printf "\n=== %s ===\n" "$*"; }

step "1. create ${VENV} (python ${PY_VER})"
if [ ! -x "${VENV}/bin/python" ]; then
    uv venv --python "${PY_VER}" "${VENV}"
fi

step "2. install build helpers"
uv pip install --python "${VENV}/bin/python" pip 'setuptools<81' wheel cython hatchling editables

step "3. install ttsds ${TTSDS_VERSION}"
"${UV_PIP[@]}" "ttsds==${TTSDS_VERSION}"

step "4. rebuild pyworld against NumPy 2"
uv pip install --python "${VENV}/bin/python" 'numpy>=2,<2.5'
"${UV_PIP[@]}" --force-reinstall --no-cache pyworld==0.3.5

step "5. pin torch/torchaudio ${TORCH}+${CUDA_TAG}"
"${UV_PIP[@]}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "torch==${TORCH}" "torchaudio==${TORCHAUDIO}" "torchvision==${TORCHVISION}"

TORCH_PINS=("torch==${TORCH}+${CUDA_TAG}" "torchaudio==${TORCHAUDIO}+${CUDA_TAG}" "torchvision==${TORCHVISION}+${CUDA_TAG}")

step "6. install compatible pyannote.audio"
"${UV_PIP[@]}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "pyannote.audio>=${PYANNOTE_MIN}" "${TORCH_PINS[@]}"

step "7. pin transformers ${TRANSFORMERS}"
"${UV_PIP[@]}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "transformers==${TRANSFORMERS}" "${TORCH_PINS[@]}"

step "8. smoke import TTSDS MPM"
"${VENV}/bin/python" - <<'PY'
import warnings
warnings.filterwarnings("ignore")
from ttsds.benchmarks.prosody.mpm import MPMBenchmark
import ttsds
print("ttsds:", ttsds.__file__)
print("MPM benchmark:", MPMBenchmark.__name__)
PY

echo
echo "done. activate: source ${VENV}/bin/activate"
