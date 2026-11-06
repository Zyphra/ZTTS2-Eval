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

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TTSDS_VENV="${TTSDS_VENV:-$HOME/.venvs/ttsds}"
PY_VER="${PY_VER:-3.12}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH="${TORCH:-2.5.1}"
TORCHAUDIO="${TORCHAUDIO:-2.5.1}"
TORCHVISION="${TORCHVISION:-0.20.1}"
TRANSFORMERS="${TRANSFORMERS:-4.46.3}"
PYANNOTE_MIN="${PYANNOTE_MIN:-3.3.0}"
TTSDS_VERSION="${TTSDS_VERSION:-2.1.1}"

. "${REPO_ROOT}/scripts/setup/lib_setup.sh"
setup_force_from_args "$@"

UV_PIP=(uv pip install --python "${TTSDS_VENV}/bin/python" --no-build-isolation)

setup_step "1. create ${TTSDS_VENV} (python ${PY_VER})"
setup_venv "${TTSDS_VENV}" "${PY_VER}"

setup_step "2. install build helpers"
uv pip install --python "${TTSDS_VENV}/bin/python" pip 'setuptools<81' wheel cython hatchling editables

setup_step "3. install ttsds ${TTSDS_VERSION}"
"${UV_PIP[@]}" "ttsds==${TTSDS_VERSION}"

setup_step "4. rebuild pyworld against NumPy 2"
uv pip install --python "${TTSDS_VENV}/bin/python" 'numpy>=2,<2.5'
"${UV_PIP[@]}" --force-reinstall --no-cache pyworld==0.3.5

setup_step "5. pin torch/torchaudio ${TORCH}+${CUDA_TAG}"
"${UV_PIP[@]}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "torch==${TORCH}" "torchaudio==${TORCHAUDIO}" "torchvision==${TORCHVISION}"

TORCH_PINS=("torch==${TORCH}+${CUDA_TAG}" "torchaudio==${TORCHAUDIO}+${CUDA_TAG}" "torchvision==${TORCHVISION}+${CUDA_TAG}")

setup_step "6. install compatible pyannote.audio"
"${UV_PIP[@]}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "pyannote.audio>=${PYANNOTE_MIN}" "${TORCH_PINS[@]}"

setup_step "7. pin transformers ${TRANSFORMERS}"
"${UV_PIP[@]}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "transformers==${TRANSFORMERS}" "${TORCH_PINS[@]}"

setup_step "8. smoke import TTSDS MPM"
"${TTSDS_VENV}/bin/python" - <<'PY'
import warnings
warnings.filterwarnings("ignore")
from ttsds.benchmarks.prosody.mpm import MPMBenchmark
import ttsds
print("ttsds:", ttsds.__file__)
print("MPM benchmark:", MPMBenchmark.__name__)
PY

echo
echo "done. activate: source ${TTSDS_VENV}/bin/activate"
