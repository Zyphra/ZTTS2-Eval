#!/usr/bin/env bash
# One-shot reproducible install for the VibeVoice-Realtime-0.5B (Microsoft) venv.
#
# VibeVoice-Realtime-0.5B is a ~0.5B streaming English preset-voice TTS (NOT a cloner).
# Runs in-process (own venv) via the cloned repo's streaming inference code. Like the
# neutts recipe, pin cu128 torch FIRST so the nvidia stack matches the CUDA-12.8 driver.
#
# Idempotent; pass --force to nuke the venv.
#   bash scripts/setup/setup_vibevoice_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VIBEVOICE_VENV="${VIBEVOICE_VENV:-$HOME/.venvs/vibevoice}"
SRC="${VIBEVOICE_SRC:-${REPO_ROOT}/third_party/VibeVoice}"
PY_VER="${PY_VER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu128}"
VIBEVOICE_REPO="${VIBEVOICE_REPO:-https://github.com/microsoft/VibeVoice}"
VIBEVOICE_COMMIT="${VIBEVOICE_COMMIT:-16fb2cb1217c9934a886e1948ffb06120caa2df5}"

. "${REPO_ROOT}/scripts/setup/lib_setup.sh"
setup_force_from_args "$@"

setup_step "1. VibeVoice @ ${VIBEVOICE_COMMIT:0:10} -> ${SRC}"
setup_checkout "${VIBEVOICE_REPO}" "${SRC}" "${VIBEVOICE_COMMIT}"

setup_step "2. uv venv ${VIBEVOICE_VENV} (python ${PY_VER})"
setup_venv "${VIBEVOICE_VENV}" "${PY_VER}"

setup_step "3. pin cu128 torch FIRST, then VibeVoice[streamingtts] (transformers==4.51.3)"
uv pip install --python "${VIBEVOICE_VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --index-strategy unsafe-best-match \
    "torch==2.8.0" torchaudio
uv pip install --python "${VIBEVOICE_VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --index-strategy unsafe-best-match \
    -e "${SRC}[streamingtts]" soundfile

setup_step "4. smoke check"
"${VIBEVOICE_VENV}/bin/python" - <<'PY'
import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference  # noqa: F401
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor  # noqa: F401
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "-- VibeVoice import OK")
PY

cat <<EOF

VibeVoice venv ready at ${VIBEVOICE_VENV}. Model microsoft/VibeVoice-Realtime-0.5B downloads
to the HF cache on first run. Generate (English only):
  bash scripts/gen/run_vibevoice.sh
  bash scripts/gen/run_vibevoice.sh --limit 3
EOF
