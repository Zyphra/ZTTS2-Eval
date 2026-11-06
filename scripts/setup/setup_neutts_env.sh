#!/usr/bin/env bash
# One-shot reproducible install for the NeuTTS-2E (Neuphonic) generation venv.
#
# NeuTTS-2E (neuphonic/neutts-2e) is a small (~125M) on-device English emotional TTS
# model with 4 fixed preset speakers -- NOT a zero-shot cloner. It runs in-process
# (like Kokoro/Echo) with its own venv. The backbone + NeuCodec download from HF.
#
# Idempotent; pass --force to nuke the venv.
#
#   bash scripts/setup/setup_neutts_env.sh
#   NEUTTS_VENV=/some/path bash scripts/setup/setup_neutts_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NEUTTS_VENV="${NEUTTS_VENV:-$HOME/.venvs/neutts}"
PY_VER="${PY_VER:-3.11}"

. "${REPO_ROOT}/scripts/setup/lib_setup.sh"
setup_force_from_args "$@"

setup_step "1. uv venv ${NEUTTS_VENV} (python ${PY_VER})"
setup_venv "${NEUTTS_VENV}" "${PY_VER}"

setup_step "2a. pin torch/torchaudio to cu128 FIRST (matches the CUDA 12.8 driver)"
# neutts[all] otherwise pulls the LATEST torch (2.13+cu130), whose CUDA-13 runtime
# does not initialise on this cluster's CUDA-12.8 driver (cuda.is_available()=False ->
# CPU, ~13 s/utt). Establishing a cu128 torch first makes the whole nvidia stack cu12.
uv pip install --python "${NEUTTS_VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/cu128" \
    --index-strategy unsafe-best-match \
    "torch==2.8.0" "torchaudio==2.8.0"

setup_step "2b. install neutts[all] + soundfile + tqdm (keeps the pinned cu128 torch)"
uv pip install --python "${NEUTTS_VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/cu128" \
    --index-strategy unsafe-best-match \
    "neutts[all]" soundfile tqdm

setup_step "3. warm the HF cache: backbone + NeuCodec"
"${NEUTTS_VENV}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for m in ["neuphonic/neutts-2e", "neuphonic/neucodec"]:
    snapshot_download(m); print("cached", m)
import torch; print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
from neutts import NeuTTS2E  # noqa: F401
print("NeuTTS2E import OK")
PY

cat <<EOF

NeuTTS-2E venv ready at ${NEUTTS_VENV}.
Generate with (English only):
  bash scripts/gen/run_neutts.sh
  bash scripts/gen/run_neutts.sh --limit 3    # smoke test
EOF
