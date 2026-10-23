#!/usr/bin/env bash
# One-shot reproducible install for the NeuTTS-2E (Neuphonic) generation venv.
#
# NeuTTS-2E (neuphonic/neutts-2e) is a small (~125M) on-device English emotional TTS
# model with 4 fixed preset speakers -- NOT a zero-shot cloner. It runs in-process
# (like Kokoro/Echo) with its own venv. The backbone + NeuCodec download from HF.
#
# Idempotent; pass --force to nuke the venv.
#
#   bash scripts/setup_neutts_env.sh
#   VENV=/some/path bash scripts/setup_neutts_env.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/neutts}"
PY_VER="${PY_VER:-3.11}"

if [ "${1:-}" = "--force" ]; then rm -rf "$VENV"; fi

step() { printf "\n=== %s ===\n" "$*"; }

step "1. uv venv ${VENV} (python ${PY_VER})"
if [ ! -x "${VENV}/bin/python" ]; then uv venv --python "${PY_VER}" "${VENV}"; fi

step "2a. pin torch/torchaudio to cu128 FIRST (matches the CUDA 12.8 driver)"
# neutts[all] otherwise pulls the LATEST torch (2.13+cu130), whose CUDA-13 runtime
# does not initialise on this cluster's CUDA-12.8 driver (cuda.is_available()=False ->
# CPU, ~13 s/utt). Establishing a cu128 torch first makes the whole nvidia stack cu12.
uv pip install --python "${VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/cu128" \
    --index-strategy unsafe-best-match \
    "torch==2.8.0" "torchaudio==2.8.0"

step "2b. install neutts[all] + soundfile + tqdm (keeps the pinned cu128 torch)"
uv pip install --python "${VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/cu128" \
    --index-strategy unsafe-best-match \
    "neutts[all]" soundfile tqdm

step "3. warm the HF cache: backbone + NeuCodec"
"${VENV}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for m in ["neuphonic/neutts-2e", "neuphonic/neucodec"]:
    snapshot_download(m); print("cached", m)
import torch; print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
from neutts import NeuTTS2E  # noqa: F401
print("NeuTTS2E import OK")
PY

cat <<EOF

NeuTTS-2E venv ready at ${VENV}.
Generate with (English only):
  bash run_generate_neutts.sh
  bash run_generate_neutts.sh --limit 3    # smoke test
EOF
