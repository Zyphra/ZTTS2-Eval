#!/usr/bin/env bash
# One-shot reproducible install for the MOSS-TTS (native upstream) generation venv.
#
# We run the official OpenMOSS/MOSS-TTS code IN-PROCESS via transformers rather than
# through vLLM-Omni: vllm-omni's moss_tts codec reimplementation is weight-
# incompatible with the published MOSS-Audio-Tokenizer checkpoint (loads 1600/1606,
# missing 6 encoder/decoder proj weights), while the upstream HF processor carries
# its OWN matching audio_tokenizer. Like the Echo/Kokoro/CosyVoice recipes it gets
# its own venv (torch 2.9.1/cu128, transformers 5.0.0). Model OpenMOSS-Team/MOSS-TTS
# (8B MossTTSDelay), HF-cached.
#
# Idempotent; pass --force to nuke the venv.
#
#   bash scripts/setup_moss_tts_env.sh
#   VENV=/some/path bash scripts/setup_moss_tts_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/.venvs/moss-tts}"
SRC="${MOSS_TTS_SRC:-${REPO_ROOT}/third_party/MOSS-TTS}"
PY_VER="${PY_VER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu128}"
MOSS_REPO="${MOSS_REPO:-https://github.com/OpenMOSS/MOSS-TTS}"

if [ "${1:-}" = "--force" ]; then
    rm -rf "$VENV"
fi

step() { printf "\n=== %s ===\n" "$*"; }

step "1. clone MOSS-TTS -> ${SRC}"
if [ ! -d "${SRC}/.git" ]; then
    mkdir -p "$(dirname "${SRC}")"
    git clone --depth 1 "${MOSS_REPO}" "${SRC}"
else
    echo "already cloned"
fi

step "2. uv venv ${VENV} (python ${PY_VER})"
if [ ! -x "${VENV}/bin/python" ]; then
    uv venv --python "${PY_VER}" "${VENV}"
fi

step "3. install MOSS-TTS + its torch-runtime extra (${CUDA_TAG})"
# The repo's [torch-runtime] extra pins torch==2.9.1+cu128 / transformers==5.0.0 /
# torchcodec + the codec deps. Multiple extra-index-urls -> unsafe-best-match.
uv pip install --python "${VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    --index-strategy unsafe-best-match \
    "${SRC}[torch-runtime]" soundfile sentencepiece
# sentencepiece: the MOSS-TTS-Nano variant's processor needs it (AutoProcessor fails
# to instantiate without it). Harmless for the other variants.

step "4. smoke check: import transformers + torch CUDA (custom code loads at runtime)"
PYTHONPATH="${SRC}" "${VENV}/bin/python" - <<'PY'
import torch, transformers, soundfile, librosa  # noqa: F401
from transformers import AutoModel, AutoProcessor  # noqa: F401
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__, "-- CosyVoice3-style native path OK")
PY

cat <<EOF

MOSS-TTS venv ready at ${VENV}.
Model OpenMOSS-Team/MOSS-TTS (+ MOSS-Audio-Tokenizer) downloads to the HF cache on
first run. Generate with:
  bash run_generate_moss_tts.sh                          # zero_shot, all 9 langs
  bash run_generate_moss_tts.sh --test_set en --limit 3  # smoke test
EOF
