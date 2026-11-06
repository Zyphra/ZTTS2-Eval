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
#   bash scripts/setup/setup_moss_tts_env.sh
#   MOSS_TTS_VENV=/some/path bash scripts/setup/setup_moss_tts_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MOSS_TTS_VENV="${MOSS_TTS_VENV:-$HOME/.venvs/moss-tts}"
SRC="${MOSS_TTS_SRC:-${REPO_ROOT}/third_party/MOSS-TTS}"
PY_VER="${PY_VER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu128}"
MOSS_TTS_REPO="${MOSS_TTS_REPO:-https://github.com/OpenMOSS/MOSS-TTS}"
MOSS_TTS_COMMIT="${MOSS_TTS_COMMIT:-934d6826b084c46a0d033402174d5f8ac4ed2519}"

. "${REPO_ROOT}/scripts/setup/lib_setup.sh"
setup_force_from_args "$@"

setup_step "1. MOSS-TTS @ ${MOSS_TTS_COMMIT:0:10} -> ${SRC}"
setup_checkout "${MOSS_TTS_REPO}" "${SRC}" "${MOSS_TTS_COMMIT}"

setup_step "2. uv venv ${MOSS_TTS_VENV} (python ${PY_VER})"
setup_venv "${MOSS_TTS_VENV}" "${PY_VER}"

setup_step "3. install MOSS-TTS + its torch-runtime extra (${CUDA_TAG})"
# The repo's [torch-runtime] extra pins torch==2.9.1+cu128 / transformers==5.0.0 /
# torchcodec + the codec deps. Multiple extra-index-urls -> unsafe-best-match.
uv pip install --python "${MOSS_TTS_VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    --index-strategy unsafe-best-match \
    "${SRC}[torch-runtime]" soundfile sentencepiece
# sentencepiece: the MOSS-TTS-Nano variant's processor needs it (AutoProcessor fails
# to instantiate without it). Harmless for the other variants.

setup_step "4. smoke check: import transformers + torch CUDA (custom code loads at runtime)"
PYTHONPATH="${SRC}" "${MOSS_TTS_VENV}/bin/python" - <<'PY'
import torch, transformers, soundfile, librosa  # noqa: F401
from transformers import AutoModel, AutoProcessor  # noqa: F401
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__, "-- CosyVoice3-style native path OK")
PY

cat <<EOF

MOSS-TTS venv ready at ${MOSS_TTS_VENV}.
Model OpenMOSS-Team/MOSS-TTS (+ MOSS-Audio-Tokenizer) downloads to the HF cache on
first run. Generate with:
  bash scripts/gen/run_moss_tts.sh                          # clean, all 9 langs
  bash scripts/gen/run_moss_tts.sh --test-set en --limit 3  # smoke test
EOF
