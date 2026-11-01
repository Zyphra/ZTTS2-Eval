#!/usr/bin/env bash
# One-shot reproducible install for the Audio8-TTS (SGLang-Omni adapter) server venv.
#
# Audio8-TTS-Realistic-0.6B is served by the upstream optimized path: the
# Edge0-AI/Audio8_TTS sglang_omni/ adapter installed into a pinned sglang-omni
# checkout (paged attention, dynamic batching, CUDA graphs, OpenAI-style
# /v1/audio/speech). Pins follow Audio8_TTS guides/deployment.md: sglang-omni
# @68a5723, SGLang 0.5.8, torch 2.9.1+cu128, transformers 4.57.1, Python 3.12.
# Both checkouts live under gitignored third_party/; the model snapshot goes to the
# HF cache at a pinned revision.
#
# Idempotent; pass --force to nuke the venv.
#
#   bash scripts/setup_audio8_env.sh
#   VENV=/some/path bash scripts/setup_audio8_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/.venvs/audio8-sglang}"
PY_VER="${PY_VER:-3.12}"
AUDIO8_SRC="${AUDIO8_SRC:-${REPO_ROOT}/third_party/Audio8_TTS}"
AUDIO8_COMMIT=4eb54a69113eedc65677cbc09a30aee809bb39ee
SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT:-${REPO_ROOT}/third_party/sglang-omni}"
SGLANG_OMNI_COMMIT=68a572348837f7b004857b4b07993c20ade4c017
AUDIO8_MODEL=Edge0/Audio8-TTS-Realistic-0.6b
AUDIO8_REVISION=1e857568bfd4f02931aca3988ee773a44d2d26d3

if [ "${1:-}" = "--force" ]; then
    rm -rf "$VENV"
fi

step() { printf "\n=== %s ===\n" "$*"; }

# checkout REPO_URL DIR COMMIT: clone once, then always pin DIR to COMMIT.
checkout() {
    if [ ! -d "$2/.git" ]; then
        mkdir -p "$(dirname "$2")"
        git clone "$1" "$2"
    fi
    git -C "$2" fetch -q origin
    git -C "$2" checkout -q "$3"
    echo "$2 @ $(git -C "$2" rev-parse --short HEAD)"
}

step "1. Audio8_TTS -> ${AUDIO8_SRC} @ ${AUDIO8_COMMIT}"
checkout https://github.com/Edge0-AI/Audio8_TTS "${AUDIO8_SRC}" "${AUDIO8_COMMIT}"

step "2. sglang-omni -> ${SGLANG_OMNI_ROOT} @ ${SGLANG_OMNI_COMMIT}"
checkout https://github.com/sgl-project/sglang-omni.git "${SGLANG_OMNI_ROOT}" "${SGLANG_OMNI_COMMIT}"

step "3. uv venv ${VENV} (python ${PY_VER})"
if [ ! -x "${VENV}/bin/python" ]; then
    uv venv --python "${PY_VER}" "${VENV}"
fi

step "4. install sglang-omni (editable; pulls sglang 0.5.8 / torch 2.9.1 / transformers 4.57.1)"
uv pip install --python "${VENV}/bin/python" -e "${SGLANG_OMNI_ROOT}"

step "5. install the Audio8 adapter into sglang-omni"
bash "${AUDIO8_SRC}/sglang_omni/scripts/install_adapter.sh" "${SGLANG_OMNI_ROOT}"

step "6. model snapshot ${AUDIO8_MODEL} @ ${AUDIO8_REVISION} + version check"
model_log=$(mktemp)
AUDIO8_MODEL="${AUDIO8_MODEL}" AUDIO8_REVISION="${AUDIO8_REVISION}" "${VENV}/bin/python" - <<'PY' | tee "${model_log}"
import os

import sglang
import torch
import transformers
from huggingface_hub import snapshot_download

path = snapshot_download(os.environ["AUDIO8_MODEL"], revision=os.environ["AUDIO8_REVISION"])
print("model dir:", path)
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sglang", sglang.__version__)
# guides/deployment.md: transformers 5.x makes the model emit all-zero codes.
assert transformers.__version__.startswith("4."), "transformers must be 4.x"
PY
model_dir=$(sed -n 's/^model dir: //p' "${model_log}")
rm -f "${model_log}"

step "7. adapter import check"
"${VENV}/bin/python" "${AUDIO8_SRC}/sglang_omni/scripts/verify_install.py" --model-path "${model_dir}"

cat <<EOF

Audio8 SGLang-Omni venv ready at ${VENV} (model ${model_dir}).
Generate with:
  bash run_generate_audio8.sh --test_set en --limit 3     # smoke test, own server on GPU 0
  bash run_generate_audio8.sh --task zero_shot_voxblink   # full ITW set
EOF
