#!/usr/bin/env bash
# One-shot reproducible install for the CosyVoice3 (native upstream) generation venv.
#
# We run the official FunAudioLLM/CosyVoice code IN-PROCESS rather than through
# vLLM-Omni (the vllm-omni cosyvoice3 path was too slow). Like the Echo/Kokoro
# recipes it gets its own venv so its torch==2.3.1/cu121 stack never collides with
# the main repo's vllm pins. Model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (HF-cached).
#
# Idempotent: re-running re-checks each step. Pass --force to nuke the venv.
#
#   bash scripts/setup/setup_cosyvoice_env.sh
#   COSYVOICE_VENV=/some/path bash scripts/setup/setup_cosyvoice_env.sh   # override venv location
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COSYVOICE_VENV="${COSYVOICE_VENV:-$HOME/.venvs/cosyvoice}"
SRC="${COSYVOICE_SRC:-${REPO_ROOT}/third_party/CosyVoice}"
PY_VER="${PY_VER:-3.10}"
COSYVOICE_REPO="${COSYVOICE_REPO:-https://github.com/FunAudioLLM/CosyVoice}"
COSYVOICE_COMMIT="${COSYVOICE_COMMIT:-074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc}"

. "${REPO_ROOT}/scripts/setup/lib_setup.sh"
setup_force_from_args "$@"

setup_step "1. CosyVoice (+ Matcha-TTS submodule) @ ${COSYVOICE_COMMIT:0:10} -> ${SRC}"
setup_checkout "${COSYVOICE_REPO}" "${SRC}" "${COSYVOICE_COMMIT}" --recursive

setup_step "2. uv venv ${COSYVOICE_VENV} (python ${PY_VER})"
setup_venv "${COSYVOICE_VENV}" "${PY_VER}"
PY="${COSYVOICE_VENV}/bin/python"

setup_step "3. install CosyVoice requirements (minus heavy/optional accel + openai-whisper)"
# requirements.txt carries multiple --extra-index-url lines (pytorch cu121 +
# onnxruntime), so uv needs --index-strategy unsafe-best-match to resolve torch==2.3.1.
# Drop tensorrt/deepspeed/vllm (only needed for accel we don't use) and openai-whisper
# (its sdist build lacks a pkg_resources build-dep and aborts the whole install; we
# install it separately below with --no-build-isolation).
grep -viE "tensorrt|deepspeed|^vllm|openai-whisper|^# " "${SRC}/requirements.txt" \
    > "${COSYVOICE_VENV}/reqs_min.txt"
uv pip install --python "${PY}" --index-strategy unsafe-best-match \
    -r "${COSYVOICE_VENV}/reqs_min.txt"

setup_step "4. openai-whisper (imported by cosyvoice frontend) + setuptools<81 for pkg_resources"
# CosyVoice's frontend.py imports `whisper` at module load, so it IS required even
# though we don't use its ASR. Build it against the already-installed torch
# (--no-build-isolation). setuptools>=81 dropped pkg_resources, which cosyvoice
# imports at runtime -> pin below 81.
uv pip install --python "${PY}" "setuptools<81" wheel
uv pip install --python "${PY}" --no-build-isolation --index-strategy unsafe-best-match \
    openai-whisper
# pyopenjtalk: ja->katakana for the --katakana flag (CosyVoice3 requires katakana
# input for Japanese; raw kanji gives ~60% WER).
uv pip install --python "${PY}" pyopenjtalk

setup_step "5. smoke check: import CosyVoice3 + torch/onnxruntime CUDA"
PYTHONPATH="${SRC}:${SRC}/third_party/Matcha-TTS" "${PY}" - <<'PY'
import torch, onnxruntime, pkg_resources  # noqa: F401
from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: F401
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("ort providers:", onnxruntime.get_available_providers())
print("CosyVoice3 import OK")
PY

cat <<EOF

CosyVoice3 venv ready at ${COSYVOICE_VENV}.
Model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 downloads to the HF cache on first run.
Generate with:
  bash scripts/gen/run_cosyvoice.sh                          # clean, all langs
  bash scripts/gen/run_cosyvoice.sh --test-set en --limit 3  # smoke test
EOF
