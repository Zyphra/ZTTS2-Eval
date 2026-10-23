#!/usr/bin/env bash
# One-shot reproducible install for the VibeVoice-1.5B (Microsoft) venv.
#
# VibeVoice-1.5B is a long-form multi-speaker ZERO-SHOT CLONER (unlike the realtime
# 0.5B, which is preset-voice). English + Chinese only. Runs in-process, own venv.
#
# NOTE: this recipe does NOT clone github.com/microsoft/VibeVoice. Upstream withdrew
# the TTS inference code, so it is VENDORED at third_party/VibeVoice-1.5B and used via
# PYTHONPATH -- read that tree's PROVENANCE.md before touching this. The vendored tree
# ships no packaging metadata on purpose (no `pip install -e`): keeping it off the venv
# means it can never shadow third_party/VibeVoice, which the realtime 0.5B recipe needs
# at a DIFFERENT version of the same module names.
#
# Like the neutts/vibevoice recipes, pin cu128 torch FIRST so the nvidia stack matches
# the CUDA-12.8 driver.
#
# Idempotent; pass --force to nuke the venv.
#   bash scripts/setup_vibevoice15_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/.venvs/vibevoice15}"
SRC="${VIBEVOICE15_SRC:-${REPO_ROOT}/third_party/VibeVoice-1.5B}"
PY_VER="${PY_VER:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu128}"

if [ "${1:-}" = "--force" ]; then rm -rf "$VENV"; fi
step() { printf "\n=== %s ===\n" "$*"; }

step "1. check the vendored inference tree"
if [ ! -f "${SRC}/vibevoice/modular/modeling_vibevoice_inference.py" ]; then
    echo "error: ${SRC} is missing modeling_vibevoice_inference.py" 1>&2
    echo "       (it is vendored, not cloned -- see ${SRC}/PROVENANCE.md)" 1>&2
    exit 1
fi
echo "ok: ${SRC}"

step "2. uv venv ${VENV} (python ${PY_VER})"
[ -x "${VENV}/bin/python" ] || uv venv --python "${PY_VER}" "${VENV}"

step "3. pin cu128 torch FIRST"
uv pip install --python "${VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --index-strategy unsafe-best-match \
    "torch==2.8.0" torchaudio

step "4. runtime deps (transformers pinned to match the realtime recipe)"
# diffusers: dpm_solver.py subclasses its SchedulerMixin/ConfigMixin.
# librosa + soundfile: vibevoice_tokenizer_processor.py loads/saves audio with them.
uv pip install --python "${VENV}/bin/python" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --index-strategy unsafe-best-match \
    "transformers==4.51.3" diffusers accelerate librosa soundfile numpy tqdm

step "5. smoke check (imports the vendored tree via PYTHONPATH, no install)"
PYTHONPATH="${SRC}" "${VENV}/bin/python" - <<'PY'
import torch, transformers
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("inference class:", VibeVoiceForConditionalGenerationInference.__name__)
print("processor:", VibeVoiceProcessor.__name__)
PY

cat <<EOF

done. next:
  bash run_generate_vibevoice15.sh --limit 3            # smoke (zero_shot/en + zh)
  bash slurm/submit_eval.sh --engine vibevoice15 --task zero_shot
EOF
