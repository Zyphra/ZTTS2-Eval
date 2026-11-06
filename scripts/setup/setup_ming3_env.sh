#!/usr/bin/env bash
# One-shot reproducible install for the Ming-3.0-tts (native upstream) generation venvs.
#
# inclusionAI/Ming-3.0-tts ships two checkpoints with separate reference stacks, so each
# gets its own venv, built from the upstream requirement pins:
#   0.5b (dense)            requirements.txt       torch 2.6.0 (cu124), transformers 4.46.3
#   tiny (7.9B-A1.3B MoE)   requirements-tiny.txt  torch 2.7.1+cu118, fla-core (Python 3.12; see step 3)
# The code runs IN-PROCESS from a checkout pinned to MING3_COMMIT (vllm-omni's ming_tts
# pipeline implements the earlier Ming-omni-tts, not 3.0). Models download to the HF
# cache in step 4, since scripts/gen/run_ming3.sh loads them offline.
#
# Idempotent; pass --force to rebuild both venvs.
#
#   bash scripts/setup/setup_ming3_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${MING3_SRC:-${REPO_ROOT}/third_party/Ming-3.0-tts}"
MING3_REPO="${MING3_REPO:-https://github.com/inclusionAI/Ming-3.0-tts}"
MING3_COMMIT="${MING3_COMMIT:-a39b80660bdc5a4394d5e1d41af6dbfd9c443a54}"
MING3_VENV="${MING3_VENV:-$HOME/.venvs/ming3}"
MING3_TINY_VENV="${MING3_TINY_VENV:-$HOME/.venvs/ming3-tiny}"

. "${REPO_ROOT}/scripts/setup/lib_setup.sh"
setup_force_from_args "$@"

setup_step "1. Ming-3.0-tts @ ${MING3_COMMIT:0:10} -> ${SRC}"
setup_checkout "${MING3_REPO}" "${SRC}" "${MING3_COMMIT}"

setup_step "2. 0.5b venv ${MING3_VENV} (python 3.10, requirements.txt, cu124)"
setup_venv "${MING3_VENV}" 3.10
# librosa: generate_ming3.py --trim-prompt (the *-trim systems).
uv pip install --python "${MING3_VENV}/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    -r "${SRC}/requirements.txt" librosa==0.11.0

# Python 3.12, not the reference 3.10.15: triton 3.3.1's kernel launcher over-decrefs
# None, and on 3.10 every tiny process dies after ~120-150 utterances with "Fatal Python
# error: none_dealloc" (None is immortal from 3.12; fla itself recommends >= 3.11). The
# package pins are unchanged.
setup_step "3. tiny venv ${MING3_TINY_VENV} (python 3.12, requirements-tiny.txt, cu118)"
setup_venv "${MING3_TINY_VENV}" 3.12
uv pip install --python "${MING3_TINY_VENV}/bin/python" \
    --index-strategy unsafe-best-match \
    -r "${SRC}/requirements-tiny.txt" librosa==0.11.0

setup_step "4. models -> HF cache"
"${MING3_VENV}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("inclusionAI/Ming-3.0-tts-0.5b", "inclusionAI/Ming-3.0-tts-tiny"):
    print(repo, snapshot_download(repo))
PY

setup_step "5. smoke check: imports + CUDA"
(cd "${SRC}" && "${MING3_VENV}/bin/python" -c \
    "import torch, ming_tts; print('0.5b torch', torch.__version__, 'cuda?', torch.cuda.is_available())")
(cd "${SRC}" && "${MING3_TINY_VENV}/bin/python" -c \
    "import torch, ming_tiny; print('tiny torch', torch.__version__, 'cuda?', torch.cuda.is_available())")

cat <<EOF

Ming-3.0-tts venvs ready: ${MING3_VENV} (0.5b), ${MING3_TINY_VENV} (tiny). Generate with:
  bash scripts/gen/run_ming3.sh --model ming-3.0-tts-0.5b --test-set en --limit 3
EOF
