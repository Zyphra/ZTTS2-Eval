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
VENV_05B="${MING3_VENV:-$HOME/.venvs/ming3}"
VENV_TINY="${MING3_TINY_VENV:-$HOME/.venvs/ming3-tiny}"

if [ "${1:-}" = "--force" ]; then
    rm -rf "${VENV_05B}" "${VENV_TINY}"
fi

step() { printf "\n=== %s ===\n" "$*"; }

step "1. Ming-3.0-tts @ ${MING3_COMMIT:0:10} -> ${SRC}"
if [ ! -d "${SRC}/.git" ]; then
    mkdir -p "$(dirname "${SRC}")"
    git clone "${MING3_REPO}" "${SRC}"
fi
git -C "${SRC}" fetch -q origin
git -C "${SRC}" checkout -q "${MING3_COMMIT}"

step "2. 0.5b venv ${VENV_05B} (python 3.10, requirements.txt, cu124)"
[ -x "${VENV_05B}/bin/python" ] || uv venv --python 3.10 "${VENV_05B}"
# librosa: generate_ming3.py --trim-prompt (the *-trim systems).
uv pip install --python "${VENV_05B}/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    -r "${SRC}/requirements.txt" librosa==0.11.0

# Python 3.12, not the reference 3.10.15: triton 3.3.1's kernel launcher over-decrefs
# None, and on 3.10 every tiny process dies after ~120-150 utterances with "Fatal Python
# error: none_dealloc" (None is immortal from 3.12; fla itself recommends >= 3.11). The
# package pins are unchanged.
step "3. tiny venv ${VENV_TINY} (python 3.12, requirements-tiny.txt, cu118)"
[ -x "${VENV_TINY}/bin/python" ] || uv venv --python 3.12 "${VENV_TINY}"
uv pip install --python "${VENV_TINY}/bin/python" \
    --index-strategy unsafe-best-match \
    -r "${SRC}/requirements-tiny.txt" librosa==0.11.0

step "4. models -> HF cache"
"${VENV_05B}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("inclusionAI/Ming-3.0-tts-0.5b", "inclusionAI/Ming-3.0-tts-tiny"):
    print(repo, snapshot_download(repo))
PY

step "5. smoke check: imports + CUDA"
(cd "${SRC}" && "${VENV_05B}/bin/python" -c \
    "import torch, ming_tts; print('0.5b torch', torch.__version__, 'cuda?', torch.cuda.is_available())")
(cd "${SRC}" && "${VENV_TINY}/bin/python" -c \
    "import torch, ming_tiny; print('tiny torch', torch.__version__, 'cuda?', torch.cuda.is_available())")

cat <<EOF

Ming-3.0-tts venvs ready: ${VENV_05B} (0.5b), ${VENV_TINY} (tiny). Generate with:
  bash scripts/gen/run_ming3.sh --model ming-3.0-tts-0.5b --test_set en --limit 3
EOF
