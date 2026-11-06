#!/usr/bin/env bash
# One-shot reproducible install for the Kokoro-82M generation venv.
#
# Kokoro (hexgrad/Kokoro-82M) is a small in-process TTS model. It gets its own venv
# (like the Echo and TTSDS scorer venvs) so its torch/misaki stack never collides
# with the main repo's vllm pins. The model is loaded in-process by
# scripts/gen/generate_kokoro.py; there is no server.
#
# misaki (Kokoro's G2P) ships its own espeak-ng via `espeakng-loader`, so NO system
# espeak-ng install is needed. The ja/zh extras pull language-specific G2P deps.
#
# Idempotent: re-running re-checks each step. Pass --force to nuke the venv.
#
#   bash scripts/setup/setup_kokoro_env.sh
#   VENV=/some/path bash scripts/setup/setup_kokoro_env.sh   # override venv location
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/kokoro}"
PY_VER="${PY_VER:-3.11}"
# CUDA wheels for torch; match the host driver (cu128, as the ZONOS2 checkout uses).
# Set CUDA_TAG=cpu for a CPU-only install (Kokoro is tiny and runs fine on CPU).
CUDA_TAG="${CUDA_TAG:-cu128}"
KOKORO_SPEC="${KOKORO_SPEC:-kokoro>=0.9.4}"

if [ "${1:-}" = "--force" ]; then
    rm -rf "$VENV"
fi

step() { printf "\n=== %s ===\n" "$*"; }

step "1. uv venv ${VENV} (python ${PY_VER})"
if [ ! -x "${VENV}/bin/python" ]; then
    uv venv --python "${PY_VER}" "${VENV}"
fi

step "2. install torch/torchaudio (${CUDA_TAG})"
if [ "${CUDA_TAG}" = "cpu" ]; then
    uv pip install --python "${VENV}/bin/python" torch torchaudio
else
    uv pip install --python "${VENV}/bin/python" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
        torch torchaudio
fi

step "3. install kokoro + misaki[en,ja,zh] + soundfile + tqdm"
# misaki[ja] needs the ja G2P (fugashi/jaconv); misaki[zh] needs the zh G2P.
# soundfile writes the output wavs; tqdm is the progress bar.
uv pip install --python "${VENV}/bin/python" \
    "${KOKORO_SPEC}" "misaki[en,ja,zh]" soundfile tqdm

step "3b. language G2P data: en spaCy model + Japanese unidic dictionary"
# misaki[en] tokenises English with spaCy's en_core_web_sm. It will auto-download it
# on first use, but that download can't be picked up by the SAME process -- so install
# it explicitly here (pinned wheel, offline-reproducible) before the warmup runs.
uv pip install --python "${VENV}/bin/python" \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# misaki[ja] uses fugashi+MeCab, which needs the full unidic dictionary downloaded
# into the venv (unidic ships empty). ~250 MB, one-time; persists in the venv so
# offline compute nodes are fine. Non-fatal: if it fails, only Japanese is affected.
"${VENV}/bin/python" -m unidic download || \
    echo "[warn] unidic download failed -- Japanese (lang_code j) will not work" 1>&2

step "4. warm the HF cache: model + voicepacks for every supported language"
# Pre-download hexgrad/Kokoro-82M weights and each language's default voicepack so
# runs on offline compute nodes don't try to reach HuggingFace. Also validates each
# language's G2P deps by synthesising one short phrase.
"${VENV}/bin/python" - <<'PY'
import numpy as np
from kokoro import KPipeline

# (lang_code, voice, probe text) for every language this recipe maps.
PROBES = [
    ("a", "af_heart",   "Hello, this is a test."),
    ("b", "bf_emma",    "Hello, this is a test."),
    ("e", "ef_dora",    "Hola, esto es una prueba."),
    ("f", "ff_siwis",   "Bonjour, ceci est un test."),
    ("i", "if_sara",    "Ciao, questo e un test."),
    ("p", "pf_dora",    "Ola, isto e um teste."),
    ("h", "hf_alpha",   "namaste"),
    ("j", "jf_alpha",   "こんにちは"),
    ("z", "zf_xiaobei", "你好"),
]
for lang, voice, text in PROBES:
    try:
        pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device="cpu")
        chunks = [a for _, _, a in pipe(text, voice=voice)]
        n = int(sum(np.asarray(getattr(a, "cpu", lambda: a)()).size for a in chunks)) if chunks else 0
        print(f"  lang={lang} voice={voice}: OK ({n} samples)")
    except Exception as e:
        print(f"  lang={lang} voice={voice}: FAILED -- {type(e).__name__}: {e}")
PY

cat <<EOF

Kokoro venv ready at ${VENV}.
Generate with:
  bash scripts/gen/run_kokoro.sh                       # supported langs, both tasks
  bash scripts/gen/run_kokoro.sh --test_set en --limit 3   # smoke test
EOF
