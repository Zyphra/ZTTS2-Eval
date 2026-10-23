#!/usr/bin/env bash
# Run the retained TTSDS2 Pitch distribution metric for the target systems.
#
#   bash scripts/run_ttsds_pitch.sh [TEST_SET] [LANG]
#
# Outputs results/<model>/<TEST_SET>/<LANG>/ttsds_pitch.csv. Existing non-empty
# outputs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

TEST_SET="${1:-zero_shot}"
LANG="${2:-en}"
VENV="${VENV:-$HOME/.venvs/ttsds}"
PY="${VENV}/bin/python"
REF_DIR="data/${TEST_SET}/${LANG}/waveform"

if [ ! -x "${PY}" ]; then
    echo "error: no TTSDS2 venv at ${VENV} -- run 'bash scripts/setup_ttsds_env.sh' first (or set VENV)" >&2
    exit 1
fi
if [ ! -d "${REF_DIR}" ]; then
    echo "missing reference dir ${REF_DIR}" >&2
    exit 1
fi

echo "test_set=${TEST_SET} lang=${LANG} ref=${REF_DIR}"

MODELS=(
    omni_qwen3-tts-1.7b
    omni_fish-s2-pro
    omni_voxcpm2
    zonos2
    zonos2_stability
)

for MODEL in "${MODELS[@]}"; do
    SYNTH_DIR="results/${MODEL}/${TEST_SET}/${LANG}/wavs"
    OUT="results/${MODEL}/${TEST_SET}/${LANG}/ttsds_pitch.csv"

    if [ ! -d "${SYNTH_DIR}" ]; then
        echo "[skip] missing ${SYNTH_DIR}"
        continue
    fi
    if [ -s "${OUT}" ]; then
        echo "[skip] ${MODEL}: ${OUT} already present"
        continue
    fi

    echo "=== ${MODEL} ==="
    "${PY}" utils/score_ttsds_pitch.py \
        --synth-dir "${SYNTH_DIR}" \
        --ref-dir "${REF_DIR}" \
        --label "${MODEL}" \
        --ref-label "ref_${TEST_SET}_${LANG}" \
        --out "${OUT}"
done
