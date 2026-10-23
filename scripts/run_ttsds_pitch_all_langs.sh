#!/usr/bin/env bash
# Score TTSDS2 Pitch for every configured (test set, language) pair.
# Each non-empty ttsds_pitch.csv is skipped. Pitch extraction is CPU-only.
set -euo pipefail
cd "$(dirname "$0")/.."

ZERO_SHOT_LANGS=(en de es fr it ja ko ru zh)
VOXBLINK_LANGS=(en ar de es fr hi id it ja ko pl pt ru th tl tr zh)
LOG_DIR=logs/ttsds_pitch_all_langs
mkdir -p "${LOG_DIR}"

run_pair() {
    local set_name="$1" lang="$2"
    local log="${LOG_DIR}/${set_name}_${lang}.log"
    echo "[$(date -Is)] === ${set_name}/${lang} ==="
    if ! bash scripts/run_ttsds_pitch.sh "${set_name}" "${lang}" >"${log}" 2>&1; then
        echo "[$(date -Is)] FAIL ${set_name}/${lang} (see ${log})"
        return 1
    fi
    echo "[$(date -Is)] OK   ${set_name}/${lang}"
}

for LANG in "${ZERO_SHOT_LANGS[@]}"; do
    run_pair zero_shot "${LANG}" || true
done
for LANG in "${VOXBLINK_LANGS[@]}"; do
    run_pair zero_shot_voxblink "${LANG}" || true
done

echo "[$(date -Is)] all done"
