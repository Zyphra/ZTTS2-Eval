#!/usr/bin/env bash
# Multi-seed gen + DS-WED prosody-diversity scoring for a vLLM-Omni TTS model.
#
#   bash scripts/run_dswed_omni.sh <model-key> <gpu-idx> <port> [noreftext] [testset]
#
# testset is <task>/<lang> (default zero_shot/en), e.g. zero_shot_voxblink/en.
# K=3 seeds * min(200, set size) utts. Writes
# results/omni_<model>/dswed/zero_shot/en/seed{0,1,2}/wavs/<uid>.wav and a
# corpus-mean DSWED file. Starts (and stops) the vLLM-Omni server itself.
#
# Pass a 4th arg "noreftext" for the audio-only ablation (omit the reference
# transcript): voxcpm2 -> --no-ref-text, qwen3-tts-* -> --no-ref-text
# --x-vector-only. Output then lands under results/omni_<model>_noreftext/dswed.
set -e
cd "$(dirname "$0")/.."

MODEL_KEY="${1:?usage: run_dswed_omni.sh <model-key> <gpu> <port> [noreftext]}"
GPU="${2:?gpu idx required}"
PORT="${3:?port required}"
NOREF="${4:-}"
TESTSET="${5:-zero_shot/en}"   # <task>/<lang>, e.g. zero_shot_voxblink/en

PY=.venv/bin/python
DATA=data/${TESTSET}
LANG="$(basename "${TESTSET}")"
LIMIT=200
WORKERS=8
# Expected utt count = min(LIMIT, dataset size); used for skip-gen (voxblink sets
# are smaller than LIMIT, so comparing against LIMIT alone would never skip).
DSIZE=$(wc -l < "${DATA}/text" 2>/dev/null || echo 0)
EXPECT=${LIMIT}
{ [ "${LIMIT}" -eq 0 ] || [ "${DSIZE}" -lt "${LIMIT}" ]; } && EXPECT=${DSIZE}

# --- audio-only (no ref_text) ablation flags -------------------------------
GEN_FLAGS=""
SUFFIX=""
LOGTAG="dswed"
if [ -n "${NOREF}" ]; then
    [ "${NOREF}" = "noreftext" ] || { echo "4th arg must be 'noreftext' (got '${NOREF}')"; exit 1; }
    SUFFIX="_noreftext"
    LOGTAG="dswed_noref"
    case "${MODEL_KEY}" in
        qwen3-tts-*) GEN_FLAGS="--no-ref-text --x-vector-only" ;;  # Qwen3 Base needs x_vector_only
        *)           GEN_FLAGS="--no-ref-text" ;;
    esac
fi
ROOT=results/omni_${MODEL_KEY}${SUFFIX}/dswed/${TESTSET}

# --- model registry (copied from run_generate_omni.sh) ---------------------
model_spec() {
    case "$1" in
        qwen3-tts-1.7b)      echo "Qwen/Qwen3-TTS-12Hz-1.7B-Base|clone|qwen3_tts.yaml" ;;
        qwen3-tts-0.6b)      echo "Qwen/Qwen3-TTS-12Hz-0.6B-Base|clone|qwen3_tts.yaml" ;;
        cosyvoice3)          echo "FunAudioLLM/Fun-CosyVoice3-0.5B-2512|clone|cosyvoice3.yaml" ;;
        voxcpm2)             echo "openbmb/VoxCPM2|clone|voxcpm2.yaml" ;;
        voxtral-tts)         echo "mistralai/Voxtral-4B-TTS-2603|preset|voxtral_tts.yaml" ;;
        moss-tts-nano)       echo "OpenMOSS-Team/MOSS-TTS-Nano|clone|moss_tts_nano.yaml" ;;
        fish-s2-pro)         echo "fishaudio/s2-pro|clone|fish_qwen3_omni.yaml" ;;
        *) echo "" ;;
    esac
}
SPEC=$(model_spec "${MODEL_KEY}")
[ -z "${SPEC}" ] && { echo "unknown model ${MODEL_KEY}"; exit 1; }
MODEL_ID="${SPEC%%|*}"
MODE="$(echo "${SPEC}" | cut -d'|' -f2)"
DEPLOY_YAML="${SPEC##*|}"

REPO_ROOT="$(pwd -P)"
# find_spec resolves the path without importing the package: an import runs
# vLLM's start-up logging, which goes to stdout and would land in DEPLOY_DIR.
DEPLOY_DIR="$(${PY} -c 'import importlib.util,os;print(os.path.join(os.path.dirname(importlib.util.find_spec("vllm_omni").origin),"deploy"))')"
SERVE_ARGS="--allowed-local-media-path ${REPO_ROOT} --deploy-config ${DEPLOY_DIR}/${DEPLOY_YAML}"

# Point serve_omni.sh at the in-venv vllm-omni binary (otherwise PATH lookup fails).
if [ -x .venv/bin/vllm-omni ]; then
    export OMNI_VLLM_BIN=.venv/bin/vllm-omni
fi

# --- start vLLM-Omni server ------------------------------------------------
export OMNI_HOST=127.0.0.1
export OMNI_PORT="${PORT}"
export OMNI_GPU="${GPU}"
export OMNI_MODEL="${MODEL_ID}"
export OMNI_SERVE_ARGS="${SERVE_ARGS}"
export OMNI_LOG="/tmp/vllm_omni_${MODEL_KEY}_${LOGTAG}.log"
. scripts/serve_omni.sh
trap 'stop_omni_server' EXIT INT TERM
start_omni_server || { echo "server failed; see ${OMNI_LOG}"; exit 1; }

# --- K=3 seeds gen ---------------------------------------------------------
mkdir -p "${ROOT}"
for s in 0 1 2; do
    OUT="${ROOT}/seed${s}"
    have=$(ls -1 ${OUT}/wavs/*.wav 2>/dev/null | wc -l)
    if [ "${EXPECT}" -gt 0 ] && [ "${have}" -ge "${EXPECT}" ]; then
        echo "[skip-gen] seed=${s} already has ${have} wavs at ${OUT}/wavs"
        continue
    fi
    echo "=== [gen] ${MODEL_KEY} seed=${s} -> ${OUT} ==="
    "${PY}" scripts/generate_omni.py \
        --server "http://127.0.0.1:${PORT}" \
        --mode "${MODE}" \
        --model "${MODEL_ID}" \
        --data-dir "${DATA}" \
        --out-dir "${OUT}" \
        --language "${LANG}" \
        --seed "${s}" \
        --temperature 0.8 \
        --limit "${LIMIT}" \
        --workers "${WORKERS}" \
        ${GEN_FLAGS}
done

# --- DS-WED scoring (HuBERT runs on the GPU -- pin it to the same card) ----
echo "=== [dswed] ${MODEL_KEY}${SUFFIX} scoring ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/score_dswed.py \
    --seed-dirs "${ROOT}/seed0" "${ROOT}/seed1" "${ROOT}/seed2" \
    --out "${ROOT}/dswed_per_uid.tsv" \
    --mean-out "${ROOT}/dswed_mean.txt"

echo "=== [done] ${ROOT}/dswed_mean.txt ==="
cat "${ROOT}/dswed_mean.txt"
