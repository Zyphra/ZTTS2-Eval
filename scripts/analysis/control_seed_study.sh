#!/usr/bin/env bash
# CONTROL (2026-08 methodology review): generation sampling variance.
#
# Every published score is ONE generation per prompt (zonos2: fixed seed 0;
# vLLM-Omni-served systems: unseeded server-default sampling), so all bootstrap
# CIs condition on the realized draws and exclude run-to-run generation variance.
# This job regenerates one system under its PUBLISHED sampling config on the
# in-the-wild en+tr sets and scores it, so the run-to-run sd of the per-language
# means can be compared against published rank margins (itw top-4 gaps ~0.26-0.32
# WER_CLIP). Repeat runs land under results/_seeds/ (underscore dirs are invisible
# to results discovery); published leaves are untouched.
#
#   zonos2_accurate       seeds 1,2 vs the published seed 0 (deterministic config)
#   omnivoice-reftext / voxcpm2 / fish-s2-pro
#                         fresh unseeded draws, exactly the published config
#
# One (system, run) per job:
#   CTL_SYSTEM=voxcpm2 CTL_RUN=1 sbatch --job-name=ztts-seed-voxcpm2-1 \
#     --gpus=1 --cpus-per-task=28 --mem=200G \
#     --time=24:00:00 --chdir="$(pwd)" --output=logs/slurm/ztts-seed-%j.out \
#     --export=ALL,CTL_SYSTEM=voxcpm2,CTL_RUN=1 scripts/analysis/control_seed_study.sh
set -e
[ -f scripts/score/serve_asr.sh ] || cd "$(dirname "$0")/../.." || exit 1

SYSTEM="${CTL_SYSTEM:?set CTL_SYSTEM (zonos2_accurate|omnivoice-reftext|voxcpm2|fish-s2-pro)}"
RUN="${CTL_RUN:?set CTL_RUN (1|2)}"
LANGS="${CTL_LANGS:-en tr}"
TASK=in-the-wild
ROOT="results/_seeds/run${RUN}"
. slurm/lib_slurm.sh
job_id="${SLURM_JOB_ID:-$$}"
port_base=$(slurm_port_base 0)
port=$(slurm_role_port "${port_base}" tts)

# --- 1. generate under the published config --------------------------------
if [ "${SYSTEM}" = zonos2_accurate ]; then
    leaf_root="${ROOT}/zonos2_accurate/${TASK}"
    # Published run used --seed 0; runs 1/2 use seeds 1/2 (same temp/topk/min_p).
    bash scripts/gen/run_zonos2.sh --task "${TASK}" --test-set "${LANGS}" \
        --out-root "${ROOT}/zonos2_accurate" --gpu 0 --port "${port}" \
        --accurate-mode true --gen-args "--seed ${RUN}"
else
    leaf_root="${ROOT}/omni_${SYSTEM}/${TASK}"
    # No seed / no temperature: the published config is unseeded server defaults,
    # so a fresh run IS the sampling-variance probe. (If a backend turns out to be
    # deterministic, run-vs-published will be ~identical — that is a finding, not
    # a failure.)
    bash scripts/gen/run_omni.sh --model "${SYSTEM}" --out-root "${ROOT}/omni_${SYSTEM}" \
        --task "${TASK}" --test-set "${LANGS}" --gpu 0 --port "${port}"
fi

# --- 2. score (ASR + Whisper co-located on the same GPU) -------------------
set +u   # serve_asr.sh references optional unset vars
export ASR_PORT=$(slurm_role_port "${port_base}" asr) WHISPER_PORT=$(slurm_role_port "${port_base}" whisper)
export ASR_GPU=0 WHISPER_GPU=0 SCORE_GPU=0
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
export SERVE_LOG_DIR=$(slurm_server_log_dir "seed-${job_id}")
. scripts/score/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || { echo "[seed-study] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper || exit 1

PY=.venv/bin/python
. scripts/score/lib_infer_scoring.sh
for lg in ${LANGS}; do
    score_lang_dir "data/${TASK}/${lg}" "${leaf_root}/${lg}" "${lg}"
done
echo "[seed-study] ${SYSTEM} run${RUN}: SCORE_FAILS=${SCORE_FAILS:-0}"
[ "${SCORE_FAILS:-0}" -eq 0 ] && echo "CTL-SEED-OK"
