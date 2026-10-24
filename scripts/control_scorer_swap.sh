#!/usr/bin/env bash
# CONTROL (2026-08 methodology review): does Qwen3-ASR favor its own family?
#
# Qwen3-ASR-1.7B is the WER scorer for QWEN_LANGS (en zh ko th vi hi fa) and two
# ranked systems are Qwen3-TTS-0.6b/1.7b -- same model family, plausibly shared
# corpora/tokenizer. This control rescores zero_shot en+zh for the qwen3 systems
# AND a cohort (fish, omnivoice-reftext, voxcpm2, zonos2_accurate, GT) with
# Whisper-large-v3 forced-language instead, writing to ALTERNATE filenames
# (*.whisperctl.*) so published scores are untouched. The readout is
# difference-in-differences: qwen3's (whisper - qwen) WER shift vs the cohort's.
#
# Mechanism: QWEN_LANGS is env-overridable in utils/run_wer.py; shrinking it to
# exclude en/zh routes them to Whisper with the language forced (WHISPER_LANG_CODE
# lists en/zh for exactly this use).
#
#   sbatch --job-name=ztts-ctl-scorer --gpus=1 \
#          --cpus-per-task=28 --mem=200G --time=8:00:00 --chdir="$(pwd)" \
#          --output=logs/slurm/ztts-ctl-scorer-%j.out scripts/control_scorer_swap.sh
#
# NB: set +u -- serve_asr.sh references optional unset vars.
set +u
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[ctl-scorer] ERROR: not at repo root"; exit 1; }

LANGS="${CTL_LANGS:-en zh}"
MODELS="${CTL_MODELS:-omni_qwen3-tts-1.7b omni_qwen3-tts-0.6b omni_fish-s2-pro omni_omnivoice-reftext omni_voxcpm2 zonos2_accurate gt_zero_shot}"

job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 21000 + (job_id % 1000) * 40 ))
export ASR_PORT=$(( port_base + 1 ))
export WHISPER_PORT=$(( port_base + 2 ))
export ASR_GPU=0 WHISPER_GPU=0 ASR_ROUTING=1
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
srv="logs/slurm/servers/ctl-scorer-${job_id}"; mkdir -p "${srv}"
export ASR_LOG="${srv}/asr.log" WHISPER_LOG="${srv}/whisper.log"

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || { echo "[ctl-scorer] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper || exit 1
[ -n "${WHISPER_BASE_URL:-}" ] || { echo "[ctl-scorer] ERROR: WHISPER_BASE_URL unset"; exit 1; }

# The swap: en/zh leave QWEN_LANGS -> forced-language Whisper.
export QWEN_LANGS="ko th vi hi fa yo"
echo "[ctl-scorer] QWEN_LANGS='${QWEN_LANGS}' -> en/zh routed to ${WHISPER_BASE_URL}"

repo="$(pwd -P)"
n_ok=0; n_fail=0
for m in ${MODELS}; do
  for lg in ${LANGS}; do
    # GT baselines are flat (results/gt_zero_shot/<lang>), systems are nested.
    if [ "${m}" = gt_zero_shot ]; then dd="${repo}/results/${m}/${lg}"; else dd="${repo}/results/${m}/zero_shot/${lg}"; fi
    man="${dd}/wav_res_ref_text"
    if [ ! -f "${man}" ]; then echo "[skip] ${m}/${lg} (no manifest)"; continue; fi
    out_raw="${dd}/wav_res_ref_text.whisperctl.raw"
    out_wer="${dd}/wav_res_ref_text.whisperctl.wer"
    if [ -s "${out_wer}" ] && grep -q '^WER:' "${out_wer}" && ! grep -q '^WER: nan' "${out_wer}"; then
      echo "[skip] ${m}/${lg} (already scored)"; n_ok=$((n_ok+1)); continue
    fi
    echo "=== ctl-scorer ${m}/${lg} ==="
    if (cd utils && python3 run_wer.py "${man}" "${out_raw}" "${lg}" \
        && python3 average_wer.py "${out_raw}" "${out_wer}"); then
      n_ok=$((n_ok+1))
    else
      echo "[ctl-scorer] FAIL ${m}/${lg}"; n_fail=$((n_fail+1))
    fi
  done
done
echo "[ctl-scorer] done: ${n_ok} ok, ${n_fail} failed"
[ "${n_fail}" -eq 0 ] && echo "CTL-SCORER-OK"
