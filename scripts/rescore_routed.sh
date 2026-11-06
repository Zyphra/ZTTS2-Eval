#!/usr/bin/env bash
# Re-score existing synth WER with per-language forced-ASR routing (ASR_ROUTING=1),
# matching how the GT baselines were scored. Fixes the langs Qwen3-ASR auto-detect
# misroutes to a phonetic neighbour (uk->Czech, bg->Russian, ca->Spanish, nb->Danish,
# sk->Czech, hr->Serbian), which inflated synth WER to 50-99% while GT stayed low.
#
# WER-only: deletes each leaf's wav_res_ref_text.wer/.raw and re-runs the WER scorer
# with a Whisper large-v3 server up. Speaker-sim / MSR-UTMOS are language-agnostic and
# left untouched. Starts ONE ASR + ONE Whisper server (both on the cgroup GPU 0) and
# loops all (model, lang) pairs through it.
#
#   sbatch --job-name=ztts-rescore --gpus=1 \
#          --cpus-per-task=28 --mem=200G --time=12:00:00 \
#          --chdir="$(pwd)" --output=logs/slurm/ztts-rescore-%j.out scripts/rescore_routed.sh
#
# Override the defaults via env: RESCORE_LANGS="bg uk", RESCORE_MODELS="cosyvoice3".
# NB: set +u -- serve_asr.sh references optional unset vars (ASR_EXTRA_ARGS etc.),
# same as scripts/run_gt.sh.
set +u
# Under sbatch the script is copied to spool, so $0 is not the repo path; the job is
# already at the repo root via `sbatch --chdir`. Only fall back to $0-relative cd when
# run directly and not already at the root.
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[rescore] ERROR: not at repo root"; exit 1; }

LANGS="${RESCORE_LANGS:-bg ca hr nb sk uk}"
MODELS="${RESCORE_MODELS:-omni_qwen3-tts-1.7b omni_qwen3-tts-0.6b omni_fish-s2-pro omni_voxcpm2 omni_higgs-audio-v3 omni_voxtral-tts omni_omnivoice-reftext zonos2_accurate zonos2_expressive cosyvoice3 moss-tts}"

# Shardable: several copies of this script can run inside one multi-GPU allocation,
# each pinned to its own GPU and its own port window (RESCORE_SHARD / RESCORE_GPU).
# scripts/rescore_routed_all.sh drives that; running the script directly is unchanged.
job_id="${SLURM_JOB_ID:-$$}"
shard="${RESCORE_SHARD:-0}"
port_base=$(( 20000 + (job_id % 1000) * 40 + shard * 4 ))
export ASR_PORT=$(( port_base + 1 ))
export WHISPER_PORT=$(( port_base + 2 ))
export ASR_GPU="${RESCORE_GPU:-0}" WHISPER_GPU="${RESCORE_GPU:-0}" ASR_ROUTING=1
# Co-locate both ASR servers on the single GPU: each defaults to 0.5 mem-util which
# together = 100% and Whisper OOMs at init. 0.35 each (28GB of 80GB) is ample for
# Qwen3-ASR-1.7B + Whisper-large-v3 (small models, KV-light for offline scoring).
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
srv="logs/slurm/servers/rescore-${job_id}-s${shard}"; mkdir -p "${srv}"
export ASR_LOG="${srv}/asr.log" WHISPER_LOG="${srv}/whisper.log"

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || { echo "[rescore] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper || exit 1
[ -n "${WHISPER_BASE_URL:-}" ] || { echo "[rescore] ERROR: WHISPER_BASE_URL unset -- routing OFF, aborting"; exit 1; }
echo "[rescore] routing via ${WHISPER_BASE_URL}; langs='${LANGS}'"

n_ok=0; n_skip=0
for m in ${MODELS}; do
  for lg in ${LANGS}; do
    dd="results/${m}/clean/${lg}"
    if [ ! -d "${dd}/wavs" ]; then echo "[skip] ${m}/${lg} (no wavs)"; n_skip=$((n_skip+1)); continue; fi
    txt="data/clean/${lg}/text"
    if [ ! -f "${txt}" ]; then echo "[skip] ${m}/${lg} (no ref text)"; n_skip=$((n_skip+1)); continue; fi
    rm -f "${dd}/wav_res_ref_text.wer" "${dd}/wav_res_ref_text.wer.raw"
    echo "=== rescore ${m}/${lg} ==="
    bash scripts/run_score_wer.sh "${txt}" "${dd}" "${lg}" && n_ok=$((n_ok+1))
  done
done
echo "[rescore] done: ${n_ok} rescored, ${n_skip} skipped"
