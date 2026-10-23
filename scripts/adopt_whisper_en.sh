#!/usr/bin/env bash
# ADOPTION (2026-08-29 scorer-swap control): rescore every `en` leaf with the new
# routing (en -> Whisper-large-v3 forced-language; see utils/run_wer.py QWEN_LANGS).
#
# The control (scripts/control_scorer_swap.sh) showed Qwen3-ASR favors its own
# family on en (qwen3-tts DiD +0.51/+0.70 WER_CLIP vs a 4-system cohort, CIs
# exclude 0) -- a differential bias between evaluated systems, which matters more
# for a ranking benchmark than Qwen's slightly lower common-mode GT floor. zh was
# ns and keeps Qwen. Symmetric: ALL systems + GT are rescored the same way.
#
# Rescores IN PLACE (canonical filenames) from each leaf's existing manifest;
# every en leaf under results/ is discovered dynamically, both tasks + flat GT.
#
#   sbatch --job-name=ztts-adopt-en --gpus=1 \
#          --cpus-per-task=28 --mem=200G --time=10:00:00 --chdir="$(pwd)" \
#          --output=logs/slurm/ztts-adopt-en-%j.out scripts/adopt_whisper_en.sh
set +u   # serve_asr.sh references optional unset vars
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[adopt-en] ERROR: not at repo root"; exit 1; }

job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 23000 + (job_id % 1000) * 40 ))
export ASR_PORT=$(( port_base + 1 )) WHISPER_PORT=$(( port_base + 2 ))
export ASR_GPU=0 WHISPER_GPU=0 ASR_ROUTING=1
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
srv="logs/slurm/servers/adopt-en-${job_id}"; mkdir -p "${srv}"
export ASR_LOG="${srv}/asr.log" WHISPER_LOG="${srv}/whisper.log"

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
# The ASR (Qwen) server still serves any leaf that mixes langs; Whisper does en.
start_asr_server || { echo "[adopt-en] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper
[ -n "${WHISPER_BASE_URL:-}" ] || { echo "[adopt-en] ERROR: WHISPER_BASE_URL unset"; exit 1; }

repo="$(pwd -P)"
# ADOPT_LEAVES overrides discovery (space-separated leaf dirs) -- used to bring
# control/variant trees (e.g. results/_seeds/) onto the same routing, since a run
# scored before the QWEN_LANGS change is not comparable to one scored after.
if [ -n "${ADOPT_LEAVES:-}" ]; then
  leaves="${ADOPT_LEAVES}"
else
  leaves=$(ls -d results/*/zero_shot/en results/*/zero_shot_voxblink/en \
                 results/gt_zero_shot/en results/gt_zero_shot_voxblink/en 2>/dev/null \
           | grep -v "^results/whisper_" | grep -v "^results/_")
fi
n_ok=0; n_fail=0; n_skip=0
for dd in ${leaves}; do
  man="${repo}/${dd}/wav_res_ref_text"
  [ -f "${man}" ] || { echo "[skip] ${dd} (no manifest)"; n_skip=$((n_skip+1)); continue; }
  echo "=== adopt-en ${dd} ==="
  raw="${man}.wer.raw"; wer="${man}.wer"
  rm -f "${raw}" "${wer}" "${raw}.meta"
  if (cd utils && python3 run_wer.py "${man}" "${raw}" en \
      && python3 average_wer.py "${raw}" "${wer}"); then
    n_ok=$((n_ok+1))
  else
    echo "[adopt-en] FAIL ${dd}"; n_fail=$((n_fail+1))
  fi
done
echo "[adopt-en] done: ${n_ok} ok, ${n_fail} failed, ${n_skip} skipped"
[ "${n_fail}" -eq 0 ] && echo "ADOPT-EN-OK"
