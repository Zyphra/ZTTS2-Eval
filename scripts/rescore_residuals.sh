#!/usr/bin/env bash
# Backfill every utterance the WER scorer never returned a row for (2026-09-01).
#
# Motivation and semantics live in scripts/wer_residuals.py and the long note in
# utils/run_wer.py. In one line: leaf means were survivor means, so a system whose
# output the reference ASR could not read got scored only on the part it could.
# Each leaf's RESIDUAL is re-measured here under today's uniform policy (current
# retries, loop guard, per-language routing) and only what still fails to
# transcribe is scored as maximum error, by run_wer.py itself.
#
# One array task processes one shard of leaves, prepared by scripts/shard_residuals.py:
#
#   .venv/bin/python scripts/shard_residuals.py --shards 8
#   sbatch --job-name=ztts-residuals --gpus=1 \
#          --cpus-per-task=28 --mem=200G --time=10:00:00 --array=0-7%8 \
#          --chdir="$(pwd)" --output=logs/slurm/ztts-residuals-%A_%a.out \
#          scripts/rescore_residuals.sh
#
# Re-runnable: a leaf with no residual left is skipped, and the merge never
# duplicates or overwrites an existing row.
#
# NB: set +u -- serve_asr.sh references optional unset vars.
set +u
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[residuals] ERROR: not at repo root"; exit 1; }

repo="$(pwd -P)"
PY="${repo}/.venv/bin/python"
shard="${RESIDUAL_SHARD:-logs/slurm/residual_shards/shard_${SLURM_ARRAY_TASK_ID:-0}.txt}"
[ -f "${shard}" ] || { echo "[residuals] ERROR: no shard file ${shard}"; exit 1; }

job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 23000 + (job_id % 1000) * 40 ))
export ASR_PORT=$(( port_base + 1 ))
export WHISPER_PORT=$(( port_base + 2 ))
export ASR_GPU=0 WHISPER_GPU=0 ASR_ROUTING=1
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
srv="logs/slurm/servers/residuals-${job_id}"; mkdir -p "${srv}"
export ASR_LOG="${srv}/asr.log" WHISPER_LOG="${srv}/whisper.log"

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || { echo "[residuals] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper || exit 1
[ -n "${WHISPER_BASE_URL:-}" ] || { echo "[residuals] ERROR: WHISPER_BASE_URL unset"; exit 1; }
# QWEN_LANGS is left at the run_wer.py default so the backfill routes each language
# exactly the way the published scores are routed (en on Whisper since control C1).
echo "[residuals] shard=${shard} routing default; whisper=${WHISPER_BASE_URL}"

work="${repo}/logs/slurm/residual_shards/work-${job_id}"; mkdir -p "${work}"
n_ok=0; n_skip=0; n_fail=0
while read -r leaf; do
  case "${leaf}" in ""|\#*) continue ;; esac
  lg="$(basename "${leaf}")"
  tag="$(echo "${leaf}" | tr '/' '_')"
  res_man="${work}/${tag}.manifest"
  res_raw="${work}/${tag}.raw"

  if ! "${PY}" scripts/wer_residuals.py manifest "${leaf}" "${res_man}" --require; then
    n_skip=$((n_skip+1)); continue
  fi

  echo "=== residuals ${leaf} (${lg}) ==="
  if ! (cd utils && "${PY}" run_wer.py "${res_man}" "${res_raw}" "${lg}"); then
    echo "[residuals] FAIL scoring ${leaf}"; n_fail=$((n_fail+1)); continue
  fi
  if ! "${PY}" scripts/wer_residuals.py merge "${leaf}" "${res_raw}"; then
    echo "[residuals] FAIL merging ${leaf}"; n_fail=$((n_fail+1)); continue
  fi
  # Re-aggregate. A leaf that is now complete but heavily imputed is a legitimate
  # (bad) score, not an untrustworthy one, so the coverage gate is expected to pass
  # here; average_wer.py prints the imputed share and writes it to the footer.
  if ! (cd utils && "${PY}" average_wer.py \
        "${repo}/${leaf}/wav_res_ref_text.wer.raw" \
        "${repo}/${leaf}/wav_res_ref_text.wer"); then
    echo "[residuals] FAIL aggregating ${leaf}"; n_fail=$((n_fail+1)); continue
  fi
  n_ok=$((n_ok+1))
done < "${shard}"

echo "[residuals] done: ${n_ok} backfilled, ${n_skip} already complete, ${n_fail} failed"
[ "${n_fail}" -eq 0 ] && echo "RESIDUALS-OK"
