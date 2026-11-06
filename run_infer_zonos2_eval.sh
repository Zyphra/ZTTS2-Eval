#!/usr/bin/env bash
# Score generated audio against the FLEURS-R zero-shot eval
# (data/clean/<lang>). Pins task=clean; defaults to the frozen Clean basket
# (utils/benchmark.py, `python3 -m utils.benchmark langs clean`). The script name
# is historical and model-agnostic — it scores any compatible DECODE_DIR.
#
#   bash run_infer_zonos2_eval.sh                              # all Clean langs
#   bash run_infer_zonos2_eval.sh --test_set "en zh de es fr it ja ko ru"
#   DECODE_DIR=results/zonos2_clean bash run_infer_zonos2_eval.sh
#
# GPU placement is handled by scripts/serve_asr.sh -- if ASR_GPU is unset it
# auto-picks a GPU with enough free memory. Override with: export ASR_GPU=N.
# Speaker-sim / MSR-UTMOS pick their own GPU the same way (override: SCORE_GPU=N).
#
# Exits non-zero when any language's scoring step failed (after attempting the
# remaining languages), so SLURM jobs surface scoring failures instead of
# silently exiting 0 with broken score files.
#
# -u is deliberately omitted: the sourced helpers (serve_asr.sh,
# parse_options.sh) probe optionally-unset environment variables.
set -eo pipefail

task=clean
test_set=""   # empty -> the task's basket, resolved after option parsing
# Override task / test_set on the CLI, e.g.:
#   bash run_infer_zonos2_eval.sh --task clean --test_set "en zh de es fr it ja ko ru"
#   bash run_infer_zonos2_eval.sh --task clean_mini   # a coreset panel (its own langs)

. utils/parse_options.sh || exit 1;
[ -n "${test_set}" ] || test_set=$(python3 -m utils.benchmark langs "${task}")
# IMPORTANT: dumpdir / decode_dir reference ${task} AFTER parse_options
# updates it -- otherwise --task overrides silently use the default dumpdir.
dumpdir=data/${task}
decode_dir=${DECODE_DIR:-results/${task}}

# Start the Qwen3-ASR vLLM server; the trap stops it on success, error, or Ctrl-C.
# maybe_start_whisper starts the Whisper server that per-language ASR routing needs
# (ASR_ROUTING=0 is a diagnostic all-Qwen mode whose results build_results.py refuses).
. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper || exit 1

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

# Shared scoring loop (resume gates, SCORE_GPU pinning, per-leaf fail tracking)
# -- one copy for this script and run_infer_voxblink_eval.sh so fixes land once.
. scripts/lib_infer_scoring.sh
pick_score_gpu

for lang in ${test_set}; do
    [ -f "${dumpdir}/${lang}/text" ] || { echo "[skip] ${dumpdir}/${lang}: no data"; continue; }
    score_lang_dir "${dumpdir}/${lang}" "${decode_dir}/${lang}" "${lang}"
done

if [ "${SCORE_FAILS}" -gt 0 ]; then
    echo "[score] ERROR: ${SCORE_FAILS} language(s) had failed scoring steps" 1>&2
    exit 1
fi
