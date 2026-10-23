#!/usr/bin/env bash
# Score generated audio against the FLEURS-R zero-shot eval
# (data/zero_shot/<lang>). Pins task=zero_shot; defaults to en. The script name
# is historical and model-agnostic — it scores any compatible DECODE_DIR.
#
#   bash run_infer_zonos2_eval.sh                              # en
#   bash run_infer_zonos2_eval.sh --test_set "en zh de es fr it ja ko ru"
#   DECODE_DIR=results/zonos2_zero_shot bash run_infer_zonos2_eval.sh
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

task=zero_shot
test_set="en"
# Override task / test_set on the CLI, e.g.:
#   bash run_infer_zonos2_eval.sh --task zero_shot --test_set "en zh de es fr it ja ko ru"
#   bash run_infer_zonos2_eval.sh --task zero_shot_voxblink --test_set "en zh de fr"
# zero_shot languages:          en zh de es fr it ja ko ru
# zero_shot_voxblink languages: en zh de es fr it ja ko ru pt ar hi tr tl pl id th yo nl uk vi

. utils/parse_options.sh || exit 1;
# IMPORTANT: dumpdir / decode_dir reference ${task} AFTER parse_options
# updates it -- otherwise --task overrides silently use the default dumpdir.
dumpdir=data/${task}
decode_dir=${DECODE_DIR:-results/${task}}

# Start the Qwen3-ASR vLLM server; the trap stops it on success, error, or Ctrl-C.
# maybe_start_whisper adds a Whisper server for per-lang routing when ASR_ROUTING=1
# or WHISPER_GPU is set (no-op otherwise -- all-Qwen, unchanged).
. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

# Shared scoring loop (resume gates, SCORE_GPU pinning, per-leaf fail tracking)
# -- one copy for this script and run_infer_voxblink_eval.sh so fixes land once.
. scripts/lib_infer_scoring.sh
pick_score_gpu

for lang in ${test_set}; do
    score_lang_dir "${dumpdir}/${lang}" "${decode_dir}/${lang}" "${lang}"
done

if [ "${SCORE_FAILS}" -gt 0 ]; then
    echo "[score] ERROR: ${SCORE_FAILS} language(s) had failed scoring steps" 1>&2
    exit 1
fi
