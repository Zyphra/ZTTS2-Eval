#!/usr/bin/env bash
# Score generated audio: WER (Qwen3-ASR / routed Whisper), speaker similarity and
# MSR-UTMOS for every language leaf of one system's task directory. Any task works
# (clean, in-the-wild, the coreset panels); it is read from the --decode_dir basename
# unless --task is given, and data/<task>/<lang> supplies the references.
#
#   bash scripts/score/run_score.sh --decode_dir results/<system>/clean              # its whole basket
#   bash scripts/score/run_score.sh --decode_dir results/<system>/in-the-wild --test_set "en zh ja"
#   bash scripts/score/run_score.sh --decode_dir /tmp/runs/x --task clean_mini --test_set en
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --decode_dir  <system results>/<task>, holding <lang>/wavs/       (required)
#   --task        task name (default: basename of --decode_dir)
#   --test_set    languages to score (default: the task's frozen basket,
#                 `python3 -m utils.benchmark langs <task>`)
#
# GPU placement is handled by scripts/score/serve_asr.sh -- if ASR_GPU is unset it
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

decode_dir=""
task=""
test_set=""

help_message="usage: bash scripts/score/run_score.sh --decode_dir <system results>/<task> [--task T] [--test_set \"...\"]"
. utils/parse_options.sh || exit 1

# All validation happens before the ASR servers start.
if [ -z "${decode_dir}" ]; then
    echo "error: --decode_dir is required (<system results>/<task>)" 1>&2
    exit 1
fi
[ -n "${task}" ] || task=$(basename "${decode_dir%/}")
langs=$(python3 -m utils.benchmark langs "${task}" 2>/dev/null) || {
    echo "error: unknown task '${task}' (from --decode_dir basename; pass --task)" 1>&2
    exit 1
}
[ -n "${test_set}" ] || test_set=${langs}
dumpdir=data/${task}

# Start the Qwen3-ASR vLLM server; the trap stops it on success, error, or Ctrl-C.
# maybe_start_whisper starts the Whisper server that per-language ASR routing needs
# (ASR_ROUTING=0 is a diagnostic all-Qwen mode whose results build_results.py refuses).
. scripts/score/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper || exit 1

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

# Shared scoring loop (resume gates, SCORE_GPU pinning, per-leaf fail tracking).
. scripts/score/lib_infer_scoring.sh
pick_score_gpu

for lang in ${test_set}; do
    [ -f "${dumpdir}/${lang}/text" ] || { echo "[skip] ${dumpdir}/${lang}: no data"; continue; }
    score_lang_dir "${dumpdir}/${lang}" "${decode_dir}/${lang}" "${lang}"
done

if [ "${SCORE_FAILS}" -gt 0 ]; then
    echo "[score] ERROR: ${SCORE_FAILS} language(s) had failed scoring steps" 1>&2
    exit 1
fi
