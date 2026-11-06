#!/usr/bin/env bash
# Score generated audio against the VoxBlink2-derived zero-shot eval
# (data/in-the-wild/<lang>). Mirrors run_infer_zonos2_eval.sh but
# pins task=in-the-wild. Defaults to the frozen ITW basket (utils/benchmark.py,
# `python3 -m utils.benchmark langs in-the-wild`).
#
#   bash run_infer_voxblink_eval.sh                            # all ITW langs
#   bash run_infer_voxblink_eval.sh --test_set "en zh ja"
#   DECODE_DIR=results/zonos2/in-the-wild bash run_infer_voxblink_eval.sh
#
# Exits non-zero when any language's scoring step failed (after attempting the
# remaining languages), so SLURM jobs surface scoring failures instead of
# silently exiting 0 with broken score files.
#
# -u is deliberately omitted: the sourced helpers (serve_asr.sh,
# parse_options.sh) probe optionally-unset environment variables.
set -eo pipefail

task=in-the-wild
test_set=""   # empty -> the task's basket, resolved after option parsing
. utils/parse_options.sh || exit 1;
[ -n "${test_set}" ] || test_set=$(python3 -m utils.benchmark langs "${task}")

dumpdir=data/${task}
decode_dir=${DECODE_DIR:-results/${task}}

# Explicit venv python for the torch scorers (sfi_utmos etc. are installed there;
# bare `python` may not resolve to the venv in every launch context).
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper || exit 1

# Shared scoring loop (resume gates, SCORE_GPU pinning, per-leaf fail tracking)
# -- one copy for this script and run_infer_zonos2_eval.sh so fixes land once.
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
