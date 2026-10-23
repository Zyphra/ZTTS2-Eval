#!/usr/bin/env bash
# Score generated audio against the VoxBlink2-derived zero-shot eval
# (data/zero_shot_voxblink/<lang>). Mirrors run_infer_zonos2_eval.sh but
# pins task=zero_shot_voxblink. Defaults to the 9 languages that overlap the
# zero_shot set; the noisy voxblink-only tail is disabled by default (see below).
#
#   bash run_infer_voxblink_eval.sh                            # 9 zs-overlap langs
#   bash run_infer_voxblink_eval.sh --test_set "en zh ja"
#   DECODE_DIR=results/zonos2/zero_shot_voxblink bash run_infer_voxblink_eval.sh
#
# Exits non-zero when any language's scoring step failed (after attempting the
# remaining languages), so SLURM jobs surface scoring failures instead of
# silently exiting 0 with broken score files.
#
# -u is deliberately omitted: the sourced helpers (serve_asr.sh,
# parse_options.sh) probe optionally-unset environment variables.
set -eo pipefail

task=zero_shot_voxblink
# Score only the 9 langs shared with zero_shot. The voxblink-only tail
# (pt ar hi tr tl pl id th yo nl uk vi) is noisy -- runaway WER on ar/hi/th/tl/tr -- so it's
# off by default. Re-enable via the full list below or an explicit --test_set.
#   full 21: "en zh de es fr it ja ko ru pt ar hi tr tl pl id th yo nl uk vi"
test_set="en zh de es fr it ja ko ru"
. utils/parse_options.sh || exit 1;

dumpdir=data/${task}
decode_dir=${DECODE_DIR:-results/${task}}

# Explicit venv python for the torch scorers (sfi_utmos etc. are installed there;
# bare `python` may not resolve to the venv in every launch context).
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper   # per-lang routing when ASR_ROUTING=1 / WHISPER_GPU set (else no-op)

# Shared scoring loop (resume gates, SCORE_GPU pinning, per-leaf fail tracking)
# -- one copy for this script and run_infer_zonos2_eval.sh so fixes land once.
. scripts/lib_infer_scoring.sh
pick_score_gpu

for lang in ${test_set}; do
    score_lang_dir "${dumpdir}/${lang}" "${decode_dir}/${lang}" "${lang}"
done

if [ "${SCORE_FAILS}" -gt 0 ]; then
    echo "[score] ERROR: ${SCORE_FAILS} language(s) had failed scoring steps" 1>&2
    exit 1
fi
