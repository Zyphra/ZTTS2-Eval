#!/usr/bin/env bash
# Ground-truth baselines: score REAL human audio with the same pipeline as the systems.
# Output is results/gt_<task>/<lang>/ (default; --out-root overrides), one leaf per language:
#
#   --task clean            (Clean, FLEURS-R)  the real PROMPT clips scored against
#       prompt_text. WER = the ASR error floor on genuine human speech; MSR-UTMOS = the
#       real-audio quality ceiling. Speaker similarity is omitted (prompt vs itself is
#       degenerate ~100). The clean set retains prompt audio but not target audio, so
#       the GT is measured on the prompt clips rather than the target text.
#   --task in-the-wild   (ITW, VoxBlink2)  the real TARGET recordings
#       (data/in-the-wild/<lang>/target_wav.scp) scored against `text`. WER and
#       MSR-UTMOS as above; speaker similarity is prompt vs. real target -- the
#       held-out ceiling of the primary ITW speaker-similarity score.
#
#   bash scripts/score/run_gt.sh                                  # all Clean langs
#   bash scripts/score/run_gt.sh --test-set "en zh"
#   bash scripts/score/run_gt.sh --task in-the-wild        # all ITW langs
#
# Scoring is scripts/score/lib_infer_scoring.sh score_lang_dir in its gt_clean / gt_itw
# mode: each score kind of a leaf is rescored only when it is not current
# (utils/benchmark.py stale_kinds). Exits non-zero when any leaf had a failed step.
set +u
cd "$(dirname "$0")/../.." || exit 1

task=clean
test_set=""   # empty -> the task's basket, resolved after option parsing
out_root=""   # empty -> results/gt_<task>
help_message="usage: bash scripts/score/run_gt.sh [--task clean|in-the-wild] [--test-set \"...\"] [--out-root D]"
. utils/parse_options.sh || exit 1
case "${task}" in
    clean) mode=gt_clean ;;
    in-the-wild) mode=gt_itw ;;
    *) echo "error: --task must be clean or in-the-wild (got '${task}')" 1>&2; exit 1 ;;
esac
[ -n "${test_set}" ] || test_set=$(python3 -m utils.benchmark langs "${task}")
[ -n "${out_root}" ] || out_root=results/gt_${task}

. utils/lib_common.sh || { echo "error: run from the repo root" 1>&2; exit 1; }
repo_py

. scripts/score/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper || exit 1

. scripts/score/lib_infer_scoring.sh
pick_score_gpu

for lang in ${test_set}; do
    data=data/${task}/${lang}
    sd=${out_root}/${lang}
    [ -f "${data}/prompt_wav.scp" ] || { echo "[skip] ${lang} (no prompt_wav.scp)"; continue; }
    if [ "${task}" = "in-the-wild" ]; then
        [ -f "${data}/target_wav.scp" ] \
            || { echo "[skip] ${lang} (no target_wav.scp; run scripts/prep/build_voxblink_targets.py)"; continue; }
    fi
    echo "=== GT ${task}/${lang} ==="
    score_lang_dir "${data}" "${sd}" "${lang}" "${mode}"
    echo "  ${lang}: $(grep -h '^WER:' ${sd}/wav_res_ref_text.wer 2>/dev/null) | MSR=$(cat ${sd}/msr_utmos_mean.txt 2>/dev/null)"
done
stop_asr_server; stop_whisper_server; trap - EXIT INT TERM
echo "=== GT_${task} DONE -> ${out_root} (${SCORE_FAILS} failed step(s)) ==="
[ "${SCORE_FAILS}" -eq 0 ]
