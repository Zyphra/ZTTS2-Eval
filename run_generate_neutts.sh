#!/usr/bin/env bash
# Generate NeuTTS-2E (Neuphonic) audio for the zero-shot benchmark. Runs in-process
# via the dedicated venv from scripts/setup_neutts_env.sh (no server).
#
# NeuTTS-2E is a PRESET-voice model (4 fixed speakers x 7 emotions), NOT a zero-shot
# cloner -- speaker-similarity is not meaningful (WER + MSR-UTMOS are). It is
# ENGLISH ONLY, so only the `en` set is generated. Writes results/neutts-2e/<task>/en/.
#
#   bash run_generate_neutts.sh                       # clean/en (+ voxblink/en via --task)
#   bash run_generate_neutts.sh --limit 3             # smoke test
#   bash run_generate_neutts.sh --speaker emily --emotion neutral
#
# Options (Kaldi-style "--name value"):
#   --task       clean | in-the-wild        (default clean)
#   --speaker    emily|paul|sophie|steven | rotate     (default rotate)
#   --emotion    neutral|happy|sad|angry|...           (default neutral)
#   --out_root   results root                          (default results/neutts-2e)
#   --limit      only the first N utterances           (default 0 = all)
#   --overwrite  regenerate existing wavs              (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

task=clean
test_set=     # NeuTTS is English-only; accepted (the SLURM launcher passes it) but
              # only 'en' is generated. A non-en-containing test_set is a no-op.
speaker=rotate
emotion=neutral
out_root="results/neutts-2e"
limit=0
overwrite=false

help_message="usage: bash run_generate_neutts.sh [--task T] [--speaker S] [--emotion E] [--out_root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

# English only: if a test_set was given and doesn't include en, there's nothing to do.
if [ -n "${test_set}" ] && ! echo " ${test_set} " | grep -q " en "; then
    echo "[neutts-2e] test_set='${test_set}' has no 'en' -- NeuTTS is English-only, nothing to generate" 1>&2
    exit 0
fi

VENV="${VENV:-$HOME/.venvs/neutts}"
PY="${VENV}/bin/python"
if [ ! -x "${PY}" ]; then
    echo "error: no NeuTTS venv at ${VENV} -- run 'bash scripts/setup_neutts_env.sh' first (or set VENV)" 1>&2
    exit 1
fi

# English only.
data_dir="data/${task}/en"
out_dir="${out_root}/${task}/en"
if [ ! -f "${data_dir}/text" ]; then
    echo "error: ${data_dir}/text not found" 1>&2; exit 1
fi
ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

echo "=== [neutts-2e] ${data_dir} -> ${out_dir}  (speaker=${speaker} emotion=${emotion}) ==="
"${PY}" scripts/generate_neutts.py \
    --data-dir "${data_dir}" \
    --out-dir "${out_dir}" \
    --speaker "${speaker}" \
    --emotion "${emotion}" \
    --limit "${limit}" \
    ${ow_opt}

echo
echo "generation done -> ${out_root}/${task}/en/"
echo "score with:  DECODE_DIR=${out_root}/${task} bash run_infer_zonos2_eval.sh --task ${task} --test_set en"
echo "note: speaker-similarity is not meaningful for NeuTTS-2E (preset voice); read WER + MSR-UTMOS."
