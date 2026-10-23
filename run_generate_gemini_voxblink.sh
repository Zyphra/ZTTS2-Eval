#!/usr/bin/env bash
# Generate audio for the VoxBlink2-derived eval (data/zero_shot_voxblink) with
# native Google Gemini TTS. Mirrors run_generate_gemini.sh but pinned to the one
# voxblink task (no server -- this is just an HTTP client).
#
#   uv sync --extra gemini        # one-time: install google-genai into .venv
#   export GEMINI_API_KEY=...     # your Google API key (or GOOGLE_API_KEY)
#   bash run_generate_gemini_voxblink.sh                        # 9 zs-overlap langs
#   bash run_generate_gemini_voxblink.sh --model 2.5-pro --test_set en --limit 3
#
# Score afterwards with run_infer_voxblink_eval.sh and DECODE_DIR pointing at
# results/gemini_<model>/zero_shot_voxblink.
#
# NOTE: each voxblink lang is sorted worst->best MSR-UTMOS in bins, so `--limit N`
# samples the NOISIEST prompts, not a representative subset -- prefer the full
# run (limit 0) for comparable numbers.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      3.1-flash | 2.5-flash | 2.5-pro       (default 3.1-flash)
#   --test_set   space-separated voxblink lang dirs    (default 9 zs-overlap langs)
#   --workers    concurrent gemini requests per set    (default 4)
#   --limit      only the first N utterances per set    (default 0 = all)
#   --overwrite  regenerate existing wavs               (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=3.1-flash
# Default to the 9 langs shared with zero_shot; the noisy voxblink-only tail
# (pt ar hi tr tl pl id th yo nl uk vi) is off by default. Re-enable via an explicit
# --test_set, e.g. --test_set "en zh de es fr it ja ko ru pt ar hi tr tl pl id th yo nl uk vi".
test_set="en zh de es fr it ja ko ru"
workers=4
limit=0
overwrite=false

help_message="usage: bash run_generate_gemini_voxblink.sh [--model {3.1-flash|2.5-flash|2.5-pro}] [--test_set \"...\"] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    3.1-flash|2.5-flash|2.5-pro) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac

# Prefer the project venv's interpreter so this works without activation.
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python
fi

# --language label for voices.txt (Gemini detects the spoken language itself).
# generate_gemini.py knows {en zh ja ko de es fr it ru}; others fall back to
# per-utterance script detection for the label.
lang_for() {
    case "$1" in
        en|zh|ja|ko|de|es|fr|it|ru) echo "$1" ;;
        *) echo "" ;;
    esac
}

for nm in ${test_set}; do
    data_dir="data/zero_shot_voxblink/${nm}"
    out_dir="results/gemini_${model}/zero_shot_voxblink/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi

    lang_opt=""
    lg="$(lang_for ${nm})"
    [ -n "${lg}" ] && lang_opt="--language ${lg}"
    ow_opt=""
    [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

    echo "=== [gemini:${model}] ${data_dir} -> ${out_dir} ${lang_opt} ==="
    "${PY}" scripts/generate_gemini.py \
        --model "${model}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --limit "${limit}" \
        ${lang_opt} ${ow_opt}
done

echo
echo "generation done -> results/gemini_${model}/zero_shot_voxblink/"
echo "score with: DECODE_DIR=results/gemini_${model}/zero_shot_voxblink bash run_infer_voxblink_eval.sh"
