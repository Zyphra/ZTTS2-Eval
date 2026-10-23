#!/usr/bin/env bash
# Generate audio with native Google Gemini TTS.
#
# Generation only -- no GPU / vLLM here. Score the output afterwards with the
# existing run_infer_zonos2_eval.sh (run separately, see the hint at the end).
#
#   uv sync --extra gemini        # one-time: install google-genai into .venv
#   export GEMINI_API_KEY=...     # your Google API key (or GOOGLE_API_KEY)
#   bash run_generate_gemini.sh                                  # all test sets
#   bash run_generate_gemini.sh --model 2.5-pro --task zero_shot --test_set "en" --limit 5
#
# Run it directly -- it picks up .venv/bin/python automatically. Do NOT wrap it
# in a bare `uv run`: that re-syncs and drops the optional 'gemini' extra. If you
# must, use `uv run --extra gemini bash run_generate_gemini.sh ...`.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      3.1-flash | 2.5-flash | 2.5-pro       (default 3.1-flash)
#   --task       restrict to one task group (optional)
#   --test_set   restrict to names within --task       (optional; needs --task)
#   --workers    concurrent gemini requests per set    (default 4)
#   --limit      only the first N utterances per set    (default 0 = all)
#   --overwrite  regenerate existing wavs               (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=3.1-flash
task=
test_set=
workers=4
limit=0
overwrite=false

help_message="usage: bash run_generate_gemini.sh [--model {3.1-flash|2.5-flash|2.5-pro}] [--task T] [--test_set \"...\"] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    3.1-flash|2.5-flash|2.5-pro) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
if [ -n "${test_set}" ] && [ -z "${task}" ]; then
    echo "error: --test_set requires --task" 1>&2
    exit 1
fi

# Prefer the project venv's interpreter so this works without activation.
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python
fi

# --- test sets ---------------------------------------------------
sets_for() {
    case "$1" in
        zero_shot)              echo "zh en ja ko de es fr it ru" ;;
        *) echo "" ;;
    esac
}

# --language label for each test set (used only for voices.txt; Gemini detects
# the spoken language from the text itself).
lang_for() {
    case "$1/$2" in
        zero_shot/zh) echo zh ;;
        zero_shot/en) echo en ;;
        zero_shot/ja) echo ja ;;
        zero_shot/ko) echo ko ;;
        zero_shot/de) echo de ;;
        zero_shot/es) echo es ;;
        zero_shot/fr) echo fr ;;
        zero_shot/it) echo it ;;
        zero_shot/ru) echo ru ;;
        *) echo "" ;;
    esac
}

tasks="${task:-zero_shot}"

for tk in ${tasks}; do
    names="${test_set:-$(sets_for ${tk})}"
    if [ -z "${names}" ]; then
        echo "error: unknown task '${tk}'" 1>&2
        exit 1
    fi
    for nm in ${names}; do
        data_dir="data/${tk}/${nm}"
        out_dir="results/gemini_${model}/${tk}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lang_opt=""
        lg="$(lang_for ${tk} ${nm})"
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
done

echo
echo "generation done -> results/gemini_${model}/"
echo "score it with run_infer_zonos2_eval.sh, e.g.:"
echo "  DECODE_DIR=results/gemini_${model}/zero_shot bash run_infer_zonos2_eval.sh"
