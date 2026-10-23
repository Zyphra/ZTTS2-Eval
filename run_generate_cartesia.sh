#!/usr/bin/env bash
# Generate audio with the Cartesia (Sonic) REST TTS API.
#
# Generation only -- no GPU / vLLM here. Score the output afterwards with the
# existing run_infer_zonos2_eval.sh (run separately, see the hint at the end).
#
#   uv sync --extra cartesia       # one-time: install requests into .venv
#   export CARTESIA_API_KEY=...     # the token after 'Bearer ' in the curl
#   bash run_generate_cartesia.sh                                  # all test sets
#   bash run_generate_cartesia.sh --task zero_shot --test_set "en" --limit 5
#
# Run it directly -- it picks up .venv/bin/python automatically. Do NOT wrap it
# in a bare `uv run`: that re-syncs and drops the optional 'cartesia' extra. If
# you must, use `uv run --extra cartesia bash run_generate_cartesia.sh ...`.
#
# Like the Gemini / ElevenLabs / Inworld engines, this uses preset (public)
# voices rotated per utterance (no zero-shot cloning); Sonic is multilingual, so
# one English voice pool serves every language via the --language flag.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      sonic-3.5                            (default sonic-3.5)
#   --task       restrict to one task group (default: zero_shot zero_shot_voxblink)
#   --test_set   restrict to names within --task       (optional; needs --task)
#   --workers    concurrent cartesia requests per set   (default 4)
#   --limit      only the first N utterances per set    (default 0 = all)
#   --overwrite  regenerate existing wavs               (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=sonic-3.5
mode=preset
task=
test_set=
workers=4
limit=0
overwrite=false

help_message="usage: bash run_generate_cartesia.sh [--model sonic-3.5] [--mode preset|clone] [--task T] [--test_set \"...\"] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    sonic-3.5) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
case "${mode}" in
    preset) mode_suffix="" ;;
    clone)  mode_suffix="_clone" ;;
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
        # voxblink has 21 langs; default to the 14 Sonic supports (per
        # generate_cartesia.py's CARTESIA_LANGS -- omits ar id th tl uk vi yo,
        # which are not in Sonic's language enum)
        zero_shot_voxblink)     echo "en de es fr it ja ko ru zh pt hi nl pl tr" ;;
        *) echo "" ;;
    esac
}

# --language label for each test set (sent to the API and used for voices.txt).
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
        zero_shot_voxblink/*) echo "$2" ;;   # voxblink test-set name is the lang code
        *) echo "" ;;
    esac
}

tasks="${task:-zero_shot zero_shot_voxblink}"

for tk in ${tasks}; do
    names="${test_set:-$(sets_for ${tk})}"
    if [ -z "${names}" ]; then
        echo "error: unknown task '${tk}'" 1>&2
        exit 1
    fi
    for nm in ${names}; do
        data_dir="data/${tk}/${nm}"
        out_dir="results/cartesia_${model}${mode_suffix}/${tk}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lang_opt=""
        lg="$(lang_for ${tk} ${nm})"
        [ -n "${lg}" ] && lang_opt="--language ${lg}"
        ow_opt=""
        [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

        echo "=== [cartesia:${model}:${mode}] ${data_dir} -> ${out_dir} ${lang_opt} ==="
        "${PY}" scripts/generate_cartesia.py \
            --model "${model}" \
            --mode "${mode}" \
            --data-dir "${data_dir}" \
            --out-dir "${out_dir}" \
            --workers "${workers}" \
            --limit "${limit}" \
            ${lang_opt} ${ow_opt}
    done
done

echo
echo "generation done -> results/cartesia_${model}${mode_suffix}/"
echo "score it with run_infer_zonos2_eval.sh, e.g.:"
echo "  DECODE_DIR=results/cartesia_${model}${mode_suffix}/zero_shot bash run_infer_zonos2_eval.sh"
