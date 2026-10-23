#!/usr/bin/env bash
# Generate audio with the official ElevenLabs API.
#
# Generation only -- no GPU / vLLM here. Score the output afterwards with the
# existing run_infer_zonos2_eval.sh (run separately, see the hint at the end).
#
#   uv sync --extra elevenlabs       # one-time: install elevenlabs into .venv
#   export ELEVENLABS_API_KEY=...     # your ElevenLabs API key
#   bash run_generate_elevenlabs.sh                                  # all of zero_shot
#   bash run_generate_elevenlabs.sh --model v3 --task zero_shot --test_set "en" --limit 5
#
# Run it directly -- it picks up .venv/bin/python automatically. Do NOT wrap it
# in a bare `uv run`: that re-syncs and drops the optional 'elevenlabs' extra. If
# you must, use `uv run --extra elevenlabs bash run_generate_elevenlabs.sh ...`.
#
# Like the Gemini engine, this uses preset voices (no zero-shot cloning).
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      v3 | multilingual-v2                  (default v3)
#   --task       restrict to one task group (default: zero_shot zero_shot_voxblink)
#   --test_set   restrict to names within --task       (optional; needs --task)
#   --workers    concurrent elevenlabs requests per set (default 2; free/Starter cap)
#   --limit      only the first N utterances per set    (default 0 = all)
#   --overwrite  regenerate existing wavs               (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=v3
task=
test_set=
workers=2
limit=0
overwrite=false

help_message="usage: bash run_generate_elevenlabs.sh [--model {v3|multilingual-v2}] [--task T] [--test_set \"...\"] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    v3|multilingual-v2) ;;
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
        # Full 21 voxblink langs. Codes outside generate_elevenlabs.py's
        # ELEVEN_LANG table (tl uk vi yo nl for eleven_v3) are handled there:
        # no language_code is sent and the model auto-detects (with a 400
        # fallback), so an unsupported code never zeroes out a split.
        zero_shot_voxblink)     echo "ar de en es fr hi id it ja ko nl pl pt ru th tl tr uk vi yo zh" ;;
        *) echo "" ;;
    esac
}

# --language label for each test set (sent as language_code when en/zh/ja/ko,
# else ElevenLabs auto-detects from the text).
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
        out_dir="results/elevenlabs_${model}/${tk}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lang_opt=""
        lg="$(lang_for ${tk} ${nm})"
        [ -n "${lg}" ] && lang_opt="--language ${lg}"
        ow_opt=""
        [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

        echo "=== [elevenlabs:${model}] ${data_dir} -> ${out_dir} ${lang_opt} ==="
        "${PY}" scripts/generate_elevenlabs.py \
            --model "${model}" \
            --data-dir "${data_dir}" \
            --out-dir "${out_dir}" \
            --workers "${workers}" \
            --limit "${limit}" \
            ${lang_opt} ${ow_opt}
    done
done

echo
echo "generation done -> results/elevenlabs_${model}/"
echo "score it with run_infer_zonos2_eval.sh, e.g.:"
echo "  DECODE_DIR=results/elevenlabs_${model}/zero_shot bash run_infer_zonos2_eval.sh"
