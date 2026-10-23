#!/usr/bin/env bash
# Generate audio with ElevenLabs *Instant Voice Cloning* (zero-shot).
#
# Like run_generate_elevenlabs.sh but clones each prompt speaker (per-utterance
# IVC) instead of rotating preset voices -- so the output gives a real zero-shot
# speaker-similarity number, comparable to Zonos2 / Omni / fal cloning engines.
# Generation only; score afterwards with run_infer_zonos2_eval.sh.
#
#   uv sync --extra elevenlabs
#   export ELEVENLABS_API_KEY=...     # MUST be scoped with create_instant_voice_clone
#                                     # + voice read/delete (a plain TTS key 401s)
#   bash run_generate_elevenlabs_clone.sh                                  # all of zero_shot
#   bash run_generate_elevenlabs_clone.sh --task zero_shot --test_set en --limit 5
#
# Output -> results/elevenlabs_${model}_clone/<task>/<set>/wavs/
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      v3 | multilingual-v2                  (default v3)
#   --task       restrict to one task group (default: zero_shot zero_shot_voxblink)
#   --test_set   restrict to names within --task       (optional; needs --task)
#   --workers    concurrent IVC chains (bounded by custom-voice cap; default 2)
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

help_message="usage: bash run_generate_elevenlabs_clone.sh [--model {v3|multilingual-v2}] [--task T] [--test_set \"...\"] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    v3|multilingual-v2) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
if [ -n "${test_set}" ] && [ -z "${task}" ]; then
    echo "error: --test_set requires --task" 1>&2
    exit 1
fi

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python
fi

# --- test sets (mirror run_generate_elevenlabs.sh) ----------------
sets_for() {
    case "$1" in
        zero_shot)              echo "zh en ja ko de es fr it ru" ;;
        # Full 21 voxblink langs -- same auto-detect fallback as
        # run_generate_elevenlabs.sh (the shared client sends no language_code
        # for codes outside its ELEVEN_LANG table).
        zero_shot_voxblink)     echo "ar de en es fr hi id it ja ko nl pl pt ru th tl tr uk vi yo zh" ;;
        *) echo "" ;;
    esac
}

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
        zero_shot_voxblink/*) echo "$2" ;;
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
        out_dir="results/elevenlabs_${model}_clone/${tk}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lang_opt=""
        lg="$(lang_for ${tk} ${nm})"
        [ -n "${lg}" ] && lang_opt="--language ${lg}"
        ow_opt=""
        [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

        echo "=== [elevenlabs-clone:${model}] ${data_dir} -> ${out_dir} ${lang_opt} ==="
        "${PY}" scripts/generate_elevenlabs_clone.py \
            --model "${model}" \
            --data-dir "${data_dir}" \
            --out-dir "${out_dir}" \
            --workers "${workers}" \
            --limit "${limit}" \
            ${lang_opt} ${ow_opt}
    done
done

echo
echo "generation done -> results/elevenlabs_${model}_clone/"
echo "score it with run_infer_zonos2_eval.sh, e.g.:"
echo "  DECODE_DIR=results/elevenlabs_${model}_clone/zero_shot bash run_infer_zonos2_eval.sh"
