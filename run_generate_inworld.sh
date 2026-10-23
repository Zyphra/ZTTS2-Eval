#!/usr/bin/env bash
# Generate audio with the Inworld REST TTS API.
#
# Generation only -- no GPU / vLLM here. Score the output afterwards with the
# existing run_infer_zonos2_eval.sh (run separately, see the hint at the end).
#
#   uv sync --extra inworld          # one-time: install requests into .venv
#   export INWORLD_API_KEY=...        # the base64 token after 'Basic ' in the curl
#   bash run_generate_inworld.sh                                  # all of zero_shot
#   bash run_generate_inworld.sh --model 2 --task zero_shot --test_set "en" --limit 5
#
# Run it directly -- it picks up .venv/bin/python automatically. Do NOT wrap it
# in a bare `uv run`: that re-syncs and drops the optional 'inworld' extra. If
# you must, use `uv run --extra inworld bash run_generate_inworld.sh ...`.
#
# Two modes: default uses preset (SYSTEM) voices fetched live per language (like
# Gemini / ElevenLabs); --clone true does instant voice cloning of each prompt
# speaker (real zero-shot speaker-similarity), writing to results/inworld_<m>_clone/.
# Clones persist on the account -- remove them after scoring with --delete_clones true.
#
#   bash run_generate_inworld.sh --clone true --task zero_shot --test_set en  # spk-sim run
#   bash run_generate_inworld.sh --delete_clones true --task zero_shot --test_set en  # cleanup
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model         2 | 1.5-max | 1.5 | 1               (default 2)
#   --task          restrict to one task group (default: zero_shot zero_shot_voxblink)
#   --test_set      restrict to names within --task     (optional; needs --task)
#   --workers       concurrent inworld requests per set  (default 4)
#   --limit         only the first N utterances per set  (default 0 = all)
#   --overwrite     regenerate existing wavs             (default false)
#   --clone         clone each prompt speaker (IVC)      (default false)
#   --delete_clones delete clones.txt voices, then exit  (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=2
task=
test_set=
workers=4
limit=0
overwrite=false
clone=false
delete_clones=false

help_message="usage: bash run_generate_inworld.sh [--model {2|1.5-max|1.5|1}] [--task T] [--test_set \"...\"] [--workers N] [--limit N] [--overwrite true] [--clone true] [--delete_clones true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    2|1.5-max|1.5|1) ;;
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
        # 18 of voxblink's 21 langs. uk/vi/yo stay excluded: Inworld has no
        # SYSTEM voice pool or clone locale for them (generate_inworld.py's
        # LOCALE_FOR), so they would silently fall back to English voices.
        zero_shot_voxblink)     echo "ar de en es fr hi id it ja ko nl pl pt ru th tl tr zh" ;;
        *) echo "" ;;
    esac
}

# --language label for each test set (selects which SYSTEM voice pool is fetched
# and labels voices.txt).
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
        # Clone-mode output lives in a separate tree so it never overwrites the
        # preset-voice results (and so spk-sim is scored on the right set).
        # --delete_clones targets the same clone tree (to find clones.txt).
        if [ "${clone}" = "true" ] || [ "${delete_clones}" = "true" ]; then
            out_dir="results/inworld_${model}_clone/${tk}/${nm}"
        else
            out_dir="results/inworld_${model}/${tk}/${nm}"
        fi
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lang_opt=""
        lg="$(lang_for ${tk} ${nm})"
        [ -n "${lg}" ] && lang_opt="--language ${lg}"
        ow_opt=""
        [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
        clone_opt=""
        [ "${clone}" = "true" ] && clone_opt="--clone"
        del_opt=""
        [ "${delete_clones}" = "true" ] && del_opt="--clone --delete-clones"

        label="inworld:${model}"; [ "${clone}" = "true" ] && label="${label}+clone"
        echo "=== [${label}] ${data_dir} -> ${out_dir} ${lang_opt} ==="
        "${PY}" scripts/generate_inworld.py \
            --model "${model}" \
            --data-dir "${data_dir}" \
            --out-dir "${out_dir}" \
            --workers "${workers}" \
            --limit "${limit}" \
            ${lang_opt} ${ow_opt} ${clone_opt} ${del_opt}
    done
done

if [ "${clone}" = "true" ]; then
    out_root="results/inworld_${model}_clone"
else
    out_root="results/inworld_${model}"
fi
echo
[ "${delete_clones}" = "true" ] && exit 0
echo "generation done -> ${out_root}/"
echo "score it with run_infer_zonos2_eval.sh, e.g.:"
echo "  DECODE_DIR=${out_root}/zero_shot bash run_infer_zonos2_eval.sh"
if [ "${clone}" = "true" ]; then
    echo "clones persist on your account; delete them when done with:"
    echo "  bash run_generate_inworld.sh --model ${model} --delete_clones true${task:+ --task ${task}}${test_set:+ --test_set \"${test_set}\"}"
fi
