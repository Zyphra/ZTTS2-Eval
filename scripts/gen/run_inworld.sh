#!/usr/bin/env bash
# Generate audio with Inworld TTS (/tts/v1/voice). Default: preset SYSTEM voices
# per language; --clone true: per-utterance instant voice clone of the prompt wav
# (no transcription), deleted right after synthesis.
# HTTP client only (no GPU); loops every language of --task through
# scripts/gen/generate_inworld.py.
#
#   uv sync --inexact --extra inworld   # one-time; --inexact keeps the scorer extras
#   export INWORLD_API_KEY=...
#   bash scripts/gen/run_inworld.sh --clone true --task clean_mini --test_set "en ja" --limit 4
#   bash scripts/gen/run_inworld.sh --clone true --clone_rate_per_min 5 --task in-the-wild
#
# The benchmark language is each data dir's basename; the client maps it to the
# provider's language code. Writes wavs to <out_root>/<task>/<lang>/wavs/ and
# skips existing wavs, so re-running retries failures. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Run it directly (it picks up .venv/bin/python); a bare `uv run` re-syncs and
# drops the optional extra.
#
# Only one instance per provider account may run at a time: clone creation is
# paced to the workspace's CloneVoice rate limit (--clone_rate_per_min), which
# concurrent runs would share. Needs a key with voices write access for --clone.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model              2                                    (default 2)
#   --clone              true = instant voice cloning         (default false)
#   --task               task group                           (default clean)
#   --test_set           languages within --task (default: every language of the task)
#   --out_root           results root (default results/inworld_<model>, + _clone for --clone true)
#   --workers            concurrent requests per language, preset mode (default 4)
#   --clone_rate_per_min clone requests/minute (token bucket)  (default 5)
#   --clone_workers      clone+synth+delete pipelines         (default 4)
#   --limit              only the first N utterances per set  (default 0 = all)
#   --overwrite          regenerate existing wavs             (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=2
clone=false
task=clean
test_set=
out_root=
workers=4
clone_rate_per_min=5
clone_workers=4
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_inworld.sh [--model 2] [--clone true|false] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--clone_rate_per_min R] [--clone_workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    2) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
if [ "${clone}" != "true" ] && [ "${clone}" != "false" ]; then
    echo "error: --clone must be true or false (got '${clone}')" 1>&2
    exit 2
fi
if [ -z "${out_root}" ]; then
    out_root="results/inworld_${model}"
    [ "${clone}" = "true" ] && out_root="${out_root}_clone"
fi

gen_main_py

names=$(gen_langs "${task}" "${test_set}") || exit 1

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
clone_opt=""; [ "${clone}" = "true" ] && clone_opt="--clone"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    echo "=== [inworld:${model}${clone_opt:+ clone}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_inworld.py \
        --model "${model}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --clone-rate-per-min "${clone_rate_per_min}" \
        --clone-workers "${clone_workers}" \
        --limit "${limit}" \
        ${ow_opt} ${clone_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
