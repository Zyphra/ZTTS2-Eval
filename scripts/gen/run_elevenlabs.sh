#!/usr/bin/env bash
# Generate audio with ElevenLabs (premade voices, rotated per utterance; no cloning).
# HTTP client only (no GPU); loops every language of --task through
# scripts/gen/generate_elevenlabs.py.
#
#   uv sync --inexact --extra elevenlabs   # one-time; --inexact keeps the scorer extras
#   export ELEVENLABS_API_KEY=...
#   bash scripts/gen/run_elevenlabs.sh --task clean_mini --test_set "en ja" --limit 4
#   bash scripts/gen/run_elevenlabs.sh --task in-the-wild
#
# The benchmark language is each data dir's basename; the client maps it to the
# provider's language code. Writes wavs to <out_root>/<task>/<lang>/wavs/ and
# skips existing wavs, so re-running retries failures. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Run it directly (it picks up .venv/bin/python); a bare `uv run` re-syncs and
# drops the optional extra.
#
# Only one instance per provider account may run at a time (they share its
# concurrency limit).
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      v4                                           (default v4)
#   --task       task group                                   (default clean)
#   --test_set   languages within --task   (default: every language of the task)
#   --out_root   results root              (default results/elevenlabs_<model>)
#   --workers    concurrent requests per language             (default 4)
#   --seed       generation seed, provider best-effort        (default 0)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=v4
task=clean
test_set=
out_root=
workers=4
seed=0
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_elevenlabs.sh [--model v4] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--seed N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    v4) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
[ -z "${out_root}" ] && out_root="results/elevenlabs_${model}"

repo_py

names=$(gen_langs "${task}" "${test_set}") || exit 1

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    echo "=== [elevenlabs:${model}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_elevenlabs.py \
        --model "${model}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --seed "${seed}" \
        --limit "${limit}" \
        ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
