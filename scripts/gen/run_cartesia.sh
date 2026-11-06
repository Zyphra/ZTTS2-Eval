#!/usr/bin/env bash
# Generate audio with Cartesia Sonic (/tts/bytes). --mode preset rotates public
# voices; --mode clone instant-clones each prompt wav (deleted after use).
# HTTP client only (no GPU); loops every language of --task through
# scripts/gen/generate_cartesia.py.
#
#   uv sync --inexact --extra cartesia   # one-time; --inexact keeps the scorer extras
#   export CARTESIA_API_KEY=...
#   bash scripts/gen/run_cartesia.sh --mode clone --task clean_mini --test_set "en ja" --limit 4
#   bash scripts/gen/run_cartesia.sh --mode clone --task in-the-wild
#
# The benchmark language is each data dir's basename; the client maps it to the
# provider's language code. Writes wavs to <out_root>/<task>/<lang>/wavs/ and
# skips existing wavs, so re-running retries failures. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Run it directly (it picks up .venv/bin/python); a bare `uv run` re-syncs and
# drops the optional extra.
#
# Only one instance per provider account may run at a time: clone mode's
# startup/final purge deletes EVERY live evtmp-* voice on the account, including
# another run's in-flight clones.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      sonic-3.6                                    (default sonic-3.6)
#   --mode       preset | clone                               (default preset)
#   --task       task group                                   (default clean)
#   --test_set   languages within --task   (default: every language of the task)
#   --out_root   results root  (default results/cartesia_<model>, + _clone for --mode clone)
#   --workers    concurrent requests per language             (default 4)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=sonic-3.6
mode=preset
task=clean
test_set=
out_root=
workers=4
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_cartesia.sh [--model sonic-3.6] [--mode preset|clone] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    sonic-3.6) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
if [ "${mode}" != "preset" ] && [ "${mode}" != "clone" ]; then
    echo "error: --mode must be preset or clone (got '${mode}')" 1>&2
    exit 2
fi
if [ -z "${out_root}" ]; then
    out_root="results/cartesia_${model}"
    [ "${mode}" = "clone" ] && out_root="${out_root}_clone"
fi

gen_main_py

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
    echo "=== [cartesia:${model}:${mode}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_cartesia.py \
        --model "${model}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --mode "${mode}" \
        --workers "${workers}" \
        --limit "${limit}" \
        ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
