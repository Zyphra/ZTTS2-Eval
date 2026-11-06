#!/usr/bin/env bash
# Generate audio with native Google Gemini TTS (30 prebuilt voices, rotated per utterance).
# HTTP client only (no GPU); loops every language of --task through
# scripts/gen/generate_gemini.py.
#
#   uv sync --inexact --extra gemini   # one-time; --inexact keeps the scorer extras
#   export GEMINI_API_KEY=...
#   bash scripts/gen/run_gemini.sh --task clean_mini --test_set "en ja" --limit 4
#   bash scripts/gen/run_gemini.sh --task in-the-wild
#
# The benchmark language is each data dir's basename; the client maps it to the
# provider's language code. Writes wavs to <out_root>/<task>/<lang>/wavs/ and
# skips existing wavs, so re-running retries failures. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Run it directly (it picks up .venv/bin/python); a bare `uv run` re-syncs and
# drops the optional extra.
#
# --batch true (default) goes through the Gemini Batch API (50% of the unary price,
# async): pass 1 submits every language's missing utterances without waiting; pass 2
# runs one waiting client per language CONCURRENTLY, which writes the wavs and
# resubmits failed requests for up to 4 rounds (and anything pass 1 could not
# submit, e.g. on the enqueued-token limit). Pass-2 output lines from different
# languages interleave; each "done:" line names its wav dir. Interrupting is safe:
# job names live in <out_dir>/batch_jobs.json and the next run re-attaches.
# --batch false makes one unary call per utterance.
#
# Only one instance per provider account may run at a time (they share its
# rate limits and batch quota).
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      3.8-flash                                    (default 3.8-flash)
#   --task       task group                                   (default clean)
#   --test_set   languages within --task   (default: every language of the task)
#   --out_root   results root              (default results/gemini_<model>)
#   --batch      true = Batch API, false = unary calls         (default true)
#   --workers    concurrent unary requests per language       (default 4)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=3.8-flash
task=clean
test_set=
out_root=
batch=true
workers=4
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_gemini.sh [--model 3.8-flash] [--task T] [--test_set \"...\"] [--out_root D] [--batch true|false] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    3.8-flash) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
if [ "${batch}" != "true" ] && [ "${batch}" != "false" ]; then
    echo "error: --batch must be true or false (got '${batch}')" 1>&2
    exit 2
fi
[ -z "${out_root}" ] && out_root="results/gemini_${model}"

gen_main_py

names=$(gen_langs "${task}" "${test_set}") || exit 1

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

# One pass over every language; extra args go to generate_gemini.py. With
# parallel=true every language runs in the background and the pass waits for all.
run_pass() {
    local parallel=$1; shift
    local extra=("$@") args pids="" pid failed=0
    ran=0
    for nm in ${names}; do
        data_dir="data/${task}/${nm}"
        out_dir="${out_root}/${task}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi
        echo "=== [gemini:${model}${extra[*]:+ ${extra[*]}}] ${data_dir} -> ${out_dir} ==="
        args=(--model "${model}" --data-dir "${data_dir}" --out-dir "${out_dir}"
              --workers "${workers}" --limit "${limit}" ${ow_opt} "${extra[@]}")
        if [ "${parallel}" = "true" ]; then
            "${PY}" scripts/gen/generate_gemini.py "${args[@]}" &
            pids="${pids} $!"
        else
            "${PY}" scripts/gen/generate_gemini.py "${args[@]}"
        fi
        ran=$((ran + 1))
    done
    for pid in ${pids}; do
        wait "${pid}" || failed=1
    done
    [ "${failed}" -eq 0 ] || { echo "error: a generate_gemini.py process failed" 1>&2; exit 1; }
}

if [ "${batch}" = "true" ]; then
    run_pass false --batch --batch-no-wait
    ow_opt=""  # pass 1 already queued the overwrites; pass 2 only collects / resubmits
    run_pass true --batch
else
    run_pass false
fi

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
