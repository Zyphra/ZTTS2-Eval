#!/usr/bin/env bash
# Generate zero-shot cloned audio with ElevenLabs (per-utterance Instant Voice
# Clone of the prompt wav, deleted after use).
# HTTP client only (no GPU); loops every language of --task through
# scripts/gen/generate_elevenlabs_clone.py.
#
#   uv sync --inexact --extra elevenlabs   # one-time; --inexact keeps the scorer extras
#   export ELEVENLABS_API_KEY=...
#   bash scripts/gen/run_elevenlabs_clone.sh --task clean_mini --test-set "en ja" --limit 4
#   bash scripts/gen/run_elevenlabs_clone.sh --task in-the-wild
#
# The benchmark language is each data dir's basename; the client maps it to the
# provider's language code. Writes wavs to <out_root>/<task>/<lang>/wavs/ and
# skips existing wavs, so re-running retries failures. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Run it directly (it picks up .venv/bin/python); a bare `uv run` re-syncs and
# drops the optional extra.
#
# Only one instance per provider account may run at a time: the startup/final
# purge deletes EVERY live ztts_clone_* voice on the account, including another
# run's in-flight clones. Needs a key with text-to-speech, voices read/write
# and user-read scopes.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      v4                                           (default v4)
#   --task       task group                                   (default clean)
#   --test-set   languages within --task   (default: every language of the task)
#   --out-root   results root (default: gen_default_out_root; elevenlabs_clone row:
#                results/elevenlabs_<model>_clone)
#   --workers    concurrent requests per language             (default 4)
#   --seed       generation seed, provider best-effort; '' = none (default 0)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
#   --list-langs print the languages that would run and exit  (default false)
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
list_langs=false

help_message="usage: bash scripts/gen/run_elevenlabs_clone.sh [--model v4] [--task T] [--test-set \"...\"] [--out-root D] [--workers N] [--seed N] [--limit N] [--overwrite true] [--list-langs true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    v4) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
[ -n "${out_root}" ] || out_root=$(gen_default_out_root elevenlabs_clone "${model}")

GEN_NAME=elevenlabs_clone
names=$(gen_langs "${task}" "${test_set}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"

repo_py

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
seed_opt=(); [ -n "${seed}" ] && seed_opt=(--seed "${seed}")

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    echo "=== [elevenlabs-clone:${model}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_elevenlabs_clone.py \
        --model "${model}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        "${seed_opt[@]}" \
        --limit "${limit}" \
        ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
