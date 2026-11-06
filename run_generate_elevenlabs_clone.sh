#!/usr/bin/env bash
# Generate zero-shot cloned audio with ElevenLabs (per-utterance Instant Voice
# Clone of the prompt wav, deleted after use).
# HTTP client only (no GPU); loops every language of --task through
# scripts/generate_elevenlabs_clone.py.
#
#   uv sync --inexact --extra elevenlabs   # one-time; --inexact keeps the scorer extras
#   export ELEVENLABS_API_KEY=...
#   bash run_generate_elevenlabs_clone.sh --task clean_mini --test_set "en ja" --limit 4
#   bash run_generate_elevenlabs_clone.sh --task in-the-wild
#
# The benchmark language is each data dir's basename; the client maps it to the
# provider's language code. Writes wavs to <out_root>/<task>/<lang>/wavs/ and
# skips existing wavs, so re-running retries failures. Score afterwards with the
# hint printed at the end.
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
#   --test_set   languages within --task   (default: every language of the task)
#   --out_root   results root              (default results/elevenlabs_<model>_clone)
#   --workers    concurrent requests per language             (default 4)
#   --seed       generation seed, provider best-effort        (default 0)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=v4
task=clean
test_set=
out_root=
workers=4
seed=0
limit=0
overwrite=false

help_message="usage: bash run_generate_elevenlabs_clone.sh [--model v4] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--seed N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    v4) ;;
    *) echo "${help_message}" 1>&2; exit 1 ;;
esac
[ -z "${out_root}" ] && out_root="results/elevenlabs_${model}_clone"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

names="${test_set:-$(python3 -m utils.benchmark langs "${task}")}"
if [ -z "${names}" ]; then
    echo "error: unknown task '${task}'" 1>&2
    exit 1
fi

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    echo "=== [elevenlabs-clone:${model}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/generate_elevenlabs_clone.py \
        --model "${model}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --seed "${seed}" \
        --limit "${limit}" \
        ${ow_opt}
    ran=$((ran + 1))
done

if [ "${ran}" -eq 0 ]; then
    echo "error: nothing generated (no data/${task}/<lang>/text for: ${names})" 1>&2
    exit 1
fi

case "${task}" in
    *voxblink*) scorer=run_infer_voxblink_eval.sh ;;
    *)          scorer=run_infer_zonos2_eval.sh ;;
esac
echo
echo "generation done -> ${out_root}/"
echo "score with:  DECODE_DIR=${out_root}/${task} bash ${scorer} --task ${task} --test_set \"${names}\""
