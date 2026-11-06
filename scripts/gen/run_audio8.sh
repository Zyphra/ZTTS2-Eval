#!/usr/bin/env bash
# Generate Audio8-TTS-Realistic-0.6B audio through the upstream SGLang-Omni adapter
# server (Edge0-AI/Audio8_TTS sglang_omni/, POST /v1/audio/speech): with-transcript
# zero-shot cloning (prompt wav + prompt_text).
#
# By default this launches its own server (scripts/gen/serve_audio8.sh) on one GPU,
# loops every language of the task through it, then stops it. Pass --server <URL>
# to use an already-running server instead (no launch/stop).
#
# Set up the server venv, checkouts and model snapshot first:
#   bash scripts/setup/setup_audio8_env.sh
#
#   bash scripts/gen/run_audio8.sh --task clean --test-set "en zh" --limit 4
#   bash scripts/gen/run_audio8.sh --task in-the-wild --gpu 2
#   bash scripts/gen/run_audio8.sh --server http://127.0.0.1:8010 --task clean
#   AUDIO8_DISABLE_CUDA_GRAPH=1 bash scripts/gen/run_audio8.sh ...   # CUDA graphs off
#
# Writes wavs to <out_root>/<task>/<lang>/wavs/. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --server          external endpoint; EMPTY = launch our own   (default '')
#   --gpu             GPU index for the launched server           (default 0)
#   --port            preferred port for the launched server      (default 8010)
#   --task            task group                                  (default clean)
#   --test-set        languages within --task (default: every language of the task)
#   --out-root        results root (default: gen_default_out_root; audio8 row:
#                     results/audio8-tts-realistic-0.6b)
#   --workers         concurrent requests per language            (default 32)
#   --max-new-tokens  per-request codec-frame cap                 (default 1024)
#   --limit           only the first N utterances per set         (default 0 = all)
#   --overwrite       regenerate existing wavs                    (default false)
#   --list-langs      print the languages that would run and exit (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

server=""
gpu=0
port=8010
task=clean
test_set=
out_root=
workers=32
max_new_tokens=1024
limit=0
overwrite=false
list_langs=false

help_message="usage: bash scripts/gen/run_audio8.sh [--server URL|''] [--gpu N] [--port N] [--task T] [--test-set \"...\"] [--out-root D] [--workers N] [--max-new-tokens N] [--limit N] [--overwrite true] [--list-langs true]"
. utils/parse_options.sh || exit 1

GEN_NAME=audio8
names=$(gen_langs "${task}" "${test_set}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"

[ -n "${out_root}" ] || out_root=$(gen_default_out_root audio8 "")
repo_py

# --- server lifecycle -----------------------------------------------------------
if [ -z "${server}" ]; then
    export AUDIO8_GPU="${gpu}" AUDIO8_PORT="${port}"
    . scripts/gen/serve_audio8.sh
    trap stop_audio8_server EXIT INT TERM
    start_audio8_server || { echo "error: failed to launch Audio8 server (see log above)" 1>&2; exit 1; }
    server="${AUDIO8_BASE_URL}"
else
    echo "[audio8] using external server ${server} (not managed here)"
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
    echo "=== [audio8] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_audio8.py \
        --server "${server}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --max-new-tokens "${max_new_tokens}" \
        --limit "${limit}" \
        ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
