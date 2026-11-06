#!/usr/bin/env bash
# Generate Audio8-TTS-Realistic-0.6B audio through the upstream SGLang-Omni adapter
# server (Edge0-AI/Audio8_TTS sglang_omni/, POST /v1/audio/speech): with-transcript
# zero-shot cloning (prompt wav + prompt_text).
#
# By default this launches its own server (scripts/serve_audio8.sh) on one GPU,
# loops every language of the task through it, then stops it. Pass --server <URL>
# to use an already-running server instead (no launch/stop).
#
# Set up the server venv, checkouts and model snapshot first:
#   bash scripts/setup_audio8_env.sh
#
#   bash run_generate_audio8.sh --task clean --test_set "en zh" --limit 4
#   bash run_generate_audio8.sh --task in-the-wild --gpu 2
#   bash run_generate_audio8.sh --server http://127.0.0.1:8010 --task clean
#   AUDIO8_DISABLE_CUDA_GRAPH=1 bash run_generate_audio8.sh ...   # CUDA graphs off
#
# Writes wavs to <out_root>/<task>/<lang>/wavs/. Score afterwards with the hint
# printed at the end.
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --server          external endpoint; EMPTY = launch our own   (default '')
#   --gpu             GPU index for the launched server           (default 0)
#   --port            preferred port for the launched server      (default 8010)
#   --task            task group                                  (default clean)
#   --test_set        languages within --task (default: every language of the task)
#   --out_root        results root      (default results/audio8-tts-realistic-0.6b)
#   --workers         concurrent requests per language            (default 32)
#   --max_new_tokens  per-request codec-frame cap                 (default 1024)
#   --limit           only the first N utterances per set         (default 0 = all)
#   --overwrite       regenerate existing wavs                    (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

server=""
gpu=0
port=8010
task=clean
test_set=
out_root="results/audio8-tts-realistic-0.6b"
workers=32
max_new_tokens=1024
limit=0
overwrite=false

help_message="usage: bash run_generate_audio8.sh [--server URL|''] [--gpu N] [--port N] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--max_new_tokens N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

names="${test_set:-$(python3 -m utils.benchmark langs "${task}")}"
if [ -z "${names}" ]; then
    echo "error: unknown task '${task}'" 1>&2
    exit 1
fi

# --- server lifecycle -----------------------------------------------------------
if [ -z "${server}" ]; then
    export AUDIO8_GPU="${gpu}" AUDIO8_PORT="${port}"
    . scripts/serve_audio8.sh
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
    "${PY}" scripts/generate_audio8.py \
        --server "${server}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --max-new-tokens "${max_new_tokens}" \
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
