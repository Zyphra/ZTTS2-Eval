#!/usr/bin/env bash
# Generate ZONOS3 Mini audio through the release minisgl HTTP server
# (python -m zonos2, POST /tts/generate), one conditioning mode per call.
#
# By default this launches its own server (scripts/gen/serve_zonos3.sh) on one GPU,
# loops every language of the task through it, then stops it. Pass --server <URL>
# to use an already-running server instead (no launch/stop).
#
#   bash scripts/gen/run_zonos3.sh --speaker_mode reference --task clean_mini --test_set "en ja" --limit 4
#   bash scripts/gen/run_zonos3.sh --speaker_mode embedding --task clean --gpu 2
#   bash scripts/gen/run_zonos3.sh --speaker_mode reference --server http://127.0.0.1:1920 --task in-the-wild
#
# Writes wavs to <out_root>/<task>/<lang>/wavs/. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
# Cluster environment for the self-managed server:
#   export ZONOS3_HOME=/data/groups/audio/sofian/bingua2-z3m-accent-serve/minisgl
#   export ZONOS3_ENV=$ZONOS3_HOME/scripts/z3m_accent_env.sh
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --server        external endpoint; EMPTY = launch our own     (default '')
#   --gpu           GPU index for the launched server             (default 0)
#   --port          preferred port for the launched server        (default 1920)
#   --speaker_mode  reference | embedding                         (required)
#   --task          task group                                    (default clean)
#   --test_set      languages within --task   (default: every language of the task)
#   --out_root      results root              (default results/zonos3-mini_<speaker_mode>)
#   --workers       concurrent requests per language              (default 16)
#   --seed          sampling seed                                 (default 0)
#   --max_tokens    per-request token cap                         (default 1350)
#   --limit         only the first N utterances per set           (default 0 = all)
#   --overwrite     regenerate existing wavs                      (default false)
#   --laci          true = LACI decoding (laci=true; the server needs --laci-heads,
#                   e.g. ZONOS3_SERVE_ARGS="--laci-heads en_us=heads_en.json cmn=heads_zh.json") (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

server=""
gpu=0
port=1920
speaker_mode=
task=clean
test_set=
out_root=
workers=16
seed=0
max_tokens=1350
limit=0
overwrite=false
laci=false

help_message="usage: bash scripts/gen/run_zonos3.sh --speaker_mode reference|embedding [--server URL|''] [--gpu N] [--port N] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--seed N] [--max_tokens N] [--limit N] [--overwrite true] [--laci true]"
. utils/parse_options.sh || exit 1

if [ "${speaker_mode}" != "reference" ] && [ "${speaker_mode}" != "embedding" ]; then
    echo "error: --speaker_mode must be reference or embedding (got '${speaker_mode}')" 1>&2
    exit 2
fi
if [ "${laci}" != "true" ] && [ "${laci}" != "false" ]; then
    echo "error: --laci must be true or false (got '${laci}')" 1>&2
    exit 2
fi
[ -z "${out_root}" ] && out_root="results/zonos3-mini_${speaker_mode}"

repo_py

names=$(gen_langs "${task}" "${test_set}") || exit 1

# --- server lifecycle -----------------------------------------------------------
if [ -z "${server}" ]; then
    export ZONOS3_GPU="${gpu}" ZONOS3_PORT="${port}"
    . scripts/gen/serve_zonos3.sh
    trap stop_zonos3_server EXIT INT TERM
    start_zonos3_server || { echo "error: failed to launch ZONOS3 server (see log above)" 1>&2; exit 1; }
    server="${ZONOS3_BASE_URL}"
else
    echo "[zonos3] using external server ${server} (not managed here)"
fi

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
laci_opt=""; [ "${laci}" = "true" ] && laci_opt="--laci"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    echo "=== [zonos3 ${speaker_mode}${laci_opt:+ laci}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_zonos3.py \
        --server "${server}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --speaker-mode "${speaker_mode}" \
        --workers "${workers}" \
        --seed "${seed}" \
        --max-tokens "${max_tokens}" \
        --limit "${limit}" \
        ${ow_opt} ${laci_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
