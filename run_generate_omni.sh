#!/usr/bin/env bash
# Generate audio with vLLM-Omni TTS models -- starts/stops the server itself.
#
# For each selected model this:
#   1. starts a vLLM-Omni server   (scripts/serve_omni.sh -> `vllm-omni serve --omni`)
#   2. generates every test set through it   (scripts/generate_omni.py)
#   3. stops the server, then moves to the next model
# A trap stops the running server on success, error, or Ctrl-C.
#
# Generation only -- score with the command printed at the end (run_score.sh).
#
# Requires vLLM-Omni (the `vllm-omni` CLI), installed in this repo's .venv (Python
# 3.12) or in the venv named by OMNI_VENV. generate_omni.py is just an HTTP client and
# runs from the same venv.
#
#   bash run_generate_omni.sh --model all                                  # clean, every model
#   bash run_generate_omni.sh --model qwen3-tts-0.6b --task in-the-wild
#   bash run_generate_omni.sh --model qwen3-tts-0.6b --task clean --test_set "en" --limit 3
#   bash run_generate_omni.sh --model fish-s2-pro --out_root results/_seeds/omni_fish-s2-pro
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      model key (see ALL_MODELS below) or "all"   (default all)
#   --task       task group (clean, in-the-wild, a panel)     (default clean)
#   --test_set   languages within --task (default: every language of the task)
#   --out_root   results root for a single --model     (default results/omni_<model>)
#   --gpu        GPU index for the server                    (default 0)
#   --port       server port                                 (default 8091)
#   --workers    concurrent requests per test set            (default 16)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
set -e

. scripts/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=all
task=clean
test_set=
out_root=""
gpu=0
port=8091
workers=16
limit=0
overwrite=false

help_message="usage: bash run_generate_omni.sh --model {all|<key>} [--task T] [--test_set \"...\"] [--out_root D] [--gpu N] [--port N] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

if [ -n "${out_root}" ] && [ "${model}" = "all" ]; then
    echo "error: --out_root needs a single --model" 1>&2
    exit 1
fi
names=$(gen_langs "${task}" "${test_set}") || exit 1

# Project venv: generate_omni.py (HTTP client) runs with <venv>/bin/python; the server
# is launched with <venv>/bin/vllm-omni. <venv> is OMNI_VENV (default .venv), e.g. another
# checkout's venv that has the omni extra installed. Both fall back to PATH if absent.
venv="${OMNI_VENV:-.venv}"
if [ -x "${venv}/bin/python" ]; then PY="${venv}/bin/python"; else PY=python; fi
if [ -x "${venv}/bin/vllm-omni" ]; then
    export OMNI_VLLM_BIN="${venv}/bin/vllm-omni"
else
    export OMNI_VLLM_BIN=vllm-omni
fi

# clone mode sends each reference wav to the server as a file:// URL; vLLM only
# loads local media under an explicitly allowed path. The prompt wavs live under
# the repo, so allow the repo root.
REPO_ROOT="$(pwd -P)"

# vLLM-Omni ships a deploy-config YAML per model family inside its package; locate it.
# find_spec resolves the path without importing the package: an import runs
# vLLM's start-up logging, which goes to stdout and would land in DEPLOY_DIR.
DEPLOY_DIR="$(${PY} -c 'import importlib.util,os;print(os.path.join(os.path.dirname(importlib.util.find_spec("vllm_omni").origin),"deploy"))' 2>/dev/null || true)"
if [ -z "${DEPLOY_DIR}" ] || [ ! -d "${DEPLOY_DIR}" ]; then
    echo "[warn] could not locate vllm_omni/deploy/ -- is vLLM-Omni installed in .venv?" 1>&2
    echo "[warn] proceeding without --deploy-config (vLLM-Omni will try to auto-detect)" 1>&2
    DEPLOY_DIR=""
fi

# --- model registry --------------------------------------------------------
# Shared via scripts/omni_models.sh: ALL_MODELS, NEW_MODELS, model_spec,
# reftext_policy, refaudio_mode, extra_pythonpath, extra_serve_args, extra_env,
# extra_client_args. Add or edit models THERE.
. scripts/omni_models.sh

if [ "${model}" = "all" ]; then models="${ALL_MODELS}"; else models="${model}"; fi
for m in ${models}; do
    if [ -z "$(model_spec ${m})" ]; then
        echo "error: unknown model '${m}'" 1>&2
        echo "known models: all ${ALL_MODELS}" 1>&2
        exit 1
    fi
done

# --language override per test set: generate_omni.py knows {en zh ja ko de es fr it
# ru}; the other codes fall through to "Auto" (it warns) and rely on detect_lang() /
# server auto-detect. Everything else is sent without --language.
lang_for() {
    case "$1" in
        en|zh|ja|ko|de|es|fr|it|ru|pt|ar|hi|tr|tl|pl|id|th) echo "$1" ;;
        *) echo "" ;;
    esac
}

# --- server lifecycle ------------------------------------------------------
# Set OMNI_HOST/OMNI_PORT/OMNI_GPU before sourcing serve_omni.sh (it builds URLs).
export OMNI_HOST=127.0.0.1
export OMNI_PORT="${port}"
export OMNI_GPU="${gpu}"
. scripts/serve_omni.sh
trap 'stop_omni_server' EXIT INT TERM
ran=0

# Snapshot PYTHONPATH so per-model additions (extra_pythonpath) can be reset each
# iteration instead of accumulating across models in a `--model all` sweep.
BASE_PYTHONPATH="${PYTHONPATH:-}"

for m in ${models}; do
    spec="$(model_spec ${m})"
    model_id="${spec%%|*}"
    rest="${spec#*|}"
    mode="${rest%%|*}"
    deploy_yaml="${rest#*|}"

    # Clone-mode delivery policy: prefer audio-only cloning where the server supports
    # it, and pick the ref-audio delivery each connector accepts (see the two policy
    # functions above). Both are no-ops for preset/design modes (no ref wav/text).
    nrt_opt=""; ra_opt=""
    if [ "${mode}" = clone ]; then
        nrt_opt="$(reftext_policy ${m})"
        ra_opt="--ref-audio-mode $(refaudio_mode ${m})"
        echo "[omni:${m}] clone delivery: ref-audio=$(refaudio_mode ${m}) ref-text=$([ -n "${nrt_opt}" ] && echo "OMITTED (${nrt_opt})" || echo "sent")"
    fi

    serve_args="--allowed-local-media-path ${REPO_ROOT} $(extra_serve_args ${m})"
    if [ -n "${DEPLOY_DIR}" ] && [ -n "${deploy_yaml}" ]; then
        if [ -f "${DEPLOY_DIR}/${deploy_yaml}" ]; then
            serve_args="${serve_args} --deploy-config ${DEPLOY_DIR}/${deploy_yaml}"
        else
            echo "[warn] ${m}: ${DEPLOY_DIR}/${deploy_yaml} not found;" \
                 "letting vLLM-Omni auto-detect the deploy config" 1>&2
        fi
    fi

    echo
    echo "############################################################"
    echo "# model: ${m}  (${model_id}, mode=${mode})"
    echo "#   vllm-omni serve ${model_id} --omni ${serve_args}"
    echo "############################################################"

    # Reset PYTHONPATH to the base, then prepend any source tree this model's server
    # needs (e.g. fish-s2-pro). The server inherits this env when start_omni_server
    # backgrounds it.
    epp="$(extra_pythonpath ${m})"
    if [ -n "${epp}" ]; then
        export PYTHONPATH="${epp}${BASE_PYTHONPATH:+:${BASE_PYTHONPATH}}"
        echo "[omni:${m}] PYTHONPATH += ${epp}"
    elif [ -n "${BASE_PYTHONPATH}" ]; then
        export PYTHONPATH="${BASE_PYTHONPATH}"
    else
        unset PYTHONPATH
    fi

    # Per-model extra env (e.g. higgs-audio-v3 needs DeepGEMM off); exported here so
    # the backgrounded server inherits it. Applies only to models that declare it.
    for kv in $(extra_env ${m}); do export "${kv?}"; echo "[omni:${m}] env ${kv}"; done

    export OMNI_MODEL="${model_id}"
    export OMNI_SERVE_ARGS="${serve_args}"
    export OMNI_LOG="${OMNI_LOG_DIR:-/tmp}/vllm_omni_${m}.log"
    if ! start_omni_server; then
        echo "[skip] ${m}: server failed to start (see ${OMNI_LOG})" 1>&2
        continue
    fi

    for nm in ${names}; do
        data_dir="data/${task}/${nm}"
        out_dir="${out_root:-results/omni_${m}}/${task}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lang_opt=""
        lg="$(lang_for ${nm})"
        [ -n "${lg}" ] && lang_opt="--language ${lg}"
        ow_opt=""
        [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

        echo "=== [omni:${m}] ${data_dir} -> ${out_dir} (mode=${mode}) ${lang_opt} ==="
        "${PY}" scripts/generate_omni.py \
            --server "http://127.0.0.1:${port}" \
            --mode "${mode}" \
            --model "${model_id}" \
            --data-dir "${data_dir}" \
            --out-dir "${out_dir}" \
            --workers "${workers}" \
            --limit "${limit}" \
            ${lang_opt} ${ow_opt} ${ra_opt} ${nrt_opt} $(extra_client_args ${m})
        ran=$((ran + 1))
    done

    stop_omni_server
done

gen_require_ran "${ran}" "${names}" "${task}"
for m in ${models}; do
    gen_done "${out_root:-results/omni_${m}}" "${task}" "${names}"
done
