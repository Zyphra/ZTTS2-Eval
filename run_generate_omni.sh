#!/usr/bin/env bash
# Generate audio with vLLM-Omni TTS models -- starts/stops the server itself.
#
# For each selected model this:
#   1. starts a vLLM-Omni server   (scripts/serve_omni.sh -> `vllm-omni serve --omni`)
#   2. generates every test set through it   (scripts/generate_omni.py)
#   3. stops the server, then moves to the next model
# A trap stops the running server on success, error, or Ctrl-C.
#
# Generation only -- score the output afterwards with run_infer_zonos2_eval.sh.
#
# Requires vLLM-Omni (the `vllm-omni` CLI), installed in this repo's .venv (Python
# 3.12) or in the venv named by OMNI_VENV. generate_omni.py is just an HTTP client and
# runs from the same venv.
#
#   bash run_generate_omni.sh --model all
#   bash run_generate_omni.sh --model qwen3-tts-0.6b
#   bash run_generate_omni.sh --model qwen3-tts-0.6b --task zero_shot --test_set "en" --limit 3
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model      model key (see ALL_MODELS below) or "all"   (default all)
#   --task       restrict to one task group       (optional)
#   --test_set   restrict to names within --task             (optional; needs --task)
#   --gpu        GPU index for the server                    (default 0)
#   --port       server port                                 (default 8091)
#   --workers    concurrent requests per test set            (default 4)
#   --limit      only the first N utterances per set          (default 0 = all)
#   --overwrite  regenerate existing wavs                     (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=all
task=
test_set=
gpu=0
port=8091
workers=4
limit=0
overwrite=false

help_message="usage: bash run_generate_omni.sh --model {all|<key>} [--task T] [--test_set \"...\"] [--gpu N] [--port N] [--workers N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

if [ -n "${test_set}" ] && [ -z "${task}" ]; then
    echo "error: --test_set requires --task" 1>&2
    exit 1
fi

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
# Shared with run_generate_omni_voxblink.sh via scripts/omni_models.sh: ALL_MODELS,
# NEW_MODELS, model_spec, reftext_policy, refaudio_mode, extra_pythonpath,
# extra_serve_args, extra_env, extra_client_args. Add or edit models THERE so both
# tasks stay in sync.
. scripts/omni_models.sh

if [ "${model}" = "all" ]; then models="${ALL_MODELS}"; else models="${model}"; fi
for m in ${models}; do
    if [ -z "$(model_spec ${m})" ]; then
        echo "error: unknown model '${m}'" 1>&2
        echo "known models: all ${ALL_MODELS}" 1>&2
        exit 1
    fi
done

# --- test sets --------------------------------------------------
sets_for() {
    case "$1" in
        zero_shot)              echo "zh en ja ko de es fr it ru" ;;
        *) echo "" ;;
    esac
}

# --language override for each test set (all current sets have an explicit
# entry); the generator falls back to per-utterance auto-detect for any miss.
lang_for() {
    case "$1/$2" in
        zero_shot/zh) echo zh ;;
        zero_shot/en) echo en ;;
        zero_shot/ja) echo ja ;;
        zero_shot/ko) echo ko ;;
        zero_shot/de) echo de ;;
        zero_shot/es) echo es ;;
        zero_shot/fr) echo fr ;;
        zero_shot/it) echo it ;;
        zero_shot/ru) echo ru ;;
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

tasks="${task:-zero_shot}"

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

    for tk in ${tasks}; do
        names="${test_set:-$(sets_for ${tk})}"
        if [ -z "${names}" ]; then
            echo "error: unknown task '${tk}'" 1>&2
            stop_omni_server
            exit 1
        fi
        for nm in ${names}; do
            data_dir="data/${tk}/${nm}"
            # OMNI_OUT_ROOT overrides the results root for control/variant runs
            # (e.g. the 2026-08 seed study writes under results/_seeds/) without
            # touching the published tree. Default is the published layout.
            out_dir="${OMNI_OUT_ROOT:-results}/omni_${m}/${tk}/${nm}"
            if [ ! -f "${data_dir}/text" ]; then
                echo "[skip] ${data_dir}: no 'text' file" 1>&2
                continue
            fi

            lang_opt=""
            lg="$(lang_for ${tk} ${nm})"
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
        done
    done

    stop_omni_server
done

echo
echo "generation done -> results/omni_<model>/"
echo "score it with run_infer_zonos2_eval.sh, e.g.:"
echo "  DECODE_DIR=results/omni_qwen3-tts-0.6b/zero_shot bash run_infer_zonos2_eval.sh"
