#!/usr/bin/env bash
# Generate audio for the VoxBlink2-derived eval (data/in-the-wild) with
# vLLM-Omni TTS models. Starts and stops the server itself; mirrors
# run_generate_omni.sh but pinned to the one new task.
#
#   bash run_generate_omni_voxblink.sh --model all              # 9 zs-overlap langs
#   bash run_generate_omni_voxblink.sh --model qwen3-tts-0.6b
#   bash run_generate_omni_voxblink.sh --model qwen3-tts-0.6b --test_set en --limit 3
#
# Score afterwards with run_infer_voxblink_eval.sh and DECODE_DIR pointing at
# results/omni_<model>/in-the-wild. OMNI_VENV and OMNI_OUT_ROOT work as in
# run_generate_omni.sh.
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root" 1>&2; exit 1
fi

model=all
# Default to the 9 langs shared with clean; the noisy voxblink-only tail
# (pt ar hi tr tl pl id th yo nl uk vi) is off by default. Re-enable via the full list or
# an explicit --test_set.
#   full 21: "en zh de es fr it ja ko ru pt ar hi tr tl pl id th yo nl uk vi"
test_set="en zh de es fr it ja ko ru"
gpu=0
port=8091
workers=16     # vLLM-Omni batches; bump for ~1.6k-prompt run
limit=0
overwrite=false
. utils/parse_options.sh || exit 1

venv="${OMNI_VENV:-.venv}"
if [ -x "${venv}/bin/python" ]; then PY="${venv}/bin/python"; else PY=python; fi
if [ -x "${venv}/bin/vllm-omni" ]; then
    export OMNI_VLLM_BIN="${venv}/bin/vllm-omni"
else
    export OMNI_VLLM_BIN=vllm-omni
fi
REPO_ROOT="$(pwd -P)"

# find_spec: no package import, so vLLM's stdout start-up logging stays out.
DEPLOY_DIR="$(${PY} -c 'import importlib.util,os;print(os.path.join(os.path.dirname(importlib.util.find_spec("vllm_omni").origin),"deploy"))' 2>/dev/null || true)"
if [ -z "${DEPLOY_DIR}" ] || [ ! -d "${DEPLOY_DIR}" ]; then
    DEPLOY_DIR=""
fi

# Shared model registry (same file as run_generate_omni.sh) -- ALL_MODELS,
# model_spec, reftext_policy, refaudio_mode, extra_pythonpath, extra_serve_args,
# extra_env, extra_client_args.
. scripts/omni_models.sh

if [ "${model}" = "all" ]; then models="${ALL_MODELS}"; else models="${model}"; fi
for m in ${models}; do
    if [ -z "$(model_spec ${m})" ]; then
        echo "error: unknown model '${m}' (known: all ${ALL_MODELS})" 1>&2; exit 1
    fi
done

# generate_omni.py knows {en zh ja ko de es fr it ru}; others fall through to
# "Auto" and rely on detect_lang() / server auto-detect. Pass ISO codes anyway
# so the warning lands on the unsupported ones.
lang_for() {
    case "$1" in
        en|zh|ja|ko|de|es|fr|it|ru) echo "$1" ;;
        pt|ar|hi|tr|tl|pl|id|th)    echo "$1" ;;   # generate_omni.py warns + uses Auto
        *) echo "" ;;
    esac
}

export OMNI_HOST=127.0.0.1
export OMNI_PORT="${port}"
export OMNI_GPU="${gpu}"
. scripts/serve_omni.sh
trap 'stop_omni_server' EXIT INT TERM

# Snapshot PYTHONPATH so per-model extra_pythonpath additions reset each iteration.
BASE_PYTHONPATH="${PYTHONPATH:-}"

for m in ${models}; do
    spec="$(model_spec ${m})"
    model_id="${spec%%|*}"
    rest="${spec#*|}"
    mode="${rest%%|*}"
    deploy_yaml="${rest#*|}"

    # clone-mode delivery policy (same as the clean driver): audio-only cloning
    # where supported + the ref-audio delivery each connector accepts. No-op for preset.
    nrt_opt=""; ra_opt=""
    if [ "${mode}" = clone ]; then
        nrt_opt="$(reftext_policy ${m})"
        ra_opt="--ref-audio-mode $(refaudio_mode ${m})"
        echo "[omni:${m}] clone delivery: ref-audio=$(refaudio_mode ${m}) ref-text=$([ -n "${nrt_opt}" ] && echo "OMITTED (${nrt_opt})" || echo "sent")"
    fi

    serve_args="--allowed-local-media-path ${REPO_ROOT} $(extra_serve_args ${m})"
    if [ -n "${DEPLOY_DIR}" ] && [ -n "${deploy_yaml}" ] && [ -f "${DEPLOY_DIR}/${deploy_yaml}" ]; then
        serve_args="${serve_args} --deploy-config ${DEPLOY_DIR}/${deploy_yaml}"
    fi

    echo
    echo "############################################################"
    echo "# model: ${m}  (${model_id}, mode=${mode})"
    echo "############################################################"

    # Reset PYTHONPATH, then prepend any source tree this model's server needs
    # (e.g. fish-s2-pro's fish_speech tree), mirroring run_generate_omni.sh.
    epp="$(extra_pythonpath ${m})"
    if [ -n "${epp}" ]; then
        export PYTHONPATH="${epp}${BASE_PYTHONPATH:+:${BASE_PYTHONPATH}}"
        echo "[omni:${m}] PYTHONPATH += ${epp}"
    elif [ -n "${BASE_PYTHONPATH}" ]; then
        export PYTHONPATH="${BASE_PYTHONPATH}"
    else
        unset PYTHONPATH
    fi

    # Per-model extra env (e.g. higgs-audio-v3 needs DeepGEMM off); see run_generate_omni.sh.
    for kv in $(extra_env ${m}); do export "${kv?}"; echo "[omni:${m}] env ${kv}"; done

    export OMNI_MODEL="${model_id}"
    export OMNI_SERVE_ARGS="${serve_args}"
    export OMNI_LOG="${OMNI_LOG_DIR:-/tmp}/vllm_omni_${m}.log"
    if ! start_omni_server; then
        echo "[skip] ${m}: server failed (see ${OMNI_LOG})" 1>&2
        continue
    fi

    for nm in ${test_set}; do
        data_dir="data/in-the-wild/${nm}"
        out_dir="${OMNI_OUT_ROOT:-results}/omni_${m}/in-the-wild/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
        fi
        lang_opt=""
        lg="$(lang_for ${nm})"
        [ -n "${lg}" ] && lang_opt="--language ${lg}"
        ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

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

    stop_omni_server
done

echo
echo "generation done -> results/omni_<model>/in-the-wild/"
echo "score with: DECODE_DIR=results/omni_<model>/in-the-wild bash run_infer_voxblink_eval.sh"
