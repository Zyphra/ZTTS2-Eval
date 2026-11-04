#!/usr/bin/env bash
# Generate Phonon audio (gradium/phonon-*, Gradium's ptts Rust runtime on CPU, q8 GGUF)
# for the zero-shot benchmark. Preset-voice system: the open checkpoints cannot clone, so
# each utterance gets a catalog voice of its language (scripts/generate_phonon.py). Runs
# from ~/.venvs/phonon (pip install ptts==0.6.0 soundfile numpy) with the checkpoint and
# the gradium/phonon-voices catalog pre-downloaded to the HF cache (loaded offline; the
# 40m checkpoints are gated, so accept their terms first). Writes
# <out_root>/<task>/<lang>/wavs/.
#
#   bash run_generate_phonon.sh                                  # zero_shot, all 5 langs
#   bash run_generate_phonon.sh --task zero_shot_voxblink --test_set en --limit 3
#
# Options (Kaldi-style "--name value"):
#   --model      phonon-40m-multilingual                 (default phonon-40m-multilingual)
#   --task       zero_shot | zero_shot_voxblink                    (default zero_shot)
#   --test_set   langs to run (quote); '' = all 5 supported          (default '')
#   --out_root   results root                                (default results/<model>)
#   --workers    model processes per leaf                               (default 16)
#   --threads    CPU threads per process                                 (default 2)
#   --limit      only the first N utterances per set              (default 0 = all)
#   --overwrite  regenerate existing wavs                          (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

model=phonon-40m-multilingual
task=zero_shot
test_set=
out_root=
workers=16
threads=2
limit=0
overwrite=false

help_message="usage: bash run_generate_phonon.sh [--model phonon-40m-multilingual] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--threads N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    phonon-40m-multilingual) repo=gradium/phonon-40m-multilingual; supported="en fr de es pt" ;;
    *) echo "error: unknown Phonon model '${model}' (phonon-40m-multilingual)" 1>&2; exit 1 ;;
esac
case "${task}" in
    zero_shot|zero_shot_voxblink) ;;
    *) echo "error: unsupported task '${task}' (zero_shot | zero_shot_voxblink)" 1>&2; exit 1 ;;
esac
out_root="${out_root:-results/${model}}"
PY="${PHONON_VENV:-$HOME/.venvs/phonon}/bin/python"
[ -x "${PY}" ] || { echo "error: no Phonon venv at ${PY%/bin/python} (pip install ptts==0.6.0 soundfile numpy)" 1>&2; exit 1; }

export HF_HUB_OFFLINE=1
snapshot() {  # REPO -> local snapshot dir in the HF cache
    local d="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--${1/\//--}"
    [ -f "${d}/refs/main" ] || { echo "error: ${1} is not in the HF cache -- download it first" 1>&2; return 1; }
    echo "${d}/snapshots/$(cat "${d}/refs/main")"
}
model_dir="$(snapshot "${repo}")"
catalog_dir="$(snapshot gradium/phonon-voices)"

names="${test_set:-${supported}}"
for nm in ${names}; do
    case " ${supported} " in
        *" ${nm} "*) ;;
        *) echo "error: ${model} does not support '${nm}' (supported: ${supported})" 1>&2; exit 1 ;;
    esac
done

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
    echo "=== [${model}] ${data_dir} -> ${out_dir} (${model_dir##*/snapshots/}) ==="
    "${PY}" scripts/generate_phonon.py \
        --model-dir "${model_dir}" --voice-catalog "${catalog_dir}" --language "${nm}" \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --workers "${workers}" --threads "${threads}" --limit "${limit}" ${ow_opt}
    ran=$((ran + 1))
done

[ "${ran}" -eq 0 ] && { echo "error: nothing generated for '${names}'" 1>&2; exit 1; }

echo
echo "generation done -> ${out_root}/${task}/"
