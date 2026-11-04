#!/usr/bin/env bash
# Generate Kyutai Pocket TTS audio (official `pocket-tts` package, CPU) for the zero-shot
# benchmark: audio-only cloning of the prompt wav, one checkpoint per language
# (scripts/generate_pocket_tts.py). Runs from ~/.venvs/pocket-tts (pocket-tts with the
# CPU torch wheel; see that script) with the gated kyutai/pocket-tts weights
# pre-downloaded to the HF cache (loaded offline). Writes <out_root>/<task>/<lang>/wavs/.
#
#   bash run_generate_pocket_tts.sh                               # zero_shot, all 6 langs
#   bash run_generate_pocket_tts.sh --task zero_shot_voxblink --test_set en --limit 3
#
# Options (Kaldi-style "--name value"):
#   --model      pocket-tts                                          (default pocket-tts)
#   --task       zero_shot | zero_shot_voxblink                              (default zero_shot)
#   --test_set   langs to run (quote); '' = all 6 supported          (default '')
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

model=pocket-tts
task=zero_shot
test_set=
out_root=
workers=16
threads=2
limit=0
overwrite=false

help_message="usage: bash run_generate_pocket_tts.sh [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--threads N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

[ "${model}" = pocket-tts ] || { echo "error: unknown model '${model}' (pocket-tts)" 1>&2; exit 1; }
case "${task}" in
    zero_shot|zero_shot_voxblink) ;;
    *) echo "error: unsupported task '${task}' (zero_shot | zero_shot_voxblink)" 1>&2; exit 1 ;;
esac
out_root="${out_root:-results/${model}}"
PY="${POCKET_TTS_VENV:-$HOME/.venvs/pocket-tts}/bin/python"
[ -x "${PY}" ] || { echo "error: no Pocket TTS venv at ${PY%/bin/python}" 1>&2; exit 1; }
export HF_HUB_OFFLINE=1

supported="en fr de pt it es"
names="${test_set:-${supported}}"
for nm in ${names}; do
    case " ${supported} " in
        *" ${nm} "*) ;;
        *) echo "error: Pocket TTS does not support '${nm}' (supported: ${supported})" 1>&2; exit 1 ;;
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
    echo "=== [${model}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/generate_pocket_tts.py --language "${nm}" \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --workers "${workers}" --threads "${threads}" --limit "${limit}" ${ow_opt}
    ran=$((ran + 1))
done

[ "${ran}" -eq 0 ] && { echo "error: nothing generated for '${names}'" 1>&2; exit 1; }

echo
echo "generation done -> ${out_root}/${task}/"
