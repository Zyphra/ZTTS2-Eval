#!/usr/bin/env bash
# Generate CosyVoice3 audio (native upstream) for the zero-shot benchmark. Runs the
# official FunAudioLLM/CosyVoice code IN-PROCESS via the dedicated venv from
# scripts/setup_cosyvoice_env.sh -- NOT vLLM-Omni (that cosyvoice3 path was too slow).
#
# CosyVoice3 is a WITH-reference-text zero-shot cloner and is multilingual (text-in,
# no per-language code needed), so it covers all 9 clean languages. Writes wavs to
# results/cosyvoice3/<task>/<lang>/wavs/. Score with run_infer_zonos2_eval.sh.
#
#   bash run_generate_cosyvoice.sh                          # clean, all 9 langs
#   bash run_generate_cosyvoice.sh --test_set "en zh" --limit 3   # smoke test
#   bash run_generate_cosyvoice.sh --task in-the-wild
#   bash run_generate_cosyvoice.sh --fp16 true              # ~2x faster, lower precision
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --task       clean | in-the-wild            (default clean)
#   --test_set   langs to run (quote); '' = task default    (default '')
#   --fp16       load the model in fp16                     (default false)
#   --speed      speaking-rate multiplier                  (default 1.0)
#   --out_root   results root                              (default results/cosyvoice3)
#   --model_dir  CosyVoice3 model dir (HF id or snapshot)   (default: cached snapshot)
#   --limit      only the first N utterances per set        (default 0 = all)
#   --overwrite  regenerate existing wavs                  (default false)
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

task=clean
test_set=
fp16=false
speed=1.0
out_root="results/cosyvoice3"
model_dir=
limit=0
overwrite=false

help_message="usage: bash run_generate_cosyvoice.sh [--task T] [--test_set \"...\"] [--fp16 true] [--speed X] [--out_root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

VENV="${VENV:-$HOME/.venvs/cosyvoice}"
PY="${VENV}/bin/python"
COSY_SRC="${COSYVOICE_SRC:-third_party/CosyVoice}"
if [ ! -x "${PY}" ]; then
    echo "error: no CosyVoice venv at ${VENV} -- run 'bash scripts/setup_cosyvoice_env.sh' first (or set VENV)" 1>&2
    exit 1
fi

# The 9 clean langs (CosyVoice3 is multilingual; no language flag needed).
sets_for() {
    case "$1" in
        clean)          echo "en zh de es fr it ja ko ru" ;;
        in-the-wild) echo "en zh de es fr it ja ko ru" ;;
        *) echo "" ;;
    esac
}
names="${test_set:-$(sets_for ${task})}"
if [ -z "${names}" ]; then
    echo "error: unknown task '${task}'" 1>&2; exit 1
fi

md_opt=""; [ -n "${model_dir}" ] && md_opt="--model-dir ${model_dir}"
fp16_opt=""; [ "${fp16}" = "true" ] && fp16_opt="--fp16"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
    # CosyVoice3 requires katakana input for Japanese (else ~60% WER); auto-enable for ja.
    kata_opt=""; [ "${nm}" = ja ] && kata_opt="--katakana"
    echo "=== [cosyvoice3] ${data_dir} -> ${out_dir} ${kata_opt} ==="
    PYTHONPATH="${COSY_SRC}:${COSY_SRC}/third_party/Matcha-TTS:${PYTHONPATH:-}" \
        "${PY}" scripts/generate_cosyvoice.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --speed "${speed}" \
        --limit "${limit}" \
        ${fp16_opt} ${md_opt} ${kata_opt} ${ow_opt}
    ran=$((ran + 1))
done

[ "${ran}" -eq 0 ] && { echo "error: nothing generated for '${names}'" 1>&2; exit 1; }

echo
echo "generation done -> ${out_root}/${task}/"
echo "score with:  DECODE_DIR=${out_root}/${task} bash run_infer_zonos2_eval.sh --task ${task} --test_set \"${names}\""
