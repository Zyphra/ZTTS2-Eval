#!/usr/bin/env bash
# Generate CosyVoice3 audio (native upstream) for the zero-shot benchmark. Runs the
# official FunAudioLLM/CosyVoice code IN-PROCESS via the dedicated venv from
# scripts/setup/setup_cosyvoice_env.sh -- NOT vLLM-Omni (that cosyvoice3 path was too slow).
#
# CosyVoice3 is a WITH-reference-text zero-shot cloner and is multilingual (text-in,
# no per-language code needed), so it covers all 9 clean languages. Writes wavs to
# <out-root>/<task>/<lang>/wavs/. Score with the command printed at the end (scripts/score/run_score.sh).
#
#   bash scripts/gen/run_cosyvoice.sh                          # clean, all 9 langs
#   bash scripts/gen/run_cosyvoice.sh --test-set "en zh" --limit 3   # smoke test
#   bash scripts/gen/run_cosyvoice.sh --task in-the-wild
#   bash scripts/gen/run_cosyvoice.sh --fp16 true              # ~2x faster, lower precision
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --task        clean | in-the-wild            (default clean)
#   --test-set    langs to run (quote); '' = task default    (default '')
#   --list-langs  print the languages that would run, then exit (default false)
#   --fp16        load the model in fp16                     (default false)
#   --speed       speaking-rate multiplier                  (default 1.0)
#   --seed        seed every RNG before each utterance      (default '' = unseeded)
#   --out-root    results root    (default: gen_default_out_root -> results/cosyvoice3)
#   --model-dir   CosyVoice3 model dir (HF id or snapshot)   (default: cached snapshot)
#   --limit       only the first N utterances per set        (default 0 = all)
#   --overwrite   regenerate existing wavs                  (default false)
# Venv: COSYVOICE_VENV (default ~/.venvs/cosyvoice).
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
test_set=
list_langs=false
fp16=false
speed=1.0
seed=
out_root=
model_dir=
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_cosyvoice.sh [--task T] [--test-set \"...\"] [--list-langs true] [--fp16 true] [--speed X] [--seed N] [--out-root D] [--model-dir D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
GEN_NAME=cosyvoice3
names=$(gen_langs "${task}" "${test_set}" "") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root cosyvoice "")

gen_venv_py COSYVOICE_VENV "$HOME/.venvs/cosyvoice" "run 'bash scripts/setup/setup_cosyvoice_env.sh' first"
COSY_SRC="${COSYVOICE_SRC:-third_party/CosyVoice}"

md_opt=""; [ -n "${model_dir}" ] && md_opt="--model-dir ${model_dir}"
fp16_opt=""; [ "${fp16}" = "true" ] && fp16_opt="--fp16"
seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"

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
        "${PY}" scripts/gen/generate_cosyvoice.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --speed "${speed}" \
        --limit "${limit}" \
        ${fp16_opt} ${md_opt} ${kata_opt} ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
