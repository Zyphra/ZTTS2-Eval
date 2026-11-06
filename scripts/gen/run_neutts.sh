#!/usr/bin/env bash
# Generate NeuTTS-2E (Neuphonic) audio for the zero-shot benchmark. Runs in-process
# via the dedicated venv from scripts/setup/setup_neutts_env.sh (no server).
#
# NeuTTS-2E is a PRESET-voice model (4 fixed speakers x 7 emotions), NOT a zero-shot
# cloner -- speaker-similarity is not meaningful (WER + MSR-UTMOS are). It is
# ENGLISH ONLY, so only the `en` set is generated. Writes <out-root>/<task>/en/.
#
#   bash scripts/gen/run_neutts.sh                       # clean/en (+ in-the-wild/en via --task)
#   bash scripts/gen/run_neutts.sh --limit 3             # smoke test
#   bash scripts/gen/run_neutts.sh --speaker emily --emotion neutral
#
# Options (Kaldi-style "--name value"):
#   --task        clean | in-the-wild        (default clean)
#   --test-set    langs to run (quote); only en is supported  (default '' = en)
#   --list-langs  print the languages that would run, then exit (default false)
#   --speaker     emily|paul|sophie|steven | rotate     (default rotate)
#   --emotion     neutral|happy|sad|angry|...           (default neutral)
#   --seed        NeuTTS2E sampling seed                (default '' = generator default 0)
#   --out-root    results root    (default: gen_default_out_root -> results/neutts-2e)
#   --limit       only the first N utterances           (default 0 = all)
#   --overwrite   regenerate existing wavs              (default false)
# Venv: NEUTTS_VENV (default ~/.venvs/neutts).
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
test_set=
list_langs=false
speaker=rotate
emotion=neutral
seed=
out_root=
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_neutts.sh [--task T] [--test-set \"...\"] [--list-langs true] [--speaker S] [--emotion E] [--seed N] [--out-root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
GEN_NAME=neutts-2e
names=$(gen_langs "${task}" "${test_set}" "en") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root neutts "")

gen_venv_py NEUTTS_VENV "$HOME/.venvs/neutts" "run 'bash scripts/setup/setup_neutts_env.sh' first"

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    echo "=== [neutts-2e] ${data_dir} -> ${out_dir}  (speaker=${speaker} emotion=${emotion}) ==="
    "${PY}" scripts/gen/generate_neutts.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --speaker "${speaker}" \
        --emotion "${emotion}" \
        --limit "${limit}" \
        ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
echo "note: speaker-similarity is not meaningful for NeuTTS-2E (preset voice); read WER + MSR-UTMOS."
