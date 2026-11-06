#!/usr/bin/env bash
# Generate Ming-3.0-tts audio (native upstream, inclusionAI/Ming-3.0-tts) for the
# zero-shot benchmark. Runs scripts/gen/generate_ming3.py IN-PROCESS from the venvs and the
# pinned checkout of scripts/setup/setup_ming3_env.sh: one venv per checkpoint (their
# reference stacks differ). Clones with the prompt wav + transcript, for the seven
# languages the models support (zh en de es fr ja ko); other languages are skipped.
# The *-trim models trim the prompt wav's edge silence first (generate_ming3.py
# --trim-prompt): Ming continues from the prompt's tail, and the FLEURS-R prompts end in
# silence. Writes <out_root>/<task>/<lang>/wavs/.
#
#   bash scripts/gen/run_ming3.sh --model ming-3.0-tts-0.5b                # clean, 7 langs
#   bash scripts/gen/run_ming3.sh --model ming-3.0-tts-tiny --test-set en --limit 3
#   bash scripts/gen/run_ming3.sh --model ming-3.0-tts-0.5b --task in-the-wild
#
# Options (Kaldi-style "--name value"):
#   --model      ming-3.0-tts-{0.5b,tiny}[-trim]               (default ming-3.0-tts-0.5b)
#   --task       clean | in-the-wild                (default clean)
#   --test-set   langs to run (quote); '' = basket ∩ supported  (default '')
#   --out-root   results root   (default: gen_default_out_root; ming3 row: results/<model>)
#   --seed       sampling seed; '' = generator default (0.5b: 1895; tiny: fixed 1234)
#   --limit      only the first N utterances per set            (default 0 = all)
#   --overwrite  regenerate existing wavs                      (default false)
#   --list-langs true: print the languages that would run and exit
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=ming-3.0-tts-0.5b
task=clean
test_set=
out_root=
seed=
limit=0
overwrite=false
list_langs=false

help_message="usage: bash scripts/gen/run_ming3.sh [--model ming-3.0-tts-{0.5b,tiny}[-trim]] [--task T] [--test-set \"...\"] [--out-root D] [--seed N] [--limit N] [--overwrite true] [--list-langs true]"
. utils/parse_options.sh || exit 1
supported="zh en de es fr ja ko"
GEN_NAME="${model}"
names=$(gen_langs "${task}" "${test_set}" "${supported}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"

trim_opt=""
case "${model}" in
    ming-3.0-tts-0.5b-trim|ming-3.0-tts-tiny-trim) trim_opt="--trim-prompt" ;;
esac
case "${model%-trim}" in
    ming-3.0-tts-0.5b) variant=0.5b; venv_var=MING3_VENV; venv_dir="$HOME/.venvs/ming3" ;;
    ming-3.0-tts-tiny) variant=tiny; venv_var=MING3_TINY_VENV; venv_dir="$HOME/.venvs/ming3-tiny" ;;
    *) echo "error: unknown Ming-3.0 model '${model}' (ming-3.0-tts-{0.5b,tiny}[-trim])" 1>&2; exit 1 ;;
esac
[ -n "${out_root}" ] || out_root=$(gen_default_out_root ming3 "${model}")
SRC="${MING3_SRC:-third_party/Ming-3.0-tts}"
gen_venv_py "${venv_var}" "${venv_dir}" "run 'bash scripts/setup/setup_ming3_env.sh' first"
if [ ! -d "${SRC}" ]; then
    echo "error: no Ming-3.0 checkout at ${SRC} -- run 'bash scripts/setup/setup_ming3_env.sh' first" 1>&2
    exit 1
fi

# Models are pre-downloaded by setup_ming3_env.sh: load offline so a degraded hub never
# stalls model load.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
    seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"
    echo "=== [${model}] ${data_dir} -> ${out_dir} (variant ${variant}${trim_opt:+, prompt edge silence trimmed}, $(git -C "${SRC}" rev-parse --short HEAD)) ==="
    PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" scripts/gen/generate_ming3.py \
        --variant "${variant}" --language "${nm}" \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --limit "${limit}" ${trim_opt} ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
