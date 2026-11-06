#!/usr/bin/env bash
# Generate Kyutai Pocket TTS audio (official `pocket-tts` package, CPU) for the zero-shot
# benchmark: audio-only cloning of the prompt wav, one checkpoint per language
# (scripts/gen/generate_pocket_tts.py). Runs from ~/.venvs/pocket-tts (pocket-tts with the
# CPU torch wheel; see that script) with the gated kyutai/pocket-tts weights
# pre-downloaded to the HF cache (loaded offline). Writes <out_root>/<task>/<lang>/wavs/.
#
#   bash scripts/gen/run_pocket_tts.sh                               # clean, all 6 langs
#   bash scripts/gen/run_pocket_tts.sh --task in-the-wild --test-set en --limit 3
#
# Options (Kaldi-style "--name value"):
#   --model      pocket-tts                                          (default pocket-tts)
#   --task       clean | in-the-wild                              (default clean)
#   --test-set   langs to run (quote); '' = basket ∩ supported       (default '')
#   --out-root   results root   (default: gen_default_out_root; pocket_tts row: results/<model>)
#   --seed       sampling seed; '' = generator default (42)
#   --workers    model processes per leaf                               (default 16)
#   --threads    CPU threads per process                                 (default 2)
#   --limit      only the first N utterances per set              (default 0 = all)
#   --overwrite  regenerate existing wavs                          (default false)
#   --list-langs true: print the languages that would run and exit
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=pocket-tts
task=clean
test_set=
out_root=
workers=16
threads=2
limit=0
overwrite=false
seed=
list_langs=false

help_message="usage: bash scripts/gen/run_pocket_tts.sh [--task T] [--test-set \"...\"] [--out-root D] [--workers N] [--threads N] [--seed N] [--limit N] [--overwrite true] [--list-langs true]"
. utils/parse_options.sh || exit 1

[ "${model}" = pocket-tts ] || { echo "error: unknown model '${model}' (pocket-tts)" 1>&2; exit 1; }
supported="en fr de pt it es"
GEN_NAME="Pocket TTS"
names=$(gen_langs "${task}" "${test_set}" "${supported}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root pocket_tts "${model}")
gen_venv_py POCKET_TTS_VENV "$HOME/.venvs/pocket-tts" "install pocket-tts with the CPU torch wheel into it"
export HF_HUB_OFFLINE=1
ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
    seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"
    echo "=== [${model}] ${data_dir} -> ${out_dir} ==="
    "${PY}" scripts/gen/generate_pocket_tts.py --language "${nm}" \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --workers "${workers}" --threads "${threads}" --limit "${limit}" ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
