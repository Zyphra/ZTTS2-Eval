#!/usr/bin/env bash
# Generate Phonon audio (gradium/phonon-*, Gradium's ptts Rust runtime on CPU, q8 GGUF)
# for the zero-shot benchmark. Preset-voice system: the open checkpoints cannot clone, so
# each utterance gets a catalog voice of its language (scripts/gen/generate_phonon.py). Runs
# from ~/.venvs/phonon (pip install ptts==0.6.0 soundfile numpy) with the checkpoint and
# the gradium/phonon-voices catalog pre-downloaded to the HF cache (loaded offline; the
# 40m checkpoints are gated, so accept their terms first). Writes
# <out_root>/<task>/<lang>/wavs/.
#
#   bash scripts/gen/run_phonon.sh                                  # clean, all 5 langs
#   bash scripts/gen/run_phonon.sh --task in-the-wild --test-set en --limit 3
#
# Options (Kaldi-style "--name value"):
#   --model      phonon-40m-multilingual                 (default phonon-40m-multilingual)
#   --task       clean | in-the-wild                    (default clean)
#   --test-set   langs to run (quote); '' = basket ∩ supported       (default '')
#   --out-root   results root   (default: gen_default_out_root; phonon row: results/<model>)
#   --seed       sampling seed; '' = generator default (4242424242424242)
#   --workers    model processes per leaf                               (default 16)
#   --threads    CPU threads per process                                 (default 2)
#   --limit      only the first N utterances per set              (default 0 = all)
#   --overwrite  regenerate existing wavs                          (default false)
#   --list-langs true: print the languages that would run and exit
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=phonon-40m-multilingual
task=clean
test_set=
out_root=
workers=16
threads=2
limit=0
overwrite=false
seed=
list_langs=false

help_message="usage: bash scripts/gen/run_phonon.sh [--model phonon-40m-multilingual] [--task T] [--test-set \"...\"] [--out-root D] [--workers N] [--threads N] [--seed N] [--limit N] [--overwrite true] [--list-langs true]"
. utils/parse_options.sh || exit 1

case "${model}" in
    phonon-40m-multilingual) repo=gradium/phonon-40m-multilingual; supported="en fr de es pt" ;;
    *) echo "error: unknown Phonon model '${model}' (phonon-40m-multilingual)" 1>&2; exit 1 ;;
esac
GEN_NAME="${model}"
names=$(gen_langs "${task}" "${test_set}" "${supported}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root phonon "${model}")
gen_venv_py PHONON_VENV "$HOME/.venvs/phonon" "pip install ptts==0.6.0 soundfile numpy into it"

export HF_HUB_OFFLINE=1
snapshot() {  # REPO -> local snapshot dir in the HF cache
    local d="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--${1/\//--}"
    [ -f "${d}/refs/main" ] || { echo "error: ${1} is not in the HF cache -- download it first" 1>&2; return 1; }
    echo "${d}/snapshots/$(cat "${d}/refs/main")"
}
model_dir="$(snapshot "${repo}")"
catalog_dir="$(snapshot gradium/phonon-voices)"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
    seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"
    echo "=== [${model}] ${data_dir} -> ${out_dir} (${model_dir##*/snapshots/}) ==="
    "${PY}" scripts/gen/generate_phonon.py \
        --model-dir "${model_dir}" --voice-catalog "${catalog_dir}" --language "${nm}" \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --workers "${workers}" --threads "${threads}" --limit "${limit}" ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
