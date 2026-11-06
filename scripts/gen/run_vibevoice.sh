#!/usr/bin/env bash
# Generate VibeVoice-Realtime-0.5B (Microsoft) audio for the zero-shot benchmark.
# In-process via the dedicated venv (scripts/setup/setup_vibevoice_env.sh), no server.
#
# VibeVoice-Realtime is a PRESET-voice streaming model (fixed speakers), NOT a
# zero-shot cloner -- speaker-sim is not meaningful (WER + MSR-UTMOS are).
# Writes <out-root>/<task>/<lang>/.
#
# LANGUAGES: the model card's frontmatter says `language: [en]` and its limitations
# section says "trained only on English data", but the body ships preset voices for
# nine more -- German, French, Italian, Japanese, Korean, Dutch, Polish, Portuguese,
# Spanish -- "for users to explore". Those .pt files are in demo/voices/streaming_model
# of the upstream checkout, so we run them. Treat non-en numbers as vendor-flagged
# exploratory, not as a supported multilingual claim.
#
# The voice-file prefixes are NOT our language codes: ja->jp, ko->kr, es->sp. That
# mapping lives in voice_for() below. (demo/voices also ships an undocumented
# `in-Samuel_man`; it is not one of the nine named languages, so we leave it out.)
#
#   bash scripts/gen/run_vibevoice.sh                            # clean, all 10 langs
#   bash scripts/gen/run_vibevoice.sh --test-set "en de"         # a subset
#   bash scripts/gen/run_vibevoice.sh --limit 3                  # smoke
#
# Options: --task (clean | in-the-wild | panel), --test-set (default = task basket ∩ supported),
#          --list-langs, --speaker (rotate|Carter|...|Spk0|Spk1), --seed (default '' = unseeded),
#          --out-root (default: gen_default_out_root -> results/vibevoice-rt-0.5b),
#          --limit, --overwrite.
# Venv: VIBEVOICE_VENV (default ~/.venvs/vibevoice).
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
test_set=
list_langs=false
speaker=rotate
seed=
out_root=
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_vibevoice.sh [--task T] [--test-set \"en de\"] [--list-langs true] [--speaker S] [--seed N] [--out-root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
GEN_NAME=vibevoice
supported="en de es fr it ja ko nl pl pt"
names=$(gen_langs "${task}" "${test_set}" "${supported}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root vibevoice "")

# Our language code -> preset-voice file prefix in demo/voices/streaming_model.
voice_for() {
    case "$1" in
        en) echo en ;; de) echo de ;; fr) echo fr ;; it) echo it ;; nl) echo nl ;;
        pl) echo pl ;; pt) echo pt ;;
        ja) echo jp ;;   # upstream names it jp-*
        ko) echo kr ;;   # upstream names it kr-*
        es) echo sp ;;   # upstream names it sp-*
    esac
}

# A fixed --speaker names one voice inside ONE language's preset set, so it cannot
# apply across a multi-language run. 'rotate' is the only valid multi-language choice.
nlang=$(echo ${names} | wc -w)
if [ "${speaker}" != rotate ] && [ "${nlang}" -gt 1 ]; then
    echo "error: --speaker '${speaker}' needs a single --test-set language (got: ${names})" 1>&2
    exit 1
fi

gen_venv_py VIBEVOICE_VENV "$HOME/.venvs/vibevoice" "run 'bash scripts/setup/setup_vibevoice_env.sh' first"
VV_SRC="${VIBEVOICE_SRC:-third_party/VibeVoice}"

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"

ran=0
for l in ${names}; do
    v="$(voice_for "${l}")"
    data_dir="data/${task}/${l}"; out_dir="${out_root}/${task}/${l}"
    [ -f "${data_dir}/text" ] || { echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue; }
    echo "=== [vibevoice] ${data_dir} -> ${out_dir}  (voice=${v}-* speaker=${speaker}) ==="
    PYTHONPATH="${VV_SRC}:${PYTHONPATH:-}" "${PY}" scripts/gen/generate_vibevoice.py \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --voice-lang "${v}" --speaker "${speaker}" --limit "${limit}" ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
echo "note: speaker-similarity is not meaningful for VibeVoice (preset voice); read WER + MSR-UTMOS."
echo "note: only 'en' is a supported language upstream; the other nine are vendor-flagged exploratory."
