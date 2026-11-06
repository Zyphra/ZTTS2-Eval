#!/usr/bin/env bash
# Generate Kokoro-82M audio for the zero-shot benchmark sets. Kokoro runs in-process
# (no server) via the dedicated venv from scripts/setup/setup_kokoro_env.sh.
#
# NOTE: Kokoro is a PRESET-VOICE model, not a zero-shot cloner -- it ignores the
# reference/prompt wav and speaks in a fixed built-in voice per language. So the
# speaker-similarity metric is not meaningful for it (WER and MSR-UTMOS are). It is
# a useful intelligibility/naturalness baseline, not a cloning entry.
#
# Kokoro covers a subset of the benchmark languages. Supported here:
#   en zh es fr it ja pt hi;  every other requested language is skipped.
#
# Writes wavs to <out-root>/<task>/<lang>/wavs/. Score with the command printed
# at the end (scripts/score/run_score.sh).
#
#   bash scripts/gen/run_kokoro.sh                              # clean, all supported langs
#   bash scripts/gen/run_kokoro.sh --test-set "en zh" --limit 3 # smoke test
#   bash scripts/gen/run_kokoro.sh --task in-the-wild
#   bash scripts/gen/run_kokoro.sh --test-set en --voice am_michael  # override the voice
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --task        clean | in-the-wild            (default clean)
#   --test-set    langs to run (quote); '' = basket ∩ supported (default '')
#   --list-langs  print the languages that would run, then exit (default false)
#   --voice       override the per-language default voice    (optional)
#   --speed       speaking-rate multiplier                  (default 1.0)
#   --device      torch device (cuda|cpu; default: auto)     (optional)
#   --seed        seed every RNG before each utterance      (default '' = unseeded)
#   --out-root    results root    (default: gen_default_out_root -> results/kokoro)
#   --limit       only the first N utterances per set        (default 0 = all)
#   --overwrite   regenerate existing wavs                  (default false)
# Venv: KOKORO_VENV (default ~/.venvs/kokoro).
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
test_set=
list_langs=false
voice=
speed=1.0
device=
seed=
out_root=
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_kokoro.sh [--task T] [--test-set \"...\"] [--list-langs true] [--voice V] [--speed X] [--device cuda|cpu] [--seed N] [--out-root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
GEN_NAME=kokoro
supported="en zh es fr it ja pt hi"
names=$(gen_langs "${task}" "${test_set}" "${supported}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root kokoro "")

gen_venv_py KOKORO_VENV "$HOME/.venvs/kokoro" "run 'bash scripts/setup/setup_kokoro_env.sh' first"

# lang (ISO 639-1) -> "<kokoro lang_code> <default voice>" for every supported
# language. Defaults are the top-ranked voice per language from the model card's VOICES.md.
map_for() {
    case "$1" in
        en) echo "a af_heart" ;;
        zh) echo "z zf_xiaobei" ;;
        es) echo "e ef_dora" ;;
        fr) echo "f ff_siwis" ;;
        it) echo "i if_sara" ;;
        ja) echo "j jf_alpha" ;;
        pt) echo "p pf_dora" ;;
        hi) echo "h hf_alpha" ;;
    esac
}

dev_opt=""; [ -n "${device}" ] && dev_opt="--device ${device}"
ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"

ran=0
for nm in ${names}; do
    m="$(map_for ${nm})"
    lang_code="${m%% *}"
    def_voice="${m##* }"
    use_voice="${voice:-${def_voice}}"

    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi

    echo "=== [kokoro] ${data_dir} -> ${out_dir}  (lang_code=${lang_code} voice=${use_voice}) ==="
    "${PY}" scripts/gen/generate_kokoro.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --lang-code "${lang_code}" \
        --voice "${use_voice}" \
        --speed "${speed}" \
        --limit "${limit}" \
        ${dev_opt} ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
echo "note: speaker-similarity is not meaningful for Kokoro (preset voice, no cloning); read WER + MSR-UTMOS."
