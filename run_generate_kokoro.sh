#!/usr/bin/env bash
# Generate Kokoro-82M audio for the zero-shot benchmark sets. Kokoro runs in-process
# (no server) via the dedicated venv from scripts/setup_kokoro_env.sh.
#
# NOTE: Kokoro is a PRESET-VOICE model, not a zero-shot cloner -- it ignores the
# reference/prompt wav and speaks in a fixed built-in voice per language. So the
# speaker-similarity metric is not meaningful for it (WER and MSR-UTMOS are). It is
# a useful intelligibility/naturalness baseline, not a cloning entry.
#
# Kokoro covers a subset of the benchmark languages. Supported here:
#   en zh es fr it ja  (+ pt hi for in-the-wild);  de ko ru are NOT supported and skipped.
#
# Writes wavs to results/kokoro/<task>/<lang>/wavs/. Score with the command printed
# at the end (run_score.sh).
#
#   bash run_generate_kokoro.sh                              # clean, all supported langs
#   bash run_generate_kokoro.sh --test_set "en zh" --limit 3 # smoke test
#   bash run_generate_kokoro.sh --task in-the-wild
#   bash run_generate_kokoro.sh --test_set en --voice am_michael  # override the voice
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --task       clean | in-the-wild            (default clean)
#   --test_set   langs to run (quote); '' = basket ∩ supported (default '')
#   --voice      override the per-language default voice    (optional)
#   --speed      speaking-rate multiplier                  (default 1.0)
#   --device     torch device (cuda|cpu; default: auto)     (optional)
#   --out_root   results root                              (default results/kokoro)
#   --limit      only the first N utterances per set        (default 0 = all)
#   --overwrite  regenerate existing wavs                  (default false)
set -e

. scripts/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
test_set=
voice=
speed=1.0
device=
out_root="results/kokoro"
limit=0
overwrite=false

help_message="usage: bash run_generate_kokoro.sh [--task T] [--test_set \"...\"] [--voice V] [--speed X] [--device cuda|cpu] [--out_root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
ALL_SUPPORTED="en zh es fr it ja pt hi"
names=$(gen_langs "${task}" "${test_set}" "${ALL_SUPPORTED}") || exit 1

VENV="${VENV:-$HOME/.venvs/kokoro}"
PY="${VENV}/bin/python"
if [ ! -x "${PY}" ]; then
    echo "error: no Kokoro venv at ${VENV} -- run 'bash scripts/setup_kokoro_env.sh' first (or set VENV)" 1>&2
    exit 1
fi

# lang (ISO 639-1) -> "<kokoro lang_code> <default voice>". Unmapped langs (de ko ru
# and the rest) are unsupported by Kokoro and skipped. Defaults are the top-ranked
# voice per language from the model card's VOICES.md.
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
        *)  echo "" ;;
    esac
}

ran=0
for nm in ${names}; do
    m="$(map_for ${nm})"
    if [ -z "${m}" ]; then
        echo "[skip] ${nm}: not supported by Kokoro (no lang_code) -- skipping" 1>&2
        continue
    fi
    lang_code="${m%% *}"
    def_voice="${m##* }"
    use_voice="${voice:-${def_voice}}"

    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    dev_opt=""; [ -n "${device}" ] && dev_opt="--device ${device}"
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

    echo "=== [kokoro] ${data_dir} -> ${out_dir}  (lang_code=${lang_code} voice=${use_voice}) ==="
    "${PY}" scripts/generate_kokoro.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --lang-code "${lang_code}" \
        --voice "${use_voice}" \
        --speed "${speed}" \
        --limit "${limit}" \
        ${dev_opt} ${ow_opt}
    ran=$((ran + 1))
done

if [ "${ran}" -eq 0 ]; then
    echo "error: no supported languages in '${names}' (Kokoro supports: ${ALL_SUPPORTED})" 1>&2
    exit 1
fi

gen_done "${out_root}" "${task}" "${names}"
echo "note: speaker-similarity is not meaningful for Kokoro (preset voice, no cloning); read WER + MSR-UTMOS."
