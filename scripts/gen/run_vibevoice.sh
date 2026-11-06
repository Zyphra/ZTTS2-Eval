#!/usr/bin/env bash
# Generate VibeVoice-Realtime-0.5B (Microsoft) audio for the zero-shot benchmark.
# In-process via the dedicated venv (scripts/setup/setup_vibevoice_env.sh), no server.
#
# VibeVoice-Realtime is a PRESET-voice streaming model (fixed speakers), NOT a
# zero-shot cloner -- speaker-sim is not meaningful (WER + MSR-UTMOS are).
# Writes results/vibevoice-rt-0.5b/<task>/<lang>/.
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
#   bash scripts/gen/run_vibevoice.sh --test_set "en de"         # a subset
#   bash scripts/gen/run_vibevoice.sh --limit 3                  # smoke
#
# Options: --task (clean | in-the-wild | panel), --test_set (default = task basket ∩ supported),
#          --speaker (rotate|Carter|...|Spk0|Spk1), --out_root, --limit, --overwrite.
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
test_set=
speaker=rotate
out_root="results/vibevoice-rt-0.5b"
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_vibevoice.sh [--task T] [--test_set \"en de\"] [--speaker S] [--out_root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
ALL_LANGS="en de es fr it ja ko nl pl pt"
names=$(gen_langs "${task}" "${test_set}" "${ALL_LANGS}") || exit 1

# Our language code -> preset-voice file prefix in demo/voices/streaming_model.
# Empty result = no preset ships for that language (e.g. ru, zh) -> skip it.
voice_for() {
    case "$1" in
        en) echo en ;; de) echo de ;; fr) echo fr ;; it) echo it ;; nl) echo nl ;;
        pl) echo pl ;; pt) echo pt ;;
        ja) echo jp ;;   # upstream names it jp-*
        ko) echo kr ;;   # upstream names it kr-*
        es) echo sp ;;   # upstream names it sp-*
        *)  echo "" ;;
    esac
}

VENV="${VENV:-$HOME/.venvs/vibevoice}"
PY="${VENV}/bin/python"
VV_SRC="${VIBEVOICE_SRC:-third_party/VibeVoice}"
if [ ! -x "${PY}" ]; then
    echo "error: no VibeVoice venv at ${VENV} -- run 'bash scripts/setup/setup_vibevoice_env.sh' first" 1>&2
    exit 1
fi

# Resolve the language list: an explicit --test_set is filtered down to the languages
# a preset voice exists for (the SLURM launcher passes whole-task lists that include
# ru/zh/ar/...); an empty --test_set means the task basket ∩ supported languages.
langs=""
for l in ${names}; do
    v="$(voice_for "${l}")"
    if [ -z "${v}" ]; then
        [ -n "${test_set}" ] && echo "[vibevoice] skip '${l}': no preset voice ships for it" 1>&2
        continue
    fi
    [ -f "data/${task}/${l}/text" ] || { echo "[vibevoice] skip '${l}': no data/${task}/${l}/text" 1>&2; continue; }
    langs="${langs} ${l}"
done
langs="$(echo ${langs})"
if [ -z "${langs}" ]; then
    echo "[vibevoice] nothing to generate for task=${task} test_set='${test_set}'" 1>&2
    exit 0
fi

# A fixed --speaker names one voice inside ONE language's preset set, so it cannot
# apply across a multi-language run. 'rotate' is the only valid multi-language choice.
nlang=$(echo ${langs} | wc -w)
if [ "${speaker}" != rotate ] && [ "${nlang}" -gt 1 ]; then
    echo "error: --speaker '${speaker}' needs a single --test_set language (got: ${langs})" 1>&2
    exit 1
fi

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

for l in ${langs}; do
    v="$(voice_for "${l}")"
    data_dir="data/${task}/${l}"; out_dir="${out_root}/${task}/${l}"
    echo "=== [vibevoice] ${data_dir} -> ${out_dir}  (voice=${v}-* speaker=${speaker}) ==="
    PYTHONPATH="${VV_SRC}:${PYTHONPATH:-}" "${PY}" scripts/gen/generate_vibevoice.py \
        --data-dir "${data_dir}" --out-dir "${out_dir}" \
        --voice-lang "${v}" --speaker "${speaker}" --limit "${limit}" ${ow_opt}
done

gen_done "${out_root}" "${task}" "${langs}"
echo "note: speaker-similarity is not meaningful for VibeVoice (preset voice); read WER + MSR-UTMOS."
echo "note: only 'en' is a supported language upstream; the other nine are vendor-flagged exploratory."
