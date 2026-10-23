#!/usr/bin/env bash
# Generate VibeVoice-1.5B (Microsoft) audio for the zero-shot benchmark.
# In-process via the dedicated venv (scripts/setup_vibevoice15_env.sh), no server.
#
# VibeVoice-1.5B is a long-form multi-speaker ZERO-SHOT CLONER -- it takes a reference
# wav per utterance (prompt_wav.scp), so unlike the realtime 0.5B preset-voice recipe
# SPEAKER-SIMILARITY IS MEANINGFUL here, alongside WER + MSR-UTMOS.
# Writes results/vibevoice-1.5b/<task>/<lang>/.
#
# LANGUAGES: English and Chinese only. The HF card tags `en`/`zh` and its limitations
# say "Transcripts in languages other than English or Chinese may result in unexpected
# audio outputs". Any other language in --test_set is skipped with a reason.
#
# The TTS inference code is VENDORED (upstream withdrew it) -- see
# third_party/VibeVoice-1.5B/PROVENANCE.md. It is used via PYTHONPATH and must never
# share a path with third_party/VibeVoice (realtime 0.5B): both define the same module
# names at different versions.
#
#   bash run_generate_vibevoice15.sh                      # zero_shot, en + zh
#   bash run_generate_vibevoice15.sh --test_set "en"      # one language
#   bash run_generate_vibevoice15.sh --limit 3            # smoke
#
# Options: --task (zero_shot|zero_shot_voxblink), --test_set (default = en zh),
#          --cfg_scale, --ddpm_steps, --out_root, --limit, --overwrite.
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root" 1>&2; exit 1
fi

task=zero_shot
model=1.5b
test_set=
cfg_scale=1.3
ddpm_steps=10
out_root=            # empty = per-model default (set after --model is parsed)
allow_unsupported=false
limit=0
overwrite=false

help_message="usage: bash run_generate_vibevoice15.sh [--model 1.5b|7b] [--task T] [--allow_unsupported true] [--test_set \"en zh\"] [--cfg_scale F] [--out_root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

# Both variants share the vendored inference code and this venv -- same architecture
# (VibeVoiceForConditionalGeneration), same processor, only the backbone size differs
# (Qwen2.5-1.5B vs 7B). NOTE the 7B comes from the THIRD-PARTY `vibevoice` org, not
# `microsoft`: Microsoft withdrew VibeVoice-Large entirely (docs list its weights as
# "Disabled"), so unlike the 1.5B there is no official copy to verify against. Its
# weights are safetensors, so loading cannot execute code. See PROVENANCE.md.
case "${model}" in
    1.5b) hf_id="microsoft/VibeVoice-1.5B";  def_root="results/vibevoice-1.5b" ;;
    7b)   hf_id="vibevoice/VibeVoice-7B";    def_root="results/vibevoice-7b" ;;
    *)    echo "error: --model must be 1.5b or 7b (got '${model}')" 1>&2; exit 1 ;;
esac
[ -z "${out_root}" ] && out_root="${def_root}"

# Supported-language set, per variant, decided by MEASUREMENT not by the model card.
# The card says en/zh for both sizes, but a 50-utt/lang probe over 10 diverse languages
# (scripts/probe_vibevoice_langs.sh, 2026-08-09) showed that only holds for the 1.5B:
#
#   7B   at or BELOW the GT ASR floor on it (-1.18), ru (-0.84), hi (-2.37); within
#        2.3 pts on es/pt/ar/de/fr; worst were ko (+5.4) and ja (+8.8). Duration ratio
#        0.83-1.04 everywhere -- no runaway. So the 7B is genuinely multilingual, which
#        matches upstream's own FAQ ("strong cross-lingual transfer ... emergent").
#   1.5B degrades as the card warns: ja 35.99, ar 49.42, fr 14.20. Kept at en/zh.
#
# Re-probe before widening this for a new variant; --allow_unsupported forces anything.
case "${model}" in
    7b)  ALL_LANGS="$(ls "data/${task}" 2>/dev/null | tr '\n' ' ')" ;;
    *)   ALL_LANGS="en zh" ;;
esac

VENV="${VENV:-$HOME/.venvs/vibevoice15}"
PY="${VENV}/bin/python"
VV15_SRC="${VIBEVOICE15_SRC:-third_party/VibeVoice-1.5B}"
if [ ! -x "${PY}" ]; then
    echo "error: no VibeVoice-1.5B venv at ${VENV} -- run 'bash scripts/setup_vibevoice15_env.sh' first" 1>&2
    exit 1
fi

# Resolve the language list. The SLURM launcher passes whole-task lists, so filter an
# explicit --test_set down to the supported languages rather than failing on the rest.
langs=""
for l in ${test_set:-${ALL_LANGS}}; do
    case " ${ALL_LANGS} " in
        *" ${l} "*) ;;
        *) if [ "${allow_unsupported}" = true ]; then
               # Exploratory only. The card says en/zh; other languages are UNSUPPORTED
               # and the model may run away (generate far more audio than the text
               # warrants). Compare output duration against the GT set before trusting
               # any WER -- a duration ratio >> 1 means runaway, not accented speech.
               echo "[vibevoice15] '${l}': UNSUPPORTED language, running anyway (--allow_unsupported)" 1>&2
           else
               [ -n "${test_set}" ] && echo "[vibevoice15] skip '${l}': not in VibeVoice-${model}'s supported set (use --allow_unsupported true to probe)" 1>&2
               continue
           fi ;;
    esac
    [ -f "data/${task}/${l}/text" ] || { echo "[vibevoice15] skip '${l}': no data/${task}/${l}/text" 1>&2; continue; }
    [ -f "data/${task}/${l}/prompt_wav.scp" ] || { echo "[vibevoice15] skip '${l}': no prompt_wav.scp (clone mode needs refs)" 1>&2; continue; }
    langs="${langs} ${l}"
done
langs="$(echo ${langs})"
if [ -z "${langs}" ]; then
    echo "[vibevoice15] nothing to generate for task=${task} test_set='${test_set}'" 1>&2
    exit 0
fi

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

for l in ${langs}; do
    data_dir="data/${task}/${l}"; out_dir="${out_root}/${task}/${l}"
    echo "=== [vibevoice-${model}] ${data_dir} -> ${out_dir}  (${hf_id}, clone, cfg=${cfg_scale} ddpm=${ddpm_steps}) ==="
    PYTHONPATH="${VV15_SRC}:${PYTHONPATH:-}" "${PY}" scripts/generate_vibevoice15.py \
        --data-dir "${data_dir}" --out-dir "${out_dir}" --model-dir "${hf_id}" \
        --cfg-scale "${cfg_scale}" --ddpm-steps "${ddpm_steps}" --limit "${limit}" ${ow_opt}
done

echo
echo "generation done -> ${out_root}/${task}/ (${langs})"
echo "score with:  DECODE_DIR=${out_root}/${task} bash run_infer_zonos2_eval.sh --task ${task} --test_set \"${langs}\""
echo "note: VibeVoice-${model} IS a cloner -- speaker-similarity is meaningful (unlike vibevoice-rt-0.5b)."
if [ "${model}" = 7b ]; then
    echo "note: 7B weights come from the third-party 'vibevoice' org (Microsoft withdrew VibeVoice-Large); see third_party/VibeVoice-1.5B/PROVENANCE.md."
fi
