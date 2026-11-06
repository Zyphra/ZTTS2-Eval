#!/usr/bin/env bash
# Generate VibeVoice-1.5B / -7B audio for the zero-shot benchmark.
# In-process via the dedicated venv (scripts/setup/setup_vibevoice15_env.sh), no server.
#
# VibeVoice-1.5B is a long-form multi-speaker ZERO-SHOT CLONER -- it takes a reference
# wav per utterance (prompt_wav.scp), so unlike the realtime 0.5B preset-voice recipe
# SPEAKER-SIMILARITY IS MEANINGFUL here, alongside WER + MSR-UTMOS.
# Writes <out-root>/<task>/<lang>/.
#
# LANGUAGES: 1.5b runs English and Chinese only (see the measured supported sets
# below); 7b runs the task's full basket. Any other language in --test-set is skipped.
#
# The TTS inference code is VENDORED (upstream withdrew it) -- see
# third_party/VibeVoice-1.5B/PROVENANCE.md. It is used via PYTHONPATH and must never
# share a path with third_party/VibeVoice (realtime 0.5B): both define the same module
# names at different versions.
#
#   bash scripts/gen/run_vibevoice15.sh                      # clean, en + zh
#   bash scripts/gen/run_vibevoice15.sh --test-set "en"      # one language
#   bash scripts/gen/run_vibevoice15.sh --limit 3            # smoke
#
# Options: --model (1.5b | 7b), --task (clean | in-the-wild | panel),
#          --test-set (default = basket ∩ en zh; 7b: basket), --list-langs,
#          --allow-unsupported (true: no language filter), --cfg-scale, --ddpm-steps,
#          --seed (default '' = unseeded),
#          --out-root (default: gen_default_out_root -> results/vibevoice-<model>),
#          --limit, --overwrite.
# Venv: VIBEVOICE15_VENV (default ~/.venvs/vibevoice15).
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

task=clean
model=1.5b
test_set=
list_langs=false
cfg_scale=1.3
ddpm_steps=10
seed=
out_root=
allow_unsupported=false
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_vibevoice15.sh [--model 1.5b|7b] [--task T] [--allow-unsupported true] [--test-set \"en zh\"] [--list-langs true] [--cfg-scale F] [--ddpm-steps N] [--seed N] [--out-root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

# Both variants share the vendored inference code and this venv -- same architecture
# (VibeVoiceForConditionalGeneration), same processor, only the backbone size differs
# (Qwen2.5-1.5B vs 7B). NOTE the 7B comes from the THIRD-PARTY `vibevoice` org, not
# `microsoft`: Microsoft withdrew VibeVoice-Large entirely (docs list its weights as
# "Disabled"), so unlike the 1.5B there is no official copy to verify against. Its
# weights are safetensors, so loading cannot execute code. See PROVENANCE.md.
case "${model}" in
    1.5b) hf_id="microsoft/VibeVoice-1.5B" ;;
    7b)   hf_id="vibevoice/VibeVoice-7B" ;;
    *)    echo "error: --model must be 1.5b or 7b (got '${model}')" 1>&2; exit 1 ;;
esac

# Supported-language set, per variant, decided by MEASUREMENT not by the model card.
# The card says en/zh for both sizes, but a 50-utt/lang probe over 10 diverse languages
# (scripts/analysis/probe_vibevoice_langs.sh, 2026-08-09) showed that only holds for the 1.5B:
#
#   7B   at or BELOW the GT ASR floor on it (-1.18), ru (-0.84), hi (-2.37); within
#        2.3 pts on es/pt/ar/de/fr; worst were ko (+5.4) and ja (+8.8). Duration ratio
#        0.83-1.04 everywhere -- no runaway. So the 7B is genuinely multilingual, which
#        matches upstream's own FAQ ("strong cross-lingual transfer ... emergent").
#   1.5B degrades as the card warns: ja 35.99, ar 49.42, fr 14.20. Kept at en/zh.
#
# Re-probe before widening this for a new variant; --allow-unsupported true forces anything.
# Exploratory only: unsupported languages may run away (generate far more audio than the
# text warrants). Compare output duration against the GT set before trusting any WER --
# a duration ratio >> 1 means runaway, not accented speech.
case "${model}" in
    7b)  supported="" ;;          # empty = the task's full basket, no filter
    *)   supported="en zh" ;;
esac
[ "${allow_unsupported}" = true ] && supported=""
GEN_NAME="vibevoice-${model}"
names=$(gen_langs "${task}" "${test_set}" "${supported}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"
[ -n "${out_root}" ] || out_root=$(gen_default_out_root vibevoice15 "${model}")

gen_venv_py VIBEVOICE15_VENV "$HOME/.venvs/vibevoice15" "run 'bash scripts/setup/setup_vibevoice15_env.sh' first"
VV15_SRC="${VIBEVOICE15_SRC:-third_party/VibeVoice-1.5B}"

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"

ran=0
for l in ${names}; do
    data_dir="data/${task}/${l}"; out_dir="${out_root}/${task}/${l}"
    [ -f "${data_dir}/text" ] || { echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue; }
    [ -f "${data_dir}/prompt_wav.scp" ] || { echo "[skip] ${data_dir}: no prompt_wav.scp (clone mode needs refs)" 1>&2; continue; }
    echo "=== [vibevoice-${model}] ${data_dir} -> ${out_dir}  (${hf_id}, clone, cfg=${cfg_scale} ddpm=${ddpm_steps}) ==="
    PYTHONPATH="${VV15_SRC}:${PYTHONPATH:-}" "${PY}" scripts/gen/generate_vibevoice15.py \
        --data-dir "${data_dir}" --out-dir "${out_dir}" --model-dir "${hf_id}" \
        --cfg-scale "${cfg_scale}" --ddpm-steps "${ddpm_steps}" --limit "${limit}" ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
echo "note: VibeVoice-${model} IS a cloner -- speaker-similarity is meaningful (unlike vibevoice-rt-0.5b)."
if [ "${model}" = 7b ]; then
    echo "note: 7B weights come from the third-party 'vibevoice' org (Microsoft withdrew VibeVoice-Large); see third_party/VibeVoice-1.5B/PROVENANCE.md."
fi
