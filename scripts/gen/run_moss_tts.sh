#!/usr/bin/env bash
# Generate MOSS-TTS audio (native upstream) for the zero-shot benchmark. Runs the
# official OpenMOSS/MOSS-TTS code IN-PROCESS via the dedicated venv from
# scripts/setup/setup_moss_tts_env.sh -- NOT vLLM-Omni (its moss_tts codec is weight-
# incompatible with the published MOSS-Audio-Tokenizer; the upstream processor
# carries its own matching audio_tokenizer).
#
# MOSS-TTS (8B MossTTSDelay) is a multilingual zero-shot cloner; clone mode
# conditions on the reference wav's timbre (audio only), so it covers all 9
# clean langs. Writes <out-root>/<task>/<lang>/wavs/. Score with the command
# printed at the end (scripts/score/run_score.sh).
#
#   bash scripts/gen/run_moss_tts.sh                          # clean, all 9 langs
#   bash scripts/gen/run_moss_tts.sh --test-set "en zh" --limit 3   # smoke test
#   bash scripts/gen/run_moss_tts.sh --task in-the-wild
#
# Options (Kaldi-style "--name value"):
#   --model       moss-tts | moss-tts-v1.5 | moss-tts-local | moss-tts-local-v1.5 (default moss-tts)
#   --model-dir   explicit HF id / path (wins over --model)  (default '')
#   --task        clean | in-the-wild            (default clean)
#   --test-set    langs to run (quote); '' = task default    (default '')
#   --list-langs  print the languages that would run, then exit (default false)
#   --seed        seed every RNG before each utterance      (default '' = unseeded)
#   --out-root    results root    (default: gen_default_out_root -> results/<model>)
#   --limit       only the first N utterances per set        (default 0 = all)
#   --overwrite   regenerate existing wavs                  (default false)
# Venv: MOSS_TTS_VENV (default ~/.venvs/moss-tts).
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=moss-tts       # variant key (see moss_id below); all share the one venv
model_dir=           # explicit HF id / path override (wins over --model)
task=clean
test_set=
list_langs=false
seed=
out_root=
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_moss_tts.sh [--model VARIANT] [--model-dir D] [--task T] [--test-set \"...\"] [--list-langs true] [--seed N] [--out-root D] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1
GEN_NAME=${model}
names=$(gen_langs "${task}" "${test_set}" "") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"

# MOSS-TTS family variants -> HF model id. All run through the same native path
# (AutoModel/AutoProcessor); the Local architectures ship their own modeling
# code loaded via trust_remote_code.
moss_id() {
    case "$1" in
        moss-tts)            echo "OpenMOSS-Team/MOSS-TTS" ;;
        moss-tts-v1.5)       echo "OpenMOSS-Team/MOSS-TTS-v1.5" ;;
        moss-tts-local)      echo "OpenMOSS-Team/MOSS-TTS-Local-Transformer" ;;
        moss-tts-local-v1.5) echo "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5" ;;
        *)                   echo "" ;;
    esac
}
mid="${model_dir:-$(moss_id ${model})}"
[ -z "${mid}" ] && { echo "error: unknown moss variant '${model}' (moss-tts | moss-tts-v1.5 | moss-tts-local | moss-tts-local-v1.5)" 1>&2; exit 1; }
[ -n "${out_root}" ] || out_root=$(gen_default_out_root moss_tts "${model}")

# Models are pre-downloaded (setup / manual). Force HF offline so from_pretrained
# never stalls on a flaky HF-hub connection during load (a non-offline job hung 28 min
# at "loading" while the hub was degraded). Set MOSS_ALLOW_DOWNLOAD=1 to re-enable
# network fetches (e.g. first run of an un-cached variant).
if [ "${MOSS_ALLOW_DOWNLOAD:-0}" != 1 ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

gen_venv_py MOSS_TTS_VENV "$HOME/.venvs/moss-tts" "run 'bash scripts/setup/setup_moss_tts_env.sh' first"
MOSS_SRC="${MOSS_TTS_SRC:-third_party/MOSS-TTS}"
seed_opt=""; [ -n "${seed}" ] && seed_opt="--seed ${seed}"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
    # MOSS-TTS-Local-v1.5's config requests flash_attention_2 (not installed) -> sdpa.
    attn_opt=""; [ "${model}" = moss-tts-local-v1.5 ] && attn_opt="--attn sdpa"
    echo "=== [${model}] ${data_dir} -> ${out_dir}  (${mid}) ${attn_opt} ==="
    PYTHONPATH="${MOSS_SRC}:${PYTHONPATH:-}" "${PY}" scripts/gen/generate_moss_tts.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --model-dir "${mid}" \
        --limit "${limit}" \
        ${attn_opt} ${seed_opt} ${ow_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
