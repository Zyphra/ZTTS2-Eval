#!/usr/bin/env bash
# Generate ZONOS3 Mini audio WITH the prompt transcript (continuation) plus the speaker embedding, in-process
# (scripts/gen/generate_zonos3_reftext.py on zonos2.tts.llm.TTSLLM; no server: /tts/generate cannot prefill the prompt's
# codec frames). Text = prompt transcript + target text; the prompt's frames (edge silence trimmed, top_db 35, last 2
# frames dropped: they carry the clip's end-of-file edge, which the LM reads as the utterance's end) follow
# the 0.2 s silence prefill; the output is the audio after the prefill. --ref-text false drops transcript and prompt
# frames (the in-process twin of the server's speaker_mode=embedding row). One process per call; writes
# <out_root>/<task>/<lang>/wavs/.
#
#   bash scripts/gen/run_zonos3_reftext.sh --task clean_mini --test-set "en zh" --limit 4
#   bash scripts/gen/run_zonos3_reftext.sh --task in-the-wild --gpu 2
#   GEN_DRIVER=scripts/gen/run_zonos3_reftext.sh GEN_MODEL=zonos3-mini-reftext_embedding \
#       sbatch --parsable --chdir="$(pwd -P)" --export=ALL --array=0-1 slurm/gen_job.sh
#
# Environment (required):
#   ZONOS3_HOME   minisgl checkout (its git rev is recorded in gen_meta.json)
#   ZONOS3_ENV    site env script, sourced before the run (sets PYTHON / ZONOS3_PY, PYTHONPATH, registries)
#   ZONOS3_MODEL  checkpoint (--model-path)
#   ZONOS3_PY     python; empty -> $PYTHON from ZONOS3_ENV
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --model       system name                (default zonos3-mini-reftext_embedding)
#   --task        task group                 (default clean)
#   --test-set    languages within --task    (default: every language of the task)
#   --list-langs  true = print the languages that would run and exit (default false)
#   --out-root    results root (default: gen_default_out_root; zonos3_reftext row: results/<model>)
#   --gpu         GPU index                  (default ${CUDA_VISIBLE_DEVICES:-0})
#   --ref-text    true = continuation from the prompt transcript + frames (default true)
#   --trim-prompt true = trim the prompt wav's edge silence before encoding its frames (default true)
#   --drop-prompt-frames  drop the last N codec frames of the prompt encoding (default 2)
#   --seed        sampling seed              (default 0)
#   --max-tokens  generated frames cap       (default 1350)
#   --batch       requests per engine call   (default 64)
#   --limit       only the first N utterances per set (default 0 = all)
#   --overwrite   regenerate existing wavs   (default false)
set -e

. scripts/gen/lib_gen.sh || { echo "error: run this from the repo root" 1>&2; exit 1; }

model=zonos3-mini-reftext_embedding
task=clean
test_set=
out_root=
list_langs=false
gpu=${CUDA_VISIBLE_DEVICES:-0}
ref_text=true
trim_prompt=true
drop_prompt_frames=2
seed=0
max_tokens=1350
batch=64
limit=0
overwrite=false

help_message="usage: bash scripts/gen/run_zonos3_reftext.sh [--model M] [--task T] [--test-set \"...\"] [--list-langs true] [--out-root D] [--gpu N] [--ref-text true|false] [--trim-prompt true|false] [--drop-prompt-frames N] [--seed N] [--max-tokens N] [--batch N] [--limit N] [--overwrite true]"
. utils/parse_options.sh || exit 1

# Resolved before the env rewrites PATH / PYTHONPATH.
GEN_NAME="${model}"
names=$(gen_langs "${task}" "${test_set}") || exit 1
gen_maybe_list "${list_langs}" "${names}"
gen_nothing_to_do "${names}"

for opt in ref_text trim_prompt overwrite; do
    case "${!opt}" in
        true|false) ;;
        *) echo "error: --${opt//_/-} must be true or false (got '${!opt}')" 1>&2; exit 2 ;;
    esac
done
[ -d "${ZONOS3_HOME:-}" ] || { echo "error: ZONOS3_HOME is not set" 1>&2; exit 1; }
[ -f "${ZONOS3_ENV:-}" ] || { echo "error: ZONOS3_ENV is not set" 1>&2; exit 1; }
[ -n "${ZONOS3_MODEL:-}" ] || { echo "error: ZONOS3_MODEL is not set" 1>&2; exit 1; }
[ -n "${out_root}" ] || out_root=$(gen_default_out_root zonos3_reftext "${model}")

# The single-rank engine's rendezvous defaults to :23333, shared by every process on the node.
port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
export ZONOS2_DISTRIBUTED_ADDR="tcp://127.0.0.1:${port}"
export CUDA_VISIBLE_DEVICES="${gpu}" ZONOS3_GPU="${gpu}"   # ZONOS3_GPU keys the env's per-GPU TMPDIR
ZONOS3_RUNTIME_REV=$(git -C "${ZONOS3_HOME}" rev-parse --short HEAD 2>/dev/null || true)
export ZONOS3_RUNTIME_REV
[ -n "${ZONOS3_RUNTIME_REV}" ] || echo "[warn] no git rev for ${ZONOS3_HOME} (gen_meta.json runtime_rev = null)" 1>&2

. "${ZONOS3_ENV}" || exit 1
py="${ZONOS3_PY:-${PYTHON:-}}"
[ -x "${py}" ] || { echo "error: no runtime python (ZONOS3_PY / PYTHON from ${ZONOS3_ENV}: '${py}')" 1>&2; exit 1; }
export ZONOS2_AUDIO_CODEC_DEVICE=cuda:0 ZONOS2_SPEAKER_EMBEDDER_DEVICE=cuda:0

ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"
rt_opt=""; [ "${ref_text}" = "false" ] && rt_opt="--no-ref-text"
tp_opt=""; [ "${trim_prompt}" = "false" ] && tp_opt="--no-trim-prompt"

ran=0
for nm in ${names}; do
    data_dir="data/${task}/${nm}"
    out_dir="${out_root}/${task}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2
        continue
    fi
    echo "=== [${model} (ref_text=${ref_text}, trim_prompt=${trim_prompt}, drop_prompt_frames=${drop_prompt_frames})] ${data_dir} -> ${out_dir} (runtime ${ZONOS3_RUNTIME_REV:-?}, gpu ${gpu}) ==="
    "${py}" scripts/gen/generate_zonos3_reftext.py \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --model-path "${ZONOS3_MODEL}" \
        ${seed:+--seed "${seed}"} \
        --max-tokens "${max_tokens}" \
        --batch "${batch}" \
        --limit "${limit}" \
        --drop-prompt-frames "${drop_prompt_frames}" \
        ${ow_opt} ${rt_opt} ${tp_opt}
    ran=$((ran + 1))
done

gen_require_ran "${ran}" "${names}" "${task}"

gen_done "${out_root}" "${task}" "${names}"
