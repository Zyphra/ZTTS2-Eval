#!/usr/bin/env bash
# Re-score the zero_shot/en leaves that were left as smoke-test stubs: the wavs are
# complete (1279) but only 3-8 utterances ever got scored, and the resume guard in
# run_infer_zonos2_eval.sh ("WER: line present AND spk file non-empty") means a normal
# re-run skips them forever. Deletes the stale score files and re-scores all three
# metrics. No regeneration -- the audio is already on disk.
#
#   results/omni_qwen3-tts-0.6b/zero_shot/en     wer=3  spk=4  msr=3   wavs=1279
#   results/omni_omnivoice-reftext/zero_shot/en  wer=8  spk=9  msr=8   wavs=1279
#   results/omni_glm-tts/zero_shot/en            wer=8  spk=9  msr=8   wavs=1279
#
#   sbatch --job-name=ztts-rescore-en --gpus=1 \
#          --cpus-per-task=28 --mem=200G --time=4:00:00 \
#          --chdir="$(pwd)" --output=logs/slurm/ztts-rescore-en-%j.out scripts/rescore_stale_en.sh
#
# Override the target list with RESCORE_MODELS. `en` is in QWEN_LANGS, so it always
# routes to Qwen3-ASR -- no Whisper server is needed here (unlike rescore_routed.sh).
# NB: set +u -- serve_asr.sh references optional unset vars, same as rescore_routed.sh.
set +u
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[rescore-en] ERROR: not at repo root"; exit 1; }

MODELS="${RESCORE_MODELS:-omni_qwen3-tts-0.6b omni_omnivoice-reftext omni_glm-tts}"
LANG_=en
TASK=zero_shot

job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 20000 + (job_id % 10000) * 4 ))
export ASR_PORT=$(( port_base + 1 ))
export ASR_GPU=0
srv="logs/slurm/servers/rescore-en-${job_id}"; mkdir -p "${srv}"
export ASR_LOG="${srv}/asr.log"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

. scripts/serve_asr.sh
trap 'stop_asr_server' EXIT INT TERM
start_asr_server || { echo "[rescore-en] ERROR: ASR server failed"; exit 1; }

n_ok=0; n_skip=0
for m in ${MODELS}; do
    dd="results/${m}/${TASK}/${LANG_}"
    txt="data/${TASK}/${LANG_}/text"
    if [ ! -d "${dd}/wavs" ]; then echo "[skip] ${m} (no wavs)"; n_skip=$((n_skip+1)); continue; fi
    n_wav=$(find "${dd}/wavs" -name '*.wav' | wc -l)
    if [ "${n_wav}" -lt 100 ]; then echo "[skip] ${m} (only ${n_wav} wavs)"; n_skip=$((n_skip+1)); continue; fi

    echo "=== rescore ${m}/${LANG_} (${n_wav} wavs) ==="
    # Clear every artifact the resume guards key on, so all three metrics re-run.
    rm -f "${dd}/wav_res_ref_text" "${dd}/wav_res_ref_text.wer" "${dd}/wav_res_ref_text.wer.raw" \
          "${dd}/spk_simi_scores.txt" \
          "${dd}/msr_utmos_scores.txt" "${dd}/msr_utmos_mean.txt" "${dd}/msr_utmos_ci.txt"

    bash scripts/run_score_wer.sh "${txt}" "${dd}" "${LANG_}" || { echo "[rescore-en] WER failed: ${m}"; continue; }

    find "${dd}/wavs" -name '*.wav' | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > "${dd}/wav.scp"

    "${PY}" scripts/eval_speaker_similarity.py \
        --prompt_wavs "data/${TASK}/${LANG_}/prompt_wav.scp" \
        --hyp_wavs "${dd}/wav.scp" \
        --log_file "${dd}/spk_simi_scores.txt" \
        --device cuda || { echo "[rescore-en] spk failed: ${m}"; continue; }

    "${PY}" utils/score_msr_utmos.py \
        -t "${dd}/wav.scp" \
        -o "${dd}/msr_utmos_scores.txt" \
        -m "${dd}/msr_utmos_mean.txt" || { echo "[rescore-en] msr failed: ${m}"; continue; }

    n_ok=$((n_ok+1))
done
echo "[rescore-en] done: ${n_ok} rescored, ${n_skip} skipped"
