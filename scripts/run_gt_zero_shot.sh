#!/usr/bin/env bash
# Ground-truth baseline for the clean (FLEURS-R) zero_shot set: score the real
# human PROMPT clips against prompt_text. WER = the ASR error floor on genuine
# human speech; MSR-UTMOS = the real-audio quality ceiling. Speaker similarity is
# omitted (prompt vs itself is degenerate ~100). Output mirrors the voxblink GT:
#   results/gt_zero_shot/<lang>/{wavs/, wav.scp, wav_res_ref_text.wer, msr_utmos_mean.txt}
#
# The clean set retains prompt audio but not target audio, so — like the CV3-Eval
# baseline — the GT is measured on the prompt clips (still genuine FLEURS-R speech
# of the same distribution) rather than the synthesized target text.
#
#   bash scripts/run_gt_zero_shot.sh                      # all zero_shot langs
#   bash scripts/run_gt_zero_shot.sh --test_set "en zh"
set +u
cd "$(dirname "$0")/.." || exit 1

task=zero_shot
# Default: every language dir under data/zero_shot with prompt audio.
test_set=$(for d in data/${task}/*/; do
    l=$(basename "$d"); [ -f "${d}/prompt_wav.scp" ] && printf '%s ' "$l"; done)
. utils/parse_options.sh || exit 1

OUT_ROOT=${DECODE_DIR:-results/gt_${task}}
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper   # per-lang routing when ASR_ROUTING=1 / WHISPER_GPU set (else no-op)

# Pick a GPU for the MSR-UTMOS torch model (>=5 GB free, not the ASR GPU).
SCORE_GPU="${SCORE_GPU:-}"
if [ -z "${SCORE_GPU}" ]; then
    SCORE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', ' '$2>=5000 && $1!='"${ASR_GPU:-99}"'{print $1; exit}')
fi
[ -z "${SCORE_GPU}" ] && SCORE_GPU=0
echo "[score] MSR-UTMOS on GPU ${SCORE_GPU}"

for lang in ${test_set}; do
    data=data/${task}/${lang}
    sd=${OUT_ROOT}/${lang}
    [ -f "${data}/prompt_wav.scp" ] || { echo "[skip] ${lang} (no prompt_wav.scp)"; continue; }

    if grep -q '^WER:' "${sd}/wav_res_ref_text.wer" 2>/dev/null \
       && [ -s "${sd}/msr_utmos_mean.txt" ]; then
        echo "[skip] ${lang} already scored"
        continue
    fi

    mkdir -p "${sd}/wavs"
    # Bridge uid -> prompt wav: get_wav_res_ref_text expects wavs/<uid>.wav, but the
    # files are named prompt_audio_<k>.wav. Symlink them and build wav.scp.
    "${PY}" - "${data}" "${sd}" <<'PYEOF'
import os, sys
data, sd = sys.argv[1], sys.argv[2]
wavs_dir = os.path.join(sd, "wavs")
lines = []
with open(f"{data}/prompt_wav.scp", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        uid, path = line.split(maxsplit=1)
        src = os.path.abspath(path)
        dst = os.path.join(wavs_dir, uid + ".wav")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        lines.append(f"{uid} {dst}")
with open(f"{sd}/wav.scp", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"{len(lines)} prompt clips -> {sd}/wavs")
PYEOF

    echo "=== GT ${task}/${lang} (n=$(wc -l < ${sd}/wav.scp)) ==="
    bash scripts/run_score_wer.sh "${data}/prompt_text" "${sd}" "${lang}"
    CUDA_VISIBLE_DEVICES=${SCORE_GPU} "${PY}" utils/score_msr_utmos.py \
        -t "${sd}/wav.scp" -o "${sd}/msr_utmos_scores.txt" -m "${sd}/msr_utmos_mean.txt"
    echo "  ${lang}: $(grep -h '^WER:' ${sd}/wav_res_ref_text.wer 2>/dev/null) | MSR=$(cat ${sd}/msr_utmos_mean.txt 2>/dev/null)"
done
stop_asr_server; stop_whisper_server; trap - EXIT INT TERM
echo "=== GT_ZERO_SHOT DONE -> ${OUT_ROOT} ==="
