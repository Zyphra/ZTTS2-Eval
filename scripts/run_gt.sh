#!/usr/bin/env bash
# Ground-truth baselines: score REAL human audio with the same pipeline as the systems.
# Output is results/gt_<task>/<lang>/ (default; --out_root overrides), one leaf per language:
#
#   --task clean            (Clean, FLEURS-R)  the real PROMPT clips scored against
#       prompt_text. WER = the ASR error floor on genuine human speech; MSR-UTMOS = the
#       real-audio quality ceiling. Speaker similarity is omitted (prompt vs itself is
#       degenerate ~100). The clean set retains prompt audio but not target audio, so
#       the GT is measured on the prompt clips rather than the target text.
#   --task in-the-wild   (ITW, VoxBlink2)  the real TARGET recordings
#       (data/in-the-wild/<lang>/target_wav.scp) scored against `text`. WER and
#       MSR-UTMOS as above; speaker similarity is prompt vs. real target -- the
#       held-out ceiling of the primary ITW speaker-similarity score.
#
#   bash scripts/run_gt.sh                                  # all Clean langs
#   bash scripts/run_gt.sh --test_set "en zh"
#   bash scripts/run_gt.sh --task in-the-wild        # all ITW langs
#
# Leaves are version-aware: a leaf whose score files do not carry the current
# BENCHMARK_VERSION stamps (utils/benchmark.py) is rescored.
set +u
cd "$(dirname "$0")/.." || exit 1

task=clean
test_set=""   # empty -> the task's basket, resolved after option parsing
out_root=""   # empty -> results/gt_<task>
. utils/parse_options.sh || exit 1
[ -n "${test_set}" ] || test_set=$(python3 -m utils.benchmark langs "${task}")
[ -n "${out_root}" ] || out_root=results/gt_${task}

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || exit 1
maybe_start_whisper || exit 1

# Shared resume gates (wer_scored_ok / spk_scored_ok / BENCHMARK_VERSION) and GPU pick.
. scripts/lib_infer_scoring.sh
pick_score_gpu

fails=0
for lang in ${test_set}; do
    data=data/${task}/${lang}
    sd=${out_root}/${lang}
    [ -f "${data}/prompt_wav.scp" ] || { echo "[skip] ${lang} (no prompt_wav.scp)"; continue; }

    if [ "${task}" = "in-the-wild" ]; then
        [ -f "${data}/target_wav.scp" ] \
            || { echo "[skip] ${lang} (no target_wav.scp; run scripts/build_voxblink_targets.py)"; continue; }
        audio_scp=${data}/target_wav.scp     # real target recordings
        ref_text=${data}/text
    else
        audio_scp=${data}/prompt_wav.scp     # real prompt clips
        ref_text=${data}/prompt_text
    fi

    if wer_scored_ok "${sd}/wav_res_ref_text.wer" \
       && { [ "${task}" != "in-the-wild" ] || spk_scored_ok "${sd}/spk_simi_scores.txt"; } \
       && msr_scored_ok "${sd}"; then
        echo "[skip] ${lang} already scored"
        continue
    fi

    mkdir -p "${sd}"
    # Published GT leaves carry symlinks into another checkout (wavs, wav.scp, the raw
    # ASR intermediates). The python block below os.remove()s the paths it rebuilds, which
    # through a symlinked wavs/ directory would delete the files it points at. Drop the
    # symlinks themselves first.
    for p in wavs wav.scp wav_res_ref_text wav_res_ref_text.wer.raw wav_res_ref_text.wer.raw.meta; do
        [ -L "${sd}/${p}" ] && rm "${sd}/${p}"
    done
    mkdir -p "${sd}/wavs"
    # Bridge uid -> wav: get_wav_res_ref_text expects wavs/<uid>.wav, but the source
    # files are named prompt_audio_<k>.wav / <uid>.wav elsewhere. Symlink them and
    # build wav.scp.
    "${PY}" - "${audio_scp}" "${sd}" <<'PYEOF'
import os, sys
scp, sd = sys.argv[1], sys.argv[2]
wavs_dir = os.path.join(sd, "wavs")
lines = []
with open(scp, encoding="utf-8") as f:
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
print(f"{len(lines)} real clips -> {sd}/wavs")
PYEOF

    echo "=== GT ${task}/${lang} (n=$(wc -l < ${sd}/wav.scp)) ==="
    bash scripts/run_score_wer.sh "${ref_text}" "${sd}" "${lang}" || fails=$((fails + 1))
    if [ "${task}" = "in-the-wild" ]; then
        # Held-out ceiling: the real prompt vs the real target of the same speaker.
        CUDA_VISIBLE_DEVICES=${SCORE_GPU} "${PY}" scripts/eval_speaker_similarity.py \
            --prompt_wavs "${data}/prompt_wav.scp" --hyp_wavs "${sd}/wav.scp" \
            --log_file "${sd}/spk_simi_scores.txt" --device cuda || fails=$((fails + 1))
    fi
    CUDA_VISIBLE_DEVICES=${SCORE_GPU} "${PY}" utils/score_msr_utmos.py \
        -t "${sd}/wav.scp" -o "${sd}/msr_utmos_scores.txt" -m "${sd}/msr_utmos_mean.txt" \
        || fails=$((fails + 1))
    echo "  ${lang}: $(grep -h '^WER:' ${sd}/wav_res_ref_text.wer 2>/dev/null) | MSR=$(cat ${sd}/msr_utmos_mean.txt 2>/dev/null)"
done
stop_asr_server; stop_whisper_server; trap - EXIT INT TERM
echo "=== GT_${task} DONE -> ${out_root} (${fails} failed step(s)) ==="
[ "${fails}" -eq 0 ]
