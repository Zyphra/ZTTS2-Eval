#!/usr/bin/env bash
# Complete the ASR-routing re-score of the FLEURS-R (clean) set.
#
# WHY THIS EXISTS
# ---------------
# `utils/run_wer.py` routes a language to Whisper-large-v3 (forced language) when it is
# NOT in QWEN_LANGS (`en zh ko th vi hi fa yo`) and ASR_ROUTING=1. The ground-truth
# baselines were scored that way from the start. The synthesized audio was not: the
# 2026-07-30 fix (`scripts/rescore_routed.sh`) re-scored only the six languages where the
# misrouting was most obvious (bg ca hr nb sk uk), leaving **22 more** languages scored by
# Qwen3-ASR auto-detect while GT used Whisper.
#
# That is not a small effect. Re-scoring zonos2_accurate revealed inflation of:
#
#     he +53.1   fi +14.6   fil +14.5   hu +13.9   ro +11.7   el +10.1
#     cs  +9.5   sv  +8.9   da  +5.5    pl  +5.0   tr  +4.6   ms  +3.3
#     (and ~0.3-1.5 on ar id fr it nl de ja ru es pt, where auto-detect mostly copes)
#
# and moved its 26-tail-language WER_CLIP from 13.87 to 7.77. Because GT is routed and
# synth is not, every "Δ vs GT floor" on those languages is inflated by the full amount.
#
# Diagnosis was unambiguous: languages showing EXACTLY zero change on re-scoring are
# precisely QWEN_LANGS plus the six already fixed, and the stored `he` hypotheses are 90%
# Hebrew with Arabic/Cyrillic drift where GT's are 100% Hebrew.
#
# WHAT IT DOES
# ------------
# Shards the 22 languages across the GPUs of one allocation, running one
# `scripts/rescore_routed.sh` worker per GPU (own ASR + Whisper server, own port window).
# WER only -- speaker similarity and MSR-UTMOS are language-agnostic and untouched.
#
#   sbatch --job-name=ztts-rescore-all --gpus=8 \
#          --cpus-per-task=112 --mem=1500G --time=24:00:00 \
#          --chdir="$(pwd)" --output=logs/slurm/ztts-rescore-all-%j.out \
#          scripts/rescore_routed_all.sh
#
# Env: RESCORE_ALL_LANGS, RESCORE_ALL_MODELS, RESCORE_NGPU (default: all visible GPUs).
set +u
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[rescore-all] ERROR: not at repo root"; exit 1; }

# The 35 FLEURS-R languages minus QWEN_LANGS (en zh ko th vi hi fa) minus the six already
# re-scored on 2026-07-30 (bg ca hr nb sk uk).
LANGS=${RESCORE_ALL_LANGS:-"ar cs da de el es fi fil fr he hu id it ja ms nl pl pt ro ru sv tr"}
# The 11 dense systems, plus the partial-coverage ones -- those are the novel-family rows
# the coreset validation leans on, so they must be scored consistently too.
MODELS=${RESCORE_ALL_MODELS:-"zonos2_accurate zonos2_expressive omni_qwen3-tts-1.7b \
omni_qwen3-tts-0.6b omni_voxcpm2 omni_fish-s2-pro omni_higgs-audio-v3 \
omni_omnivoice-reftext moss-tts cosyvoice3 omni_voxtral-tts \
omni_omnivoice moss-tts-v1.5 moss-tts-local-v1.5 omni_glm-tts kokoro"}

n_gpu=${RESCORE_NGPU:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}
[ "${n_gpu}" -ge 1 ] 2>/dev/null || n_gpu=1
set -- ${LANGS}
n_lang=$#
echo "[rescore-all] ${n_lang} languages x $(echo ${MODELS} | wc -w) systems across ${n_gpu} GPU(s)"
echo "[rescore-all] langs: ${LANGS}"

# Round-robin languages across shards so the (uneven) per-language costs even out.
pids=""
for ((s = 0; s < n_gpu; s++)); do
    shard_langs=""
    i=0
    for lg in ${LANGS}; do
        [ $((i % n_gpu)) -eq "${s}" ] && shard_langs="${shard_langs} ${lg}"
        i=$((i + 1))
    done
    [ -n "${shard_langs}" ] || continue
    echo "[rescore-all] shard ${s} (GPU ${s}):${shard_langs}"
    RESCORE_SHARD="${s}" RESCORE_GPU="${s}" \
    RESCORE_LANGS="${shard_langs# }" RESCORE_MODELS="${MODELS}" \
        bash scripts/rescore_routed.sh > "logs/slurm/rescore-all-shard${s}.log" 2>&1 &
    pids="${pids} $!"
done

fail=0
for p in ${pids}; do
    wait "${p}" || fail=$((fail + 1))
done
echo "[rescore-all] all shards done (${fail} shard(s) reported failure)"
grep -h '^\[rescore\] done' logs/slurm/rescore-all-shard*.log 2>/dev/null

echo
echo "[rescore-all] next: rebuild the matrix and the clean panels"
echo "  python scripts/coreset/build_matrix.py --set clean --verify"
echo "  python scripts/coreset/denoise.py --set clean"
echo "  python scripts/coreset/validate.py --set clean --ks 200,500,1000,2000,3500 --seeds 0,1,2,3,4"
