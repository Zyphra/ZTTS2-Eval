# scripts/lib_infer_scoring.sh -- shared per-language scoring loop for the
# run_infer_* eval scripts (WER + speaker-sim + MSR-UTMOS).
#
# SOURCE this (`. scripts/lib_infer_scoring.sh`); do not execute it. Both
# run_infer_zonos2_eval.sh and run_infer_voxblink_eval.sh use this single copy
# so resume-gate / GPU-pinning fixes land once for both tasks (the two scripts
# had already drifted apart once: the nan-WER / avg-0.00 resume conditions and
# the SCORE_GPU pinning existed only in the voxblink sibling).
#
# Requires from the caller (set before sourcing):
#   PY                  python interpreter for the torch scorers (venv-aware)
# Optional environment:
#   SCORE_GPU           GPU index for speaker-sim / MSR-UTMOS; empty = auto-pick
#   ASR_GPU             excluded from the auto-pick (the ASR server's GPU)
#   SCORE_MIN_COVERAGE  min scored/expected fraction for an existing score file
#                       to count as "already scored" on resume (default 0.98,
#                       matching the scorers' own coverage gate)
#
# Provides:
#   pick_score_gpu               resolve SCORE_GPU (auto-pick a free GPU)
#   wer_scored_ok FILE           .wer file is trustworthy (see below)
#   spk_scored_ok FILE           spk_simi_scores.txt is trustworthy
#   score_lang_dir DATA DEC LG   score one language leaf (resume-aware)
#   SCORE_FAILS                  count of leaves with a failed scoring step;
#                                callers should exit non-zero when > 0

SCORE_FAILS=0
SCORE_MIN_COVERAGE="${SCORE_MIN_COVERAGE:-0.98}"

# Pick a GPU for the speaker-sim / MSR-UTMOS torch models. Default `cuda` lands
# on GPU 0, which is often pinned by a co-resident TTS server -- ReDimNet2-B6
# then OOMs every utt and silently produces n=0 / avg=0.00. Override with
# SCORE_GPU (slurm/eval_job.sh pins it to the single cgroup-visible GPU).
pick_score_gpu() {
    if [ -z "${SCORE_GPU:-}" ]; then
        SCORE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F', ' '$2>=5000 && $1!='"${ASR_GPU:-99}"'{print $1; exit}' || true)
    fi
    if [ -z "${SCORE_GPU}" ]; then
        echo "[score] WARN: no GPU >=5 GB free; falling back to cuda:0 (may OOM)" 1>&2
        SCORE_GPU=0
    fi
    echo "[score] speaker-sim / MSR-UTMOS will run on GPU ${SCORE_GPU}"
}

# 0 iff n_scored/n_expected >= SCORE_MIN_COVERAGE (n_expected <= 0 passes).
_coverage_ok() {  # <n_scored> <n_expected>
    awk -v s="$1" -v e="$2" -v c="${SCORE_MIN_COVERAGE}" 'BEGIN{exit !(e<=0 || s>=c*e)}'
}

# A trustworthy .wer (mirrors wer_ok in scripts/rescore_gaps.sh): has a real
# `WER:` footer, is not `WER: nan` (ASR connection died -> empty wer.raw ->
# np.mean=nan), and when the coverage footer (N_SCORED:/N_EXPECTED:, written
# since 2026-08) is present, meets SCORE_MIN_COVERAGE. Legacy files without the
# coverage lines pass on the WER: line alone.
wer_scored_ok() {  # <path to wav_res_ref_text.wer>
    local f="$1" ns ne
    grep -q '^WER:' "${f}" 2>/dev/null || return 1
    if grep -q '^WER: nan' "${f}" 2>/dev/null; then return 1; fi
    ns=$(sed -n 's/^N_SCORED: //p' "${f}" 2>/dev/null | head -n1)
    ne=$(sed -n 's/^N_EXPECTED: //p' "${f}" 2>/dev/null | head -n1)
    if [ -n "${ns}" ] && [ -n "${ne}" ]; then
        _coverage_ok "${ns}" "${ne}" || return 1
    fi
    return 0
}

# A trustworthy spk_simi_scores.txt (mirrors spk_ok in scripts/rescore_gaps.sh):
# has an `avg` line that is not the n=0 stub `avg 0.00 0.00` (every utt
# OOM-failed), and the #N_SCORED:/#N_EXPECTED: coverage footer (when present)
# meets SCORE_MIN_COVERAGE.
spk_scored_ok() {  # <path to spk_simi_scores.txt>
    local f="$1" ns ne
    grep -q '^avg ' "${f}" 2>/dev/null || return 1
    if grep -q '^avg 0.00 0.00$' "${f}" 2>/dev/null; then return 1; fi
    ns=$(sed -n 's/^#N_SCORED: //p' "${f}" 2>/dev/null | head -n1)
    ne=$(sed -n 's/^#N_EXPECTED: //p' "${f}" 2>/dev/null | head -n1)
    if [ -n "${ns}" ] && [ -n "${ne}" ]; then
        _coverage_ok "${ns}" "${ne}" || return 1
    fi
    return 0
}

_build_wav_scp() {  # <leaf dir with wavs/> -> writes <leaf>/wav.scp
    find "$1/wavs" -name '*.wav' \
        | awk -F '/' '{print $NF, $0}' | sed 's@\.wav @ @g' > "$1/wav.scp"
}

# Score one language leaf: WER + speaker-sim (resume-gated together, so broken
# artifacts -- nan WER, all-zero spk, under-coverage -- are re-scored instead of
# silently skipped), then MSR-UTMOS (gated separately so it backfills dirs
# scored before that step existed). Individual scorer failures (including the
# scorers' own non-zero exit on coverage < SCORE_MIN_COVERAGE) are counted in
# SCORE_FAILS rather than aborting, so the remaining languages still run;
# callers exit non-zero when SCORE_FAILS > 0.
score_lang_dir() {  # <data dir (text/prompt_wav.scp)> <decode dir (wavs/)> <lang>
    local dd="$1" sd="$2" lg="$3" failed=0

    if wer_scored_ok "${sd}/wav_res_ref_text.wer" \
       && spk_scored_ok "${sd}/spk_simi_scores.txt"; then
        echo "[skip] ${sd} already scored (WER/spk)"
    else
        echo "Score WER for ${sd}"
        if ! bash scripts/run_score_wer.sh "${dd}/text" "${sd}" "${lg}"; then
            echo "[score] ERROR: WER scoring failed for ${sd}" 1>&2
            failed=1
        fi

        _build_wav_scp "${sd}" || failed=1

        echo "Score speaker similarity for ${sd} (GPU ${SCORE_GPU})"
        if ! CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" scripts/eval_speaker_similarity.py \
            --prompt_wavs "${dd}/prompt_wav.scp" \
            --hyp_wavs "${sd}/wav.scp" \
            --log_file "${sd}/spk_simi_scores.txt" \
            --device cuda; then
            echo "[score] ERROR: speaker-sim scoring failed for ${sd}" 1>&2
            failed=1
        fi
        # If every utt failed (n=0 => avg 0.00 0.00), flag it loudly so we don't
        # quietly compare a broken set to real numbers.
        if grep -q '^avg 0.00 0.00$' "${sd}/spk_simi_scores.txt" 2>/dev/null; then
            echo "[score] WARN: ${lg} speaker-sim is all-zero (n=0). Likely GPU OOM." 1>&2
        fi
    fi

    # MSR-UTMOS (gated independently so it backfills dirs scored before this
    # step existed without redoing WER/spk). Needs wav.scp.
    if [ ! -s "${sd}/msr_utmos_mean.txt" ]; then
        [ -s "${sd}/wav.scp" ] || _build_wav_scp "${sd}" || failed=1
        echo "Score MSR-UTMOS for ${sd} (GPU ${SCORE_GPU})"
        if ! CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" utils/score_msr_utmos.py \
            -t "${sd}/wav.scp" \
            -o "${sd}/msr_utmos_scores.txt" \
            -m "${sd}/msr_utmos_mean.txt"; then
            echo "[score] ERROR: MSR-UTMOS scoring failed for ${sd}" 1>&2
            failed=1
        fi
    fi

    if [ "${failed}" -ne 0 ]; then
        SCORE_FAILS=$((SCORE_FAILS + 1))
    fi
    return 0
}
