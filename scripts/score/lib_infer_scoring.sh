# scripts/score/lib_infer_scoring.sh -- the one per-language scoring loop (WER +
# speaker-sim + MSR-UTMOS) of scripts/score/run_score.sh, scripts/score/run_gt.sh,
# scripts/score/rescore_all.sh and scripts/analysis/control_seed_study.sh.
#
# SOURCE this from the repo root (`. scripts/score/lib_infer_scoring.sh`); do not execute it.
#
# Requires from the caller (set before sourcing):
#   PY                  python interpreter for the scorers (venv-aware)
# Optional environment:
#   SCORE_GPU           GPU index for speaker-sim / MSR-UTMOS; empty = auto-pick
#   SCORE_GPU_MIN_FREE_MIB  free memory the auto-pick requires (default 5000)
#   ASR_GPU             excluded from the auto-pick (the ASR server's GPU)
#   SCORE_MIN_COVERAGE  min scored/expected fraction for a score file to count as
#                       current (default 0.98; utils/average_wer.py MIN_COVERAGE)
#
# Provides:
#   pick_score_gpu                     resolve SCORE_GPU (auto-pick a free GPU)
#   score_wer REF_TEXT LEAF LANG       WER of LEAF/wavs against REF_TEXT
#   leaf_current DATA LEAF [MODE]      0 iff every expected score file is current
#   score_lang_dir DATA LEAF LANG [MODE]  score one language leaf (resume-aware)
#   SCORE_FAILS                        count of leaves with a failed scoring step;
#                                      callers should exit non-zero when > 0
#
# MODE is the leaf kind (utils.benchmark.stale_kinds):
#   system    generated audio in LEAF/wavs: wer, spk, msr (+ xutt when DATA has
#             target_wav.scp, the held-out pass against the real recording)
#   gt_clean  the real Clean prompt clips scored against prompt_text: wer, msr
#   gt_itw    the real ITW target recordings scored against text: wer, spk (prompt vs
#             real target, the held-out ceiling), msr
# The resume gates (stamps, nan / all-zero results, coverage) live in utils/benchmark.py.

SCORE_FAILS=0

# Pick a GPU for the speaker-sim / MSR-UTMOS torch models. Default `cuda` lands
# on GPU 0, which is often pinned by a co-resident TTS server -- ReDimNet2-B6
# then OOMs every utt and silently produces n=0 / avg=0.00. Override with
# SCORE_GPU (slurm/eval_job.sh pins it to the single cgroup-visible GPU).
pick_score_gpu() {
    local min=${SCORE_GPU_MIN_FREE_MIB:-5000}
    if [ -z "${SCORE_GPU:-}" ]; then
        SCORE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F', ' -v n="${min}" -v skip="${ASR_GPU:-99}" '$2>=n && $1!=skip {print $1; exit}' || true)
    fi
    if [ -z "${SCORE_GPU}" ]; then
        echo "[score] WARN: no GPU with >= ${min} MiB free; falling back to cuda:0 (may OOM)" 1>&2
        SCORE_GPU=0
    fi
    echo "[score] speaker-sim / MSR-UTMOS will run on GPU ${SCORE_GPU}"
}

# Fail fast: a run_wer.py that dies partway (OOM, SLURM preemption, uncaught exception)
# must not have its partial .raw averaged into a complete-looking .wer. average_wer.py
# additionally exits non-zero when scored/expected coverage is below MIN_COVERAGE.
score_wer() {  # <ref text> <leaf dir with wavs/> <lang>
    "${PY}" scripts/score/get_wav_res_ref_text.py "$1" "$2/wavs" "$2/wav_res_ref_text" \
        && "${PY}" scripts/score/run_wer.py "$2/wav_res_ref_text" "$2/wav_res_ref_text.wer.raw" "$3" \
        && "${PY}" utils/average_wer.py "$2/wav_res_ref_text.wer.raw" "$2/wav_res_ref_text.wer"
}

# <leaf dir with wavs/> <data dir with text> -> writes <leaf>/wav.scp, keeping only
# uids present in <data dir>/text so items removed from the benchmark (near-duplicate
# filter) never reach the speaker-sim / MSR scorers.
_build_wav_scp() {
    find "$1/wavs" -name '*.wav' \
        | awk -F '/' '{print $NF, $0}' | sed 's@\.wav @ @g' \
        | awk 'NR==FNR{keep[$1]=1; next} ($1 in keep)' "$2/text" - > "$1/wav.scp"
}

# Ground truth: bridge uid -> real wav. get_wav_res_ref_text expects wavs/<uid>.wav, but
# the source files are named prompt_audio_<k>.wav / <uid>.wav elsewhere: symlink them
# and build wav.scp from <scp>.
_bridge_audio() {  # <audio scp> <leaf dir>
    local p
    mkdir -p "$2"
    # Published GT leaves carry symlinks into another checkout (wavs, wav.scp, the raw
    # ASR intermediates). The python block below os.remove()s the paths it rebuilds,
    # which through a symlinked wavs/ directory would delete the files it points at.
    # Drop the symlinks themselves first.
    for p in wavs wav.scp wav_res_ref_text wav_res_ref_text.wer.raw wav_res_ref_text.wer.raw.meta; do
        [ -L "$2/${p}" ] && rm "$2/${p}"
    done
    mkdir -p "$2/wavs"
    "${PY}" - "$1" "$2" <<'PYEOF'
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
}

_stale() {  # <data dir> <leaf> <lang> <mode> -> the kinds still to score (stdout)
    "${PY}" -m utils.benchmark stale "$4" "$1" "$2" "$3"
}

leaf_current() {  # <data dir> <leaf> [mode]; the leaf dir is named after its language
    local stale
    stale=$(_stale "$1" "$2" "$(basename "${2%/}")" "${3:-system}") || return 1
    [ -z "${stale}" ]
}

# Score one language leaf. Each score kind has its own resume gate, so a broken or stale
# artifact (nan WER, all-zero spk, under-coverage, stale stamp) is rescored without
# redoing the kinds that are current. Individual scorer failures (including the
# scorers' own non-zero exit on low coverage) are counted in SCORE_FAILS rather than
# aborting, so the remaining languages still run; callers exit non-zero when
# SCORE_FAILS > 0.
score_lang_dir() {  # <data dir> <leaf dir> <lang> [mode]
    local dd="$1" sd="$2" lg="$3" mode="${4:-system}" failed=0 stale audio ref spk_ref

    if [ "${mode}" = system ]; then
        # results/<sys>/<panel>/ symlinked to the full set (utils.benchmark.is_panel_alias)
        # only serves the explorer drawer; scoring it would rewrite the full leaf's files.
        if "${PY}" -c 'import os, sys
from utils.benchmark import is_panel_alias
t = os.path.dirname(os.path.normpath(sys.argv[1]))
sys.exit(0 if is_panel_alias(os.path.dirname(t), os.path.basename(t)) else 1)' "${sd}"; then
            echo "[score] ERROR: ${sd} is a symlink to the full set, not a panel run; remove the symlink to score the panel" 1>&2
            SCORE_FAILS=$((SCORE_FAILS + 1))
            return 1
        fi
    fi

    if ! stale=$(_stale "${dd}" "${sd}" "${lg}" "${mode}"); then
        echo "[score] ERROR: resume gate failed for ${sd}" 1>&2
        SCORE_FAILS=$((SCORE_FAILS + 1))
        return 1
    fi
    if [ -z "${stale}" ]; then
        echo "[skip] ${sd} already scored"
        return 0
    fi
    echo "[score] ${sd}: scoring ${stale}"

    case "${mode}" in
        system)
            ref="${dd}/text"; spk_ref="${dd}/prompt_wav.scp"
            # wav.scp is cheap; rebuild it so a stale unfiltered list never reaches a scorer.
            _build_wav_scp "${sd}" "${dd}" || failed=1 ;;
        gt_clean)
            audio="${dd}/prompt_wav.scp"; ref="${dd}/prompt_text"
            _bridge_audio "${audio}" "${sd}" || failed=1 ;;
        gt_itw)
            audio="${dd}/target_wav.scp"; ref="${dd}/text"; spk_ref="${dd}/prompt_wav.scp"
            _bridge_audio "${audio}" "${sd}" || failed=1 ;;
        *)
            echo "[score] ERROR: unknown scoring mode '${mode}'" 1>&2
            SCORE_FAILS=$((SCORE_FAILS + 1))
            return 1 ;;
    esac

    case " ${stale} " in *" wer "*)
        echo "Score WER for ${sd}"
        if ! score_wer "${ref}" "${sd}" "${lg}"; then
            echo "[score] ERROR: WER scoring failed for ${sd}" 1>&2
            failed=1
        fi ;;
    esac

    case " ${stale} " in *" spk "*)
        echo "Score speaker similarity for ${sd} (GPU ${SCORE_GPU})"
        if ! CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" scripts/score/eval_speaker_similarity.py \
            --prompt-wavs "${spk_ref}" \
            --hyp-wavs "${sd}/wav.scp" \
            --log-file "${sd}/spk_simi_scores.txt" \
            --device cuda; then
            echo "[score] ERROR: speaker-sim scoring failed for ${sd}" 1>&2
            failed=1
        fi
        # If every utt failed (n=0 => avg 0.00 0.00), flag it loudly so we don't
        # quietly compare a broken set to real numbers.
        if grep -q '^avg 0.00 0.00$' "${sd}/spk_simi_scores.txt" 2>/dev/null; then
            echo "[score] WARN: ${lg} speaker-sim is all-zero (n=0). Likely GPU OOM." 1>&2
        fi ;;
    esac

    case " ${stale} " in *" xutt "*)
        awk 'NR==FNR{k[$1]=1; next} ($1 in k)' "${dd}/target_wav.scp" "${sd}/wav.scp" \
            > "${sd}/wav_xutt.scp" || failed=1
        echo "Score held-out speaker similarity for ${sd} (GPU ${SCORE_GPU})"
        if ! CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" scripts/score/eval_speaker_similarity.py \
            --prompt-wavs "${dd}/target_wav.scp" \
            --hyp-wavs "${sd}/wav_xutt.scp" \
            --log-file "${sd}/spk_simi_scores.xutt.txt" \
            --device cuda; then
            echo "[score] ERROR: held-out speaker-sim scoring failed for ${sd}" 1>&2
            failed=1
        fi ;;
    esac

    case " ${stale} " in *" msr "*)
        echo "Score MSR-UTMOS for ${sd} (GPU ${SCORE_GPU})"
        if ! CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" scripts/score/score_msr_utmos.py \
            --wav-scp "${sd}/wav.scp" \
            -o "${sd}/msr_utmos_scores.txt" \
            --mean-out "${sd}/msr_utmos_mean.txt"; then
            echo "[score] ERROR: MSR-UTMOS scoring failed for ${sd}" 1>&2
            failed=1
        fi ;;
    esac

    if [ "${failed}" -ne 0 ]; then
        SCORE_FAILS=$((SCORE_FAILS + 1))
    fi
    return 0
}
