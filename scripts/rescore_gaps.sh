#!/usr/bin/env bash
# Re-score leaves whose WAVS ARE COMPLETE but whose scores are missing or partial,
# because the ASR server died mid-job (GPU OOM from job co-residency). Those jobs exit
# 0: the WER step's failure is not fatal, so they leave a header-only
# wav_res_ref_text.wer behind and a coverage audit that only checks "does the file
# exist" counts the cell as done. See slurm/README.md.
#
# NO GENERATION -- the audio is already on disk. One ASR (+ Whisper) server start
# serves every target, so this must run as ONE job, alone on its GPU.
#
#   sbatch --job-name=ztts-rescore-gaps --gpus=2 \
#          --export=ALL,SCORE_GPU=1 \
#          --cpus-per-task=28 --mem=200G --time=12:00:00 --chdir="$(pwd)" \
#          --output=logs/slurm/ztts-rescore-gaps-%j.out scripts/rescore_gaps.sh
#
# Targets are "model/task/lang" triples; override with RESCORE_TARGETS.
#
# All THREE metrics are recomputed per leaf, not just the missing one. A leaf that was
# scored while generation was still filling has metrics over DIFFERENT wav subsets
# (e.g. moss-tts-local-v1.5 voxblink: spk/msr present, WER absent) -- recomputing only
# the gap would leave the trio mutually inconsistent.
#
# ASR_ROUTING=1: langs outside QWEN_LANGS (en zh ko th vi hi fa yo) go to Whisper
# large-v3 with a forced language, matching how the GT baselines were scored.
# NB: set +u -- serve_asr.sh references optional unset vars.
set +u
[ -f scripts/serve_asr.sh ] || cd "$(dirname "$0")/.." || exit 1
[ -f scripts/serve_asr.sh ] || { echo "[rescore-gaps] ERROR: not at repo root"; exit 1; }

# Targets are DISCOVERED at run time, not hardcoded: which leaves the OOM corrupts
# depends on which jobs happened to share a node, so a list fixed at submit time is
# stale by the time the job starts (sbatch snapshots the script, so editing this file
# does not reach an already-queued job either). A leaf qualifies when its wavs are
# essentially complete but any of the three metrics is missing or untrustworthy --
# a header-only .wer (dead ASR server), a `WER: nan` footer, an `avg 0.00 0.00`
# spk stub, or a coverage footer (N_SCORED:/N_EXPECTED:) below SCORE_MIN_COVERAGE.
if [ -x .venv/bin/python ]; then _DPY=.venv/bin/python; else _DPY=python3; fi
discover() {
    "${_DPY}" - <<'PY'
import os
R, D = 'results', 'data'
exp = {}
for task in ('zero_shot', 'zero_shot_voxblink'):
    d = os.path.join(D, task)
    if not os.path.isdir(d):
        continue
    for lg in os.listdir(d):
        t = os.path.join(d, lg, 'text')
        if os.path.isfile(t):
            exp[(task, lg)] = sum(1 for _ in open(t, encoding='utf-8', errors='replace'))

def nonempty(p):
    return os.path.isfile(p) and os.path.getsize(p) > 0

MIN_COV = float(os.environ.get('SCORE_MIN_COVERAGE', '0.98'))

def wer_ok(p):
    """A trustworthy .wer: has a real WER: footer, and when the coverage footer
    (N_SCORED:/N_EXPECTED:, written since 2026-08) is present, meets MIN_COV.
    Legacy files without the coverage lines pass on the WER: line alone."""
    if not nonempty(p):
        return False
    wer_line = False
    n_scored = n_expected = None
    for l in open(p, errors='replace'):
        if l.startswith('WER:'):
            if l.startswith('WER: nan'):
                return False          # ASR connection died mid-leaf
            wer_line = True
        elif l.startswith('N_SCORED:'):
            n_scored = int(l.split(':', 1)[1])
        elif l.startswith('N_EXPECTED:'):
            n_expected = int(l.split(':', 1)[1])
    if not wer_line:
        return False
    if n_scored is not None and n_expected:
        return n_scored / n_expected >= MIN_COV
    return True

def spk_ok(p):
    """A trustworthy spk file: per-utt rows + a non-stub avg line, and the
    #N_SCORED:/#N_EXPECTED: coverage footer (when present) meets MIN_COV."""
    if not nonempty(p):
        return False
    avg = None
    n_scored = n_expected = None
    for l in open(p, errors='replace'):
        if l.startswith('avg '):
            avg = l.strip()
        elif l.startswith('#N_SCORED:'):
            n_scored = int(l.split(':', 1)[1])
        elif l.startswith('#N_EXPECTED:'):
            n_expected = int(l.split(':', 1)[1])
    if avg is None or avg == 'avg 0.00 0.00':   # every utt failed (OOM-to-n=0)
        return False
    if n_scored is not None and n_expected:
        return n_scored / n_expected >= MIN_COV
    return True

def msr_ok(p):
    """A trustworthy MSR mean: non-empty, parses > 0, the per-utt scores file beside
    it has rows, and the coverage footer in msr_utmos_ci.txt (N_SCORED:/N_EXPECTED:)
    meets MIN_COV. A mean of exactly 0.0 with a 0-byte scores file is the scorer's
    crash stub; a mean over a partial set is the GPU-co-residency OOM case -- the
    scorer exits non-zero but still writes the mean, so the mean's existence alone
    never proves the leaf is complete."""
    if not nonempty(p):
        return False
    try:
        mean = float(open(p, errors='replace').readline().split()[0])
    except (ValueError, IndexError):
        return False
    if mean <= 0.0:
        return False
    leaf = os.path.dirname(p)
    if not nonempty(os.path.join(leaf, 'msr_utmos_scores.txt')):
        return False
    ci = os.path.join(leaf, 'msr_utmos_ci.txt')
    n_scored = n_expected = None
    if os.path.isfile(ci):
        for l in open(ci, errors='replace'):
            if l.startswith('N_SCORED:'):
                n_scored = int(l.split(':', 1)[1])
            elif l.startswith('N_EXPECTED:'):
                n_expected = int(l.split(':', 1)[1])
    if n_scored is not None and n_expected:
        return n_scored / n_expected >= MIN_COV
    return True

out = []
for sysd in sorted(os.listdir(R)):
    if sysd.startswith(('_', 'gt_', 'whisper_')):
        continue
    for task in ('zero_shot', 'zero_shot_voxblink'):
        base = os.path.join(R, sysd, task)
        if not os.path.isdir(base):
            continue
        for lg in sorted(os.listdir(base)):
            leaf = os.path.join(base, lg)
            wavs = os.path.join(leaf, 'wavs')
            if not os.path.isdir(wavs):
                continue
            n = len(os.listdir(wavs))
            e = exp.get((task, lg), 0)
            if not e or n < e * 0.995:      # still generating -- leave it alone
                continue
            wer = wer_ok(os.path.join(leaf, 'wav_res_ref_text.wer'))
            spk = spk_ok(os.path.join(leaf, 'spk_simi_scores.txt'))
            msr = msr_ok(os.path.join(leaf, 'msr_utmos_mean.txt'))
            if not (wer and spk and msr):
                miss = ','.join(k for k, v in (('wer', wer), ('spk', spk), ('msr', msr)) if not v)
                out.append((f"{sysd}/{task}/{lg}", miss))
for t, miss in out:
    print(t, f"# missing:{miss}")
PY
}

if [ -n "${RESCORE_TARGETS}" ]; then
    TARGETS="${RESCORE_TARGETS}"
    echo "[rescore-gaps] using RESCORE_TARGETS override"
else
    echo "[rescore-gaps] discovering leaves with complete wavs but missing scores ..."
    _disc="$(discover)"
    echo "${_disc}" | sed 's/^/  /'
    TARGETS="$(echo "${_disc}" | awk '{print $1}')"
fi
if [ -z "${TARGETS}" ]; then
    echo "[rescore-gaps] nothing to rescore -- every complete leaf already has all 3 metrics"
    exit 0
fi

job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 20000 + (job_id % 10000) * 4 ))
export ASR_PORT=$(( port_base + 1 ))
export WHISPER_PORT=$(( port_base + 2 ))
export ASR_GPU=0 WHISPER_GPU=0 ASR_ROUTING=1
# Both ASR servers co-locate on the one cgroup GPU; 0.5 each would total 100% and
# Whisper would OOM at init. 0.35 each (28GB of 80GB) fits both.
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
# The torch scorers (ReDimNet speaker-sim, MSR-UTMOS) then have to fit in what the
# two vLLM servers left behind. On big leaves (1500 utts, long wavs) they do not:
# a 2026-08-29 run lost 23 of 44 leaves to "CUDA out of memory ... 4.00 GiB" during
# speaker-sim, AFTER the script had already deleted the old scores -- turning a
# partial leaf into an empty one. With --gpus=2, set SCORE_GPU=1 to give the
# scorers their own card. Defaults to the ASR GPU, preserving 1-GPU behaviour.
SCORE_GPU="${SCORE_GPU:-${ASR_GPU}}"
echo "[rescore-gaps] ASR/Whisper on GPU ${ASR_GPU}; torch scorers on GPU ${SCORE_GPU}"
srv="logs/slurm/servers/rescore-gaps-${job_id}"; mkdir -p "${srv}"
export ASR_LOG="${srv}/asr.log" WHISPER_LOG="${srv}/whisper.log"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

. scripts/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || { echo "[rescore-gaps] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper
[ -n "${WHISPER_BASE_URL:-}" ] || { echo "[rescore-gaps] ERROR: WHISPER_BASE_URL unset -- routing OFF, aborting"; exit 1; }
echo "[rescore-gaps] routing via ${WHISPER_BASE_URL}"

n_ok=0; n_skip=0; n_fail=0
for tgt in ${TARGETS}; do
    m="${tgt%%/*}"; rest="${tgt#*/}"; task="${rest%%/*}"; lg="${rest##*/}"
    dd="results/${m}/${task}/${lg}"
    txt="data/${task}/${lg}/text"
    ref="data/${task}/${lg}/prompt_wav.scp"

    if [ ! -d "${dd}/wavs" ]; then echo "[skip] ${tgt} (no wavs)"; n_skip=$((n_skip+1)); continue; fi
    if [ ! -f "${txt}" ]; then echo "[skip] ${tgt} (no ref text)"; n_skip=$((n_skip+1)); continue; fi
    n_wav=$(find "${dd}/wavs" -name '*.wav' | wc -l)
    n_exp=$(wc -l < "${txt}")
    if [ "${n_wav}" -lt 100 ]; then echo "[skip] ${tgt} (only ${n_wav} wavs)"; n_skip=$((n_skip+1)); continue; fi

    echo "=== rescore ${tgt} (${n_wav}/${n_exp} wavs) ==="
    # Clear every artifact the resume guards key on, so all three metrics re-run.
    rm -f "${dd}/wav_res_ref_text" "${dd}/wav_res_ref_text.wer" "${dd}/wav_res_ref_text.wer.raw" \
          "${dd}/wav_res_ref_text.wer.raw.meta" \
          "${dd}/spk_simi_scores.txt" \
          "${dd}/msr_utmos_scores.txt" "${dd}/msr_utmos_mean.txt" "${dd}/msr_utmos_ci.txt"

    if ! bash scripts/run_score_wer.sh "${txt}" "${dd}" "${lg}"; then
        echo "[rescore-gaps] WER FAILED: ${tgt}"; n_fail=$((n_fail+1)); continue
    fi
    # A surviving-but-empty .wer means the ASR server died again -- do not let that
    # pass as success, it is exactly the silent failure this script exists to fix.
    if ! grep -q '^WER:' "${dd}/wav_res_ref_text.wer" 2>/dev/null; then
        echo "[rescore-gaps] WER EMPTY (ASR server down?): ${tgt}"; n_fail=$((n_fail+1)); continue
    fi

    find "${dd}/wavs" -name '*.wav' | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > "${dd}/wav.scp"

    if [ -f "${ref}" ]; then
        CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" scripts/eval_speaker_similarity.py \
            --prompt_wavs "${ref}" --hyp_wavs "${dd}/wav.scp" \
            --log_file "${dd}/spk_simi_scores.txt" --device cuda \
            || { echo "[rescore-gaps] spk FAILED: ${tgt}"; n_fail=$((n_fail+1)); continue; }
    else
        echo "[rescore-gaps] no ${ref} -- skipping speaker-sim for ${tgt}"
    fi

    CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PY}" utils/score_msr_utmos.py \
        -t "${dd}/wav.scp" -o "${dd}/msr_utmos_scores.txt" -m "${dd}/msr_utmos_mean.txt" \
        || { echo "[rescore-gaps] msr FAILED: ${tgt}"; n_fail=$((n_fail+1)); continue; }

    n_ok=$((n_ok+1))
done
echo "[rescore-gaps] done: ${n_ok} rescored, ${n_skip} skipped, ${n_fail} failed"
[ "${n_fail}" -eq 0 ]
