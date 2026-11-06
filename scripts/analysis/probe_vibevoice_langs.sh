#!/usr/bin/env bash
# Cheap capability probe: does VibeVoice actually work outside en/zh?
#
# The card says "trained only on English and Chinese data; outputs in other languages
# are unsupported", but that is a vendor claim, not a structural limit -- VibeVoice is a
# CLONER, so it will accept any reference wav and any text. The realtime 0.5B taught us
# the claim is not uniform: of its nine disclaimed languages, es (5.22) and pt (7.82)
# were usable while nl (19.66) and fr (19.23) were not.
#
# Rather than spend ~150h finding out at full scale, generate a SMALL sample per
# language and read two numbers:
#
#   duration ratio = mean(gen audio) / mean(GT audio)
#       The glm-tts diagnostic. Across 8 languages, ratio predicted WER monotonically:
#       0.91x -> WER 4, 1.58x -> 91, 4.77x -> 153. A ratio >> 1 means the model is
#       running away (generating past the text), not merely speaking with an accent.
#       This needs NO ASR, so it is the cheap signal.
#   WER   the real metric, scored on the same sample.
#
# Output lands in a PROBE tree, never in results/<system>/, so a 50-utterance sample can
# never be mistaken for a real cell by the coverage audit or the results builder.
#
#   sbatch --job-name=ztts-probe-vv --gpus=1 \
#          --cpus-per-task=28 --mem=200G --time=8:00:00 --chdir="$(pwd)" \
#          --output=logs/slurm/ztts-probe-vv-%j.out scripts/analysis/probe_vibevoice_langs.sh
#
# Env: PROBE_MODEL (1.5b|7b, default 7b), PROBE_LANGS, PROBE_N (utts per lang),
#      PROBE_TASK, PROBE_ROOT.
set +u
[ -f scripts/score/serve_asr.sh ] || cd "$(dirname "$0")/../.." || exit 1
[ -f scripts/score/serve_asr.sh ] || { echo "[probe] ERROR: not at repo root"; exit 1; }

MODEL="${PROBE_MODEL:-7b}"
TASK="${PROBE_TASK:-clean}"
N="${PROBE_N:-50}"
# A spread across scripts and families: Latin (de es fr it pt), Cyrillic (ru),
# CJK-adjacent (ja ko), Indic (hi), Arabic (ar). Deliberately includes languages the
# realtime 0.5B handled well (es pt) and badly (fr nl) as calibration.
LANGS="${PROBE_LANGS:-de es fr it pt ru ja ko hi ar}"
ROOT="${PROBE_ROOT:-results/_probe/vibevoice-${MODEL}}"

. slurm/lib_slurm.sh
job_id="${SLURM_JOB_ID:-$$}"
port_base=$(slurm_port_base 0)
export ASR_PORT=$(slurm_role_port "${port_base}" asr) WHISPER_PORT=$(slurm_role_port "${port_base}" whisper)
export ASR_GPU=0 WHISPER_GPU=0
export ASR_GPU_MEM_UTIL=0.35 WHISPER_GPU_MEM_UTIL=0.35
export SERVE_LOG_DIR=$(slurm_server_log_dir "probe-${job_id}")
PY=.venv/bin/python

echo "=== probe: model=${MODEL} task=${TASK} n=${N}/lang langs='${LANGS}' -> ${ROOT}"

# ---- 1. generate (small sample per language) --------------------------------
for l in ${LANGS}; do
    [ -f "data/${TASK}/${l}/text" ] || { echo "[probe] skip ${l}: no data"; continue; }
    bash scripts/gen/run_vibevoice15.sh --model "${MODEL}" --task "${TASK}" \
        --test-set "${l}" --allow-unsupported true --limit "${N}" --out-root "${ROOT}" \
        || echo "[probe] generation failed for ${l}"
done

# ---- 2. score WER on what was generated -------------------------------------
. scripts/score/serve_asr.sh
trap 'stop_asr_server; stop_whisper_server' EXIT INT TERM
start_asr_server || { echo "[probe] ERROR: ASR server failed"; exit 1; }
maybe_start_whisper || exit 1

for l in ${LANGS}; do
    dd="${ROOT}/${TASK}/${l}"
    [ -d "${dd}/wavs" ] || continue
    rm -f "${dd}/wav_res_ref_text.wer" "${dd}/wav_res_ref_text.wer.raw"
    # run_score_wer.sh scores every utt in the ref text; our sample is the first N, so
    # restrict the reference to the uids actually generated or it counts the rest as
    # deletions and reports a meaningless ~100% WER.
    ls "${dd}/wavs" | sed 's/\.wav$//' | sort > "${dd}/.probe_uids"
    awk 'NR==FNR{k[$1];next} ($1 in k)' "${dd}/.probe_uids" "data/${TASK}/${l}/text" > "${dd}/.probe_text"
    echo "=== score ${l} ($(wc -l < "${dd}/.probe_text") utts) ==="
    bash scripts/score/run_score_wer.sh "${dd}/.probe_text" "${dd}" "${l}" || echo "[probe] WER failed: ${l}"
done

# ---- 3. report: duration ratio vs GT, next to WER ---------------------------
echo
echo "================ PROBE SUMMARY (${MODEL}, ${TASK}) ================"
ROOT="${ROOT}" TASK="${TASK}" LANGS="${LANGS}" "${PY}" - <<'PY'
import os, random, soundfile as sf
root, task = os.environ['ROOT'], os.environ['TASK']
random.seed(0)
def mean_dur(d, uids=None):
    if not os.path.isdir(d): return None
    fs = [f for f in os.listdir(d) if f.endswith('.wav')]
    if uids is not None: fs = [f for f in fs if f[:-4] in uids]
    if not fs: return None
    fs = random.sample(fs, min(60, len(fs))); ds = []
    for f in fs:
        try:
            i = sf.info(os.path.join(d, f)); ds.append(i.frames / i.samplerate)
        except Exception: pass
    return sum(ds) / len(ds) if ds else None
def wer(p):
    if not os.path.isfile(p): return None
    for l in open(p, errors='replace'):
        if l.startswith('WER:'):
            try: return float(l.split()[1].rstrip(','))
            except Exception: return None
print(f"{'lang':<5} {'n':>4} {'gen dur':>8} {'GT dur':>8} {'ratio':>7} {'WER':>8}  verdict")
for lg in os.environ['LANGS'].split():
    dd = os.path.join(root, task, lg)
    wd = os.path.join(dd, 'wavs')
    if not os.path.isdir(wd): continue
    uids = {f[:-4] for f in os.listdir(wd) if f.endswith('.wav')}
    g = mean_dur(wd)
    # compare against the SAME utterances in the GT set, not the whole set
    t = mean_dur(f'results/gt_{task}/{lg}/wavs', uids)
    w = wer(os.path.join(dd, 'wav_res_ref_text.wer'))
    r = g / t if (g and t) else None
    if r is None:      v = '?'
    elif r > 1.5:      v = 'RUNAWAY - unusable'
    elif r > 1.15:     v = 'overruns - suspect'
    elif w is None:    v = 'duration ok, no WER'
    elif w < 15:       v = 'USABLE'
    elif w < 30:       v = 'marginal'
    else:              v = 'poor'
    print(f"{lg:<5} {len(uids):>4} {g or 0:8.2f} {t or 0:8.2f} "
          f"{(f'{r:.2f}x' if r else '-'):>7} {(f'{w:.2f}' if w is not None else '-'):>8}  {v}")
print("\nratio >> 1 = model generating past the text (the glm-tts failure mode).")
print("Sample-sized WER is indicative only -- confirm a full run before reporting.")
PY
