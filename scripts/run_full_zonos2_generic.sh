#!/usr/bin/env bash
# Generic full ZTTS2-Eval driver (both sets). Generates and scores the FLEURS-R
# (zero_shot) and VoxBlink2 (zero_shot_voxblink) sets for one Zonos2 checkpoint.
#
# Usage:
#   bash scripts/run_full_zonos2_generic.sh <TAG> <SERVER_URL>
#
# Example:
#   bash scripts/run_full_zonos2_generic.sh zonos2_my_ckpt http://127.0.0.1:20241
#
# Reuses the long-lived ASR vLLM server on :8000.
set +u
cd "$(dirname "$0")/.."

TAG="${1:?tag required, e.g. zonos2_my_ckpt}"
SERVER="${2:?server URL required, e.g. http://127.0.0.1:20243}"
OUT_ROOT="results/${TAG}"
LOG_DIR="/tmp/full_${TAG}"
mkdir -p "${LOG_DIR}"

echo "=== [1/4] gen zero_shot ${SERVER} ==="
bash run_generate_zonos2.sh \
    --server "${SERVER}" \
    --out_root "${OUT_ROOT}" \
    --workers 8 \
    > "${LOG_DIR}/gen_main.log" 2>&1
echo "gen_main exit $?"
tail -5 "${LOG_DIR}/gen_main.log"

echo "=== [2/4] gen zero_shot_voxblink ${SERVER} ==="
bash run_generate_zonos2_voxblink.sh \
    --server "${SERVER}" \
    --out_root "${OUT_ROOT}/zero_shot_voxblink" \
    --workers 16 \
    > "${LOG_DIR}/gen_voxblink.log" 2>&1
echo "gen_voxblink exit $?"
tail -5 "${LOG_DIR}/gen_voxblink.log"

echo "=== [3/4] score zero_shot ==="
DECODE_DIR="${OUT_ROOT}/zero_shot" bash run_infer_zonos2_eval.sh \
    --task zero_shot \
    --test_set "en zh de es fr it ja ko ru" \
    > "${LOG_DIR}/score_zero_shot.log" 2>&1
echo "score_zero_shot exit $?"
tail -3 "${LOG_DIR}/score_zero_shot.log"

echo "=== [4/4] score zero_shot_voxblink ==="
DECODE_DIR="${OUT_ROOT}/zero_shot_voxblink" bash run_infer_voxblink_eval.sh \
    > "${LOG_DIR}/score_voxblink.log" 2>&1
echo "score_voxblink exit $?"
tail -3 "${LOG_DIR}/score_voxblink.log"

echo
echo "=== summary (${TAG}) ==="
.venv/bin/python - <<PY
import os
ROOT='${OUT_ROOT}'
def parse_wer(p):
    if not os.path.exists(p): return None
    for ln in open(p):
        if ln.startswith('WER:'):
            return ln.strip()
    return None
def parse_spk(p):
    if not os.path.exists(p): return None
    for ln in open(p):
        if ln.startswith('avg '): return ln.strip()
    return None
def parse_mean(p):
    if not os.path.exists(p): return None
    try: return float(open(p).read().strip())
    except: return None

for task, sets in [
    ('zero_shot', 'en zh de es fr it ja ko ru'.split()),
    ('zero_shot_voxblink', 'en zh de es fr it ja ko ru pt ar hi tr tl pl id th'.split()),
]:
    print(f'--- {task} ---')
    for s in sets:
        sd = f'{ROOT}/{task}/{s}'
        w = parse_wer(f'{sd}/wav_res_ref_text.wer')
        sp = parse_spk(f'{sd}/spk_simi_scores.txt')
        msr = parse_mean(f'{sd}/msr_utmos_mean.txt')
        bits = [s]
        if w: bits.append(w)
        if sp: bits.append(sp)
        if msr is not None: bits.append(f'MSR={msr:.3f}')
        print('  ' + ' | '.join(bits))
PY
