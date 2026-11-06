#!/usr/bin/env bash
# End-to-end coreset pipeline: matrix -> fingerprints -> validation sweep -> panels.
#
# Everything here is offline and CPU-only. It reads the per-utterance scores already in
# results/ and never regenerates audio, so the whole selection and validation loop runs
# without a GPU.
#
#   bash scripts/coreset/run.sh              # full rebuild, both sets
#   CORESET_SKIP_MATRIX=1 bash scripts/coreset/run.sh   # reuse matrix_*.npz
#   CORESET_SKIP_VALIDATE=1 bash scripts/coreset/run.sh # skip the (slow) sweep
#
# Shipped panel sizes are the knee of the error-vs-k curve in RESULTS_CORESET.md, not
# round numbers: see docs/coreset.md for how they were chosen.
#
# RESULTS_CORESET.md is regenerated as the final step (step 7). It needs the validation
# CSVs from step 4 (skipping the sweep reuses the committed CSVs), the matrix and
# fingerprints from steps 2-3 for the live CI-coverage and delta-power sections, and the
# shipped mini panels from step 5 -- so regenerate it only via this pipeline, never by
# hand-editing the .md.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
[ -x "${PY}" ] || PY=python

MINI_SELECTOR=${MINI_SELECTOR:-strat_meta}
CLEAN_MINI_K=${CLEAN_MINI_K:-3500}
ITW_MINI_K=${ITW_MINI_K:-2000}
CLEAN_DELTA_K=${CLEAN_DELTA_K:-200}
ITW_DELTA_K=${ITW_DELTA_K:-200}

echo "=== 1. repair any tab/newline-corrupted .wer.raw records ==="
"${PY}" scripts/repair_wer_raw.py --apply

if [ -z "${CORESET_SKIP_MATRIX:-}" ]; then
    echo "=== 2. build the system x item matrix (+ reproduction gate) ==="
    "${PY}" scripts/coreset/build_matrix.py --set clean --set itw --verify
else
    echo "=== 2. skipped (CORESET_SKIP_MATRIX) ==="
fi

echo "=== 3. denoise item fingerprints ==="
"${PY}" scripts/coreset/denoise.py --set clean --set itw

if [ -z "${CORESET_SKIP_VALIDATE:-}" ]; then
    echo "=== 4. leave-family-out validation sweep (slow) ==="
    # One shared selector list for both sets (this is what produced the shipped CSVs).
    # itw additionally sweeps greedy; it is left off the clean set, where its O(k*N*M)
    # cost dominates the sweep for no new insight (validation_clean.csv has no greedy rows).
    SELECTORS=random_global,random_perlang,strat_meta,strat_quantile,median_quantile,strat_response,medoid_response,medoid_raw,highvar
    "${PY}" scripts/coreset/validate.py --set itw \
        --ks 200,500,1000,2000 --seeds 0,1,2,3,4 \
        --selectors "${SELECTORS},greedy" \
        --report-selector "${MINI_SELECTOR}" --report-k "${ITW_MINI_K}"
    "${PY}" scripts/coreset/validate.py --set clean \
        --ks 200,500,1000,2000,3500 --seeds 0,1,2,3,4 \
        --selectors "${SELECTORS}" \
        --report-selector "${MINI_SELECTOR}" --report-k "${CLEAN_MINI_K}"
else
    echo "=== 4. skipped (CORESET_SKIP_VALIDATE) ==="
fi

echo "=== 5. select the four shipped panels ==="
"${PY}" scripts/coreset/panel_select.py --set clean --kind mini  --selector "${MINI_SELECTOR}" -k "${CLEAN_MINI_K}"
"${PY}" scripts/coreset/panel_select.py --set itw   --kind mini  --selector "${MINI_SELECTOR}" -k "${ITW_MINI_K}"
"${PY}" scripts/coreset/panel_select.py --set clean --kind delta --selector delta -k "${CLEAN_DELTA_K}"
"${PY}" scripts/coreset/panel_select.py --set itw   --kind delta --selector delta -k "${ITW_DELTA_K}"

echo "=== 6. export panels as runnable task directories ==="
P=assets/coreset/panels
"${PY}" scripts/coreset/export_panel.py "${P}/clean_mini_${MINI_SELECTOR}_k${CLEAN_MINI_K}.json" --task clean_mini           --overwrite
"${PY}" scripts/coreset/export_panel.py "${P}/itw_mini_${MINI_SELECTOR}_k${ITW_MINI_K}.json"     --task in-the-wild_mini  --overwrite
"${PY}" scripts/coreset/export_panel.py "${P}/clean_delta_delta_k${CLEAN_DELTA_K}.json"          --task clean_delta          --overwrite
"${PY}" scripts/coreset/export_panel.py "${P}/itw_delta_delta_k${ITW_DELTA_K}.json"              --task in-the-wild_delta --overwrite

echo "=== 7. regenerate RESULTS_CORESET.md from the CSVs + shipped panels ==="
"${PY}" scripts/coreset/gen_results_md.py > RESULTS_CORESET.md.tmp
mv RESULTS_CORESET.md.tmp RESULTS_CORESET.md
echo "wrote RESULTS_CORESET.md"

cat <<'EOF'

=== done ===
Run a system on a panel with the unmodified pipeline, e.g.:

  bash run_generate_omni.sh --model qwen3-tts-0.6b --task clean_mini --test_set en
  bash run_score.sh --decode_dir results/omni_qwen3-tts-0.6b/clean_mini --test_set en

Then read the weighted estimate of the FULL benchmark score:

  python scripts/coreset/score_panel.py --decode_dir results/omni_qwen3-tts-0.6b/clean_mini
EOF
