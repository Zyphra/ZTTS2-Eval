#!/usr/bin/env bash
# CONTROL (2026-08 methodology review): seen-speaker embedder contamination.
#
# The shipped speaker-similarity embedder (ReDimNet2-B6 `vb2+vox2_v0`) is trained
# ON VoxBlink2 -- the in-the-wild eval corpus -- so every ITW spk score is computed
# in a seen-speaker regime. This control rescores a language sample with the
# vox2-only variant (`--dataset vox2`, VoxBlink2-free training) into ALTERNATE
# filenames (spk_simi_scores.vox2.txt); published scores are untouched. Readout:
# per-lang system rank agreement (Kendall tau) + adjacent-rank flips vs the
# published ReDimNet-vb2 scores.
#
# Prereq: ~/.cache/torch/hub/checkpoints/b6-vox2-lm.pt (prefetched on a login node;
# GPU nodes may lack GitHub egress).
#
#   sbatch --job-name=ztts-ctl-spk --gpus=1 \
#          --cpus-per-task=14 --mem=100G --time=8:00:00 --chdir="$(pwd)" \
#          --output=logs/slurm/ztts-ctl-spk-%j.out scripts/control_spk_embedder.sh
set -euo pipefail
[ -f scripts/eval_speaker_similarity.py ] || cd "$(dirname "$0")/.." || exit 1

LANGS="${CTL_LANGS:-en zh de th}"
PY="${PY:-.venv/bin/python}"

[ -f "$HOME/.cache/torch/hub/checkpoints/b6-vox2-lm.pt" ] || {
  echo "[ctl-spk] ERROR: b6-vox2-lm.pt not prefetched"; exit 1; }

n_ok=0; n_fail=0; n_skip=0
for lg in ${LANGS}; do
  pscp="data/in-the-wild/${lg}/prompt_wav.scp"
  [ -f "${pscp}" ] || { echo "[skip] ${lg}: no prompt scp"; continue; }
  for d in results/*/in-the-wild/"${lg}" results/gt_in-the-wild/"${lg}"; do
    [ -d "${d}" ] || continue
    [ -s "${d}/wav.scp" ] || { echo "[skip] ${d} (no wav.scp)"; n_skip=$((n_skip+1)); continue; }
    out="${d}/spk_simi_scores.vox2.txt"
    if [ -s "${out}" ] && grep -q '^avg ' "${out}" && ! grep -q '^avg 0.00 0.00$' "${out}"; then
      echo "[skip] ${d} (already scored)"; n_ok=$((n_ok+1)); continue
    fi
    echo "=== ctl-spk ${d} ==="
    if "${PY}" scripts/eval_speaker_similarity.py \
        --prompt_wavs "${pscp}" --hyp_wavs "${d}/wav.scp" \
        --log_file "${out}" --device cuda --dataset vox2; then
      n_ok=$((n_ok+1))
    else
      echo "[ctl-spk] FAIL ${d}"; n_fail=$((n_fail+1))
    fi
  done
done
echo "[ctl-spk] done: ${n_ok} ok, ${n_fail} failed, ${n_skip} skipped"
[ "${n_fail}" -eq 0 ] && echo "CTL-SPK-OK"
