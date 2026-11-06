#!/usr/bin/env bash
# Submit the whole benchmark release compute DAG. Run from the repo root on the login
# node (the pinned scorer revisions must already be in the shared HF/torch caches).
#
#   J_tg  build_voxblink_targets.py --import-from results/gt_in-the-wild
#   J_hf  upload_release_data.py            (only when HF_TOKEN is set; nothing depends on it)
#   J_rs  rescore_all.sh  array             (needs J_tg: held-out speaker sim reads target_wav.scp)
#   J_gc  run_gt.sh --task clean
#   J_gi  run_gt.sh --task in-the-wild     (needs J_tg)
#   J_m   mpm + rhythm + pitch_range arrays (need J_rs, J_gc, J_gi)
#   J_g   mpm_gt array                      (reads only data/; no dependency)
#   J_fin results_doc/run.sh + RESULTS_CORESET.md           (needs J_m, J_g)
#
# Environment:
#   RELEASE_PARTITION / RELEASE_ACCOUNT / RELEASE_QOS   site values (none hard-coded)
#   RELEASE_RESCORE_NODES   rescore array size (default 4)
#   RELEASE_SKIP_TARGETS=1  skip J_tg and J_hf (restart after the targets exist)
#   HF_TOKEN                enables the HF upload job
set -euo pipefail

if [ ! -f slurm/submit_release_run.sh ]; then
    echo "run from the repository root" >&2
    exit 1
fi

site_args=()
if [ -n "${RELEASE_PARTITION:-}" ]; then site_args+=(--partition="${RELEASE_PARTITION}"); fi
if [ -n "${RELEASE_ACCOUNT:-}" ]; then site_args+=(--account="${RELEASE_ACCOUNT}"); fi
if [ -n "${RELEASE_QOS:-}" ]; then site_args+=(--qos="${RELEASE_QOS}"); fi
export METRICS_PARTITION="${RELEASE_PARTITION:-}" METRICS_ACCOUNT="${RELEASE_ACCOUNT:-}" \
       METRICS_QOS="${RELEASE_QOS:-}"
nodes="${RELEASE_RESCORE_NODES:-4}"
skip_targets="${RELEASE_SKIP_TARGETS:-0}"
repo="$(pwd -P)"
run_dir="${repo}/logs/slurm/release-run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${run_dir}" logs/slurm

# sbatch with the flags every job shares; the output file is per job name unless the
# script's own #SBATCH header owns it (J_rs).
sb() {
    sbatch --parsable --chdir="${repo}" --export=ALL \
        --output=logs/slurm/%x-%j.out ${site_args[@]+"${site_args[@]}"} "$@"
}

J_tg=""; J_hf=""
if [ "${skip_targets}" != "1" ]; then
    J_tg=$(sb --job-name=ztts-targets --cpus-per-task=16 --mem=64G --time=04:00:00 \
        --wrap ".venv/bin/python scripts/prep/build_voxblink_targets.py --import-from results/gt_in-the-wild")
    if [ -n "${HF_TOKEN:-}" ]; then
        J_hf=$(sb --job-name=ztts-hf-upload --dependency="afterok:${J_tg}" \
            --cpus-per-task=8 --mem=32G --time=10:00:00 \
            --wrap ".venv/bin/python scripts/prep/upload_release_data.py")
    else
        echo "HF upload skipped: HF_TOKEN unset"
    fi
fi
after_tg=()
if [ -n "${J_tg}" ]; then after_tg=(--dependency="afterok:${J_tg}"); fi

J_rs=$(sbatch --parsable --chdir="${repo}" --export=ALL ${site_args[@]+"${site_args[@]}"} \
    ${after_tg[@]+"${after_tg[@]}"} --array="0-$((nodes - 1))" scripts/score/rescore_all.sh)
J_gc=$(sb --job-name=ztts-gt-clean slurm/score_job.sh \
    bash scripts/score/run_gt.sh --task clean)
J_gi=$(sb --job-name=ztts-gt-itw ${after_tg[@]+"${after_tg[@]}"} slurm/score_job.sh \
    bash scripts/score/run_gt.sh --task in-the-wild)

# speaking is omitted: the final run.sh step recomputes char/s, GT halves included.
out=$(METRICS_KINDS="mpm rhythm pitch_range" METRICS_FINALIZE=0 \
      METRICS_DEPENDENCY="afterok:${J_rs}:${J_gc}:${J_gi}" bash slurm/submit_metrics.sh)
printf '%s\n' "${out}"
J_m=$(printf '%s\n' "${out}" | sed -n 's/^METRICS_JOBS=//p' | tail -n1)
out=$(METRICS_KINDS="mpm_gt" METRICS_FINALIZE=0 bash slurm/submit_metrics.sh)
printf '%s\n' "${out}"
J_g=$(printf '%s\n' "${out}" | sed -n 's/^METRICS_JOBS=//p' | tail -n1)
if [ -z "${J_m}" ] || [ -z "${J_g}" ]; then
    echo "metrics submitters returned no job ids" >&2
    exit 1
fi

J_fin=$(sb --job-name=ztts-release-build --dependency="afterok:${J_m}:${J_g}" \
    --cpus-per-task=112 --mem=400G --time=10:00:00 \
    --wrap "bash scripts/results_doc/run.sh && .venv/bin/python scripts/coreset/gen_results_md.py > RESULTS_CORESET.md.tmp && mv RESULTS_CORESET.md.tmp RESULTS_CORESET.md")

cat > "${run_dir}/jobs.env" <<EOF
J_TG=${J_tg}
J_HF=${J_hf}
J_RS=${J_rs}
J_GC=${J_gc}
J_GI=${J_gi}
J_M=${J_m}
J_G=${J_g}
J_FIN=${J_fin}
EOF
ids=$(printf '%s:%s:%s:%s:%s:%s:%s:%s' "${J_tg}" "${J_hf}" "${J_rs}" "${J_gc}" "${J_gi}" "${J_m}" "${J_g}" "${J_fin}" \
      | tr ':' '\n' | grep . | paste -sd,)
echo "jobs written to ${run_dir}/jobs.env"
echo "watch: squeue -j ${ids}"
