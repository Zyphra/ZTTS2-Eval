#!/usr/bin/env bash
# One-GPU scoring job: runs any scoring driver given as its arguments, with its own
# ASR/Whisper port window and the cgroup GPU pinned (the one allocated GPU is index 0).
#
#   sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
#          bash scripts/run_gt.sh --task clean
#   sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
#          bash run_score.sh --decode_dir logs/smoke/in-the-wild --test_set en
#
# Partition / account / QoS come from the submitter (sbatch flags or SBATCH_* env); none
# is hard-coded here.
#SBATCH --job-name=ztts-score
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -eo pipefail

[ -f scripts/serve_asr.sh ] || { echo "submit with --chdir=<repo root>" >&2; exit 1; }

# Unique localhost ports per job so 1-GPU jobs packed onto one node never share a port
# (same 4-port window as slurm/eval_job.sh).
job_id="${SLURM_JOB_ID:-$$}"
port_base=$(( 20000 + (job_id % 10000) * 4 ))
export ASR_PORT=$(( port_base + 1 )) WHISPER_PORT=$(( port_base + 2 ))
export ASR_GPU=0 WHISPER_GPU=0 SCORE_GPU=0

srv="logs/slurm/servers"
mkdir -p "${srv}"
export ASR_LOG="${srv}/score-${job_id}-asr.log" WHISPER_LOG="${srv}/score-${job_id}-whisper.log"

exec "$@"
