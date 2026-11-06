#!/usr/bin/env bash
# One-GPU scoring job: runs any scoring driver given as its arguments, with its own
# ASR/Whisper port window and the cgroup GPU pinned (the one allocated GPU is index 0).
#
#   sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
#          bash scripts/score/run_gt.sh --task clean
#   sbatch --chdir="$(pwd -P)" --export=ALL slurm/score_job.sh \
#          bash scripts/score/run_score.sh --decode-dir logs/smoke/in-the-wild --test-set en
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

[ -f scripts/score/serve_asr.sh ] || { echo "submit with --chdir=<repo root>" >&2; exit 1; }
. slurm/lib_slurm.sh

# Unique localhost ports per job so 1-GPU jobs packed onto one node never share a port
# (slot 0 of the job's slurm/lib_slurm.sh window, as in slurm/eval_job.sh).
job_id="${SLURM_JOB_ID:-$$}"
port_base=$(slurm_port_base 0)
export ASR_PORT=$(slurm_role_port "${port_base}" asr) WHISPER_PORT=$(slurm_role_port "${port_base}" whisper)
export ASR_GPU=0 WHISPER_GPU=0 SCORE_GPU=0
export SERVE_LOG_DIR=$(slurm_server_log_dir "score-${job_id}")

exec "$@"
