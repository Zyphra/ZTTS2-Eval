# Fail fast: a run_wer.py that dies partway (OOM, SLURM preemption, uncaught
# exception) must not have its partial .raw silently averaged into a
# complete-looking .wer -- that is exactly the stub-leaf failure the resume
# guards then skip forever. average_wer.py additionally exits non-zero itself
# when scored/expected coverage falls below SCORE_MIN_COVERAGE (default 0.98).
set -euo pipefail
set -x

meta_lst=$1
output_dir=$2
lang=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

# ASR runs against the vLLM Qwen3-ASR server (continuous batching server-side), so
# the old per-GPU file split is gone -- run_wer.py drives concurrency as a client.
python3 get_wav_res_ref_text.py "$meta_lst" "$output_dir/wavs" "$wav_wav_text"
python3 run_wer.py "$wav_wav_text" "$score_file.raw" "$lang"
python3 average_wer.py "$score_file.raw" "$score_file"
