#!/usr/bin/env bash
# WER for one leaf: <ref text> <leaf dir with wavs/> <lang>.
#
# Fail fast: a run_wer.py that dies partway (OOM, SLURM preemption, uncaught
# exception) must not have its partial .raw silently averaged into a
# complete-looking .wer -- that is exactly the stub-leaf failure the resume
# guards then skip forever. average_wer.py additionally exits non-zero itself
# when scored/expected coverage falls below SCORE_MIN_COVERAGE (default 0.98).
set -euo pipefail
ref_text=$(readlink -f "$1"); leaf=$(readlink -f "$2"); lang=$3
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
set -x
"${PY}" utils/get_wav_res_ref_text.py "${ref_text}" "${leaf}/wavs" "${leaf}/wav_res_ref_text"
"${PY}" utils/run_wer.py "${leaf}/wav_res_ref_text" "${leaf}/wav_res_ref_text.wer.raw" "${lang}"
"${PY}" utils/average_wer.py "${leaf}/wav_res_ref_text.wer.raw" "${leaf}/wav_res_ref_text.wer"
