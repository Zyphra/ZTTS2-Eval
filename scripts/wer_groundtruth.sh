# Baseline WER: transcribe the prompt (ground-truth human) clips of a test set
# with Qwen3-ASR and score them against prompt_text. This is the ASR's own error
# floor on genuine human speech -- a reference point for a TTS system's WER.
#
# Run from the repo root, inside the venv:
#   uv run bash scripts/wer_groundtruth.sh <data-dir> <lang> <out-dir>
# e.g.
#   uv run bash scripts/wer_groundtruth.sh data/zero_shot/en en results/gt_zero_shot_en
set -e

data_dir=$1          # test-set dir with prompt_wav.scp + prompt_text
lang=$2              # language code (en, zh, ...)
out_dir=$3           # output directory
mkdir -p "$out_dir"
lst="$out_dir/gt.lst"

# Build a '<wav>|<reference text>' list from the prompt clips and their transcripts.
python - "$data_dir" "$lst" <<'PYEOF'
import sys

def load(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            uid, val = line.split(maxsplit=1)
            d[uid] = val
    return d

data_dir, lst = sys.argv[1], sys.argv[2]
wav = load(f"{data_dir}/prompt_wav.scp")
txt = load(f"{data_dir}/prompt_text")
with open(lst, "w", encoding="utf-8") as f:
    n = 0
    for uid in wav:
        if uid in txt:
            # '|' is run_wer.py's field delimiter; strip any from the transcript
            # (it is punctuation and dropped during WER scoring anyway).
            f.write(f"{wav[uid]}|{txt[uid].replace('|', ' ')}\n")
            n += 1
print(f"{n} ground-truth clips -> {lst}")
PYEOF

# Start the Qwen3-ASR server, score WER, stop the server.
. scripts/serve_asr.sh
trap 'stop_asr_server' EXIT INT TERM
start_asr_server || exit 1
python utils/run_wer.py "$lst" "$out_dir/gt.wer.raw" "$lang"
python utils/average_wer.py "$out_dir/gt.wer.raw" "$out_dir/gt.wer"
echo "ground-truth WER -> $out_dir/gt.wer"
tail -1 "$out_dir/gt.wer"
