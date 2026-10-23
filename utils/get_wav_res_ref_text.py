import sys, os

try:  # tqdm is cosmetic here; keep this script stdlib-runnable (smoke-testable, no venv)
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_res_ref_text = sys.argv[3]
print(metalst, wav_dir, wav_res_ref_text)
with open(metalst) as f:
    lines = f.readlines()

n_kept = 0
with open(wav_res_ref_text, 'w') as f_w:
    for line in tqdm(lines):
        utt, infer_text = line.strip().split(maxsplit=1)
        if not os.path.exists(os.path.join(wav_dir, utt + '.wav')):
            continue
        out_line = '|'.join([os.path.join(wav_dir, utt + '.wav'), infer_text])
        f_w.write(out_line + '\n')
        n_kept += 1

# Missing wavs are dropped before WER ever sees them, so surface the coverage here
# (run_wer.py reports its own scored/total over the lines that survive this filter).
n_missing = len(lines) - n_kept
print(f"[wav_res_ref_text] kept {n_kept}/{len(lines)} utterances"
      + (f" -- WARNING: {n_missing} missing wavs in {wav_dir}" if n_missing else ""))
