#!/usr/bin/env python3
"""Per-language text-length-bucket WEIGHTED WER for voxblink.

weighted_wer[bucket] = sum(edits over utts) / sum(ref_len over utts)

Equivalent to total-error-rate over the bucket's pooled tokens — the correct
apples-to-apples comparison across length buckets, since simple averaging of
per-utt WER amplifies short refs (1 error in 5 words = 20% WER; same error in
30 words = 3.3%).

Tokenization is scripts/tufte_common.py's ref_token_count — the exact
reconstruction of utils/run_wer.py's units (whitespace words, or characters for
the CHAR_LANGS zh/ja/ko/th) — imported rather than re-implemented so the pooled
∑edits/∑ref_len weights can never drift from the scorer's own token counts.
"""
import os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import data_langs, discover_systems, write_json_atomic  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from tufte_common import ref_token_count  # noqa: E402  (run_wer.py's exact units)

ROOT_RES = os.path.join(REPO_ROOT, "results")
ROOT_DATA = os.path.join(REPO_ROOT, "data", "zero_shot_voxblink")

VOX_LANGS = data_langs(ROOT_DATA)   # every materialized voxblink language
N_BUCKETS = 6

uid_bucket = {}
for lang in VOX_LANGS:
    p = f"{ROOT_DATA}/{lang}/text"
    if not os.path.isfile(p): continue
    pairs = []
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) < 2: continue
            uid, text = parts
            pairs.append((uid, len(text)))
    pairs.sort(key=lambda x: x[1])
    n = len(pairs)
    for b in range(N_BUCKETS):
        lo = (b * n) // N_BUCKETS
        hi = ((b+1) * n) // N_BUCKETS
        for uid, _L in pairs[lo:hi]:
            uid_bucket[(lang, uid)] = b
print(f"bucketed {len(uid_bucket)} utts")

WAVPATH_RE = re.compile(r"/wavs/(uttid_\d+)\.wav")   # uid scheme unified to uttid_N

def parse_wer_raw_with_ref(path):
    if not os.path.isfile(path): return None
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            mp = WAVPATH_RE.search(parts[0])
            if not mp: continue
            uid = mp.group(1)
            try: w = float(parts[1])
            except: continue
            ref = parts[2]
            out.append((uid, min(w, 1.0), ref))
    return out

# Baselines live in flat results/<model>/<lang>/ dirs; every TTS system is
# discovered from the results tree (mirrors build_perbin.py — hard-coded model
# lists are how this cache came to silently hold only stale baselines).
BASELINES = {"gt": "gt_zero_shot_voxblink"}
MODELS = dict(BASELINES)
MODELS.update(discover_systems("zero_shot_voxblink"))
print(f"models: {', '.join(sorted(MODELS))}")

per_bucket = {}
for label, mdir in MODELS.items():
    for lang in VOX_LANGS:
        task = "" if label in BASELINES else "zero_shot_voxblink"
        base = os.path.join(ROOT_RES, mdir, task, lang) if task else os.path.join(ROOT_RES, mdir, lang)
        path = os.path.join(base, "wav_res_ref_text.wer.raw")
        recs = parse_wer_raw_with_ref(path)
        if not recs: continue
        edits = [0.0]*N_BUCKETS
        ref_lens = [0]*N_BUCKETS
        for uid, w, ref in recs:
            b = uid_bucket.get((lang, uid))
            if b is None: continue
            rl = ref_token_count(ref, lang)
            edits[b] += w * rl
            ref_lens[b] += rl
        per_bucket[(label, lang)] = [
            (edits[b] / ref_lens[b] * 100) if ref_lens[b] > 0 else None
            for b in range(N_BUCKETS)
        ]

if not per_bucket:
    # Same guard as build_perlen.py: an empty payload means the parser broke,
    # not that there is nothing to report.
    sys.exit("build_perlen_weighted: 0 entries parsed from a non-empty results tree -- refusing to write an empty payload")

write_json_atomic(os.path.join(CACHE, "voxblink_perlen_weighted.json"),
                  {f"{l}|{lang}": vals for (l,lang), vals in per_bucket.items()},
                  indent=2)
print(f"wrote {len(per_bucket)} entries")
