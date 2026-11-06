#!/usr/bin/env python3
"""Same per-length bucketing applied to the controlled-content `clean` set
(FLEURS-R-derived, uniform register). The inverse-with-length voxblink trend
disappears here, exposing a U-shape — see voxblink-length-confound memory."""
import os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import discover_systems, write_json_atomic  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from tufte_common import ref_token_count  # noqa: E402  (run_wer.py's exact units)

ROOT_RES = os.path.join(REPO_ROOT, "results")
ROOT_DATA = os.path.join(REPO_ROOT, "data", "clean")

LANGS = ["en","zh","de","es","fr","it","ja","ko","ru"]  # skip hard_* (only 60-64 utts)
N_BUCKETS = 6

uid_bucket = {}
for lang in LANGS:
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
        lo = (b*n)//N_BUCKETS
        hi = ((b+1)*n)//N_BUCKETS
        for uid, _L in pairs[lo:hi]:
            uid_bucket[(lang, uid)] = b
print(f"bucketed {len(uid_bucket)}")

WAVPATH_RE = re.compile(r"/wavs/(uttid_\d+)\.wav")

def parse_wer_raw(path):
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

# Every TTS system with a clean leaf is discovered from the results tree
# (mirrors build_perbin.py); no hard-coded model list to go stale.
MODELS = discover_systems("clean")
print(f"models: {', '.join(sorted(MODELS))}")

simple = {}
weighted = {}
for label, mdir in MODELS.items():
    for lang in LANGS:
        path = os.path.join(ROOT_RES, mdir, "clean", lang, "wav_res_ref_text.wer.raw")
        recs = parse_wer_raw(path)
        if not recs: continue
        bin_vals = [[] for _ in range(N_BUCKETS)]
        edits = [0.0]*N_BUCKETS
        ref_lens = [0]*N_BUCKETS
        for uid, w, ref in recs:
            b = uid_bucket.get((lang, uid))
            if b is None: continue
            bin_vals[b].append(w*100)
            rl = ref_token_count(ref, lang)
            edits[b] += w * rl
            ref_lens[b] += rl
        simple[(label,lang)] = [(sum(v)/len(v)) if v else None for v in bin_vals]
        weighted[(label,lang)] = [
            (edits[b]/ref_lens[b]*100) if ref_lens[b]>0 else None
            for b in range(N_BUCKETS)
        ]

write_json_atomic(os.path.join(CACHE, "clean_perlen_simple.json"),
                  {f"{l}|{lang}": v for (l,lang),v in simple.items()}, indent=2)
write_json_atomic(os.path.join(CACHE, "clean_perlen_weighted.json"),
                  {f"{l}|{lang}": v for (l,lang),v in weighted.items()}, indent=2)
print(f"wrote {len(simple)} simple, {len(weighted)} weighted")
