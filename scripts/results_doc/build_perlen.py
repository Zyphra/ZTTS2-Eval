#!/usr/bin/env python3
"""Per-language text-length-bucket WER for voxblink (simple-mean clipped WER)."""
import os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import data_langs, discover_systems, write_json_atomic  # noqa: E402

ROOT_RES = os.path.join(REPO_ROOT, "results")
ROOT_DATA = os.path.join(REPO_ROOT, "data", "zero_shot_voxblink")

VOX_LANGS = data_langs(ROOT_DATA)   # every materialized voxblink language
N_BUCKETS = 6

uid_bucket = {}
bucket_ranges = {}
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
    ranges = []
    for b in range(N_BUCKETS):
        lo = (b * n) // N_BUCKETS
        hi = ((b+1) * n) // N_BUCKETS
        chunk = pairs[lo:hi]
        if chunk:
            ranges.append((chunk[0][1], chunk[-1][1], len(chunk)))
        else:
            ranges.append((0, 0, 0))
        for uid, _L in chunk:
            uid_bucket[(lang, uid)] = b
    bucket_ranges[lang] = ranges

print(f"assigned {len(uid_bucket)} uid->bucket")
write_json_atomic(os.path.join(CACHE, "voxblink_uid_lenbucket.json"),
                  {f"{l}|{u}": b for (l,u), b in uid_bucket.items()})
write_json_atomic(os.path.join(CACHE, "voxblink_lenbucket_ranges.json"), bucket_ranges)

WAVPATH_RE = re.compile(r"/wavs/(uttid_\d+)\.wav")   # uid scheme unified to uttid_N

def parse_wer_raw(path):
    if not os.path.isfile(path): return None
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            mp = WAVPATH_RE.search(parts[0])
            if not mp: continue
            uid = mp.group(1)
            try: w = float(parts[1])
            except: continue
            out.append((uid, min(w, 1.0)))
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
        recs = parse_wer_raw(path)
        if not recs: continue
        bin_vals = [[] for _ in range(N_BUCKETS)]
        for uid, w in recs:
            b = uid_bucket.get((lang, uid))
            if b is None: continue
            bin_vals[b].append(w * 100)
        per_bucket[(label, lang)] = [
            (sum(v)/len(v)) if v else None for v in bin_vals
        ]

if not per_bucket:
    # A results tree with wavs but zero parsed entries means the parser (regex,
    # layout) is broken -- publishing an empty {} silently blanks every
    # downstream plot, so fail loudly instead.
    sys.exit("build_perlen: 0 (model,lang) entries parsed from a non-empty results tree -- refusing to write an empty payload")

write_json_atomic(os.path.join(CACHE, "voxblink_perlen.json"),
                  {f"{l}|{lang}": vals for (l,lang), vals in per_bucket.items()},
                  indent=2)
print(f"wrote {len(per_bucket)} (model,lang) per-bucket entries")
