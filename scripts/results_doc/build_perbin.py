#!/usr/bin/env python3
"""Map each materialized voxblink prompt uid -> its MSR-UTMOS bin, then compute
per-bin WER per model.

Joins each materialized <lang>/prompt_text back to _stage_a/manifest.tsv to recover
the prompt clip, looks up its MSR-UTMOS score in _msr_utmos_cache.tsv, and bins it
(matching prepare_voxblink2_eval_stage_b.py). Outputs:
  _cache/voxblink_uid_bin.json
  _cache/voxblink_perbin.json
  _cache/voxblink_bin_counts.json
"""
import csv, os, sys, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from rd_common import data_langs, discover_systems, write_json_atomic  # noqa: E402

DATA = os.path.join(REPO_ROOT, "data", "zero_shot_voxblink")
MAN = f"{DATA}/_stage_a/manifest.tsv"
MSR_CACHE = f"{DATA}/_msr_utmos_cache.tsv"
ROOT_RES = os.path.join(REPO_ROOT, "results")

N_BINS = 5
MSR_UTMOS_MIN = 1.0
MSR_UTMOS_MAX = 4.0
bin_w = (MSR_UTMOS_MAX - MSR_UTMOS_MIN) / N_BINS

# Load MSR-UTMOS cache: (spk,vid,seg,start,end) -> score
cache = {}
with open(MSR_CACHE) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 6: continue
        spk, vid, seg, s, e, sc = parts
        cache[(spk, vid, seg, float(s), float(e))] = float(sc)
print(f"loaded {len(cache)} MSR-UTMOS cache entries", file=sys.stderr)

# Index manifest by (lang, prompt_text) -> prompt candidate_id (first match wins)
man_by_key = {}
with open(MAN) as f:
    for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
        cid = (row["speaker"], row["prompt_vid"], row["prompt_seg"],
               float(row["prompt_start"]), float(row["prompt_end"]))
        key = (row["lang"], " ".join(row["prompt_text"].split()))
        man_by_key.setdefault(key, cid)
print(f"indexed {len(man_by_key)} manifest prompt_text keys", file=sys.stderr)

# Bin each MATERIALIZED prompt by its MSR-UTMOS score, joining prompt_text back to
# the manifest to recover the source clip. Robust to the round-robin selection order.
uid_bin = {}
for lang in sorted(os.listdir(DATA)):
    ptext_path = os.path.join(DATA, lang, "prompt_text")
    if not os.path.isfile(ptext_path):
        continue
    with open(ptext_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, ptext = line.split(" ", 1)
            cid = man_by_key.get((lang, " ".join(ptext.split())))
            if cid is None:
                continue
            score = cache.get(cid, float("nan"))
            if score != score:  # NaN
                continue
            b = min(max(int((score - MSR_UTMOS_MIN) / bin_w), 0), N_BINS - 1)
            uid_bin[(lang, uid)] = b

write_json_atomic(os.path.join(CACHE, "voxblink_uid_bin.json"),
                  {f"{l}|{u}": b for (l,u), b in uid_bin.items()})
print(f"wrote {len(uid_bin)} uid->bin entries", file=sys.stderr)

WAVPATH_RE = re.compile(r"/wavs/(uttid_\d+)\.wav")   # uid scheme unified to uttid_N

def parse_wer_raw_clipped(path):
    out = []
    if not os.path.isfile(path): return None
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

# The two reference baselines live in flat results/<model>/<lang>/ dirs; every TTS
# system is discovered from the results tree instead of being listed here, so a newly
# scored system shows up on the chart the next time this runs rather than silently
# missing from it (which is how the cache came to hold only the baselines).
# Labels and discovery are the shared scripts/results_doc/rd_common.py versions, so
# this cache's keys match gen_md.py's labels exactly.
BASELINES = {"gt": "gt_zero_shot_voxblink", "whisper": "whisper_zero_shot_voxblink"}
VOX_TASK = "zero_shot_voxblink"

MODELS = dict(BASELINES)
MODELS.update(discover_systems(VOX_TASK))
print(f"models: {', '.join(sorted(MODELS))}", file=sys.stderr)
VOX_LANGS = data_langs(DATA)   # every materialized voxblink language

per_bin = {}
for label, mdir in MODELS.items():
    for lang in VOX_LANGS:
        task = "" if label in BASELINES else VOX_TASK
        base = os.path.join(ROOT_RES, mdir, task, lang) if task else os.path.join(ROOT_RES, mdir, lang)
        path = os.path.join(base, "wav_res_ref_text.wer.raw")
        recs = parse_wer_raw_clipped(path)
        if not recs: continue
        bin_vals = [[] for _ in range(N_BINS)]
        for uid, w in recs:
            b = uid_bin.get((lang, uid))
            if b is None: continue
            bin_vals[b].append(w * 100)
        per_bin[(label, lang)] = [
            (sum(v)/len(v)) if v else None for v in bin_vals
        ]

write_json_atomic(os.path.join(CACHE, "voxblink_perbin.json"),
                  {f"{l}|{lang}": vals for (l,lang), vals in per_bin.items()},
                  indent=2)
print(f"wrote {len(per_bin)} (model,lang) per-bin entries")

bin_counts = {}
for lang in VOX_LANGS:
    cnt = [0]*N_BINS
    for (l,u), b in uid_bin.items():
        if l==lang: cnt[b]+=1
    bin_counts[lang] = cnt
write_json_atomic(os.path.join(CACHE, "voxblink_bin_counts.json"), bin_counts, indent=2)
print("done")
