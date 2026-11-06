#!/usr/bin/env python3
"""Build the explorer's speaking-rate payload — the interactive twin of
scripts/prosody/speaking_rate_dist.py.

The metric is a distributional distance between a system's per-utterance speaking
rate (characters per second of ACTUAL SPEECH) and the reference set's, scored with
the exact same 2-Wasserstein as TTSDS2. We import speaking_rate_dist so the numbers
here match the CSV it writes (same trim, same cap, same ceiling, same W2).

For each (test_set, language) that has at least one system, this collapses the
per-utterance char/s arrays into a compact, portable JSON the browser can render
without numpy or librosa:

  - the reference: n, median char/s, lead+trail silence, and a quantile function
    (char/s at a fixed probability grid P — plotting (q, P) gives the ECDF);
  - per system: n, heard_pct (share of its utterances the ASR heard with WER <= 20%,
    the only ones scored: speaking_rate_dist's content filter), median,
    truncation_pct, d_ref + bootstrap 95% CI, x_ceiling, and the same quantile function.

char/s is only comparable WITHIN a language (a Chinese character is a whole
syllable), so every panel is one language — never cross-lang. Languages/systems
whose wav dirs are absent are simply skipped.

    _cache/speaking_rate.json  ->  published to explorer/snapshot/ (publish_snapshot.py)

Run from the pipeline (run.sh) or standalone:
    python scripts/results_doc/build_speaking_rate.py
    python scripts/results_doc/build_speaking_rate.py --sets clean --langs en de
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
CACHE_DIR = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(CACHE_DIR, "speaking_rate.json")

# reuse the exact metric (trim, cap, W2, ceiling, CI) — same import trick as
# build_reliability.py reusing tufte_common.
from scripts.prosody import speaking_rate_dist as sr  # noqa: E402
from utils.ground_truth_split import split_indices  # noqa: E402
from utils.scp import load_text  # noqa: E402
from scripts.prosody.common import boot_ci, ceiling, find_systems, heard_files, w2, w2_fair  # noqa: E402
from utils.atomic import write_json_atomic  # noqa: E402

# one label per test_set, mirroring build_reliability.py's dataset labels
SET_LABEL = {
    "clean": ("FLEURS-R — read speech", "FLEURS-R"),
    "in-the-wild": ("VoxBlink2 — in the wild", "VoxBlink2"),
}
# quantile function grid: char/s at these probabilities. Interior only (the 0/1
# tails are the noisy min/max); 49 points draw a smooth ECDF. Shared across series.
P = [round(float(x), 4) for x in np.linspace(0.02, 0.98, 49)]


def quantiles(a):
    return [round(float(v), 3) for v in np.quantile(a, P)] if a.size else []


def langs_for(test_set, only):
    base = os.path.join(REPO, "data", test_set)
    if not os.path.isdir(base):
        return []
    out = []
    for lang in sorted(os.listdir(base)):
        d = os.path.join(base, lang)
        if all(os.path.exists(os.path.join(d, p)) for p in ("waveform", "prompt_text", "text")):
            if not only or lang in only:
                out.append(lang)
    return out


def build_panel(test_set, lang, args):
    base = os.path.join(REPO, "data", test_set, lang)
    systems = {} if args.ground_truth_only else find_systems(test_set, lang)
    prompt_text = load_text(os.path.join(base, "prompt_text"))
    target_text = load_text(os.path.join(base, "text"))

    ref_files = sorted(glob.glob(os.path.join(base, "waveform", "*.wav")))
    ref_all, ref_pad = sr.char_rates(ref_files, prompt_text, args.workers)
    ref = ref_all[ref_all <= args.cap]
    if ref.size < (2 if args.ground_truth_only else args.min_utts):
        return None
    ceil = ceiling(ref)

    series = []
    left_idx, right_idx = split_indices(ref.size, args.seed, f"{test_set}/{lang}")
    gt = ref[left_idx]
    gt_ref = ref[right_idx]
    gt_d_ref = w2(gt, gt_ref)
    gt_lo, gt_hi = boot_ci(gt, gt_ref, args.n_boot, args.seed)
    series.append({
        "name": f"gt_{test_set}",
        "n": int(gt.size),
        "median": round(float(np.median(gt)), 2),
        "pad_s": round(ref_pad, 3),
        "truncation_pct": 0.0,
        "d_ref": round(float(gt_d_ref), 3),
        "ci": [round(float(gt_lo), 3), round(float(gt_hi), 3)],
        "x_ceiling": round(float(gt_d_ref / ceil), 2) if ceil > 0 else None,
        "q": quantiles(gt),
        "split": "random 50/50",
        "seed": args.seed,
    })
    for name, d in sorted(systems.items()):
        files, heard_pct = heard_files(d)             # content filter: WER <= common.MAX_WER only
        if len(files) < args.min_utts:
            continue
        rates, pad = sr.char_rates(files, target_text, args.workers)
        if rates.size < args.min_utts:
            continue
        keep = rates[rates <= args.cap]
        trunc = 100.0 * (rates.size - keep.size) / rates.size
        d_ref = w2_fair(keep, ref)
        lo, hi = boot_ci(keep, ref, args.n_boot)
        series.append({
            "name": name,
            "n": int(rates.size),
            "heard_pct": round(100.0 * heard_pct, 1),
            "median": round(float(np.median(keep)), 2),
            "pad_s": round(pad, 3),
            "truncation_pct": round(trunc, 2),
            "d_ref": round(float(d_ref), 3),
            "ci": [round(float(lo), 3), round(float(hi), 3)],
            "x_ceiling": round(float(d_ref / ceil), 2) if ceil > 0 else None,
            "q": quantiles(keep),
        })
    if not series:
        return None
    series.sort(key=lambda r: r["d_ref"])
    label, short = SET_LABEL.get(test_set, (test_set, test_set))
    return {
        "key": f"{test_set}|{lang}",
        "set": test_set,
        "set_label": label,
        "chip": f"{short} · {lang}",
        "lang": lang,
        "unit": "char/s",
        "ceiling": round(float(ceil), 3),
        "ref": {
            "n": int(ref.size),
            "median": round(float(np.median(ref)), 2),
            "pad_s": round(ref_pad, 3),
            "q": quantiles(ref),
        },
        "series": series,
    }


def merge_shards(shard_dir, out_path, base_path=None):
    """Merge one-panel payloads, optionally overlaying them onto a base payload."""
    panels = {}
    cap = None
    if base_path:
        with open(base_path) as fh:
            base = json.load(fh)
        cap = base.get("cap")
        panels = {panel["key"]: panel for panel in base.get("panels", [])}

    paths = sorted(glob.glob(os.path.join(shard_dir, "*.json")))
    if not paths:
        raise SystemExit(f"no speaking-rate shards in {shard_dir}")
    for path in paths:
        with open(path) as fh:
            shard = json.load(fh)
        if shard.get("P") != P:
            raise SystemExit(f"incompatible probability grid in {path}")
        if cap is None:
            cap = shard.get("cap")
        elif shard.get("cap") != cap:
            raise SystemExit(f"incompatible cap in {path}")
        for panel in shard.get("panels", []):
            key = panel["key"]
            if key not in panels:
                panels[key] = panel
                continue
            if not base_path:
                raise SystemExit(f"duplicate speaking-rate panel {key} in {path}")
            updates = {series["name"]: series for series in panel["series"]}
            current = [series for series in panels[key]["series"]
                       if series["name"] not in updates]
            panels[key]["series"] = sorted(current + list(updates.values()),
                                            key=lambda row: row["d_ref"])

    merged = {"P": P, "cap": cap, "panels": [panels[k] for k in sorted(panels)]}
    write_json_atomic(out_path, merged, separators=(",", ":"))
    print(f"merged {len(paths)} shards / {len(panels)} panels -> "
          f"{os.path.relpath(out_path, REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["clean", "in-the-wild"])
    ap.add_argument("--langs", nargs="*", default=None, help="subset of langs; default = all present")
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--min-utts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default=OUT, help="output JSON path")
    ap.add_argument("--merge-shards", default=None,
                    help="merge per-panel JSON files from this directory and exit")
    ap.add_argument("--merge-base", default=None,
                    help="existing payload to overlay when merging shards")
    ap.add_argument("--ground-truth-only", action="store_true",
                    help="score only a reproducible random reference half against the other half")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.merge_shards:
        merge_shards(args.merge_shards, args.out, args.merge_base)
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    panels = []
    for test_set in args.sets:
        for lang in langs_for(test_set, args.langs):
            panel = build_panel(test_set, lang, args)
            if panel:
                panels.append(panel)
                print(f"[speaking_rate] {panel['key']}: {len(panel['series'])} system(s) · "
                      f"ref n={panel['ref']['n']} · ceiling {panel['ceiling']}")
            else:
                print(f"[speaking_rate] {test_set}|{lang}: no eligible system — skipped",
                      file=sys.stderr)

    out = {"P": P, "cap": args.cap, "panels": panels}
    if not panels:
        print("[speaking_rate] no panels had inputs — writing empty payload", file=sys.stderr)
    out_path = os.path.abspath(args.out)
    write_json_atomic(out_path, out, separators=(",", ":"))
    print(f"wrote {os.path.relpath(out_path, REPO)} "
          f"({os.path.getsize(out_path) / 1024:.0f} KB, {len(panels)} panels)")


if __name__ == "__main__":
    main()
