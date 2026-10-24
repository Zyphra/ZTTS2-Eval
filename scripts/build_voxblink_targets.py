#!/usr/bin/env python3
"""Validate the held-out target recordings of the ITW eval and write their scp files.

Published ITW speaker similarity used to compare each generation with the prompt it was
conditioned on, which rewards copying that one recording (channel, mic, session) as much
as the voice. The primary score instead compares against a DIFFERENT real recording of
the same speaker: the recording behind the item's target text, stored at
`data/zero_shot_voxblink/<lang>/target_wav/<uid>.wav` (ships on the HF dataset
Zyphra/ZTTS2-Eval; rebuilt from raw VoxBlink2 by scripts/extract_voxblink_gt.py).

Every uid of the current `text` is validated against `_stage_a/manifest.tsv`, joined on
prompt_text exactly as extract_voxblink_gt.py does:
  - the joined row is unambiguous (cross-speaker prompt_text collisions excluded),
  - prompt and target come from different videos of that speaker,
  - the target wav exists, and its duration matches the manifest target segment
    (catches a stale or misaligned extraction).

Output per language:
  data/zero_shot_voxblink/<lang>/target_wav.scp          <uid> <repo-relative wav>
  data/zero_shot_voxblink/<lang>/target_provenance.json  pairing rule, counts, exclusions
  data/zero_shot_voxblink_{mini,delta}/<lang>/target_wav.scp   the panel subset

The reference wav shares the generation's target text, so content is matched for
generations but not for the ceiling (real prompt vs. real target).

Usage:
    python scripts/build_voxblink_targets.py --import-from results/gt_zero_shot_voxblink
    python scripts/build_voxblink_targets.py --langs en zh
"""
import argparse
import csv
import json
import os
import shutil
import sys

import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.benchmark import ITW_LANGS, data_uids  # noqa: E402
from utils.scp import load_scp_items  # noqa: E402

TASK = "zero_shot_voxblink"
DATA_ROOT = f"data/{TASK}"
PANELS = (f"data/{TASK}_mini", f"data/{TASK}_delta")
DUR_TOL_S = 0.05


def import_wavs(src_root, lang, uids):
    """Copy <src_root>/<lang>/wavs/<uid>.wav into target_wav/ (follows symlinks)."""
    dst_dir = os.path.join(DATA_ROOT, lang, "target_wav")
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for uid in uids:
        src = os.path.join(src_root, lang, "wavs", f"{uid}.wav")
        dst = os.path.join(dst_dir, f"{uid}.wav")
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        shutil.copyfile(src, dst)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--langs", nargs="+", default=list(ITW_LANGS))
    ap.add_argument("--import-from", default=None, metavar="DIR",
                    help="copy DIR/<lang>/wavs/<uid>.wav into target_wav/ first")
    args = ap.parse_args()
    os.chdir(REPO)

    rows_by_key = {}
    with open(os.path.join(DATA_ROOT, "_stage_a/manifest.tsv"), newline="",
              encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows_by_key.setdefault((row["lang"], row["prompt_text"]), []).append(row)

    failed = False
    for lang in args.langs:
        uids = data_uids(TASK, lang)
        if args.import_from:
            n = import_wavs(args.import_from, lang, uids)
            print(f"{lang}: imported {n} wavs from {args.import_from}")
        prompts = dict(load_scp_items(os.path.join(DATA_ROOT, lang, "prompt_text")))
        refs, excluded, speakers = [], {}, set()
        for uid in uids:
            matches = rows_by_key.get((lang, prompts.get(uid)), [])
            if not matches:
                excluded[uid] = "no manifest row for prompt_text"
                continue
            if len({m["speaker"] for m in matches}) > 1:
                excluded[uid] = "prompt_text shared across speakers (ambiguous)"
                continue
            row = matches[0]
            if row["prompt_vid"] == row["target_vid"]:
                excluded[uid] = "prompt and target from the same video"
                continue
            ref = f"{DATA_ROOT}/{lang}/target_wav/{uid}.wav"
            if not os.path.isfile(ref):
                excluded[uid] = "target wav missing"
                continue
            want = float(row["target_end"]) - float(row["target_start"])
            got = sf.info(ref).duration
            if abs(got - want) > DUR_TOL_S:
                excluded[uid] = f"duration {got:.3f}s != manifest {want:.3f}s"
                continue
            refs.append((uid, ref))
            speakers.add(row["speaker"])

        out_dir = os.path.join(DATA_ROOT, lang)
        with open(os.path.join(out_dir, "target_wav.scp"), "w", encoding="utf-8") as fh:
            fh.writelines(f"{uid} {path}\n" for uid, path in refs)
        provenance = {
            "source_task": TASK,
            "lang": lang,
            "reference": "real recording of the item's target text (same speaker, "
                         "different video from the prompt)",
            "n_items": len(uids),
            "n_references": len(refs),
            "n_speakers": len(speakers),
            "excluded": excluded,
        }
        with open(os.path.join(out_dir, "target_provenance.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)
            fh.write("\n")
        # Panel subsets: the same lines, restricted to the panel's uids.
        ref_by_uid = dict(refs)
        for panel in PANELS:
            pdir = os.path.join(panel, lang)
            if not os.path.isfile(os.path.join(pdir, "text")):
                continue
            puids = [u for u, _ in load_scp_items(os.path.join(pdir, "text"))]
            with open(os.path.join(pdir, "target_wav.scp"), "w", encoding="utf-8") as fh:
                fh.writelines(f"{u} {ref_by_uid[u]}\n" for u in puids if u in ref_by_uid)
        print(f"{lang}: {len(refs)}/{len(uids)} references, {len(speakers)} speakers, "
              f"{len(excluded)} excluded -> {out_dir}/target_wav.scp")
        for uid, why in list(excluded.items())[:10]:
            print(f"  excluded {uid}: {why}")
        failed |= not refs
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
