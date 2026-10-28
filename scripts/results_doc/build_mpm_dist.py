#!/usr/bin/env python3
"""Build the explorer's MPM prosody-realism payload from the shuffle-floor tsvs.

MPM `span` = (d_ref - d_ref_floor) / (d_noise_floor - d_ref_floor): each system's
distance from real speech, normalized PER LANGUAGE onto [0 = indistinguishable from
ground truth .. 1 = fully contour-scrambled]. Because it is normalized to each
language's own real/scramble anchors it is comparable across languages, so the
explorer draws a small-multiple dot plot (one row per language) on the shared span
axis. span > 1 means the system is FURTHER from real than the scrambled floor.

Reads results/_mpm_shuffle_floor/zero_shot_<lang>.tsv (and the ground-truth split rows
in results/_mpm_ground_truth/) and writes

    _cache/mpm_dist.json  ->  published to explorer/snapshot/ (publish_snapshot.py)

Run from the pipeline (run.sh) or standalone:
    python scripts/results_doc/build_mpm_dist.py
"""
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DIST_DIR = os.path.join(REPO_ROOT, "results", "_mpm_shuffle_floor")
GT_DIST_DIR = os.path.join(REPO_ROOT, "results", "_mpm_ground_truth")
CACHE = os.path.join(SCRIPT_DIR, "_cache")
OUT = os.path.join(CACHE, "mpm_dist.json")

sys.path.insert(0, SCRIPT_DIR)
from rd_common import label as _clean, write_json_atomic  # noqa: E402

ORDER = ["en", "de", "es", "fr", "it", "ja", "ko", "ru", "zh"]


def _rows(path):
    """(system, span) rows from a shuffle-floor tsv."""
    out = {}
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(hdr)}
        if "system" not in ix or "span" not in ix:
            return out
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(ix["system"], ix["span"]):
                continue
            try:
                out[_clean(p[ix["system"]])] = float(p[ix["span"]])
            except ValueError:
                pass
    return out


def main():
    os.makedirs(CACHE, exist_ok=True)
    per_lang = {}
    # `zero_shot_*.tsv` also matches `zero_shot_voxblink_<lang>.tsv`, which would yield
    # a phantom language named "voxblink_<lang>" on the shared span axis. MPM is
    # FLEURS-R only (docs/mpm_itw_limits.md), so no such file exists today -- skip them
    # explicitly so enabling VoxBlink MPM cannot silently corrupt this payload.
    for f in sorted(glob.glob(os.path.join(DIST_DIR, "zero_shot_*.tsv"))):
        lang = os.path.basename(f)[len("zero_shot_"):-len(".tsv")]
        if lang.startswith("voxblink"):
            print(f"  skip {os.path.basename(f)}: not FLEURS-R (see docs/mpm_itw_limits.md)")
            continue
        per_lang[lang] = _rows(f)
    # Ground-truth rows are scored separately on a random 50/50 reference split.
    for f in sorted(glob.glob(os.path.join(GT_DIST_DIR, "zero_shot_*.tsv"))):
        lang = os.path.basename(f)[len("zero_shot_"):-len(".tsv")]
        if lang.startswith("voxblink"):
            continue
        per_lang.setdefault(lang, {}).update(_rows(f))

    if not per_lang:
        write_json_atomic(OUT, {"langs": [], "systems": [], "xmax": 1.0})
        print("wrote empty mpm_dist.json (no tsvs)")
        return

    # per-system mean span across languages -> legend order (best first) + colour index
    spans = {}
    for sys_spans in per_lang.values():
        for name, sp in sys_spans.items():
            spans.setdefault(name, []).append(sp)
    systems = sorted(({"name": n, "mean": round(sum(v) / len(v), 4)}
                      for n, v in spans.items()), key=lambda s: s["mean"])
    xmax = max(sp for v in per_lang.values() for sp in v.values())

    langs = []
    for lang in sorted(per_lang, key=lambda l: (ORDER.index(l) if l in ORDER else 99, l)):
        rows = sorted(({"name": n, "span": round(sp, 4)} for n, sp in per_lang[lang].items()),
                      key=lambda s: s["span"])
        langs.append({"lang": lang, "systems": rows})

    write_json_atomic(OUT, {"xmax": round(xmax + 0.05, 3), "systems": systems, "langs": langs},
                      separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)} "
          f"({len(langs)} langs, {len(systems)} systems, xmax {xmax:.2f})")


if __name__ == "__main__":
    main()
