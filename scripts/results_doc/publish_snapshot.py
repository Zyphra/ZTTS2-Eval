#!/usr/bin/env python3
"""Publish the explorer's data snapshot.

The build scripts (build_results.py, build_perbin.py, ...) write into
scripts/results_doc/_cache/, which is gitignored *build scratch*: it mixes large
derived intermediates (the uid->bin index, etc.) with host-specific paths and is
regenerated from the heavy private inputs (the full results/ tree, the 13MB
VoxBlink manifest, the MSR-UTMOS cache).

The explorer, however, only needs a handful of small, portable aggregate JSONs to
render its board / grid / bins. This step copies exactly those — an explicit,
reviewable allowlist — from the scratch cache into explorer/snapshot/, which IS
committed. That directory is the shareable payload: a fresh clone serves a
populated explorer from it with no rebuild and without the private inputs.

Publishing is deliberately a separate act from building, so scratch is never
committed by accident and the shipped snapshot only changes when someone runs
this. Run it after the build pipeline (run.sh does this as its last step).

    python scripts/results_doc/publish_snapshot.py            # copy _cache -> snapshot
    python scripts/results_doc/publish_snapshot.py --check    # verify, don't write (CI)
"""
import argparse
import filecmp
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE_DIR = os.path.join(SCRIPT_DIR, "_cache")
PUBLISH_DIR = os.path.join(REPO_ROOT, "explorer", "snapshot")

# The allowlist: the only files the explorer serves, and the ONLY things that get
# committed out of the scratch cache. Keep this in sync with serve.py's PUBLISHED
# resolution. Bulky/derived intermediates (voxblink_uid_bin.json, ...) are absent
# on purpose — they never ship.
PUBLISHED = [
    "results_snapshot.json",     # board + grid + per-model/lang metrics
    "voxblink_perbin.json",      # bins view: WER_CLIP per MSR-UTMOS bin
    "voxblink_bin_counts.json",  # bins view: per-bin n
    "voxblink_perlen.json",      # optional: per-REF-length trend (if built)
    "reliability.json",          # reliability view: survival/p99/slopegraph (tufte plates)
    "speaking_rate.json",        # speaking-rate view: char/s d_ref + ECDF per system
    "mpm_dist.json",             # prosody-overlap view: GT-vs-system MPM distributions
    "panels.json",               # coreset panel provenance (captions the panel boards)
]
OPTIONAL = {"voxblink_perlen.json", "reliability.json", "speaking_rate.json",
            "mpm_dist.json", "panels.json"}  # absence is fine


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify explorer/snapshot is in sync with _cache; exit 1 if not "
                         "(no writes) — for CI / pre-commit")
    args = ap.parse_args()

    os.makedirs(PUBLISH_DIR, exist_ok=True)
    missing, stale, published = [], [], []
    for name in PUBLISHED:
        src = os.path.join(CACHE_DIR, name)
        dst = os.path.join(PUBLISH_DIR, name)
        if not os.path.isfile(src):
            if name not in OPTIONAL:
                missing.append(name)
            continue
        in_sync = os.path.isfile(dst) and filecmp.cmp(src, dst, shallow=False)
        if args.check:
            if not in_sync:
                stale.append(name)
            continue
        if not in_sync:
            # copy to a temp sibling + os.replace so a reader (the explorer's
            # published-snapshot fallback) can never observe a half-copied file
            tmp = dst + ".tmp"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            published.append(name)

    rel = os.path.relpath(PUBLISH_DIR, REPO_ROOT)
    if args.check:
        if missing:
            print(f"missing in _cache (build first): {', '.join(missing)}", file=sys.stderr)
        if stale:
            print(f"out of sync — run publish_snapshot.py: {', '.join(stale)}", file=sys.stderr)
        if missing or stale:
            return 1
        print(f"{rel}/ is in sync with _cache")
        return 0

    if missing:
        print(f"warning: not in _cache, skipped (build first): {', '.join(missing)}",
              file=sys.stderr)
    print(f"published {len(published)} file(s) to {rel}/"
          + (": " + ", ".join(published) if published else " (already up to date)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
