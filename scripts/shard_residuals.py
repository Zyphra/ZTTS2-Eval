#!/usr/bin/env python3
"""Find every results leaf with unscored utterances and split them into balanced
shards for the scripts/rescore_residuals.sh SLURM array.

Balanced by residual COUNT, not leaf count: the distribution is extremely skewed
(one leaf held 1331 of the 4197 unscored utterances tree-wide), so round-robin over
leaves would leave one array task running for hours while the rest idle. Greedy
longest-first assignment to the currently-lightest shard is enough here.

Also prints the audit the backfill rests on -- how many of the unscored utterances
have their generated wav on disk. Any that do not are NOT a scorer failure at all
(nothing was generated), so they are reported separately rather than folded in.
"""

import argparse
import os

SKIP_PREFIX = "_"          # results/_probe, _seeds, _mpm_* are not published
RAW = "wav_res_ref_text.wer.raw"
MANIFEST = "wav_res_ref_text"


def leaf_residual(leaf):
    """(n_expected, n_scored, n_missing_wav) for one leaf, or None if not scoreable."""
    man_p, raw_p = os.path.join(leaf, MANIFEST), os.path.join(leaf, RAW)
    if not (os.path.isfile(man_p) and os.path.isfile(raw_p)):
        return None
    expected = []
    with open(man_p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                expected.append(ln.split("|")[0])
    scored = set()
    with open(raw_p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            c = ln.split("\t")
            if len(c) >= 7:
                scored.add(c[0])
    missing = [w for w in expected if w not in scored]
    return len(expected), len(expected) - len(missing), sum(
        1 for w in missing if not os.path.exists(w))


def discover(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(SKIP_PREFIX)]
        if RAW not in filenames:
            continue
        r = leaf_residual(dirpath)
        if r and r[0] > r[1]:
            out.append((r[0] - r[1], dirpath, r[2]))
    out.sort(reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--out-dir", default="logs/slurm/residual_shards")
    args = ap.parse_args()

    leaves = discover(args.root)
    total = sum(n for n, _, _ in leaves)
    nowav = sum(k for _, _, k in leaves)
    print(f"{len(leaves)} leaves, {total} unscored utterances "
          f"({nowav} of them with no generated wav on disk)")

    shards = [[] for _ in range(args.shards)]
    load = [0] * args.shards
    for n, leaf, _ in leaves:                      # longest-first, lightest shard
        i = load.index(min(load))
        shards[i].append(leaf)
        load[i] += n

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (sh, w) in enumerate(zip(shards, load)):
        path = os.path.join(args.out_dir, f"shard_{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(leaf + "\n" for leaf in sh))
        print(f"  shard {i}: {len(sh):3d} leaves, {w:5d} utterances -> {path}")


if __name__ == "__main__":
    main()
