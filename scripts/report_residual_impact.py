#!/usr/bin/env python3
"""What the 2026-09-01 residual backfill changed, per leaf and per system.

The backfill (scripts/rescore_residuals.sh) rewrote each affected leaf's .raw in
place, so "before" is not recoverable from the leaf alone. It IS recoverable from
the run's own artifacts: every residual .raw it produced is kept under
logs/slurm/residual_shards/work-<jobid>/, and the leaf's rows minus those rows are
exactly the rows the published number was averaged over.

    python scripts/report_residual_impact.py --work-root logs/slurm/residual_shards

Reports, per leaf: the published (survivor) mean, the mean over the complete leaf,
how many residual utterances transcribed on the retry, and how many stayed
unscoreable and are now scored as maximum error. The last two columns are the
point of the exercise -- a large `retried` count means the gap was a scoring-run
failure and the old number was simply wrong, while a large `imputed` count means
the audio really is untranscribable and the old number was survivor-biased.
"""

import argparse
import glob
import os
from collections import defaultdict


def rows_of(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 7 and c[0] != "utt":
                try:
                    out[c[0]] = (float(c[1]), c[3].startswith("<UNSCOREABLE"))
                except ValueError:
                    continue
    return out


def clip_mean(vals):
    return 100.0 * sum(min(v, 1.0) for v in vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default="logs/slurm/residual_shards")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    # The work files are named after the leaf with "/" replaced by "_", which is not
    # invertible (system names contain "_" too), so map the leaves forward instead.
    by_tag = {d.replace(os.sep, "_"): d
              for pat in ("*/*/*", "*/*")
              for d in glob.glob(os.path.join(args.results, pat)) if os.path.isdir(d)}

    per_leaf = []
    for res_raw in sorted(glob.glob(os.path.join(args.work_root, "work-*", "*.raw"))):
        leaf = by_tag.get(os.path.basename(res_raw)[:-4])
        if leaf is None:
            print(f"warning: cannot locate leaf for {res_raw}")
            continue
        residual = rows_of(res_raw)
        now = rows_of(os.path.join(leaf, "wav_res_ref_text.wer.raw"))
        if not now:
            continue
        before = [w for k, (w, _) in now.items() if k not in residual]
        after = [w for w, _ in now.values()]
        n_imp = sum(1 for _, imp in now.values() if imp)
        per_leaf.append((leaf, clip_mean(before), clip_mean(after), len(before),
                         len(residual) - n_imp, n_imp))

    per_leaf.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)
    print(f"{'before':>8} {'after':>8} {'delta':>8} {'was':>6} {'retried':>7} {'imputed':>7}  leaf")
    for leaf, b, a, nb, nr, ni in per_leaf:
        print(f"{b:8.2f} {a:8.2f} {a - b:+8.2f} {nb:6d} {nr:7d} {ni:7d}  {leaf}")

    tot_r = sum(r[4] for r in per_leaf)
    tot_i = sum(r[5] for r in per_leaf)
    print(f"\n{len(per_leaf)} leaves backfilled: {tot_r} utterances transcribed on the "
          f"retry, {tot_i} still unscoreable and scored as maximum error")

    by_sys = defaultdict(lambda: [0, 0])
    for leaf, _, _, _, nr, ni in per_leaf:
        key = "/".join(leaf.split(os.sep)[1:3])
        by_sys[key][0] += nr
        by_sys[key][1] += ni
    print(f"\n{'retried':>7} {'imputed':>7}  system / task")
    for key, (nr, ni) in sorted(by_sys.items(), key=lambda kv: -kv[1][1]):
        print(f"{nr:7d} {ni:7d}  {key}")


if __name__ == "__main__":
    main()
