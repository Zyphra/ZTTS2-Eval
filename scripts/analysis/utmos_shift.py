"""UTMOS shift (protocol S3): signed mean shift of a system's per-utterance
MSR-UTMOS distribution against the human reference's, with its spread ratio.

What it is
----------
MSR-UTMOS is already scored on every leaf (results/<sys>/<set>/<lang>/
msr_utmos_scores.txt). Its *level* is not read - it ranks real in-the-wild
speech below every TTS system (docs/controls_2026-08.md) - so the protocol
reads only the position of a system's distribution relative to the human
reference's:

  dmean      = mean(system) - mean(reference), with a 95% bootstrap CI
               (independent utterance resamples of both sides);
  sd_ratio   = sd(system) / sd(reference)  (< 1 = narrower than real speech);
  dmean_band = the null band of dmean at matched geometry: p95 of
               |mean(A*) - mean(B*)| over 1,000 pairs of independent size-n
               bootstrap re-draws of the reference (n = the reference size, the
               same n-vs-n geometry as dmean itself; a half-vs-half split would
               be sqrt(2) wider). A system whose |dmean| sits inside the band is
               indistinguishable from a re-draw of the reference.

The 2-Wasserstein distance at split-half geometry (`d_ref`, `x_ceiling`;
the S7 construction) is kept as a secondary column. Across this tree it is
the mean shift in disguise - Spearman(d_ref, |dmean|) over systems is printed
as the diagnostic - so it is never the headline and never a ranking. S3 is a
regression alarm: read the change in dmean vs. the previous release.
On in-the-wild references the reference distribution carries the channel
(VoxBlink2 is YouTube audio), so absolute dmean is not read there.

Reference = the GT leaf of the same (set, lang): results/gt_<set>/<lang>/
msr_utmos_scores.txt. On FLEURS-R that leaf holds the prompt recordings
(different text, same speakers/domain) - immaterial for a quality
distribution. The GT leaf is never listed as a system.

Usage
-----
    python scripts/analysis/utmos_shift.py                                   # clean / en
    python scripts/analysis/utmos_shift.py --test-set in-the-wild --lang en
    python scripts/analysis/utmos_shift.py --models omni_voxcpm2 --out /tmp/x.csv

Output CSV: results/utmos_shift_<set>_<lang>.csv, sorted by |dmean|
(rank 0 = the reference itself: dmean 0, sd_ratio 1, the null band).
"""
import argparse
import csv
import os
import sys

import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from scripts.prosody.rhythm_dist import w2_fair  # noqa: E402
from scripts.prosody.speaking_rate_dist import boot_ci, ceiling, w2  # noqa: E402
from utils.per_uid import parse_score_path  # noqa: E402

SCORE_FILE = "msr_utmos_scores.txt"


def read_scores(path):
    """`<uttid>\\t<wav>\\t<score>` rows -> float array (footer / malformed rows skipped)."""
    return np.asarray(list(parse_score_path(path).values()), dtype=np.float64)


def find_score_files(test_set, lang, only=None):
    """Same two-layout walk as speaking_rate_dist.find_systems, keyed on score files.

    nested: results/<sys>/<set>/<lang>/msr_utmos_scores.txt (+ results/<group>/<sys>/...)
    flat:   results/<backend>_<set>/<lang>/msr_utmos_scores.txt  (gt_, whisper_, ...)
    The `*<set>` glob keeps clean and in-the-wild disjoint.
    """
    import glob
    results_root = os.path.join(REPO, "results")
    pats = (
        f"results/*/{test_set}/{lang}/{SCORE_FILE}",
        f"results/*/*/{test_set}/{lang}/{SCORE_FILE}",
        f"results/*{test_set}/{lang}/{SCORE_FILE}",
    )
    out = {}
    for pat in pats:
        for f in glob.glob(os.path.join(REPO, pat)):
            rel = os.path.relpath(os.path.dirname(f), results_root)   # <...>/<lang>
            name = rel[: -len(f"/{lang}")]
            if name.endswith(f"/{test_set}"):
                name = name[: -len(f"/{test_set}")]
            out[name] = f
    if only:
        out = {k: v for k, v in out.items() if k in set(only)}
    return out


def dmean_ci(x, ref, n_boot, seed=0):
    """95% percentile CI of mean(x) - mean(ref) under independent utterance resamples."""
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    br = rng.choice(ref, size=(n_boot, ref.size), replace=True).mean(axis=1)
    d = bx - br
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def dmean_band(ref, n_draws=1000, seed=0):
    """p95 of |mean(A*) - mean(B*)| over independent size-n bootstrap re-draw pairs
    of the reference: the null distribution of dmean at the n-vs-n geometry dmean
    is computed at (a half-vs-half split would be sqrt(2) wider)."""
    rng = np.random.default_rng(seed)
    a = rng.choice(ref, size=(n_draws, ref.size), replace=True).mean(axis=1)
    b = rng.choice(ref, size=(n_draws, ref.size), replace=True).mean(axis=1)
    return float(np.percentile(np.abs(a - b), 95))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-set", default="clean")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--models", nargs="*", default=None, help="subset of system dirs; default = all found")
    ap.add_argument("--out", default=None, help="CSV path; default results/utmos_w2_<set>_<lang>.csv")
    ap.add_argument("--min-utts", type=int, default=250)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    gt_name = f"gt_{args.test_set}"
    files = find_score_files(args.test_set, args.lang, args.models)
    ref_path = files.pop(gt_name, None) or os.path.join(REPO, "results", gt_name, args.lang, SCORE_FILE)
    if not os.path.isfile(ref_path):
        raise SystemExit(f"missing reference leaf {ref_path}")
    ref = read_scores(ref_path)
    floor = ceiling(ref)
    band = dmean_band(ref)
    print(f"reference: {ref_path}")
    print(f"  n={ref.size}  mean {ref.mean():.3f}  sd {ref.std(ddof=1):.3f}")
    print(f"  |dmean| null band (p95, 1000 n-vs-n re-draw pairs) = {band:.4f}; W2 split-half floor d_ref = {floor:.4f}\n")

    rows = []
    for name, path in sorted(files.items()):
        x = read_scores(path)
        if x.size < args.min_utts:
            continue
        d_ref = w2_fair(x, ref)
        lo, hi = boot_ci(x, ref, args.n_boot)
        dm = float(x.mean() - ref.mean())
        dlo, dhi = dmean_ci(x, ref, args.n_boot)
        rows.append(dict(system=name, n=int(x.size), utmos_mean=float(x.mean()), utmos_sd=float(x.std(ddof=1)),
                         dmean=dm, dmean_ci_lo=dlo, dmean_ci_hi=dhi, dmean_band=band,
                         sd_ratio=float(x.std(ddof=1) / ref.std(ddof=1)),
                         d_ref=d_ref, d_ref_full=w2(x, ref), ci_lo=lo, ci_hi=hi, x_ceiling=d_ref / floor))
        print(f"  {name:34s} n={x.size:4d} mean {x.mean():5.3f} dmean {dm:+6.3f} [{dlo:+.3f},{dhi:+.3f}]  "
              f"sd_ratio {x.std(ddof=1) / ref.std(ddof=1):4.2f}  d_ref {d_ref:6.4f} x{d_ref / floor:5.2f}", flush=True)

    if not rows:
        raise SystemExit("no system leaf with enough scored utterances")
    rows.sort(key=lambda r: abs(r["dmean"]))
    out = args.out or os.path.join(REPO, "results", f"utmos_shift_{args.test_set}_{args.lang}.csv")
    fields = ["rank", "system", "n", "utmos_mean", "utmos_sd", "dmean", "dmean_ci_lo", "dmean_ci_hi", "dmean_band",
              "sd_ratio", "d_ref", "d_ref_full", "ci_lo", "ci_hi", "x_ceiling"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(rank=0, system="__REFERENCE__", n=ref.size, utmos_mean=float(ref.mean()),
                        utmos_sd=float(ref.std(ddof=1)), dmean=0.0, dmean_ci_lo="", dmean_ci_hi="", dmean_band=band,
                        sd_ratio=1.0, d_ref=floor, d_ref_full="", ci_lo="", ci_hi="", x_ceiling=1.0))
        for i, r in enumerate(rows):
            w.writerow(dict(rank=i + 1, **r))

    d = np.array([r["d_ref"] for r in rows])
    dm = np.array([abs(r["dmean"]) for r in rows])
    print(f"\n{len(rows)} systems -> {out}")
    print(f"inside the reference |dmean| band ({band:.3f}): {int((dm <= band).sum())}/{len(rows)}; "
          f"W2 at/below floor: {int((d <= floor).sum())}/{len(rows)}")
    print(f"Spearman(d_ref, |dmean|) over systems = {spearman(d, dm):.3f}  "
          f"(near 1 = the W2 distance is the mean shift; dmean is the headline)")


if __name__ == "__main__":
    main()
