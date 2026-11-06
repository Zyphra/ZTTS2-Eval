#!/usr/bin/env python3
"""Print the ground-truth baselines with 95% CIs, straight from the committed
metric files (no re-scoring). WER + CI come from each leaf's
`wav_res_ref_text.wer` footer (written by average_wer.py); MSR-UTMOS + CI from
`msr_utmos_mean.txt` / `msr_utmos_ci.txt` (written by score_msr_utmos.py).

    python scripts/report_gt.py                 # both GT sets
    python scripts/report_gt.py --set clean     # gt_clean only
    python scripts/report_gt.py --sort lang     # sort by lang instead of WER

Backend column B: Q = the lang stayed on Qwen3-ASR (QWEN_LANGS), W = re-scored
with Whisper large-v3 forced-language (see docs/asr_whisper_vs_qwen.md).
"""
import argparse
import glob
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from utils.benchmark import QWEN_LANGS  # noqa: E402
SETS = {"clean": "gt_clean", "voxblink": "gt_in-the-wild"}

WER_RE = re.compile(r"^WER:\s*([\d.]+)")
WER_CI_RE = re.compile(r"WER_CI95:\s*\[([\d.]+),\s*([\d.]+)\]")
MSR_CI_RE = re.compile(r"CI95\s+([\d.]+)\s+([\d.]+)")


def _last(path, pred):
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return None
    for ln in reversed(lines):
        if pred(ln):
            return ln
    return None


def read_leaf(d):
    werf = os.path.join(d, "wav_res_ref_text.wer")
    wl = _last(werf, lambda x: x.startswith("WER:"))
    if not wl:
        return None
    wer = float(WER_RE.match(wl).group(1))
    cil = _last(werf, lambda x: x.startswith("WER_CI95:"))
    wer_ci = tuple(map(float, WER_CI_RE.search(cil).groups())) if cil else None
    msr = msr_ci = None
    mf = os.path.join(d, "msr_utmos_mean.txt")
    if os.path.exists(mf):
        try: msr = float(open(mf).read().strip())
        except ValueError: pass
    mc = os.path.join(d, "msr_utmos_ci.txt")
    if os.path.exists(mc):
        m = MSR_CI_RE.search(open(mc).read())
        if m: msr_ci = tuple(map(float, m.groups()))
    raw = os.path.join(d, "wav_res_ref_text.wer.raw")
    n = sum(1 for _ in open(raw)) if os.path.exists(raw) else None
    return dict(lang=os.path.basename(d.rstrip("/")), wer=wer, wer_ci=wer_ci,
                msr=msr, msr_ci=msr_ci, n=n)


def fmt_ci(ci, w=5, p=2):
    if not ci:
        return "        —        "
    return f"[{ci[0]:{w}.{p}f}, {ci[1]:{w}.{p}f}]"


def report(setkey, sort):
    root = os.path.join(REPO, "results", SETS[setkey])
    rows = [r for d in sorted(glob.glob(root + "/*/")) if (r := read_leaf(d))]
    if not rows:
        print(f"  (no results under {root})")
        return
    rows.sort(key=(lambda r: r["lang"]) if sort == "lang" else (lambda r: r["wer"]))
    print(f"===== {SETS[setkey]} ({len(rows)} langs) =====")
    print(f"  {'lang':<5} {'B':<1} {'n':>5}  {'WER':>6}  {'WER 95% CI':>16}   "
          f"{'MSR':>5}  {'MSR 95% CI':>14}")
    for r in rows:
        b = "Q" if r["lang"] in QWEN_LANGS else "W"
        msr = f"{r['msr']:5.2f}" if r["msr"] is not None else "   — "
        print(f"  {r['lang']:<5} {b:<1} {str(r['n'] or '?'):>5}  {r['wer']:6.2f}  "
              f"{fmt_ci(r['wer_ci']):>16}   {msr}  {fmt_ci(r['msr_ci'], 4, 2):>14}")
    macro = sum(r["wer"] for r in rows) / len(rows)
    print(f"  --- macro-WER over {len(rows)} langs = {macro:.2f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["clean", "voxblink", "both"], default="both")
    ap.add_argument("--sort", choices=["wer", "lang"], default="wer")
    args = ap.parse_args()
    keys = ["clean", "voxblink"] if args.set == "both" else [args.set]
    for k in keys:
        report(k, args.sort)


if __name__ == "__main__":
    main()
