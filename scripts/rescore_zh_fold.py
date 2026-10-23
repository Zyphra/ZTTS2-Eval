#!/usr/bin/env python3
"""Re-score every zh leaf under the symmetric traditional->simplified fold, offline.

The 2026-08 audit moved the zhconv trad->simplified fold into
utils/text_norm.prepare(), applied to reference AND hypothesis (it was
hypothesis-only, so traditional characters in a reference were scored as
errors even when the system spoke them correctly). The stored
`wav_res_ref_text.wer.raw` already holds the reference + ASR hypothesis for
every utterance -- and the stored zh hypotheses were folded before storage by
the old code, so re-folding them is a no-op -- which makes this recompute
exact with no ASR rerun (same offline pattern as rescore_ortho.py).

Walks EVERY results leaf whose base lang is zh: <sys>/<task>/zh (all tasks,
incl. the mini/delta panel sets) and the flat gt_*/whisper_* baseline dirs.

Report-only by default; pass --write to overwrite each leaf's `.wer.raw` and
re-run average_wer.py to refresh `.wer` (with CI + coverage footers).

    python scripts/rescore_zh_fold.py            # preview old -> new WER
    python scripts/rescore_zh_fold.py --write    # commit the re-score
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "utils"))
from text_norm import base_lang, process_one  # noqa: E402

WER_RE = re.compile(r"^WER:\s*([\d.]+|nan)")


def committed_wer(werf):
    try:
        for ln in open(werf):
            if ln.startswith("WER:"):
                return float(WER_RE.match(ln).group(1))
    except (OSError, AttributeError, ValueError):
        pass
    return None


def zh_leaves():
    """Yield every results/ leaf dir holding a .wer.raw whose base lang is zh."""
    R = os.path.join(REPO, "results")
    for sysd in sorted(os.listdir(R)):
        top = os.path.join(R, sysd)
        if not os.path.isdir(top):
            continue
        for root, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d not in ("wavs", "waveform")]
            if "wav_res_ref_text.wer.raw" not in files:
                continue
            leaf = os.path.basename(root)
            if base_lang(leaf) == "zh":
                yield root


def rescore_leaf(d, write):
    leaf = os.path.basename(d.rstrip("/"))
    raw = os.path.join(d, "wav_res_ref_text.wer.raw")
    werf = os.path.join(d, "wav_res_ref_text.wer")
    old = committed_wer(werf)
    lines, wers = [], []
    for line in open(raw, errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        wav, ref, hyp = p[0], p[2], p[3]
        _, _, wer, subs, dele, inse = process_one(hyp, ref, "zh")
        wers.append(wer)
        lines.append(f"{wav}\t{wer}\t{ref}\t{hyp}\t{inse}\t{dele}\t{subs}\n")
    if not wers:
        return None
    new_mean = round(100 * sum(wers) / len(wers), 3)
    if write:
        with open(raw, "w") as f:
            f.writelines(lines)
        subprocess.run([sys.executable, os.path.join(REPO, "utils", "average_wer.py"), raw, werf],
                       check=True, stdout=subprocess.DEVNULL)
        new_mean = committed_wer(werf)          # read back the averaged value
    return old, new_mean, len(wers)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="overwrite .wer.raw + refresh .wer")
    args = ap.parse_args()
    print(f"symmetric zh fold re-score   ({'WRITING' if args.write else 'preview'})")
    print(f"{'leaf':<52}{'old WER':>9}{'new WER':>9}{'Δ':>8}{'n':>7}")
    n_leaves = 0
    for d in zh_leaves():
        r = rescore_leaf(d, args.write)
        if not r:
            continue
        old, new, n = r
        n_leaves += 1
        rel = os.path.relpath(d, os.path.join(REPO, "results"))
        delta = (new - old) if (old is not None and new is not None) else float("nan")
        olds = f"{old:9.2f}" if old is not None else f"{'--':>9}"
        print(f"{rel:<52}{olds}{new:>9.2f}{delta:>+8.2f}{n:>7}")
    print(f"{n_leaves} leaves")


if __name__ == "__main__":
    main()
