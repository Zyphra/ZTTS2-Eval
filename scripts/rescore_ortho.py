#!/usr/bin/env python3
"""Re-score the abjad GT languages under the orthographic normalization in
utils/text_norm.py, offline. The stored `wav_res_ref_text.wer.raw` already holds
the routed reference + ASR hypothesis for every utterance, so we only recompute
the per-utterance WER / ins / del / sub with the new `prepare()` — no ASR rerun.

Report-only by default; pass --write to overwrite the leaf's `.wer.raw` and
re-run average_wer.py to refresh `.wer` (with its CI footer).

    python scripts/rescore_ortho.py                    # preview old -> new WER
    python scripts/rescore_ortho.py --write            # commit the re-score

Set ORTHO_NORM_LANGS="" to get the no-op control (must reproduce committed WER).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "utils"))
from text_norm import base_lang, process_one, ORTHO_NORM_LANGS  # noqa: E402

WER_RE = re.compile(r"^WER:\s*([\d.]+)")
ROOTS = ["results/gt_clean", "results/gt_in-the-wild"]


def committed_wer(werf):
    try:
        for ln in open(werf):
            if ln.startswith("WER:"):
                return float(WER_RE.match(ln).group(1))
    except OSError:
        pass
    return None


def rescore_leaf(d, write):
    leaf = os.path.basename(d.rstrip("/"))
    base = base_lang(leaf)
    raw = os.path.join(d, "wav_res_ref_text.wer.raw")
    werf = os.path.join(d, "wav_res_ref_text.wer")
    if not os.path.isfile(raw):
        return None
    old = committed_wer(werf)
    lines, wers = [], []
    for line in open(raw):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        wav, ref, hyp = p[0], p[2], p[3]
        _, _, wer, subs, dele, inse = process_one(hyp, ref, base)
        wers.append(wer)
        lines.append(f"{wav}\t{wer}\t{ref}\t{hyp}\t{inse}\t{dele}\t{subs}\n")
    new_mean = round(100 * sum(wers) / len(wers), 3) if wers else float("nan")
    if write:
        with open(raw, "w") as f:
            f.writelines(lines)
        subprocess.run([sys.executable, os.path.join(REPO, "utils", "average_wer.py"), raw, werf],
                       check=True, stdout=subprocess.DEVNULL)
        new_mean = committed_wer(werf)          # read back the averaged value
    return leaf, base, old, new_mean, len(wers)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="overwrite .wer.raw + refresh .wer")
    ap.add_argument("--langs", default="ar,fa,he",
                    help="abjad leaves to re-score (default: ar,fa,he). Whether folding "
                         "actually applies is controlled by ORTHO_NORM_LANGS in text_norm.")
    args = ap.parse_args()
    langs = set(args.langs.split(","))
    print(f"ORTHO_NORM_LANGS = {sorted(ORTHO_NORM_LANGS)}   ({'WRITING' if args.write else 'preview'})")
    print(f"{'set':<26}{'lang':<5}{'old WER':>9}{'new WER':>9}{'Δ':>8}{'n':>7}")
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(REPO, root, "*/"))):
            if base_lang(os.path.basename(d.rstrip("/"))) not in langs:
                continue
            r = rescore_leaf(d, args.write)
            if not r:
                continue
            leaf, base, old, new, n = r
            delta = (new - old) if (old is not None) else float("nan")
            print(f"{os.path.basename(root):<26}{leaf:<5}{old:>9.2f}{new:>9.2f}{delta:>+8.2f}{n:>7}")


if __name__ == "__main__":
    main()
