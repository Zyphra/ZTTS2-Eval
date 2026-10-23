#!/usr/bin/env python3
"""Repair `wav_res_ref_text.wer.raw` records broken by tabs/newlines in the ASR hypothesis.

`utils/run_wer.py` writes one TAB-separated record per utterance:

    <wav_path> TAB wer TAB ref_text TAB hyp_text TAB ins TAB del TAB sub

but it does not escape the hypothesis, so an ASR transcription containing a literal TAB
or newline splits its own record. `utils/average_wer.py` then dies on the first such line
(`malformed line`) leaving a **truncated .wer** -- the per-utterance rows up to that point
plus no `WER:` footer at all. Anything reading the footer (`scripts/results_doc/build_results.py`,
and any hand-rolled aggregation) silently loses that leaf.

That is not hypothetical: it silently dropped one language each from three systems in
RESULTS_NEWLANGS.md (omnivoice-reftext/he, zonos2_expressive/he, voxtral-tts/th), which
moved omnivoice-reftext from +2.3 to +0.6 vs the GT floor and changed the reported
ranking.

The repair is conservative. Lines are re-accumulated until a record has at least 7 fields
whose numeric positions (wer, ins, del, sub) all parse and whose first field is a .wav
path; the stray TABs/newlines inside the hypothesis are then replaced by spaces. A record
that already parses is rewritten byte-identically. Records that cannot be reassembled are
reported and left alone rather than guessed at.

    python scripts/repair_wer_raw.py                       # dry-run scan of results/
    python scripts/repair_wer_raw.py --apply               # rewrite + regenerate .wer
    python scripts/repair_wer_raw.py --apply path/to/leaf  # a specific leaf or file
"""

import argparse
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _complete(fields):
    """True when `fields` is a whole record: .wav path, numeric wer, numeric ins/del/sub."""
    return (len(fields) >= 7
            and fields[0].endswith(".wav")
            and "\n" not in fields[0]
            and _is_float(fields[1])
            and all(_is_float(x) for x in fields[-3:]))


def _starts_record(line):
    """True when a raw line is a plausible record START: its first TAB field is a
    .wav path. Continuation lines of a newline-split hypothesis never are."""
    return line.split("\t", 1)[0].endswith(".wav")


def repair_lines(text):
    """Reassemble records from a possibly-corrupted .raw body.

    Returns (records, n_repaired, dropped) where each record is the canonical 7-field
    tuple and `dropped` holds fragments that could not be reassembled.

    A record can only START at a .wav path, so when a new record start arrives while
    an incomplete fragment is buffered, the fragment is flushed as unrecoverable
    instead of being glued onto (and corrupting) the new record -- the old behavior
    could both delete good records queued behind a dead fragment and emit records
    whose wav-path field contained a literal newline.
    """
    records, dropped = [], []
    n_repaired = 0
    buf = []
    for line in text.split("\n"):
        if not buf and not line.strip():
            continue
        if buf and _starts_record(line):
            # The buffered fragment can never be completed by lines from the record
            # that starts here: report it, keep this record intact.
            dropped.append("\n".join(buf))
            buf = []
        buf.append(line)
        fields = "\n".join(buf).split("\t")
        if not _complete(fields):
            continue
        if len(buf) > 1 or len(fields) > 7:
            n_repaired += 1
        hyp = " ".join(" ".join(fields[3:-3]).split())
        records.append([fields[0], fields[1], fields[2], hyp,
                        fields[-3], fields[-2], fields[-1]])
        buf = []
    if buf and "".join(buf).strip():
        dropped.append("\n".join(buf))
    return records, n_repaired, dropped


def process(raw_path, apply=False, regen=True):
    with open(raw_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    records, n_repaired, dropped = repair_lines(text)
    n_lines = sum(1 for line in text.split("\n") if line.strip())
    status = "ok" if not n_repaired and not dropped else "REPAIRABLE"
    if n_repaired or dropped:
        print(f"  [{status}] {raw_path}")
        print(f"      {len(records)} records from {n_lines} raw lines; "
              f"{n_repaired} reassembled, {len(dropped)} unrecoverable")
    if not apply or (not n_repaired and not dropped):
        return n_repaired, len(dropped)

    with open(raw_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write("\t".join(r) + "\n")
    if regen:
        wer_path = raw_path[: -len(".raw")]
        res = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "utils", "average_wer.py"),
                              raw_path, wer_path], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"      average_wer.py FAILED: {res.stderr.strip().splitlines()[-1:]}")
        else:
            print(f"      rewrote {os.path.basename(wer_path)}: "
                  f"{res.stdout.strip().splitlines()[0]}")
    return n_repaired, len(dropped)


def targets(paths):
    if not paths:
        return sorted(glob.glob(os.path.join(REPO_ROOT, "results", "**",
                                             "wav_res_ref_text.wer.raw"), recursive=True))
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "wav_res_ref_text.wer.raw"),
                                        recursive=True)))
        elif os.path.isfile(p):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="leaf dirs or .raw files (default: all of results/)")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the .raw files and regenerate their .wer (default: dry run)")
    ap.add_argument("--no-regen", action="store_true", help="do not re-run average_wer.py")
    args = ap.parse_args()

    files = targets(args.paths)
    print(f"scanning {len(files)} .raw files"
          f"{' (dry run -- pass --apply to rewrite)' if not args.apply else ''}\n")
    tot_r = tot_d = n_bad = 0
    for f in files:
        r, d = process(f, apply=args.apply, regen=not args.no_regen)
        if r or d:
            n_bad += 1
        tot_r += r
        tot_d += d
    print(f"\n{n_bad} file(s) affected; {tot_r} record(s) reassembled, "
          f"{tot_d} unrecoverable fragment(s)")


if __name__ == "__main__":
    main()
