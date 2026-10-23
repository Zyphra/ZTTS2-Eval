#!/usr/bin/env python3
"""Backfill the utterances a leaf's WER scorer never produced a row for.

Every published leaf was scored by handing `wav_res_ref_text` to utils/run_wer.py,
which writes one `.raw` row per utterance it managed to transcribe. Utterances that
failed were simply absent, and until 2026-09-01 the aggregate quietly averaged the
survivors -- so the more of a system's output was untranscribable, the less of it
counted against the system. Across the tree that was 4197 utterances in 135 leaves,
every one of them with its generated wav present on disk (nothing was a missing
generation), concentrated in the systems with the most degenerate output.

The fix is applied per leaf in three steps, and this script owns the first and third:

    manifest  write the RESIDUAL input list -- the manifest lines whose wav has no
              row in the .raw -- so run_wer.py can be pointed at just those.
    merge     fold the residual .raw back into the leaf's .raw and rebuild the
              .meta sidecar, then utils/average_wer.py re-aggregates.

The middle step (running run_wer.py on the residual list) is the only part that
needs a GPU, and it is what makes the backfill honest rather than a pure
imputation: the residual is re-measured under TODAY's uniform scoring policy
(current retries, loop guard, per-language routing), and only what still fails to
transcribe is scored as maximum error by run_wer.py itself. Leaves scored before
ASR_RETRIES existed therefore stop being penalised for having had no retries.

Merging is idempotent: rows already present are never duplicated or overwritten,
so a partial or repeated backfill converges to the same file.
"""

import argparse
import json
import os
import sys


def read_manifest(path):
    """[(wav_path, raw_line)] for the scorer's input list, in order."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append((s.split("|")[0], line if line.endswith("\n") else line + "\n"))
    return out


def read_rows(path):
    """{wav_path: row} for a .raw, keyed on its first column."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            c = line.split("\t")
            if len(c) >= 7 and c[0] != "utt":
                out[c[0]] = line if line.endswith("\n") else line + "\n"
    return out


def leaf_paths(leaf):
    return (os.path.join(leaf, "wav_res_ref_text"),
            os.path.join(leaf, "wav_res_ref_text.wer.raw"))


def cmd_manifest(args):
    man_path, raw_path = leaf_paths(args.leaf)
    man = read_manifest(man_path)
    have = read_rows(raw_path)
    residual = [line for wav, line in man if wav not in have]
    if not residual:
        print(f"[residuals] {args.leaf}: complete ({len(man)} utterances)")
        return 1 if args.require else 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(residual)
    print(f"[residuals] {args.leaf}: {len(residual)}/{len(man)} unscored -> {args.out}")
    return 0


def cmd_merge(args):
    man_path, raw_path = leaf_paths(args.leaf)
    man = read_manifest(man_path)
    base = read_rows(raw_path)
    extra = read_rows(args.residual_raw)

    # Rows already in the leaf win: they are the original measurement.
    added = {w: r for w, r in extra.items() if w not in base}
    merged = dict(base)
    merged.update(added)

    ordered = [merged[w] for w, _ in man if w in merged]
    stray = [r for w, r in merged.items() if w not in {m[0] for m in man}]
    if stray:
        print(f"[residuals] warning: {len(stray)} row(s) not in the manifest, kept at end",
              file=sys.stderr)
        ordered += stray

    n_imputed = sum(1 for r in ordered if r.split("\t")[3].startswith("<UNSCOREABLE"))
    still_missing = [w for w, _ in man if w not in merged]

    # Carry forward whatever the residual run recorded about each failure.
    reasons = {}
    for meta in (raw_path + ".meta", args.residual_raw + ".meta"):
        if os.path.isfile(meta):
            try:
                for u in json.load(open(meta)).get("unscored", []):
                    reasons[u.get("wav")] = u.get("reason")
            except (OSError, ValueError):
                pass

    if not args.dry_run:
        with open(raw_path, "w", encoding="utf-8") as f:
            f.writelines(ordered)
        with open(raw_path + ".meta", "w", encoding="utf-8") as f:
            json.dump({"n_expected": len(man),
                       "n_transcribed": len(ordered) - n_imputed,
                       "n_imputed": n_imputed,
                       "n_rows": len(ordered),
                       "impute_policy": True,
                       "backfilled": len(added),
                       "unscored": [{"wav": os.path.basename(w),
                                     "reason": reasons.get(os.path.basename(w), "unknown"),
                                     "imputed": False} for w in still_missing]},
                      f, indent=1)
    print(f"[residuals] {args.leaf}: +{len(added)} rows "
          f"({n_imputed} imputed of {len(ordered)}), {len(still_missing)} still unscored")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="write the residual input list for a leaf")
    m.add_argument("leaf")
    m.add_argument("out")
    m.add_argument("--require", action="store_true",
                   help="exit 1 when the leaf has no residual (drives shell skips)")
    m.set_defaults(func=cmd_manifest)

    g = sub.add_parser("merge", help="fold a residual .raw back into the leaf")
    g.add_argument("leaf")
    g.add_argument("residual_raw")
    g.add_argument("--dry-run", action="store_true")
    g.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
