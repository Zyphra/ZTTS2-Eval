#!/usr/bin/env python3
"""Materialize a selected panel as an ordinary benchmark task directory.

The panels ship as `data/<task>_<kind>/<lang>/` with exactly the layout the pipeline
already expects, so **no generation or scoring code has to change**:

    text            <uttid> <target text>
    prompt_text     <uttid> <reference speaker text>
    prompt_wav.scp  <uttid> <path to prompt wav>
    waveform/       symlinks to the original prompt audio
    panel.json      per-uid weights + provenance (read by score_panel.py)

Running the benchmark on a panel is then just:

    bash run_generate_omni.sh --model <m> --task clean_mini --test_set en
    bash run_score.sh --decode_dir results/omni_qwen3-tts-0.6b/clean_mini --test_set en

Original uids are preserved rather than renumbered, so any panel result can be traced
back to the full set. Note the two sets differ in how they write prompt paths -- the
FLEURS-R set uses repo-relative paths (with a doubled slash), VoxBlink2 absolute ones --
so paths are always rewritten for the new directory rather than copied.

    python scripts/coreset/export_panel.py assets/coreset/panels/itw_mini_strat_meta_k2000.json
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from common import Matrix, load_panel  # noqa: E402
from utils.per_uid import parse_scp, parse_text  # noqa: E402

SET_TASK = {"clean": "clean", "itw": "in-the-wild"}


def export(panel_path, out_task=None, overwrite=False, link=True, allow_missing=False):
    panel = load_panel(panel_path)
    M = Matrix(panel["set"])
    src_task = SET_TASK[panel["set"]]
    out_task = out_task or f"{src_task}_{panel['kind']}"
    out_root = os.path.join(REPO_ROOT, "data", out_task)

    if os.path.exists(out_root):
        if not overwrite:
            sys.exit(f"error: {out_root} exists (pass --overwrite to replace)")
        shutil.rmtree(out_root)

    manifest = {"source_panel": os.path.relpath(panel_path, REPO_ROOT),
                "set": panel["set"], "kind": panel["kind"],
                "selector": panel["selector"], "k_requested": panel["k_requested"],
                "source_task": src_task,
                "seed": panel.get("seed"), "alloc": panel.get("alloc"),
                "langs": {}}

    n_items = 0
    missing = []  # (lang, uid, what) for every panel item that cannot be fully exported
    for lang, sel in sorted(panel["langs"].items()):
        idx = np.asarray(sel["idx"], dtype=int)
        w = np.asarray(sel["w"], dtype=float)
        # Emit in the source manifest's own order so the panel reads like a subset of the
        # original file rather than a reshuffle.
        order = np.argsort(idx, kind="stable")
        idx, w = idx[order], w[order]

        src = os.path.join(REPO_ROOT, "data", src_task, lang)
        texts = parse_text(os.path.join(src, "text"))
        ptexts = parse_text(os.path.join(src, "prompt_text"))
        pwavs = parse_scp(os.path.join(src, "prompt_wav.scp"))

        dst = os.path.join(out_root, lang)
        os.makedirs(os.path.join(dst, "waveform"), exist_ok=True)

        rows = []
        for j, wj in zip(idx, w):
            uid = M.uid[j]
            # A panel item with no prompt wav or no text cannot be exported as a working
            # benchmark item. Record it loudly instead of silently dropping it (which
            # would redistribute its stratum's weight) or writing an empty-text line
            # (which a TTS engine would "synthesize" as silence with no error).
            src_wav = pwavs.get(uid, "")
            if not src_wav:
                missing.append((lang, uid, "prompt_wav"))
                continue
            if not texts.get(uid, "").strip():
                missing.append((lang, uid, "text"))
                continue
            if not ptexts.get(uid, "").strip():
                missing.append((lang, uid, "prompt_text"))
                continue
            abs_wav = src_wav if os.path.isabs(src_wav) else os.path.join(REPO_ROOT, src_wav)
            abs_wav = os.path.realpath(abs_wav)
            base = os.path.basename(abs_wav)
            dst_wav = os.path.join(dst, "waveform", base)
            if not os.path.lexists(dst_wav):
                if link:
                    # RELATIVE symlink: an absolute one bakes in this checkout's path and
                    # breaks the moment the panel is cloned or moved, which for a
                    # published benchmark subset is the normal case.
                    os.symlink(os.path.relpath(abs_wav, os.path.dirname(dst_wav)), dst_wav)
                else:
                    shutil.copy2(abs_wav, dst_wav)
            rows.append((uid, texts.get(uid, ""), ptexts.get(uid, ""),
                         os.path.join("data", out_task, lang, "waveform", base), float(wj)))

        with open(os.path.join(dst, "text"), "w", encoding="utf-8") as fh:
            for uid, t, _p, _w, _wt in rows:
                fh.write(f"{uid} {t}\n")
        with open(os.path.join(dst, "prompt_text"), "w", encoding="utf-8") as fh:
            for uid, _t, p, _w, _wt in rows:
                fh.write(f"{uid} {p}\n")
        with open(os.path.join(dst, "prompt_wav.scp"), "w", encoding="utf-8") as fh:
            for uid, _t, _p, wav, _wt in rows:
                fh.write(f"{uid} {wav}\n")
        # ITW: held-out target recordings (target_wav.scp), same subset of uids.
        src_target = os.path.join(src, "target_wav.scp")
        if os.path.isfile(src_target):
            twavs = parse_scp(src_target)
            with open(os.path.join(dst, "target_wav.scp"), "w", encoding="utf-8") as fh:
                for uid, *_ in rows:
                    if uid in twavs:
                        fh.write(f"{uid} {twavs[uid]}\n")

        lang_w = {r[0]: r[4] for r in rows}
        if rows and len(rows) < idx.size:
            # Items were dropped (--allow-missing): renormalize this language's weights
            # over what was actually exported so they still sum to 1 and the manifest
            # matches the shipped files rather than the original selection.
            w_sum = sum(lang_w.values())
            lang_w = {u: wt / w_sum for u, wt in lang_w.items()}
        manifest["langs"][lang] = lang_w
        n_items += len(rows)

    if missing:
        for lang, uid, what in missing[:20]:
            print(f"  MISSING {what}: {lang}/{uid}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        if not allow_missing:
            shutil.rmtree(out_root)
            sys.exit(f"error: {len(missing)} panel item(s) lack a prompt wav or text in "
                     f"data/{src_task}/ — the exported panel would silently misweight "
                     f"its strata. Fix the source data, or pass --allow-missing to "
                     f"export without them (weights are then renormalized).")
        print(f"WARNING: exported without {len(missing)} missing item(s); per-language "
              f"weights renormalized and k_actual recomputed.", file=sys.stderr)

    # k_actual reflects what was exported, not what the panel selected.
    manifest["k_actual"] = n_items
    with open(os.path.join(out_root, "panel.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    print(f"exported {n_items} items across {len(manifest['langs'])} languages "
          f"-> data/{out_task}/")
    print(f"  run it with:  --task {out_task}")
    return out_root, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel", help="panel JSON from panel_select.py")
    ap.add_argument("--task", help="output task name (default <source_task>_<kind>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy", action="store_true",
                    help="copy prompt wavs instead of symlinking (for a portable release)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="export even if some panel items lack a prompt wav or text "
                         "(dropped items are reported; weights renormalized)")
    args = ap.parse_args()
    export(args.panel, out_task=args.task, overwrite=args.overwrite, link=not args.copy,
           allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
