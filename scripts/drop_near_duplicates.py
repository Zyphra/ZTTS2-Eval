#!/usr/bin/env python3
"""Remove prompt/target near-duplicates from the benchmark data (idempotent, no args).

A pair is a near-duplicate when the prompt transcript and the target text are the same
sentence up to case/punctuation, or their `text_key`s are >= 0.8 similar
(utils/text_norm.near_duplicate). Such pairs let a system win by copying the prompt, so
they are dropped from both full sets and from every coreset panel built on them.

For each of data/clean and data/in-the-wild, per language:
  * text, prompt_text, prompt_wav.scp are rewritten without the offending uids
    (original line bytes and order preserved) and their local prompt wavs deleted;
  * every panel dir (data/clean_{mini,delta}, data/in-the-wild_{mini,delta})
    whose panel.json["source_task"] matches loses the same uids from text, prompt_text,
    prompt_wav.scp, target_wav.scp (if present) and waveform/ symlinks, its per-language
    weights are renormalised to sum 1 and k_actual is decremented.

data/near_duplicates_removed.json records what was removed (merged across reruns).
"""
import difflib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.text_norm import near_duplicate, text_key  # noqa: E402

DATA = os.path.join(REPO, "data")
TASKS = ("clean", "in-the-wild")
PANELS = ("clean_mini", "clean_delta", "in-the-wild_mini",
          "in-the-wild_delta")
CRITERION = "text_key equal or SequenceMatcher ratio >= 0.8 (utils/text_norm.near_duplicate)"


def read_lines(path):
    with open(path, "rb") as f:
        return f.readlines()


def uid_of(line):
    parts = line.split(None, 1)
    return parts[0].decode("utf-8") if parts else None


def as_dict(path):
    d = {}
    for ln in read_lines(path):
        u = uid_of(ln)
        if u is not None:
            d[u] = ln.split(None, 1)[1].decode("utf-8").strip() if len(ln.split(None, 1)) > 1 else ""
    return d


def drop_uids(path, uids):
    """Rewrite `path` without the lines whose uid is in `uids`; returns #removed."""
    if not os.path.isfile(path):
        return 0
    lines = read_lines(path)
    keep = [ln for ln in lines if uid_of(ln) not in uids]
    if len(keep) != len(lines):
        with open(path, "wb") as f:
            f.writelines(keep)
    return len(lines) - len(keep)


def find_duplicates(task):
    """{lang: [(uid, prompt_wav, ratio)]} for one full set."""
    out = {}
    root = os.path.join(DATA, task)
    for lang in sorted(os.listdir(root)):
        tp = os.path.join(root, lang, "text")
        if not os.path.isfile(tp):
            continue
        text = as_dict(tp)
        ptext = as_dict(os.path.join(root, lang, "prompt_text"))
        pwav = as_dict(os.path.join(root, lang, "prompt_wav.scp"))
        hits = []
        for uid, t in text.items():
            if uid in ptext and near_duplicate(ptext[uid], t):
                ratio = difflib.SequenceMatcher(None, text_key(ptext[uid]), text_key(t)).ratio()
                hits.append((uid, pwav.get(uid, ""), round(ratio, 4)))
        if hits:
            out[lang] = hits
    return out


def drop_from_full(task, dups):
    root = os.path.join(DATA, task)
    for lang, hits in dups.items():
        uids = {u for u, _, _ in hits}
        for name in ("text", "prompt_text", "prompt_wav.scp"):
            drop_uids(os.path.join(root, lang, name), uids)
        for _, wav, _ in hits:
            p = os.path.join(REPO, wav)
            if wav and (os.path.lexists(p)):
                os.remove(p)


def drop_from_panel(panel, dups):
    """Remove the duplicates of the panel's source task; returns #uids dropped."""
    pdir = os.path.join(DATA, panel)
    pj = os.path.join(pdir, "panel.json")
    if not os.path.isfile(pj):
        return 0
    with open(pj) as f:
        meta = json.load(f)
    removed = 0
    for lang, hits in dups.items():
        ldir = os.path.join(pdir, lang)
        if lang not in meta["langs"] or not os.path.isdir(ldir):
            continue
        in_panel = {u for u, _, _ in hits if u in meta["langs"][lang]}
        if not in_panel:
            continue
        for name in ("text", "prompt_text", "prompt_wav.scp", "target_wav.scp"):
            drop_uids(os.path.join(ldir, name), in_panel)
        wf = os.path.join(ldir, "waveform")
        by_uid = {u: w for u, w, _ in hits}
        for uid in in_panel:
            base = os.path.basename(by_uid[uid]) if by_uid[uid] else ""
            p = os.path.join(wf, base)
            if base and os.path.lexists(p):
                os.remove(p)
            del meta["langs"][lang][uid]
        w = meta["langs"][lang]
        tot = sum(w.values())
        if tot > 0:
            meta["langs"][lang] = {u: v / tot for u, v in w.items()}
        meta["k_actual"] -= len(in_panel)
        removed += len(in_panel)
    if removed:
        with open(pj, "w") as f:
            json.dump(meta, f, indent=None, sort_keys=True)
            f.write("\n")
    return removed


def main():
    log_path = os.path.join(DATA, "near_duplicates_removed.json")
    log = {"criterion": CRITERION, "removed": {}}
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log = json.load(f)
        log["criterion"] = CRITERION
    for task in TASKS:
        dups = find_duplicates(task)
        n = sum(len(v) for v in dups.values())
        panel_n = {}
        for panel in PANELS:
            pj = os.path.join(DATA, panel, "panel.json")
            if os.path.isfile(pj):
                with open(pj) as f:
                    if json.load(f)["source_task"] == task:
                        panel_n[panel] = drop_from_panel(panel, dups)
        drop_from_full(task, dups)
        for lang, hits in dups.items():
            lst = log["removed"].setdefault(f"{task}/{lang}", [])
            seen = {e["uid"] for e in lst}
            lst.extend({"uid": u, "prompt_wav": w, "ratio": r} for u, w, r in hits
                       if u not in seen)
        print(f"{task}: removed {n}" + "".join(f"; {p}: {c}" for p, c in panel_n.items()))
        if dups and task == "in-the-wild":
            print("  per lang: " + " ".join(f"{l}={len(h)}" for l, h in dups.items()))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
