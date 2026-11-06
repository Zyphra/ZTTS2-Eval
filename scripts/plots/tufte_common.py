# tufte_common.py  --  shared data loading + palette for the tufte_* plots.
#
# The tufte_* scripts are reliability / survival-tail plots. Originally they
# compared a fixed set of 9 TTS models across 6 word-languages, reading
# one merged `mega_merge_wav_res_ref_text_wer.raw`. This module makes them
# generic: point them at any set of per-leaf `wav_res_ref_text.wer.raw` files via
# a --glob and pick what a "series" is via --label (language, set/backend, model,
# a directory level, or a regex). No hardcoded model map, no WORDLANG filter.
#
# Each .raw line is:
#     wav \t wer \t ref \t hyp \t ins_frac \t del_frac \t sub_frac
# where the *_frac are (edit_count / n_ref_tokens) with the SAME tokenization
# run_wer.py applied: whitespace words, or single characters for zh/ja/ko/th
# (scored as CER). We reproduce that tokenization here so the reconstructed
# integer edit counts are exact -- which is what lets every language be plotted,
# not just the space-delimited ones.
#
# NOTE on pooling: within one language the edit unit is consistent (words, or
# characters for CJK/Thai). If a *series* pools multiple languages of different
# unit types (e.g. label=set over a whole GT tree), its edit counts mix word- and
# character-edits. That is fine for a per-language series (the default) and for a
# per-backend series scored on the same audio; be wary of it only when a single
# series deliberately spans both kinds of language.
#
# Requires: numpy, matplotlib  (no scipy)

import argparse
import glob as _glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ---- tokenization: the scorer's own (utils/text_norm), not a copy -----------
# A local "copied to match exactly" replica of run_wer.py's tokenizer drifted
# (it missed normalize_ortho, the zh variant fold, and lowercasing). Import the
# canonical implementation instead so reconstructed edit counts stay exact.
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from utils.text_norm import CHAR_LANGS, base_lang, prepare  # noqa: E402,F401
from utils.per_uid import parse_wer  # noqa: E402


def ref_token_count(text, base):
    """Number of reference tokens run_wer.py would have used for this text+lang.
    Multiplying an ins/del/sub fraction by this recovers the integer edit count."""
    return max(len(prepare(text, base).split()), 1)


# ---- palette (unchanged Tufte look) ---------------------------------------------
PAPER, INK, MUT = "#FBF9F4", "#2A2622", "#B4ABA0"
GREY, ENS, MID = "#CFC8BC", "#DDD7CB", "#8A8074"
RED, BLUE, GOLD = "#A5321E", "#2C5E8A", "#C98A16"
GREEN, PURPLE, TEAL = "#3F7A46", "#6A4C93", "#2E7D7B"
HL_PALETTE = [BLUE, RED, GOLD, GREEN, PURPLE, TEAL]


# ---- series resolution ----------------------------------------------------------
def _up(path, n):
    """basename of the directory n levels above `path` (n=1 -> the file's dir)."""
    d = os.path.abspath(path)
    for _ in range(n):
        d = os.path.dirname(d)
    return os.path.basename(d)


def resolve_label(path, scheme, label_re):
    """Series label for a .raw file. Schemes:
        lang   -> the leaf dir (the language)         [1 level up]
        set    -> the set / backend dir               [2 levels up]
        model  -> the model dir                       [3 levels up]
        dir:N  -> basename N levels above the file
        re     -> --label-re against the full path; group('series') or group(1)"""
    if scheme == "lang":
        return _up(path, 1)
    if scheme == "set":
        return _up(path, 2)
    if scheme == "model":
        return _up(path, 3)
    if scheme.startswith("dir:"):
        return _up(path, int(scheme.split(":", 1)[1]))
    if scheme == "re":
        if not label_re:
            raise SystemExit("--label re requires --label-re PATTERN")
        m = re.search(label_re, path)
        if not m:
            raise SystemExit(f"--label-re did not match {path}")
        gd = m.groupdict()
        return gd["series"] if "series" in gd else m.group(1)
    raise SystemExit(f"unknown --label scheme: {scheme}")


def _parse_rename(spec):
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, _, v = pair.partition("=")
        out[k.strip()] = v.strip()
    return out


def load_series(globs, label="lang", label_re=None, rename=None, limit=0):
    """Return (E, meta):
        E    : dict series_label -> np.ndarray of per-utterance integer edit counts
        meta : dict series_label -> {"n": int, "langs": sorted list of leaf langs}
    If `limit` > 0, keep only the worst-`limit` series by 99th-percentile edits."""
    rn = _parse_rename(rename)
    files = []
    for g in globs:
        m = _glob.glob(g)                        # as given (cwd-relative or absolute)
        if not m and not os.path.isabs(g):       # fall back to repo-root-relative
            m = _glob.glob(os.path.join(REPO, g))
        files.extend(sorted(m))
    if not files:
        raise SystemExit(f"no .raw files matched: {globs}")

    edits = defaultdict(list)
    langs = defaultdict(set)
    for path in files:
        leaf = _up(path, 1)
        base = base_lang(leaf)
        label_val = rn.get(lbl := resolve_label(path, label, label_re), lbl)
        for d in parse_wer(path).values():
            if d["sub"] is None:
                continue
            n = ref_token_count(d["ref_text"], base)
            e = round(d["sub"] * n) + round(d["del"] * n) + round(d["ins"] * n)
            edits[label_val].append(e)
            langs[label_val].add(leaf)
    E = {m: np.asarray(v, dtype=int) for m, v in edits.items()}
    meta = {m: {"n": len(E[m]), "langs": sorted(langs[m])} for m in E}

    if limit and len(E) > limit:
        keep = sorted(E, key=lambda m: np.percentile(E[m], 99), reverse=True)[:limit]
        E = {m: E[m] for m in keep}
        meta = {m: meta[m] for m in keep}
    return E, meta


def auto_highlight(E, k=3):
    """Pick up to `k` series to emphasize: best (lowest p99), worst (highest p99),
    and the most catastrophic single utterance. Returns [(label, color), ...]."""
    p99 = {m: np.percentile(e, 99) for m, e in E.items()}
    worst_max = {m: int(e.max()) for m, e in E.items()}
    picks, roles = [], [
        (min(p99, key=p99.get), BLUE),          # most reliable
        (max(p99, key=p99.get), RED),            # least reliable
        (max(worst_max, key=worst_max.get), GOLD),  # worst blow-up
    ]
    for lab, col in roles[:k]:
        if lab not in [p[0] for p in picks]:
            picks.append((lab, col))
    return picks


def resolve_highlight(E, spec, k=3):
    """User --highlight 'a,b,c' -> colored list; else auto_highlight."""
    if not spec:
        return auto_highlight(E, k)
    labs = [s.strip() for s in spec.split(",") if s.strip() and s.strip() in E]
    return [(lab, HL_PALETTE[i % len(HL_PALETTE)]) for i, lab in enumerate(labs)]


# ---- survival helper ------------------------------------------------------------
def survxy(e):
    """P(edits >= k) as a percentage, for k = 1 .. max, dropping zero-rate tail."""
    ks = np.arange(1, int(e.max()) + 1) if e.size and e.max() >= 1 else np.array([1])
    s = np.array([(e >= k).mean() * 100 for k in ks])
    keep = s > 0
    return ks[keep], s[keep]


# ---- argparse ------------------------------------------------------------------
def add_common_args(ap, default_glob, default_out):
    ap.add_argument("globs", nargs="*", default=[default_glob],
                    help=f"glob(s) for *.wer.raw files (default: {default_glob})")
    ap.add_argument("--label", default="lang",
                    help="series label scheme: lang|set|model|dir:N|re (default: lang)")
    ap.add_argument("--label-re", default=None, help="regex for --label re (group 'series')")
    ap.add_argument("--rename", default=None,
                    help="comma list old=new applied to series labels, e.g. "
                         "'gt_clean=Qwen routed,whisper_clean=Whisper'")
    ap.add_argument("--highlight", default=None,
                    help="comma list of series labels to emphasize (default: auto best/worst)")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the worst-N series by p99 (0 = all)")
    ap.add_argument("--small-n", type=int, default=300,
                    help="series with fewer utterances are flagged/greyed (default: 300)")
    ap.add_argument("--title", default=None, help="override the headline")
    ap.add_argument("--subtitle", default=None, help="override the sub-headline")
    ap.add_argument("-o", "--out", default=default_out, help=f"output PNG (default: {default_out})")


def default_subtitle(E, meta, extra="exact edit distance from wer.raw"):
    total = sum(m["n"] for m in meta.values())
    langs = sorted({l for m in meta.values() for l in m["langs"]})
    nl = len(langs)
    return (f"{len(E)} series · {nl} language{'s' if nl != 1 else ''} · "
            f"{total:,} utterances · {extra}")
