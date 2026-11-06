"""Shared parsers for the repo's uttid-keyed line files (scp / text / prompt_text).

These consolidate helpers that were previously copy-pasted across the
generation clients, scoring drivers, and data builders. All inputs follow the
Kaldi-style convention: one record per line, ``<uttid> <value>``.
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(p):
    """Resolve an scp wav path against the current checkout.

    Committed scp files store repo-relative paths. Absolute paths written on
    another checkout (``/old/checkout/data/...``) dangle elsewhere, so if ``p``
    doesn't exist as given, retry relative to the repo root; for absolute paths
    also retry each path suffix re-anchored at the repo root. Values that
    resolve nowhere are returned unchanged.
    """
    if os.path.exists(p):
        return p
    if not os.path.isabs(p):
        cand = os.path.join(_REPO_ROOT, p)
        return cand if os.path.exists(cand) else p
    parts = p.split(os.sep)
    for i in range(1, len(parts)):
        cand = os.path.join(_REPO_ROOT, *parts[i:])
        if os.path.exists(cand):
            return cand
    return p


def load_scp(path, resolve=True, missing_ok=False):
    """Parse a ``<uttid> <value>`` file into a ``{uttid: value}`` dict.

    Whitespace-collapsing: the first whitespace run separates the id from the
    value (``str.split(maxsplit=1)``). Blank lines are skipped. Values get the
    doubled slash that ``data/clean/`` prompt paths carry collapsed
    (``data/clean//en/...`` -> ``data/clean/en/...``) and, when ``resolve``, are
    re-anchored on this checkout (``resolve_path``). A missing file returns ``{}``
    when ``missing_ok``, else raises.
    """
    if missing_ok and not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, val = line.split(maxsplit=1)
            val = val.replace("//", "/")
            out[uid] = resolve_path(val) if resolve else val
    return out


def load_scp_items(path):
    """Parse a ``<uttid> <value>`` file into an ordered list of ``(uttid, value)``.

    Tolerates either a tab or a space as the first separator (tab preferred when
    present), preserving insertion order. Used by the scoring drivers and the
    voice-dir builders where order matters and duplicate ids must not collapse.
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sep = "\t" if "\t" in line else " "
            uid, val = line.split(sep, 1)
            items.append((uid, resolve_path(val)))
    return items


def load_text(path, missing_ok=False):
    """Parse a ``<uttid> <text>`` file into a ``{uttid: text}`` dict.

    Splits on the first space only (``str.partition(" ")``), so the text value
    keeps its internal spacing verbatim. A missing file returns ``{}`` when
    ``missing_ok``, else raises.
    """
    if missing_ok and not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, _, txt = line.partition(" ")
            out[uid] = txt
    return out


def load_msr_utmos_cache(path):
    """``{(spk, vid, seg, start, end): score}`` from the VoxBlink2 MSR-UTMOS clip cache
    (``data/in-the-wild/_msr_utmos_cache.tsv``); ``{}`` if the file is missing. Rows
    without exactly six tab fields are skipped."""
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 6:
                continue
            cache[(p[0], p[1], p[2], float(p[3]), float(p[4]))] = float(p[5])
    return cache
