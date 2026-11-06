#!/usr/bin/env python3
"""Classify the *nature* of the ground-truth WER edits: how much of the tail is
normalization (numbers written as digits vs words, diacritics / zero-width joiners,
CJK/Thai word-segmentation, loanword transliteration) versus a genuine ASR mishearing.

For every utterance we re-tokenize ref & hyp with run_wer.py's own `prepare()` (words,
or CER characters for zh/ja/ko/th), align them with jiwer, and bucket each individual
substitution / insertion / deletion. Reported per set, and per language.

    python scripts/analysis/classify_edits.py results/gt_clean            # clean
    python scripts/analysis/classify_edits.py results/gt_in-the-wild   # in-the-wild
"""
import glob
import os
import sys
import unicodedata
from collections import defaultdict

from jiwer import process_words  # noqa: E402

# Tokenize with the scorer's OWN prepare() (incl. normalize_ortho for ar/fa/he and
# the zh trad->simplified fold) so edit boundaries match the scored WER exactly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.text_norm import CHAR_LANGS, base_lang, prepare  # noqa: E402


ZW = {"‌", "‍", "​", "⁠"}     # ZWNJ, ZWJ, ZWSP, word-joiner


def is_num(t):
    return t != "" and all(unicodedata.category(c) == "Nd" for c in t)


def has_digit(t):
    return any(unicodedata.category(c) == "Nd" for c in t)


def ortho(t):
    """Strip combining marks + zero-width joiners; NFKD-fold. Equal orthography ->
    same base letters (diacritic / ZWNJ / full-vs-half-width difference only)."""
    t = "".join(c for c in t if c not in ZW)
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def lev(a, b, cap=3):
    """Bounded char-level Levenshtein; returns min(distance, cap)."""
    if abs(len(a) - len(b)) > cap:
        return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            best = min(best, cur[-1])
        if best >= cap:
            return cap
        prev = cur
    return min(prev[-1], cap)


def classify_sub(r, h):
    if is_num(r) and is_num(h):
        return "number: digit vs digit (misheard)"          # 1945 vs 1975 -> real
    if is_num(r) != is_num(h) and (has_digit(r) or has_digit(h)):
        return "number: digit vs word (normalization)"      # 70 vs siebzig -> norm
    if ortho(r) == ortho(h):
        return "orthographic: diacritic / ZWNJ / width (normalization)"
    if lev(r, h) <= 2 and min(len(r), len(h)) >= 3:
        return "minor: <=2-char spelling / inflection"
    return "content: different word (mishearing)"


def classify_indel(tok, kind, neighbors):
    if is_num(tok):
        return f"number {kind} (digit token)"
    if tok in neighbors:
        return f"{kind}: repeated adjacent token"
    return f"content {kind}"


def run(root):
    cats = defaultdict(int)
    per_lang = defaultdict(lambda: defaultdict(int))
    lang_edits = defaultdict(int)
    seg_edits = defaultdict(int)          # CJK/Thai edits that vanish if spaces ignored
    tot = 0
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        leaf = os.path.basename(d.rstrip("/"))
        base = base_lang(leaf)
        raw = os.path.join(d, "wav_res_ref_text.wer.raw")
        if not os.path.isfile(raw):
            continue
        for line in open(raw):
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            # Rows the reference ASR never transcribed carry an <UNSCOREABLE:Reason>
            # sentinel instead of a hypothesis and a synthetic all-deletions edit
            # decomposition (utils/run_wer.py). They are a real maximum-error score
            # but there is no transcript to classify, so they are not edits.
            if p[3].startswith("<UNSCOREABLE"):
                continue
            rt = prepare(p[2], base).split()
            ht = prepare(p[3], base).split()
            if rt == ht:
                continue
            # segmentation check (CJK/Thai): identical once inter-token spaces drop
            if base in CHAR_LANGS and "".join(rt) == "".join(ht):
                n = process_words(" ".join(rt), " ".join(ht))
                e = n.substitutions + n.insertions + n.deletions
                cats["segmentation: CJK/Thai word-boundary only (normalization)"] += e
                per_lang[leaf]["segmentation: CJK/Thai word-boundary only (normalization)"] += e
                lang_edits[leaf] += e
                seg_edits[leaf] += e
                tot += e
                continue
            out = process_words(" ".join(rt), " ".join(ht))
            for chunk in out.alignments[0]:
                if chunk.type == "equal":
                    continue
                if chunk.type == "substitute":
                    for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                        r = rt[chunk.ref_start_idx + k]
                        h = ht[chunk.hyp_start_idx + k]
                        c = classify_sub(r, h)
                        cats[c] += 1; per_lang[leaf][c] += 1; lang_edits[leaf] += 1; tot += 1
                elif chunk.type == "insert":
                    for k in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        nb = {ht[k - 1] if k > 0 else "", ht[k + 1] if k + 1 < len(ht) else ""}
                        c = classify_indel(ht[k], "insertion", nb)
                        cats[c] += 1; per_lang[leaf][c] += 1; lang_edits[leaf] += 1; tot += 1
                elif chunk.type == "delete":
                    for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
                        nb = {rt[k - 1] if k > 0 else "", rt[k + 1] if k + 1 < len(rt) else ""}
                        c = classify_indel(rt[k], "deletion", nb)
                        cats[c] += 1; per_lang[leaf][c] += 1; lang_edits[leaf] += 1; tot += 1
    return cats, per_lang, lang_edits, tot


NORM_KEYS = (
    "number: digit vs word (normalization)",
    "orthographic: diacritic / ZWNJ / width (normalization)",
    "segmentation: CJK/Thai word-boundary only (normalization)",
)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/gt_clean"
    cats, per_lang, lang_edits, tot = run(root)
    if tot == 0:
        print(f"\n===== {root}   (0 total edits) =====")
        print("no edits found (no .wer.raw files, or all utterances match "
              "after normalization)")
        return
    print(f"\n===== {root}   ({tot:,} total edits) =====")
    norm = sum(cats[k] for k in NORM_KEYS)
    print(f"clearly normalization-driven: {norm:,}  ({100*norm/tot:.1f}%)   "
          f"[numbers + diacritics/ZWNJ + CJK segmentation]")
    print(f"{'edit category':<58}{'count':>9}{'share':>8}")
    for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<56}{n:>9,}{100*n/tot:>7.1f}%")
    # languages whose tail is most normalization-inflated
    print("\n  per-language normalization share (langs with >=200 edits, top 12):")
    rows = []
    for lg, d in per_lang.items():
        e = lang_edits[lg]
        if e < 200:
            continue
        nrm = sum(d.get(k, 0) for k in NORM_KEYS)
        rows.append((100 * nrm / e, lg, e, nrm))
    for share, lg, e, nrm in sorted(rows, reverse=True)[:12]:
        print(f"    {lg:<5} {share:5.1f}% normalization   ({nrm:,}/{e:,} edits)")


if __name__ == "__main__":
    main()
