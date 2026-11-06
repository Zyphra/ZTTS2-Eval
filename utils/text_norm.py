"""Shared text preparation for WER scoring, used by run_wer.py (and offline
re-scorers). Kept side-effect-free so it can be imported anywhere.

Two stages, both applied IDENTICALLY to reference and hypothesis so nothing here
can create a spurious win:

1. `normalize_ortho` — script-level orthographic folding for abjad languages,
   where a large slice of the ground-truth WER tail is spelling convention, not a
   mispronunciation (see docs/edit_nature.md). All folds are pronunciation-neutral:
     ar  strip harakat/tanwin/shadda/sukun/tatweel; hamza-carriers -> bare letter
         (أ إ آ ٱ -> ا, ؤ -> و, ئ -> ي, ء dropped); ة -> ه; alef-maqsura ى -> ي.
     fa  same strip; Arabic<->Persian unification ي -> ی, ك -> ک; ة -> ه; ى -> ی;
         and remove the zero-width non-joiner (آن‌ها == آنها).
     he  strip niqqud/cantillation; final forms -> base (ך ם ן ף ץ). Hebrew's tail
         is mostly matres lectionis (optional vav/yod), which we deliberately do
         NOT fold: that would change consonantal spelling and could hide a real
         ASR error, violating the pronunciation-neutral rule.
   Gated by ORTHO_NORM_LANGS (default "ar fa he"); set empty to disable.

2. `prepare` — digits -> words (num2words, see `expand_numbers`), apostrophe deletion,
   Unicode punctuation -> space, lowercase, and
   tokenization: whitespace words, or single characters (CER) for zh/ja/ko/th.
   zh additionally folds traditional -> simplified (zhconv) on BOTH sides: a
   handful of data/*/zh/text lines are traditional, and folding only the ASR
   hypothesis (the old run_wer.py behaviour) made those utterances score ~100%
   CER regardless of the audio.
"""
import difflib
import os
import re
import sys
import unicodedata

import num2words as _n2w
import zhconv
from jiwer import process_words

# Every benchmark language code (mirrors utils/run_wer.py's WHISPER_LANG_CODE
# keys, incl. the 'fil' alias for tl). base_lang() only recognizes codes listed
# here, so a language missing from this set silently disables Whisper routing,
# the abjad ortho fold and CER tokenization for compound leaf names.
KNOWN_LANGS = {
    "en", "zh", "de", "es", "fr", "it", "ar",
    "ja", "ko", "ru", "hi", "id", "th", "tl", "tr", "pl", "pt",
    # 2026-07 new-language expansion (all live under results/gt_clean)
    "bg", "ca", "cs", "da", "el", "fa", "fi", "fil", "he", "hr", "hu",
    "ms", "nb", "nl", "ro", "sk", "sv", "uk", "vi", "yo",
}
CHAR_LANGS = ("zh", "ja", "ko", "th")            # no inter-word spaces -> score as CER
_CJK_TOKEN_RE = re.compile(r"[A-Za-z]+|\S")
# apostrophe glyphs: ' U+2018 U+2019 U+02BC U+02B9 U+2032 backtick
_APOS_RE = re.compile("['‘’ʼʹ′`]")

# --- orthographic normalization (abjad scripts) ---------------------------------
# default when unset; ORTHO_NORM_LANGS="" (empty) genuinely disables all folding
ORTHO_NORM_LANGS = set(os.environ.get("ORTHO_NORM_LANGS", "ar fa he").split())

# zero-width joiners / bidi controls / soft hyphen / BOM (never pronounced)
_ZW = {c: None for c in (0x200C, 0x200D, 0x200E, 0x200F, 0x00AD, 0x2060, 0xFEFF)}
# Arabic combining marks: harakat/tanwin U+064B-064F, shadda/sukun U+0650-0652,
# maddah/hamza-above/below U+0653-0655, superscript alef U+0670, tatweel U+0640
_AR_STRIP = {c: None for c in range(0x064B, 0x0656)}
_AR_STRIP.update({0x0670: None, 0x0640: None})


def _t(mapping):
    return {ord(k): v for k, v in mapping.items()}


_AR_FOLD = _t({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
               "ى": "ي", "ئ": "ي", "ؤ": "و", "ء": None, "ة": "ه"})
_FA_FOLD = _t({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
               "ي": "ی", "ك": "ک", "ى": "ی", "ئ": "ی", "ؤ": "و", "ء": None, "ة": "ه"})
_HE_STRIP = {c: None for c in range(0x0591, 0x05C8)
             if unicodedata.category(chr(c)) == "Mn"}
_HE_FOLD = _t({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})

_TABLES = {
    "ar": {**_AR_STRIP, **_ZW, **_AR_FOLD},
    "fa": {**_AR_STRIP, **_ZW, **_FA_FOLD},
    "he": {**_HE_STRIP, **_ZW, **_HE_FOLD},
}


def normalize_ortho(text, base):
    """Pronunciation-neutral orthographic folding for abjad langs; identity else."""
    if base not in ORTHO_NORM_LANGS:
        return text
    table = _TABLES.get(base)
    return text.translate(table) if table else text


def base_lang(leaf_name):
    """Map a result leaf-dir name (`en`, `hard_zh`, ...) to its base ISO code:
    the first underscore token that is a known language code."""
    for t in leaf_name.split("_"):
        if t in KNOWN_LANGS:
            return t
    fallback = leaf_name.split("_")[0]
    # An unrecognized code silently disables Whisper routing, the abjad fold and
    # CER tokenization -- warn loudly so a new language gets added to KNOWN_LANGS.
    print(f"[text_norm] WARNING: no known language code in leaf '{leaf_name}'; "
          f"falling back to base lang '{fallback}'", file=sys.stderr)
    return fallback


# --- number expansion ----------------------------------------------------------------
# ASR output and references disagree on digits vs words ("20" vs "twenty") and that
# disagreement is not a pronunciation error, so both sides are expanded identically
# before scoring. Languages num2words cannot handle are left in digits.
_N2W_CODE = {"nb": "no"}
NUM2WORDS_LANGS = frozenset(
    l for l in KNOWN_LANGS if _N2W_CODE.get(l, l) in _n2w.CONVERTER_CLASSES)
_NUM_RE = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)*)(st|nd|rd|th)?(?![\w])")
_SEP_RE = re.compile(r"[.,]")


def _say(value, code, **kw):
    return " " + _n2w.num2words(value, lang=code, **kw) + " "


def expand_numbers(text, base):
    """Replace digit tokens by words; identity for languages without num2words support."""
    if base not in NUM2WORDS_LANGS:
        return text
    code = _N2W_CODE.get(base, base)

    def repl(m):
        tok, suffix = m.group(1), m.group(2)
        if suffix and base != "en":
            return m.group(0)
        try:
            groups = _SEP_RE.split(tok)
            if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]) \
                    and not groups[0].startswith("0"):
                tok = "".join(groups)               # 1,000 / 1.000.000 -> integer
            elif len(groups) == 2:
                return _say(f"{groups[0]}.{groups[1]}", code)   # decimal
            elif len(groups) > 2:
                return " ".join(_say(int(g), code) for g in groups)
            n = int(tok)
            if base == "en" and suffix:
                return _say(n, code, to="ordinal")
            if base == "en" and 1100 <= n <= 2099 and len(tok) == 4:
                return _say(n, code, to="year")
            return _say(n, code)
        except Exception:  # num2words: NotImplementedError/OverflowError/... -> keep digits
            return m.group(0)

    return _NUM_RE.sub(repl, text)


def text_key(text):
    """Case/punctuation-insensitive key for matching the same sentence."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def near_duplicate(a, b, threshold=0.8):
    """True when two texts are the same sentence up to case/punctuation, or their
    text_keys are >= `threshold` similar (difflib ratio)."""
    ka, kb = text_key(a), text_key(b)
    return ka == kb or difflib.SequenceMatcher(None, ka, kb).ratio() >= threshold


def prepare(text, base):
    """Return a space-joined token string ready for jiwer.process_words(ref, hyp)."""
    if base == "zh":
        # Traditional -> simplified, applied identically to ref and hyp (folding
        # only the hypothesis turned traditional-script references into ~100% CER).
        text = zhconv.convert(text, "zh-cn")
    text = normalize_ortho(text, base)
    text = expand_numbers(text, base)
    text = _APOS_RE.sub("", text)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    if base in CHAR_LANGS:
        toks = [t.lower() for t in _CJK_TOKEN_RE.findall(text)]
    else:
        toks = text.lower().split()
    return " ".join(toks)


def process_one(hypo, truth, base):
    """(raw_truth, raw_hypo, wer, sub_frac, del_frac, ins_frac) for one pair.
    Fractions are edit-count / reference-token-count, matching the .wer.raw format."""
    raw_truth, raw_hypo = truth, hypo
    truth = prepare(truth, base)
    hypo = prepare(hypo, base)
    if not truth:
        # degenerate reference (all punctuation/empty after prep) -- avoid div-by-zero
        nhyp = len(hypo.split())
        return (raw_truth, raw_hypo, float(nhyp > 0), 0.0, 0.0, float(nhyp > 0))
    measures = process_words(truth, hypo)
    n = max(len(truth.split(" ")), 1)
    return (raw_truth, raw_hypo, measures.wer,
            measures.substitutions / n, measures.deletions / n, measures.insertions / n)
