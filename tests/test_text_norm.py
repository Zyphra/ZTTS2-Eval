"""Unit tests for utils/text_norm.py -- the text prep every WER number flows through.

No GPU, no network (pure jiwer + zhconv + stdlib). Run from the repo root with:
    .venv/bin/python tests/test_text_norm.py

Pins: tokenization per script family (latin words, CJK/Thai chars), the zh
traditional->simplified fold (applied to BOTH sides), the hand-built ar/fa/he
orthographic fold tables, the ORTHO_NORM_LANGS gate (read once at import time),
process_one's degenerate-reference guard, and base_lang's compound-leaf parsing.
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

# Make `utils` importable regardless of how this file is launched.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import text_norm  # noqa: E402
from utils.text_norm import (  # noqa: E402
    CHAR_LANGS, KNOWN_LANGS, base_lang, normalize_ortho, prepare, process_one,
)


class PrepareLatinTest(unittest.TestCase):
    def test_lowercase_and_word_tokens(self):
        self.assertEqual(prepare("Hello World", "en"), "hello world")

    def test_apostrophes_deleted_not_spaced(self):
        # contractions are orthography, not pronunciation
        self.assertEqual(prepare("Don't stop", "en"), "dont stop")

    def test_unicode_apostrophe_variants_deleted(self):
        self.assertEqual(prepare("l’été", "fr"), "lété")

    def test_punctuation_becomes_space(self):
        self.assertEqual(prepare("hello, world!  (yes)", "en"), "hello world yes")


class PrepareCharLangTest(unittest.TestCase):
    def test_char_langs_tuple(self):
        self.assertEqual(set(CHAR_LANGS), {"zh", "ja", "ko", "th"})

    def test_zh_character_tokens(self):
        self.assertEqual(prepare("你好世界", "zh"),
                         "你 好 世 界")

    def test_latin_runs_stay_whole_inside_cjk(self):
        # _CJK_TOKEN_RE keeps [A-Za-z]+ runs as one (lowercased) token
        self.assertEqual(prepare("我用GPT写", "zh"),
                         "我 用 gpt 写")

    def test_thai_is_char_tokenized(self):
        text = "สวัสดี"  # 6 codepoints, no spaces
        self.assertEqual(prepare(text, "th").split(), list(text))

    def test_english_is_not_char_tokenized(self):
        self.assertEqual(prepare("hello world", "en"), "hello world")


class ZhVariantFoldTest(unittest.TestCase):
    """The traditional->simplified fold is applied inside prepare(), i.e. to the
    reference AND the hypothesis. Folding only the hypothesis (the old run_wer.py
    behavior) made a traditional-script reference score ~100% CER regardless of
    the audio (e.g. data/zero_shot_voxblink/zh uttid_400)."""

    TRAD = "這個節目"   # 這個節目
    SIMP = "这个节目"   # 这个节目

    def test_fold_is_symmetric(self):
        self.assertEqual(prepare(self.TRAD, "zh"), prepare(self.SIMP, "zh"))

    def test_traditional_ref_simplified_hyp_scores_zero(self):
        _, _, wer, sub, dele, ins = process_one(self.SIMP, self.TRAD, "zh")
        self.assertEqual((wer, sub, dele, ins), (0.0, 0.0, 0.0, 0.0))

    def test_fold_only_applies_to_zh(self):
        self.assertEqual(prepare(self.TRAD, "ja"),
                         " ".join(self.TRAD))  # ja char-tokenizes but never folds


class OrthoNormTest(unittest.TestCase):
    def test_ar_hamza_and_harakat(self):
        # أَحْمَد -> احمد (hamza-carrier fold + harakat strip)
        self.assertEqual(normalize_ortho("أَحْمَد", "ar"),
                         "احمد")

    def test_ar_ta_marbuta(self):
        # مدرسة -> مدرسه
        self.assertEqual(normalize_ortho("مدرسة", "ar"),
                         "مدرسه")

    def test_fa_zwnj_removed(self):
        # آن‌ها == آنها (docstring's own example)
        self.assertEqual(normalize_ortho("آن‌ها", "fa"),
                         normalize_ortho("آنها", "fa"))

    def test_fa_arabic_yeh_kaf_unified(self):
        # Arabic ي/ك -> Persian ی/ک
        self.assertEqual(normalize_ortho("يك", "fa"), "یک")

    def test_he_niqqud_stripped_and_final_forms_folded(self):
        # שָׁלוֹם -> שלומ (niqqud stripped, final mem -> base)
        self.assertEqual(
            normalize_ortho("שָׁלוֹם", "he"),
            "שלומ")

    def test_fold_is_idempotent(self):
        for base, text in (("ar", "أَحْمَد"),
                           ("fa", "آن‌ها"),
                           ("he", "שָׁלוֹם")):
            once = normalize_ortho(text, base)
            self.assertEqual(normalize_ortho(once, base), once)

    def test_identity_for_non_abjad_langs(self):
        self.assertEqual(normalize_ortho("hello", "en"), "hello")

    def test_gate_disables_folding(self):
        saved = text_norm.ORTHO_NORM_LANGS
        try:
            text_norm.ORTHO_NORM_LANGS = set()
            self.assertEqual(normalize_ortho("أ", "ar"), "أ")
        finally:
            text_norm.ORTHO_NORM_LANGS = saved

    def test_env_after_import_is_a_no_op(self):
        # ORTHO_NORM_LANGS is read ONCE at import; setting it later cannot change
        # behavior. This test documents that trap.
        import os
        saved = os.environ.get("ORTHO_NORM_LANGS")
        try:
            os.environ["ORTHO_NORM_LANGS"] = ""
            self.assertEqual(normalize_ortho("أ", "ar"), "ا")
        finally:
            if saved is None:
                os.environ.pop("ORTHO_NORM_LANGS", None)
            else:
                os.environ["ORTHO_NORM_LANGS"] = saved


class ProcessOneTest(unittest.TestCase):
    def test_perfect_match(self):
        _, _, wer, sub, dele, ins = process_one("Hello world.", "hello, world", "en")
        self.assertEqual((wer, sub, dele, ins), (0.0, 0.0, 0.0, 0.0))

    def test_one_substitution_of_two_words(self):
        _, _, wer, sub, dele, ins = process_one("hello there", "hello world", "en")
        self.assertEqual((wer, sub, dele, ins), (0.5, 0.5, 0.0, 0.0))

    def test_insertion_fraction_uses_ref_length(self):
        _, _, wer, sub, dele, ins = process_one("a b c", "a b", "en")
        self.assertEqual((wer, sub, dele, ins), (0.5, 0.0, 0.0, 0.5))

    def test_degenerate_reference_nonempty_hyp_is_full_error(self):
        # all-punctuation ref: wer 1.0 (charged as insertion), no div-by-zero
        raw_truth, raw_hypo, wer, sub, dele, ins = process_one("something", "!!!", "en")
        self.assertEqual((raw_truth, raw_hypo), ("!!!", "something"))
        self.assertEqual((wer, sub, dele, ins), (1.0, 0.0, 0.0, 1.0))

    def test_degenerate_reference_empty_hyp_is_perfect(self):
        _, _, wer, sub, dele, ins = process_one("...", "!!!", "en")
        self.assertEqual((wer, sub, dele, ins), (0.0, 0.0, 0.0, 0.0))

    def test_zh_cer_counts_characters(self):
        # 1 wrong char of 4 -> CER 0.25
        _, _, wer, sub, dele, ins = process_one(
            "你好世人", "你好世界", "zh")
        self.assertEqual((wer, sub), (0.25, 0.25))


class BaseLangTest(unittest.TestCase):
    def test_bare_codes(self):
        for code in ("en", "zh", "uk", "yo", "fil", "nb"):
            self.assertEqual(base_lang(code), code)

    def test_compound_leaf_names(self):
        self.assertEqual(base_lang("hard_zh"), "zh")
        self.assertEqual(base_lang("to_en"), "en")
        # new-language compound leaves must resolve too (the pre-2026-08
        # 17-language KNOWN_LANGS returned 'hard' here, silently disabling
        # Whisper routing / ortho fold / CER)
        self.assertEqual(base_lang("hard_uk"), "uk")
        self.assertEqual(base_lang("hard_he"), "he")

    def test_known_langs_covers_the_full_benchmark_set(self):
        benchmark = set(
            "en zh de es fr it ar ja ko ru hi id th tl tr pl pt "
            "bg ca cs da el fa fi fil he hr hu ms nb nl ro sk sv uk vi yo".split())
        self.assertTrue(benchmark <= KNOWN_LANGS,
                        f"missing: {sorted(benchmark - KNOWN_LANGS)}")

    def test_unknown_leaf_falls_back_with_warning(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(base_lang("hardmode_v2"), "hardmode")
        self.assertIn("WARNING", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
