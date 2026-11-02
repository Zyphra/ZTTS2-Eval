"""Unit tests for the benchmark-v2 scoring changes.

No GPU, no network. Run from anywhere:
    .venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py' -v

Pins: digit expansion on both sides of the WER text prep, near-duplicate matching, ASR
routing (and its hard failure without a Whisper server), the footer stamps that
build_results refuses to aggregate without, the sentence-clustered bootstrap, the
split-half geometry shared by char/s, rhythm and pitch range, the MPM span formula,
and pitch-range extraction on a synthetic glide.
"""
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from utils import average_wer, benchmark  # noqa: E402
from utils.text_norm import (KNOWN_LANGS, NUM2WORDS_LANGS, near_duplicate,  # noqa: E402
                             prepare)

PY = sys.executable


def _env(**over):
    env = dict(os.environ)
    env.pop("WHISPER_BASE_URL", None)
    env.update(over)
    return env


class NumberExpansionTest(unittest.TestCase):
    def test_digits_become_words_on_both_sides(self):
        a = prepare("It costs 20 dollars on the 12th, 1,000 in 1999", "en")
        b = prepare("it costs twenty dollars on the twelfth one thousand in "
                    "nineteen ninety-nine", "en")
        self.assertEqual(a, b)

    def test_unsupported_language_keeps_digits(self):
        rest = sorted(KNOWN_LANGS - NUM2WORDS_LANGS)
        if not rest:
            self.skipTest("num2words covers every benchmark language")
        self.assertIn("12", prepare("abc 12 def", rest[0]).split())


class NearDuplicateTest(unittest.TestCase):
    def test_case_and_punctuation_only(self):
        self.assertTrue(near_duplicate("Hello, world!", "hello world"))

    def test_unrelated_sentences(self):
        self.assertFalse(near_duplicate("The earthquake was centered north of Dillon.",
                                        "Sharks rarely attack humans."))


class RoutingTest(unittest.TestCase):
    def test_backend_per_language(self):
        for lang in ("en", "fil", "tl"):
            self.assertEqual(benchmark.asr_backend(lang), "whisper", lang)
        for lang in ("zh", "hi"):
            self.assertEqual(benchmark.asr_backend(lang), "qwen", lang)

    def _manifest(self, d):
        m = Path(d) / "manifest"
        m.write_text(f"{d}/does_not_exist.wav|hello world\n", encoding="utf-8")
        return m

    def test_whisper_lang_without_server_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([PY, "run_wer.py", str(self._manifest(d)), f"{d}/out.raw", "en"],
                               cwd=REPO / "utils", env=_env(ASR_ROUTING="1"),
                               capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("routes to Whisper-large-v3 but WHISPER_BASE_URL is unset", r.stderr)

    def test_missing_wav_is_imputed_and_footer_is_stamped(self):
        with tempfile.TemporaryDirectory() as d:
            raw, wer = f"{d}/out.raw", f"{d}/out.wer"
            r = subprocess.run([PY, "run_wer.py", str(self._manifest(d)), raw, "zh"],
                               cwd=REPO / "utils", env=_env(ASR_ROUTING="0"),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(Path(raw).read_text(encoding="utf-8").split("\t")[3]
                            .startswith("<UNSCOREABLE:missing_wav>"))
            subprocess.run([PY, "average_wer.py", raw, wer], cwd=REPO / "utils", check=False,
                           capture_output=True, text=True)
            foot = Path(wer).read_text(encoding="utf-8")
        for line in ("WER_CLIP: 100.0", "N_IMPUTED: 1", "WER_TAIL50: 100.000",
                     "BENCHMARK_VERSION: 2",
                     "ASR: qwen Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5"):
            self.assertIn(line, foot)


class ClusteredBootstrapTest(unittest.TestCase):
    def test_single_cluster_has_zero_width(self):
        lo, hi = average_wer.boot_ci95([0, .5, 1, .2], clusters=["a"] * 4)
        self.assertEqual(lo, hi)

    def test_unclustered_matches_the_original_algorithm(self):
        vals = [0.0, 0.1, 0.5, 1.0, 0.2, 0.0, 0.3]
        a = np.asarray(vals, dtype=float)
        rng = np.random.default_rng(7)
        means = a[rng.integers(0, a.size, size=(300, a.size))].mean(axis=1) * 100.0
        want = (round(float(np.percentile(means, 2.5)), 3),
                round(float(np.percentile(means, 97.5)), 3))
        self.assertEqual(average_wer.boot_ci95(vals, 100.0, B=300, seed=7), want)

    def test_clusters_widen_the_interval_when_rows_repeat(self):
        vals = [0.0] * 20 + [1.0] * 20
        keys = [str(i // 4) for i in range(40)]      # 10 sentences x 4 repeats
        plain = average_wer.boot_ci95(vals, B=2000, seed=1)
        clust = average_wer.boot_ci95(vals, B=2000, seed=1, clusters=keys)
        self.assertGreaterEqual(clust[1] - clust[0], plain[1] - plain[0])


class SplitHalfGeometryTest(unittest.TestCase):
    def test_w2_fair_of_identical_samples_sits_at_the_ceiling(self):
        import speaking_rate_dist as sr
        ref = np.random.default_rng(0).normal(15, 3, 1000)
        self.assertEqual(sr.w2(ref, ref), 0.0)
        ratio = sr.w2_fair(ref, ref) / sr.ceiling(ref)
        self.assertTrue(0.5 < ratio < 2, ratio)


class CharSecContentFilterTest(unittest.TestCase):
    # char/s divides the TARGET text by the duration, so a truncated or runaway
    # utterance reads as a pace error: only the ones the ASR heard (WER <= MAX_WER) count.
    def _leaf(self, d, wers):
        wavs = Path(d) / "wavs"
        wavs.mkdir()
        lines = []
        for i, w in enumerate(wers, 1):
            sf.write(wavs / f"uttid_{i}.wav", np.zeros(1600, np.float32), 16000)
            if w is not None:
                lines.append(f"{wavs}/uttid_{i}.wav\t{w}\tref\thyp\n")
        if lines:
            (Path(d) / "wav_res_ref_text.wer.raw").write_text("".join(lines))
        return str(wavs)

    def test_only_utterances_the_asr_heard_are_scored(self):
        import speaking_rate_dist as sr
        with tempfile.TemporaryDirectory() as d:
            files, share = sr.heard_files(self._leaf(d, [0.0, sr.MAX_WER, 0.75, None]))
        self.assertEqual([Path(f).stem for f in files], ["uttid_1", "uttid_2"])
        self.assertAlmostEqual(share, 0.5)

    def test_untranscribed_leaf_gets_no_char_s(self):
        import speaking_rate_dist as sr
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sr.heard_files(self._leaf(d, [None, None])), ([], 0.0))


class MpmSpanTest(unittest.TestCase):
    def test_span_anchors(self):
        from utils.score_mpm_shuffle_floor import mpm_span
        floor, noise = 2.0, 10.0
        self.assertEqual(mpm_span(floor, floor, noise), 0.0)
        self.assertEqual(mpm_span(noise, floor, noise), 1.0)
        self.assertAlmostEqual(mpm_span(6.0, floor, noise), 0.5)
        self.assertGreater(mpm_span(14.0, floor, noise), 1.0)

    def test_leading_and_trailing_silence_never_reach_the_features(self):
        # The FLEURS-R references carry ~2.45 s of edge silence and most systems almost
        # none; untrimmed, MPM span measured that padding instead of prosody.
        from utils.score_mpm_shuffle_floor import SR, _load_wav
        t = np.arange(int(1.5 * SR)) / SR
        voice = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        padded = np.concatenate([np.zeros(SR, np.float32), voice, np.zeros(int(0.8 * SR), np.float32)])
        with tempfile.TemporaryDirectory() as d:
            sf.write(f"{d}/plain.wav", voice, SR)
            sf.write(f"{d}/padded.wav", padded, SR)
            plain, pad = _load_wav(f"{d}/plain.wav"), _load_wav(f"{d}/padded.wav")
        self.assertLess(abs(len(pad) - len(plain)), 2048)   # within librosa's trim framing


class PitchRangeTest(unittest.TestCase):
    def test_octave_glide_range(self):
        from utils.pitch_range import range_st
        sr_, dur = 16000, 2.0
        t = np.arange(int(sr_ * dur)) / sr_
        # log-linear glide 150 -> 300 Hz: 12 st total, so p90 - p10 = 0.8 * 12 = 9.6 st
        phase = 2 * np.pi * 150.0 * (dur / math.log(2)) * (2 ** (t / dur) - 1)
        wav = sum(np.sin(h * phase) / h for h in (1, 2, 3)) * 0.2
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/glide.wav"
            sf.write(p, wav.astype(np.float32), sr_)
            r = range_st(p)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 9.6, delta=1.0)


class LeafProblemsTest(unittest.TestCase):
    def _leaf(self, d, wer_lines):
        (Path(d) / "wav_res_ref_text.wer").write_text("WER: 1.0\n" + "".join(wer_lines),
                                                       encoding="utf-8")

    def test_unstamped_wer_is_stale(self):
        with tempfile.TemporaryDirectory() as d:
            self._leaf(d, [])
            self.assertTrue(benchmark.leaf_problems(d, "en"))

    def test_current_en_leaf_is_clean_and_wrong_backend_is_not(self):
        v = benchmark.BENCHMARK_VERSION
        with tempfile.TemporaryDirectory() as d:
            self._leaf(d, [f"BENCHMARK_VERSION: {v}\n",
                           f"ASR: {benchmark.expected_asr_stamp('en')}\n"])
            self.assertEqual(benchmark.leaf_problems(d, "en"), [])
            self._leaf(d, [f"BENCHMARK_VERSION: {v}\n",
                           f"ASR: {benchmark.asr_stamp('qwen', *benchmark.ASR_MODELS['qwen'])}\n"])
            self.assertTrue(benchmark.leaf_problems(d, "en"))

    def test_speaker_and_msr_stamps_are_checked(self):
        v = benchmark.BENCHMARK_VERSION
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "spk_simi_scores.txt").write_text(
                f"avg 70.00 60.00\n#BENCHMARK_VERSION: {v}\n#SPK_MODEL: other\n")
            self.assertTrue(benchmark.leaf_problems(d, "en"))
            (Path(d) / "spk_simi_scores.txt").write_text(
                f"avg 70.00 60.00\n#BENCHMARK_VERSION: {v}\n"
                f"#SPK_MODEL: {benchmark.EXPECTED_SPK_STAMP}\n")
            self.assertEqual(benchmark.leaf_problems(d, "en"), [])
            (Path(d) / "msr_utmos_mean.txt").write_text("3.0\n")
            self.assertTrue(benchmark.leaf_problems(d, "en"))
            (Path(d) / "msr_utmos_ci.txt").write_text(
                f"CI95 1 2 n 3\nBENCHMARK_VERSION: {v}\n"
                f"MSR_UTMOS_CKPT: {benchmark.MSR_UTMOS_CKPT}\n")
            self.assertEqual(benchmark.leaf_problems(d, "en"), [])


if __name__ == "__main__":
    unittest.main()
