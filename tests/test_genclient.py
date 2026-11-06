"""Unit tests for utils/genclient.py (the shared generation-client harness).

Run from the repo root with either:
    python -m tests.test_genclient
    python tests/test_genclient.py

Covers the pure helpers (language detection/normalization, retry
classification incl. permanent-4xx and Retry-After handling, voice rotation,
--limit/--stratified-bins selection, L16 mime parsing, error formatting), the
scp-triple loader, the atomic WAV writer + audio decode round trip, and the
shared generate loop. No GPU, no network.
"""
import hashlib
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

# Make `utils` importable regardless of how this file is launched.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import genclient as gc  # noqa: E402


class DetectLangTest(unittest.TestCase):
    def test_latin_defaults_to_en(self):
        self.assertEqual(gc.detect_lang("Hello world."), "en")
        self.assertEqual(gc.detect_lang(""), "en")

    def test_hangul(self):
        self.assertEqual(gc.detect_lang("안녕하세요"), "ko")

    def test_kana(self):
        self.assertEqual(gc.detect_lang("こんにちは"), "ja")

    def test_han(self):
        self.assertEqual(gc.detect_lang("你好世界"), "zh")

    def test_kana_beats_han(self):
        # Japanese text mixes kanji + kana; kana must win.
        self.assertEqual(gc.detect_lang("日本語のテキストです"), "ja")

    def test_hangul_beats_all(self):
        self.assertEqual(gc.detect_lang("한국어 テキスト 漢字"), "ko")


class NormalizeLangTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(gc.normalize_lang(""), "")
        self.assertEqual(gc.normalize_lang(None), "")

    def test_test_set_names(self):
        self.assertEqual(gc.normalize_lang("to_hard_zh"), "zh")
        self.assertEqual(gc.normalize_lang("to_en"), "en")

    def test_aliases(self):
        self.assertEqual(gc.normalize_lang("cmn"), "zh")
        self.assertEqual(gc.normalize_lang("Mandarin"), "zh")
        self.assertEqual(gc.normalize_lang("JP"), "ja")
        self.assertEqual(gc.normalize_lang("korean"), "ko")
        self.assertEqual(gc.normalize_lang(" English "), "en")

    def test_passthrough(self):
        self.assertEqual(gc.normalize_lang("de"), "de")
        self.assertEqual(gc.normalize_lang("fr"), "fr")


class PickVoiceTest(unittest.TestCase):
    POOL = ["a", "b", "c", "d"]

    def test_deterministic(self):
        self.assertEqual(gc.pick_voice("uttid_7", self.POOL),
                         gc.pick_voice("uttid_7", self.POOL))

    def test_matches_md5_definition(self):
        # The exact historical definition every client used: md5(uid) % len(pool).
        for uid in ("uttid_0", "uttid_1", "uttid_1499"):
            idx = int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % 4
            self.assertEqual(gc.pick_voice(uid, self.POOL), self.POOL[idx])
            self.assertEqual(gc.pick_index(uid, 4), idx)

    def test_in_range(self):
        for i in range(50):
            self.assertIn(gc.pick_voice(f"uttid_{i}", self.POOL), self.POOL)


class FakeHTTPError(Exception):
    """requests.HTTPError shape: .response.status_code / .response.headers."""

    def __init__(self, status, headers=None):
        super().__init__(f"{status} error")
        self.response = mock.Mock(status_code=status, headers=headers or {})


class RetryClassificationTest(unittest.TestCase):
    def test_status_from_requests_shape(self):
        self.assertEqual(gc.http_status_of(FakeHTTPError(404)), 404)

    def test_status_from_status_code_attr(self):
        e = Exception("boom")
        e.status_code = 401
        self.assertEqual(gc.http_status_of(e), 401)

    def test_status_from_urllib_httperror(self):
        e = urllib.error.HTTPError("http://x", 503, "unavailable", {}, None)
        self.assertEqual(gc.http_status_of(e), 503)

    def test_status_from_runtimeerror_prefix(self):
        # generate_omni.http_post raises RuntimeError('HTTP 400: ...').
        self.assertEqual(gc.http_status_of(RuntimeError("HTTP 400: bad lang")), 400)
        self.assertIsNone(gc.http_status_of(RuntimeError("no audio in response")))

    def test_plain_exception_has_no_status(self):
        self.assertIsNone(gc.http_status_of(ValueError("nope")))
        self.assertIsNone(gc.http_status_of(FileNotFoundError(2, "missing")))

    def test_is_permanent_table(self):
        self.assertTrue(gc.is_permanent(FakeHTTPError(400)))
        self.assertTrue(gc.is_permanent(FakeHTTPError(401)))
        self.assertTrue(gc.is_permanent(FakeHTTPError(404)))
        self.assertFalse(gc.is_permanent(FakeHTTPError(408)))  # timeout: retry
        self.assertFalse(gc.is_permanent(FakeHTTPError(429)))  # rate limit: retry
        self.assertFalse(gc.is_permanent(FakeHTTPError(500)))
        self.assertFalse(gc.is_permanent(FakeHTTPError(503)))
        self.assertFalse(gc.is_permanent(ValueError("transient-looking")))

    def test_retry_after_seconds(self):
        self.assertEqual(gc.retry_after_of(
            FakeHTTPError(429, headers={"Retry-After": "7"})), 7.0)
        self.assertIsNone(gc.retry_after_of(FakeHTTPError(429)))
        # HTTP-date form is ignored, not crashed on.
        self.assertIsNone(gc.retry_after_of(
            FakeHTTPError(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})))


class WithRetryTest(unittest.TestCase):
    def _run(self, fn, attempts=4, base_delay=2.0):
        sleeps = []
        with mock.patch.object(gc.time, "sleep", side_effect=sleeps.append):
            result = gc.with_retry(fn, attempts, base_delay)
        return result, sleeps

    def test_success_first_try(self):
        result, sleeps = self._run(lambda: "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [])

    def test_transient_then_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeHTTPError(500)
            return "ok"

        result, sleeps = self._run(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(sleeps), 2)
        # exponential backoff: base*2^i plus jitter in [0, base)
        self.assertGreaterEqual(sleeps[0], 2.0)
        self.assertGreaterEqual(sleeps[1], 4.0)

    def test_exhausted_reraises_last(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise FakeHTTPError(500)

        with mock.patch.object(gc.time, "sleep"):
            with self.assertRaises(FakeHTTPError):
                gc.with_retry(fn, 4, 0.01)
        self.assertEqual(calls["n"], 4)

    def test_permanent_4xx_raises_immediately(self):
        for status in (400, 401, 403, 404, 422):
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                raise FakeHTTPError(status)

            with mock.patch.object(gc.time, "sleep") as sl:
                with self.assertRaises(FakeHTTPError):
                    gc.with_retry(fn, 4, 2.0)
            self.assertEqual(calls["n"], 1, f"status {status} was retried")
            sl.assert_not_called()

    def test_408_and_429_are_retried(self):
        for status in (408, 429):
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                raise FakeHTTPError(status)

            with mock.patch.object(gc.time, "sleep"):
                with self.assertRaises(FakeHTTPError):
                    gc.with_retry(fn, 3, 0.01)
            self.assertEqual(calls["n"], 3)

    def test_429_honors_retry_after(self):
        def fn():
            raise FakeHTTPError(429, headers={"Retry-After": "17"})

        sleeps = []
        with mock.patch.object(gc.time, "sleep", side_effect=sleeps.append):
            with self.assertRaises(FakeHTTPError):
                gc.with_retry(fn, 2, 0.01)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 17.0)

    def test_429_retry_after_capped(self):
        def fn():
            raise FakeHTTPError(429, headers={"Retry-After": "3600"})

        sleeps = []
        with mock.patch.object(gc.time, "sleep", side_effect=sleeps.append):
            with self.assertRaises(FakeHTTPError):
                gc.with_retry(fn, 2, 0.01)
        self.assertLessEqual(sleeps[0], 61.0)

    def test_permanent_runtimeerror_http_prefix(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise RuntimeError("HTTP 400: Invalid language 'Klingon'")

        with mock.patch.object(gc.time, "sleep"):
            with self.assertRaises(RuntimeError):
                gc.with_retry(fn, 4, 2.0)
        self.assertEqual(calls["n"], 1)


class FormatErrorTest(unittest.TestCase):
    def test_includes_type_and_message(self):
        self.assertEqual(gc.format_error(ValueError("boom")),
                         "error: ValueError: boom")

    def test_empty_message_keeps_type(self):
        # The 61/403 in-the-wild/ko failure mode: str(e) == "".
        self.assertEqual(gc.format_error(ValueError()), "error: ValueError: ")


class ParseL16RateTest(unittest.TestCase):
    def test_missing_mime_uses_default(self):
        self.assertEqual(gc.parse_l16_rate(None, 24000), 24000)
        self.assertEqual(gc.parse_l16_rate("", 24000), 24000)

    def test_l16_with_rate(self):
        self.assertEqual(gc.parse_l16_rate("audio/L16;codec=pcm;rate=24000", 8000),
                         24000)
        self.assertEqual(gc.parse_l16_rate("audio/l16; rate=16000", 24000), 16000)

    def test_l16_without_rate_uses_default(self):
        self.assertEqual(gc.parse_l16_rate("audio/L16", 24000), 24000)

    def test_non_l16_raises(self):
        with self.assertRaises(RuntimeError):
            gc.parse_l16_rate("audio/mp3", 24000)
        with self.assertRaises(RuntimeError):
            gc.parse_l16_rate("audio/ogg;codec=opus", 24000)

    def test_bad_rate_raises(self):
        with self.assertRaises(RuntimeError):
            gc.parse_l16_rate("audio/L16;rate=fast", 24000)


class ApplyLimitTest(unittest.TestCase):
    UIDS = [f"uttid_{i}" for i in range(100)]

    def test_noop(self):
        self.assertEqual(gc.apply_limit(self.UIDS), self.UIDS)
        self.assertEqual(gc.apply_limit(self.UIDS, 0, 0), self.UIDS)

    def test_limit(self):
        self.assertEqual(gc.apply_limit(self.UIDS, limit=5), self.UIDS[:5])

    def test_stratified_evenly_spaced(self):
        picked = gc.apply_limit(self.UIDS, stratified_bins=5)
        self.assertEqual(picked, [self.UIDS[i] for i in (0, 20, 40, 60, 80)])

    def test_stratified_wins_over_limit(self):
        picked = gc.apply_limit(self.UIDS, limit=3, stratified_bins=5)
        self.assertEqual(len(picked), 5)

    def test_stratified_larger_than_set_is_noop(self):
        self.assertEqual(gc.apply_limit(self.UIDS[:3], stratified_bins=5),
                         self.UIDS[:3])


class LoadTestSetTest(unittest.TestCase):
    def _make_dir(self, text=None, prompt_wav=None, prompt_text=None):
        d = tempfile.mkdtemp(prefix="genclient_test_")
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        if text is not None:
            Path(d, "text").write_text(text, encoding="utf-8")
        if prompt_wav is not None:
            Path(d, "prompt_wav.scp").write_text(prompt_wav, encoding="utf-8")
        if prompt_text is not None:
            Path(d, "prompt_text").write_text(prompt_text, encoding="utf-8")
        return d

    def test_intersection(self):
        d = self._make_dir(
            text="uttid_1 hello there\nuttid_2 second\nuttid_3 third\n",
            prompt_wav="uttid_1 /nope/a.wav\nuttid_2 /nope/b.wav\n",
            prompt_text="uttid_1 ref one\nuttid_2 ref two\nuttid_9 orphan\n")
        texts, prompts, prompt_texts, uids = gc.load_test_set(d)
        self.assertEqual(uids, ["uttid_1", "uttid_2"])
        self.assertEqual(texts["uttid_1"], "hello there")
        self.assertEqual(len(texts), 3)
        self.assertEqual(len(prompt_texts), 3)

    def test_no_text_file_exits(self):
        d = self._make_dir()
        with self.assertRaises(SystemExit):
            gc.load_test_set(d)

    def test_empty_intersection_exits(self):
        d = self._make_dir(text="uttid_1 hello\n", prompt_wav="uttid_2 /x.wav\n",
                           prompt_text="uttid_2 x\n")
        with self.assertRaises(SystemExit):
            gc.load_test_set(d)

    def test_optional_prompt_text(self):
        d = self._make_dir(text="uttid_1 hello\n", prompt_wav="uttid_1 /x.wav\n")
        texts, prompts, prompt_texts, uids = gc.load_test_set(
            d, need_prompt_text=False)
        self.assertEqual(uids, ["uttid_1"])
        self.assertEqual(prompt_texts, {})


class AudioIOTest(unittest.TestCase):
    """Round-trip tests for the audio writers (main venv has numpy/soundfile)."""

    def setUp(self):
        try:
            import numpy  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("numpy/soundfile not installed in this venv")
        self.dir = tempfile.mkdtemp(prefix="genclient_audio_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir,
                                                            ignore_errors=True))

    def test_save_wav_atomic_roundtrip(self):
        import numpy as np
        import soundfile as sf
        sr = 16000
        data = np.sin(np.linspace(0, 40 * np.pi, sr)).astype("float32") * 0.5
        out = os.path.join(self.dir, "a.wav")
        dur = gc.save_wav_atomic(data, sr, out)
        self.assertAlmostEqual(dur, 1.0, places=3)
        self.assertTrue(os.path.exists(out))
        self.assertFalse(os.path.exists(out + ".tmp.wav"))  # tmp cleaned up
        back, back_sr = sf.read(out, dtype="float32")
        self.assertEqual(back_sr, sr)
        self.assertEqual(back.shape[0], sr)
        info = sf.info(out)
        self.assertEqual(info.subtype, "PCM_16")

    def test_save_wav_atomic_empty_raises(self):
        out = os.path.join(self.dir, "b.wav")
        with self.assertRaises(RuntimeError):
            gc.save_wav_atomic([], 16000, out)
        self.assertFalse(os.path.exists(out))

    def test_bytes_to_wav_decodes_wav_bytes(self):
        import io
        import numpy as np
        import soundfile as sf
        sr = 8000
        data = np.sin(np.linspace(0, 20 * np.pi, sr // 2)).astype("float32") * 0.3
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        out = os.path.join(self.dir, "c.wav")
        dur = gc.bytes_to_wav(buf.getvalue(), out)
        self.assertAlmostEqual(dur, 0.5, places=3)
        back, back_sr = sf.read(out, dtype="float32")
        self.assertEqual(back_sr, sr)

    def test_decode_audio_garbage_raises(self):
        with self.assertRaises(Exception):
            gc.decode_audio(b"this is not audio at all")


class RunGenerateLoopTest(unittest.TestCase):
    def _generate(self, uid):
        if uid.endswith("_skip"):
            return uid, "skip", 0.0
        if uid.endswith("_bad"):
            return uid, gc.format_error(ValueError("nope")), 0.0
        return uid, "ok", 2.0

    def test_sequential_counts(self):
        uids = ["u_1", "u_2_skip", "u_3_bad", "u_4"]
        ok, skip, fail = gc.run_generate_loop(uids, self._generate, workers=1,
                                              desc="t", wav_dir="/tmp/x")
        self.assertEqual((ok, skip, fail), (2, 1, 1))

    def test_threaded_counts(self):
        uids = [f"u_{i}" for i in range(10)] + ["u_a_bad", "u_b_skip"]
        ok, skip, fail = gc.run_generate_loop(uids, self._generate, workers=4,
                                              desc="t", wav_dir="/tmp/x")
        self.assertEqual((ok, skip, fail), (10, 1, 1))


if __name__ == "__main__":
    unittest.main()
