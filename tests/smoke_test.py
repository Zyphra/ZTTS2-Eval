"""No-GPU, no-network smoke test for the shared utils modules.

Run from the repo root with either:
    python -m tests.smoke_test
    python tests/smoke_test.py

Covers the shared scp helpers (utils.scp), the average_wer aggregation, the
wav/text join, the explorer server API, plus an import-check of every
scripts/generate_*.py client and a light touch on utils/genclient.py's pure
helpers (full coverage lives in tests/test_genclient.py). It does NOT import
the heavy scoring drivers (torch / vllm), so it stays fast and runnable
anywhere.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

# Make `utils` importable regardless of how this file is launched.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.scp import load_scp, load_scp_items, load_text  # noqa: E402
from utils.ground_truth_split import split_indices  # noqa: E402

_TMP_FILES = []


def _write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".scp", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    _TMP_FILES.append(f.name)
    return f.name


def tearDownModule():
    for p in _TMP_FILES:
        try:
            os.unlink(p)
        except OSError:
            pass


class ScpTest(unittest.TestCase):
    def test_load_scp_dict(self):
        p = _write("u1 hello world\nu2 foo\n\n")  # blank line skipped
        self.assertEqual(load_scp(p), {"u1": "hello world", "u2": "foo"})

    def test_load_scp_items_order_and_tab(self):
        p = _write("u1\t/a/b.wav\nu2 /c/d.wav\n")  # tab or space accepted
        self.assertEqual(load_scp_items(p),
                         [("u1", "/a/b.wav"), ("u2", "/c/d.wav")])

    def test_load_scp_items_preserves_duplicates(self):
        p = _write("u1 /x.wav\nu1 /y.wav\n")  # list keeps both; dict would collapse
        self.assertEqual(len(load_scp_items(p)), 2)

    def test_load_text_keeps_internal_spacing(self):
        p = _write("u1 a sentence  with  spaces\n")
        self.assertEqual(load_text(p), {"u1": "a sentence  with  spaces"})

    def test_load_scp_collapses_double_slash_without_resolving(self):
        p = _write("u1 data/clean//en/x.wav\n")
        self.assertEqual(load_scp(p, resolve=False), {"u1": "data/clean/en/x.wav"})

    def test_missing_ok_returns_empty(self):
        missing = "/nonexistent/does/not/exist"
        self.assertEqual(load_scp(missing, missing_ok=True), {})
        self.assertEqual(load_text(missing, missing_ok=True), {})
        with self.assertRaises(OSError):
            load_text(missing)


class EntryPointValidationTest(unittest.TestCase):
    """run_score.sh / scripts/lib_gen.sh reject bad arguments before any server starts."""

    def _run(self, *cmd):
        t0 = time.monotonic()
        proc = subprocess.run(list(cmd), cwd=REPO, capture_output=True, text=True, timeout=60)
        return proc, time.monotonic() - t0

    def test_run_score_requires_decode_dir(self):
        proc, _ = self._run("bash", "run_score.sh")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("--decode_dir is required", proc.stderr)

    def test_run_score_unknown_task_fails_fast(self):
        proc, dt = self._run("bash", "run_score.sh", "--decode_dir", "/tmp/x/bogus")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("unknown task 'bogus'", proc.stderr)
        self.assertLess(dt, 5.0)  # the ASR servers never started

    def test_gen_langs_basket_filter_and_unknown_task(self):
        proc, _ = self._run("bash", "-c",
                            '. scripts/lib_gen.sh; gen_langs in-the-wild "" "en fr xx"')
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout.split(), ["en", "fr"])
        proc, _ = self._run("bash", "-c", '. scripts/lib_gen.sh; gen_langs nosuch ""')
        self.assertEqual(proc.returncode, 1)
        self.assertIn("unknown task 'nosuch'", proc.stderr)


class AverageWerTest(unittest.TestCase):
    """End-to-end run of utils/average_wer.py on synthetic .raw rows.

    Row format (tab-separated): wav_path  wer  text_ref  text_res  inse  dele  subs
    """

    def _run(self, rows):
        infile = _write("".join(rows))
        outfile = infile + ".out"
        _TMP_FILES.append(outfile)
        proc = subprocess.run(
            [sys.executable, str(REPO / "utils" / "average_wer.py"), infile, outfile],
            capture_output=True, text=True)
        return proc, outfile

    @staticmethod
    def _row(wer, inse=0.0, dele=0.0, subs=0.0):
        return f"/x.wav\t{wer}\tref text\thyp text\t{inse}\t{dele}\t{subs}\n"

    @staticmethod
    def _summary(outfile):
        """The `WER: ...` summary line. A `WER_CI95:` line now follows it (CIs are
        emitted by default), so the summary is no longer the last line of the file."""
        lines = Path(outfile).read_text(encoding="utf-8").splitlines()
        return next(l for l in lines if l.startswith("WER:"))

    def test_mixed_wers(self):
        proc, outfile = self._run([self._row(0.2, subs=0.2), self._row(0.8, subs=0.8),
                                   self._row(1.5, inse=1.5)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = self._summary(outfile)
        # mean(0.2, 0.8, 1.5)*100 = 83.333; clip(0.2, 0.8, 1.0) -> 66.667; L50: only 0.2
        self.assertIn("WER: 83.333", summary)
        self.assertIn("WER_L50: 20.0", summary)
        self.assertIn("WER_CLIP: 66.667", summary)

    def test_all_zero_wer_does_not_crash(self):
        # Regression: perfect-WER runs used to hit dead-code ratios dividing by mean WER 0.
        proc, outfile = self._run([self._row(0.0), self._row(0.0)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("WER: 0.0,", self._summary(outfile))

    def test_all_above_50_falls_back(self):
        # No <=50% utterances: L50 falls back to the full-set mean instead of NaN.
        proc, outfile = self._run([self._row(0.9), self._row(0.7)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = self._summary(outfile)
        self.assertIn("WER: 80.0", summary)
        self.assertIn("WER_L50: 80.0", summary)
        self.assertNotIn("nan", summary)

    def test_empty_input_fails_cleanly(self):
        proc, _ = self._run([])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no scored utterances", proc.stderr + proc.stdout)

    def test_malformed_line_reports_lineno(self):
        proc, _ = self._run(["only three\tfields\there\n"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(":1:", proc.stderr + proc.stdout)


class GetWavResRefTextTest(unittest.TestCase):
    """utils/get_wav_res_ref_text.py joins the target text with the generated wavs;
    missing wavs are dropped HERE (before run_wer.py counts anything), so the
    coverage warning it prints is the only trace of incomplete generation."""

    def test_join_and_missing_wav_warning(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wav_dir = Path(d) / "wavs"
        wav_dir.mkdir()
        (wav_dir / "u1.wav").write_bytes(b"RIFF")  # only u1 was generated
        meta = Path(d) / "text"
        meta.write_text("u1 first target text\nu2 second target text\n", encoding="utf-8")
        out = Path(d) / "wav_res_ref_text"

        proc = subprocess.run(
            [sys.executable, str(REPO / "utils" / "get_wav_res_ref_text.py"),
             str(meta), str(wav_dir), str(out)],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows, [f"{wav_dir / 'u1.wav'}|first target text"])
        self.assertIn("kept 1/2", proc.stdout)
        self.assertIn("WARNING: 1 missing", proc.stdout)


class GroundTruthSplitTest(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_exhaustive(self):
        left, right = split_indices(7, seed=0, key="clean/en")
        self.assertEqual((left, right), split_indices(7, seed=0, key="clean/en"))
        self.assertEqual((len(left), len(right)), (3, 4))
        self.assertFalse(set(left) & set(right))
        self.assertEqual(sorted(left + right), list(range(7)))

    def test_dataset_key_changes_the_sampling(self):
        left, _ = split_indices(20, seed=0, key="clean/en")
        other, _ = split_indices(20, seed=0, key="clean/de")
        self.assertNotEqual(left, other)


class PanelAliasTest(unittest.TestCase):
    """utils.benchmark.is_panel_alias: only a panel dir symlinked to the system's OWN
    full-set dir is an alias. Scoring and the results walk skip aliases, so a genuine
    panel run (a real dir, or a symlink into another store) must never count as one."""

    def test_only_a_link_to_the_own_full_set_is_an_alias(self):
        from utils.benchmark import is_panel_alias
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        store = root / "store" / "clean_mini"
        store.mkdir(parents=True)
        sysd = root / "results" / "s"
        (sysd / "clean").mkdir(parents=True)
        (sysd / "in-the-wild_mini").mkdir()                    # genuine panel run
        os.symlink("clean", sysd / "clean_mini")       # browse-only alias
        self.assertTrue(is_panel_alias(str(sysd), "clean_mini"))
        self.assertFalse(is_panel_alias(str(sysd), "in-the-wild_mini"))
        self.assertFalse(is_panel_alias(str(sysd), "clean"))
        self.assertFalse(is_panel_alias(str(sysd), "en"))     # GT leaves are flat
        os.remove(sysd / "clean_mini")
        os.symlink(store, sysd / "clean_mini")            # panel run in another store
        self.assertFalse(is_panel_alias(str(sysd), "clean_mini"))



class ExplorerServerTest(unittest.TestCase):
    """Live checks of explorer/serve.py (stdlib server): API basics plus the
    path-traversal / symlink-escape protections of /api/audio, and --allow-root
    (a shared store that results/<system> symlinks point into)."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        wavdir = Path(cls.root) / "data" / "waveform"
        wavdir.mkdir(parents=True)
        (wavdir / "ok.wav").write_bytes(b"\x00" * 256)
        # a file outside --root, reachable only through a symlink inside it
        cls.outside = tempfile.mkdtemp()
        (Path(cls.outside) / "secret.wav").write_bytes(b"\x00" * 16)
        os.symlink(cls.outside, Path(cls.root) / "escape_link")

        # a coreset panel: two of the leaf's three utterances, with weights
        pdir = Path(cls.root) / "data" / "clean_mini"
        (pdir / "en").mkdir(parents=True)
        (pdir / "panel.json").write_text(json.dumps({
            "kind": "mini", "source_task": "clean", "selector": "strat_meta",
            "k_requested": 2, "k_actual": 2, "seed": 0,
            "langs": {"en": [["uttid_1", 0.75], ["uttid_2", 0.25]]}}))
        (pdir / "en" / "text").write_text("uttid_1 one\nuttid_2 two\nuttid_3 three\n")
        leaf = Path(cls.root) / "results" / "m" / "clean_mini" / "en"
        leaf.mkdir(parents=True)
        (leaf / "wav_res_ref_text.wer").write_text(
            "".join(f"/w/{u}.wav\t{w}\tref\thyp\t0.0\t0.0\t{w}\n"
                    for u, w in (("uttid_1", 0.1), ("uttid_2", 0.5), ("uttid_3", 0.9))))

        # a system dir that is a symlink into a shared store passed as --allow-root
        cls.store = tempfile.mkdtemp()
        sleaf = Path(cls.store) / "s" / "clean_mini" / "en"
        (sleaf / "wavs").mkdir(parents=True)
        (sleaf / "wavs" / "uttid_1.wav").write_bytes(b"\x00" * 64)
        (sleaf / "wav_res_ref_text.wer").write_text("/w/uttid_1.wav\t0.2\tref\thyp\t0.0\t0.0\t0.2\n")
        # wav.scp names where the leaf was scored, not where it is served from
        (sleaf / "wav.scp").write_text("uttid_1 /elsewhere/results/s/old_task/en/wavs/uttid_1.wav\n")
        os.symlink(Path(cls.store) / "s", Path(cls.root) / "results" / "s")

        with socket.socket() as s:  # free port
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        cls.proc = subprocess.Popen(
            [sys.executable, str(REPO / "explorer" / "serve.py"),
             "--port", str(cls.port), "--root", cls.root, "--allow-root", cls.store],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):  # wait for the server to come up
            try:
                cls._get("/api/snapshot")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("explorer server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.outside, ignore_errors=True)
        shutil.rmtree(cls.store, ignore_errors=True)

    @classmethod
    def _get(cls, path, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{cls.port}{path}", headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_snapshot_endpoint(self):
        status, body = self._get("/api/snapshot")
        self.assertEqual(status, 200)
        self.assertIn(b'"models"', body)

    def test_audio_serves_in_root_wav(self):
        status, body = self._get("/api/audio?path=data/waveform/ok.wav")
        self.assertEqual(status, 200)
        self.assertEqual(len(body), 256)

    def test_audio_range_request(self):
        status, body = self._get("/api/audio?path=data/waveform/ok.wav",
                                 headers={"Range": "bytes=0-99"})
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 100)

    def test_audio_rejects_non_wav(self):
        status, _ = self._get("/api/audio?path=/etc/passwd")
        self.assertEqual(status, 400)

    def test_audio_rejects_dotdot_traversal(self):
        status, _ = self._get("/api/audio?path=../../../etc/passwd.wav")
        self.assertEqual(status, 404)

    def test_audio_rejects_symlink_escape(self):
        status, _ = self._get("/api/audio?path=escape_link/secret.wav")
        self.assertEqual(status, 404)

    def test_allow_root_serves_symlinked_store_but_nothing_else(self):
        """--allow-root admits the store a results/ symlink resolves into; a symlink to
        any other directory still cannot escape --root."""
        _, body = self._get("/api/utterances?model=s&task=clean_mini&lang=en")
        wer = {r["uid"]: r["wer"] for r in json.loads(body)["rows"]}
        self.assertEqual(wer["uttid_1"], 0.2)  # scored from the store leaf
        status, body = self._get("/api/audio?path=results/s/clean_mini/en/wavs/uttid_1.wav")
        self.assertEqual((status, len(body)), (200, 64))
        status, _ = self._get("/api/audio?path=escape_link/secret.wav")
        self.assertEqual(status, 404)

    def test_generated_audio_url_follows_the_served_leaf(self):
        """A leaf scored in another checkout or under an older task name keeps that
        path in wav.scp; the drawer must still stream the leaf's own wavs/."""
        _, body = self._get("/api/utterances?model=s&task=clean_mini&lang=en")
        url = {r["uid"]: r.get("gen_url") for r in json.loads(body)["rows"]}["uttid_1"]
        status, body = self._get(url)
        self.assertEqual((status, len(body)), (200, 64))

    def test_meta_advertises_coreset_panels(self):
        status, body = self._get("/api/meta")
        self.assertEqual(status, 200)
        panels = json.loads(body)["panels"]
        self.assertEqual(panels["clean_mini"]["kind"], "mini")
        self.assertEqual(panels["clean_mini"]["source_task"], "clean")
        self.assertEqual(panels["clean_mini"]["langs"], 1)

    def test_utterances_panel_restricts_and_weights(self):
        """?panel= must cut the leaf down to the panel's items and attach each
        weight — the drawer has to summarise the same population the board scored."""
        _, body = self._get("/api/utterances?model=m&task=clean_mini&lang=en")
        self.assertEqual(json.loads(body)["total"], 3)  # unfiltered leaf

        _, body = self._get("/api/utterances?model=m&task=clean_mini&lang=en"
                            "&panel=clean_mini")
        rows = json.loads(body)
        self.assertEqual(rows["total"], 2)
        got = {r["uid"]: r["w"] for r in rows["rows"]}
        self.assertEqual(got, {"uttid_1": 0.75, "uttid_2": 0.25})

    def test_utterances_panel_name_cannot_escape_data_dir(self):
        """`panel` is joined onto data/, so it must stay a single path component."""
        _, body = self._get("/api/utterances?model=m&task=clean_mini&lang=en"
                            "&panel=../../etc")
        rows = json.loads(body)
        self.assertEqual(rows["total"], 3)                       # no restriction applied
        self.assertTrue(all("w" not in r for r in rows["rows"]))  # and no weights


class PerUidParserTest(unittest.TestCase):
    """utils.per_uid — the shared per-utterance score-file parsers."""

    def test_wer_handles_raw_and_footered_variants(self):
        from utils.per_uid import parse_wer
        rows = ("/x/wavs/uttid_1.wav\t0.25\tref one\thyp one\t0.0\t0.1\t0.15\n"
                "/x/wavs/uttid_2.wav\t1.5\tref two\thyp two\t0.5\t0.0\t1.0\n")
        raw = _write(rows)
        full = _write("utt\twav_res\tres_wer\ttext_ref\ttext_res\tins\tdel\tsub\n"
                      + rows
                      + "WER: 87.5, WER_L50: 25.0, WER_CLIP: 62.5\n"
                      + "WER_CI95: [1.0, 2.0], WER_CLIP_CI95: [1.0, 2.0], n: 2\n")
        for path in (raw, full):
            d = parse_wer(path)
            self.assertEqual(sorted(d), ["uttid_1", "uttid_2"])
            self.assertAlmostEqual(d["uttid_1"]["wer"], 0.25)
            self.assertAlmostEqual(d["uttid_2"]["sub"], 1.0)

    def test_spk_takes_last_column_and_skips_avg(self):
        from utils.per_uid import parse_spk
        p = _write("uttid_1 0.00 77.00\nuttid_2 0.00 64.50\navg 0.00 70.75\n")
        self.assertEqual(parse_spk(p), {"uttid_1": 77.0, "uttid_2": 64.5})

    def test_missing_file_is_empty_not_an_error(self):
        from utils.per_uid import parse_score_path, parse_spk, parse_wer
        for fn in (parse_wer, parse_spk, parse_score_path):
            self.assertEqual(fn("/nonexistent/does/not/exist"), {})

    def test_wer_clip_matches_average_wer(self):
        from utils.per_uid import wer_clip
        # average_wer.py:118 -- min(wer, 1.0); x100 at aggregation time (average_wer.py:131).
        self.assertAlmostEqual(wer_clip(0.25), 25.0)
        self.assertAlmostEqual(wer_clip(1.5), 100.0)


class WerRawRepairTest(unittest.TestCase):
    """scripts/repair_wer_raw.py — reassembly of records split by tabs/newlines."""

    @staticmethod
    def _repair(text):
        sys.path.insert(0, str(REPO / "scripts"))
        from repair_wer_raw import repair_lines
        return repair_lines(text)

    def test_clean_input_is_untouched(self):
        text = ("/x/wavs/uttid_1.wav\t0.0\tref\thyp\t0.0\t0.0\t0.0\n"
                "/x/wavs/uttid_2.wav\t0.5\tref two\thyp two\t0.1\t0.2\t0.2\n")
        recs, n_rep, dropped = self._repair(text)
        self.assertEqual(len(recs), 2)
        self.assertEqual((n_rep, dropped), (0, []))
        self.assertEqual("\n".join("\t".join(r) for r in recs) + "\n", text)

    def test_embedded_tab_in_hypothesis(self):
        text = "/x/wavs/uttid_1.wav\t0.5\tref text\thyp part a\thyp part b\t0.1\t0.0\t0.4\n"
        recs, n_rep, dropped = self._repair(text)
        self.assertEqual(len(recs), 1)
        self.assertEqual(n_rep, 1)
        self.assertEqual(dropped, [])
        self.assertEqual(recs[0][3], "hyp part a hyp part b")
        self.assertEqual(recs[0][-3:], ["0.1", "0.0", "0.4"])

    def test_embedded_newline_in_hypothesis(self):
        text = ("/x/wavs/uttid_1.wav\t0.5\tref\thyp line one\n"
                "\n"
                "hyp line two\t0.1\t0.0\t0.4\n"
                "/x/wavs/uttid_2.wav\t0.0\tr\th\t0.0\t0.0\t0.0\n")
        recs, n_rep, dropped = self._repair(text)
        self.assertEqual(len(recs), 2)
        self.assertEqual(n_rep, 1)
        self.assertEqual(dropped, [])
        self.assertEqual(recs[0][3], "hyp line one hyp line two")
        self.assertEqual(recs[1][0], "/x/wavs/uttid_2.wav")


class CoresetEstimatorTest(unittest.TestCase):
    """The weighted panel estimator must be exact when the panel is the whole set.

    This is the property everything else rests on: if the estimator is not the same
    macro-average the benchmark reports, no amount of clever selection helps.
    """

    @staticmethod
    def _panel_score(y, per_lang):
        import numpy as np
        vals = []
        for idx, w in per_lang:
            idx, w = np.asarray(idx), np.asarray(w, dtype=float)
            vals.append(float(np.sum(w * y[idx]) / np.sum(w)))
        return float(np.mean(vals))

    def test_full_set_panel_recovers_the_true_mean_exactly(self):
        import numpy as np
        rng = np.random.default_rng(0)
        y = rng.normal(size=50)
        langs = {"en": np.arange(0, 20), "fr": np.arange(20, 50)}
        per_lang = [(cols, np.full(cols.size, 1.0 / cols.size)) for cols in langs.values()]
        truth = float(np.mean([y[c].mean() for c in langs.values()]))
        self.assertAlmostEqual(self._panel_score(y, per_lang), truth, places=12)

    def test_stratum_mass_weights_are_unbiased_in_expectation(self):
        import numpy as np
        rng = np.random.default_rng(7)
        y = np.concatenate([rng.normal(0, 1, 80), rng.normal(10, 1, 20)])
        strata = [np.arange(0, 80), np.arange(80, 100)]
        truth = y.mean()
        draws = []
        for _ in range(4000):
            idx = [int(rng.choice(s)) for s in strata]
            w = np.array([s.size / y.size for s in strata])
            draws.append(float(np.sum(w * y[idx]) / np.sum(w)))
        # Unbiased by construction, so the Monte-Carlo mean converges on the truth.
        self.assertAlmostEqual(float(np.mean(draws)), truth, delta=0.06)

    def test_weights_sum_to_one_per_language(self):
        import numpy as np
        sys.path.insert(0, str(REPO / "scripts" / "coreset"))
        from common import check_weights
        good = {"langs": {"en": {"idx": [0, 1], "w": [0.5, 0.5]},
                          "fr": {"idx": [2], "w": [1.0]}}}
        bad = {"langs": {"en": {"idx": [0, 1], "w": [0.5, 0.4]}}}
        self.assertEqual(check_weights(good), [])
        self.assertEqual([lg for lg, _ in check_weights(bad)], ["en"])
        _ = np


class GenerateClientImportTest(unittest.TestCase):
    """Every scripts/generate_*.py must import cleanly: utils/genclient.py keeps
    third-party imports out of module scope, so an import-time failure here is a
    real regression (syntax error, moved helper, argv parsing at import), not a
    missing optional extra. Clients whose own deps genuinely are not installed
    in this venv are skipped, not failed."""

    def test_all_generate_clients_import(self):
        import importlib
        scripts_dir = str(REPO / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        paths = sorted((REPO / "scripts").glob("generate_*.py"))
        self.assertGreaterEqual(len(paths), 13)  # the current client roster
        for p in paths:
            with self.subTest(client=p.stem):
                try:
                    mod = importlib.import_module(p.stem)
                except ModuleNotFoundError as e:
                    self.skipTest(f"optional dep not installed: {e.name}")
                self.assertTrue(hasattr(mod, "main"),
                                f"{p.stem} has no main() entry point")


class GenClientHelpersSmokeTest(unittest.TestCase):
    """Light smoke of utils/genclient.py's pure helpers — the shared substrate of
    all 13 clients. Edge cases live in tests/test_genclient.py; this only guards
    the import path and one representative behaviour per helper family."""

    @classmethod
    def setUpClass(cls):
        from utils import genclient
        cls.gc = genclient

    def test_lang_helpers(self):
        self.assertEqual(self.gc.detect_lang("hello world"), "en")
        self.assertEqual(self.gc.detect_lang("你好世界"), "zh")
        self.assertEqual(self.gc.detect_lang("こんにちは"), "ja")
        self.assertEqual(self.gc.detect_lang("안녕하세요"), "ko")
        self.assertEqual(self.gc.normalize_lang("to_hard_zh"), "zh")
        self.assertEqual(self.gc.normalize_lang("English"), "en")

    def test_voice_rotation_is_deterministic_and_in_range(self):
        pool = ["a", "b", "c"]
        for uid in ("uttid_1", "uttid_2", "uttid_999"):
            idx = self.gc.pick_index(uid, len(pool))
            self.assertTrue(0 <= idx < len(pool))
            self.assertEqual(self.gc.pick_voice(uid, pool), pool[idx])  # stable

    def test_retry_classification_and_error_format(self):
        class Http(Exception):
            def __init__(self, code):
                super().__init__(f"HTTP {code}: nope")
        self.assertTrue(self.gc.is_permanent(Http(403)))
        self.assertFalse(self.gc.is_permanent(Http(429)))  # retryable 4xx
        self.assertFalse(self.gc.is_permanent(Http(500)))
        self.assertEqual(self.gc.format_error(ValueError("")), "error: ValueError: ")

    def test_parse_l16_rate(self):
        self.assertEqual(
            self.gc.parse_l16_rate("audio/L16;codec=pcm;rate=24000", 16000), 24000)
        self.assertEqual(self.gc.parse_l16_rate("", 16000), 16000)
        with self.assertRaises(RuntimeError):
            self.gc.parse_l16_rate("audio/mp3", 16000)

    def test_apply_limit_and_stratified_bins(self):
        uids = [f"u{i}" for i in range(10)]
        self.assertEqual(self.gc.apply_limit(uids, limit=3), uids[:3])
        picked = self.gc.apply_limit(uids, stratified_bins=4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(picked, sorted(picked, key=uids.index))  # order kept
        self.assertEqual(self.gc.apply_limit(uids), uids)


class PanelSnapshotTest(unittest.TestCase):
    """scripts/results_doc/build_results.py must summarise a panel leaf with the
    panel's weights, not with the leaf's own `.wer` footer (which is an unweighted
    mean over the sampled items — a different estimator that happens to look close)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / "scripts" / "results_doc"))
        import build_results
        cls.br = build_results

    def _leaf(self, tmp, wers, spks):
        leaf = Path(tmp) / "leaf"
        leaf.mkdir()
        (leaf / "wav_res_ref_text.wer.raw").write_text(
            "".join(f"/w/{u}.wav\t{v}\tref\thyp\t0.0\t0.0\t{v}\n" for u, v in wers.items()))
        (leaf / "spk_simi_scores.txt").write_text(
            "".join(f"{u} 0.00 {v}\n" for u, v in spks.items()) + "avg 0.00 0.00\n")
        return str(leaf)

    def test_panel_leaf_uses_panel_weights(self):
        wers = {"uttid_1": 0.10, "uttid_2": 0.50}
        spks = {"uttid_1": 60.0, "uttid_2": 80.0}
        panel = {"kind": "mini", "source_task": "clean", "selector": "s",
                 "w": {"en": {"uttid_1": 0.75, "uttid_2": 0.25}}}
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.br.collect_panel_leaf("m", "clean_mini", "en",
                                             self._leaf(tmp, wers, spks), panel)
        # weighted: .75*10 + .25*50 = 20.0   (unweighted would be 30.0)
        self.assertAlmostEqual(rec["wer"][0], 20.0, places=6)
        self.assertAlmostEqual(rec["wer"][2], 20.0, places=6)
        self.assertAlmostEqual(rec["spk"], 65.0, places=6)   # unweighted would be 70.0
        self.assertEqual(rec["n"], 2)
        self.assertEqual(rec["panel"]["kind"], "mini")

    def test_panel_leaf_ignores_items_outside_the_panel(self):
        """A leaf may hold more than the panel (the GT baselines are scored on the
        full task and filtered down here), so extra uids must not leak into the mean."""
        wers = {"uttid_1": 0.10, "uttid_2": 0.50, "uttid_9": 1.00}
        panel = {"kind": "mini", "source_task": "clean", "selector": "s",
                 "w": {"en": {"uttid_1": 0.5, "uttid_2": 0.5}}}
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.br.collect_panel_leaf("gt_clean", "clean_mini", "en",
                                             self._leaf(tmp, wers, {}), panel)
        self.assertEqual(rec["n"], 2)
        self.assertAlmostEqual(rec["wer"][0], 30.0, places=6)

    def test_unscored_language_yields_no_row(self):
        panel = {"kind": "mini", "source_task": "clean", "selector": "s",
                 "w": {"en": {"uttid_1": 1.0}}}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "leaf").mkdir()
            self.assertIsNone(self.br.collect_panel_leaf(
                "m", "clean_mini", "en", str(Path(tmp) / "leaf"), panel))
            self.assertIsNone(self.br.collect_panel_leaf(  # language not in the panel
                "m", "clean_mini", "zz", str(Path(tmp) / "leaf"), panel))


class FullSetMacroTest(unittest.TestCase):
    """build_results.full_aggregates: the full sets' ALL row is the macro over the set's
    frozen language basket, defined only where a system covers EVERY basket language —
    a mean over a different language subset is a different quantity."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / "scripts" / "results_doc"))
        import build_results
        from utils.benchmark import ITW_LANGS
        cls.br, cls.langs = build_results, ITW_LANGS

    def _rec(self, model, task, lang, i):
        r = self.br.new_rec(model, task, lang)
        r.update(wer=[2.0 * i, 1.0 * i, 1.5 * i], spk_xutt=50.0 + i, msr_utmos=3.0, n=10)
        return r

    def test_macro_needs_every_language(self):
        task = "in-the-wild"
        recs = [self._rec("full", task, l, i) for i, l in enumerate(self.langs)]
        recs[0]["msr_utmos"] = None                       # one language lacks MSR
        recs += [self._rec("part", task, l, i) for i, l in enumerate(self.langs[1:])]
        recs += [self._rec("gt_in-the-wild", "", l, i) for i, l in enumerate(self.langs)]
        out = {r["model"]: r for r in self.br.full_aggregates(recs) if r["task"] in (task, "")}

        mean_i = (len(self.langs) - 1) / 2
        full = out["full"]
        self.assertEqual(full["set"], "ALL")
        self.assertEqual(full["wer"], [round(2 * mean_i, 3), round(mean_i, 3), round(1.5 * mean_i, 3)])
        self.assertAlmostEqual(full["spk_xutt"], 50.0 + mean_i, places=4)
        self.assertIsNone(full["msr_utmos"])              # not a mean over 19 languages
        self.assertEqual(full["macro"], {"langs": 20, "of": 20})
        self.assertEqual(full["n"], 200)

        part = out["part"]
        self.assertEqual(part["macro"], {"langs": 19, "of": 20})
        self.assertIsNone(part["wer"])
        self.assertIsNone(part["spk_xutt"])

        self.assertEqual(out["gt_in-the-wild"]["task"], "")   # flat, like its rows
        self.assertIsNotNone(out["gt_in-the-wild"]["wer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
