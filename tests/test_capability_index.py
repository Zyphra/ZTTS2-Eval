"""Capability index fit (scripts/results_doc/build_capability_index.py).

Noise-free synthetic scores drawn from the model itself, so the expected placement
is known exactly.

    .venv/bin/python -m unittest tests.test_capability_index
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "results_doc"))

import build_capability_index as ci  # noqa: E402


def synthetic(capability, coverage):
    """(system, item, dimension, performance) from sigmoid(a·(C − D)); `coverage`
    maps a system to the item indices it was scored on (default: all)."""
    rng = np.random.default_rng(0)
    n_items = 48
    difficulty = np.linspace(-2.5, 2.5, n_items)
    slope = rng.uniform(0.6, 1.8, n_items)
    obs = []
    for s, c in capability.items():
        for i in coverage.get(s, range(n_items)):
            p = 1 / (1 + np.exp(-slope[i] * (c - difficulty[i])))
            obs.append((s, f"m{i % 4}|t|l{i}", ci.DIMENSIONS[i % 4], float(p)))
    return obs, difficulty


class CapabilityIndexFitTest(unittest.TestCase):
    def fit(self, obs, anchors):
        P = ci.Problem(obs)
        C = P.unpack(P.fit(P.si, P.ii, P.y, P.w, P.x0()))[0]
        a, b = ci.anchor_map(P.systems, C, anchors)
        return dict(zip(P.systems, a + b * C))

    def test_partial_coverage_is_placed_by_item_difficulty_not_raw_mean(self):
        capability = {"s0": -1.5, "s1": -0.5, "s2": 0.5, "s3": 1.5, "s4": 2.5}
        obs, difficulty = synthetic(capability, {})
        easy = [i for i in range(len(difficulty)) if difficulty[i] < 0]
        # s2 is scored only on the easy half: its raw mean beats s3's full-set mean
        obs = [o for o in obs if o[0] != "s2" or int(o[1].rsplit("l", 1)[1]) in easy]
        raw = {s: np.mean([o[3] for o in obs if o[0] == s]) for s in capability}
        self.assertGreater(raw["s2"], raw["s3"])

        index = self.fit(obs, (("s4", 100.0), ("s0", 50.0)))
        self.assertEqual(sorted(index, key=index.get), ["s0", "s1", "s2", "s3", "s4"])
        self.assertAlmostEqual(index["s4"], 100.0, places=9)
        self.assertAlmostEqual(index["s0"], 50.0, places=9)
        # equally spaced true capabilities stay roughly equally spaced on the index
        gaps = np.diff([index[s] for s in ("s0", "s1", "s2", "s3", "s4")])
        self.assertLess(gaps.max() / gaps.min(), 1.5)

    def test_inverted_anchors_refuse_to_define_a_scale(self):
        obs, _ = synthetic({"lo": -1.0, "hi": 1.0}, {})
        with self.assertRaises(SystemExit):
            self.fit(obs, (("lo", 100.0), ("hi", 50.0)))


class CharSecItemsTest(unittest.TestCase):
    def test_char_s_counts_only_leaves_the_snapshot_scored(self):
        # A partial generation that build_results' coverage gate dropped still gets a
        # speaking-rate series; it must not credit the system with that language.
        sr = {"panels": [{"key": "zero_shot|ar", "set": "zero_shot", "lang": "ar", "series": [
            {"name": "gt_zero_shot", "d_ref": 0.2, "x_ceiling": 1.0, "split": "random 50/50"},
            {"name": "partial", "d_ref": 0.4, "x_ceiling": 1.5}]}]}
        rows = [{"model": "partial", "task": "zero_shot", "set": "en"},
                {"model": "partial", "task": "zero_shot", "set": "ALL"}]
        names = {o[0] for o in ci.charsec_observations(sr, ci.scored_leaves(rows))}
        self.assertEqual(names, {ci.REAL})


if __name__ == "__main__":
    unittest.main()
