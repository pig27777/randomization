"""S2 sampling interface tests (round 2): constraints always hold, categories are respected, speed.

    /home/ubuntu/Dev/Envs/mjwarp/bin/python -m unittest discover -s tests -v      (from contents/segment)
"""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import build, load_table           # noqa: E402
from sampling import CATEGORIES, check_kappa, sample_kappa   # noqa: E402

N = 10000


class SamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_table()
        rng = np.random.default_rng(12345)
        t = time.perf_counter()
        cls.samples = [sample_kappa(rng, "mixed", cls.table) for _ in range(N)]
        cls.rate = N / (time.perf_counter() - t)

    def test_all_constraints_hold(self):
        bad = [(i, check_kappa(k, None, self.table)) for i, k in enumerate(self.samples)]
        bad = [b for b in bad if b[1]]
        self.assertEqual(bad, [], f"{len(bad)} of {N} samples violate constraints, e.g. {bad[:3]}")

    def test_speed(self):
        self.assertGreaterEqual(self.rate, 1000, f"{self.rate:.0f} samples/s")

    def test_categories(self):
        rng = np.random.default_rng(7)
        for cat in CATEGORIES:
            for _ in range(200):
                k = sample_kappa(rng, cat, self.table)
                self.assertEqual(k.rigid, cat == "hard")
                self.assertEqual(check_kappa(k, None, self.table), [])
        mix = [k.rigid for k in self.samples]
        self.assertTrue(0.25 < np.mean(mix) < 0.42, "mixed should be about one third hard")
        with self.assertRaises(ValueError):
            sample_kappa(rng, "liquid", self.table)

    def test_reproducible_and_buildable(self):
        a = sample_kappa(np.random.default_rng(3), "mixed", self.table)
        b = sample_kappa(np.random.default_rng(3), "mixed", self.table)
        self.assertEqual(a, b)
        for k in self.samples[:50]:
            desc = build(k, table=self.table)
            self.assertAlmostEqual(sum(s.mass for s in desc.segments) + (desc.mobile.mass if desc.mobile else 0),
                                   k.mass, places=9)


if __name__ == "__main__":
    unittest.main()
