"""Check the frozen evidence and the Milestone 1 sampling contract."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from build_stage1_envelopes import EXPANSION, MATERIALS, ROOT, UNITS, build


class Stage1EnvelopesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = build()

    def test_artifact_and_provenance(self):
        folder = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proxy_material_envelopes.yaml"
            subprocess.run([sys.executable, "-B", str(folder / "build_stage1_envelopes.py"), "--output", str(output)],
                           stdout=subprocess.DEVNULL, check=True)
            self.assertEqual(output.read_bytes(), (folder / output.name).read_bytes())
            self.assertEqual(json.loads(output.read_bytes()), self.output)
        for path, digest in self.output["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(self.output["sample_counts"], dict(actions=59, action_samples=28607,
                                                          load_curves=14, load_samples=9394))
        self.assertEqual(self.output["geometry"]["sample_count"], 128)
        self.assertEqual(self.output["proxy_contract"]["state_shape"], ["N_envs", 2, 6])

    def test_ranges_and_prior_covariances(self):
        self.assertEqual(list(self.output["materials"]), list(MATERIALS))
        for material, row in self.output["materials"].items():
            with self.subTest(material=material):
                sampling = row["tier1_ranges"]
                self.assertEqual(sampling.keys(), UNITS.keys())
                self.assertTrue(all(0. <= low < high and np.isfinite(high) for low, high in sampling.values()))
                low, high = row["anchors"]["k_N_per_m"]
                np.testing.assert_allclose(sampling["k"], [max(5., low / EXPANSION), min(2000., high * EXPANSION)])
                dependency = row["dependency"]
                self.assertEqual(dependency["order"], list(UNITS))
                self.assertIsNone(dependency["empirical_covariance"])
                covariance = np.asarray(dependency["covariance_si"])
                correlation = np.asarray(dependency["correlation"])
                np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(np.diag(correlation), 1.)
                self.assertGreater(np.linalg.eigvalsh(correlation).min(), 0.)
                self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_measures_are_not_interchangeable(self):
        rows = self.output["materials"]
        self.assertIsNotNone(rows["pe"]["anchors"]["alpha_local"])
        self.assertIsNone(rows["bubble"]["anchors"]["beta_local_m_per_N_s"])
        self.assertIsNone(rows["paperboard"]["anchors"]["beta_local_m_per_N_s"])
        for row in rows.values():
            self.assertIsNone(row["anchors"]["s_crit"])
            self.assertIsNone(row["anchors"]["mu_eff"])
        ridge = self.output["pe_whole_bag_ridge_diagnostic"]
        self.assertNotEqual(ridge["spatial_fold_slope_m_per_m"], ridge["reference_excess_slope_m_per_m"])
        self.assertFalse(ridge["calibrated"])
        release = rows["paperboard"]["anchors"]["release_applied_load_by_axis"]
        self.assertAlmostEqual(release["vertical"][0]["added_N"], 7.7095)
        self.assertEqual(len(release["tangential"]), 2)


if __name__ == "__main__":
    unittest.main()
