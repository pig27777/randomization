"""Physical-response and input-boundary checks for the local kappa preview."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parameter_ui import PRESETS, Preview, Settings


class ParameterPreviewTest(unittest.TestCase):
    def test_limits_reject_outside_and_nonfinite_parameters(self):
        for value in (float('nan'), float('inf'), True, -1., 100.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Settings.parse(dict(asdict(Settings()), ky=value))
        with self.assertRaises(ValueError):
            Settings.parse({'unknown': 1})

    def test_both_models_and_all_boundary_modes_step(self):
        for category in PRESETS:
            for mode in ('squeeze_y', 'squeeze_z', 'sag'):
                with self.subTest(category=category, mode=mode):
                    cfg = Settings(category=category, mode=mode, **PRESETS[category]['defaults'])
                    preview = Preview(Settings.parse(asdict(cfg)))
                    for _ in range(10):
                        preview.advance()
                    data = preview.snapshot()
                    self.assertIsNone(data['failure'])
                    self.assertFalse(data['warnings'])
                    self.assertTrue(np.isfinite(data['centers']).all())
                    self.assertAlmostEqual(data['t'], .1)
                    self.assertEqual(len(data['centers']), 36 if category == 'bendable' else 48)

    def test_stiffness_changes_actual_loaded_displacement(self):
        values = []
        for ky in (.12, 1.):
            preview = Preview(replace(Settings(), ky=ky))
            for _ in range(120):
                preview.advance()
            data = preview.snapshot()
            self.assertIsNone(data['failure'])
            self.assertFalse(data['warnings'])
            values.append(data['metrics']['compression_pct'])
        self.assertGreater(values[0], values[1] + 2.)


if __name__ == '__main__':
    unittest.main()
