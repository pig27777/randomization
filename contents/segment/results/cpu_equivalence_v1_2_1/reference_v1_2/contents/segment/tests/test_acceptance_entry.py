"""Regression: acceptance entry must keep the same platform auto-lock limit as R1/R2/R3."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import freeze_v1 as acceptance
from model import Kappa, load_table


class AcceptanceEntryTest(unittest.TestCase):
    def test_random_entry_preserves_required_auto_locks(self):
        # Seed-20261106 case 8 diverged when the driver accidentally built with omega_max=inf.
        k = Kappa(category="compressible", size=(.3576288470291566, .2801740881433323, .1645246024937686),
                  mass=.9285564757959872, com_offset=(.1367726338180353, .09352524767098613, -.11986472415549165),
                  segments=(6, 3, 3), squeeze_stiffness_y=.14813222827630426,
                  squeeze_stiffness_z=.2643844322354988, lock_strain_y=.2796768717420721,
                  lock_strain_z=.2996549277197144, droop_ratio=.08829161008992539,
                  damping_ratio=.32363606255592503)
        seen = []
        def protocol(kappa, *, desc):
            seen.append(desc)
            self.assertTrue(any(j.fixed for j in desc.joints), "platform auto locks were lost")
            return {"fail": None, "fail_t": None}
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(dir=acceptance.ROOT / "results") as tmp:
            try:
                with patch.object(acceptance, "rest", protocol), patch.object(acceptance, "r3", protocol):
                    row = acceptance.random_case((0, k, str(Path(tmp) / "case"), load_table()))
                self.assertFalse(row["execution_error"], row.get("traceback"))
                self.assertEqual(len(seen), 2)
                self.assertFalse(row["confirmed_failure"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
