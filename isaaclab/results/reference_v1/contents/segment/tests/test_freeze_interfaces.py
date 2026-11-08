"""Frozen interfaces: category, all sampling constraints, RJ pairing and diagnostics."""
import unittest
from dataclasses import replace

import mujoco
import numpy as np

from model import ANCHORS, analytic_internal, build, internal_params, load_table
from mj_adapter import Sim, scene
from sampling import CATEGORIES, check_kappa, sample_kappa, sample_rj, rj_internal_bounds


class FreezeInterfacesTest(unittest.TestCase):
    def test_specified_category_and_constraints(self):
        for category in CATEGORIES:
            rng = np.random.default_rng(20261106)
            for _ in range(100):
                k = sample_kappa(rng, category)
                self.assertEqual(k.category, category)
                self.assertEqual(check_kappa(k, category), [])
                self.assertEqual(k.rigid, category == "hard")
                self.assertEqual(type(k).from_dict(k.to_dict()), k)

    def test_invalid_boundary_inputs(self):
        k = ANCHORS["bendable"]
        for invalid in (replace(k, mass=-1), replace(k, size=(0, .1, .1)),
                        replace(k, damping_ratio=float("nan")), replace(k, segments=(0, 3, 2)),
                        replace(k, mobile=True, mobile_frac=1.1), replace(k, category="hard")):
            self.assertTrue(check_kappa(invalid))

    def test_outside_new_mapping_is_rejected(self):
        k = ANCHORS["bendable"]
        entry = dict(mapping_version="test", squeeze_y=[[.4, 1], [.6, 1]],
                     squeeze_z=[[.2, 1], [.4, 1]], droop=[[.2, 1], [.4, 1]])
        table = {"bendable:6x3x2": entry}
        internal_params(k, table)
        with self.assertRaisesRegex(ValueError, "outside calibration"):
            build(replace(k, squeeze_stiffness_y=.61), table=table)
        self.assertTrue(check_kappa(replace(k, squeeze_stiffness_y=.61), table=table))

    def test_rj_geometry_mass_pairing_and_log_ranges(self):
        table = load_table()
        for cat in CATEGORIES:
            for seed in range(10):
                mapped = sample_kappa(np.random.default_rng(seed), cat, table)
                direct = sample_rj(np.random.default_rng(seed), cat, table)
                self.assertEqual(mapped, direct.kappa)
                desc = build(mapped, table=table)
                for a, b in zip(desc.segments, direct.segments):
                    self.assertEqual(a.mass, b.mass)
                    np.testing.assert_array_equal(a.pos, b.pos)
                    np.testing.assert_array_equal(a.inertia, b.inertia)
                if cat != "hard":
                    for name, (lo, hi) in rj_internal_bounds(table, cat).items():
                        self.assertLessEqual(lo, direct.internal[name])
                        self.assertLessEqual(direct.internal[name], hi)
                    again = sample_rj(np.random.default_rng(seed), cat, table)
                    self.assertEqual(direct.internal, again.internal)

    def test_speed_only_is_diagnostic_but_overlap_fails(self):
        k = ANCHORS["bendable"]
        desc = build(k, internal=analytic_internal(k))
        sim = Sim(scene(desc, (0, 0, .5), table=False, gravity=False))
        b = sim.seg_bodies[0]
        j = sim.m.body_jntadr[b]
        sim.d.qvel[sim.m.jnt_dofadr[j]] = 2.0
        mujoco.mj_forward(sim.m, sim.d)
        self.assertIsNone(sim.failure())
        self.assertGreater(sim.velocity_diagnostic()["com_relative_m_s"], 1.)
        edge = desc.joints[0]
        q = sim.m.jnt_qposadr[sim.m.body_jntadr[sim.seg_bodies[edge.b]]]
        sim.d.qpos[q + edge.axis] -= .55 * edge.rest
        mujoco.mj_forward(sim.m, sim.d)
        self.assertIn("overlap", sim.failure())


if __name__ == "__main__":
    unittest.main()
