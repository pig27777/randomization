"""I-27 scene composition and isolation of caller-owned tendons and contacts."""
import unittest
from dataclasses import replace

import mujoco
import numpy as np

from model import ANCHORS, build
from mj_adapter import Sim, scene


class SceneExtensionsTest(unittest.TestCase):
    def test_default_and_explicit_solver(self):
        desc = build(ANCHORS["hard"])
        for name in (None, "CG", "Newton"):
            sim = Sim(scene(desc, (0, 0, .5), solver=name))
            expected = mujoco.mjtSolver.mjSOL_CG if name == "CG" else mujoco.mjtSolver.mjSOL_NEWTON
            self.assertEqual(sim.m.opt.solver, expected)
            self.assertEqual(sim.m.opt.iterations, 100)
            self.assertEqual(sim.m.opt.tolerance, 1e-8)
        with self.assertRaisesRegex(ValueError, "unknown MuJoCo solver"):
            scene(desc, (0, 0, .5), solver="invalid")

    def test_table_contact_switch_preserves_table_and_self_contact(self):
        for cat in ("hard", "bendable"):
            desc = build(ANCHORS[cat])
            for enabled in (True, False):
                sim = Sim(scene(desc, (0, 0, ANCHORS[cat].size[2] / 2 - .001),
                                table_contact=enabled))
                self.assertGreaterEqual(sim.table_geom, 0)
                self.assertEqual(bool(sim.d.ncon), enabled)
                for g in sim.obj_geoms:
                    self.assertEqual(int(sim.m.geom_conaffinity[g]) & 1, int(enabled))
                    self.assertEqual(int(sim.m.geom_conaffinity[g]) & 2, 2)

    def test_pad_switch_and_caller_contact(self):
        desc = build(ANCHORS["hard"])
        pads = [("pA", 1, (0, -.17 / 2 - .003, .5))]
        for enabled in (True, False):
            sim = Sim(scene(desc, (0, 0, .5), pads, table=False, pad_contact=enabled))
            self.assertEqual(sim.m.npair, 5 * len(desc.segments) if enabled else 0)
            self.assertEqual(bool(sim.d.ncon), enabled)
        sim = Sim(scene(desc, (0, 0, .5), pads, table=False, pad_contact=False,
                        contact='<pair geom1="pA_core" geom2="s0"/>'))
        self.assertEqual(sim.m.npair, 1)

    def test_all_fragments_with_external_tendon_and_mobile_mass(self):
        worldbody = '''<body name="extra" pos="1 0 .5"><joint name="extra_slide" type="slide" axis="1 0 0"/>
          <geom name="extra_geom" size=".01" mass=".01" contype="0" conaffinity="0"/>
          <site name="extra_site"/></body><site name="extra_anchor" pos="1.1 0 .5"/>'''
        tendon = '''<spatial name="extra_tendon" stiffness="100" springlength=".05">
          <site site="extra_site"/><site site="extra_anchor"/></spatial>'''
        equality = '<joint name="extra_eq" joint1="extra_slide" polycoef="0 1 0 0 0" active="false"/>'
        for cat in ("hard", "bendable"):
            k = ANCHORS[cat]
            if cat == "bendable":
                k = replace(k, mobile=True, mobile_frac=.1, mobile_range=.005)
            desc = build(k)
            plain = Sim(scene(desc, (0, 0, .5), table=False, gravity=False))
            built = scene(desc, (0, 0, .5), table=False, gravity=False,
                          worldbody=worldbody, tendon=tendon, equality=equality)
            sim = Sim(built)
            self.assertEqual(sim.m.ntendon, len(built.tendons) + 1)
            self.assertEqual(sim.m.equality("extra_eq").active0, False)
            self.assertAlmostEqual(sim.mass, plain.mass)
            self.assertAlmostEqual(sim.elastic_energy(), plain.elastic_energy())
            self.assertGreater(sim.d.ten_length[sim.m.tendon("extra_tendon").id], .09)
            np.testing.assert_array_equal(sim.content_tendons, np.arange(len(built.tendons)))
            sim.step()
            self.assertIsNone(sim.failure())


if __name__ == "__main__":
    unittest.main()
