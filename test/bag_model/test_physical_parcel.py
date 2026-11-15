"""Static and force-law regressions for the existing-envelope parcel."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import contents  # noqa: E402,F401  (package import must coexist with legacy helpers)
from bag_sim.physical_parcel import Config, Parcel, material_parameters, compression_force  # noqa: E402
from bag_sim.bag_model import dihedrals  # noqa: E402


class PhysicalParcelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parcel = Parcel()

    def test_existing_topology_and_positive_mass(self):
        p = self.parcel
        with np.load(ROOT / 'bag_sim/geometry/kraft_coarse_35.npz') as original:
            np.testing.assert_array_equal(p.envelope['faces'], original['faces'])
        self.assertEqual(p.m.nflexvert, 200)
        self.assertEqual(p.m.nflexelem, 396)
        self.assertEqual(p.m.neq, 0)
        self.assertTrue(np.all(p.envelope['masses'] > 0))

    def test_reference_geometry_is_independent(self):
        p = self.parcel
        self.assertGreater(np.max(abs(p.envelope['current']-p.envelope['reference'])), .001)
        rest = mujoco.MjData(p.m)
        p.forward(p.m, rest)
        self.assertLess(np.max(abs(rest.qfrc_passive)), 1e-7)

    def test_bending_internal_force_and_torque(self):
        p = self.parcel
        points = p.envelope['current']
        theta, gradients = dihedrals(points, p.hinges)
        delta = np.arctan2(np.sin(theta-p.rest_angles), np.cos(theta-p.rest_angles))
        forces = np.zeros_like(points)
        np.add.at(forces, p.hinges.ravel(), (-p.hinge_stiffness[:,None,None]*delta[:,None,None]*gradients).reshape(-1,3))
        self.assertLess(np.linalg.norm(forces.sum(0)), 1e-10)
        self.assertLess(np.linalg.norm(np.cross(points, forces).sum(0)), 1e-10)

    def test_unilateral_compression(self):
        cfg = Config()
        self.assertAlmostEqual(float(compression_force(.004, 0, cfg)), 3.52)
        self.assertEqual(float(compression_force(0, -1, cfg)), 0)
        self.assertGreater(float(compression_force(.005, 0, cfg)), float(compression_force(.004, 0, cfg)))

    def test_material_parameters_are_correlated(self):
        first = material_parameters(6e6, .00015, 920)
        second = material_parameters(6e6, .0003, 920)
        self.assertAlmostEqual(second['membrane']/first['membrane'], 2)
        self.assertAlmostEqual(second['areal_mass']/first['areal_mass'], 2)
        self.assertAlmostEqual(second['bending']/first['bending'], 8)

    def test_payload_mass_and_torque(self):
        p = Parcel(replace(Config(), mode='bendable'))
        self.assertAlmostEqual(float(p.m.body_mass[p.payload_bodies].sum()), .3)
        joint = p.m.joint('bend_1').id
        p.d.qpos[p.m.jnt_qposadr[joint]] = .1
        mujoco.mj_step1(p.m, p.d)
        self.assertAlmostEqual(float(p.d.qfrc_spring[p.m.jnt_dofadr[joint]]), -.008)

    def test_unresolved_bending_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not resolve'):
            Parcel(replace(Config(), bending=.00135, timestep=.0004))


if __name__ == '__main__':
    unittest.main()
