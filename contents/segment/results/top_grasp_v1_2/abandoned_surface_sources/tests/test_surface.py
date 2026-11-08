"""Local surface mass moments, native dynamics, and YAM hard-object range."""
from dataclasses import replace
import unittest

import mujoco
import numpy as np

from model import ANCHORS, build
from mj_adapter import OMEGA_MAX, Sim, scene
from sampling import sample_kappa, sample_rj


def moments(desc):
    mass = sum(s.mass for s in desc.segments)
    com = sum(s.mass * (s.pos + s.com) for s in desc.segments) / mass
    inertia = np.zeros((3, 3))
    for s in desc.segments:
        r = s.pos + s.com - com
        inertia += np.diag(s.inertia) + s.mass * (r @ r * np.eye(3) - np.outer(r, r))
    return mass, com, inertia


class SurfaceTest(unittest.TestCase):
    def test_partition_preserves_mass_moments(self):
        for category, k in ANCHORS.items():
            for offset in ((0., 0., 0.), (.1, -.1, .1)):
                with self.subTest(category=category, offset=offset):
                    base = build(replace(k, com_offset=offset), omega_max=OMEGA_MAX)
                    fine = build(replace(k, com_offset=offset, surface=True), omega_max=OMEGA_MAX)
                    for a, b in zip(moments(base), moments(fine)):
                        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
                    self.assertEqual(len(fine.joints), len(base.joints))
                    if not k.rigid:
                        for link in fine.surface_links:
                            s = fine.segments[link.child]
                            self.assertTrue(np.all(2 * s.half[:2] >= .015))
                            self.assertTrue(np.all(2 * s.half[:2] <= .020 + 1e-12))
                            self.assertAlmostEqual(2 * s.half[2], min(.020, k.size[2] / 3))

    def test_thin_three_layer_and_native_surface(self):
        k = replace(ANCHORS['bendable'], segments=(6, 3, 3), size=(.24, .16, .03), surface=True)
        # Internal parameters are explicit for this geometry-only boundary test.
        desc = build(k, internal=dict(E_y=10000., E_z=800., D=.2), omega_max=OMEGA_MAX)
        self.assertTrue(all(s.mass > 0 and min(s.inertia) > 0 for s in desc.segments))
        sim = Sim(scene(desc, (0, 0, .5), table=False, gravity=False))
        self.assertEqual(sim.surface.q.shape, (len(desc.surface_links), 3))
        link = desc.surface_links[len(desc.surface_links) // 2]
        b = sim.m.body(f's{link.child}').id
        p = sim.m.body(f's{link.parent}').id
        self.assertEqual(sim.m.body_parentid[b], p)
        qz = sim.surface.q[len(desc.surface_links) // 2, 2]
        sim.d.qpos[qz] = -.001
        mujoco.mj_forward(sim.m, sim.d)
        self.assertGreater(sim.elastic_energy(), 0)
        for _ in range(5):
            sim.step()
        self.assertIsNone(sim.failure())
        self.assertTrue(np.all(sim.d.qfrc_applied == 0), 'explicit extra force must not accumulate')

    def test_hard_range_and_rj_pairing(self):
        for seed in range(200):
            k = sample_kappa(np.random.default_rng(seed), 'hard')
            rj = sample_rj(np.random.default_rng(seed), 'hard')
            self.assertLessEqual(min(k.size[:2]), .080)
            self.assertEqual(k, rj.kappa)
            self.assertTrue(k.rigid)
            self.assertEqual(rj.surface_links, [])
        for category in ('bendable', 'compressible'):
            k = sample_kappa(np.random.default_rng(20261118), category)
            self.assertTrue(k.surface)


if __name__ == '__main__':
    unittest.main()
