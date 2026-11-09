"""Integration checks for frozen contents and dynamic-palm point attachments."""
import unittest

import mujoco
import numpy as np

from segmented import Config, SegmentPinch, scene
import test_platform


class SegmentedAdapterTest(unittest.TestCase):
    def test_resized_geometry_and_mass_reach_the_physics_model(self):
        for size, mass in (((.18,.12,.0375),.12234375), ((.32,.12,.05),.29),
                           ((.20,.24,.04),.29), ((.16,.15,.08),.29)):
            with self.subTest(size=size):
                sim = scene(Config(size=size, mass=mass, yaw=45)).sim
                centers = sim.seg_pos()
                half = np.array([s.half for s in sim.desc.segments])
                np.testing.assert_allclose((centers+half).max(0)-(centers-half).min(0), size, atol=1e-12)
                self.assertAlmostEqual(sim.mass, mass)
                self.assertEqual(sim.bushings.n, 72)
        for size in ((0,.1,.1), (.1,.1), (float('nan'),.1,.1)):
            with self.assertRaises(ValueError):
                Config(size=size)

    def test_released_models_keep_mass_and_connections(self):
        for category, count, connections, mass in (
            ('bendable', 36, 72, .29), ('compressible', 48, 104, .5)
        ):
            with self.subTest(category=category):
                robot = scene(Config(category=category))
                sim = robot.sim
                self.assertEqual(len(sim.seg_bodies), count)
                self.assertEqual(sim.bushings.n, connections)
                self.assertAlmostEqual(sim.mass, mass)
                self.assertAlmostEqual(sim.m.opt.timestep, .0004)
                self.assertEqual(sim.m.nmocap, 0)
                self.assertEqual(sim.m.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
                for _ in range(50):
                    sim.step()
                self.assertIsNone(sim.failure())

    def test_binding_preserves_material_and_world_anchor(self):
        robot = scene(Config(yaw=45))
        sim, pinch = robot.sim, SegmentPinch(robot)
        point = sim.d.xpos[sim.seg_bodies[pinch.top[0]]] + [0, 0, .005]
        before = sim.d.qpos.copy()
        pinch.set_frame(point)
        pinch.bind_segments(pinch.top[:2])
        np.testing.assert_array_equal(before, sim.d.qpos)
        for eq in pinch.active:
            body1, body2 = sim.m.eq_obj1id[eq], sim.m.eq_obj2id[eq]
            self.assertEqual(body1, pinch.frame)
            a = sim.d.xpos[body1] + sim.d.xmat[body1].reshape(3, 3) @ sim.m.eq_data[eq, :3]
            b = sim.d.xpos[body2] + sim.d.xmat[body2].reshape(3, 3) @ sim.m.eq_data[eq, 3:6]
            np.testing.assert_allclose(a, point, atol=1e-12)
            np.testing.assert_allclose(b, point, atol=1e-12)
        pinch.release()
        self.assertFalse(sim.d.eq_active[list(pinch.equations.values())].any())

    def test_actual_reaction_direction_and_release(self):
        fixture = test_platform.LoadProjectionTest()
        for gravity, expected in ((9.81, False), (-9.81, True)):
            pinch = SegmentPinch.__new__(SegmentPinch)
            pinch.__dict__.update(fixture.attachment(gravity).__dict__)
            released = pinch.check_load(.001, 1.)
            self.assertEqual(released, expected)
        pinch = SegmentPinch.__new__(SegmentPinch)
        pinch.__dict__.update(fixture.attachment(0.).__dict__)
        self.assertTrue(pinch.check_load(.001, 1., np.array([10., 0., 100.])))


if __name__ == '__main__':
    unittest.main()
