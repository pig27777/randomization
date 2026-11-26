"""Regression for force and contact masks across independently stepped worlds."""

import unittest

import numpy as np
import warp as wp

from mujoco_warp._src.types import OverflowType, vec5
from policy_v3.yam_gpu import apply_pad_film_contact, control_after_solve, find_bad_contact, sum_contact_forces
from yam_platform.yam_warp_kernels import Commands, Geometry, Mat66


def device(values, dtype):
    return wp.array(np.asarray(values), dtype=dtype, device="cpu")


class YamSubstepTest(unittest.TestCase):
    def test_soft_depth_allowance_preserves_rigid_collision_and_world_boundary(self):
        # IDs are base, pad, wall, table and core. Include both contact endpoint orders.
        geom = device([[2, -1], [-1, 3], [2, -1], [-1, 3], [1, -1], [1, 2], [4, 2], [4, 3]], wp.vec2i)
        flex = device([[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [-1, -1], [0, -1], [0, -1]], wp.vec2i)
        bad = device([20] * 7, int)
        minimum = device(np.full((7, 2), np.inf), float)
        wp.launch(find_bad_contact, len(geom), [
            device([len(geom)], int), device([0, 0, 1, 2, 3, 4, 5, 6], int), geom, flex,
            device([-.00200164, -.005, -.005001, -.005001, -.002001, -.000501, -.001001, -.001001], float),
            device([2, 1, 3, 3, 0], int), device([0, -1, 1, 2, 0], int),
            3, .001, bad, minimum,
        ], device="cpu")
        np.testing.assert_array_equal(bad.numpy(), [20, 2, 3, 4, 5, 6, 7])
        np.testing.assert_allclose(minimum.numpy(), [[-.00200164, -.005], [-.005001, np.inf],
                                                     [np.inf, -.005001], *([[np.inf, np.inf]] * 4)])

    def test_force_uses_contact_world_and_detects_even_small_positive_direct_load(self):
        geom = device([[0, -1], [1, -1], [0, 4], [0, -1], [2, 3], [0, 1]], wp.vec2i)
        flex = device([[-1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, -1], [-1, 0]], wp.vec2i)
        loads = wp.zeros((2, 5), dtype=float, device="cpu")
        direct = device([20, 20], int)
        wp.launch(sum_contact_forces, len(geom), [
            device([len(geom)], int), device([0, 1, 0, 1, 0, 0], int), geom, flex,
            device([[0], [0], [1], [1], [2], [3]], int),
            device([-.0006, -.0004, -.0003, .0002, -.01, -.00002], float),
            device([[.9, 1e-7, 99., 7.], [8.2, 2., 0., 0.]], float),
            device([1, 1, 2, 3, 0], int), 3, device([0, 1], int), 4, loads, direct,
            wp.zeros(2, dtype=int, device="cpu"),
        ], device="cpu")
        np.testing.assert_allclose(loads.numpy(), [
            [7.9000001, .9, 0., -.0006, 1e-7], [10.2, 2., 8.2, -.0004, 0.],
        ], rtol=1e-6, atol=1e-9)
        np.testing.assert_array_equal(direct.numpy(), [2, 20])

    def test_stale_rigid_contact_tags_do_not_change_material(self):
        original = [[.02, 1.]] * 3
        solref = device(original, wp.vec2)
        friction = device(original, wp.vec2)
        impedance = wp.zeros(3, dtype=vec5, device="cpu")
        wp.launch(apply_pad_film_contact, 3, [
            device([3], int), device([[0, 1], [0, -1], [-1, 1]], wp.vec2i),
            device([[-1, 0], [-1, 0], [0, -1]], wp.vec2i), device([0, 1], int),
            wp.vec2(.004, 1.), vec5(.95, .999, .0003, .5, 2.), solref, friction, impedance,
        ], device="cpu")
        np.testing.assert_allclose(solref.numpy(), [[.02, 1.], [.004, 1.], [.004, 1.]])
        np.testing.assert_allclose(friction.numpy(), solref.numpy())
        np.testing.assert_allclose(impedance.numpy()[0], 0.)


    def test_probe_unloads_and_grasp_freezes_or_opens_independently(self):
        worlds = 6
        robot = Geometry()
        robot.joints = device(np.arange(6), int)
        robot.pos = device(np.arange(8), int)
        robot.lower = device([-3.] * 6 + [0.], float)
        robot.upper = device([3.] * 6 + [.095], float)
        robot.max_gap = .095
        robot.site = 0
        commands = Commands()
        for name in ("requested", "target", "velocity", "desired_velocity", "feedforward", "action"):
            setattr(commands, name, device(np.full((worlds, 7), .01), float))
        q = np.zeros((worlds, 8))
        q[:, 6:] = .03
        safe_q = device(np.zeros((worlds, 6)), float)
        jac = device(np.tile(np.diag([4., 4., 4., 1., 1., 1.]), (worlds, 1, 1)), Mat66)
        zeros = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device="cpu")
        state = zeros((worlds, 6))
        trigger, jaw, stops = zeros((worlds, 3)), zeros(worlds, int), zeros(worlds, int)
        counts = zeros((worlds, 2), int)
        xyz = zeros((worlds, 6), wp.vec3)
        loads = [[1.85, .95, .90, -.0005, 0], [8.3, 8.3, 0, -.0007, 0],
                 [11.2, 11.2, 0, -.0008, 0], [1.9, 1.9, 0, -.0004, 0], [0.] * 5, [0.] * 5]
        limits = [[3.5, 0, 1.8], [130, 25, 0], [130, 25, 0], [3.5, 0, 1.8], [3.5, 0, 0], [3.5, 0, 0]]
        overflow = device([0, 0, 0, 0, OverflowType.ITERATIONS, OverflowType.ITERATIONS | OverflowType.NEFC], int)
        wp.launch(control_after_solve, worlds, [
            robot, commands, device(q, float), xyz, xyz, xyz, zeros(worlds), .03,
            device(loads, float), device(limits, float), safe_q, jac, device([1, 0, 0, 0, 0, 0], int),
            state, trigger, jaw, zeros((worlds, 5)), counts, zeros(worlds, int), zeros(worlds, int),
            zeros(1, wp.vec2i), zeros(1), overflow, stops,
            zeros((worlds, 4), int), zeros((worlds, 3)),
        ], device="cpu")
        np.testing.assert_allclose(state.numpy()[0], [0., 0., .003, 0., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(commands.target.numpy()[:, 6], [.064, .060, .0604, .010, .010, .010], atol=1e-7)
        np.testing.assert_allclose(trigger.numpy()[:, 0], [1.85, 0., 0., 0., 0., 0.], atol=1e-6)
        np.testing.assert_array_equal(counts.numpy(), [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(stops.numpy(), [0, 0, 0, 10, 11, 1])


if __name__ == "__main__":
    unittest.main()
