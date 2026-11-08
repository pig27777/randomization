"""Checks the measurement code against energy differences and exact integrals."""
import unittest
import numpy as np
from metrics import pressure_weights, surface_height, tet_response


class MeasurementTests(unittest.TestCase):
    def test_tet_force_matches_energy_gradient(self):
        rest = np.array([[0., 0., 0.], [.04, 0., 0.], [0., .03, 0.], [0., 0., .02]])
        tets = np.array([[0, 1, 2, 3]])
        inv = np.linalg.inv(np.stack([rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]], axis=1))[None]
        mats = np.array([[1500., 1000., 1.]])
        q = rest @ np.array([[1.05, .02, 0.], [0., .98, .04], [0., 0., .85]])
        _, force, j = tet_response(q, tets, inv, mats)
        numerical = np.zeros_like(q)
        for node in range(4):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[node, axis] += 1e-7
                minus[node, axis] -= 1e-7
                numerical[node, axis] = -(tet_response(plus, tets, inv, mats)[0] - tet_response(minus, tets, inv, mats)[0]) / 2e-7
        np.testing.assert_allclose(force, numerical, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(force.sum(axis=0), 0., atol=1e-12)
        self.assertTrue(np.all(j > 0))

    def test_off_grid_pressure_preserves_force_and_moment(self):
        q = np.array([[0., 0., .1], [1., 0., .1], [1., 1., .1], [0., 1., .1]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        weights = pressure_weights(q, faces, (.1, .15), (.4, .3))
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertTrue(np.all(weights >= 0))
        np.testing.assert_allclose(weights @ q[:, :2], [.4, .3], atol=1e-12)

    def test_ray_follows_deformed_surface_and_rejects_outside(self):
        q = np.array([[0., 0., 0.], [1., 0., 1.], [0., 1., 2.]])
        heights = surface_height(q, np.array([[0, 1, 2]]), np.array([[.2, .3], [2., 2.]]))
        self.assertAlmostEqual(heights[0], .8)
        self.assertTrue(np.isnan(heights[1]))


if __name__ == '__main__':
    unittest.main()
