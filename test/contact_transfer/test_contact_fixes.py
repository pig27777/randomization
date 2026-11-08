"""Check friction at rest against a force derivative, including the actual contact-law caller."""
import unittest

import numpy as np
import warp as wp
from newton._src.solvers.vbd import rigid_vbd_kernels as rigid

from contact_fixes import enable_friction_zero_fix, projected_friction, surface_friction, mat32

HAS_LOG_BARRIER = 'use_log_barrier' in rigid._compute_body_particle_contact_force.input_types


@wp.kernel
def _sample(displacement: wp.array[wp.vec3], force: wp.array[wp.vec3], stiffness: wp.array[wp.mat33],
            surface_force: wp.array[wp.vec3], surface_stiffness: wp.array[wp.mat33]):
    i = wp.tid()
    u = displacement[i]
    f, h = projected_friction(0.5, 4.0, wp.vec3(0., 0., 1.), u, 1.e-5)
    force[i] = f
    stiffness[i] = h
    f2, h2 = surface_friction(0.5, 4.0, mat32(1., 0., 0., 1., 0., 0.), wp.vec2(u[0], u[1]), 1.e-5)
    surface_force[i] = f2
    surface_stiffness[i] = h2


@wp.kernel
def _caller(stiffness: wp.array[wp.mat33]):
    # Positive penetration and zero displacement: tangent stiffness must survive through the caller.
    if wp.static(HAS_LOG_BARRIER):
        f, h = rigid._compute_body_particle_contact_force(0.003, 0.004, wp.vec3(0., 0., 1.),
            wp.vec3(0.), 1000., 0., 0.5, 0.01, 0.001, False)
    else:
        f, h = rigid._compute_body_particle_contact_force(0.003, 0.004, wp.vec3(0., 0., 1.),
            wp.vec3(0.), 1000., 0., 0.5, 0.01, 0.001)
    stiffness[0] = h


class ContactFixTests(unittest.TestCase):
    def test_planar_quadrature_covers_off_node_tool_and_preserves_moments(self):
        import benchmark as bm
        for case in ('finger_off', 'plate'):
            exp = bm.Experiment(bm.Config(case=case, planar_quadrature=True, drive='position', tool_z_mm=63.))
            exp.s0.body_q.assign(np.array([[*exp.xy, .063, 0., 0., 0., 1.]], np.float32))
            exp.pipeline.collide(exp.s0, exp.contacts)
            count = int(exp.contacts.soft_contact_count.numpy()[0])
            self.assertGreater(count, 0)
            self.assertLessEqual(count, exp.contacts.soft_contact_max)
            ids = exp.contacts.soft_contact_indices.numpy()[:count]
            bary = exp.contacts.soft_contact_barycentric.numpy()[:count]
            weight = exp.solver.quadrature_weights.numpy()[:count].astype(float)
            points = (exp.q0[ids] * bary[..., None]).sum(axis=1)
            area = 4 * exp.half[0] * exp.half[1]
            np.testing.assert_allclose(bary.sum(axis=1), 1., atol=2e-7)
            self.assertTrue((bary >= -1e-7).all())
            self.assertAlmostEqual(weight.sum() * .025**2 / area, 1., places=5)
            for axis in (0, 1):
                self.assertTrue((np.abs(points[:, axis] - exp.xy[axis]) <= exp.half[axis] + 1e-7).all())
                self.assertAlmostEqual(float(weight @ points[:, axis] / weight.sum()), exp.xy[axis], places=6)
                expected = exp.xy[axis]**2 + exp.half[axis]**2 / 3
                self.assertAlmostEqual(float(weight @ points[:, axis]**2 / weight.sum()), expected, places=7)

    def test_zero_slip_matches_force_derivative_and_surface_path(self):
        device = 'cuda:0'
        delta = 1.e-9
        u = wp.array([[0., 0., 0.], [delta, 0., 0.], [-delta, 0., 0.], [2.e-5, 0., 0.]], dtype=wp.vec3, device=device)
        f, fs = [wp.zeros(4, dtype=wp.vec3, device=device) for _ in range(2)]
        h, hs = [wp.zeros(4, dtype=wp.mat33, device=device) for _ in range(2)]
        wp.launch(_sample, 4, inputs=[u, f, h, fs, hs], device=device)
        force, tangent = f.numpy(), h.numpy()
        np.testing.assert_array_equal(force[0], 0.)
        derivative = -(force[1, 0] - force[2, 0]) / (2 * delta)
        self.assertAlmostEqual(tangent[0, 0, 0] / derivative, 1., places=3)
        np.testing.assert_array_equal(tangent[0, 2], 0.)
        np.testing.assert_allclose(force[3], [-2., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(force, fs.numpy(), atol=1e-6)
        np.testing.assert_allclose(tangent, hs.numpy(), atol=1e-2)

    def test_patch_reaches_contact_law(self):
        enable_friction_zero_fix()
        h = wp.zeros(1, dtype=wp.mat33, device='cuda:0')
        wp.launch(_caller, 1, inputs=[h], device='cuda:0')
        np.testing.assert_allclose(h.numpy()[0].diagonal(), [1e5, 1e5, 1000.], rtol=1e-5)


if __name__ == '__main__':
    wp.config.quiet = True
    unittest.main()
