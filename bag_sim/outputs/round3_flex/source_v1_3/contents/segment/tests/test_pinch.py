"""Effect-rule eligibility, native point-connection loads (v1.3), and duration-based release."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from model import ANCHORS, build
from mj_adapter import OMEGA_MAX, Sim, scene
from pinch import DEPTH_MIN, LOAD_FILTER_TAU, PinchAttachment, PinchSpring, fragments
from sampling import check_kappa, sample_kappa, sample_rj


def fixture(category='bendable'):
    k = ANCHORS[category]
    desc = build(k, omega_max=OMEGA_MAX)
    body, eq = fragments(desc)
    sim = Sim(scene(desc, (0, 0, .2), table=False, gravity=False, worldbody=body, equality=eq))
    return sim, PinchAttachment(sim, .2 + k.size[2] / 2)


class PinchTest(unittest.TestCase):
    def test_real_contact_and_spring_force_add(self):
        from top_grasp import real_grip_force
        # Native compressive reactions are read even before the effect is armed.
        pads = {'pA': np.array([0., -8., 4.]), 'pB': np.array([0., 12., 3.])}
        spring = PinchSpring()
        self.assertEqual(real_grip_force(pads) + spring.force(.04), 10.)
        spring.arm(.04, True)
        self.assertAlmostEqual(real_grip_force(pads) + spring.force(.029875), 15.)

    def test_first_contact_starts_press_with_only_one_finger(self):
        from top_grasp import evaluate
        class ObservedFirstContact(Exception):
            pass
        contacts = []
        def observe(sim, binding, phase, t, force):
            if phase in ('press', 'close'):
                contacts.extend(np.linalg.norm(v) for v in sim.contact_forces()[0].values())
                raise ObservedFirstContact
        with self.assertRaises(ObservedFirstContact):
            evaluate(ANCHORS['bendable'], dict(v=.45, w=.08, d=.03, F=15.), observer=observe)
        self.assertGreater(max(contacts), .05)
        self.assertLess(min(contacts), .05)

    def test_native_table_contact_and_clearance(self):
        from top_grasp import task_scene, TopGrasp, table_contacts
        sim, _, _ = task_scene(ANCHORS['hard'], TopGrasp(), 0.)
        for _ in range(100):
            sim.step()
        self.assertTrue(table_contacts(sim))
        for j in range(sim.m.njnt):
            if sim.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                sim.d.qpos[sim.m.jnt_qposadr[j] + 2] += .5
        mujoco.mj_forward(sim.m, sim.d)
        self.assertFalse(table_contacts(sim))

    def test_native_top_contacts_and_yam_dimensions(self):
        from top_grasp import task_scene, TopGrasp, WIDTH, HEIGHT
        sim, starts, top = task_scene(ANCHORS['bendable'], TopGrasp(), 0.)
        for p, start in starts.items():
            sim.set_pad(p, start - [0, 0, .008], np.zeros(3))
        mujoco.mj_forward(sim.m, sim.d)
        binding = PinchAttachment(sim, top)
        self.assertTrue(all(binding.top_contacts()))
        tips = np.array([sim.pad_pos(p) - [0, 0, HEIGHT / 2] for p in starts])
        self.assertTrue(binding.try_bind(tips))
        core = sim.m.geom('pA_core').size
        np.testing.assert_allclose(core + [.002, 0, .002], [WIDTH / 2, .002, HEIGHT / 2])

    def test_spring_requires_eligibility_and_10mm_gathering(self):
        spring = PinchSpring()
        spring.arm(.04, False)
        self.assertEqual(spring.force(.020), 0.)
        spring.arm(.04, True)
        self.assertEqual(spring.force(.040), 0.)
        self.assertEqual(spring.force(.031), 0.)
        self.assertAlmostEqual(spring.force(.02975), 10.)
        self.assertAlmostEqual(spring.force(.02950), 20.)
        spring.release()
        self.assertEqual(spring.force(.020), 0.)

    def test_eligibility_depth_and_both_contacts(self):
        sim, p = fixture()
        i = min(p.top, key=lambda j: np.linalg.norm(sim.d.xpos[sim.seg_bodies[j], :2]))
        tips = np.array([[0., -.01, p.original_top - DEPTH_MIN], [0., .01, p.original_top - DEPTH_MIN]])
        before = sim.d.qpos.copy()
        with patch.object(p, 'top_contacts', return_value=({i}, {i})):
            shallow = tips.copy(); shallow[0, 2] += 1e-6
            self.assertFalse(p.try_bind(shallow))
        with patch.object(p, 'top_contacts', return_value=({i}, set())):
            self.assertFalse(p.try_bind(tips))
        with patch.object(p, 'top_contacts', return_value=({i}, {i})):
            self.assertTrue(p.try_bind(tips))
        self.assertTrue(sim.d.eq_active[p.active].all())
        self.assertIn(p.segment, p.top)
        np.testing.assert_array_equal(sim.d.qpos, before, 'activation must not teleport the content')
        mujoco.mj_forward(sim.m, sim.d)
        mask = (sim.d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(sim.d.efc_id, p.active)
        self.assertEqual(mask.sum(), 3 * len(p.active), 'point connections constrain position only')
        np.testing.assert_allclose(sim.d.efc_pos[mask], 0., atol=1e-12)
        p.release()
        self.assertFalse(any(sim.d.eq_active[list(p.equations.values())]))

    def test_connections_span_a_seam_and_share_the_weight(self):
        # The anchor's x has 6 columns: the top-centre pinch line lies on the seam of two segments.
        k = ANCHORS['bendable']
        desc = build(k, omega_max=OMEGA_MAX)
        body, eq = fragments(desc)
        sim = Sim(scene(desc, (0, 0, .2), table=False, gravity=True, worldbody=body, equality=eq))
        p = PinchAttachment(sim, .2 + k.size[2] / 2)
        tips = np.array([[0., -.02, p.original_top - .005], [0., .02, p.original_top - .005]])
        near = min(p.top, key=lambda j: np.linalg.norm(sim.d.xpos[sim.seg_bodies[j], :2]))
        with patch.object(p, 'top_contacts', return_value=({near}, {near})):
            self.assertTrue(p.try_bind(tips))
        self.assertEqual(len(p.segments), 2)
        self.assertEqual({sim.desc.segments[i].idx[1] for i in p.segments}, {1})
        self.assertEqual(sorted(sim.desc.segments[i].idx[0] for i in p.segments), [2, 3])
        np.testing.assert_allclose(sim.d.mocap_pos[p.mocap], tips.mean(0))
        for _ in range(6250):   # the hanging droop settles within about 2 s
            sim.step()
        weight = sim.mass * 9.81
        self.assertLess(abs(-p.force_vector()[2] / weight - 1), .05)
        rows = np.flatnonzero((sim.d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(sim.d.efc_id, p.active))
        shares = -sim.d.efc_force[rows].reshape(-1, 3)[:, 2] / weight
        self.assertTrue(np.all(shares > .1), f'both seam segments carry load: {shares}')
        self.assertIsNone(sim.failure())

    def test_band_keeps_both_sides_of_an_opened_seam(self):
        sim, p = fixture()
        tips = np.array([[0., -.02, p.original_top - .005], [0., .02, p.original_top - .005]])
        self.assertEqual(len(p.covered(tips)), 2)
        for i in p.top:   # open the x = 0 seam by 1 mm, as squeezing can
            q = sim.m.jnt_qposadr[sim.m.body_jntadr[sim.seg_bodies[i]]]
            sim.d.qpos[q] += .0005 * np.sign(sim.d.qpos[q])
        mujoco.mj_forward(sim.m, sim.d)
        self.assertEqual(len(p.covered(tips)), 2)

    def test_connection_fixes_the_point_but_not_rotation(self):
        sim, p = fixture()
        i = p.top[0]
        b = sim.seg_bodies[i]
        point = sim.d.xpos[b] + [0, 0, sim.desc.segments[i].half[2]]
        p.set_frame(point)
        p.bind_segment(i)
        local = sim.point_local(b, point)
        q0 = sim.d.xquat[b].copy()
        sim.d.xfrc_applied[b, 3] = .02   # a pure couple about x on the pinched segment
        for _ in range(250):
            sim.step()
        sim.d.xfrc_applied[b, 3] = 0
        turned = 2 * np.degrees(np.arccos(min(1., abs(float(q0 @ sim.d.xquat[b])))))
        self.assertGreater(turned, 2., 'a point connection must not resist rotation')
        self.assertLess(np.linalg.norm(sim.point_world(b, local) - point), 1e-3)

    def test_multi_segment_release_and_tips_outside(self):
        sim, p = fixture()
        top = p.original_top - .005
        near = min(p.top, key=lambda j: np.linalg.norm(sim.d.xpos[sim.seg_bodies[j], :2]))
        with patch.object(p, 'top_contacts', return_value=({near}, {near})):
            outside = np.array([[0., .5, top], [0., .6, top]])
            self.assertIsNone(p.candidate(outside), 'fingertips beyond the object cannot pinch it')
            self.assertTrue(p.try_bind(np.array([[0., -.04, top], [0., .04, top]])))
        self.assertGreater(len(p.active), 1)
        p.release()
        self.assertFalse(any(sim.d.eq_active[list(p.equations.values())]))
        self.assertIsNone(p.active)

    def test_rigid_never_binds(self):
        sim, p = fixture('hard')
        with patch.object(p, 'top_contacts', return_value=({0}, {0})):
            self.assertFalse(p.try_bind(np.array([[0, -.01, .1], [0, .01, .1]])))
        self.assertEqual(p.equations, {})
        with self.assertRaises(ValueError):
            p.bind_segment(0)

    def test_actual_current_force_and_low_passed_release(self):
        sim, p = fixture()
        p.bind_segment(p.top[0])
        dt = sim.m.opt.timestep
        a = dt / (LOAD_FILTER_TAU + dt)
        # A single-step spike to 3x capacity is filtered out.
        with patch.object(p, 'force_vector', return_value=np.array([0., 0., 30.])):
            self.assertFalse(p.check_load(dt, 10.))
        with patch.object(p, 'force_vector', return_value=np.array([0., 0., 5.])):
            for _ in range(100):
                self.assertFalse(p.check_load(dt, 10.))
        # A constant 11 N load against a 10 N capacity releases on the step the low-pass crosses it,
        # whatever the loading history; a larger CURRENT grip force raises the capacity.
        steps, x = 0, p.load
        while x <= 10.:
            x += a * (11. - x); steps += 1
        with patch.object(p, 'force_vector', return_value=np.array([0., 0., 11.])):
            for _ in range(steps - 1):
                self.assertFalse(p.check_load(dt, 10.))
            self.assertFalse(p.check_load(dt, 12.))
            self.assertTrue(p.check_load(dt, 10.))
        self.assertTrue(p.broken)
        self.assertIsNone(p.active)
        self.assertGreater(p.events[-1]['filtered_force'], 10.)
        self.assertEqual(p.events[-1]['limit'], 10.)
        self.assertFalse(p.try_bind(np.zeros((2, 3))), 'a ruptured grasp must not rebind in flight')

    def test_connection_carries_native_load_and_detaches(self):
        sim, p = fixture()
        i = p.top[0]
        p.set_frame(sim.d.xpos[sim.seg_bodies[i]])
        p.bind_segment(i)
        # A load on the selected segment must appear as a solver constraint force.
        sim.d.xfrc_applied[sim.seg_bodies[i], 2] = -3.
        for _ in range(500):
            sim.step()
        np.testing.assert_allclose(p.force_vector(), [0, 0, -3], atol=.08)
        self.assertIsNone(sim.failure())
        for _ in range(50):
            sim.step()
            p.check_load(sim.m.opt.timestep, .5)
        self.assertIsNone(p.active)
        self.assertEqual(p.events[-1]['event'], 'overload')

    def test_parameter_boundary_and_sampling_pair(self):
        for category in ('hard', 'bendable', 'compressible'):
            for seed in range(20):
                k = sample_kappa(np.random.default_rng(seed), category)
                rj = sample_rj(np.random.default_rng(seed), category)
                self.assertEqual(k, rj.kappa)
                self.assertEqual(check_kappa(k), [])
                self.assertEqual(k.pinch_coeff == 0, k.rigid)
        for value in (-1., float('inf'), float('nan')):
            self.assertTrue(check_kappa(replace(ANCHORS['bendable'], pinch_coeff=value)))
        sim, p = fixture()
        for dt, f in ((0, 10), (.0004, -1), (.0004, float('nan'))):
            with self.assertRaises(ValueError):
                p.check_load(dt, f)


if __name__ == '__main__':
    unittest.main()
