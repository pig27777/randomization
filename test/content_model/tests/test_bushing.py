"""Physical checks: stiffness, work/energy, objectivity and momentum balance."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "contents" / "segment"))
from model import ANCHORS, ContentDesc, build, load_table
from mj_adapter import Sim, scene, object_mjcf
from bushing import joint_damping


def pair(name, axis):
    d = build(ANCHORS[name], table=load_table(), geometry=name)
    j = next(j for j in d.joints if j.axis == axis)
    sa, sb = d.segments[j.a], d.segments[j.b]
    mid = (sa.pos + sb.pos) / 2
    # Nonzero COM offsets exercise the wrench-to-COM conversion.
    segs = [replace(s, pos=s.pos - mid, com=np.array([.001, -.002, .001])) for s in (sa, sb)]
    desc = ContentDesc(d.kappa, segs, [replace(j, a=0, b=1)], [(0, 1)], None, False)
    return Sim(scene(desc, (0, 0, 0), table=False, gravity=False, mode="bushing"))


def pose(sim, u, phi):
    d, m = sim.d, sim.m
    d.qpos[:] = m.qpos0
    j = sim.desc.joints[0]
    a, b = sim.seg_bodies
    q = m.jnt_qposadr[m.body_jntadr[b]]
    quat = np.zeros(4)
    angle = np.linalg.norm(phi)
    quat[0] = np.cos(angle / 2)
    quat[1:] = phi * (np.sin(angle / 2) / angle if angle else .5)
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, quat)
    arm = np.eye(3)[j.axis] * j.rest / 2
    d.qpos[q:q + 3] = sim.desc.segments[0].pos + arm + u + r.reshape(3, 3) @ arm
    d.qpos[q + 3:q + 7] = quat
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


class BushingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.measurements = []
        cls.gradients = []
        cls.releases = []

    def test_declared_stiffness_and_force(self):
        self.stiffness_errors = []
        for name in ("bendable", "compressible"):
            for axis in range(3):
                sim = pair(name, axis)
                j = sim.desc.joints[0]
                other = [a for a in range(3) if a != axis]
                cases = [("compression", axis, -1, j.k_comp, False),
                         ("tension", axis, 1, j.k_tens, False),
                         ("twist", axis, 1, j.k_twist, True)]
                cases += [(f"shear{i}", a, 1, j.k_shear[i], False) for i, a in enumerate(other)]
                cases += [(f"bend{i}", a, 1, j.k_bend[i], True) for i, a in enumerate(other)]
                for tag, a, sign, k, angular in cases:
                    with self.subTest(anchor=name, axis=axis, mode=tag):
                        h = sign * (1e-5 if angular else 1e-6)
                        u, phi = np.zeros(3), np.zeros(3)
                        (phi if angular else u)[a] = h
                        pose(sim, u, phi)
                        measured = 2 * sim.elastic_energy() / h**2
                        self.assertLess(abs(measured / k - 1), .01)
                        sim.d.xfrc_applied[:] = 0
                        sim.bushings.apply(sim.m, sim.d)
                        f = sim.d.xfrc_applied[sim.seg_bodies[1]]
                        # Recover the couple about the connection from the COM wrench.
                        b = sim.seg_bodies[1]
                        face = sim.d.xpos[sim.seg_bodies[0]] + np.eye(3)[axis] * j.rest / 2
                        couple = f[3:] - np.cross(face - sim.d.xipos[b], f[:3])
                        response = couple[a] if angular else f[a]
                        self.assertLess(abs(response / (-k * h) - 1), .01)
                        type(self).measurements.append(dict(anchor=name, axis=axis, mode=tag, target=k,
                                                            energy_stiffness=measured, force_stiffness=-response / h))

    def test_energy_gradient_momentum_and_damping(self):
        rng = np.random.default_rng(10)
        for name in ("bendable", "compressible"):
            for axis in range(3):
                sim = pair(name, axis)
                m, d = sim.m, sim.d
                phi = rng.normal(0, .15, 3)
                phi[next(a for a in range(3) if a != axis)] = 1.05
                u = rng.normal(0, .001, 3)
                j = sim.desc.joints[0]
                u[axis] = -1.05 * j.rest * j.lock_comp
                pose(sim, u, phi)
                q0 = d.qpos.copy()
                d.xfrc_applied[:] = 0
                sim.bushings.apply(m, d)
                wrench = d.xfrc_applied.copy()
                force = wrench[sim.seg_bodies, :3]
                torque = wrench[sim.seg_bodies, 3:]
                self.assertLess(np.linalg.norm(force.sum(0)), 1e-10)
                net_moment = (torque + np.cross(d.xipos[sim.seg_bodies], force)).sum(0)
                self.assertLess(np.linalg.norm(net_moment), 1e-10)
                qforce = np.zeros(m.nv)
                for b in sim.seg_bodies:
                    mujoco.mj_applyFT(m, d, wrench[b, :3], wrench[b, 3:], d.xipos[b], b, qforce)
                gradient = []
                h = 1e-7
                for i in range(m.nv):
                    velocity = np.eye(m.nv)[i]
                    energies = []
                    for sign in (1, -1):
                        d.qpos[:] = q0
                        mujoco.mj_integratePos(m, d.qpos, velocity, sign * h)
                        mujoco.mj_forward(m, d)
                        energies.append(sim.elastic_energy())
                    gradient.append((energies[0] - energies[1]) / (2 * h))
                error = np.linalg.norm(qforce + gradient) / np.linalg.norm(qforce)
                self.assertLess(error, 1e-5)
                type(self).gradients.append(dict(anchor=name, axis=axis, relative_error=float(error),
                                                net_moment_residual=float(np.linalg.norm(net_moment))))
                # Native tendon damping dissipates at this finite deformation too.
                d.qpos[:] = q0
                d.qvel[:] = rng.normal(0, .2, m.nv)
                mujoco.mj_forward(m, d)
                self.assertLess(float(d.qfrc_damper @ d.qvel), 0)
                # A common rigid transform must preserve strain energy.
                before = sim.elastic_energy()
                rot = np.array([np.cos(.3), 0, np.sin(.3), 0])
                mat = np.zeros(9)
                mujoco.mju_quat2Mat(mat, rot)
                for b in sim.seg_bodies:
                    q = m.jnt_qposadr[m.body_jntadr[b]]
                    d.qpos[q:q + 3] = mat.reshape(3, 3) @ d.qpos[q:q + 3] + [.4, -.2, .1]
                    old = d.qpos[q + 3:q + 7].copy()
                    mujoco.mju_mulQuat(d.qpos[q + 3:q + 7], rot, old)
                mujoco.mj_forward(m, d)
                self.assertAlmostEqual(sim.elastic_energy() / before, 1, places=10)

    def test_native_damping_matches_declared_coefficients(self):
        """At the rest pose the tendon dampers give the face-frame damping diag(c_trans, c_rot)."""
        type(self).damping = []
        for name in ("bendable", "compressible"):
            for axis in range(3):
                sim = pair(name, axis)
                m, d = sim.m, sim.d
                j = sim.desc.joints[0]
                pose(sim, np.zeros(3), np.zeros(3))
                face = d.xpos[sim.seg_bodies[0]] + np.eye(3)[axis] * j.rest / 2
                dof = m.jnt_dofadr[m.body_jntadr[sim.seg_bodies[1]]]
                modes = []
                for i in range(6):
                    v = np.zeros(m.nv)
                    w = np.eye(3)[i - 3] if i >= 3 else np.zeros(3)
                    v[dof:dof + 3] = (np.eye(3)[i] if i < 3 else 0) + np.cross(w, d.xpos[sim.seg_bodies[1]] - face)
                    v[dof + 3:dof + 6] = w
                    modes.append(v)
                damping = np.zeros((6, 6))
                for i, v in enumerate(modes):
                    d.qvel[:] = v
                    mujoco.mj_forward(m, d)
                    damping[:, i] = [-float(d.qfrc_damper @ u) for u in modes]
                target = np.concatenate(joint_damping(j))
                with self.subTest(anchor=name, axis=axis):
                    error = np.abs(np.diag(damping) / target - 1).max()
                    cross = np.abs(damping - np.diag(np.diag(damping))).max() / target.max()
                    self.assertLess(error, .01)
                    self.assertLess(cross, .01)
                    type(self).damping.append(dict(anchor=name, axis=axis, relative_error=float(error),
                                                   cross_ratio=float(cross)))

    def test_native_stops_engage_at_the_declared_locks(self):
        """Inner fibres reach their lower limit at straight compression to the lock and at lock_bend unloaded;
        outer fibres reach their upper limit at the same pure bending lock."""
        type(self).stop_errors = []
        for name in ("bendable", "compressible"):
            for axis in range(3):
                sim = pair(name, axis)
                j = sim.desc.joints[0]
                stops = np.where(sim.t_kind == "stop")[0]
                outer = np.where(sim.t_kind == "stop_upper")[0]
                self.assertEqual((len(stops), len(outer)), (4, 4))
                self.assertTrue(np.all(sim.m.tendon_stiffness[np.r_[stops, outer]] == 0))
                low, up = sim.m.tendon_range[stops, 0], sim.m.tendon_range[outer, 1]
                u, phi = np.zeros(3), np.zeros(3)
                u[axis] = -j.lock_comp * j.rest
                cases = [("compression", u, phi)]
                for a in range(3):
                    if a != axis:
                        bend = np.zeros(3)
                        bend[a] = j.lock_bend
                        cases.append((f"bend{a}", np.zeros(3), bend))
                for tag, u_, phi_ in cases:
                    with self.subTest(anchor=name, axis=axis, mode=tag):
                        pose(sim, u_, phi_)
                        error = abs(sim.d.ten_length[stops].min() / low[0] - 1)
                        self.assertLess(error, 1e-6)
                        if tag != "compression":
                            outer_error = abs(sim.d.ten_length[outer].max() / up[0] - 1)
                            self.assertLess(outer_error, 1e-6)
                            error = max(error, outer_error)
                        type(self).stop_errors.append(dict(anchor=name, axis=axis, mode=tag, relative_error=error))

    def test_compaction_welds_switch_across_the_squeeze(self):
        """Squeezing a 2x2x2 block along z to 0.95 x lock welds its x and y connections at their current
        pose, not the z connections; returning to rest releases them."""
        k = replace(ANCHORS["compressible"], segments=(2, 2, 2))
        sim = Sim(scene(build(k, table={}), (0, 0, 0), table=False, gravity=False))
        m, d = sim.m, sim.d
        lock = sim.desc.joints[next(i for i, j in enumerate(sim.desc.joints) if j.axis == 2)].lock_comp
        for b in sim.seg_bodies:
            q = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[q + 2] *= 1 - .95 * lock
            d.xfrc_applied[b, 2] = -np.sign(d.qpos[q + 2])
        mujoco.mj_forward(m, d)
        sim.step()
        active = d.eq_active[sim.welds].astype(bool)
        self.assertTrue(np.array_equal(active, sim.bushings.axis != 2))
        mujoco.mj_forward(m, d)
        rows = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self.assertEqual(rows.sum(), 6 * active.sum())
        self.assertLess(np.abs(d.efc_pos[:d.nefc][rows]).max(), 1e-9)
        d.qpos[:], d.qvel[:] = m.qpos0, 0
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        sim.step()
        self.assertFalse(d.eq_active[sim.welds].any())

    def test_compaction_recovers_after_real_unloading(self):
        """Remove the balancing loads, without resetting position/velocity: uniaxial and biaxial blocks
        must unlock and recover through their elastic forces rather than keep each other welded."""
        for name in ("bendable", "compressible"):
            for axes in ((2,), (1, 2)):
                with self.subTest(anchor=name, axes=axes):
                    k = replace(ANCHORS[name], segments=(2, 2, 2))
                    sim = Sim(scene(build(k, table={}), (0, 0, 0), table=False, gravity=False))
                    m, d = sim.m, sim.d
                    for b in sim.seg_bodies:
                        q = m.jnt_qposadr[m.body_jntadr[b]]
                        for axis in axes:
                            lock = next(j.lock_comp for j in sim.desc.joints if j.axis == axis)
                            d.qpos[q + axis] *= 1 - .95 * lock
                    mujoco.mj_forward(m, d)
                    sim.bushings.apply(m, d)
                    d.xfrc_applied[:] *= -1
                    self.assertLess(np.linalg.norm(d.xfrc_applied[:, 3:]), 1e-10)
                    initial_pos = d.xipos.copy()
                    rng = np.random.default_rng(61)
                    d.qvel[:] = rng.normal(0, .01, m.nv)
                    activation_energy = [sim.elastic_energy() + sim.kinetic_energy()]
                    for _ in range(round(.04 / m.opt.timestep)):
                        sim.step()
                        mujoco.mj_forward(m, d)
                        work = np.sum(d.xfrc_applied[:, :3] * (d.xipos - initial_pos))
                        activation_energy.append(sim.elastic_energy() + sim.kinetic_energy() - work)
                    activation_energy = np.array(activation_energy)
                    activation_rise = np.max(activation_energy - np.minimum.accumulate(activation_energy)) / activation_energy[0]
                    self.assertLess(activation_rise, .01)
                    self.assertTrue(d.eq_active[sim.welds].any())
                    d.xfrc_applied[:] = 0
                    energies = [sim.elastic_energy() + sim.kinetic_energy()]
                    for _ in range(round(1 / m.opt.timestep)):
                        sim.step()
                        energies.append(sim.elastic_energy() + sim.kinetic_energy())
                    self.assertFalse(d.eq_active[sim.welds].any())
                    *_, u, _ = sim.bushings.coordinates(d)
                    residual = np.max(np.abs(u[sim.bushings.row, sim.bushings.axis]) / sim.bushings.compression_travel)
                    self.assertLess(residual, .05)
                    energies = np.array(energies)
                    rise = np.max(energies - np.minimum.accumulate(energies)) / energies[0]
                    self.assertLess(rise, .01)
                    type(self).releases.append(dict(anchor=name, axes=axes, residual=float(residual),
                                                   energy_rise=float(rise), activation_energy_rise=float(activation_rise)))

    def test_biaxial_pad_release_without_reset(self):
        """Four pads physically squeeze a free block from rest, then leave; no pose/velocity reset."""
        from protocols import Runner, GAP
        from mj_adapter import PAD_T
        k = replace(ANCHORS["compressible"], size=(.06, .06, .06), segments=(2, 2, 2))
        desc = build(k, table={})
        pads, direction, travel = [], {}, {}
        for axis in (1, 2):
            lock = next(j.lock_comp for j in desc.joints if j.axis == axis)
            for sign in (-1, 1):
                name = f"p{axis}{'p' if sign > 0 else 'n'}"
                p = np.eye(3)[axis] * sign * (.03 + GAP + PAD_T)
                pads.append((name, axis, p))
                direction[name] = -sign * np.eye(3)[axis]
                travel[name] = GAP + .95 * lock * .03
        sim = Sim(scene(desc, (0, 0, 0), pads, table=False, gravity=False), [p[0] for p in pads])
        run = Runner(sim)
        peak = 0
        dt = sim.m.opt.timestep
        for _ in range(round(2 / dt)):
            t = run.t
            if t < .6:
                fraction = min(t / .5, 1)
            else:
                fraction = max(0., 1 - (t - .6) / .3)
            run.step({name: p + direction[name] * travel[name] * fraction for name, _, p in pads})
            peak = max(peak, int(sim.d.eq_active[sim.welds].sum()))
            self.assertIsNone(run.fail)
        self.assertGreater(peak, 0)
        self.assertFalse(sim.d.eq_active[sim.welds].any())
        *_, u, _ = sim.bushings.coordinates(sim.d)
        residual = np.max(np.abs(u[sim.bushings.row, sim.bushings.axis]) / sim.bushings.compression_travel)
        self.assertLess(residual, .05)
        type(self).releases.append(dict(anchor="compressible_four_pads", peak_welds=peak,
                                       residual=float(residual), dt=dt))

    def test_export_cannot_silently_omit_pair_forces(self):
        sim = pair("compressible", 0)
        with self.assertRaisesRegex(ValueError, "Sim.step"):
            object_mjcf(sim.desc, (0, 0, 0))
        self.assertIn("tendon", object_mjcf(sim.desc, (0, 0, 0), mode="truss"))

    def test_native_step_receives_the_elastic_impulse(self):
        sim = pair("compressible", 1)
        m, d = sim.m, sim.d
        m.tendon_damping[:] = 0     # isolate the elastic impulse from the implicit damping solve
        pose(sim, np.array([.0001, -.0001, .0001]), np.zeros(3))
        sim.bushings.apply(m, d)
        expected = np.zeros(m.nv)
        for b in sim.seg_bodies:
            w = d.xfrc_applied[b]
            mujoco.mj_applyFT(m, d, w[:3], w[3:], d.xipos[b], b, expected)
        d.xfrc_applied[:] = 0
        sim.step()
        impulse = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, impulse, d.qvel)
        self.assertLess(np.linalg.norm(impulse - expected * m.opt.timestep)
                        / np.linalg.norm(expected * m.opt.timestep), 1e-8)
        self.assertEqual(np.linalg.norm(d.xfrc_applied), 0)

    def test_energy_gradient_across_a_closed_block(self):
        k = replace(ANCHORS["compressible"], segments=(2, 2, 2))
        desc = build(k, table={})
        sim = Sim(scene(desc, (0, 0, 0), table=False, gravity=False))
        m, d = sim.m, sim.d
        rng = np.random.default_rng(33)
        for b in sim.seg_bodies:
            q = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[q:q + 3] *= [.96, .90, .94]
            d.qpos[q:q + 3] += rng.normal(0, 1e-4, 3)
            angle = rng.normal(0, .04, 3)
            theta = np.linalg.norm(angle)
            d.qpos[q + 3] = np.cos(theta / 2)
            d.qpos[q + 4:q + 7] = angle * np.sin(theta / 2) / theta
        mujoco.mj_forward(m, d)
        sim.bushings.apply(m, d)
        q0, wrench = d.qpos.copy(), d.xfrc_applied.copy()
        qforce = np.zeros(m.nv)
        for b in sim.seg_bodies:
            mujoco.mj_applyFT(m, d, wrench[b, :3], wrench[b, 3:], d.xipos[b], b, qforce)
        net_force = wrench[sim.seg_bodies, :3].sum(0)
        net_moment = (wrench[sim.seg_bodies, 3:] + np.cross(d.xipos[sim.seg_bodies],
                                                        wrench[sim.seg_bodies, :3])).sum(0)
        self.assertLess(np.linalg.norm(net_force), 1e-10)
        self.assertLess(np.linalg.norm(net_moment), 1e-10)
        gradient = np.zeros(m.nv)
        h = 1e-7
        for i in range(m.nv):
            energies = []
            for sign in (1, -1):
                d.qpos[:] = q0
                mujoco.mj_integratePos(m, d.qpos, np.eye(m.nv)[i], sign * h)
                mujoco.mj_forward(m, d)
                energies.append(sim.elastic_energy())
            gradient[i] = (energies[0] - energies[1]) / (2 * h)
        error = np.linalg.norm(qforce + gradient) / np.linalg.norm(qforce)
        self.assertLess(error, 1e-5)
        type(self).gradients.append(dict(anchor="compressible_2x2x2", axis="all", relative_error=float(error),
                                        net_moment_residual=float(np.linalg.norm(net_moment))))


if __name__ == "__main__":
    unittest.main()
