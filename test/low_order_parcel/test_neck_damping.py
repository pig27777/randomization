"""Passivity and disabled-path checks for power-conjugate neck damping."""

from dataclasses import replace
import unittest

import torch
from torch.utils._pytree import tree_flatten

from audit_energy_passivity import torsion_fixture
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import ParcelAction
from src.low_order_parcel.neck_damping import NeckDamping, NeckDampingParams


class NeckDampingTests(unittest.TestCase):
    def test_rotating_counterexample_and_rigid_comotion(self):
        mass = torch.ones((1, 1), dtype=torch.float64)
        contact = mass.new_zeros((1, 2, 6))
        contact[0, 0, ContactState.CAPTURE] = 1.
        damper = NeckDamping(mass, mass.new_ones((1, 1, 2)), .002, NeckDampingParams(15., .1))
        tip = mass.new_zeros((1, 2, 3))
        anchor = tip + mass.new_tensor([0., 0., -.10])
        omega = tip + mass.new_tensor([0., -4., 0.])
        anchor_vel = tip + mass.new_tensor([.05, 0., 0.])
        motion = mass.new_zeros((1, 2, 2))
        normal = mass.new_full((1, 2), 2.)

        def step(pad_v, anchor_v, pad_omega=omega, point=anchor, before=contact, mobility=motion):
            relative = damper.relative(tip, point, pad_v, anchor_v, pad_omega)
            return damper.resolve(relative, before, contact, normal, mobility, pad_v, anchor_v, pad_omega)

        damper.c.fill_(6.)  # length == characteristic, so c_eff == 3 Ns/m
        result = step(tip, anchor_vel)
        work = .002 * (result.anchor_power - result.pad_force_power - result.pad_couple_power).sum()
        self.assertAlmostEqual(float(work), -.000735, places=11)
        self.assertGreater(float(result.dissipation_J.sum()), 0.)
        self.assertTrue(torch.equal(torch.linalg.cross(anchor - tip, result.force) + result.pad_couple, tip))
        self.assertEqual(float(step(tip, torch.linalg.cross(omega, anchor - tip)).force.abs().max()), 0.)
        boost = mass.new_tensor([.8, -.5, 1.4])
        torch.testing.assert_close(step(tip + boost, anchor_vel + boost).force, result.force)
        self.assertEqual(float(step(tip, anchor_vel, before=torch.zeros_like(contact)).force.abs().max()), 0.)
        damper.c.fill_(5000.)
        capped = step(tip, anchor_vel)
        self.assertAlmostEqual(float(capped.force.norm()), 2.)
        motion[0, 0, 0] = 500.
        clamped = step(tip, anchor_vel)
        self.assertAlmostEqual(float(clamped.force.norm()) * (.002 * 504.), .35, places=11)

    def test_zero_coefficient_matches_legacy_steps(self):
        sim, original, command = torsion_fixture(42.)
        old = current = original
        for _ in range(16):
            sim.neck = None
            reference = sim.step(old, command)
            sim.neck = NeckDamping(sim.mass, sim.proxy.params.mu_eff[:, None], .002, NeckDampingParams())
            candidate = sim.step(current, command)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                tree_flatten(reference)[0], tree_flatten(candidate)[0], strict=True)))
            old, current = reference.state, candidate.state

        episode = CornerSuite(DomainRandomizer()).select(torch.tensor([5]))
        disabled = replace(episode, neck_damping=NeckDampingParams())
        first, second = disabled.make_sim(), disabled.make_sim()
        first.neck = None
        one = first.mass.new_ones((1, 2))
        tips = first.mass.new_tensor([[[-.024, 0., .080], [.024, 0., .080]]])
        action = ParcelAction(tips, torch.zeros_like(tips), 12. * one, one, .02 * one, one, tips[:, 0] * 0.,
                              first.mass * 0.)
        command = DualArmAction(action, action, torch.tensor([[True, False]]))
        old, current = disabled.table_state(first), disabled.table_state(second)
        for _ in range(16):
            reference = first.step(old, command)
            candidate = second.step(current, command)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                tree_flatten(reference)[0], tree_flatten(candidate)[0], strict=True)))
            old, current = reference.state, candidate.state
