"""Checks for finite pad torque, the rotational work port and disabled parity."""

import math
import unittest

import torch

from audit_energy_passivity import torsion_episode, torsion_fixture
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim


class PadTorsionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_eccentric_book_slips_at_low_grip_and_holds_at_high_grip(self):
        low = torsion_episode(18.)
        high = torsion_episode(90.)
        self.assertLess(low[0]["torque_limit_Nm"], .40)
        self.assertGreater(high[0]["torque_limit_Nm"], .40)
        self.assertGreater(sum(row["torsion_slip_rad"] > 1e-6 for row in low), 10)
        self.assertGreater(abs(low[-1]["roll_rad"]), math.radians(30.))
        self.assertEqual(max(row["torsion_slip_rad"] for row in high), 0.)
        self.assertLess(abs(high[-1]["roll_rad"]), math.radians(12.))
        self.assertLess(max(abs(row["roll_speed_rad_s"]) for row in high[-100:]), .035)
        self.assertLess(max(row["com_speed_mps"] for row in high[-100:]), .020)
        for rows in (low, high):
            self.assertTrue(all(row["captured"] == 2 and row["clearance_m"] > .001 for row in rows))
            self.assertGreaterEqual(min(row["net_dissipation_J"] for row in rows), -1e-4)
            self.assertGreaterEqual(min(row["torsion_net_dissipation_J"] for row in rows), -1e-8)
            self.assertGreaterEqual(min(row["torsion_slip_loss_J"] for row in rows), -1e-8)
        print("off-center book, low/high roll degrees", *(math.degrees(rows[-1]["roll_rad"]) for rows in (low, high)),
              "low/high accumulated slip rad", *(sum(row["torsion_slip_rad"] for row in rows) for rows in (low, high)))

    @torch.inference_mode()
    def test_pure_couple_and_relative_angular_power(self):
        sim, state, action = torsion_fixture(42.)
        payload, torsion = sim.payload, sim.torsion
        p = state.payload
        r = p.pos - state.com[:, None]
        spin = p.pos.new_tensor([[.8, .6, -.2]])
        vel = torch.linalg.cross(spin[:, None], r)
        moving = p._replace(vel=vel)
        separation = action.pinch_position[:, None] - p.pos[:, None, :2]
        active = state.contacts[:, None]
        twist = state.torsion
        normal = action.normal_force[:, None]
        zero_load = torch.zeros_like(p.pos)
        frame = payload.frame(p)
        matched, _, result = torsion.couples(payload, moving, frame, twist, active, active, normal,
                                             separation, zero_load, None, spin[:, None])
        self.assertLess(float(matched.abs().max()), 5e-7)
        self.assertLess(float(result.energy.max()), 1e-12)
        torque, force, result = torsion.couples(payload, moving, frame, twist, active, active, normal,
                                               separation, zero_load, None, None)
        self.assertLessEqual(float(torque.norm(dim=-1).max()), float(result.capacity.max()))
        torch.testing.assert_close(force.sum(dim=1), torch.zeros_like(spin), atol=1e-6, rtol=0.)
        torch.testing.assert_close(torch.linalg.cross(r, force).sum(dim=1), torque[:, 0], atol=1e-7, rtol=0.)
        net = result.jaw_work - result.body_work - result.energy
        torch.testing.assert_close(net, result.damping + result.slip_loss + result.integration_loss,
                                   atol=1e-8, rtol=1e-5)
        uncaptured = active.clone()
        uncaptured[..., ContactState.CAPTURE] = 0.
        _, _, edge = torsion.couples(payload, moving, frame, twist, uncaptured, active, normal,
                                     separation, zero_load, None, None)
        self.assertFalse(edge.energy.any() | edge.torque.any())

    @torch.inference_mode()
    def test_zero_radius_preserves_parcel_step(self):
        off, state, action = torsion_fixture(42., radius=0.)
        legacy = MacroParcelSim(off.proxy, payload=off.payload)
        legacy_state = state._replace(torsion=())
        for _ in range(32):
            new = off.step(state, action)
            old = legacy.step(legacy_state, action)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(new[1:], old[1:], strict=True)))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(new.state.payload, old.state.payload, strict=True)))
            self.assertTrue(torch.equal(new.state.contacts, old.state.contacts))
            self.assertFalse(new.state.torsion.energy.any())
            state, legacy_state = new.state, old.state
