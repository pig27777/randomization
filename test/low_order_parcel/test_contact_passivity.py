"""Regression for passive creation, retention and release of a rope reference."""

import unittest

import torch

from audit_energy_passivity import ObservePayload, bench
from test_stage2_proxy import inputs, make_proxy
from src.low_order_parcel.lumped_contact_proxy import ContactState as C, LumpedContactProxy


class ContactPassivityTest(unittest.TestCase):
    def test_reference_latches_without_force_and_resets_on_release(self):
        for ablation in (False, True):
            with self.subTest(ablation=ablation):
                base = make_proxy(dt=.002, h_min=.012)
                model = LumpedContactProxy(base.params, base.dt, ablation_rigid_snap=ablation)
                state = model.initial_state()
                state[..., C.SLACK] = .003
                reference = model.params.L0
                acquired = inputs(model, distance=.110)._replace(feed_depth=torch.full_like(reference, .014))
                velocity = torch.ones_like(acquired.pinch_velocity)
                edge, latched = model.step_with_reference(state, acquired._replace(pinch_velocity=velocity), reference)
                self.assertEqual(edge.state.shape, state.shape)
                self.assertTrue((edge.state[..., C.CAPTURE] == 1.).all())
                self.assertFalse(edge.force.any() | edge.elastic_energy.any() | edge.tension.any())
                torch.testing.assert_close(latched, torch.full_like(reference, .110))
                settled, unchanged = model.step_with_reference(edge.state, acquired, latched)
                self.assertFalse(settled.elastic_energy.any() | settled.tension.any())
                torch.testing.assert_close(latched, unchanged, atol=0., rtol=0.)
                moved = acquired._replace(pinch_position=acquired.pinch_position + .002 * velocity[..., :1]
                                          * velocity.new_tensor([0., 0., 1.]))
                stretched, latched = model.step_with_reference(settled.state, moved, unchanged)
                self.assertTrue((stretched.elastic_energy > 0.).all())
                release, reset = model.step_with_reference(
                    stretched.state, moved._replace(normal_force=torch.zeros_like(reference)), latched)
                self.assertFalse(release.force.any() | release.elastic_energy.any())
                torch.testing.assert_close(reset, reference, atol=0., rtol=0.)

    def test_damping_impulse_cannot_reverse_its_relative_motion(self):
        proxy = make_proxy(dt=.002, c_d=1500., beta=0.)
        state = proxy.initial_state()
        state[..., C.SLACK] = .004
        state[..., C.CAPTURE] = 1.
        cmd = inputs(proxy, normal=6., distance=.100)
        speed = cmd.pinch_position.new_tensor([[[0., 0., 2.], [0., 0., 2.]]])
        inverse_mass = proxy.params.L0.new_full((1, 2), 60.)
        result, _ = proxy.step_with_reference(state, cmd._replace(pinch_velocity=speed),
                                              proxy.params.L0, inverse_mass)
        removal = proxy.dt * inverse_mass * result.tension
        self.assertTrue((removal > 0.).all())
        self.assertTrue((removal <= speed[..., 2]).all())

    def test_isolated_hard_bend_is_passive(self):
        sim, _, _, _ = bench("bend-limit")
        payload = sim.payload
        observe = ObservePayload(sim)
        state = payload.initial_state()
        mode = payload.bend_sign[None, :, None] * payload.up
        pos = payload.rest + .45 * payload.up + mode * (.90 * payload.bend_limit)[:, :, None]
        state = state._replace(pos=pos, vel=.11 * mode)
        previous = sum(observe.storage(state).values())
        active_steps = 0
        minimum = float("inf")
        for _ in range(80):
            output = payload.step(state, torch.zeros_like(pos), state.d_surf)
            self.assertFalse(output.support_N.any())
            energy = sum(observe.storage(output.state).values())
            minimum = min(minimum, previous - energy)
            active_steps += int(observe.hard_bend)
            previous, state = energy, output.state
        self.assertGreater(active_steps, 0)
        self.assertGreaterEqual(minimum, -1e-4)
        print(f"Isolated hard bend: {active_steps} active steps, min W-dE {minimum:.9g} J")
