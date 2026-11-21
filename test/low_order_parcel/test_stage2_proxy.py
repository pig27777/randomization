"""Stage 2 Gate B for the public batched contact step."""

from dataclasses import fields
import math
import unittest

import torch

from src.low_order_parcel.lumped_contact_proxy import ContactInputs, ContactState as C, LumpedContactProxy, ProxyParams


def make_proxy(dt=.001, n=1, **overrides):
    values = dict(S_bag=.025, s_max=.010, alpha=.7, k=100., c_d=.12, mu_eff=[.35, .42], beta=.12,
                  s_crit=.001, F_min=.35, F_lock=.55, F_release=.2, h_min=.004, cos_theta_max=.65,
                  tau_gate=.012, v_gather_max=.08, L0=[.10, .10])
    values.update(overrides)
    params = ProxyParams.from_scalars(values)
    return LumpedContactProxy(ProxyParams(**{f.name: getattr(params, f.name).repeat(n, 1) for f in fields(params)}), dt)


def inputs(proxy, *, normal=1., inward=.06, distance=.10):
    p = proxy.params
    attachment = p.S_bag.new_zeros((p.S_bag.shape[0], 2, 3))
    attachment[:, :, 0] = torch.tensor([-.04, .04])
    pinch = attachment.clone()
    pinch[:, :, 2] = distance
    zeros = torch.zeros_like(pinch)
    pair = torch.ones_like(p.mu_eff)
    return ContactInputs(pinch, attachment, zeros, zeros, zeros[:, 0], pair * normal, pair * inward,
                         pair * .01, pair)


def close(proxy):
    state = proxy.initial_state()
    contact = inputs(proxy)
    proxy.validate_inputs(state, contact)
    for _ in range(math.ceil(.15 / proxy.dt)):
        state = proxy.step(state, contact).state
    return state


class Stage2ProxyTest(unittest.TestCase):
    def test_global_slack_budget(self):
        proxy = make_proxy(n=16, v_gather_max=100.)
        gen = torch.Generator().manual_seed(10)
        proxy.params.S_bag.copy_(torch.linspace(0., .011, 16).view(-1, 1))
        contact = inputs(proxy)
        trajectory = []
        state = proxy.initial_state()
        for _ in range(100):
            speed = torch.rand((16, 2), generator=gen) * 10_000.
            state = proxy.step(state, contact._replace(inward_velocity=speed)).state
            trajectory.append(state[:, :, C.SLACK])
        slack = torch.stack(trajectory)
        self.assertTrue((slack.sum(dim=-1) <= proxy.params.S_bag[:, 0] + 1e-7).all())
        self.assertTrue((slack >= 0).all())
        self.assertTrue((slack[:, 0] == 0).all())
        self.assertGreater(slack[-1, -1].sum().item(), .010)

    def test_jitter_cannot_accumulate_slack(self):
        proxy = make_proxy(n=8)
        contact = inputs(proxy)
        frequencies = torch.linspace(50., 250., 8).view(-1, 1)
        state = proxy.initial_state()
        for i in range(1000):
            wave = torch.sin(2 * math.pi * i * proxy.dt * frequencies)
            normals = .2 + torch.cat((wave, -wave), dim=1) * .3
            inward = torch.cat((wave, -wave), dim=1) * .002
            state = proxy.step(state, contact._replace(normal_force=normals, inward_velocity=inward)).state
            self.assertTrue((state[:, :, C.HOLD_TIME] < proxy.params.tau_gate).all())
        self.assertEqual(state[:, :, C.SLACK].abs().max().item(), 0.)
        self.assertEqual(state[:, :, C.CAPTURE].abs().max().item(), 0.)
        self.assertTrue((close(proxy)[:, :, C.CAPTURE] == 1.).all())

    def test_slip_dissipates_and_detachment_has_no_impulse(self):
        proxy = make_proxy()
        state = close(proxy)
        contact = inputs(proxy, normal=.60, inward=0., distance=.115)
        previous_energy = .5 * proxy.params.k * .015 ** 2
        energy, rates, tension = [], [], []
        for _ in range(70):
            result = proxy.step(state, contact)
            self.assertTrue((result.state[:, :, C.SLIP_CUM] >= state[:, :, C.SLIP_CUM]).all())
            self.assertTrue((result.elastic_energy <= previous_energy + 1e-7).all())
            if (result.state[:, :, C.CAPTURE] < state[:, :, C.CAPTURE]).any():
                released = result.state[:, :, C.CAPTURE] == 0.
                self.assertTrue((result.force[released] == 0.).all())
            state, previous_energy = result.state, result.elastic_energy
            energy.append(result.elastic_energy.sum().item())
            rates.append(result.slip_rate.max().item())
            tension.append(result.tension.max().item())
        self.assertGreater(rates[0], 0.)
        self.assertGreater(energy[0], 0.)
        self.assertEqual(tension[-1], 0.)
        self.assertEqual(state[:, :, C.CAPTURE].sum().item(), 0.)

        # With fixed pads and a free, initially stored-energy-loaded mass, separation
        # cannot impart extra velocity. Use a small physical reserve so both pads release.
        state = close(proxy)
        state[:, :, C.SLACK] = .0015
        position, velocity, mass = 0., 0., .20
        initial_energy = proxy.params.k.item() * .015 ** 2
        for _ in range(40):
            attachment = contact.bag_attachment.clone()
            attachment[:, :, 2] += position
            speed = torch.zeros_like(attachment)
            speed[:, :, 2] = velocity
            result = proxy.step(state, contact._replace(bag_attachment=attachment, attachment_velocity=speed))
            next_velocity = velocity + proxy.dt * result.force[:, :, 2].sum().item() / mass
            self.assertLessEqual(.5 * mass * next_velocity ** 2 + result.elastic_energy.sum().item(), initial_energy)
            state, velocity = result.state, next_velocity
            position += proxy.dt * velocity
        self.assertEqual(state[:, :, C.CAPTURE].sum().item(), 0.)

    def test_hysteresis_retains_both_captures(self):
        proxy = make_proxy()
        state = close(proxy)
        self.assertTrue((state[:, :, C.CAPTURE] == 1.).all())
        for index in range(450):
            contact = inputs(proxy, normal=(.23, .53)[index % 2], inward=0.)
            state = proxy.step(state, contact).state
            self.assertTrue((state[:, :, C.CAPTURE] == 1.).all())
        state = proxy.step(state, inputs(proxy, normal=.19, inward=0.)).state
        self.assertEqual(state[:, :, C.CAPTURE].sum().item(), 0.)
        state = proxy.step(state, inputs(proxy, normal=.53, inward=0.)).state
        self.assertEqual(state[:, :, C.CAPTURE].sum().item(), 0.)

    def test_capture_requires_both_geometric_gates_under_full_normal_load(self):
        proxy = make_proxy()
        contact = inputs(proxy)
        depth = contact.feed_depth.clone()
        depth[:, 0] = 0.
        alignment = contact.alignment_cos.clone()
        alignment[:, 1] = 0.
        bad_geometry = contact._replace(feed_depth=depth, alignment_cos=alignment)
        state = proxy.initial_state()
        for _ in range(80):
            state = proxy.step(state, bad_geometry).state
        self.assertTrue((state[:, :, C.SLACK] >= proxy.params.s_crit).all())
        self.assertTrue((state[:, :, C.CAPTURE] == 0.).all())
        good_geometry = proxy.step(state, contact)
        self.assertTrue((good_geometry.state[:, :, C.CAPTURE] == 1.).all())

    def test_million_extreme_steps_at_both_step_sizes(self):
        gen = torch.Generator().manual_seed(56484)
        size = 257, 16, 2
        normal = torch.rand(size, generator=gen)
        normal = torch.where(normal < .2, normal * (-1e4), torch.where(normal < .9, normal * 2., normal * 1e4))
        inward = torch.randn(size, generator=gen) * 50.
        distance = torch.rand(size, generator=gen) * 3.
        speed = torch.randn(size + (3,), generator=gen).clamp(-4., 4.) * 20.
        depth = torch.rand(size, generator=gen) * .030
        alignment = torch.rand(size, generator=gen) * 1.5 - .5
        for dt in (.001, .002):
            with self.subTest(dt=dt):
                proxy = make_proxy(dt, n=16, k=2000., c_d=10.)
                proxy.params.S_bag.copy_(torch.linspace(0., .04, 16).view(-1, 1))
                base = inputs(proxy)
                frame = []
                for i in range(len(distance)):
                    pinch = base.pinch_position.clone()
                    pinch[:, :, 2] = distance[i]
                    frame.append(base._replace(pinch_position=pinch, pinch_velocity=speed[i], normal_force=normal[i],
                                               inward_velocity=inward[i], feed_depth=depth[i], alignment_cos=alignment[i]))
                state = proxy.initial_state()
                max_tension, detached, peak_slip = 0., 0, 0.
                steps = 1_000_000
                for first in range(0, steps, len(frame)):
                    history = []
                    prior = state
                    for action in frame[:min(len(frame), steps - first)]:
                        result = proxy.step(state, action)
                        state = result.state
                        history.append(result)
                    states, forces, torques, tensions, rates, energies, elastic = (torch.stack(x) for x in zip(*history))
                    self.assertTrue(all(torch.isfinite(x).all()
                                        for x in (states, forces, torques, tensions, rates, energies, elastic)))
                    slack = states[:, :, :, C.SLACK]
                    self.assertTrue((slack >= 0.).all())
                    self.assertTrue((slack.sum(dim=-1) <= proxy.params.S_bag[:, 0] + 1e-7).all())
                    cumulative = torch.cat((prior[None, :, :, C.SLIP_CUM], states[:, :, :, C.SLIP_CUM]))
                    self.assertTrue((cumulative.diff(dim=0) >= 0.).all())
                    capture = torch.cat((prior[None, :, :, C.CAPTURE], states[:, :, :, C.CAPTURE]))
                    detached += int((capture.diff(dim=0) < 0.).sum().item())
                    max_tension = max(max_tension, float(tensions.max()))
                    peak_slip = max(peak_slip, float(rates.max()))
                self.assertGreater(detached, 0)
                self.assertGreater(peak_slip, 0.)
                self.assertLessEqual(max_tension, 2000. * (3. - .10) + 10. * 80. + .01)
                print(f"dt={dt * 1000:g} ms, B=16, steps={steps}, releases={detached}, peak tension={max_tension:.3f} N",
                      flush=True)

    def test_rigid_snap_switch_keeps_the_tensor_interface(self):
        normal = make_proxy(S_bag=0.)
        rigid = LumpedContactProxy(normal.params, normal.dt, ablation_rigid_snap=True)
        state = rigid.step(rigid.initial_state(), inputs(rigid)).state
        result = rigid.step(state, inputs(rigid, distance=.103))
        self.assertEqual(result.state.shape, (1, 2, 6))
        self.assertTrue((result.state[:, :, C.CAPTURE] == 1.).all())
        self.assertEqual(result.state[:, :, C.SLACK].sum().item(), 0.)
        self.assertEqual(result.slip_rate.sum().item(), 0.)
        self.assertTrue((result.tension > 0.).all())
