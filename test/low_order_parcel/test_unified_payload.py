"""Acceptance probes for the four-point payload and its frozen proxy coupling."""

from dataclasses import fields, replace
import json
import math
from pathlib import Path
import unittest

import torch

from src.low_order_parcel.lumped_contact_proxy import ContactInputs, ContactState, LumpedContactProxy, ProxyParams
from src.low_order_parcel.unified_payload import DT, GRAVITY, PayloadParamsX, UnifiedPayloadTemplate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/payload_template_X.yaml").read_bytes())


def model(*names, **overrides):
    return UnifiedPayloadTemplate(PayloadParamsX.from_presets(CONFIG, names, **overrides))


class UnifiedPayloadTest(unittest.TestCase):
    @torch.inference_mode()
    def test_p2_progressive_lift_and_drape(self):
        sim = model("folded_clothes", "heavy_quilt")
        state = sim.initial_state()
        target0 = state.pos[:, 3, 2].clone()
        # Two pads lift along a diagonal, allowing the other corners to hang down.
        load_shares = state.pos.new_tensor([.01, .46, .52, .01])
        lift_s, hold_s = .70, .15
        hold_start = round(lift_s / DT)
        hold_end = round((lift_s + hold_s) / DT)
        force = torch.zeros_like(state.pos)
        support, bend, pull = [], [], []
        for i in range(1200):
            t = (i + 1) * DT
            micro = .025 * .5 * (1. - math.cos(math.pi * min(t / lift_s, 1.)))
            further = .22 * .5 * (1. - math.cos(math.pi * min(max((t - lift_s - hold_s) / .85, 0.), 1.)))
            target = target0 + micro + further
            lifted_point = (state.pos[:, :, 2] * load_shares).sum(dim=1)
            speed = (state.vel[:, :, 2] * load_shares).sum(dim=1)
            tension = (650. * (target - lifted_point) - 12. * speed).clamp_min(0.)
            force[:, :, 2] = tension[:, None] * load_shares
            result = sim.step(state, force)
            state = result.state
            support.append(result.clearance > .0003)
            bend.append(sim.bend_angle(state)[:, 0])
            pull.append(tension)
        airborne, theta, tension = map(torch.stack, (support, bend, pull))
        for j, name in enumerate(("folded_clothes", "heavy_quilt")):
            first = airborne[:, j].to(torch.int32).argmax(dim=0)
            rise = tensor_mean(tension[:hold_start, j].diff().clamp_min(0.))
            hold = tension[hold_start:hold_end, j]
            ratio = tensor_mean(hold[-20:]) / tensor_mean(hold[:20])
            angle = tensor_mean(theta[-50:, j])
            print(f"P2 {name}: lift frames {first.tolist()}, mean positive rise {rise:.4f} N/step, hold ratio {ratio:.3f}, angle {angle:.3f}")
            self.assertTrue(airborne[-50:, j].all())
            self.assertEqual(first.unique().numel(), 4)
            self.assertGreater(int(first.max() - first.min()), 5)
            self.assertLess(rise, .05 * float(sim.params.total_mass[j]) * GRAVITY)
            self.assertAlmostEqual(ratio, 1., delta=.15)
            self.assertGreater(angle, .3)

    @torch.inference_mode()
    def test_p3_indentation_hardening_and_rebound(self):
        sim = model("fluffy_pillow", "rigid_box")
        state = sim.initial_state()
        weights = torch.eye(4)[:2].expand(2, -1, -1)
        engaged = torch.ones((2, 2), dtype=torch.bool)
        dmax = sim.depth_max
        forces = []
        for ratio in (.2, .4, .6, .8, 1.):
            depth = (dmax * ratio).expand(-1, 2)
            result = sim.surface_contact(state, depth, weights, engaged)
            settled = state._replace(d_surf=result.depth)
            result = sim.surface_contact(settled, depth, weights, engaged)
            forces.append(result.pad_force_N[0, 0])
        self.assertTrue((torch.stack(forces).diff().diff() > 0.).all())
        overpress = sim.surface_contact(state, torch.full((2, 2), .5), weights, engaged)
        self.assertTrue((overpress.core_force_N > 0.).all())
        torch.testing.assert_close(overpress.pad_force_N.sum(1), overpress.block_force_N.sum(1))
        self.assertEqual(float(overpress.depth[1].max()), 0.)
        zeros = torch.zeros((2, 2))
        slack = torch.full((2, 1), .01)
        feed, cosine, budget = sim.contact_geometry(overpress.depth, weights, zeros, zeros, slack)
        torch.testing.assert_close(feed[0], torch.full_like(feed[0], .020))
        self.assertTrue((cosine[0] > .65).all())
        self.assertTrue((budget - slack <= 2. * sim.depth_max).all())
        self.assertTrue((feed[1] == 0.).all() and (cosine[1] == 0.).all() and (budget[1] == slack[1]).all())
        moderate = sim.surface_contact(state, torch.full((2, 2), .018), weights, engaged)
        feed, cosine, budget = sim.contact_geometry(moderate.depth, weights, zeros, zeros, slack)
        self.assertTrue((feed[0] < .020).all())
        nominal = json.loads((ROOT / "configs/proxy_material_envelopes.yaml").read_bytes())["materials"]["pe"]["nominal"]
        base = ProxyParams.from_scalars(nominal["proxy"])
        params = ProxyParams(**{f.name: getattr(base, f.name).expand(2, -1) for f in fields(base)} | {"S_bag": budget})
        proxy = LumpedContactProxy(params, DT)
        xyz = torch.zeros((2, 2, 3))
        inputs = ContactInputs(xyz, xyz, xyz, xyz, xyz[:, 0], torch.ones_like(feed), torch.full_like(feed, .04), feed, cosine)
        contacts = proxy.initial_state()
        for _ in range(100):
            contacts = proxy.step(contacts, inputs).state
        self.assertTrue((contacts[0, :, ContactState.CAPTURE] == 1.).all())
        self.assertTrue((contacts[1, :, ContactState.CAPTURE] == 0.).all())
        loaded = state._replace(d_surf=overpress.depth)
        for _ in range(75):
            result = sim.surface_contact(loaded, zeros, weights, engaged)
            self.assertFalse(result.pad_force_N.any())
            loaded = loaded._replace(d_surf=result.depth)
        torch.testing.assert_close(loaded.d_surf, overpress.depth * torch.exp(-.15 / sim.params.tau_rebound),
                                   rtol=1e-5, atol=1e-8)
        print(f"P3: settled force {torch.stack(forces).tolist()} N, rebound ratio {float(loaded.d_surf[0, 0] / dmax[0, 0]):.3f}")

    @torch.inference_mode()
    def test_p4_slip_impulse_and_momentum(self):
        sim = model("loose_shifting", dtype=torch.float64)
        state = sim.initial_state()
        # Start clear of the table, tilted around the y axis, with one edge receiving a lateral impulse.
        angle = .5
        rotation = state.pos.new_tensor([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                                         [-math.sin(angle), 0., math.cos(angle)]])
        state = state._replace(pos=state.pos @ rotation.T + sim.up * .5)
        weight = sim.mass.unsqueeze(-1) * GRAVITY * sim.up
        force = weight.clone()
        force[:, 3, 0] = 3.
        momentum = torch.zeros_like(force[:, 0])
        impacts, transfers, moments = [], [], []
        for i in range(90):
            applied = force if i < 28 else weight
            old = state
            result = sim.step(state, applied)
            state = result.state
            impulse = (applied - weight) * DT
            momentum = momentum + impulse.sum(dim=1)
            torch.testing.assert_close((sim.mass.unsqueeze(-1) * state.vel).sum(dim=1), momentum, atol=1e-9, rtol=1e-7)
            actual_impulse = sim.mass.unsqueeze(-1) * (state.vel - old.vel)
            internal = actual_impulse - impulse
            transfers.append(float(internal[:, :3, 0].sum()))
            moments.append(float(torch.linalg.cross(old.pos[:, 3] - sim.com(old), applied[:, 3] - weight[:, 3]).norm()))
            free = old.pos + DT * old.vel + DT * impulse / sim.mass.unsqueeze(-1)
            impacts.append(float((free[:, 3] - free[:, 0]).norm() - sim.diagonal[0, 0]))
        self.assertGreater(max(impacts), float(sim.slide[0, 0]))
        first_stop = next(i for i, d in enumerate(impacts) if d > float(sim.slide[0, 0]))
        self.assertGreater(first_stop, 3)
        self.assertGreater(max(transfers[first_stop:]), max(transfers[:3]))
        self.assertGreater(max(moments), .01)
        # At suspended equilibrium, uniform support per unit mass balances gravity, including eccentric mass.
        equilibrium = state._replace(pos=sim.initial_state().pos + .5 * sim.up, vel=torch.zeros_like(state.vel))
        suspended = sim.step(equilibrium, weight)
        torch.testing.assert_close(suspended.state.vel, torch.zeros_like(state.vel), atol=1e-10, rtol=0.)
        torch.testing.assert_close(weight.sum(dim=1)[:, 2], sim.params.total_mass[:, 0] * GRAVITY)
        self.assertEqual(float(suspended.support_N.sum()), 0.)
        print(f"P4: diagonal stop at {first_stop * DT:.3f} s, peak internal impulse {max(transfers):.6f} N s")

    @unittest.skipUnless(torch.cuda.is_available(), "P5 requires CUDA")
    @torch.inference_mode()
    def test_p5_cuda_100000_heterogeneous_steps(self):
        device = "cuda"
        count, steps = 32, 100_000
        gen = torch.Generator(device=device).manual_seed(102198)
        sampled = PayloadParamsX.sample(CONFIG, count, device=device, generator=gen)
        names = ("rigid_box", "folded_clothes", "fluffy_pillow", "heavy_quilt", "loose_shifting")
        anchors = PayloadParamsX.from_presets(CONFIG, names, device=device)
        params = replace(sampled, **{f.name: torch.cat((getattr(anchors, f.name), getattr(sampled, f.name)[len(names):]))
                                     for f in fields(sampled)})
        sim = UnifiedPayloadTemplate(params)
        state = sim.initial_state()
        force_table = (torch.randn((127, count, 4, 3), generator=gen, device=device) * 300.).clamp(-1200., 1200.)
        # This repeated pulse exercises acceleration, support, and release without indefinitely driving translation.
        force_table[1::2] *= -1.
        force_table -= force_table.mean(dim=0)
        depth_table = torch.rand((127, count, 2), generator=gen, device=device) * .35 - .10
        weights = torch.eye(4, device=device)[:2].expand(count, -1, -1)
        engaged = torch.ones((count, 2), dtype=torch.bool, device=device)
        cursor = torch.zeros(1, dtype=torch.long, device=device)
        valid = torch.ones((), dtype=torch.bool, device=device)
        min_clearance = torch.zeros((), device=device)
        max_span = torch.zeros((), device=device)
        def advance():
            force = force_table.index_select(0, cursor)[0]
            raw = depth_table.index_select(0, cursor)[0]
            surface = sim.surface_contact(state, raw, weights, engaged)
            result = sim.step(state, force - surface.block_force_N.unsqueeze(-1) * sim.up, surface.depth)
            for old, new in zip(state, result.state, strict=True):
                old.copy_(new)
                valid.logical_and_(new.isfinite().all())
            valid.logical_and_(((state.d_surf >= 0.) & (state.d_surf <= sim.depth_max)).all())
            valid.logical_and_(result.clearance.isfinite().all() & result.support_N.isfinite().all())
            min_clearance.copy_(torch.minimum(min_clearance, result.clearance.min()))
            max_span.copy_(torch.maximum(max_span, (state.pos - sim.com(state)[:, None]).norm(dim=2).max()))
            cursor.add_(1).remainder_(len(force_table))

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            advance()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            advance()
        for current, initial in zip(state, sim.initial_state(), strict=True):
            current.copy_(initial)
        cursor.zero_()
        valid.fill_(True)
        min_clearance.zero_()
        max_span.zero_()
        for _ in range(steps):
            graph.replay()
        self.assertEqual(cursor.item(), steps % len(force_table))
        self.assertTrue(valid.item())
        self.assertEqual(min_clearance.item(), 0.)
        self.assertLess(max_span.item(), 2.)
        print(f"P5: {count} × {steps} CUDA steps, minimum clearance {min_clearance.item():g} m, max internal radius {max_span.item():.4f} m")

    def test_top_anchor_power_matches_force_distribution(self):
        sim = model("fluffy_pillow", dtype=torch.float64)
        initial = sim.initial_state()
        gen = torch.Generator().manual_seed(561)
        vel = torch.randn(initial.vel.shape, generator=gen, dtype=initial.vel.dtype)
        state = initial._replace(pos=initial.pos + .01 * vel, vel=vel, d_surf=torch.full_like(initial.d_surf, .01))
        weights = sim.weights_at(state.pos.new_tensor([[[-.02, -.03], [.02, .035]]]))
        depth_rate = state.d_surf.new_tensor([[.10, -.02, .15, -.25]])
        force = state.pos.new_tensor([[[1., -3., 2.], [-2., 2., .5]]])
        anchor, speed = sim.anchors(state, weights, depth_rate)
        block_force, depth_force = sim.distribute_anchor_forces(state, weights, force)
        torch.testing.assert_close(block_force.sum(dim=1), force.sum(dim=1))
        torch.testing.assert_close((speed * force).sum(), (block_force * state.vel).sum() + (depth_force * depth_rate).sum())
        com = sim.com(state)[:, None]
        torch.testing.assert_close(torch.linalg.cross(anchor - com, force).sum(dim=1),
                                   torch.linalg.cross(state.pos - com, block_force).sum(dim=1))
        self.assertEqual(sim.get_privileged_state(state).shape, (1, 54))


def tensor_mean(value):
    return float(value.mean())
