"""Acceptance probes for the four-point payload and its frozen proxy coupling."""

from dataclasses import fields, replace
import json
import math
from pathlib import Path
import unittest

import torch

from benchmark_stage2_cuda import build_sim, timed_episode
from src.low_order_parcel.lumped_contact_proxy import ContactInputs, ContactState, LumpedContactProxy, ProxyParams
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.unified_payload import DT, GRAVITY, PayloadParamsX, UnifiedPayloadTemplate
from src.low_order_parcel.verify_stage2_alignment import airborne, both_captured, longest, verify

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/payload_template_X.yaml").read_bytes())
STAGE2 = json.loads((ROOT / "configs/proxy_material_envelopes.yaml").read_bytes())


def model(*names, **overrides):
    return UnifiedPayloadTemplate(PayloadParamsX.from_presets(CONFIG, names, **overrides))


def rigid_probe(nominal, dt, *, slack_m=None, com_offset=(0., 0., 0.)):
    values = nominal["proxy"].copy()
    if slack_m is not None:
        values["S_bag"] = slack_m
    params = ProxyParams.from_scalars(values, dtype=torch.float64)
    hx = CONFIG["defaults"]["bag_half_extents"][0]
    signs = params.S_bag.new_tensor([[-1., 1., -1., 1.]])
    mass_weights = .25 + signs * (com_offset[0] / (2. * hx))
    payload = model("rigid_box", dtype=torch.float64, mass_weights=mass_weights,
                    total_mass=params.S_bag.new_tensor([[nominal["mass_kg"]]]))
    return MacroParcelSim(LumpedContactProxy(params, dt), payload=payload)


def pe_proxy(batch):
    nominal = STAGE2["materials"]["pe"]["nominal"]["proxy"]
    base = ProxyParams.from_scalars(nominal, dtype=torch.float64)
    return LumpedContactProxy(ProxyParams(**{f.name: getattr(base, f.name).expand(batch, -1)
                                            for f in fields(base)}), DT)


def heterogeneous_probe(config, worlds):
    legacy, materials = build_sim(config, worlds)
    gen = torch.Generator(device=legacy.mass.device).manual_seed(1726481)
    params = PayloadParamsX.sample(CONFIG, worlds, device=legacy.mass.device, generator=gen)
    return MacroParcelSim(legacy.proxy, payload=UnifiedPayloadTemplate(params)), materials


def hover_grip(sim):
    # Allocate enough per-pad friction for the sampled weight during the nominal lift.
    return (2. * sim.mass * GRAVITY / sim.proxy.params.mu_eff.amin(dim=1, keepdim=True)).clamp_min(6.)


def hover_steering(sim, state, action, offset):
    relative = state.velocity - action.pinch_velocity.mean(dim=1)
    # Common pad motion damps the pendulum and the vertical spring mode.
    proxy = sim.proxy.params
    segment = proxy.L0.mean(dim=1, keepdim=True) + .5 * sim.mass * GRAVITY / proxy.k
    lateral_gain = 2. * torch.sqrt(segment / GRAVITY)
    vertical_gain = torch.sqrt(2. * sim.mass / proxy.k)
    lateral = .35 * state.com[:, :2] - lateral_gain * relative[:, :2]
    target = torch.cat((lateral, -vertical_gain * relative[:, 2:3]), dim=1).clamp(-.10, .10)
    return torch.lerp(offset, target, .18)


class UnifiedPayloadTest(unittest.TestCase):
    @torch.inference_mode()
    def test_p1_rigid_template_passes_frozen_gate_a(self):
        report, _ = verify(STAGE2, DT, sim_factory=rigid_probe)
        regimes = report["regimes"]
        print("P1:", {key: value["pass_gate"] for key, value in regimes.items()},
              {key: value["hover_s"] for key, value in regimes["3_side_hover"]["materials"].items()}, flush=True)
        self.assertTrue(report["gate_A_pass"], [key for key, value in regimes.items() if not value["pass_gate"]])
        self.assertEqual(len(regimes["1_top_press"]["cases"]), 12)
        for material, result in regimes["3_side_hover"]["materials"].items():
            self.assertGreaterEqual(result["hover_s"], 3., material)

    @torch.inference_mode()
    def test_p2_same_pe_bag_top_grasp_bifurcation(self):
        proxy = pe_proxy(3)
        like = proxy.params.S_bag
        payload = model("rigid_box", "folded_clothes", "fluffy_pillow", dtype=like.dtype,
                        bag_half_extents=like.new_tensor([[.085, .105, .050]]).expand(3, -1),
                        total_mass=like.new_full((3, 1), .24))
        sim = MacroParcelSim(proxy, payload=payload)
        state = sim.initial_state()
        z = like.new_full((3, 1), .115)
        feedback = torch.zeros_like(z)
        side = z.new_tensor([[-1., 1.]])
        zeros = z.new_zeros((3, 2))
        old_position = None
        values = []
        for i in range(2300):
            time = i * DT
            if time < 1.:
                # Same pressure servo for all three environments, sensing the actual wrist force.
                z = z + DT * (.006 * (feedback - 20.)).clamp(-.13, .065)
                press_z = z
            else:
                lift = min(1., (time - 1.) / 1.3)
                z = press_z + .5 * (1. - math.cos(math.pi * lift)) * (.315 - press_z)
            close = min(1., max(0., (time - .30) / .65))
            x = (.056 - .043 * close) * side.expand_as(zeros)
            pos = torch.stack((x, zeros, z.expand_as(x)), dim=-1)
            vel = torch.zeros_like(pos) if old_position is None else (pos - old_position) / DT
            old_position = pos
            inward = (-vel[:, :, 0] * side).clamp_min(0.)
            action = ParcelAction(pos, vel, torch.full_like(x, 6.), inward, zeros, zeros, pos[:, 0] * 0., z * 0.)
            result = sim.step(state, action)
            state = result.state
            feedback = result.wrist_wrench[:, 2:3]
            _, _, budget = payload.contact_geometry(state.payload.d_surf, sim.payload_mount_weights,
                                                    zeros, zeros, sim.base_slack)
            values.append((state.contacts[:, :, ContactState.CAPTURE], result.tension,
                           state.payload.d_surf, budget - sim.base_slack, result.table_clearance))
        capture, tension, indentation, slack_gain, clearance = (torch.stack(c) for c in zip(*values, strict=True))
        self.assertFalse(capture[:, 0].any() or tension[:, 0].any() or indentation[:, 0].any() or slack_gain[:, 0].any())
        self.assertTrue((indentation[:, 1:].amax(dim=(0, 2)) > .010).all())
        self.assertTrue((slack_gain[:, 1:].amax(dim=(0, 2)) > .005).all())
        self.assertTrue((capture[-100:, 1:] == 1.).all())
        self.assertTrue((clearance[-100:, 1:] > .10).all())
        side_trace = sim.run_side_grasp_and_hover().channels
        self.assertGreaterEqual(longest(airborne(side_trace) & both_captured(side_trace), DT), 3.)
        print("P2 same PE: max indentation (mm)", (1000. * indentation.amax(dim=(0, 2))).tolist(),
              "max extra slack (mm)", (1000. * slack_gain.amax(dim=(0, 2))).tolist(),
              "final clearance (mm)", (1000. * clearance[-1, :, 0]).tolist(), flush=True)

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
        asymmetric_depth = overpress.depth.clone()
        asymmetric_depth[:, 0] = 0.
        _, _, asymmetric_budget = sim.contact_geometry(asymmetric_depth, weights, zeros, zeros, slack)
        width = 2. * sim.params.bag_half_extents[:, :1]
        perimeter_drop = width + dmax - torch.hypot(width, dmax)
        self.assertTrue((asymmetric_budget - slack <= perimeter_drop + 1e-8).all())
        moderate = sim.surface_contact(state, torch.full((2, 2), .018), weights, engaged)
        feed, cosine, budget = sim.contact_geometry(moderate.depth, weights, zeros, zeros, slack)
        self.assertTrue((feed[0] < .020).all())
        nominal = STAGE2["materials"]["pe"]["nominal"]
        base = ProxyParams.from_scalars(nominal["proxy"])
        params = ProxyParams(**{f.name: getattr(base, f.name).expand(2, -1) for f in fields(base)})
        proxy = LumpedContactProxy(params, DT)
        xyz = torch.zeros((2, 2, 3))
        inputs = ContactInputs(xyz, xyz, xyz, xyz, xyz[:, 0], torch.ones_like(feed),
                               torch.full_like(feed, .04), feed, cosine, budget)
        contacts = proxy.initial_state()
        proxy.validate_inputs(contacts, inputs)
        loaded_contact = contacts.clone()
        loaded_contact[:, :, ContactState.SLACK] = .004
        default = inputs._replace(S_bag_eff=None)
        for a, b in zip(proxy.step(loaded_contact, default), proxy.step(loaded_contact, inputs._replace(S_bag_eff=params.S_bag)),
                        strict=True):
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
        starved = proxy.step(loaded_contact, inputs._replace(S_bag_eff=torch.zeros_like(budget)))
        self.assertEqual(float(starved.state[:, :, ContactState.SLACK].max()), 0.)
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
        parcel = MacroParcelSim(proxy, payload=sim)
        initial = parcel.initial_state()
        press = initial.payload.pos[:, :2].clone()
        press[:, :, 2] = -.15
        action = ParcelAction(press, torch.zeros_like(press), zeros, zeros, zeros, zeros, xyz[:, 0], slack * 0.)
        pressed = parcel.step(initial, action)
        self.assertTrue((pressed.wrist_wrench[:, 2] > 0.).all())
        self.assertTrue((pressed.support_N[:, 0] > sim.params.total_mass[:, 0] * GRAVITY).all())
        torch.testing.assert_close(proxy.params.S_bag, base.S_bag.expand_as(budget))
        print(f"P3: settled force {torch.stack(forces).tolist()} N, rebound ratio {float(loaded.d_surf[0, 0] / dmax[0, 0]):.3f}")

    @torch.inference_mode()
    def test_p3_head_on_side_approach_pushes_the_light_parcel(self):
        proxy = pe_proxy(2)
        like = proxy.params.S_bag
        sim = MacroParcelSim(proxy, payload=model("rigid_box", "rigid_box", dtype=like.dtype,
                             total_mass=like.new_full((2, 1), .065), table_friction=like.new_full((2, 1), .06)))
        state = sim.initial_state()
        side = like.new_tensor([[-1., 1.]])
        heights = like.new_tensor([[.012], [.045]])
        old_position = None
        net_feed, push_force, capture = [], [], []
        for i in range(245):
            t = min(i * DT, .34)
            x = (.030 - .025 * t / .34) * side.expand(2, -1)
            y = torch.full_like(x, -.107 + .12 * t)
            pos = torch.stack((x, y, heights.expand_as(y)), dim=-1)
            vel = torch.zeros_like(pos) if old_position is None else (pos - old_position) / DT
            old_position = pos
            nominal_feed = (y + .10).clamp_min(0.)
            action = ParcelAction(pos, vel, torch.full_like(x, 6.), (-vel[:, :, 0] * side).clamp_min(0.),
                                  nominal_feed, torch.ones_like(x), state.com * 0., like * 0.)
            result = sim.step(state, action)
            state = result.state
            net_feed.append(state.contacts[:, :, ContactState.FEED_DEPTH])
            capture.append(state.contacts[:, :, ContactState.CAPTURE])
            push_force.append(-result.wrist_wrench[:, 1])
        observed_feed, capture, peak_force = (torch.stack(net_feed), torch.stack(capture),
                                              torch.stack(push_force).amax(dim=0))
        self.assertGreater(float(peak_force[0]), .06 * .065 * GRAVITY)
        self.assertGreater(float(state.com[0, 1]), float(nominal_feed[0, 0]) - .005)
        self.assertLess(float(observed_feed[:, 0].max()), float(proxy.params.h_min[0, 0]))
        self.assertFalse(capture[:, 0].any())
        self.assertTrue((capture[-1, 1] == 1.).all())
        self.assertTrue((result.table_clearance[0] == 0.).all())
        print(f"P3 side push: {float(state.com[0, 1]) * 1000.:.2f} mm parcel travel, "
              f"{float(observed_feed[:, 0].max()) * 1000.:.3f} mm max net feed, {float(peak_force[0]):.3f} N peak reaction")

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

        nominal = STAGE2["materials"]["pe"]["nominal"]["proxy"]
        base = ProxyParams.from_scalars(nominal, dtype=torch.float64)
        balanced_proxy = ProxyParams(**{f.name: getattr(base, f.name).expand(2, -1) for f in fields(base)})
        payload = model("loose_shifting", "loose_shifting", dtype=torch.float64,
                        slide_slack_max=base.S_bag.new_tensor([[.04], [0.]]))
        parcel = MacroParcelSim(LumpedContactProxy(balanced_proxy, DT), payload=payload)
        times, _, _, actions = parcel.probe_actions("side", grip_N=hover_grip(parcel))
        state = parcel.initial_state()
        correction = torch.zeros_like(state.com)
        history, shifts, settled = [], [], []
        for i in range(len(times)):
            action = ParcelAction(*(c[i] for c in actions))
            target = hover_steering(parcel, state, action, correction)[:1].expand_as(correction)
            phase = min(1., max(0., (i * DT - .55) / .70))
            tilt = .02 * (1. - math.cos(2. * math.pi * phase))
            rate = .04 * math.pi / .70 * math.sin(2. * math.pi * phase)
            side = state.com.new_tensor([[-1.], [1.]]) * parcel.up[:, None]
            action = action._replace(pinch_position=action.pinch_position + target[:, None] + tilt * side,
                                     pinch_velocity=action.pinch_velocity + ((target - correction) / DT)[:, None] + rate * side)
            correction = target
            out = parcel.step(state, action)
            state = out.state
            shifts.append((state.payload.pos[:, 3] - state.payload.pos[:, 0]).norm(dim=1) - payload.diagonal[:, 0])
            history.append(out.wrist_wrench)
            settled.append((out.support_N[0, 0] == 0.) & (out.table_clearance[0, 0] > .001))
        wrench = torch.stack(history)
        shift = torch.stack(shifts)
        impact = torch.nonzero(shift[:, 0] >= .99 * payload.slide[0, 0])[0, 0].item()
        self.assertGreater(impact * DT, .55)
        self.assertLess(float(shift[:, 1].abs().max()), .002)
        diff = wrench[275:1000, 0] - wrench[275:1000, 1]
        self.assertGreater(float(diff[:, :3].norm(dim=1).max()), .1)
        self.assertGreater(float(diff[:, 3:].norm(dim=1).max()), 1e-4)
        self.assertTrue(torch.stack(settled)[-100:].all())
        expected = -(parcel.mass[0, 0] * GRAVITY) * parcel.up[0]
        hover_wrench = wrench[-100:, 0, :3]
        torch.testing.assert_close(hover_wrench, expected.expand_as(hover_wrench), atol=1e-3, rtol=0.)
        print(f"P4 coupled: delayed stop {impact * DT:.3f} s, "
              f"hover force {float(-hover_wrench[:, 2].mean()):.6f} N, weight {float(-expected[2]):.6f} N")

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
        dense, loose = model("rigid_box", "rigid_box",
                             fill_ratio=torch.tensor([[.95], [.30]])).params.fill_slack_capacity()
        self.assertLess(float(dense), .034)
        self.assertGreater(float(loose), .034)

    @unittest.skipUnless(torch.cuda.is_available() and "5090" in torch.cuda.get_device_name(), "P6 requires an RTX 5090")
    @torch.inference_mode()
    def test_p6_rtx_5090_8192_heterogeneous_full_episodes(self):
        result = timed_episode(STAGE2, 8192, {}, make_sim=heterogeneous_probe, grip_N=hover_grip, steer=hover_steering)
        result["all_worlds_met_hold"] = result.pop("passed")
        print(json.dumps(result, allow_nan=False), flush=True)
        self.assertEqual(result["steps_per_world"], 2385)
        self.assertEqual(result["nonfinite_worlds"], 0)
        self.assertEqual(result["invalid_geometry_worlds"], 0)
        self.assertLessEqual(result["wall_s"], 10. * 1.119)
        self.assertGreaterEqual(result["verified_sim_seconds_per_wall_second"], 8192 * 4.77 / (10. * 1.119))


def tensor_mean(value):
    return float(value.mean())
