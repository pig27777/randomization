"""Step 1 physics and sensor gates, with the same float32 probes on CPU and CUDA."""

from dataclasses import fields
import json
import math
import os
from pathlib import Path
import unittest

import torch

from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction, DualArmParcelSim
from src.low_order_parcel.hardware_filters import YAMCurrentEstimatorFilter, YAMParameters
from src.low_order_parcel.lumped_contact_proxy import ContactInputs, ContactState, LumpedContactProxy, ProxyParams
from src.low_order_parcel.macro_parcel_sim import ParcelAction
from src.low_order_parcel.unified_payload import DT, PayloadParamsX, UnifiedPayloadTemplate
from src.low_order_parcel.virtual_rgb import CameraPose, RGBParameters, VirtualRGBProjector

ROOT = Path(__file__).resolve().parents[2]
MATERIAL = json.loads((ROOT / "configs/proxy_material_envelopes.yaml").read_bytes())["materials"]["pe"]["nominal"]["proxy"]
PRESETS = json.loads((ROOT / "configs/payload_template_X.yaml").read_bytes())
DEVICE = torch.device(os.environ.get("STAGE3_TEST_DEVICE", "cpu"))


def tensor(value):
    return torch.tensor(value, device=DEVICE, dtype=torch.float32)


def fixture(*presets, F_half=0., T_tear=40., mass=None, mounts=None, extents=None, **overrides):
    base = ProxyParams.from_scalars(MATERIAL | overrides, device=DEVICE)
    params = ProxyParams(*(getattr(base, field.name).expand(len(presets), -1) for field in fields(base)))
    kwargs = {} if mass is None else {"total_mass": tensor(mass).expand(len(presets), 1)}
    if extents is not None:
        kwargs["bag_half_extents"] = tensor(extents).expand(len(presets), 3)
    payload = UnifiedPayloadTemplate(PayloadParamsX.from_presets(PRESETS, presets, device=DEVICE, **kwargs))
    return DualArmParcelSim(LumpedContactProxy(params, DT), LumpedContactProxy(params, DT),
                            payload, F_half=F_half, T_tear=T_tear, mount_xy=mounts,
                            generator=torch.Generator(device=DEVICE).manual_seed(20261121))


def action(pos, *, velocity=None, normal=6., inward=0., feed=0., press=0.):
    pads = pos[..., 0]
    return ParcelAction(pos, torch.zeros_like(pos) if velocity is None else velocity,
                        torch.broadcast_to(torch.as_tensor(normal, device=pos.device), pads.shape).to(pos.dtype),
                        torch.full_like(pads, inward), torch.full_like(pads, feed), torch.ones_like(pads),
                        torch.zeros_like(pos[:, 0]), torch.zeros_like(pads[:, :1]) + press)


def pair(arm_A, arm_B=None):
    enabled = torch.tensor([True, arm_B is not None], device=DEVICE).expand(len(arm_A.pinch_position), -1)
    return DualArmAction(arm_A, arm_A if arm_B is None else arm_B, enabled)


def contact_state(proxy, slack, capture):
    state = proxy.initial_state()
    state[..., ContactState.SLACK] = slack
    state[..., ContactState.CAPTURE] = capture
    state[..., ContactState.HOLD_TIME] = .10
    return state


def clean_yam(batch, **overrides):
    one = tensor(1.).expand(batch, 2, 1)
    params = YAMParameters(one, 0. * one, 0. * one, 15. * one, 0. * one,
                           torch.zeros_like(one, dtype=torch.long), .1 * one, one, .001 * one)
    return YAMCurrentEstimatorFilter(params._replace(**overrides),
                                      generator=torch.Generator(device=DEVICE).manual_seed(51))


def camera(position, frame, batch):
    return CameraPose(tensor(position).expand(batch, -1), tensor(frame).expand(batch, -1, -1))


TOP = ((1., 0., 0.), (0., -1., 0.), (0., 0., -1.))
WRIST = ((1., 0., 0.), (0., 0., 1.), (0., -1., 0.))


def rgb_projector(payload, *, frame_period_steps=1, **overrides):
    batch = len(payload.mass)
    zeros = tensor(0.).expand(batch, 2, 1)
    params = RGBParameters(zeros.expand(-1, -1, 3), zeros.expand(-1, -1, 3), zeros, zeros,
                           zeros.long(), zeros[:, 0] + .12, zeros[:, 0] + math.radians(75))
    return VirtualRGBProjector(payload, camera((.05, .20, .75), TOP, batch), params._replace(**overrides),
                               table_bounds=tensor((-.4, .4, -.4, .4)).expand(batch, -1),
                               frame_period_steps=frame_period_steps,
                               generator=torch.Generator(device=DEVICE).manual_seed(52))


class DualArmTests(unittest.TestCase):
    @torch.inference_mode()
    def test_zero_half_force_matches_frozen_proxy(self):
        sim = fixture("rigid_box", "rigid_box")
        state = sim.initial_state()
        current = state.payload
        weights = sim.mount_weights[:, :2]
        anchor, anchor_vel = sim.payload.anchors(current, weights, torch.zeros_like(current.d_surf))
        prior = contact_state(sim.proxy_A, .003, 1.)
        state = state._replace(contacts_A=prior)
        pos = anchor + tensor([0., 0., .070])
        normal = tensor([[.25, .25], [6., 6.]])
        cmd = action(pos, normal=normal, inward=.05, feed=.01)
        frozen = sim.proxy_A.step(prior, ContactInputs(
            pos, anchor, cmd.pinch_velocity, anchor_vel, sim.payload.com(current), normal,
            cmd.inward_velocity, cmd.feed_depth, cmd.alignment_cos, sim.base_slack))
        result = sim.step(state, pair(cmd))
        torch.testing.assert_close(result.state.contacts_A, frozen.state, atol=0., rtol=0.)
        torch.testing.assert_close(result.tension[:, 0], frozen.tension, atol=0., rtol=0.)
        torch.testing.assert_close(result.wrist_wrench[:, 0, :3], -frozen.force.sum(dim=1), atol=0., rtol=0.)
        self.assertFalse(result.state.contacts_B.any())
        self.assertEqual(sim.get_privileged_state(result.state).shape, (2, 54))
        print("F_half=0: contact state, tension and net wrist force match exactly", flush=True)

    @torch.inference_mode()
    def test_same_clothes_light_vs_heavy_press(self):
        sim = fixture("folded_clothes", "folded_clothes", F_half=1., mass=.20)
        state = sim.initial_state()
        pressure = tensor([[.3], [3.]])
        z = tensor(.058).expand(2, 1)
        filtered_force = torch.zeros_like(z)
        signs = tensor([[-1., 1.]])
        old_position = None
        press_samples = []
        max_indentation = torch.zeros_like(z)
        for step in range(1450):
            t = step * DT
            if t < 1.45:
                z = z + DT * (.05 * (filtered_force - pressure)).clamp(-.06, .06)
                press_z = z
            else:
                phase = min(1., (t - 1.45) / 1.05)
                z = press_z + .5 * (1. - math.cos(math.pi * phase)) * .135
            close = min(1., max(0., (t - 1.15) / .22))
            x = ((.027 - .008 * close) * signs).expand(2, -1)
            pos = torch.stack((x, torch.zeros_like(x), z.expand_as(x)), dim=-1)
            vel = torch.zeros_like(pos) if old_position is None else (pos - old_position) / DT
            old_position = pos
            cmd = action(pos, velocity=vel, inward=.008 / .22 if 1.15 < t < 1.37 else 0., feed=.004)
            result = sim.step(state, pair(cmd))
            state = result.state
            filtered_force = torch.lerp(filtered_force, result.press_N[:, :1], .06)
            if 1.15 < t < 1.37:
                press_samples.append(result.press_N[:, :1])
                max_indentation = torch.maximum(max_indentation, state.payload.d_surf.amax(dim=1, keepdim=True))
        measured_press = torch.stack(press_samples).mean(dim=0)
        print(f"Clothes: press {measured_press.squeeze(1).tolist()} N, clearance "
              f"{(1000. * result.table_clearance).squeeze(1).tolist()} mm, captures "
              f"{state.contacts_A[..., ContactState.CAPTURE].tolist()}, indentation "
              f"{(1000. * max_indentation).flatten().tolist()} mm", flush=True)
        torch.testing.assert_close(measured_press, pressure, atol=.12, rtol=0.)
        self.assertFalse(state.contacts_A[0, :, ContactState.CAPTURE].any())
        self.assertTrue(state.contacts_A[1, :, ContactState.CAPTURE].bool().all())
        self.assertGreater(result.table_clearance[1].item(), .001)
        self.assertEqual(result.support_N[1].item(), 0.)

    @torch.inference_mode()
    def test_tear_is_per_finger_immediate_and_latched(self):
        sim = fixture("rigid_box", "rigid_box", "rigid_box", T_tear=tensor([[20.], [60.], [60.]]),
                      k=4000., c_d=0., beta=0.)
        state = sim.initial_state()._replace(contacts_A=contact_state(sim.proxy_A, .004, 1.))
        anchors, vel = sim.payload.anchors(state.payload, sim.mount_weights[:, :2], state.payload.d_surf)
        extension = tensor([[.010, .002], [.0859925, .002], [.010, .002]])
        pos = anchors + (sim.proxy_A.params.L0 + extension).unsqueeze(-1) * sim.payload.up
        cmd = action(pos, feed=.01)
        trial = sim.proxy_A.step(state.contacts_A, ContactInputs(
            pos, anchors, cmd.pinch_velocity, vel, sim.payload.com(state.payload), cmd.normal_force,
            cmd.inward_velocity, cmd.feed_depth, cmd.alignment_cos, sim.base_slack))
        result = sim.step(state, pair(cmd))
        self.assertAlmostEqual(trial.tension[1, 0].item(), 343.97, places=2)
        self.assertEqual(result.state.torn_flag[:, 0].tolist(), [[True, False], [True, False], [False, False]])
        self.assertFalse(result.tension[:2, 0, 0].any())
        self.assertFalse(result.state.contacts_A[:2, 0].any())
        torch.testing.assert_close(result.wrist_wrench[:2, 0, 2], -trial.tension[:2, 1])
        self.assertTrue((result.tension <= sim.T_tear[:, None]).all())
        again = sim.step(result.state, pair(cmd))
        self.assertTrue(again.state.torn_flag[0, 0, 0])
        self.assertEqual(again.tension[0, 0, 0], 0.)
        reset = sim.reset(again.state, torch.tensor([True, True, False], device=DEVICE))
        self.assertFalse(reset.torn_flag.any())
        self.assertTrue((sim.T_tear >= 20.).all() & (sim.T_tear <= 60.).all())
        self.assertEqual(sim.T_tear[2].item(), 60.)
        torch.testing.assert_close(reset.payload.pos[2], again.state.payload.pos[2])
        print(f"Tear: frozen {trial.tension.tolist()} N, transmitted {result.tension[:, 0].tolist()} N", flush=True)

    @torch.inference_mode()
    def test_simultaneous_shared_slack(self):
        sim = fixture("rigid_box", "rigid_box", S_bag=.00015)
        state = sim.initial_state()
        start = state.contacts_A.clone()
        start[1, :, ContactState.SLACK] = .00006
        state = state._replace(contacts_A=start)
        xy = tensor([[-.024, 0.], [.024, 0.]]).expand(2, -1, -1)
        pos = torch.cat((xy, torch.full_like(xy[..., :1], .050)), dim=-1)
        cmd = action(pos, inward=.075, feed=.01)
        max_slack = torch.zeros_like(sim.base_slack)
        for _ in range(12):
            previous = torch.stack((state.contacts_A, state.contacts_B), dim=1)[..., ContactState.SLACK]
            result = sim.step(state, pair(cmd, cmd))
            state = result.state
            expected = (result.S_bag_total_eff - previous.sum(dim=-1).flip(dims=(1,))).clamp_min(0.)
            torch.testing.assert_close(result.S_bag_available, expected)
            held = state.contacts_A[..., ContactState.SLACK] + state.contacts_B[..., ContactState.SLACK]
            max_slack = torch.maximum(max_slack, held.sum(dim=1, keepdim=True))
            self.assertTrue((held.sum(dim=1, keepdim=True) <= result.S_bag_total_eff + 1e-9).all())
        torch.testing.assert_close(state.contacts_A[0], state.contacts_B[0])
        self.assertGreater(max_slack.min().item(), .00014)
        print(f"Shared reserve: maximum {max_slack.flatten().tolist()} m", flush=True)

        # Slip within a step consumes the same reserve from which both arms collect film.
        sim = fixture("rigid_box", S_bag=.00015, k=4000., beta=.1)
        state = sim.initial_state()._replace(contacts_A=contact_state(sim.proxy_A, .00003, 1.),
                                            contacts_B=contact_state(sim.proxy_B, .00003, 1.))
        anchors, _ = sim.payload.anchors(state.payload, sim.mount_weights, state.payload.d_surf)
        tips = (anchors + tensor([0., 0., .08])).unflatten(1, (2, 2))
        result = sim.step(state, pair(action(tips[:, 0], inward=.075), action(tips[:, 1], inward=.075)))
        after = result.state.contacts_A + result.state.contacts_B
        film = after[..., ContactState.SLACK].sum() + after[..., ContactState.SLIP_CUM].sum()
        self.assertGreater(after[..., ContactState.SLIP_CUM].sum().item(), .00012)
        self.assertLessEqual(film.item(), result.S_bag_total_eff.item() + 1e-9)
        print(f"Shared reserve: retained plus withdrawn {film.item():.6f} m", flush=True)

    @torch.inference_mode()
    def test_airborne_side_push_reduces_net_feed(self):
        sim = fixture("rigid_box", mass=.065)
        state = sim.initial_state()
        state = state._replace(payload=state.payload._replace(pos=state.payload.pos + tensor([0., 0., .30])))
        first_center = sim.payload.com(state.payload)
        peak_force = tensor(0.)
        peak_crush = tensor(0.)
        cmd = None
        for step in range(50):
            y = -.103 + .15 * (step * DT)
            pos = tensor([[[-.024, y, .316], [.024, y, .316]]])
            cmd = action(pos, velocity=tensor([[[0., .15, 0.], [0., .15, 0.]]]),
                         normal=0., feed=max(0., y + .10))
            # Stand in for Arm B's partner holding altitude without constraining planar swing.
            lift = tensor([0., 0., .065 * 9.81])
            cmd = cmd._replace(external_force=lift[None])
            result = sim.step(state, pair(cmd))
            peak_force = torch.maximum(peak_force, -result.wrist_wrench[0, 0, 1])
            peak_crush = torch.maximum(peak_crush, (result.state.payload.pos[..., 0]
                                                   - sim.payload.rest[..., 0]).abs().max())
            state = result.state
        travel = (sim.payload.com(state.payload) - first_center)[0, 1]
        self.assertGreater(travel.item(), .001)
        self.assertGreater(peak_force.item(), .05)
        self.assertLess(peak_crush.item(), .001)
        self.assertLess(result.net_feed[0, 0, 0].item(), cmd.feed_depth[0, 0].item())
        self.assertEqual(result.support_N.item(), 0.)
        print(f"Air-side push: {1000. * travel.item():.3f} mm travel, {peak_force.item():.3f} N; "
              f"net/nominal {1000. * result.net_feed[0, 0, 0].item():.3f}/"
              f"{1000. * cmd.feed_depth[0, 0].item():.3f} mm", flush=True)

    @torch.inference_mode()
    def test_65g_active_tug_survives_worst_coulomb_and_delay(self):
        mounts = tensor([[[[-.024, 0.], [.024, 0.]], [[-.024, 0.], [.024, 0.]]]])
        sim = fixture("rigid_box", mass=.065, mounts=mounts, extents=(.075, .050, .016), k=380., c_d=2.)
        state = sim.initial_state()
        state = state._replace(payload=state.payload._replace(pos=state.payload.pos + tensor([0., 0., .16])),
                               contacts_A=contact_state(sim.proxy_A, .003, 1.),
                               contacts_B=contact_state(sim.proxy_B, .003, 1.))
        sensor = clean_yam(1, sensor_gain=tensor(.75).expand(1, 2, 1),
                           noise_N=tensor(.6).expand(1, 2, 1), coulomb_N=tensor(2.).expand(1, 2, 1),
                           cutoff_hz=tensor(4.).expand(1, 2, 1),
                           delay_steps=torch.full((1, 2, 1), 25, device=DEVICE, dtype=torch.long))
        xy = tensor([[-.024, .068], [.024, .068]])
        directions = tensor([[[-1.], [1.]]])
        zero = tensor(0.).expand(1, 2, 3)
        gripper = tensor(16.).expand(1, 2, 1)
        opening = tensor(.048).expand_as(gripper)
        true_peak, sensed_peak = (tensor(0.).expand(1, 2) for _ in range(2))
        min_clearance = tensor(1.)
        pulse_length = longest_pulse = tensor(0.)
        for step in range(225):
            t = min(step * DT / .27, 1.)
            offset = .0036 * .5 * (1. - math.cos(math.pi * t))
            velocity = .0036 * .5 * math.pi / .27 * math.sin(math.pi * t)
            y = directions * (xy[:, 1] + offset)
            pos = torch.stack((xy[:, 0].expand_as(y), y, torch.full_like(y, .201)), dim=-1)
            v = torch.zeros_like(pos)
            v[..., 1] = (directions * velocity).expand_as(y)
            cmd = [action(pos[:, i], velocity=v[:, i], normal=16., feed=.01) for i in range(2)]
            out = sim.step(state, pair(*cmd))
            state = out.state
            force = out.wrist_wrench[..., :3]
            sensed = sensor.step(force, v.mean(dim=2), zero, gripper, opening, pos.mean(dim=2), pos.mean(dim=2))
            true_peak = torch.maximum(true_peak, (force[..., 1] * -directions.squeeze(-1)))
            sensed_pull = sensed.force[..., 1] * -directions.squeeze(-1)
            sensed_peak = torch.maximum(sensed_peak, sensed_pull)
            pulse_length = torch.where((sensed_pull > 3.5).all(), pulse_length + 1, 0.)
            longest_pulse = torch.maximum(longest_pulse, pulse_length)
            min_clearance = torch.minimum(min_clearance, out.table_clearance.min())
        print(f"65 g tug: actual {true_peak.tolist()} N, observed {sensed_peak.tolist()} N, "
              f"simultaneous pulse {1000. * DT * longest_pulse:.0f} ms, "
              f"minimum clearance {1000. * min_clearance:.2f} mm", flush=True)
        self.assertTrue((true_peak > 3.5).all())
        self.assertTrue((true_peak <= 5.5).all())
        self.assertTrue((sensed_peak > 3.5).all())
        self.assertGreaterEqual(longest_pulse.item(), 10)
        self.assertGreater(min_clearance.item(), .001)
        self.assertFalse(state.torn_flag.any())
        self.assertTrue((state.contacts_A[..., ContactState.CAPTURE] == 1.).all())
        self.assertTrue((state.contacts_B[..., ContactState.CAPTURE] == 1.).all())


class HardwareTests(unittest.TestCase):
    @torch.inference_mode()
    def test_yam_hysteresis_independent_gripper_delay_and_reset(self):
        delays = torch.tensor([[[5], [25]], [[0], [0]]], device=DEVICE)
        sensor = clean_yam(2, delay_steps=delays, coulomb_N=tensor(2.).expand(2, 2, 1),
                           pose_bias_N=tensor(1.5).expand(2, 2, 1))
        zeros = tensor(0.).expand(2, 2, 3)
        v = tensor([.05, 0., 0.]).expand_as(zeros)
        gravity = tensor([0., 0., -1.]).expand_as(zeros)
        open_at_reset = tensor(.040).expand_as(delays)
        commanded = tensor([.01, .02, .10]).expand_as(zeros)
        actual = commanded + tensor([.001, 0., -.004])
        opening, force = open_at_reset, tensor(.08).expand_as(delays)
        for step in range(60):
            if step == 1:
                opening = open_at_reset - .0005
            if step == 15:
                opening = open_at_reset - .004
                force = force + .32
            reading = sensor.step(zeros, v, gravity, force, opening, actual, commanded)
            if step in (4, 5, 24, 25):
                self.assertEqual(reading.valid[0, :, 0].tolist(), [step >= 5, step >= 25])
            if step == 6:
                torch.testing.assert_close(reading.opening[0, 0], open_at_reset[0, 0])
            if step == 20:
                torch.testing.assert_close(reading.gripper_current[0, :, 0], tensor([.3, 0.]))
        self.assertGreater(reading.force[0, 0, 0].item(), 1.99)
        self.assertLess(reading.force[0, 0, 2].item(), -1.49)
        torch.testing.assert_close(reading.opening[0], tensor([[.037], [.037]]))
        torch.testing.assert_close(reading.position_error[0, 0], tensor([.001, 0., -.004]))

        switched = sensor.step(zeros, -v, gravity, force, opening, actual, commanded)
        self.assertGreater(switched.force[1, 0, 0].item(), 0.)
        for _ in range(55):
            switched = sensor.step(zeros, -v, gravity, force, opening, actual, commanded)
        self.assertLess(switched.force[0, 1, 0].item(), -1.98)
        new_noise = tensor(.3).expand_as(sensor.params.noise_N)
        sensor.reset(torch.tensor([True, False], device=DEVICE), params=sensor.params._replace(noise_N=new_noise))
        torch.testing.assert_close(sensor.params.noise_N[:, 0, 0], tensor([.3, 0.]))
        reset = sensor.step(zeros, -v, gravity, force, opening, actual, commanded)
        self.assertFalse(reset.valid[0].any())
        self.assertFalse(reset.force[0].any())
        self.assertTrue(reset.valid[1].all())
        self.assertLess(reset.force[1, 1, 0].item(), -1.99)

        for step in range(26):
            p, _ = sensor.impedance_step(zeros, zeros, commanded, zeros, zeros)
            self.assertEqual((p[0, :, 2] > 0.).tolist(), [step >= 5, step >= 25])
        self.assertTrue((p[1, :, 2] > 0.).all())
        print("YAM: 10/50 ms independent delay, ±2 N reversal, pose drift, encoder and gripper checks", flush=True)

    @torch.inference_mode()
    def test_sensor_parameter_draws_cover_declared_bounds_independently(self):
        reference = tensor(0.).expand(256, 1)
        seed = torch.Generator(device=DEVICE).manual_seed(53)
        yaw = YAMParameters.sample(reference, generator=seed)
        rgb = RGBParameters.sample(reference, generator=seed)
        for value, lower, upper in ((yaw.sensor_gain, .75, 1.25), (yaw.coulomb_N, .5, 2.),
                                    (yaw.pose_bias_N, -1.5, 1.5), (yaw.cutoff_hz, 4., 15.),
                                    (yaw.noise_N, .2, .6), (yaw.delay_steps, 5, 25),
                                    (yaw.gripper_deadzone_N, .05, .25), (yaw.gripper_gain, .8, 1.2),
                                    (yaw.opening_deadzone_m, .0005, .002),
                                    (rgb.translation_m, -.005, .005),
                                    (rgb.rotation_rad, -math.radians(1.5), math.radians(1.5)),
                                    (rgb.mask_jitter, 0., .12), (rgb.drop_probability, .05, .15),
                                    (rgb.delay_steps, 30, 60), (rgb.wrist_min_range_m, .10, .20),
                                    (rgb.wrist_fov_rad, math.radians(65.), math.radians(85.))):
            self.assertTrue(((value >= lower) & (value <= upper)).all())
        self.assertFalse(torch.equal(yaw.sensor_gain[:, 0], yaw.sensor_gain[:, 1]))
        self.assertFalse(torch.equal(rgb.delay_steps[:, 0], rgb.delay_steps[:, 1]))

    @torch.inference_mode()
    def test_rgb_top_parallax_lift_and_fk_depth(self):
        sim = fixture("rigid_box", "rigid_box")
        projector = rgb_projector(sim.payload)
        state = sim.initial_state().payload
        wrist = camera((0., -.30, .045), WRIST, 2)
        fk = tensor([0., -.010, .034]).expand(2, -1)
        lift = torch.ones(2, dtype=torch.bool, device=DEVICE)
        initial = projector.step(state, wrist, fk, lift)
        self.assertEqual(projector.vertices(state).shape, (2, 32, 3))
        self.assertTrue(initial.top_valid.all() & initial.wrist_vis_valid.all())
        self.assertTrue((initial.top_box[:, 2] > .20).all())
        self.assertTrue((initial.top_box[:, 1] < 0.).all())
        torch.testing.assert_close(initial.wrist_target_rel[:, 2], tensor([.29, .29]))
        self.assertTrue((initial.wrist_profile > 0.).all())
        self.assertTrue((initial.top_clearance > 0.).all())
        rise = tensor([[0., 0., .025], [0., 0., 0.]])
        moved = state._replace(pos=state.pos + rise[:, None])
        after = projector.step(moved, wrist, fk + tensor([0., 0., .025]), lift)
        self.assertGreater(after.vis_lift_ratio[0].item(), .85)
        self.assertEqual(after.vis_lift_ratio[1].item(), 0.)
        self.assertGreater(after.top_box[0, 2].item(), initial.top_box[0, 2].item())
        self.assertLess(after.wrist_sway_est[0, 1].item(), -.020)
        print(f"RGB: off-axis lift ratios {after.vis_lift_ratio.flatten().tolist()}, "
              f"wrist depths {after.wrist_target_rel[:, 2].tolist()} m", flush=True)

    @torch.inference_mode()
    def test_rgb_delay_dropout_near_field_fov_and_partial_reset(self):
        sim = fixture("rigid_box", "rigid_box")
        delays = torch.tensor([[[30], [60]], [[0], [0]]], device=DEVICE)
        projector = rgb_projector(sim.payload, delay_steps=delays)
        state = sim.initial_state().payload
        wrist = camera((0., -.30, .045), WRIST, 2)
        fk = tensor([0., 0., .032]).expand(2, -1)
        lift = torch.zeros(2, dtype=torch.bool, device=DEVICE)
        for step in range(61):
            initial = projector.step(state, wrist, fk, lift)
            self.assertEqual(initial.top_valid[:, 0].tolist(), [step >= 30, True])
            self.assertEqual(initial.wrist_vis_valid[:, 0].tolist(), [step >= 60, True])
        near = camera((0., -.07, .032), WRIST, 2)
        blind = projector.step(state, near, fk, lift)
        self.assertEqual(blind.wrist_vis_valid[:, 0].tolist(), [True, False])
        torch.testing.assert_close(blind.wrist_target_rel, initial.wrist_target_rel)
        out_of_fov = camera((1., -.30, .045), WRIST, 2)
        blind = projector.step(state, out_of_fov, fk, lift)
        self.assertFalse(blind.wrist_vis_valid[1])
        torch.testing.assert_close(blind.wrist_target_rel[1], initial.wrist_target_rel[1])
        projector.reset(torch.tensor([True, False], device=DEVICE))
        reset = projector.step(state, wrist, fk, lift)
        self.assertEqual(reset.top_valid[:, 0].tolist(), [False, True])
        self.assertEqual(reset.wrist_vis_valid[:, 0].tolist(), [False, True])

        probability = tensor([[[1.], [1.]], [[0.], [0.]]])
        projector = rgb_projector(sim.payload, drop_probability=probability)
        for _ in range(5):
            dropped = projector.step(state, wrist, fk, lift)
        self.assertEqual(dropped.top_valid[:, 0].tolist(), [False, True])
        self.assertEqual(dropped.wrist_vis_valid[:, 0].tolist(), [False, True])
        # An intervening dropped frame must not replace an already delivered valid target.
        probability[1] = 1.
        frozen = projector.step(state, out_of_fov, fk, lift)
        for old, new in zip(dropped, frozen):
            torch.testing.assert_close(old, new)
        print("RGB: independent 60/120 ms delay, blind target hold, frame freeze and reset", flush=True)

    @torch.inference_mode()
    def test_rgb_calibration_jitter_and_50_hz_exposure(self):
        sim = fixture("rigid_box", "rigid_box")
        projector = rgb_projector(sim.payload, frame_period_steps=10)
        state = sim.initial_state().payload
        wrist = camera((0., -.30, .045), WRIST, 2)
        fk = tensor([0., 0., .032]).expand(2, -1)
        lift = torch.zeros(2, dtype=torch.bool, device=DEVICE)
        before = projector.step(state, wrist, fk, lift)
        moved = state._replace(pos=state.pos + tensor([.020, 0., 0.]))
        for _ in range(9):
            reading = projector.step(moved, wrist, fk, lift)
            torch.testing.assert_close(reading.top_box, before.top_box)
        after = projector.step(moved, wrist, fk, lift)
        self.assertTrue((after.top_box[:, 0] - before.top_box[:, 0] > .019).all())

        p = projector.params._replace(translation_m=tensor([.005, 0., 0.]).expand_as(projector.params.translation_m),
                                      rotation_rad=tensor([0., .025, 0.]).expand_as(projector.params.rotation_rad),
                                      mask_jitter=torch.full_like(projector.params.mask_jitter, .12))
        projector.reset(torch.tensor([True, False], device=DEVICE), params=p)
        for _ in range(10):
            perturbed = projector.step(moved, wrist, fk, lift)
        torch.testing.assert_close(perturbed.top_box[1], after.top_box[1])
        self.assertTrue(perturbed.top_valid.all())
        self.assertGreater((perturbed.top_box[0] - after.top_box[0]).abs().max().item(), .005)
