"""Open-loop pad-space feasibility checks for the fixed Stage 3 dual-arm corners."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils._pytree import tree_map

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction
from src.low_order_parcel.hardware_filters import YAMCurrentEstimatorFilter
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction, PROBE_ACTION_CHANGE_S
from src.low_order_parcel.unified_payload import DT, GRAVITY, PayloadState
from src.low_order_parcel.verify_stage2_alignment import detachment_metrics
from src.low_order_parcel.virtual_rgb import CameraPose, VirtualRGBProjector

HOLD_STEPS = 1500
TRANSFER_STEPS = 250


def smooth(value):
    return .5 - .5 * (math.pi * value.clamp(0., 1.)).cos()


def action(pos, velocity, normal, inward, feed, alignment):
    return ParcelAction(pos, velocity, normal.expand(-1, 2), inward.expand(-1, 2),
                        feed.expand(-1, 2), alignment.expand(-1, 2),
                        torch.zeros_like(pos[:, 0]), torch.zeros_like(normal))


def replay(tick, reference, steps):
    """Use one common tick graph per fixed batch, carrying state in place."""
    if not reference.is_cuda:
        for _ in range(steps):
            tick()
        return
    stream = torch.cuda.Stream(device=reference.device)
    stream.wait_stream(torch.cuda.current_stream(reference.device))
    with torch.cuda.stream(stream):
        tick()
    torch.cuda.current_stream(reference.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        tick()
    for _ in range(steps - 1):
        graph.replay()
    torch.cuda.synchronize(reference.device)


def capture(contacts):
    return (contacts[..., ContactState.CAPTURE] > .5).all(dim=1, keepdim=True)


def deflect(offset, velocity, load, stiffness, damping, *, mass=.25, speed=.24, travel=.055):
    """Implicit passive wrist compliance driven by the pad contact reaction."""
    next_velocity = (mass * velocity + DT * (load - stiffness * offset)) / (
        mass + DT * damping + DT**2 * stiffness)
    next_offset = (offset + DT * next_velocity.clamp(-speed, speed)).clamp(-travel, travel)
    velocity.copy_((next_offset - offset) / DT)
    offset.copy_(next_offset)


@torch.inference_mode()
def pick_search(suite, points, corner_ids=None, *, trace=None, seconds=21.6):
    like = suite.episodes.F_half
    corners = torch.arange(8, device=like.device) if corner_ids is None else corner_ids
    count = len(points)
    ids = corners.repeat_interleave(count)
    p = suite.select(ids)
    params = points.repeat(len(corners), 1)
    route, depth, close_stroke, grip, hold_offset, approach, lift_time, entry = params.split(1, dim=1)
    heavy = (ids == 2)[:, None]
    loose = (ids == 6)[:, None]
    heavy_or_loose = heavy | loose
    # The loose parcel can be grasped nearer the loaded corner. Each candidate
    # declares its two material attachment sites before the episode starts.
    site_x = loose * (.10 * (hold_offset - .85))
    site_y = loose * (.09 * (entry - .85))
    mount = like.new_tensor([[[-.024, 0.], [.024, 0.]], [[-.024, .04], [.024, .04]]])
    site = torch.stack((site_x, site_y), dim=-1)
    mounts_A = mount[0] + site
    mounts_B = mount[1].expand(len(ids), -1, -1)
    sim = p.make_sim(mount_xy=torch.stack((mounts_A, mounts_B), dim=1))
    state = tree_map(torch.Tensor.clone, p.table_state(sim))
    t = torch.zeros((1, 1), device=like.device)
    side = (route >= .5) & (p.clearance_m.amax(dim=1, keepdim=True) >= .020)
    lift_start = torch.where(side, 2.40, 1.77)
    side_pair = torch.cat((side, torch.zeros_like(side)), dim=1)
    enabled = torch.tensor([True, False], device=like.device).expand(len(ids), -1)
    half = p.payload.bag_half_extents
    hx, hy, hz = half.split(1, dim=1)
    direction = torch.where(p.clearance_m[:, :1] >= p.clearance_m[:, 1:], 1., -1.)
    across = torch.cat((p.yaw_rad.cos(), p.yaw_rad.sin(), torch.zeros_like(hz)), dim=1)
    along = direction * torch.cat((-p.yaw_rad.sin(), p.yaw_rad.cos(), torch.zeros_like(hz)), dim=1)
    signs = t.new_tensor([[-1., 1.]])
    top = 2. * hz
    top_lower = (top - (.006 + .070 * depth)).clamp_min(.009)
    side_z = top + .002 + .012 * entry
    side_y = torch.where(loose, site_y - (.004 + .018 * (1. - depth)),
                         -torch.minimum(.012 + .036 * (1. - depth), .55 * hy))
    open_half = (.026 + .033 * approach).expand_as(hx)
    stroke = .012 + .043 * close_stroke
    closed = (open_half - stroke).clamp_min(.004) + .5 * p.yam.opening_deadzone_m[:, 0]
    target_grip = torch.where(heavy_or_loose, 13. + 24. * grip, .8 + 34. * grip)
    # Prescribed pad trajectories use only initial workcell geometry and candidate actions.
    # The slow lift covers both low-stiffness film extension and 1.5 kg load cases.
    lift_duration = torch.where(heavy_or_loose, 3.0 + 4.0 * lift_time, .9 + 1.9 * lift_time)
    final_top = top + .245 + .085 * hold_offset + p.proxy.L0.mean(dim=1, keepdim=True)
    delay = p.yam.delay_steps[:, 0] * DT
    zero = torch.zeros_like(hz)
    streak = zero.clone()
    best = zero.clone()
    peak = zero.clone()
    capture_peak = zero.clone()
    captured = torch.zeros_like(zero, dtype=torch.bool)
    airborne = torch.zeros_like(captured)
    fell = torch.zeros_like(captured)
    finite = torch.ones_like(captured)
    offset = hz.new_zeros((len(ids), 3))
    offset_velocity = torch.zeros_like(offset)
    wrist_load = torch.zeros_like(offset)
    wrist_torque = torch.zeros_like(offset)
    angle = torch.zeros_like(offset)
    angular_velocity = torch.zeros_like(offset)
    kxy = 18. + 170. * approach
    dxy = 4. + 30. * close_stroke
    wrist_k = torch.where(loose, torch.cat((kxy, kxy, 200. + 600. * grip), dim=1),
                          hz.new_tensor([70., 70., 650.]))
    wrist_d = torch.where(loose, torch.cat((dxy, dxy, 20. + 30. * grip), dim=1),
                          hz.new_tensor([13., 13., 28.]))
    angular_k = .08 + 1.5 * depth
    angular_d = .035 + .28 * lift_time
    final_clearance, final_speed = zero.clone(), zero.clone()
    final_capture = captured.clone()
    steps = round(seconds / DT)
    if trace is not None:
        recorded = hz.new_zeros((steps, len(corners), 20))
        selected = torch.arange(len(corners), device=hz.device) * count + trace["index"]
        cursor = torch.zeros(1, dtype=torch.long, device=hz.device)

    def target(time):
        time = time - delay
        near = smooth(time / .35)
        approach_to_fold = smooth((time - .35) / (.46 + .24 * approach))
        grasp = smooth((time - .89) / .75)
        rise = smooth((time - lift_start) / lift_duration)
        z_entry = torch.where(side, side_z, top_lower)
        z = torch.lerp(top + .028, z_entry, near)
        z = torch.lerp(z, final_top, rise)
        path = torch.lerp(-hy - .025, side_y, approach_to_fold)
        # Move the gathered fold over the center while both ropes still have slack.
        center = smooth((time - 1.66) / .63)
        lateral = heavy * (.075 * (entry - .5)) + site_y
        y = torch.where(side, path * (1. - center), 0.) + center * lateral
        h = torch.lerp(open_half, closed, grasp)
        transverse = heavy * (.09 * (hold_offset - .5))
        x = signs * h + center * transverse + site_x
        xy = along[:, None] * y.unsqueeze(-1) + across[:, None] * x.unsqueeze(-1)
        xyz = xy + sim.payload.up * z.unsqueeze(-1)
        return xyz, h, y, time, path

    def tick():
        pos, gap, _, time, path = target(t)
        before, old_gap, _, _, _ = target(t - DT)
        # Pad compliance damps the heavy and internally loose cases without reading COM motion.
        lift_compliance = heavy_or_loose * smooth((time - 1.66) / .65)
        deflect(offset, offset_velocity, wrist_load * lift_compliance, wrist_k, wrist_d)
        # The passive wrist can also turn under differential pad loads.
        deflect(angle, angular_velocity, wrist_torque * lift_compliance * loose,
                angular_k, angular_d, mass=.006, speed=1.5, travel=.24)
        arm = pos - pos.mean(dim=1, keepdim=True)
        arm_rotation = torch.linalg.cross(angle[:, None].expand_as(arm), arm)
        arm_velocity = torch.linalg.cross(angular_velocity[:, None].expand_as(arm), arm)
        normal = (target_grip * smooth((time - .68) / .25) * p.yam.gripper_gain[:, 0]
                  - p.yam.gripper_deadzone_N[:, 0]).clamp_min(0.)
        inward = (old_gap - gap) / DT
        feed = torch.where(side, (path + hy).clamp_min(0.), 0.)
        align = (side & (feed > 0.)).float()
        cmd = action(pos + offset[:, None] + arm_rotation,
                     (pos - before) / DT + offset_velocity[:, None] + arm_velocity,
                     normal, inward, feed, align)
        was_captured = state.contacts_A[..., ContactState.CAPTURE] > .5
        out = sim.step(state, DualArmAction(cmd, cmd, enabled, side_pair))
        wrist_load.copy_(out.wrist_wrench[:, 0, :3])
        wrist_torque.copy_(out.wrist_wrench[:, 0, 3:])
        newly = ~was_captured & (out.state.contacts_A[..., ContactState.CAPTURE] > .5)
        capture_peak.copy_(torch.maximum(capture_peak, (out.tension[:, 0] * newly).amax(dim=1, keepdim=True)))
        tree_map(lambda dst, src: dst.copy_(src), state, out.state)
        both = capture(state.contacts_A)
        clear = (out.table_clearance > .001) & (out.support_N == 0.)
        speed = sim.payload.com_velocity(state.payload).norm(dim=1, keepdim=True)
        quiet = speed < .020
        finite.logical_and_(state.payload.pos.flatten(1).isfinite().all(dim=1, keepdim=True))
        fell.logical_or_(airborne & ~clear)
        airborne.logical_or_(clear & both)
        captured.logical_or_(both)
        untorn = ~state.torn_flag.flatten(1).any(dim=1, keepdim=True)
        valid = clear & both & quiet & finite & untorn & (time >= lift_start + lift_duration)
        streak.copy_(torch.where(valid, streak + 1., 0.))
        best.copy_(torch.maximum(best, streak))
        peak.copy_(torch.maximum(peak, out.tension.flatten(1).amax(dim=1, keepdim=True)))
        final_capture.copy_(both)
        final_clearance.copy_(out.table_clearance)
        final_speed.copy_(speed)
        if trace is not None:
            center = sim.payload.com(state.payload)
            velocity = sim.payload.com_velocity(state.payload)
            channels = torch.cat((center, velocity, out.wrist_wrench[:, 0, :3],
                                  cmd.pinch_position.mean(dim=1), out.table_clearance, out.tension[:, 0],
                                  both.float(), state.payload.pos[..., 2]), dim=1)
            recorded.index_copy_(0, cursor, channels[selected].unsqueeze(0))
            cursor.add_(1)
        t.add_(DT)

    replay(tick, like, steps)
    if trace is not None:
        trace["channels"] = ("com_x", "com_y", "com_z", "v_x", "v_y", "v_z",
                             "force_x", "force_y", "force_z", "pad_x", "pad_y", "pad_z",
                             "clearance", "tension_left", "tension_right", "bilateral",
                             "block_0_z", "block_1_z", "block_2_z", "block_3_z")
        trace["data"] = recorded[::10].cpu().tolist()
    details = dict(final_capture=final_capture.cpu(), final_clearance_m=final_clearance.cpu(),
                   final_speed_mps=final_speed.cpu())
    return summarize(corners, points, best, finite & ~fell, captured, airborne, state.torn_flag,
                     peak, capture_peak, side, extra=details)


def summarize(corners, points, best, valid, captured, airborne, torn, peak, capture_peak, route,
              *, hold_steps=HOLD_STEPS, extra=None):
    count = len(points)
    arrays = [x.reshape(len(corners), count).cpu() for x in
              (best, valid, captured, airborne, torn.flatten(1).any(dim=1), peak, capture_peak, route)]
    hold, valid, captured, airborne, torn, peak, capture_peak, route = arrays
    results = {}
    for row, corner in enumerate(corners.tolist()):
        passes = valid[row] & (hold[row] >= hold_steps) & ~torn[row]
        # Prefer a feasible trace whose capture begins without rope tension.
        scores = hold[row] + 20000. * passes + 10000. * (passes & (capture_peak[row] < 1e-4))
        best_index = int(scores.argmax())
        results[f"C{corner + 1}"] = dict(
            successes=int(passes.sum()), captured=int(captured[row].sum()), airborne=int(airborne[row].sum()),
            torn=int(torn[row].sum()), best_hold_s=float(hold[row, best_index]) * DT,
            best_mode="side" if bool(route[row, best_index]) else "top",
            best_action=points[best_index].tolist(), best_index=best_index,
            hold_ticks=hold[row].int().tolist(), valid=valid[row].tolist(),
            captured_rows=captured[row].tolist(), airborne_rows=airborne[row].tolist(), torn_rows=torn[row].tolist(),
            peak_tension_N=peak[row].tolist(), capture_tension_N=capture_peak[row].tolist())
        if extra is not None:
            for key, values in extra.items():
                results[f"C{corner + 1}"][key] = values.reshape(len(corners), count, -1)[row].tolist()
    return results


@torch.inference_mode()
def measure_release():
    """Rerun the frozen 5 N/s load probe with C1's fixed Stage 3A pressure gate."""
    dr = DomainRandomizer("cpu")
    c1 = CornerSuite(dr).select(torch.tensor([0]))
    sim = c1.make_sim()
    adapter = MacroParcelSim(sim.proxy_A, payload=sim.payload)
    _, _, applied, trajectory = adapter.probe_actions("ramp", grip_N=.62, axis="tangential")
    state = sim.initial_state()
    enabled = torch.tensor([[True, False]])
    flags = enabled.clone()
    records = []
    for args in zip(*trajectory, strict=True):
        cmd = ParcelAction(*args)
        result = sim.step(state, DualArmAction(cmd, cmd, enabled, flags))
        slip = (result.state.contacts_A[..., ContactState.SLIP_CUM]
                - state.contacts_A[..., ContactState.SLIP_CUM]).clamp_min(0.) / DT
        state = result.state
        records.append((result.support_N[:, 0], result.table_clearance[:, 0], result.tension[:, 0],
                        state.contacts_A, slip, result.wrist_wrench[:, 0]))
    support, clearance, tension, contact, slip, wrench = (torch.stack(c)[:, 0].numpy()
                                                          for c in zip(*records, strict=True))
    trace = dict(time_s=(torch.arange(len(support)) * DT).numpy(), external_load_N=applied[:, 0, 0].numpy(),
                 support_N=support, table_clearance_m=clearance, tension_N=tension, contact_state=contact,
                 slip_rate_mps=slip, wrench=wrench,
                 action_change_s=PROBE_ACTION_CHANGE_S)
    return detachment_metrics(trace, DT, "external_load_N", [.55])


@torch.inference_mode()
def transfer_search(suite, points, corners):
    count = len(points)
    ids = corners.repeat_interleave(count)
    p = suite.select(ids)
    reference = p.payload.total_mass
    mounts = reference.new_tensor([[[-.024, 0.], [.024, 0.]], [[-.024, 0.], [.024, 0.]]])
    sim = p.make_sim(mount_xy=mounts.expand(len(ids), -1, -1, -1))
    state = tree_map(torch.Tensor.clone, sim.initial_state())
    payload = sim.payload
    x, y, z = payload.rest.unbind(dim=-1)
    fold = payload.diagonal * (p.fold_rad / 2.).sin() * payload.bend_sign
    c, s = p.swing_rad.cos(), p.swing_rad.sin()
    y, z = c * y - s * fold, s * y + c * fold + .30
    state.payload.pos.copy_(torch.stack((x, y, z), dim=-1))
    state.contacts_A[..., ContactState.SLACK].copy_(torch.minimum(.22 * sim.base_slack, reference.new_full((), .006)))
    state.contacts_A[..., ContactState.CAPTURE].fill_(1.)
    state.contacts_A[..., ContactState.HOLD_TIME].fill_(.1)
    initial_bend = payload.bend_angle(state.payload)
    anchors, _ = payload.anchors(state.payload, sim.mount_weights[:, :2], state.payload.d_surf)
    top = anchors[..., 2].mean(dim=1, keepdim=True)
    rest = p.proxy.L0.mean(dim=1, keepdim=True)
    equilibrium = rest + reference * (GRAVITY / 2.) / p.proxy.k
    hx, hy, _ = p.payload.bag_half_extents.split(1, dim=1)
    u0, u1, u2, u3, u4, u5, u6, u7 = points.repeat(len(corners), 1).split(1, dim=1)
    light = (ids == 9)[:, None]
    capture_y = .021 + .030 * u0
    entry_z = top + .002 + .022 * u1
    tug_y = .066 + .010 * u2
    tug_z = .006 + .010 * u1
    radius = torch.hypot(tug_y, tug_z)
    duration = .40 + .37 * u3
    carry_start = torch.where(light, 2.92, 1.55)
    carry_duration = .65 + .62 * u4
    top_B = top + equilibrium + .025 + .066 * u5
    half_B = .006 + .019 * u6
    normal_A = (12. + reference * 18.).clamp_max(32.)
    normal_B = 12. + 20. * u7
    sign = reference.new_tensor([[-1., 1.]])
    delays = p.yam.delay_steps[:, :, 0] * DT
    enabled = torch.ones((len(ids), 2), dtype=torch.bool, device=ids.device)
    routes = torch.cat((torch.zeros_like(light), torch.ones_like(light)), dim=1)
    zero = torch.zeros_like(reference)
    streak, best, peak, capture_peak = (zero.clone() for _ in range(4))
    captured, dropped, offered = (torch.zeros_like(zero, dtype=torch.bool) for _ in range(3))
    finite = torch.ones_like(captured)
    steps = 5500
    cursor = torch.zeros(1, dtype=torch.long, device=ids.device)
    tug_force = reference.new_zeros((steps, len(ids), 2, 3))
    camera_ticks = torch.tensor([90, 640], device=ids.device)
    camera_pos = reference.new_zeros((2, len(ids), 4, 3))
    camera_depth = reference.new_zeros((2, len(ids), 4))
    camera_fk = reference.new_zeros((2, len(ids), 2, 3))
    offset = reference.new_zeros((len(ids), 2, 3))
    offset_velocity = torch.zeros_like(offset)
    wrist_load = torch.zeros_like(offset)
    wrist_k = reference.new_tensor([85., 85., 650.])
    wrist_d = reference.new_tensor([12., 12., 28.])

    def target(time):
        t_A, t_B = (time - delays).split(1, dim=1)
        prepare = smooth((t_A - 1.55) / duration)
        y_A = torch.where(light, -tug_y * prepare, 0.)
        spherical = (radius.square() - y_A.square()).clamp_min(0.).sqrt()
        z_A = top + torch.where(light, torch.lerp(equilibrium, spherical, prepare), equilibrium)
        z_A = z_A + .022 * smooth((t_A - 4.18) / .40)
        x_A = sign * .024
        pos_A = torch.stack((x_A.expand_as(anchors[..., 0]), y_A.expand_as(x_A + zero),
                             z_A.expand_as(x_A + zero)), dim=-1)
        approach = smooth((t_B - .20) / .68)
        close = smooth((t_B - .88) / .59)
        tug = smooth((t_B - 1.62) / duration)
        carry = smooth((t_B - carry_start) / carry_duration)
        y_B = torch.lerp(.30 + 0. * capture_y, capture_y, approach)
        y_B = torch.where(light, torch.lerp(y_B, tug_y, tug), y_B)
        y_B = torch.lerp(y_B, zero, carry)
        z_B = torch.where(light, torch.lerp(entry_z, top + tug_z, tug), entry_z)
        z_B = torch.lerp(z_B, top_B, carry)
        h_B = torch.lerp(torch.minimum(.78 * hx, reference.new_full((), .060)), half_B, close)
        x_B = sign * h_B
        pos_B = torch.stack((x_B, y_B.expand_as(x_B), z_B.expand_as(x_B)), dim=-1)
        return pos_A, pos_B, h_B, t_A, t_B, y_B

    def tick():
        time = cursor * DT
        A, B, width, time_A, time_B, y_B = target(time)
        oldA, oldB, old_width, _, _, _ = target(time - DT)
        # Passive wrist deflection under the last mechanical reaction. No parcel
        # state is used to move either command during or after the handover.
        stiff_tug = (light * (1. - smooth((time - 2.90) / .38)))[:, :, None]
        k = wrist_k * (1. + 19. * stiff_tug)
        deflect(offset, offset_velocity, wrist_load, k, wrist_d)
        force_A = normal_A * (1. - smooth((time_A - 3.72) / .39))
        force_B = normal_B * smooth((time_B - .86) / .23)
        grip = lambda f, arm: (p.yam.gripper_gain[:, arm] * f - p.yam.gripper_deadzone_N[:, arm]).clamp_min(0.)
        feed = (hy - y_B).clamp_min(0.)
        pad_A, pad_B = offset[:, 0:1], offset[:, 1:2]
        vel_A, vel_B = offset_velocity[:, 0:1], offset_velocity[:, 1:2]
        cmd_A = action(A + pad_A, (A - oldA) / DT + vel_A, grip(force_A, 0), zero, zero, 1. + zero)
        cmd_B = action(B + pad_B, (B - oldB) / DT + vel_B, grip(force_B, 1), (old_width - width) / DT,
                       feed, (time_B >= .86).float())
        prior_B = state.contacts_B[..., ContactState.CAPTURE] > .5
        out = sim.step(state, DualArmAction(cmd_A, cmd_B, enabled, routes))
        wrist_load.copy_(out.wrist_wrench[..., :3])
        new_capture = ~prior_B & (out.state.contacts_B[..., ContactState.CAPTURE] > .5)
        capture_peak.copy_(torch.maximum(capture_peak, (out.tension[:, 1] * new_capture).amax(dim=1, keepdim=True)))
        tree_map(lambda dst, src: dst.copy_(src), state, out.state)
        cap_A = capture(state.contacts_A)
        cap_B = capture(state.contacts_B)
        clear = (out.table_clearance > .001) & (out.support_N == 0.)
        quiet = payload.com_velocity(state.payload).norm(dim=1, keepdim=True) < .020
        finite.logical_and_(state.payload.pos.flatten(1).isfinite().all(dim=1, keepdim=True))
        held = ((state.contacts_A[..., ContactState.CAPTURE] > .5).any(dim=1, keepdim=True)
                | (state.contacts_B[..., ContactState.CAPTURE] > .5).any(dim=1, keepdim=True))
        dropped.logical_or_(~(clear & held))
        captured.logical_or_(cap_B)
        offered.logical_or_(cap_A & cap_B & clear)
        untorn = ~state.torn_flag.flatten(1).any(dim=1, keepdim=True)
        A_released = (state.contacts_A[..., ContactState.CAPTURE] == 0.).all(dim=1, keepdim=True)
        stable = (time_B > 4.15) & A_released & cap_B & clear & quiet & finite & untorn & offered
        streak.copy_(torch.where(stable, streak + 1., 0.))
        best.copy_(torch.maximum(best, streak))
        peak.copy_(torch.maximum(peak, out.tension.flatten(1).amax(dim=1, keepdim=True)))
        velocity_y = torch.stack((cmd_A.pinch_velocity.mean(dim=1)[:, 1], cmd_B.pinch_velocity.mean(dim=1)[:, 1]), dim=1)
        bilateral = cap_A & cap_B & clear
        signal = torch.stack((out.wrist_wrench[:, :, 1], velocity_y, bilateral.expand_as(velocity_y)), dim=-1)
        tug_force.index_copy_(0, cursor, signal.unsqueeze(0))
        snapshot = (camera_ticks == cursor).reshape(2, 1, 1, 1)
        camera_pos.copy_(torch.where(snapshot, state.payload.pos.unsqueeze(0), camera_pos))
        camera_depth.copy_(torch.where(snapshot.squeeze(-1), state.payload.d_surf.unsqueeze(0), camera_depth))
        fk = torch.stack((cmd_A.pinch_position.mean(dim=1), cmd_B.pinch_position.mean(dim=1)), dim=1)
        camera_fk.copy_(torch.where(snapshot, fk.unsqueeze(0), camera_fk))
        cursor.add_(1)

    replay(tick, reference, steps)
    # Replay the actual lateral wrench and FK velocity through the Stage 3A estimator.
    # Processing after the trial keeps noisy readings out of the open-loop trajectory.
    sensor = YAMCurrentEstimatorFilter(p.yam, generator=torch.Generator(device=ids.device).manual_seed(347))
    zero3 = reference.new_zeros((len(ids), 2, 3))
    zero1 = zero3[..., :1]
    directions = reference.new_tensor([[1., -1.]])
    gravity_direction = reference.new_tensor([0., 0., -1.]).expand_as(zero3)
    truth_peak = zero.clone()
    greatest_arm_pull = zero.clone()
    sensed_peak = zero.clone()
    pulse, longest = zero.clone(), zero.clone()
    for tick_id in range(1460):
        force_y, velocity_y, bilateral = tug_force[tick_id].unbind(dim=-1)
        force = torch.stack((zero3[..., 0], force_y, zero3[..., 2]), dim=-1)
        velocity = torch.stack((zero3[..., 0], velocity_y, zero3[..., 2]), dim=-1)
        reading = sensor.step(force, velocity, gravity_direction, zero1, zero1, zero3, zero3)
        if 2.2 <= tick_id * DT < 2.92:
            true_pull = force_y * directions * bilateral
            measured = reading.force[..., 1] * directions
            truth_peak = torch.maximum(truth_peak, true_pull.amin(dim=1, keepdim=True))
            greatest_arm_pull = torch.maximum(greatest_arm_pull, true_pull.amax(dim=1, keepdim=True))
            sensed_peak = torch.maximum(sensed_peak, measured.amin(dim=1, keepdim=True))
            signal = (measured > 3.5).all(dim=1, keepdim=True) & bilateral[:, :1].bool()
            pulse = torch.where(signal, pulse + 1., 0.)
            longest = torch.maximum(longest, pulse)
    # Replay the frozen look and interception frames through the actual RGB pipeline.
    batch = len(ids)
    top_pose = CameraPose(reference.new_tensor([.12, .22, .95]).expand(batch, -1),
                          reference.new_tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]).expand(batch, -1, -1))
    wrist_frame = reference.new_tensor([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]]).expand(batch, -1, -1)
    camera = VirtualRGBProjector(payload, top_pose, p.rgb,
                                 table_bounds=reference.new_tensor([-.5, .5, -.5, .5]).expand(batch, -1),
                                 generator=torch.Generator(device=ids.device).manual_seed(349))
    visibility = []
    for position, depth, fk in zip(camera_pos, camera_depth, camera_fk, strict=True):
        wrist = CameraPose(fk[:, 1] + reference.new_tensor([0., .015, .015]), wrist_frame)
        snapshot = PayloadState(position, torch.zeros_like(position), depth)
        for _ in range(95):
            observation = camera.step(snapshot, wrist, fk[:, 0], torch.zeros_like(light))
        visibility.append(observation.wrist_vis_valid)
    seen, near_visible = visibility
    qualifies = seen & ~near_visible & (~light | ((longest >= 10.) & (truth_peak > 3.5) & (greatest_arm_pull <= 5.5)))
    details = dict(true_pull_peak_N=truth_peak.cpu(), observed_pull_peak_N=sensed_peak.cpu(),
                   greatest_arm_pull_N=greatest_arm_pull.cpu(),
                   pull_pulse_ms=(longest * (DT * 1000)).cpu(), initial_fold_rad=initial_bend.cpu(),
                   initial_swing_rad=p.swing_rad.cpu(), wrist_near_field_m=p.rgb.wrist_min_range_m.cpu(),
                   wrist_seen_far=seen.cpu(), wrist_seen_near=near_visible.cpu())
    results = summarize(corners, points, best, finite & ~dropped & qualifies, captured, offered, state.torn_flag,
                        peak, capture_peak, torch.ones_like(finite), hold_steps=TRANSFER_STEPS, extra=details)
    for result in results.values():
        result["best_mode"] = "handover"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=347)
    parser.add_argument("--trace-row", type=int)
    parser.add_argument("--pick-seconds", type=float, default=21.6)
    parser.add_argument("--loose-seconds", type=float, default=16.5, help="fixed C7 command duration")
    parser.add_argument("--corners", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.candidates < 1 or any(not math.isfinite(t) or t < 4.
                                   for t in (args.pick_seconds, args.loose_seconds))
            or any(not 1 <= c <= 10 for c in args.corners)
            or len(set(args.corners)) != len(args.corners) or args.output.exists()):
        parser.error("provide a positive candidate count, corner IDs 1–10 and a fresh output path")
    start = perf_counter()
    dr = DomainRandomizer(args.device, args.seed)
    suite = CornerSuite(dr)
    sobol = torch.quasirandom.SobolEngine(8, scramble=True, seed=args.seed)
    points = sobol.draw(args.candidates).to(dr.device)
    if args.trace_row is not None and not 0 <= args.trace_row < args.candidates:
        parser.error("trace row must identify a candidate in the batch")
    picked = [c - 1 for c in args.corners if c <= 8 and c != 7]
    if args.trace_row is not None and 7 in args.corners and picked:
        parser.error("trace C7 separately from other pickup corners")
    trace = None if args.trace_row is None else dict(index=args.trace_row)
    results = pick_search(suite, points, dr.tensor(picked, dtype=torch.long),
                          trace=trace, seconds=args.pick_seconds) if picked else {}
    if 7 in args.corners:
        results.update(pick_search(suite, points, dr.tensor([6], dtype=torch.long),
                                   trace=trace, seconds=args.loose_seconds))
    handed = [c - 1 for c in args.corners if c > 8]
    if handed:
        results.update(transfer_search(suite, points, dr.tensor(handed, dtype=torch.long)))
    release = measure_release() if 1 in args.corners else None
    path = ROOT / "configs/stage3_dr_and_corners.yaml"
    outcome = dict(machine=platform.node(), torch=torch.__version__, device=args.device, seed=args.seed, dt_s=DT,
                   config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), wall_s=perf_counter() - start,
                   candidates=args.candidates, pick_seconds=args.pick_seconds, loose_seconds=args.loose_seconds,
                   release=release, corners=results,
                   passed=len(results) == len(args.corners) and all(v["successes"] > 0 for v in results.values())
                   and (release is None or (release["pass_gate"] and .45 <= release.get("value", -1.) <= .65)))
    args.output.write_text(json.dumps(outcome, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n")
    if trace is not None and (picked or 7 in args.corners):
        args.output.with_suffix(".trace.json").write_text(json.dumps(trace, allow_nan=False) + "\n")
    for corner, result in results.items():
        print(corner, result["successes"], f"{result['best_hold_s']:.3f}s", result["best_mode"],
              "captured", result["captured"], "airborne", result["airborne"], "torn", result["torn"], flush=True)
    print("C1 release", release, flush=True)
    raise SystemExit(0 if outcome["passed"] else 1)
