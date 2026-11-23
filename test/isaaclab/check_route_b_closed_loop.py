"""Run deterministic ABC pad-space commands through real YAM joint targets."""

import argparse
import json
from time import perf_counter

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.controllers import DifferentialIKControllerCfg

from src.isaac_yam_parcel.joint_ik import PadIK, transition
from src.isaac_yam_parcel.yam_parcel_base_env import YAMHandoverPlaceEnv, YAMPickEnv, yam_parcel_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.lumped_contact_proxy import ContactState


class C5PressThenClose:
    """Regulate measured top load in the physical acceptance harness."""

    def __init__(self, ready, center, top):
        self.approach = center.expand_as(ready).clone()
        self.approach[:, 2] = top + .012
        self.z = self.approach[:, 2].clone()
        self.in_band = self.z.new_zeros(len(ready))
        self.close_started = self.z.new_full((len(ready),), -1.)
        self.closed_at = self.close_started.clone()
        self.stroke = self.z.new_zeros(len(ready))

    def __call__(self, env, time, ready, offer, open_rail, closed):
        command = torch.lerp(ready, self.approach, transition(time, .35, 2.05))
        if time < 2.15:
            return command, open_rail
        dt = env.step_dt
        force = env.result.press_N[:, 0]
        pad_floor_gap = env.ports[0].read().pads[..., 2].amin(dim=1)
        safe_table_gap = pad_floor_gap >= .004
        in_band = (force >= 8.) & (force <= 12.) & safe_table_gap
        self.in_band = torch.where(in_band, self.in_band + dt, 0.)
        latch = (self.close_started < 0.) & (self.in_band >= .1)
        self.close_started = torch.where(latch, time, self.close_started)
        # Change Z slowly under load and more quickly when the open pads are in air.
        dz = ((force - 10.) * .00013).clamp(-.0014, .0014)
        dz = torch.where(safe_table_gap, dz, dz.clamp_min(.001))
        # Use limited tracking-error compensation only while the pads are
        # physically loading the cloth and remain clear of the tabletop.
        min_z = torch.where(safe_table_gap & (force >= .5), -.006, 0.)
        self.z = torch.maximum(self.z + dz, min_z)
        command[:, 2] = self.z
        width = closed - open_rail
        self.stroke = torch.minimum(
            width, self.stroke + dt * .035 * ((self.close_started >= 0.) & in_band))
        complete = self.stroke >= width - 1e-6
        self.closed_at = torch.where(complete & (self.closed_at < 0.), time, self.closed_at)
        phase = torch.where(complete, ((time - self.closed_at - .16) / 2.8).clamp(0., 1.), 0.)
        rise = .5 - .5 * (torch.pi * phase).cos()
        command = torch.lerp(command, offer, rise[:, None])
        preload = env.step_limit[-1] * 15. / (1000. * env.rail_overtravel)
        rail = open_rail + self.stroke + complete * preload
        return command, rail


@torch.inference_mode()
def acquisition(env, rows, seconds, *, seat_fold=False, rapid_close=False, cross_center=False,
                lift_toward_base=False, loaded_sweep=False, preview_target=None):
    """Drive the approach using only planned workcell coordinates and YAM FK."""
    env.reset()
    robot = env.robots[0]
    port = env.ports[0]
    sample = port.read()
    q_ready = sample.wrist_quat_xyzw.clone()
    start = sample.pads.mean(dim=1).clone()
    shift = env.pick_center
    start_rail = robot.data.default_joint_pos.torch[:, robot.rail_ids].mean(dim=1)
    ik = PadIK(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                       ik_method="dls", ik_params={"lambda_val": .035}),
                                  env.num_envs, str(env.device))
    start_goal = start.clone()
    top_row = start.new_tensor([row[1] == "top" for row in rows]).bool()
    clothes = start.new_tensor([row[0] == "C5" for row in rows]).bool()
    heavy = start.new_tensor([row[0] == "C3" for row in rows]).bool()
    high_entry = start.new_tensor([row[1] == "side-high" for row in rows]).bool()
    light_high = high_entry & ~heavy
    # Rows encode each fixture's known starting dimensions. These are fixed
    # initial workcell parameters, with no parcel tracking after the start.
    half = env.episode.payload.bag_half_extents.clone()
    top = 2. * half[:, 2]
    entry = torch.stack((torch.zeros_like(top), -(half[:, 1] + .037),
                         top + start.new_tensor([row[2] for row in rows])), dim=-1)
    engage = entry.clone()
    engage[:, 1] = torch.where(top_row, 0., -start.new_tensor([row[3] for row in rows]))
    if cross_center:
        # Seat the fold just past center, outside the entry-wall contact region.
        engage[high_entry, 1] = .012
    entry[:, 1] = torch.where(top_row, 0., entry[:, 1])
    entry[:, 2] = torch.where(top_row, top + .028, entry[:, 2])
    offer = engage.clone()
    offer[:, 1] = -.027
    offer[:, 2] = .235
    if lift_toward_base:
        offer[heavy, 0] -= .115
        offer[heavy, 1] = -.055
    if loaded_sweep:
        offer[heavy, 0] -= .052
        offer[heavy, 2] = .300
    entry += shift
    engage += shift
    offer += shift
    requested_preload = start.new_tensor([row[4] for row in rows])
    rail_target = torch.full_like(start_rail, env.rail_close)
    top_press = C5PressThenClose(start_goal, shift, top)
    phased_c5_enabled = any(row[0] == "C5" and row[1] == "top" for row in rows)
    timestamps = []
    observations = []
    tracking = []
    for step in range(round(seconds / env.step_dt)):
        time = step * env.step_dt
        initial = transition(time, .35, 1.85)
        approach = transition(time, 1.85, 3.20)
        closure = transition(time, 2.80, 4.05)
        rise = transition(time, 4.65, 8.1)
        command = torch.lerp(torch.lerp(start_goal, entry, initial), engage, approach)
        command = torch.lerp(command, offer, rise)
        rail = torch.lerp(start_rail, rail_target, closure)
        # Keep the preset downward sweep moving as the two pads gather the soft film.
        command[:, 2] -= clothes * .004 * transition(time, 2.9, 4.25) * (1. - rise)
        rail = torch.where(clothes, torch.lerp(start_rail, rail_target, transition(time, 2.65, 3.70)), rail)
        if seat_fold:
            # Enter above the rim first, then lower at fixed XY while closing.
            # It is a preset geometric trajectory, with no parcel state feedback.
            seat = transition(time, 3.24, 3.78) * (1. - rise)
            command[:, 2] -= high_entry * (engage[:, 2] - top + .002) * seat
            rail = torch.where(high_entry, torch.lerp(start_rail, rail_target,
                                                      transition(time, 3.45, 4.32)), rail)
        preload = requested_preload * transition(time, 3.54, 4.40)
        if rapid_close:
            side_close = (3.72 if cross_center else 3.50) if seat_fold else 3.28
            close_now = (high_entry & (time >= side_close)) | (clothes & (time >= 2.88))
            rail = torch.where(high_entry | clothes, torch.where(close_now, rail_target, start_rail), rail)
            preload = torch.where(close_now, requested_preload, preload)
        if loaded_sweep:
            # Close soft parcels while the wrist is still sweeping into the upper surface.
            approach_z = torch.lerp(start_goal[:, 2], top + .010, transition(time, .35, 2.1))
            press_z = approach_z - .046 * min(1., max(0., (time - 2.18) / 1.20))
            command[:, 2] = torch.where(clothes, torch.lerp(press_z, offer[:, 2], rise), command[:, 2])
            rail = torch.where(clothes, rail_target if time >= 2.94 else start_rail, rail)
            preload = torch.where(clothes & (time >= 2.94), requested_preload, preload)
            # Move quickly through clearance, then sweep the remaining fold at a lower rail speed.
            pre_gap = torch.full_like(rail, env.rail_close - .008)
            side_rail = torch.lerp(start_rail, pre_gap, transition(time, 3.08, 3.48))
            side_rail = torch.lerp(side_rail, rail_target, transition(time, 3.48, 3.76))
            rail = torch.where(light_high, side_rail, rail)
            preload = torch.where(light_high, requested_preload * transition(time, 3.53, 3.80), preload)
            # Use the same closing sweep on both heavy lanes. Compare holding
            # torques only after the two fingers have finished gathering.
            if rapid_close and time >= side_close:
                heavy_preload = torch.lerp(requested_preload.clamp_max(8.), requested_preload,
                                           transition(time, 3.62, 4.32))
                preload = torch.where(heavy, heavy_preload, preload)
            ik.angular_weight = torch.where(heavy, 1. - .95 * transition(time, 4.65, 5.75),
                                             torch.ones_like(top))
        rail += env.step_limit[-1] * preload / (1000. * env.rail_overtravel)
        sample = port.read()
        if preview_target is not None:
            command, rail = preview_target(env, time, start_goal, offer, start_rail, rail_target)
        elif phased_c5_enabled:
            c5_command, c5_rail = top_press(env, time, start_goal, offer, start_rail, rail_target)
            c5_top = clothes & top_row
            command = torch.where(c5_top[:, None], c5_command, command)
            rail = torch.where(c5_top, c5_rail, rail)
        pad_center = sample.pads.mean(dim=1)
        jac = torch.cat((sample.pad_jacobian.mean(dim=1), sample.angular_jacobian.mean(dim=1)), dim=1)[..., :6]
        q = robot.data.joint_pos.torch
        ik.set_command(torch.cat((command, q_ready), dim=-1))
        target = ik.compute(pad_center, sample.wrist_quat_xyzw, jac, q[:, :6])
        arm_delta = target - q[:, :6]
        rail_delta = rail - q[:, robot.rail_ids].mean(dim=1)
        action = torch.cat((arm_delta, rail_delta[:, None]), dim=1) / env.step_limit
        env.step(action)
        state = env.parcel_state
        sample = port.read()
        measured = sample.pads.mean(dim=1)
        speed = env.parcel_sim.payload.com_velocity(state.payload).norm(dim=1, keepdim=True)
        capture = state.contacts_A[..., ContactState.CAPTURE]
        observations.append(torch.cat((capture, speed, env.result.table_clearance, measured,
                                       robot.actuators.applied_effort.torch[:, robot.rail_ids],
                                       state.contacts_A[..., ContactState.SLACK],
                                       env.result.net_feed[:, 0], env.result.press_N[:, :1],
                                       env.result.tension[:, 0],
                                       env.parcel_sim.payload.com(state.payload),
                                       robot.rail_load, (sample.pads[:, 1:] - sample.pads[:, :1]).norm(dim=-1),
                                       state.torsion.slip_angle[:, 0],
                                       env.result.support_N, state.torn_flag.flatten(1).any(dim=1, keepdim=True),
                                       env.film_load[:, 0], env.pad_normal[:, 0], env.side[:, :1],
                                       env.fold_work[:, 0]),
                                      dim=-1))
        tracking.append(command)
        timestamps.append(time)

    data = torch.stack(observations).cpu()
    goals = torch.stack(tracking).cpu()
    approach_end = min(round(3.2 / env.step_dt), len(timestamps) - 1)
    tail = round(3. / env.step_dt)
    settle = slice(round(4.45 / env.step_dt), round(4.65 / env.step_dt))
    results = []
    for index, row in enumerate(rows):
        series = data[:, index]
        both = (series[:, :2] == 1.).all(dim=1)
        closing = (series[:-1, 21] - series[1:, 21]) > .0002
        sweep_pressure = series[1:, 13][closing]
        sweep_time = series.new_tensor(timestamps[1:])
        loaded_sweep_frames = closing & (sweep_time >= 2.6) & (sweep_time < 4.45)
        in_same_load_band = (series[1:, 13] >= 8.) & (series[1:, 13] <= 12.)
        phased_c5 = row[0] == "C5" and row[1] == "top" and preview_target is None
        if phased_c5:
            started = top_press.close_started[index].item()
            finished = top_press.closed_at[index].item()
            loaded_sweep_frames = (closing & (started >= 0.) & (sweep_time >= started)
                                   & ((finished < 0.) | (sweep_time <= finished + .12)))
        results.append(dict(corner=row[0], route=row[1], entry_offset_m=row[2],
                            any_double_capture=bool(both.any()), captured_frames=int(both.sum()),
                            bilateral_airborne_frames=int((both & (series[:, 3] > .001)
                                                          & (series[:, 23] == 0.)).sum()),
                            any_finger_frames=int((series[:, :2] == 1.).any(dim=1).sum()),
                            last3_captured_frames=int(both[-tail:].sum()),
                            peak_clearance_mm=1000 * series[:, 3].amax().item(),
                            last3_min_clearance_mm=1000 * series[-tail:, 3].amin().item(),
                            last3_max_speed_mm_s=1000 * series[-tail:, 2].amax().item(),
                            longest_hold_s=env.longest_hold[index].item() * .002,
                            peak_motor_effort_N=series[:, 7:9].abs().amax().item(),
                            peak_gathered_mm=1000 * series[:, 9:11].amax().item(),
                            peak_bilateral_gather_mm=1000 * series[:, 9:11].amin(dim=1).amax().item(),
                            peak_feed_mm=1000 * series[:, 11:13].amax().item(),
                            peak_press_N=series[:, 13].amax().item(),
                            peak_package_translation_mm=1000 * (series[:, 16:18] -
                                  series[:1, 16:18]).norm(dim=1).amax().item(),
                            peak_rail_external_N=series[:, 19:21].abs().amax().item(),
                            hold_motor_mean_N=series[settle, 7:9].mean().item() if seconds > 4.65 else None,
                            hold_rail_external_mean_N=series[settle, 19:21].abs().mean().item() if seconds > 4.65 else None,
                            hold_effective_normal_mean_N=series[settle, 26:28].mean().item() if seconds > 4.65 else None,
                            peak_film_load_N=series[:, 25].amax().item(),
                            pressure_frames_during_closure=int((series[66:107, 13] > .3).sum()),
                            closing_frames=int(closing.sum()),
                            closing_frames_2_to_4_N=int(((sweep_pressure >= 2.) & (sweep_pressure <= 4.)).sum()),
                            loaded_sweep_frames=int(loaded_sweep_frames.sum()),
                            loaded_sweep_frames_8_to_12_N=int((loaded_sweep_frames & in_same_load_band).sum()),
                            fold_work_min_mJ=1000. * series[:, 29].amin().item(),
                            fold_work_max_mJ=1000. * series[:, 29].amax().item(),
                            side_route_frames=int(series[:, 28].sum()),
                            final_pad_gap_mm=1000. * series[-1, 21].item(),
                            peak_torsional_slip_rad=series[:, 22].abs().amax().item(),
                            torn=bool(series[:, 24].any()),
                            approach_end_z_target_mm=1000 * goals[approach_end, index, 2].item(),
                            approach_end_z_actual_mm=1000 * series[approach_end, 6].item(),
                            approach_end_z_error_mm=1000 * (series[approach_end, 6] -
                                                                   goals[approach_end, index, 2]).item(),
                            pad_min_height_mm=1000 * series[:, 6].amin().item(),
                            finite=bool(series.isfinite().all())))
        if phased_c5:
            contact = (series[:, 13] >= .5).nonzero().flatten()
            first_contact = int(contact[0]) if len(contact) else None
            results[-1].update(
                first_top_contact_s=timestamps[first_contact] + env.step_dt if len(contact) else None,
                aperture_at_contact_mm=1000. * series[first_contact, 21].item() if len(contact) else None,
                open_aperture_mm=1000. * series[0, 21].item(),
                close_started_s=started if started >= 0. else None,
                closed_at_s=finished if finished >= 0. else None,
                contact_before_close=bool(len(contact) and started >= timestamps[first_contact] + env.step_dt + .1
                                          and series[first_contact, 21] >= .95 * series[0, 21]))
    return results, dict(times_s=timestamps, observations=data.tolist(), pad_target_m=goals.tolist())


def c5_gate(clothes):
    return (clothes.get("contact_before_close", False)
            and clothes["loaded_sweep_frames"] > 0
            and clothes["loaded_sweep_frames"] == clothes["loaded_sweep_frames_8_to_12_N"]
            and clothes["peak_bilateral_gather_mm"] > 1.50
            and clothes["bilateral_airborne_frames"] > 0)


def right_arm_gates(result):
    hard, clothes, low_entry, *high_entry, low_grip, high_grip = result
    return {
        "C4_top": hard["any_finger_frames"] == 0 and hard["peak_clearance_mm"] < 1.
        and hard["peak_film_load_N"] == 0.,
        "C5_top": c5_gate(clothes),
        "C6_low": low_entry["peak_package_translation_mm"] > 1. and low_entry["any_finger_frames"] == 0
        and low_entry["peak_film_load_N"] == 0.,
        "C6_high": any(row["longest_hold_s"] >= 3. and row["last3_max_speed_mm_s"] < 20.
                       and row["last3_min_clearance_mm"] > 1. and row["last3_captured_frames"] == 75
                       and 2. <= row["hold_effective_normal_mean_N"] <= 6. for row in high_entry),
        "C3_twist": low_grip["any_double_capture"] and low_grip["peak_torsional_slip_rad"] > 1e-6
        and high_grip["last3_captured_frames"] == 75 and high_grip["last3_min_clearance_mm"] > 1.
        and high_grip["peak_torsional_slip_rad"] <= 1e-6,
        "finite_and_intact": all(row["finite"] and not row["torn"] for row in result),
    }


@torch.inference_mode()
def empty_clamp(env):
    env.reset()
    ready = torch.stack([env.motor_channels(robot.data.default_joint_pos.torch, robot) for robot in env.robots], dim=1)
    preload = ready.new_tensor([[0., 35.], [4., 30.], [30., 4.], [35., 0.]])[:, :env.arm_count]
    peak_normal = torch.zeros_like(preload)
    peak_capture = torch.zeros_like(preload)
    apply_action = env._apply_action

    def record_substep():
        apply_action()
        torch.maximum(peak_normal, env.pad_normal[:, :env.arm_count].amax(dim=-1), out=peak_normal)
        contacts = (env.parcel_state.contacts_A, env.parcel_state.contacts_B)[:env.arm_count]
        capture = torch.stack(contacts, dim=1)[..., ContactState.CAPTURE].amax(dim=-1)
        torch.maximum(peak_capture, capture, out=peak_capture)

    env._apply_action = record_substep
    try:
        for frame in range(40):
            close = transition(frame, 0, 24)
            request = transition(frame, 24, 31)
            desired = ready.clone()
            desired[..., -1] = torch.lerp(ready[..., -1], torch.full_like(preload, env.rail_close), close)
            desired[..., -1] += request * env.step_limit[-1] * preload / (1000. * env.rail_overtravel)
            q = torch.stack([env.motor_channels(robot.data.joint_pos.torch, robot) for robot in env.robots], dim=1)
            env.step(((desired - q) / env.step_limit).flatten(1))
    finally:
        env._apply_action = apply_action
    drive = torch.stack([robot.actuators.applied_effort.torch[:, robot.rail_ids].mean(dim=1)
                         for robot in env.robots], dim=1)
    pads = torch.stack([port.read().pads for port in env.ports], dim=1)
    gap = (pads[:, :, 1] - pads[:, :, 0]).norm(dim=-1)
    result = dict(commanded_N=preload.tolist(), motor_N=drive.tolist(), gap_mm=(1000. * gap).tolist(),
                  peak_effective_normal_N=peak_normal.tolist(), peak_capture=peak_capture.tolist())
    print("ROUTE_B_EMPTY_CLAMP " + json.dumps(result), flush=True)
    assert (gap < .0003).all() and (peak_normal == 0.).all() and (peak_capture == 0.).all()
    torch.testing.assert_close(drive, preload, atol=.35, rtol=0.)


@torch.inference_mode()
def benchmark(env, steps, free_before):
    env.reset()
    target = torch.stack([env.motor_channels(q, robot) for q, robot in zip(env.q_des, env.robots, strict=True)],
                         dim=1).clone()
    initial_pad_z = torch.stack([port.read().pads[..., 2].mean(dim=1) for port in env.ports], dim=1)

    def tick():
        q = torch.stack([env.motor_channels(robot.data.joint_pos.torch, robot) for robot in env.robots], dim=1)
        return env.step(((target - q) / env.step_limit).flatten(1))

    for _ in range(3):
        tick()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for _ in range(steps):
        obs, _, _, _, _ = tick()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    assert obs["policy"].isfinite().all() and env.parcel_state.payload.pos.isfinite().all()
    free_after, total = torch.cuda.mem_get_info()
    final_pad_z = torch.stack([port.read().pads[..., 2].mean(dim=1) for port in env.ports], dim=1)
    return dict(envs=env.num_envs, arms=env.arm_count, physical_steps=steps * 20, seconds=elapsed,
                env_steps_per_s=env.num_envs * steps * 20 / elapsed, policy_steps_per_s=env.num_envs * steps / elapsed,
                incremental_device_used_mib=(free_before - free_after) / 2**20, device_used_mib=(total - free_after) / 2**20,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                hover_max_abs_z_error_mm=1000. * (final_pad_z - initial_pad_z).abs().max().item(),
                use_newton_actuators=env.cfg.sim.use_newton_actuators)


def configure_implicit_arm_benchmark(cfg):
    """Use solver-native MIT drives for the arm, preserving the tested explicit rail pilot."""
    import warp as wp
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_physx.assets.articulation.articulation import Articulation

    from src.isaac_yam_parcel.yam_load_port import RailLoadedYam

    class ImplicitArmYam(RailLoadedYam):
        def write_data_to_sim(self):
            Articulation.write_data_to_sim(self)
            # The implicit drive adds its own PD effort. Submit the stock
            # arm feedforward, not the approximate combined current estimate.
            self._total_effort.copy_(wp.to_torch(self.actuators._joint_effort_target_sim))
            self._total_effort.index_add_(1, self.rail_indices, self.rail_load)
            if self._joint_order is None:
                self._backend_effort.copy_(self._total_effort)
            else:
                torch.index_select(self._total_effort, 1, self._joint_order, out=self._backend_effort)
            self.root_view.set_dof_actuation_forces(self._backend_effort_view, self._ALL_INDICES)

    for arm in (cfg.scene.arm_a, getattr(cfg.scene, "arm_b", None)):
        if arm is None:
            continue
        arm.class_type = ImplicitArmYam
        for name, limit in (("proximal", 28.), ("distal", 10.)):
            existing = arm.actuators[name]
            arm.actuators[name] = ImplicitActuatorCfg(
                joint_names_expr=existing.joint_names_expr, stiffness=existing.stiffness, damping=existing.damping,
                joint_effort_limit=limit, joint_velocity_limit=1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--benchmark", type=int)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--dual-arm", action="store_true")
    parser.add_argument("--native-actuators", action="store_true")
    parser.add_argument("--implicit-arm", action="store_true")
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--near-workcell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tracking-probe", action="store_true", help="Measure the 3.20 s C6 entry gate only.")
    parser.add_argument("--seat-fold", action="store_true", help="Try a shallow set-down after a high side entry.")
    parser.add_argument("--rapid-close", action="store_true", help="Use the maximum bounded rail step at engagement.")
    parser.add_argument("--cross-center", action="store_true", help="Seat the side fold just past the parcel center.")
    parser.add_argument("--lift-toward-base", action="store_true", help="Reduce the arm moment during the heavy lift.")
    parser.add_argument("--loaded-sweep", action="store_true", help="Use the calibrated fold sweep and loaded lift.")
    parser.add_argument("--loaded-arm-gains", action="store_true", help="Evaluate higher proximal gains within the same effort limits.")
    parser.add_argument("--empty-clamp", action="store_true", help="Verify 0/4/30/35 N close commands in free space.")
    parser.add_argument("--only-c4", action="store_true", help="Repeat the hard-book lane in an isolated environment.")
    parser.add_argument("--only-c5", action="store_true", help="Verify open-jaw contact and loaded closure in one C5 lane.")
    parser.add_argument("--trace")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream:
        parser.error("Use --headless and disable visualization/livestream.")
    if args.dual_arm and not (args.benchmark or args.empty_clamp):
        parser.error("Use --dual-arm with --benchmark or --empty-clamp.")
    if args.tracking_probe and args.seconds < 3.24:
        parser.error("The tracking probe requires at least 3.24 simulated seconds.")
    if args.implicit_arm and (not args.benchmark or args.native_actuators):
        parser.error("Use --implicit-arm with --benchmark and the standard Lab actuator path.")
    rows = [("C4", "top", -.00565, 0., 0.), ("C5", "top", -.036, 0., 4.),
            ("C6", "side-low", -.010, .020, 4.), ("C6", "side-high", .006, .020, 4.),
            ("C6", "side-high", .003, .030, 4.), ("C6", "side-high", .009, .024, 4.),
            ("C6", "side-high", .007, .024, 4.),
            ("C3", "side-high", .006, .020, 8.), ("C3", "side-high", .006, .020, 30.)]
    if args.only_c4:
        rows = rows[:1]
    if args.only_c5:
        if args.only_c4 or args.empty_clamp or args.benchmark:
            parser.error("--only-c5 selects the independent soft-cloth acceptance lane.")
        rows = rows[1:2]
    count = args.num_envs if args.benchmark else 4 if args.empty_clamp else len(rows)
    cfg = yam_parcel_cfg(count, dual_arm=args.dual_arm,
                         use_newton_actuators=args.native_actuators)
    if args.loaded_arm_gains or args.only_c5:
        cfg.scene.arm_a.actuators["proximal"].stiffness = 140.
        cfg.scene.arm_a.actuators["proximal"].damping = 6.6
    env_type = YAMHandoverPlaceEnv if args.dual_arm else YAMPickEnv
    with launch_simulation(cfg.sim, args):
        before = torch.cuda.mem_get_info()[0]
        if args.implicit_arm:
            configure_implicit_arm_benchmark(cfg)
        episode = None
        if not args.benchmark and not args.empty_clamp:
            dr = DomainRandomizer("cuda:0", 17)
            episode = CornerSuite(dr).select(dr.tensor([int(row[0][1:]) - 1 for row in rows], dtype=torch.long))
        env = env_type(cfg, episode=episode)
        try:
            if args.only_c5:
                # Align FK, contact, and wrench accounting to the lower CAD edge.
                env.ports[0].frames.offsets[1:, 2] -= env.scene_definition["pad_box_size_m"][2] / 2
            assert not env.sim.has_gui and not env.sim.has_offscreen_render and not env.scene.sensors
            if args.benchmark:
                result = benchmark(env, args.benchmark, before)
                print("ROUTE_B_THROUGHPUT " + json.dumps(dict(result, implicit_arm=args.implicit_arm)), flush=True)
            elif args.empty_clamp:
                empty_clamp(env)
            else:
                result, trace = acquisition(env, rows, args.seconds, seat_fold=args.seat_fold,
                                            rapid_close=args.rapid_close, cross_center=args.cross_center,
                                            lift_toward_base=args.lift_toward_base, loaded_sweep=args.loaded_sweep)
                if args.trace:
                    with open(args.trace, "w") as file:
                        json.dump(trace, file)
                print("ROUTE_B_CLOSED_LOOP " + json.dumps(result), flush=True)
                if args.only_c4:
                    assert result[0]["any_finger_frames"] == 0 and result[0]["peak_film_load_N"] == 0.
                    return
                if args.only_c5:
                    passed = c5_gate(result[0])
                    print("ROUTE_B_C5_GATE " + json.dumps(passed), flush=True)
                    if not passed:
                        raise SystemExit(1)
                    return
                if args.tracking_probe:
                    passed = all(abs(row["approach_end_z_error_mm"]) < 2. for row in result
                                 if row["corner"] == "C6" and row["route"] == "side-high")
                    if not passed:
                        raise SystemExit(1)
                    return
                gates = right_arm_gates(result)
                print("ROUTE_B_RIGHT_ARM_GATES " + json.dumps(gates), flush=True)
                if not all(gates.values()):
                    raise SystemExit(1)
        finally:
            env.close()


if __name__ == "__main__":
    main()
