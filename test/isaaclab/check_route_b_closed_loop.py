"""Run deterministic ABC pad-space commands through real YAM joint targets."""

import argparse
import json
import math
from time import perf_counter

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from src.isaac_yam_parcel.yam_parcel_base_env import YAMHandoverPlaceEnv, YAMPickEnv, yam_parcel_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.lumped_contact_proxy import ContactState


def transition(time, start, end):
    phase = min(1., max(0., (time - start) / (end - start)))
    return .5 - .5 * math.cos(math.pi * phase)


class PadIK(DifferentialIKController):
    """Track a pad center in metres and give orientation lower task-space weight."""

    def _compute_pose_task(self, ee_pos, ee_quat, jacobian):
        jac, error = super()._compute_pose_task(ee_pos, ee_quat, jacobian)
        scale = error.new_tensor([6., 6., 6., 1., 1., 1.])
        return jac * scale[:, None], error * scale


@torch.inference_mode()
def acquisition(env, rows, seconds, *, near_workcell, seat_fold=False):
    """Drive the approach using only planned workcell coordinates and YAM FK."""
    env.reset()
    robot = env.robots[0]
    port = env.ports[0]
    sample = port.read()
    q_ready = sample.wrist_quat_xyzw.clone()
    start = sample.pads.mean(dim=1).clone()
    workcell_xy = start.new_tensor([-.03, .225]) if near_workcell else start.new_zeros((2,))
    shift = torch.cat((workcell_xy, start.new_zeros((1,))))
    initial_state = env.parcel_state
    env.parcel_state = initial_state._replace(
        payload=initial_state.payload._replace(pos=initial_state.payload.pos + shift),
        side_origin=initial_state.side_origin + shift)
    start_rail = robot.data.default_joint_pos.torch[:, robot.rail_ids].mean(dim=1)
    ik = PadIK(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                       ik_method="dls", ik_params={"lambda_val": .035}),
                                  env.num_envs, str(env.device))
    start_goal = start.clone()
    top_row = start.new_tensor([row[1] == "top" for row in rows]).bool()
    high_entry = start.new_tensor([row[1] == "side-high" for row in rows]).bool()
    # Rows encode each fixture's known starting dimensions. These are fixed
    # initial workcell parameters, with no parcel tracking after the start.
    half = env.episode.payload.bag_half_extents.clone()
    top = 2. * half[:, 2]
    entry = torch.stack((torch.zeros_like(top), half[:, 1] + .037,
                         top + start.new_tensor([row[2] for row in rows])), dim=-1)
    engage = entry.clone()
    engage[:, 1] = torch.where(top_row, 0., start.new_tensor([row[3] for row in rows]))
    entry[:, 1] = torch.where(top_row, 0., entry[:, 1])
    entry[:, 2] = torch.where(top_row, top + .028, entry[:, 2])
    offer = engage.clone()
    offer[:, 1] = .027
    offer[:, 2] = .235
    entry += shift
    engage += shift
    offer += shift
    rail_target = start.new_tensor([row[4] for row in rows])
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
        if seat_fold:
            # Enter above the rim first, then lower at fixed XY while closing.
            # It is a preset geometric trajectory, with no parcel state feedback.
            seat = transition(time, 3.24, 3.78) * (1. - rise)
            command[:, 2] -= high_entry * (engage[:, 2] - top + .002) * seat
            rail = torch.where(high_entry, torch.lerp(start_rail, rail_target,
                                                      transition(time, 3.45, 4.32)), rail)
        sample = port.read()
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
                                       env.parcel_sim.payload.com(state.payload)), dim=-1))
        tracking.append(command)
        timestamps.append(time)

    data = torch.stack(observations).cpu()
    goals = torch.stack(tracking).cpu()
    approach_end = min(round(3.2 / env.step_dt), len(timestamps) - 1)
    tail = round(3. / env.step_dt)
    results = []
    for index, row in enumerate(rows):
        series = data[:, index]
        both = (series[:, :2] == 1.).all(dim=1)
        results.append(dict(corner=row[0], route=row[1], entry_offset_m=row[2],
                            any_double_capture=bool(both.any()), captured_frames=int(both.sum()),
                            last3_captured_frames=int(both[-tail:].sum()),
                            peak_clearance_mm=1000 * series[:, 3].amax().item(),
                            last3_min_clearance_mm=1000 * series[-tail:, 3].amin().item(),
                            last3_max_speed_mm_s=1000 * series[-tail:, 2].amax().item(),
                            longest_hold_s=env.longest_hold[index].item() * .002,
                            peak_motor_effort_N=series[:, 7:9].abs().amax().item(),
                            peak_gathered_mm=1000 * series[:, 9:11].amax().item(),
                            peak_feed_mm=1000 * series[:, 11:13].amax().item(),
                            peak_press_N=series[:, 13].amax().item(),
                            peak_package_translation_mm=1000 * (series[:, -3:-1] -
                                  series[:1, -3:-1]).norm(dim=1).amax().item(),
                            approach_end_z_target_mm=1000 * goals[approach_end, index, 2].item(),
                            approach_end_z_actual_mm=1000 * series[approach_end, 6].item(),
                            approach_end_z_error_mm=1000 * (series[approach_end, 6] -
                                                                   goals[approach_end, index, 2]).item(),
                            pad_min_height_mm=1000 * series[:, 6].amin().item(),
                            finite=bool(series.isfinite().all())))
    return results, dict(times_s=timestamps, observations=data.tolist(), pad_target_m=goals.tolist())


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
    parser.add_argument("--near-workcell", action="store_true")
    parser.add_argument("--tracking-probe", action="store_true", help="Measure the 3.20 s C6 entry gate only.")
    parser.add_argument("--seat-fold", action="store_true", help="Try a shallow set-down after a high side entry.")
    parser.add_argument("--trace")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream:
        parser.error("Use --headless and disable visualization/livestream.")
    if args.tracking_probe and args.seconds < 3.24:
        parser.error("The tracking probe requires at least 3.24 simulated seconds.")
    if args.implicit_arm and (not args.benchmark or args.native_actuators):
        parser.error("Use --implicit-arm with --benchmark and the standard Lab actuator path.")
    rows = [("C4", "top", -.012, 0., -.006), ("C5", "top", -.014, 0., -.006),
            ("C6", "side-low", -.010, .020, -.006), ("C6", "side-high", .006, .020, -.006),
            ("C6", "side-high", .003, .030, -.003), ("C6", "side-high", .009, .024, -.005),
            ("C3", "top", -.014, 0., -.012), ("C3", "top", -.014, 0., -.003)]
    cfg = yam_parcel_cfg(args.num_envs if args.benchmark else len(rows), dual_arm=args.dual_arm,
                         use_newton_actuators=args.native_actuators)
    env_type = YAMHandoverPlaceEnv if args.dual_arm else YAMPickEnv
    with launch_simulation(cfg.sim, args):
        before = torch.cuda.mem_get_info()[0]
        if args.implicit_arm:
            configure_implicit_arm_benchmark(cfg)
        episode = None
        if not args.benchmark:
            dr = DomainRandomizer("cuda:0", 17)
            episode = CornerSuite(dr).select(dr.tensor([int(row[0][1:]) - 1 for row in rows], dtype=torch.long))
        env = env_type(cfg, episode=episode)
        try:
            assert not env.sim.has_gui and not env.sim.has_offscreen_render and not env.scene.sensors
            if args.benchmark:
                result = benchmark(env, args.benchmark, before)
                print("ROUTE_B_THROUGHPUT " + json.dumps(dict(result, implicit_arm=args.implicit_arm)), flush=True)
            else:
                result, trace = acquisition(env, rows, args.seconds, near_workcell=args.near_workcell,
                                            seat_fold=args.seat_fold)
                if args.trace:
                    with open(args.trace, "w") as file:
                        json.dump(trace, file)
                print("ROUTE_B_CLOSED_LOOP " + json.dumps(result), flush=True)
                if args.tracking_probe:
                    passed = all(abs(row["approach_end_z_error_mm"]) < 2. for row in result
                                 if row["corner"] == "C6" and row["route"] == "side-high")
                    if not passed:
                        raise SystemExit(1)
                    return
                c6_pass = any(row["corner"] == "C6" and row["route"] == "side-high" and row["finite"]
                              and row["longest_hold_s"] >= 3. and row["last3_max_speed_mm_s"] < 20.
                              and row["last3_captured_frames"] == 75 for row in result)
                if not c6_pass:
                    raise SystemExit(1)
        finally:
            env.close()


if __name__ == "__main__":
    main()
