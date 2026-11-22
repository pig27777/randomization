"""Run deterministic ABC pad-space commands through real YAM joint targets."""

import argparse
import json
import math
from time import perf_counter

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from src.isaac_yam_parcel.yam_parcel_base_env import YAMHandoverPlaceEnv, YAMPickEnv, yam_parcel_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.lumped_contact_proxy import ContactState


def transition(time, start, end):
    phase = min(1., max(0., (time - start) / (end - start)))
    return .5 - .5 * math.cos(math.pi * phase)


@torch.inference_mode()
def acquisition(env, rows, seconds, *, near_workcell):
    """Drive the approach using only planned workcell coordinates and YAM FK."""
    env.reset()
    robot = env.robots[0]
    port = env.ports[0]
    sample = port.read()
    q_ready = sample.wrist_quat_xyzw.clone()
    wrist_pose = robot.data.body_link_pose_w.torch[:, port.wrist_id]
    origin = env.scene.env_origins
    start = sample.pads.mean(dim=1).clone()
    workcell_xy = start.new_tensor([-.03, .225]) if near_workcell else start.new_zeros((2,))
    shift = torch.cat((workcell_xy, start.new_zeros((1,))))
    initial_state = env.parcel_state
    env.parcel_state = initial_state._replace(
        payload=initial_state.payload._replace(pos=initial_state.payload.pos + shift),
        side_origin=initial_state.side_origin + shift)
    offset = quat_apply_inverse(wrist_pose[:, 3:], start + origin - wrist_pose[:, :3])
    start_rail = robot.data.default_joint_pos.torch[:, robot.rail_ids].mean(dim=1)
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                       ik_method="dls", ik_params={"lambda_val": .035}),
                                  env.num_envs, str(env.device))
    start_goal = start.clone()
    top_row = start.new_tensor([row[1] == "top" for row in rows]).bool()
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
    for step in range(round(seconds / env.step_dt)):
        time = step * env.step_dt
        initial = transition(time, .35, 1.85)
        approach = transition(time, 1.85, 3.20)
        closure = transition(time, 2.80, 4.05)
        rise = transition(time, 4.65, 8.1)
        command = torch.lerp(torch.lerp(start_goal, entry, initial), engage, approach)
        command = torch.lerp(command, offer, rise)
        rail = torch.lerp(start_rail, rail_target, closure)
        wrist_pose = robot.data.body_link_pose_w.torch[:, port.wrist_id]
        wrist = wrist_pose[:, :3] - origin
        jac = robot.data.body_link_jacobian_w.torch[:, port.wrist_id - 1, :, :6]
        q = robot.data.joint_pos.torch
        goal = command - quat_apply(q_ready, offset)
        ik.set_command(torch.cat((goal, q_ready), dim=-1))
        target = ik.compute(wrist, wrist_pose[:, 3:], jac, q[:, :6])
        # Kinematic gravity compensation uses only the arm's URDF and joint state.
        stiffness = start.new_tensor([80.] * 3 + [10.] * 3)
        compensation = robot.data.gravity_compensation_forces.torch[:, :6] / stiffness
        arm_delta = target + compensation - q[:, :6]
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
        timestamps.append(time)

    data = torch.stack(observations).cpu()
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
                            pad_min_height_mm=1000 * series[:, 6].amin().item(),
                            finite=bool(series.isfinite().all())))
    return results, dict(times_s=timestamps, observations=data.tolist())


@torch.inference_mode()
def benchmark(env, steps, free_before):
    env.reset()
    target = torch.stack([env.motor_channels(q, robot) for q, robot in zip(env.q_des, env.robots, strict=True)],
                         dim=1).clone()

    def tick():
        q = torch.stack([env.motor_channels(robot.data.joint_pos.torch, robot) for robot in env.robots], dim=1)
        return env.step(((target - q) / env.step_limit).flatten(1))

    tick()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for _ in range(steps):
        obs, _, _, _, _ = tick()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    assert obs["policy"].isfinite().all()
    free_after, total = torch.cuda.mem_get_info()
    return dict(envs=env.num_envs, arms=env.arm_count, physical_steps=steps * 20, seconds=elapsed,
                env_steps_per_s=env.num_envs * steps * 20 / elapsed, policy_steps_per_s=env.num_envs * steps / elapsed,
                incremental_device_used_mib=(free_before - free_after) / 2**20, device_used_mib=(total - free_after) / 2**20,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                use_newton_actuators=env.cfg.sim.use_newton_actuators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--benchmark", type=int)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--dual-arm", action="store_true")
    parser.add_argument("--native-actuators", action="store_true")
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--near-workcell", action="store_true")
    parser.add_argument("--trace")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream:
        parser.error("Use --headless and disable visualization/livestream.")
    rows = [("C4", "top", -.012, 0., -.006), ("C5", "top", -.014, 0., -.006),
            ("C6", "side-low", -.010, .020, -.006), ("C6", "side-high", .006, .020, -.006),
            ("C6", "side-high", .003, .030, -.003), ("C6", "side-high", .009, .024, -.005),
            ("C3", "top", -.014, 0., -.012), ("C3", "top", -.014, 0., -.003)]
    cfg = yam_parcel_cfg(args.num_envs if args.benchmark else len(rows), dual_arm=args.dual_arm,
                         use_newton_actuators=args.native_actuators)
    env_type = YAMHandoverPlaceEnv if args.dual_arm else YAMPickEnv
    with launch_simulation(cfg.sim, args):
        before = torch.cuda.mem_get_info()[0]
        episode = None
        if not args.benchmark:
            dr = DomainRandomizer("cuda:0", 17)
            episode = CornerSuite(dr).select(dr.tensor([int(row[0][1:]) - 1 for row in rows], dtype=torch.long))
        env = env_type(cfg, episode=episode)
        try:
            assert not env.sim.has_gui and not env.sim.has_offscreen_render and not env.scene.sensors
            if args.benchmark:
                print("ROUTE_B_THROUGHPUT " + json.dumps(benchmark(env, args.benchmark, before)), flush=True)
            else:
                result, trace = acquisition(env, rows, args.seconds, near_workcell=args.near_workcell)
                if args.trace:
                    with open(args.trace, "w") as file:
                        json.dump(trace, file)
                print("ROUTE_B_CLOSED_LOOP " + json.dumps(result), flush=True)
                c6_pass = any(row["corner"] == "C6" and row["route"] == "side-high" and row["finite"]
                              and row["longest_hold_s"] >= 3. and row["last3_max_speed_mm_s"] < 20.
                              and row["last3_captured_frames"] == 75 for row in result)
                if not c6_pass:
                    raise SystemExit(1)
        finally:
            env.close()


if __name__ == "__main__":
    main()
