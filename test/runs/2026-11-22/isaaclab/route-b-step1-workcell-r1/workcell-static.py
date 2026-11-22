import argparse
import json
import math
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from src.isaac_yam_parcel.yam_parcel_base_env import YAMHandoverPlaceEnv, yam_parcel_cfg


def get_positions(env):
    return torch.stack([env.motor_channels(robot.data.joint_pos.torch, robot) for robot in env.robots], dim=1)


def pair_events(target):
    from pxr import PhysicsSchemaTools

    def record(headers, _data, _friction):
        for header in headers:
            pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(key))
                                for key in (header.collider0, header.collider1)))
            if (any("Yam_A/" in body or "Yam_B/" in body for body in pair)
                    and all("/env_" in body for body in pair)):
                target.add(pair)

    return record


@torch.inference_mode()
def trial(env, candidates, events, interpolate):
    env.reset()
    q = [robot.data.default_joint_pos.torch.clone() for robot in env.robots]
    ready = get_positions(env)
    for i, robot in enumerate(env.robots):
        for lane, pair in enumerate(candidates, start=3):
            q[i][lane, :6] = q[i].new_tensor(pair[i]["joints"])
            q[i][lane, robot.rail_ids] = -.024
        if not interpolate:
            robot.write_joint_position_to_sim_index(position=q[i])
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q[i]))
    commands = torch.stack([env.motor_channels(joints, robot) for joints, robot in zip(q, env.robots)], dim=1)
    rail_goal = commands[:, 0, -1].clone()
    preload = commands.new_zeros((env.num_envs, 1))
    preload[:3, 0] = preload.new_tensor([0., 4., 30.])
    env.enabled.zero_()
    target = commands.clone()
    for step in range(120 if interpolate else 90):
        if step == 2:
            initial_gap = pad_gap(env.ports[0])
            events.clear()
        fraction = min(1., max(0., (step - 8) / 25.))
        commands[:3, 0, -1] = rail_goal[:3].lerp(torch.full_like(rail_goal[:3], env.rail_close), fraction)
        env.set_grip_preload(0, preload * min(1., max(0., (step - 33) / 12.)))
        if interpolate:
            phase = min(1., max(0., (step - 10) / 83.))
            fraction = .5 - .5 * math.cos(math.pi * phase)
            commands[3:] = ready[3:].lerp(target[3:], fraction)
        action = (commands - get_positions(env)) / env.step_limit
        env.step(action.flatten(1))
    motor = env.robots[0].actuators.applied_effort.torch.index_select(1, env.robots[0].rail_indices)
    actual = env.robots[0].data.joint_pos.torch.index_select(1, env.robots[0].rail_indices)
    gap = pad_gap(env.ports[0])
    center = torch.stack([port.read().pads.mean(dim=1) for port in env.ports], dim=1)
    root = lambda lane: f"/env_{lane}/"
    results = []
    for lane in range(env.num_envs):
        contacts = [pair for pair in sorted(events) if any(root(lane) in body for body in pair)]
        if lane < 3:
            results.append(dict(lane=lane, preload_N=preload[lane, 0].item(),
                                initial_gap_mm=initial_gap[lane].item(), gap_mm=gap[lane].item(),
                                rail_pos_mm=(1000. * actual[lane]).tolist(),
                                motor_N=motor[lane].tolist(),
                                external_N=env.robots[0].rail_load[lane].tolist(), contact_pairs=contacts))
        else:
            desired = center.new_tensor([arm["target"] for arm in candidates[lane - 3]])
            results.append(dict(lane=lane, center_m=center[lane].tolist(), target_m=desired.tolist(),
                                error_mm=(1000. * (center[lane] - desired).norm(dim=-1)).tolist(),
                                largest_joint_error_rad=(get_positions(env)[lane, :, :6] -
                                                         commands[lane, :, :6]).abs().amax(dim=-1).tolist(),
                                contact_pairs=contacts))
    print("ROUTE_B_WORKCELL_STATIC " + json.dumps(results), flush=True)


def pad_gap(port):
    tips = port.read().pads
    return (tips[:, 1] - tips[:, 0]).norm(dim=-1) * 1000.


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", required=True)
parser.add_argument("--candidates", required=True, type=Path)
parser.add_argument("--interpolate", action="store_true")
add_launcher_args(parser)
parser.set_defaults(visualizer=[], livestream=0)
args = parser.parse_args()
candidates = [json.loads(line) for line in args.candidates.read_text().splitlines()]
assert candidates
cfg = yam_parcel_cfg(3 + len(candidates), dual_arm=True)
cfg.scene.arm_a.spawn.activate_contact_sensors = True
cfg.scene.arm_b.spawn.activate_contact_sensors = True
with launch_simulation(cfg.sim, args):
    from omni.physics.core import get_physics_simulation_interface

    env = YAMHandoverPlaceEnv(cfg, curriculum_alpha=0.)
    events = set()
    listener = get_physics_simulation_interface().subscribe_physics_contact_report_events(pair_events(events))
    try:
        trial(env, candidates, events, args.interpolate)
    finally:
        listener = None
        env.close()
