import argparse
import json
from pathlib import Path
import runpy

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from src.isaac_yam_parcel.joint_current_filter import JointCurrentParams
from src.isaac_yam_parcel.yam_parcel_base_env import YAMHandoverPlaceEnv, yam_parcel_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.lumped_contact_proxy import ContactState

replay = runpy.run_path("test/isaaclab/check_route_b_closed_loop.py")
transition = replay["transition"]


class C10Env(YAMHandoverPlaceEnv):
    record_tug = False

    def _current_params(self):
        shape = self.scene.env_origins.new_ones((self.num_envs, 2, 7))
        lever = shape.new_tensor([.25, .30, .25, .12, .09, .055])
        lever = torch.cat((lever, lever.new_tensor([1.])))
        gain = shape * shape.new_tensor([.75, .75, .75, .75, .75, .75, .8])
        deadzone = shape * lever * shape.new_tensor([2., 2., 2., 2., 2., 2., .25])
        bias = shape * lever * shape.new_tensor([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.])
        noise = shape * lever * shape.new_tensor([.6, .6, .6, .6, .6, .6, .04])
        return JointCurrentParams(gain, deadzone, .5 * deadzone, bias, 4. * shape, noise,
                                  torch.full_like(shape, 25, dtype=torch.long), shape[..., :1] * .002)

    def step(self, actions):
        if actions.shape[1] == 7:
            actions = torch.cat((actions, torch.zeros_like(actions)), dim=1)
        return super().step(actions)

    def _apply_action(self):
        super()._apply_action()
        torch.maximum(self.peak_left_film, self.film_load[:, 1, 0], out=self.peak_left_film)
        torch.maximum(self.peak_left_gather, self.parcel_state.contacts_B[..., ContactState.SLACK].amin(dim=-1),
                      out=self.peak_left_gather)
        if self.record_tug:
            self.tug_truth.append(self.result.wrist_wrench[0, :, 1].clone())


def motor_pose(env):
    return torch.stack([env.motor_channels(robot.data.joint_pos.torch, robot)
                        for robot in env.robots], dim=1)


def status(env, label):
    state = env.parcel_state
    payload = env.parcel_sim.payload
    captures = [getattr(state, arm)[0, :, ContactState.CAPTURE].tolist() for arm in ("contacts_A", "contacts_B")]
    local_pads = [(p.read().pads - state.payload.pos.mean(dim=1, keepdim=True)) @ payload.frame(state.payload)
                  for p in env.ports]
    skin = [(payload.weights_at(points[..., :2]) @ payload.half_thickness(state.payload.d_surf).unsqueeze(-1)).squeeze(-1)
            for points in local_pads]
    flags = dict(label=label, contact_pairs=sorted(env.arm_contacts), peak_left_film_N=env.peak_left_film[0].item(),
                 peak_left_bilateral_gather_mm=1000. * env.peak_left_gather[0].item(),
                 folds_armed=env.entered_above_lip[0].tolist(),
                 lip_gap_mm=[(1000. * (points[..., 2] - height)[0]).tolist()
                             for points, height in zip(local_pads, skin)],
                 film_load_N=env.film_load[0, :, 0].tolist(), com_m=payload.com(state.payload)[0].tolist(),
                 gap_mm=env.result.table_clearance[0, 0].item() * 1000.,
                 supports_N=env.result.support_N[0, 0].item(), captures=captures,
                 handover_seen=env.handover_seen[0].item(),
                 pad_mm=[(1000. * p.read().pads[0].mean(dim=0)).tolist() for p in env.ports],
                 net_feed_mm=(1000. * env.result.net_feed[0]).tolist(),
                 gathered_mm=[(1000. * getattr(state, arm)[0, :, ContactState.SLACK]).tolist()
                              for arm in ("contacts_A", "contacts_B")],
                 torn=bool(state.torn_flag.any()))
    print("ROUTE_B_C10_STATE " + json.dumps(flags), flush=True)
    return flags


def joint_goal(env, target, seconds, grip, *, rail=None, sensor=None):
    start = motor_pose(env)
    steps = round(seconds / env.step_dt)
    for step in range(steps):
        now = step * env.step_dt
        blend = transition(now, 0., max(seconds - .24, env.step_dt))
        desired = start.lerp(target, blend)
        preload = grip(now) if callable(grip) else grip
        if rail is not None:
            desired[..., -1] = rail(now, desired[..., -1])
        desired[..., -1] += env.step_limit[-1] * desired.new_tensor(preload) / (1000. * env.rail_overtravel)
        try:
            action = ((desired - motor_pose(env)) / env.step_limit).flatten(1)
            env.step(action)
        except RuntimeError as error:
            raise RuntimeError(f"C10 motion step {step} failed") from error
        if sensor is not None:
            sensor()


@torch.inference_mode()
def run(env, seed):
    trace, _ = replay["acquisition"](env, [("C10", "side-high", .006, .020, 9.)],
                                      10.5, rapid_close=True, loaded_sweep=True)
    print("ROUTE_B_C10_PICK " + json.dumps(trace[0]), flush=True)
    right = status(env, "right_real_pick")
    if right["captures"][0] != [1., 1.] or right["gap_mm"] <= 1.:
        return False
    closed = env.rail_close
    physical = motor_pose(env)
    physical[:, 0, -1] = closed
    target = physical.clone()

    def pose(arm, label, opening=None):
        target[:, arm, :6] = physical.new_tensor(seed[label]["q"])
        if opening is not None:
            target[:, arm, -1] = opening

    pose(0, "right_present")
    joint_goal(env, target, 3.52, (9., 0.))
    status(env, "right_present")
    pose(1, "left_observe", -.036)
    joint_goal(env, target, 2.8, (9., 0.))
    status(env, "left_observe")
    pose(1, "left_ready_feed")
    joint_goal(env, target, 1.64, (9., 0.))
    status(env, "receiver_ready_feed")
    pose(1, "left_capture")
    joint_goal(env, target, 1.52, (9., 0.))
    status(env, "receiver_entry")
    if env.arm_contacts:
        return False
    target[:, 1, -1] = closed

    def receiver_rail(t, previous):
        planned = previous.clone()
        move = transition(t, .02, .42)
        sweep = transition(t, .42, .76)
        planned[:, 0] = closed
        planned[:, 1] = (-.036 * (1. - move) + (closed - .008) * move) * (1. - sweep) + closed * sweep
        return planned

    joint_goal(env, target, 1.40, lambda t: (9., 4. * transition(t, .47, .82) + 8. * transition(t, .88, 1.23)), rail=receiver_rail)
    received = status(env, "receiver_close")
    if received["captures"] != [[1., 1.], [1., 1.]] or not received["handover_seen"]:
        return False
    pose(1, "left_taut")
    joint_goal(env, target, 1.76, (9., 12.))
    status(env, "receiver_preload")
    baseline = env.sensed[0, :, :6].clone()
    signal = []
    env.tug_truth = []
    env.record_tug = True
    hold = target.clone()

    def sensor():
        effort = env.sensed[0, :, :6] - baseline
        axis = torch.stack([port.read().pad_jacobian[0, :, 1, :6].mean(dim=0) for port in env.ports])
        signal.append((-(effort * axis).sum(dim=-1) / axis.square().sum(dim=-1).clamp_min(1e-5)).clone())

    pose(0, "right_pull")
    pose(1, "left_pull")
    joint_goal(env, target, .24, (9., 12.), sensor=sensor)
    target.copy_(hold)
    joint_goal(env, target, .40, (9., 12.), sensor=sensor)
    env.record_tug = False
    sign = target.new_tensor([1., -1.])
    currents = torch.stack(signal) * sign
    loads = torch.stack(env.tug_truth) * sign
    consecutive = longest = 0
    for detected in ((currents > 1.5).all(dim=-1)).tolist():
        consecutive = consecutive + 1 if detected else 0
        longest = max(longest, consecutive)
    sustained = (loads.amin(dim=-1) >= 5.5) & (loads.amax(dim=-1) <= 6.5)
    duration_ms = 2 * max((len(segment) for segment in "".join("1" if x else "0" for x in
                                                               sustained.tolist()).split("0")), default=0)
    print("ROUTE_B_C10_TUG " + json.dumps(dict(
        grip_current_projected_N=currents.tolist(), true_peak_pair_N=loads.amin(dim=-1).amax().item(),
        sustained_ms=duration_ms, detected_frames=longest)), flush=True)
    if duration_ms < 130 or longest < 4:
        return False
    joint_goal(env, target, .64, lambda t: (9. * (1. - transition(t, .06, .55)), 12.))
    target[:, 0, -1] = -.030
    joint_goal(env, target, 1.04, (0., 12.))
    status(env, "right_release")
    pose(1, "left_carry")
    joint_goal(env, target, 4.16, (0., 12.))
    pose(1, "left_unload")
    joint_goal(env, target, 5.28, (0., 12.))
    status(env, "left_unload")
    joint_goal(env, target, .72, lambda t: (0., 12. * (1. - transition(t, .1, .58))))
    target[:, 1, -1] = -.034
    joint_goal(env, target, .86, (0., 0.))
    target[:, 1] = env.motor_channels(env.robots[1].data.default_joint_pos.torch, env.robots[1])
    joint_goal(env, target, 1.8, (0., 0.))
    placed = status(env, "left_retract")
    return env.place_streak[0].item() >= 250 and not placed["torn"]


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", required=True)
parser.add_argument("--poses", type=Path, required=True)
add_launcher_args(parser)
parser.set_defaults(visualizer=[], livestream=0)
args = parser.parse_args()
poses = {record["name"]: record for record in map(json.loads, args.poses.read_text().splitlines())}
cfg = yam_parcel_cfg(1, dual_arm=True)
cfg.episode_length_s = 45.
cfg.scene.arm_a.spawn.activate_contact_sensors = True
cfg.scene.arm_b.spawn.activate_contact_sensors = True
for arm in (cfg.scene.arm_a, cfg.scene.arm_b):
    arm.actuators["proximal"].stiffness = 140.
    arm.actuators["proximal"].damping = 6.6
with launch_simulation(cfg.sim, args):
    dr = DomainRandomizer("cuda:0", 17)
    episode = CornerSuite(dr).select(dr.tensor([9], dtype=torch.long))
    env = C10Env(cfg, episode=episode)
    from omni.physics.core import get_physics_simulation_interface
    from pxr import PhysicsSchemaTools

    env.arm_contacts = set()

    def contacts(headers, _data, _friction):
        for header in headers:
            ids = (header.collider0, header.collider1)
            a, b = sorted(str(PhysicsSchemaTools.intToSdfPath(x)) for x in ids)
            if ("Yam_A/" in a and "Yam_B/" in b) or ("/Table" in a and "Yam_" in b):
                env.arm_contacts.add((a, b))

    subscription = get_physics_simulation_interface().subscribe_physics_contact_report_events(contacts)
    env.peak_left_film = torch.zeros(env.num_envs, device=env.device)
    env.peak_left_gather = torch.zeros_like(env.peak_left_film)
    try:
        passed = run(env, poses)
        print("ROUTE_B_C10_PASS " + str(passed), flush=True)
        raise SystemExit(0 if passed else 1)
    finally:
        subscription = None
        env.close()
