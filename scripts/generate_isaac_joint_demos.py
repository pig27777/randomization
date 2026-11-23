"""Collect successful 25 Hz Yam_B joint demonstrations in the ABC basket scene."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.controllers import DifferentialIKControllerCfg

from src.isaac_yam_parcel.yam_bin_env import YAMSingleArmBinEnv, load_bin_layout, yam_bin_cfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.isaac_yam_parcel.joint_ik import PadIK, transition


class DemonstrationEnv(YAMSingleArmBinEnv):
    def __init__(self, cfg, layout, episode, pick_xy):
        self.pick_xy = pick_xy
        super().__init__(cfg, layout, episode=episode)

    def _parcel_table_state(self):
        # The teacher knows the fixture only at reset. It never steers from
        # the loaded parcel's true position, velocity, or capture state.
        state = self.episode.table_state(self.parcel_sim)
        shift = torch.cat((self.pick_xy, torch.zeros_like(self.pick_xy[:, :1])), dim=1)
        return state._replace(payload=state.payload._replace(pos=state.payload.pos + shift[:, None]),
                              side_origin=state.side_origin + shift[:, None])

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        self.extras["terminal_phys"] = torch.stack((
            self.rim_scrape, self.peak_contact > .02, self.dropped_early,
            self.parcel_state.torn_flag.flatten(1).any(dim=1)), dim=1)
        return terminated, timeout


class ProbeEnv(DemonstrationEnv):
    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        force = self.contacts.data.net_normal_forces_w.torch.norm(dim=-1)
        self.extras["contact_forces"] = force
        kinematics = self.ports[0].read()
        self.extras["probe_pad"] = kinematics.pads.mean(dim=1)
        separation = kinematics.pads[:, 1] - kinematics.pads[:, 0]
        self.extras["probe_signed_gap_mm"] = 1000. * (separation * kinematics.rail_axis[:, 0]).sum(dim=1)
        return terminated, timeout


def make_fixtures(layout, count, corner_numbers, seed, jitter, device):
    dr = DomainRandomizer(device, seed)
    corners = dr.tensor(corner_numbers, dtype=torch.long)[torch.arange(count, device=device) % len(corner_numbers)]
    episode = CornerSuite(dr).select(corners - 1)
    rank = dr._uniform((count, 3)) - .5
    episode = replace(episode, yaw_rad=episode.yaw_rad + rank[:, :1] * .12 * jitter)
    pick = dr.tensor(layout["pick"]["center_m"][:2]) + rank[:, 1:] * .016 * jitter
    # Leave room for the small displacement during the approach along +Y.
    pick[:, 1] -= .026
    basket = layout["bin"]
    near = basket["center_m"][1] - basket["inner_half_xy_m"][1] - basket["wall_thickness_m"] - .002
    x, y = episode.payload.bag_half_extents[:, :2].unbind(dim=1)
    yaw = episode.yaw_rad[:, 0]
    pick[:, 1] = torch.minimum(pick[:, 1], near - (x * yaw.sin().abs() + y * yaw.cos().abs()))
    return episode, corners, pick


class BinIKTeacher:
    def __init__(self, env, corners, rescue_fraction):
        self.env = env
        self.robot = env.robots[0]
        self.port = env.ports[0]
        geometry = self.port.read()
        self.ready = geometry.pads.mean(dim=1).clone()
        self.orientation = geometry.wrist_quat_xyzw.clone()
        self.ik = PadIK(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                   ik_method="dls", ik_params={"lambda_val": .035}),
                        env.num_envs, str(env.device))
        self.heavy = corners == 3
        self.book = corners == 4
        self.light = corners == 6
        self.top = corners == 5
        self.miss = torch.rand(env.num_envs, generator=env.randomizer.rng, device=env.device) < rescue_fraction
        self.recover = torch.zeros_like(self.miss)
        self.retry_decided = False
        self.height = 2. * env.episode.payload.bag_half_extents[:, 2]
        hy = env.episode.payload.bag_half_extents[:, 1]
        x, y = env.pick_xy.unbind(dim=1)
        z = self.height + .006
        self.entry = torch.stack((x, y - hy - .037, z), dim=1)
        self.seat = torch.stack((x, y - torch.where(self.book, .024, .020), z), dim=1)
        self.entry[self.top, 1] = y[self.top]
        self.entry[self.top, 2] = self.height[self.top] + .010
        self.seat[self.top, 1] = y[self.top]
        self.seat[self.top, 2] = .006
        self.lift = torch.stack((x - self.heavy * .052, y - .027,
                                 torch.where(self.heavy, .300, .235)), dim=1)
        self.over = self.lift.clone()
        self.over[:, 0] = torch.where(self.heavy, -.120, -.085)
        self.over[:, 2] = torch.where(self.heavy, .525, .360)
        self.bin_high = self.over.clone()
        self.bin_high[:, :2] = env.bin_center[:2]
        # Use the basket quadrant nearest the base to keep the descent away
        # from the arm's extended-reach posture.
        self.bin_high[:, 0] -= .065
        self.bin_high[:, 1] -= .012
        self.lower = self.bin_high.clone()
        self.lower[:, 2] = torch.where(self.heavy, .245, .215)
        self.withdraw = self.lower.clone()
        self.withdraw[:, 2] = .31
        self.open_rail = self.ready.new_full((env.num_envs,), -.040)
        self.closed = torch.full_like(self.open_rail, env.rail_close)

    def grasp(self, t):
        command = torch.lerp(self.ready, self.entry, transition(t, .35, 1.85))
        command = torch.lerp(command, self.seat, transition(t, 1.85, 3.20))
        # C4 can enter above the rim without advancing the paperboard. Seat
        # the jaws into vertical overlap only after that high-entry approach
        # to supply real transverse contact during the closing stroke.
        command[:, 2] -= self.book * .012 * transition(t, 3.20, 3.58)
        command = torch.lerp(command, self.lift, transition(t, 4.65, 8.10))
        slow = torch.lerp(self.open_rail, self.closed - .008, transition(t, 3.08, 3.48))
        slow = torch.lerp(slow, self.closed, transition(t, 3.48, 3.76))
        fast = self.closed if t >= 3.28 else self.open_rail
        rail = torch.where(self.light, slow, fast)
        book_close = torch.lerp(self.open_rail, self.closed - .008, transition(t, 3.52, 4.00))
        book_close = torch.lerp(book_close, self.closed, transition(t, 4.00, 4.40))
        rail = torch.where(self.book, book_close, rail)
        rail = torch.where(self.top, torch.lerp(self.open_rail, self.closed, transition(t, 2.92, 3.26)), rail)
        grip = torch.where(self.light, 4. * transition(t, 3.53, 3.80),
                           8. * float(t >= 3.28) + self.heavy * 22. * transition(t, 3.62, 4.32))
        grip = torch.where(self.book, 14.5 * transition(t, 4.25, 4.48), grip)
        grip = torch.where(self.top, 15. * transition(t, 3.22, 3.45), grip)
        return command, rail, grip

    def task(self, t):
        command, rail, grip = self.grasp(t)
        command = torch.lerp(command, self.over, transition(t, 8.10, 9.50))
        carry = torch.where(self.heavy, transition(t, 10.0, 12.5), transition(t, 9.5, 12.0))
        lower = torch.where(self.heavy, transition(t, 12.7, 14.4), transition(t, 12.2, 14.2))
        command = torch.lerp(command, self.bin_high, carry[:, None])
        command = torch.lerp(command, self.lower, lower[:, None])
        grip = grip * (1. - .7 * transition(t, 14.45, 14.85))
        rail = torch.lerp(rail, self.open_rail, transition(t, 14.85, 15.45))
        grip = grip * (1. - transition(t, 14.85, 15.25))
        command = torch.lerp(command, self.withdraw, transition(t, 15.7, 17.15))
        return command, rail, grip

    def action(self, step, obs):
        t = step * self.env.step_dt
        command, rail, grip = self.task(t)
        # A deliberate empty pinch stays above the tabletop and outside the
        # parcel. Stop short of the mechanical jaw stop during this first probe.
        if t < 3.2:
            probe = self.entry.clone()
            probe[:, 2] = .135
            probe = torch.lerp(self.ready, probe, transition(t, .15, 1.0))
            probe[:, 2] += .062 * transition(t, 1.70, 2.38)
            probe_rail = torch.lerp(self.open_rail, self.closed - .008, transition(t, 1.03, 1.60))
            if t >= 2.36 and not self.retry_decided:
                # Top-view delivered visual lift ratio and validity are in the actor observation.
                self.recover = self.miss & ((obs[:, 46] < .35) | (obs[:, 56] < .5))
                self.retry_decided = True
            probe_rail = torch.where(self.recover, torch.lerp(probe_rail, self.open_rail, transition(t, 2.40, 2.72)),
                                     probe_rail)
            entry = self.task(t - 1.4)[0]
            probe = torch.lerp(probe, entry, transition(t, 2.44, 3.18))
            command = torch.where(self.miss[:, None], probe, command)
            rail = torch.where(self.miss, probe_rail, rail)
            grip = torch.where(self.miss, 0., grip)
        else:
            retry_pos, retry_rail, retry_grip = self.task(t - 1.4)
            command = torch.where(self.recover[:, None], retry_pos, command)
            rail = torch.where(self.recover, retry_rail, rail)
            grip = torch.where(self.recover, retry_grip, grip)
        centering = transition(t, 3.38, 3.72) * (1. - transition(t, 4.55, 4.9))
        centering_retry = transition(t - 1.4, 3.38, 3.72) * (1. - transition(t - 1.4, 4.55, 4.9))
        centering_gain = torch.where(self.recover, centering_retry, centering)
        image_shift = (obs[:, 36] - self.env.pick_xy[:, 0]).clamp(-.03, .03)
        command[:, 0] += self.book * (obs[:, 47] > .5) * centering_gain * image_shift

        actual = self.port.read()
        jac = torch.cat((actual.pad_jacobian.mean(dim=1), actual.angular_jacobian.mean(dim=1)), dim=1)[..., :6]
        q = self.env.motor_channels(self.robot.data.joint_pos.torch, self.robot)
        takeoff = 1. - .95 * transition(t, 4.65, 5.75)
        carry = 1. - .97 * transition(t, 7.9, 9.25)
        retry_carry = 1. - .97 * transition(t - 1.4, 7.9, 9.25)
        # The approach requires the calibrated jaw angle. During transport,
        # prioritize wrist position to stay clear of the bin and joint limits.
        self.ik.angular_weight = torch.minimum(torch.where(self.heavy, takeoff, 1.),
                                               torch.where(self.recover, retry_carry, carry))
        self.ik.set_command(torch.cat((command, self.orientation), dim=1))
        target = self.ik.compute(actual.pads.mean(dim=1), actual.wrist_quat_xyzw, jac, q[:, :6])
        rail += self.env.step_limit[:, -1] * grip / (1000. * self.env.rail_overtravel)
        return ((torch.cat((target, rail[:, None]), dim=1) - q) / self.env.step_limit).clamp(-1., 1.)


@torch.inference_mode()
def collect(env, corners, args):
    obs, _ = env.reset()
    teacher = BinIKTeacher(env, corners, args.rescue_fraction)
    duration = args.probe_seconds if args.probe_seconds is not None else (19. if args.rescue_fraction else 17.6)
    n, steps = env.num_envs, round(duration / env.step_dt)
    episode_end = torch.full((n,), steps, device=env.device, dtype=torch.long)
    active = torch.ones_like(episode_end, dtype=torch.bool)
    success = torch.zeros_like(active)
    captured = torch.zeros_like(active)
    airborne = torch.zeros_like(active)
    crossed = torch.zeros_like(active)
    drop = torch.zeros_like(active)
    terms = torch.zeros_like(active)
    failure_reasons = torch.zeros((n, 4), device=env.device, dtype=torch.bool)
    buffers = {}
    if not args.probe:
        widths = dict(policy=env.cfg.observation_space, critic=env.cfg.state_space, action=7)
        buffers = {name: torch.empty((steps, n, width), device=env.device, dtype=torch.float16)
                   for name, width in widths.items()}
    start = perf_counter()
    for i in range(steps):
        action = teacher.action(i, obs["policy"]).masked_fill(~active[:, None], 0.)
        if buffers:
            for name in ("policy", "critic"):
                buffers[name][i].copy_(obs[name])
            buffers["action"][i].copy_(action)
        obs, reward, terminated, timeout, extras = env.step(action)
        finished = active & (terminated | timeout)
        episode_end.masked_fill_(finished, i + 1)
        success |= finished & (reward > 0.)
        terms |= finished & (reward < 0.)
        failure_reasons |= finished[:, None] & extras["terminal_phys"]
        if args.probe and finished.any():
            for lane in finished.nonzero().flatten().tolist():
                force = extras["contact_forces"][lane].tolist()
                print("BIN_TERMINAL " + json.dumps(dict(
                    lane=lane, corner=int(corners[lane]), time_s=(i + 1) * env.step_dt,
                    pad_m=extras["probe_pad"][lane].tolist(),
                    signed_gap_mm=float(extras["probe_signed_gap_mm"][lane]),
                    flags=extras["terminal_phys"][lane].tolist(),
                    contact_N=dict(zip(env.contacts.body_names, force, strict=True)))), flush=True)
        captured |= active & env.ever_captured
        airborne |= active & env.ever_airborne
        crossed |= active & env.entered_bin
        drop |= active & env.valid_release
        active &= ~finished
        if i % 50 == 49 or i + 1 == steps:
            print("BIN_DEMO_PROGRESS " + json.dumps(dict(
                time_s=(i + 1) * env.step_dt, active=int(active.sum()), captured=int(captured.sum()),
                airborne=int(airborne.sum()), crossed=int(crossed.sum()), released=int(drop.sum()),
                success=int(success.sum()), wall_seconds=perf_counter() - start)), flush=True)
            if args.probe:
                sample = env.ports[0].read()
                hull = env.projectors[0].vertices(env.parcel_state.payload)
                print("BIN_PROBE_STAGE " + json.dumps(dict(
                    time_s=(i + 1) * env.step_dt,
                    feed_mm=(1000. * env.result.net_feed[:, 0]).tolist(),
                    gap_mm=(1000. * (sample.pads[:, 1] - sample.pads[:, 0]).norm(dim=1)).tolist(),
                    ee_mm=(1000. * sample.pads.mean(dim=1)).tolist(),
                    box_min_mm=(1000. * hull.amin(dim=1)).tolist(), box_max_mm=(1000. * hull.amax(dim=1)).tolist(),
                    gather_mm=(1000. * env.parcel_state.contacts_A[:, :, 0]).tolist(),
                    clearance_mm=(1000. * env.result.table_clearance[:, 0]).tolist())), flush=True)
            if not active.any():
                break
    by_corner = {}
    for index in args.corners:
        mask = corners == index
        by_corner[index] = {name: int((mask & flag).sum()) for name, flag in (
            ("attempts", torch.ones_like(mask)), ("captured", captured), ("airborne", airborne),
            ("crossed", crossed), ("released", drop), ("success", success), ("terminated", terms))}
        for column, name in enumerate(("rim", "collision", "drop", "tear")):
            by_corner[index][name] = int((mask & failure_reasons[:, column]).sum())
    summary = dict(corners=by_corner, successful=int(success.sum()), recovered=int((teacher.recover & success).sum()),
                   collected=not args.probe and bool(success.any()), seconds=perf_counter() - start)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BIN_DEMO_RESULT " + json.dumps(summary), flush=True)
    if buffers and success.any():
        selection = success.nonzero().flatten()
        data = {name: buf[:i + 1, selection].cpu() for name, buf in buffers.items()}
        data.update(lengths=episode_end[selection].cpu(), corner=corners[selection].cpu(),
                    recovery=teacher.recover[selection].cpu())
        scratch = args.output / "demos.pt.tmp"
        torch.save(data, scratch)
        scratch.replace(args.output / "demos.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--corners", type=int, nargs="+", default=[6, 3, 4, 5])
    parser.add_argument("--rescue-fraction", type=float, default=.17)
    parser.add_argument("--jitter", type=float, default=1.)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if (not args.headless or args.visualizer or args.livestream or args.num_envs < 1
            or not set(args.corners).issubset({3, 4, 5, 6}) or not 0. <= args.rescue_fraction <= 1.):
        parser.error("Requires headless mode, positive environments, corners 3–6 and a valid recovery fraction.")
    if args.probe_seconds is not None and (not args.probe or not .04 <= args.probe_seconds <= 19.):
        parser.error("--probe-seconds only applies to probes and must lie in [0.04, 19].")
    layout = load_bin_layout()
    cfg = yam_bin_cfg(args.num_envs, layout, seed=args.seed)
    cfg.episode_length_s = 20.
    with launch_simulation(cfg.sim, args):
        episode, corners, pick = make_fixtures(layout, args.num_envs, args.corners, args.seed, args.jitter, cfg.sim.device)
        env = (ProbeEnv if args.probe else DemonstrationEnv)(cfg, layout, episode, pick)
        subscription = None
        if args.probe:
            from omni.physics.core import get_physics_simulation_interface
            from pxr import PhysicsSchemaTools

            pairs = set()

            def contact(headers, _data, _friction):
                for head in headers:
                    pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(cell))
                                        for cell in (head.collider0, head.collider1)))
                    if pair not in pairs:
                        pairs.add(pair)
                        print("BIN_LINK_CONTACT " + json.dumps(pair), flush=True)

            subscription = get_physics_simulation_interface().subscribe_physics_contact_report_events(contact)
        try:
            collect(env, corners, args)
        finally:
            if subscription is not None:
                subscription = None
            env.close()


if __name__ == "__main__":
    main()
