"""Measure the single-arm bin scene with calibrated 190 mm hover targets."""

import argparse
import json
from time import perf_counter

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from src.isaac_yam_parcel.yam_bin_env import YAMSingleArmBinEnv, load_bin_layout, yam_bin_cfg
from src.low_order_parcel.lumped_contact_proxy import ContactState


class HoverBenchmarkEnv(YAMSingleArmBinEnv):
    def _track_bin(self, z_ee, rail_speed):
        if hasattr(self, "hover_reference"):
            torch.maximum(self.hover_drift, (z_ee - self.hover_reference).abs(), out=self.hover_drift)
        super()._track_bin(z_ee, rail_speed)


@torch.inference_mode()
def check_bin_gates(env):
    """Exercise the terminal logic using controlled parcel states, separate from hover timing."""
    start = env.parcel_sim.initial_state()
    payload = env.parcel_sim.payload
    corners = env.projectors[0].vertices(start.payload)
    offset_z = corners[..., 2].amin(dim=1)
    like = env.scene.env_origins

    def sample(relative_y, bottom, z_ee, *, opening=False, holding=True, supported=False):
        center = payload.com(start.payload)
        shift = env.bin_center.expand_as(center).clone() - center
        shift[:, 1] += like.new_tensor(relative_y)
        shift[:, 2] = like.new_tensor(bottom) - offset_z
        state = start.payload._replace(pos=start.payload.pos + shift[:, None])
        contacts = start.contacts_A.clone()
        contacts[..., ContactState.CAPTURE] = float(holding)
        env.parcel_state = start._replace(payload=state, contacts_A=contacts)
        env.result = env.result._replace(
            table_clearance=like.new_tensor(bottom).unsqueeze(1),
            support_N=like.new_full((4, 1), float(supported)),
        )
        env._track_bin(like.new_tensor(z_ee), like.new_full((4,), -.02 if opening else 0.))

    sample([-.4] * 4, [.16, .10, .16, .16], [.23] * 4)
    sample([0.] * 4, [.16, .10, .16, .16], [.23] * 4)
    sample([0., 0., 0., .14], [.06, .10, .20, .16], [.21, .21, .29, .23], opening=True)
    sample([0., 0., 0., .14], [0.] * 4, [.21, .21, .29, .23], opening=True, holding=False, supported=True)
    for _ in range(round(env.bin_cfg["settled_duration_s"] / env.cfg.sim.dt)):
        sample([0., 0., 0., .14], [0.] * 4, [.25] * 4, holding=False, supported=True)
    expected = [True, False, False, False]
    assert env._success().tolist() == expected
    assert env.rim_scrape[1].item() and env.dropped_early[2:].all().item()
    return dict(valid_drop=expected[0], low_crossing_rejected=True,
                high_release_rejected=True, protruding_parcel_rejected=True)


@torch.inference_mode()
def check(env, steps, memory_before):
    obs, _ = env.reset()
    robot = env.robots[0]
    assert len(env.scene.articulations) == 1 and robot.cfg.prim_path.endswith("/Yam_B")
    assert set(env.contacts.body_names) == set(robot.body_names)
    assert obs["policy"].shape == (env.num_envs, env.cfg.observation_space)
    assert obs["critic"].shape == (env.num_envs, env.cfg.state_space)
    origin = env.ports[0].read().pads[..., 2].mean(dim=1)
    env.hover_reference = origin
    env.hover_drift = torch.zeros_like(origin)
    initial = env.projectors[0].vertices(env.parcel_state.payload)
    starts_clear = initial[..., 1].amax(dim=1) <= env.randomizer.near_wall_y + 1e-5
    assert starts_clear.all(), "A randomized initial parcel overlaps the near basket wall."
    target = env.motor_channels(env.q_des[0], robot).clone()
    terminals = torch.zeros_like(starts_clear)

    def tick():
        q = env.motor_channels(robot.data.joint_pos.torch, robot)
        obs, _, ended, timed_out, _ = env.step((target - q) / env.step_limit)
        terminals.logical_or_(ended | timed_out)
        return obs

    for _ in range(3):
        tick()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for _ in range(steps):
        obs = tick()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    free, total = torch.cuda.mem_get_info()
    final_z = env.ports[0].read().pads[..., 2].mean(dim=1)
    drift = torch.maximum(env.hover_drift, (final_z - origin).abs())
    assert obs["policy"].isfinite().all() and obs["critic"].isfinite().all()
    return {
        "envs": env.num_envs, "arms": len(env.scene.articulations), "actions": env.cfg.action_space,
        "contacts_monitored_per_arm": len(env.contacts.body_names),
        "initial_pose_clear": int(starts_clear.sum()), "terminal_lanes": int(terminals.sum()),
        "physical_steps": steps * env.cfg.decimation, "elapsed_s": elapsed,
        "env_steps_per_s": env.num_envs * steps * env.cfg.decimation / elapsed,
        "device_used_mib": (total - free) / 2**20, "incremental_device_used_mib": (memory_before - free) / 2**20,
        "torch_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "mean_ready_pad_z_mm": origin.mean().item() * 1000.,
        "hover_peak_drift_mm": drift.max().item() * 1000.,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--benchmark", type=int, default=12)
    parser.add_argument("--opening-m", type=float, nargs=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-gates", action="store_true", help="Run the four-lane terminal-logic check.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream or min(args.num_envs, args.benchmark) < 1:
        parser.error("Supply positive counts and use --headless without visualizers or livestream.")
    if args.check_gates and args.num_envs != 4:
        parser.error("--check-gates uses exactly four environments.")
    layout = load_bin_layout(args.opening_m)
    cfg = yam_bin_cfg(args.num_envs, layout, seed=args.seed)
    with launch_simulation(cfg.sim, args):
        before = torch.cuda.mem_get_info()[0]
        env = HoverBenchmarkEnv(cfg, layout)
        try:
            data = check(env, args.benchmark, before)
            data["pass"] = data["terminal_lanes"] == 0 and data["hover_peak_drift_mm"] < 1.
            if args.check_gates:
                data["terminal_logic"] = check_bin_gates(env)
            print("ROUTE_B_SINGLE_ARM_BIN " + json.dumps(data), flush=True)
            return int(not data["pass"])
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
