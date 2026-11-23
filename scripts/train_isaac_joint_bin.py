"""Warm-start the 7-D Yam_B policy and evaluate it before a first PPO run."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from generate_isaac_joint_demos import DemonstrationEnv, make_fixtures
from src.isaac_yam_parcel.joint_bc_ppo import ActorHistory, JointPolicy, behavior_clone, ppo_update, rollout
from src.isaac_yam_parcel.yam_bin_env import load_bin_layout, yam_bin_cfg


@torch.inference_mode()
def evaluate(env, model, corners):
    obs, _ = env.reset()
    history = ActorHistory(obs["policy"])
    completed = torch.zeros(len(corners), device=env.device, dtype=torch.bool)
    success = torch.zeros_like(completed)
    unsafe = torch.zeros_like(completed)
    for _ in range(env.max_episode_length):
        action = model.actor(history.tensor).masked_fill(completed[:, None], 0.)
        obs, reward, terminal, timeout, _ = env.step(action)
        ended = terminal | timeout
        success |= ~completed & terminal & (reward > 0.)
        unsafe |= ~completed & terminal & (reward < 0.)
        completed |= ended
        history.advance(obs["policy"], ended)
    return {int(corner): dict(attempted=int((corners == corner).sum()), success=int((success & (corners == corner)).sum()),
                              unsafe=int((unsafe & (corners == corner)).sum()))
            for corner in torch.unique(corners).tolist()}


def save_policy(model, destination):
    target = destination.with_suffix(".tmp")
    torch.save(model.state_dict(), target)
    target.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--phase", choices=("bc", "ppo"), required=True)
    parser.add_argument("--demos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corners", type=int, nargs="+", default=[6, 3, 4])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-envs", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--min-demos", type=int, default=4000)
    parser.add_argument("--bc-updates", type=int, default=1500)
    parser.add_argument("--ppo-updates", type=int, default=11)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--min-bc-success", type=float, default=.5)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if (not args.headless or args.visualizer or args.livestream
            or min(args.eval_envs, args.num_envs, args.bc_updates, args.rollout_steps, args.min_demos) < 1
            or not 0. <= args.min_bc_success <= 1.):
        parser.error("Use headless mode, positive batch sizes and a valid BC evaluation threshold.")
    if args.phase == "ppo" and args.ppo_updates < 11:
        parser.error("First PPO uses ten critic-only rounds followed by a joint round.")

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / "result.json"
    if args.phase == "bc":
        if args.demos is None:
            parser.error("BC requires --demos.")
        demonstration = torch.load(args.demos, map_location="cpu", weights_only=True)
        lane = torch.isin(demonstration["corner"], torch.tensor(args.corners))
        count = int(lane.sum())
        if count < args.min_demos or not set(args.corners) <= set(demonstration["corner"].tolist()):
            parser.error(f"Only {count} valid selected demonstrations or missing corner coverage.")
    else:
        result = json.loads(result_file.read_text())
        if not result["bc_passed"] or result["seed"] != args.seed or result["corners"] != args.corners:
            parser.error("PPO requires a passing BC evaluation for this seed and corner set.")

    layout = load_bin_layout()
    pose_seed = args.seed + 1000 if args.phase == "bc" else args.seed
    count_envs = args.eval_envs if args.phase == "bc" else args.num_envs
    cfg = yam_bin_cfg(count_envs, layout, seed=pose_seed)
    cfg.episode_length_s = 20.
    with launch_simulation(cfg.sim, args):
        device = cfg.sim.device
        model = JointPolicy(cfg.observation_space, cfg.state_space).to(device)
        episode, indices, pick = make_fixtures(layout, count_envs, args.corners, pose_seed, .7, device)
        env = DemonstrationEnv(cfg, layout, episode, pick)
        try:
            if args.phase == "bc":
                data = {name: value[:, lane].to(device) if name in ("policy", "critic", "action")
                        else value[lane].to(device) for name, value in demonstration.items()}
                loss = behavior_clone(model, data, updates=args.bc_updates, minibatch=2048)
                del data, demonstration
                save_policy(model, args.output / "bc.pt")
                scores = evaluate(env, model, indices)
                passed = all(row["attempted"] and row["success"] / row["attempted"] >= args.min_bc_success
                             for row in scores.values())
                result = dict(seed=args.seed, corners=args.corners, demos=count, bc_loss=loss,
                              bc_eval=scores, bc_passed=passed, ppo=[])
                result_file.write_text(json.dumps(result, indent=2) + "\n")
                print("BIN_BC " + json.dumps(result), flush=True)
                return int(not passed)

            state = torch.load(args.output / "bc.pt", weights_only=True, map_location=device)
            model.load_state_dict(state)
            optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
            obs, _ = env.reset()
            history = ActorHistory(obs["policy"])
            for i in range(args.ppo_updates):
                started = perf_counter()
                trajectory, obs, history, good, episodes = rollout(env, model, obs, history, args.rollout_steps)
                loss = ppo_update(model, optimizer, trajectory, minibatch=8192, epochs=2, critic_only=i < 10)
                del trajectory
                row = dict(update=i + 1, critic_only=i < 10, successful=int(good), episodes=int(episodes),
                           loss=loss, seconds=perf_counter() - started)
                result["ppo"].append(row)
                result_file.write_text(json.dumps(result, indent=2) + "\n")
                print("BIN_PPO " + json.dumps(row), flush=True)
                save_policy(model, args.output / "ppo.pt")
            actor = torch.jit.script(model.actor.cpu().eval())
            actor.save(str(args.output / "yam_b_joint_actor.ts"))
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
