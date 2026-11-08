"""L1: run the unmodified official cloth task and measure complete environment steps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import validate
import isaaclab_tasks  # registers official tasks
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.entrypoints.simple_agents import create_zero_action_policy


def process_memory_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return sum(float(line.split(",")[1]) for line in result.stdout.splitlines()
               if int(line.split(",")[0]) == os.getpid())


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["none"], headless=True, device="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("num_envs/steps must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    task = "Isaac-Lift-Cloth-Franka"
    cfg, _ = resolve_task_config(task, "", overrides=["physics=newton_mjwarp_vbd_proxy"])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    validate(cfg)
    torch.manual_seed(args.seed)
    init_start = time.perf_counter()
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg)
        try:
            env.reset()
            policy = create_zero_action_policy(env)
            torch.cuda.synchronize()
            init_s = time.perf_counter() - init_start
            warm_start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(args.warmup):
                    env.step(policy())
                torch.cuda.synchronize()
                warm_s = time.perf_counter() - warm_start
                memory_samples = [process_memory_mib()]
                memory_errors = []
                stop = threading.Event()

                def sample():
                    while not stop.wait(0.5):
                        try:
                            memory_samples.append(process_memory_mib())
                        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                            memory_errors.append(str(exc))
                            return

                monitor = threading.Thread(target=sample, daemon=True)
                monitor.start()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    for _ in range(args.steps):
                        last = env.step(policy())
                    torch.cuda.synchronize()
                    wall_s = time.perf_counter() - start
                finally:
                    stop.set()
                    monitor.join()
                memory_samples.append(process_memory_mib())
            result = {
                "task": task, "preset": "newton_mjwarp_vbd_proxy", "agent": "hold/zero",
                "num_envs": args.num_envs, "seed": args.seed, "warmup_steps": args.warmup,
                "measured_steps": args.steps, "initialization_s": init_s, "warmup_s": warm_s,
                "wall_s": wall_s, "env_steps_per_s": args.num_envs * args.steps / wall_s,
                "batch_steps_per_s": args.steps / wall_s,
                "physics_dt_s": cfg.sim.dt, "decimation": cfg.decimation,
                "num_substeps": cfg.sim.physics.num_substeps,
                "process_memory_peak_sampled_mib": max(memory_samples),
                "process_memory_sampling_interval_s": 0.5,
                "process_memory_sampling_errors": memory_errors,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "final_observation_reward_finite": finite(last[:2]),
                "note": "L1 task execution only; does not measure completed grasps or establish L5/L6",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            if memory_errors or not result["final_observation_reward_finite"]:
                raise RuntimeError("memory sampling or final state validation failed; see output")
        finally:
            env.close()


if __name__ == "__main__":
    main()
