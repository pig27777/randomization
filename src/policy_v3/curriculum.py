"""Policy V3 C0–C4 command-line entry point."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

from .data import read_records
from .environment import MacroBagGraspEnv
from .geometry import CandidateRules, InfoMode
from .training import InformationProfile
from .workers import CentralLearner, CollectionConfig, write_json


def scripted(output, args):
    from .yam_backend import YamBackend

    device = args.devices[0]
    if len(args.devices) != 1:
        raise ValueError("C0 takes exactly one device")
    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        os.environ.setdefault("WARP_CACHE_DIR", f"/tmp/warp_cache_gpu_{device}")
        os.environ["WARP_CACHE_PATH"] = os.environ["WARP_CACHE_DIR"]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        device = "cuda:0"
    b = YamBackend(args.seed, scene=args.scene, device=device, scripted=True, camera=args.camera)
    try:
        rng = np.random.default_rng(args.seed)
        env = MacroBagGraspEnv(b, rng, info_mode=args.info_mode, rules=CandidateRules(args.candidates))
        v0, _, _ = env.reset()
        outcome = dict(host=socket.gethostname(), v0=v0.to_dict(), driver=b.drive.name, neutral_hover=b.at_hover())
        if args.probe_only:
            action = np.zeros(13)
            action[8] = -.45
            action[9] = .30
            action[-1] = -.50
            if b.scene.is_flex:
                # Use a shallow software check clear of the near basket wall.
                fixture_xy = b.common_sampling_region[0][:2] + [0., -.015]
                local_xy = (fixture_xy - v0.center[:2]) @ v0.rotation[:2, :2]
                action[:2] = np.clip(local_xy / v0.half_extent[:2], -1., 1.)
                action[8] = -.04
                action[9] = .12
                action[-1] = .30
            history, length, v1, info = env.execute_probe(action)
            outcome.update(probe=info, a_try=action.tolist(), valid_len=length,
                           v1=v1.to_dict() if v1 is not None else None)
            np.savez_compressed(output / "probe.npz", H_try=history, H_try_raw=env.raw_history,
                                timestamps=env.trace.timestamps, trajectory_world=env.trace.tcp_world,
                                phases=env.trace.stages, valid_len=length)
        else:
            outcome.update(env.execute_grasp(b.scripted_grasp()))
        if hasattr(b.drive, "sim"):
            from yam_platform.yam_model import save_yam_xml
            save_yam_xml(b.drive.sim.xml, output / "scene.xml")
            outcome["peak_robot_contact_N"] = float(b.drive.peak_contact.numpy()[0])
        write_json(output / "scripted.json", outcome)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return outcome
    finally:
        b.close()


def load_replay(learner, paths):
    for path in paths:
        paths = sorted(path.rglob("episodes.jsonl.gz")) if path.is_dir() else [path]
        if not paths:
            raise ValueError("No C1 worker datasets found")
        for file in paths:
            learner.replay.extend(read_records(file))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("c0", "c1", "c2", "c3", "c4", "smoke"))
    p.add_argument("output", type=Path)
    p.add_argument("--scene", choices=("rigid", "flex"), default="rigid")
    p.add_argument("--devices", type=lambda x: tuple(x.split(",")), default=("cpu",))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--ppo-episodes", type=int, default=16)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--matched-repeats", type=int, default=1)
    p.add_argument("--bce-epochs", type=int, default=12)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=.10)
    p.add_argument("--info-mode", choices=[m.value for m in InfoMode], default="v1_allowed")
    p.add_argument("--history", choices=("all", "last", "none"), default="all")
    p.add_argument("--without-effort", action="store_true")
    p.add_argument("--camera", action="store_true")
    p.add_argument("--shared-world-candidates", action="store_true",
                   help="Freeze candidate input to the shared workcell for matched modality training")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--factory", help="Optional module:callable providing the Yam backend contract")
    p.add_argument("--records", nargs="+", type=Path, default=[])
    p.add_argument("--checkpoint", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.rounds < 1 or args.seed < 0:
        raise ValueError("Use a positive round count and nonnegative seed")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    if args.stage == "c0":
        result = scripted(args.output, args)
        return 0 if result.get("task_success") or (args.probe_only and result["probe"]["retracted"]
                                                   and result["valid_len"] > 0) else 1

    config = CollectionConfig(scene=args.scene, info_mode=args.info_mode, rules=CandidateRules(args.candidates),
                              epsilon=args.epsilon, matched_repeats=args.matched_repeats, camera=args.camera,
                              shared_world_candidates=args.shared_world_candidates, factory=args.factory)
    write_json(args.output / "configuration.json", asdict(config))
    learner = CentralLearner(seed=args.seed, devices=args.devices,
                             profile=InformationProfile(args.info_mode, args.history, not args.without_effort))
    load_replay(learner, args.records)
    if args.checkpoint:
        learner.load_checkpoint(args.checkpoint)
    if args.stage == "c1":
        result = learner.collect_random(args.output / "c1", config, args.episodes)
    elif args.stage == "c2":
        result = learner.pretrain(args.output, epochs=args.bce_epochs)
    elif args.stage in ("c3", "c4"):
        if args.checkpoint is None:
            raise ValueError("C3/C4 require a frozen C2 or later checkpoint")
        for i in range(args.rounds):
            result = learner.train_round(args.output / f"round-{i}", config, args.episodes,
                                         alternate=args.stage == "c4", bce_epochs=args.bce_epochs,
                                         ppo_epochs=args.ppo_epochs)
    else:
        if len(args.devices) < 2 or args.episodes < 100:
            raise ValueError("Smoke requires at least 100 episodes and two worker processes")
        c0 = args.output / "c0"
        c0.mkdir()
        if len(set(args.devices)) != 1 or args.devices[0] != "cpu":
            raise ValueError("The concurrent rigid smoke uses simulated GPU workers on CPU")
        args.devices = ("cpu",)
        c0_result = scripted(c0, args)
        stats = learner.collect_random(args.output / "c1", config, args.episodes)
        c2 = args.output / "c2"
        c2.mkdir()
        bce = learner.pretrain(c2, epochs=args.bce_epochs)
        from dataclasses import replace
        ppo = learner.train_round(args.output / "c3", replace(config, matched_repeats=1),
                                  args.ppo_episodes, ppo_epochs=args.ppo_epochs)
        result = dict(c0=c0_result, c1=stats, c2=bce, c3=ppo)
        write_json(args.output / "smoke.json", result)
        if (not c0_result["task_success"] or not stats["positive"] or not stats["bce_negative"]
                or not bce["train"]["bce"] < bce["initial_train"]["bce"]
                or not ppo["ppo"]["examples"] or "actor_loss" not in ppo["ppo"]):
            return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
