"""Worker entry point. Device visibility and thread limits are set by the parent."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import torch

from .data import EpisodeRecord, write_records
from .environment import MacroBagGraspEnv
from .geometry import CandidateRules, InfoMode
from .models import GraspScorer, ProbeActor, ProprioHistoryEncoder
from .training import InformationProfile, score_candidates
from .workers import CollectionConfig, episode_statistics, fingerprint, write_json


def build_backend(config, seed, device):
    if config.factory:
        module, name = config.factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    else:
        from .yam_backend import YamBackend
        factory = YamBackend
    return factory(seed, scene=config.scene, device=device, camera=config.camera)


def state_id(*parts):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def run(directory):
    job = json.loads((directory / "job.json").read_text())
    version, seed, learned = (job[key] for key in ("version", "seed", "learned"))
    checkpoint = job["checkpoint"]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if fingerprint(checkpoint) != job["token"]:
        raise ValueError("Policy snapshot checksum mismatch")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if frozen["version"] != version or frozen["configuration"] != job["configuration"]:
        raise ValueError("Snapshot version or candidate rules mismatch")
    config = CollectionConfig(**{**job["configuration"], "rules": CandidateRules(**job["configuration"]["rules"])})
    actor, encoder, scorer = ProbeActor(), ProprioHistoryEncoder(), GraspScorer()
    for key, model in (("actor", actor), ("encoder", encoder), ("scorer", scorer)):
        model.load_state_dict(frozen[key])
        model.eval()
        model.requires_grad_(False)
    profile = InformationProfile(**frozen["profile"])
    rng = np.random.default_rng(seed)
    backend = build_backend(config, seed % (2**29), job["device"])
    env = MacroBagGraspEnv(backend, rng, info_mode=config.info_mode, rules=config.rules)
    training_profile = profile.mode
    if training_profile is not None and training_profile != env.info_mode:
        raise ValueError("Collection mode must match the frozen scorer's training mode")
    rows = []
    ppo = []
    nonce = uuid4().hex
    try:
        for i in range(job["count"]):
            if i % config.matched_repeats == 0:
                reset_seed = int(rng.integers(2**32))
                action_seed = int(rng.integers(2**32))
            # Deterministic independent reinstantiations support matched evaluation while
            # each Try still executes exactly one grasp (B=1).
            v0, private, info = env.reset(seed=reset_seed)
            v0_features = v0.features()
            if learned:
                generator = torch.Generator().manual_seed(action_seed)
                u, a, old = actor.sample(torch.from_numpy(v0_features), generator)
                action = a.numpy()
                ppo.append((v0_features, private, u.numpy(), action, float(old)))
            else:
                action = np.random.default_rng(action_seed).uniform(-1, 1, 13)
            h, length, v1, probe_info = env.execute_probe(action)
            initial = state_id(info.get("initial_state", private), info["randomization_params"])
            post_state = backend.state_identity() if env.phase == "ready" else None
            probe = state_id(initial, action) if post_state is None else state_id(post_state)
            candidate_mode = InfoMode.WORLD_COMMON if config.shared_world_candidates else env.info_mode
            probe_info.update(candidate_count=0, candidate_info_mode=candidate_mode.value)
            grasp = None
            if env.phase == "ready":
                candidates = env.generate_grasp_candidates(candidate_mode)
                probe_info["candidate_count"] = len(candidates)
                if len(candidates):
                    if not learned or rng.random() < config.epsilon:
                        idx = rng.integers(len(candidates))
                    else:
                        idx = score_candidates(encoder, scorer, env.public_context(), h, candidates, profile).argmax()
                    grasp = candidates[idx]
                env.execute_grasp(grasp)
            summary = env.get_task_result()
            rows.append(EpisodeRecord(f"{nonce}-{i}", initial, probe, v0, v1, action, h, length,
                                      grasp, config.info_mode, summary["task_success"], summary["physics_validity"],
                                      summary["invalid_stage"], summary["failure_reason"], info["randomization_params"],
                                      version, env.trace.timestamps, env.trace.tcp_world, env.trace.stages,
                                      probe_info, summary, H_try_raw=env.raw_history))
            if (i + 1) % 25 == 0:
                print(f"worker {job['worker']} episodes {i + 1}/{job['count']}", flush=True)
    finally:
        backend.close()
    write_records(directory / "episodes.jsonl.gz", rows)
    if learned:
        names = ("v0", "private_s0", "u", "a_try", "old_log_prob")
        np.savez_compressed(directory / "on_policy.npz", episode_id=[r.episode_id for r in rows],
                            **{name: np.stack(values) for name, values in zip(names, zip(*ppo), strict=True)})
    meta = dict(episode_statistics(rows), token=job["token"], version=version, seed=seed, pid=os.getpid(),
                devices=os.environ["CUDA_VISIBLE_DEVICES"], cache=os.environ["WARP_CACHE_DIR"],
                omp=os.environ["OMP_NUM_THREADS"], openblas=os.environ["OPENBLAS_NUM_THREADS"])
    write_json(directory / "complete.json", meta)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
