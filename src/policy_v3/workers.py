"""Isolated simulator subprocesses and the version barrier at the learner."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import numpy as np
import torch

from .data import OnPolicyBuffer, ReplayBuffer, read_records
from .geometry import CandidateRules, InfoMode
from .training import ContextualPPOEntry, InformationProfile, SupervisedTrainEntry


@dataclass(frozen=True)
class CollectionConfig:
    scene: str = "rigid"
    info_mode: str = "v1_allowed"
    rules: CandidateRules = CandidateRules()
    epsilon: float = .10
    matched_repeats: int = 1
    camera: bool = False
    shared_world_candidates: bool = False
    factory: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "info_mode", InfoMode(self.info_mode).value)
        if not 0 <= self.epsilon <= 1 or self.matched_repeats < 1:
            raise ValueError("Invalid exploration or matched-repeat configuration")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def episode_statistics(rows):
    valid = sum(row.physics_validity for row in rows)
    positive = sum(row.task_success for row in rows)
    valid_grasps = len(ReplayBuffer.bce_records(rows))
    return dict(episodes=len(rows), physics_valid=valid, positive=positive, valid_negative=valid - positive,
                valid_executed_grasps=valid_grasps, bce_negative=valid_grasps - positive,
                physics_valid_rate=valid / max(1, len(rows)), physics_invalid_rate=1 - valid / max(1, len(rows)),
                failure_reasons=Counter(row.failure_reason for row in rows if row.failure_reason),
                invalid_stages=Counter(row.invalid_stage for row in rows if not row.physics_validity),
                probe_paths_rejected=sum(row.probe_info.get("stop_reason") == "probe_path_rejected" for row in rows),
                nonempty_probe_histories=sum(row.valid_len > 0 for row in rows))


def worker_environment(device, worker=0):
    cache = f"/tmp/warp_cache_gpu_{device}" if device != "cpu" else f"/tmp/warp_cache_gpu_cpu_{worker}"
    return dict(os.environ, CUDA_VISIBLE_DEVICES="" if device == "cpu" else device,
                WARP_CACHE_DIR=cache, WARP_CACHE_PATH=cache, MUJOCO_EGL_DEVICE_ID="0",
                OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")


class SingleMultiGPUWorker:
    def __init__(self, devices=("cpu",)):
        self.devices = tuple(devices)
        gpu = [d for d in self.devices if d != "cpu"]
        if (not self.devices or len(set(gpu)) != len(gpu)
                or any(not d.isdecimal() for d in gpu) or (gpu and len(gpu) != len(self.devices))):
            raise ValueError("Specify all cpu workers or distinct physical GPU numbers")

    def collect(self, checkpoint, token, version, config, episodes, seed, output, *, learned):
        if episodes < len(self.devices) or episodes % config.matched_repeats:
            raise ValueError("Allocate episodes to all workers in complete matched groups")
        groups = episodes // config.matched_repeats
        if groups < len(self.devices):
            raise ValueError("Every worker needs at least one complete group")
        processes = []
        try:
            for i, (device, chunk, sequence) in enumerate(zip(self.devices, np.array_split(
                    np.arange(groups), len(self.devices)), np.random.SeedSequence(seed).spawn(len(self.devices)),
                    strict=True)):
                directory = output / f"worker-{i}"
                directory.mkdir()
                job = dict(worker=i, count=len(chunk) * config.matched_repeats,
                           seed=int(sequence.generate_state(1)[0]), version=version, token=token,
                           checkpoint=str(checkpoint.resolve()), configuration=asdict(config),
                           round_id=output.name, device="cpu" if device == "cpu" else "cuda:0", learned=learned)
                write_json(directory / "job.json", job)
                with (directory / "worker.log").open("w") as log:
                    process = subprocess.Popen([sys.executable, "-B", "-m", "policy_v3.worker_process",
                                                str(directory.resolve())], env=worker_environment(device, i),
                                               stdout=log, stderr=subprocess.STDOUT)
                processes.append((process, directory, job))
            failures = []
            for process, directory, _ in processes:
                if process.wait():
                    failures.append(str(directory / "worker.log"))
            if failures:
                raise RuntimeError(f"Worker failure, see {', '.join(failures)}")
            results = []
            for _, directory, job in processes:
                meta = json.loads((directory / "complete.json").read_text())
                rows = read_records(directory / "episodes.jsonl.gz")
                if (meta["token"] != token or meta["version"] != version or meta["seed"] != job["seed"]
                        or len(rows) != job["count"] or any(r.policy_version != version for r in rows)):
                    raise ValueError("Frozen policy barrier rejected a worker result")
                results.append((rows, directory, meta))
            return results
        finally:
            for process, _, _ in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


class CentralLearner:
    def __init__(self, *, seed=0, devices=("cpu",), profile=InformationProfile()):
        torch.manual_seed(seed)
        self.seed = seed
        self.version = 0
        self.collection_index = 0
        self.workers = SingleMultiGPUWorker(devices)
        self.replay = ReplayBuffer()
        self.supervised = SupervisedTrainEntry(seed=seed, profile=profile)
        self.ppo = ContextualPPOEntry(seed=seed)

    def _worker_state(self, config=None):
        profile = self.supervised.profile
        return dict(version=self.version, encoder=self.supervised.encoder.state_dict(),
                    scorer=self.supervised.scorer.state_dict(), actor=self.ppo.actor.state_dict(),
                    profile=dict(asdict(profile), mode=profile.mode.value if profile.mode is not None else None),
                    configuration=asdict(config) if config is not None else None)

    def _state(self):
        return dict(self._worker_state(), critic=self.ppo.critic.state_dict(),
                    split_assignments=self.replay.assignments, collection_index=self.collection_index,
                    supervised_optimizer=self.supervised.optimizer.state_dict(),
                    actor_optimizer=self.ppo.actor_opt.state_dict(), critic_optimizer=self.ppo.critic_opt.state_dict(),
                    supervised_rng=self.supervised.rng.bit_generator.state,
                    ppo_rng=self.ppo.rng.bit_generator.state)

    def checkpoint(self, path):
        tmp = path.with_suffix(".tmp")
        try:
            torch.save(self._state(), tmp)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def _restore(self, state):
        self.version = state["version"]
        self.collection_index = state.get("collection_index", 0)
        self.replay.assignments = state.get("split_assignments", {})
        for trainer, keys in ((self.supervised, ("encoder", "scorer")), (self.ppo, ("actor", "critic"))):
            for key in keys:
                getattr(trainer, key).load_state_dict(state[key])
        self.supervised.profile = InformationProfile(**state["profile"])
        for key, optimizer in (("supervised_optimizer", self.supervised.optimizer),
                               ("actor_optimizer", self.ppo.actor_opt), ("critic_optimizer", self.ppo.critic_opt)):
            if key in state:
                optimizer.load_state_dict(state[key])
        for key, trainer in (("supervised_rng", self.supervised), ("ppo_rng", self.ppo)):
            if key in state:
                trainer.rng.bit_generator.state = state[key]

    def load_checkpoint(self, path):
        self._restore(torch.load(path, map_location="cpu", weights_only=True))

    def _collect(self, output, config, episodes, *, learned):
        output = Path(output)
        output.mkdir(parents=True)
        version = self.version
        # One immutable snapshot, including candidate rules, serves exactly one entire round.
        with tempfile.TemporaryDirectory(prefix="snapshot-", dir=output) as tmp:
            path = Path(tmp) / "frozen.pt"
            torch.save(self._worker_state(config), path)
            token = fingerprint(path)
            seed = np.random.SeedSequence((self.seed, version, self.collection_index))
            results = self.workers.collect(path, token, version, config, episodes,
                                           int(seed.generate_state(1)[0]), output, learned=learned)
        rows = [row for records, _, _ in results for row in records]
        on_policy = OnPolicyBuffer(version)
        if learned:
            for records, directory, _ in results:
                with np.load(directory / "on_policy.npz") as data:
                    if not np.array_equal(data["episode_id"], [r.episode_id for r in records]):
                        raise ValueError("On-policy samples must align with physics outcomes")
                    for i, record in enumerate(records):
                        on_policy.append(record, *(data[name][i] for name in
                                                    ("v0", "private_s0", "u", "a_try", "old_log_prob")))
        self.replay.extend(rows)
        self.collection_index += 1
        stats = dict(episode_statistics(rows), version=version, frozen_sha256=token,
                     host=socket.gethostname(), workers=[meta for _, _, meta in results])
        write_json(output / "collection.json", stats)
        return stats, on_policy

    def collect_random(self, output, config, episodes):
        stats, _ = self._collect(output, config, episodes, learned=False)
        return stats

    def pretrain(self, output, *, epochs=10):
        previous = deepcopy(self._state())
        try:
            stats = self.supervised.fit(self.replay.splits(self.seed), epochs=epochs)
            self.version += 1
            write_json(Path(output) / "bce.json", stats)
            self.checkpoint(Path(output) / "checkpoint.pt")
        except Exception:
            self._restore(previous)
            raise
        return stats

    def train_round(self, output, config, episodes, *, alternate=False, bce_epochs=1, ppo_epochs=4):
        stats, buffer = self._collect(output, config, episodes, learned=True)
        previous = deepcopy(self._state())
        try:
            result = dict(collection=stats, ppo=self.ppo.update(buffer, epochs=ppo_epochs))
            if alternate:
                result["bce"] = self.supervised.fit(self.replay.splits(self.seed), epochs=bce_epochs)
            if result["ppo"]["examples"] or alternate:
                self.version += 1
            result["next_policy_version"] = self.version
            write_json(Path(output) / "update.json", result)
            self.checkpoint(Path(output) / "checkpoint.pt")
        except Exception:
            self._restore(previous)
            raise
        return result
