"""Episode records, grouped holdouts and separate replay/on-policy consumers."""

from dataclasses import dataclass
import gzip
import hashlib
import json

import numpy as np

from .geometry import InfoMode, PublicContext, VisualObservation, vector
from .history import HISTORY_DIM, RAW_DIM, encoder_frames


@dataclass
class EpisodeRecord:
    episode_id: str
    initial_state_id: str
    probe_state_id: str
    v0: VisualObservation | None
    v1: VisualObservation | None
    a_try: np.ndarray
    H_try: np.ndarray
    valid_len: int
    grasp_parameters_world: np.ndarray | None
    info_mode: str
    task_success: bool
    physics_validity: bool
    invalid_stage: str | None
    failure_reason: str | None
    randomization_params: dict
    policy_version: int
    timestamps: np.ndarray
    trajectory_world: np.ndarray
    probe_stages: list[str]
    probe_info: dict
    task_summary: dict
    H_try_raw: np.ndarray | None = None

    def __post_init__(self):
        self.info_mode = InfoMode(self.info_mode).value
        if not all((self.episode_id, self.initial_state_id, self.probe_state_id)):
            raise ValueError("Episode and state identities must be populated")
        self.a_try = vector(self.a_try, 13)
        if np.any(np.abs(self.a_try) > 1):
            raise ValueError("Try outside [-1, 1]")
        if self.grasp_parameters_world is not None:
            self.grasp_parameters_world = vector(self.grasp_parameters_world, 11)
        if self.H_try_raw is None:
            # Early V3 archives placed the full raw layout under H_try.
            self.H_try_raw = np.asarray(self.H_try, dtype=float).reshape(-1, RAW_DIM)
            self.H_try = encoder_frames(self.H_try_raw)
        else:
            self.H_try_raw = np.asarray(self.H_try_raw, dtype=float).reshape(-1, RAW_DIM)
            self.H_try = np.asarray(self.H_try, dtype=np.float32).reshape(-1, HISTORY_DIM)
            if not np.array_equal(self.H_try, encoder_frames(self.H_try_raw)):
                raise ValueError("Policy history differs from the recorded raw frames")
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.trajectory_world = np.asarray(self.trajectory_world, dtype=float).reshape(-1, 3)
        arrays = (self.H_try, self.H_try_raw, self.timestamps, self.trajectory_world)
        if any(len(a) != self.valid_len or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid raw history, timestamps or trajectory")
        if (self.timestamps.shape != (self.valid_len,) or np.any(np.diff(self.timestamps) <= 0)
                or len(self.probe_stages) != self.valid_len):
            raise ValueError("Invalid probe frame alignment")
        if ((self.physics_validity and self.invalid_stage is not None)
                or (not self.physics_validity and self.invalid_stage not in ("reset", "probe", "grasp"))):
            raise ValueError("Physics validity and invalid stage disagree")
        if self.v0 is None and (self.invalid_stage != "reset" or self.valid_len or self.grasp_parameters_world is not None):
            raise ValueError("Missing v0 is allowed only when reset fails before motion")
        if self.task_success and (not self.physics_validity or self.grasp_parameters_world is None):
            raise ValueError("Success requires a valid executed grasp")

    def public_context(self, mode=None):
        mode = InfoMode(self.info_mode if mode is None else mode)
        if mode is InfoMode.V1_ALLOWED and InfoMode(self.info_mode) is not InfoMode.V1_ALLOWED:
            raise ValueError("Cannot elevate a record collected without v1")
        if self.v0 is None:
            raise ValueError("A failed reset has no public observation")
        return PublicContext(mode, self.v0, lambda: self.v1, self.a_try)

    def history(self):
        return self.H_try

    def to_dict(self):
        row = vars(self).copy()
        row.update(v0=None if self.v0 is None else self.v0.to_dict(),
                   v1=None if self.v1 is None else self.v1.to_dict())
        for name in ("a_try", "H_try", "H_try_raw", "timestamps", "trajectory_world", "grasp_parameters_world"):
            if row[name] is not None:
                row[name] = row[name].tolist()
        return row

    @classmethod
    def from_dict(cls, row):
        row = row.copy()
        for name in ("v0", "v1"):
            if row[name] is not None:
                row[name] = VisualObservation(**row[name])
        return cls(**row)


def write_records(path, records):
    with gzip.open(path, "xt", compresslevel=3, encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row.to_dict(), allow_nan=False, separators=(",", ":")) + "\n")


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [EpisodeRecord.from_dict(json.loads(line)) for line in stream]


def assert_disjoint(splits):
    names = ("episode_id", "initial_state_id", "probe_state_id")
    seen = {name: {} for name in names}
    for split, rows in splits.items():
        for row in rows:
            for name in names:
                identity = getattr(row, name)
                previous = seen[name].setdefault(identity, split)
                if previous != split:
                    raise AssertionError(f"{name} {identity} crosses {previous} and {split}")


class GroupSplit:
    """A connected component joins every row linked by either state identity."""

    @staticmethod
    def components(rows):
        parents = list(range(len(rows)))
        owners = {}

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for i, row in enumerate(rows):
            for name in ("initial_state_id", "probe_state_id"):
                key = (name, getattr(row, name))
                prior = owners.setdefault(key, i)
                parents[root(i)] = root(prior)
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(root(i), []).append(row)
        return list(groups.values())

    @staticmethod
    def partition(rows, seed=0, fractions=(.70, .15, .15)):
        if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1.):
            raise ValueError("Expected three positive split fractions summing to one")
        groups = GroupSplit.components(rows)
        ids = list(range(len(groups)))
        np.random.default_rng(seed).shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError("GroupSplit needs at least three independent state groups")
        n_train = max(1, min(n - 2, round(n * fractions[0])))
        n_val = max(1, min(n - n_train - 1, round(n * fractions[1])))
        ranges = (ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:])
        splits = {split: [r for group in chunk for r in groups[group]]
                  for split, chunk in zip(("train", "val", "test"), ranges, strict=True)}
        assert_disjoint(splits)
        return splits


class ReplayBuffer:
    def __init__(self, rows=()):
        self.records = []
        self.episode_ids = set()
        self.assignments = {}
        self.extend(rows)

    def extend(self, rows):
        rows = list(rows)
        ids = {row.episode_id for row in rows}
        if len(ids) != len(rows) or ids & self.episode_ids:
            raise ValueError("Duplicate replay episode ID")
        self.episode_ids.update(ids)
        self.records.extend(rows)

    @staticmethod
    def _group_keys(rows):
        return {f"{kind}:{getattr(row, kind)}" for row in rows
                for kind in ("initial_state_id", "probe_state_id")}

    def splits(self, seed=0):
        # Pin holdouts across incremental C4 training and fresh learner processes.
        if not self.assignments:
            result = GroupSplit.partition(self.records, seed)
        else:
            result = {split: [] for split in ("train", "val", "test")}
            for rows in GroupSplit.components(self.records):
                keys = self._group_keys(rows)
                previous = {self.assignments[key] for key in keys if key in self.assignments}
                if len(previous) > 1:
                    raise ValueError("New replay records join states assigned to different pinned splits")
                if previous:
                    split = previous.pop()
                else:
                    digest = hashlib.sha256(f"{seed}:{min(keys)}".encode()).digest()
                    draw = int.from_bytes(digest[:8], "big") / 2**64
                    split = "train" if draw < .70 else "val" if draw < .85 else "test"
                result[split].extend(rows)
        assert_disjoint(result)
        for split, rows in result.items():
            self.assignments.update(dict.fromkeys(self._group_keys(rows), split))
        return result

    @staticmethod
    def bce_records(rows):
        return [r for r in rows if r.physics_validity and r.grasp_parameters_world is not None]


@dataclass(frozen=True)
class PPOTransition:
    episode_id: str
    policy_version: int
    v0: np.ndarray
    private_s0: np.ndarray
    u: np.ndarray
    a_try: np.ndarray
    old_log_prob: float
    reward: float

    def __post_init__(self):
        if (not np.isfinite(self.old_log_prob) or not np.isfinite(self.reward)
                or not np.allclose(np.tanh(vector(self.u, 13)), vector(self.a_try, 13), atol=5e-7)):
            raise ValueError("Invalid on-policy sample")


@dataclass(frozen=True)
class RewardCosts:
    time_per_second: float = .015
    motion_per_meter: float = .08
    unsafe: float = 1.

    def reward(self, row):
        s = row.task_summary
        return (float(row.task_success) - self.time_per_second * s["probe_ticks"] / 25
                - self.motion_per_meter * s["probe_motion_m"] - self.unsafe * float(s["unsafe"]))


class OnPolicyBuffer:
    def __init__(self, policy_version, costs=RewardCosts()):
        self.policy_version = policy_version
        self.costs = costs
        self.transitions = []
        self.episode_ids = set()
        self.invalid_count = 0
        self.excluded_count = 0

    def append(self, row, v0, private_s0, u, a_try, old_log_prob):
        if row.policy_version != self.policy_version:
            raise ValueError("Mixed policy versions in on-policy batch")
        if row.episode_id in self.episode_ids:
            raise ValueError("Duplicate on-policy episode ID")
        self.episode_ids.add(row.episode_id)
        if not row.physics_validity:
            self.invalid_count += 1
            if row.invalid_stage != "probe" or not row.probe_info["unsafe"]:
                self.excluded_count += 1
                return
        if not np.array_equal(row.a_try, a_try):
            raise ValueError("Executed action differs from sampled action")
        self.transitions.append(PPOTransition(row.episode_id, row.policy_version, v0, private_s0,
                                              u, a_try, old_log_prob, self.costs.reward(row)))

    @property
    def physics_invalid_rate(self):
        return self.invalid_count / max(1, len(self.episode_ids))
