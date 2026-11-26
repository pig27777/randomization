"""Policy-visible geometry and continuous world/feature grasp coordinates."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class VisualUnavailable(ValueError):
    """The designated outer object has insufficient visible depth."""


class InfoMode(str, Enum):
    WORLD_COMMON = "world_common"
    V0_ONLY = "v0_only"
    V1_ALLOWED = "v1_allowed"


def vector(value, size):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values")
    return result


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Degenerate rotation axis")
    return vector / norm


def exp_rotation(rv):
    skew = np.cross(np.eye(3), rv)
    theta = np.linalg.norm(rv)
    return np.eye(3) + np.sinc(theta / np.pi) * skew + .5 * np.sinc(theta / (2 * np.pi))**2 * (skew @ skew)


def rotation_6d(rotation):
    return np.asarray(rotation)[:, :2].T.reshape(6)


def rotation_matrix(encoded):
    first, second = vector(encoded, 6).reshape(2, 3)
    x = unit(first)
    y = unit(second - np.dot(x, second) * x)
    return np.column_stack((x, y, np.cross(x, y)))


@dataclass(frozen=True)
class VisualObservation:
    center: np.ndarray
    rotation: np.ndarray
    half_extent: np.ndarray
    source: str

    def __post_init__(self):
        center = vector(self.center, 3)
        extent = vector(self.half_extent, 3)
        rotation = np.array(self.rotation, dtype=float, copy=True)
        if (rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.) or np.any(extent <= 0)):
            raise ValueError("Invalid outer geometry frame")
        if self.source not in ("camera_depth_mask", "fixture_geometry"):
            raise ValueError("Unknown visual observation provenance")
        for name, value in (("center", center), ("rotation", rotation), ("half_extent", extent)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def features(self):
        return np.r_[self.center, rotation_6d(self.rotation), self.half_extent].astype(np.float32)

    def to_dict(self):
        return dict(center=self.center.tolist(), rotation=self.rotation.tolist(),
                    half_extent=self.half_extent.tolist(), source=self.source)


class PublicContext:
    """Validate the mode before invoking any accessor for the optional v1."""

    def __init__(self, mode, v0, read_v1, a_try):
        self.mode = InfoMode(mode)
        self.v0 = v0
        self.v1 = read_v1() if self.mode is InfoMode.V1_ALLOWED else None
        self.a_try = vector(a_try, 13)

    @property
    def frame(self):
        if self.mode is InfoMode.WORLD_COMMON:
            return None
        if self.v1 is not None:
            return self.v1
        return self.v0

    def features(self):
        post = self.v1.features() if self.v1 is not None else np.zeros(12, dtype=np.float32)
        return np.r_[self.v0.features(), post, self.a_try].astype(np.float32)

    def grasp_features(self, grasps):
        features = np.asarray(grasps, dtype=np.float32).copy()
        frame = self.frame
        if frame is not None:
            features[..., :3] = (features[..., :3] - frame.center) @ frame.rotation
            features[..., 3:9] = (features[..., 3:9].reshape(-1, 2, 3) @ frame.rotation).reshape(
                features[..., 3:9].shape)
        return features


@dataclass(frozen=True)
class CandidateRules:
    count: int = 256
    attempts_per_candidate: int = 10

    def __post_init__(self):
        if self.count < 1 or self.attempts_per_candidate < 1:
            raise ValueError("Candidate budgets must be positive")


class SE3GraspCandidateSampler:
    """Draw a continuum of approaches and reject them via the workcell planner."""

    def __init__(self, backend, rng, rules=CandidateRules()):
        self.backend = backend
        self.rng = rng
        self.rules = rules

    def sample(self, context, K=None):
        count = self.rules.count if K is None else K
        if count < 1:
            raise ValueError("K must be positive")
        center, spread, orientation = self.backend.common_sampling_region
        if (frame := context.frame) is not None:
            center, spread, orientation = frame.center, frame.half_extent, frame.rotation
        rng = self.rng
        candidates = []
        gap_low, gap_high = self.backend.grasp_width_range
        effort_low, effort_high = self.backend.grasp_effort_range
        for _ in range(count * self.rules.attempts_per_candidate):
            # A smooth distribution concentrated above the table with full rotational support.
            downward = self.backend.hover_rotation[:, 2]
            if rng.random() < .15:
                direction = unit(rng.normal(size=3))
                roll = rng.uniform(-np.pi, np.pi)
            else:
                direction = unit(downward + rng.normal(0, .32, 3))
                roll = rng.normal(0, .48)
            base_x = self.backend.hover_rotation[:, 0]
            if abs(np.dot(base_x, direction)) > .999:
                base_x = self.backend.hover_rotation[:, 1]
            x = unit(base_x - np.dot(base_x, direction) * direction)
            rot = exp_rotation(direction * roll) @ np.column_stack((x, np.cross(direction, x), direction))
            pos = center + orientation @ (spread * rng.uniform(-.72, .72, 3))
            grasp = np.r_[pos, rotation_6d(rot), rng.uniform(gap_low, gap_high),
                          rng.uniform(effort_low, effort_high)]
            if self.backend.grasp_plan(grasp) is not None:
                candidates.append(grasp)
                if len(candidates) == count:
                    break
        return np.asarray(candidates, dtype=np.float64).reshape(-1, 11)
