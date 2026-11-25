"""Raw 25 Hz frames, lossless feature layout and optional sensor augmentation."""

from dataclasses import dataclass

import numpy as np
import torch

# Stored raw layout retains the explicit joint tracking error, even though the
# encoder can recover it from channels 0:6 and 12:18.
RAW_DIM = 54
HISTORY_DIM = 48
RAW_FIELDS = {
    "joint_pos": (0, 6), "joint_vel": (6, 12), "joint_pos_cmd": (12, 18),
    "joint_tracking_error": (18, 24), "joint_eff": (24, 30), "gripper_pos": (30, 31),
    "gripper_vel": (31, 32), "gripper_eff": (32, 33), "gripper_tracking_error": (33, 34),
    "ee_rel_pos": (34, 37), "ee_tracking_error": (37, 40), "ee_vel": (40, 43),
    "ee_rel_rot_delta": (43, 46), "actual_ctrl_cmd": (46, 53), "time_progress": (53, 54),
}
EFFORT = np.r_[18:24, 26]
FEEDBACK = np.r_[0:12, 18:40]


def encoder_frames(raw):
    raw = np.asarray(raw, dtype=np.float32).reshape(-1, RAW_DIM)
    return np.concatenate((raw[:, :18], raw[:, 24:]), axis=1)


@dataclass(frozen=True)
class HistoryPreprocess:
    noise_std: float = 0.
    bias_std: float = 0.
    delay_ticks: int = 0
    effort_dropout: float = 0.

    def __post_init__(self):
        scales = np.asarray([self.noise_std, self.bias_std])
        if (not np.isfinite(scales).all() or min(scales) < 0 or self.delay_ticks < 0
                or not 0 <= self.effort_dropout <= 1):
            raise ValueError("Invalid sensor augmentation parameters")

    def __call__(self, history, rng):
        history = history.copy()
        if not len(history):
            return history
        if self.delay_ticks:
            idx = np.maximum(0, np.arange(len(history)) - self.delay_ticks)
            history[:, FEEDBACK] = history[np.ix_(idx, FEEDBACK)]
        if self.noise_std:
            history[:, FEEDBACK] += rng.normal(0, self.noise_std, (len(history), len(FEEDBACK)))
        if self.bias_std:
            history[:, FEEDBACK] += rng.normal(0, self.bias_std, len(FEEDBACK))
        if self.effort_dropout:
            history[:, EFFORT] *= rng.random(len(EFFORT)) >= self.effort_dropout
        return history


def pad_histories(histories, device="cpu"):
    """True means padding. An empty Try has one fully masked slot."""
    width = max(1, max(map(len, histories)))
    frames = np.zeros((len(histories), width, HISTORY_DIM), dtype=np.float32)
    mask = np.ones((len(histories), width), dtype=bool)
    for i, history in enumerate(histories):
        frames[i, :len(history)] = history
        mask[i, :len(history)] = False
    return torch.as_tensor(frames, device=device), torch.as_tensor(mask, device=device)
