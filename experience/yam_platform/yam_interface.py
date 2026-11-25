"""Common deployable Yam_B observation and command contract."""

from typing import Protocol

import numpy as np

from .yam_model import load_config


PROPRIO_FIELDS = {
    "joint_pos": slice(0, 6),
    "joint_vel": slice(6, 12),
    "joint_eff": slice(12, 18),
    "joint_pos_cmd": slice(18, 24),
    "gripper_pos": slice(24, 25),
    "gripper_vel": slice(25, 26),
    "gripper_eff": slice(26, 27),
    "gripper_cmd": slice(27, 28),
}


def observation_fields(packed, *, include_effort=True):
    return {key: packed[:, channels] for key, channels in PROPRIO_FIELDS.items()
            if include_effort or key not in ("joint_eff", "gripper_eff")}


class YAMInterface(Protocol):
    def reset(self, mask=None): ...
    def get_observation(self, *, include_effort=True): ...
    def send_joint_command(self, position, *, velocity=None, feedforward=None): ...
    def send_cartesian_action(self, action): ...


class RealYAMInterface:
    """Offline SDK mapping for a future independently interlocked transport.

    SDK standard YAM v1 observations use six signed joints followed by a
    calibrated gripper. Gripper effort remains DM4310 output torque in Nm.
    """

    def __init__(self, config=None, *, calibrated_stroke_m=None):
        cfg = load_config("nominal", config)
        self.max_opening = cfg["gripper_max_gap_m"]
        self.stroke = self.max_opening if calibrated_stroke_m is None else calibrated_stroke_m
        if not np.isfinite(self.stroke) or self.stroke <= 0 or self.stroke > 0.096:
            raise ValueError("Expected a positive measured linear_4310 stroke up to 96 mm")

    def pack_observation(self, sdk, *, sent_position, include_effort=True):
        sent = np.asarray(sent_position, dtype=np.float32).reshape(7)
        packed = np.zeros((1, 28), dtype=np.float32)
        fields = observation_fields(packed)
        fields["joint_pos"][:] = np.asarray(sdk["joint_pos"]).reshape(6)
        fields["joint_vel"][:] = np.asarray(sdk["joint_vel"]).reshape(6)
        fields["joint_pos_cmd"][:] = sent[:6]
        fields["gripper_pos"][:] = np.clip(float(np.asarray(sdk["gripper_pos"]).item()) * self.stroke,
                                          0.0, self.max_opening)
        fields["gripper_vel"][:] = np.asarray(sdk["gripper_vel"]).item() * self.stroke
        fields["gripper_cmd"][:] = sent[6]
        if include_effort:
            fields["joint_eff"][:] = np.asarray(sdk["joint_eff"]).reshape(6)
            fields["gripper_eff"][:] = np.asarray(sdk["gripper_eff"]).item()
        if not np.isfinite(packed).all():
            raise ValueError("Non-finite YAM SDK observation")
        return packed

    def observation_from_sdk(self, sdk, *, sent_position, include_effort=True):
        packed = self.pack_observation(sdk, sent_position=sent_position, include_effort=include_effort)
        return observation_fields(packed, include_effort=include_effort)

    def sdk_joint_position(self, checked_position):
        """Encode a position already validated by the future hardware safety layer."""
        command = np.array(checked_position, dtype=float, copy=True).reshape(7)
        if not np.isfinite(command).all() or not 0 <= command[6] <= min(self.stroke, self.max_opening):
            raise ValueError("Position must be finite with a gripper target inside the calibrated stroke")
        command[6] /= self.stroke
        return command
