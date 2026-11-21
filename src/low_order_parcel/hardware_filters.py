"""Batched YAM current-estimation, encoder, and command-delay model (SI)."""

import math
from typing import NamedTuple

import torch
from torch import Tensor

from .unified_payload import DT


class YAMParameters(NamedTuple):
    sensor_gain: Tensor
    coulomb_N: Tensor
    pose_bias_N: Tensor
    cutoff_hz: Tensor
    noise_N: Tensor
    delay_steps: Tensor
    gripper_deadzone_N: Tensor
    gripper_gain: Tensor
    opening_deadzone_m: Tensor

    @classmethod
    def sample(cls, like: Tensor, *, generator: torch.Generator | None = None):
        """One independent draw per environment and arm, each channel [B,2,1]."""
        shape = (len(like), 2, 1)

        def uniform(lo, hi):
            draw = torch.rand(shape, dtype=like.dtype, device=like.device, generator=generator)
            return lo + (hi - lo) * draw

        delay = torch.randint(5, 26, shape, device=like.device, generator=generator)
        return cls(uniform(.75, 1.25), uniform(.5, 2.), uniform(-1.5, 1.5),
                   uniform(4., 15.), uniform(.2, .6), delay,
                   uniform(.05, .25), uniform(.80, 1.20), uniform(.0005, .002))


class YAMObservation(NamedTuple):
    force: Tensor           # [B,2,3], current-based force estimate in world coordinates
    gripper_current: Tensor  # [B,2,1], current expressed in equivalent N
    opening: Tensor         # [B,2,1], m
    position_error: Tensor  # [B,2,3], encoder actual minus commanded, m
    valid: Tensor           # [B,2,1], fresh history exists for the configured delay


class YAMCurrentEstimatorFilter:
    def __init__(self, params: YAMParameters, *, generator: torch.Generator | None = None,
                 velocity_scale: float = .01):
        like = params.sensor_gain
        if (like.ndim != 3 or like.shape[1:] != (2, 1) or like.dtype != torch.float32
                or not math.isfinite(velocity_scale) or velocity_scale <= 0.):
            raise ValueError("Use float32 YAM parameters [B,2,1] and a positive velocity scale")
        for name, value in zip(params._fields, params):
            dtype = torch.long if name == "delay_steps" else like.dtype
            if value.shape != like.shape or value.device != like.device or value.dtype != dtype:
                raise ValueError(f"{name} must match the YAM batch and use {dtype}")
            if not value.isfinite().all():
                raise ValueError(f"{name} must be finite")
        if ((params.delay_steps < 0) | (params.delay_steps > 25)).any():
            raise ValueError("YAM delay must be 0–25 steps")
        if ((params.cutoff_hz <= 0.) | (params.noise_N < 0.) | (params.coulomb_N < 0.)
                | (params.sensor_gain <= 0.) | (params.gripper_gain <= 0.)
                | (params.gripper_deadzone_N < 0.) | (params.opening_deadzone_m < 0.)).any():
            raise ValueError("Gains and bandwidth must be positive; noise and deadzones must be nonnegative")
        self.params, self.generator = params, generator
        self.velocity_scale = velocity_scale
        self.blend = -torch.expm1(-2. * math.pi * params.cutoff_hz * DT)
        self._filtered = like.new_zeros((len(like), 2, 3))
        self._opening = torch.zeros_like(like)
        self._observation = like.new_zeros((26, len(like), 2, 8))
        self._command = like.new_zeros((26, len(like), 2, 6))
        self._obs_age = torch.zeros_like(params.delay_steps)
        self._cmd_age = torch.zeros_like(params.delay_steps)
        self._obs_index = self._cmd_index = 0
        self._batch = torch.arange(len(like), device=like.device)[:, None]
        self._arm = torch.arange(2, device=like.device)

    def reset(self, reset_mask: Tensor, *, params: YAMParameters | None = None) -> None:
        """Clear selected histories and optionally install new episode draws."""
        mask = reset_mask[:, None, None]
        if params is not None:
            self.params = YAMParameters(*(torch.where(mask, new, old)
                                          for new, old in zip(params, self.params, strict=True)))
            self.blend = -torch.expm1(-2. * math.pi * self.params.cutoff_hz * DT)
        self._filtered = torch.where(mask, 0., self._filtered)
        self._opening = torch.where(mask, 0., self._opening)
        self._obs_age = torch.where(mask, 0, self._obs_age)
        self._cmd_age = torch.where(mask, 0, self._cmd_age)

    def step(self, force_true: Tensor, velocity: Tensor, gravity_direction: Tensor,
             gripper_force: Tensor, gripper_opening: Tensor, actual_position: Tensor,
             command_position: Tensor) -> YAMObservation:
        """Ingest a physics sample. The gravity direction models FK-dependent bias."""
        p = self.params
        friction = p.coulomb_N * torch.tanh(velocity / self.velocity_scale)
        noise = torch.randn(force_true.shape, device=force_true.device, dtype=force_true.dtype,
                            generator=self.generator) * p.noise_N
        raw = p.sensor_gain * force_true + friction + p.pose_bias_N * gravity_direction + noise
        self._filtered = torch.lerp(self._filtered, raw, self.blend)
        current = p.gripper_gain * (gripper_force - p.gripper_deadzone_N).clamp_min(0.)
        # A play operator preserves opening until travel takes up the measured backlash.
        self._opening = torch.where(self._obs_age == 0, gripper_opening, self._opening.clamp(
            gripper_opening - p.opening_deadzone_m, gripper_opening + p.opening_deadzone_m))
        error = actual_position - command_position
        self._observation[self._obs_index] = torch.cat((self._filtered, current, self._opening, error), dim=-1)
        index = (self._obs_index - p.delay_steps.squeeze(-1)) % len(self._observation)
        delayed = self._observation[index, self._batch, self._arm]
        self._obs_index = (self._obs_index + 1) % len(self._observation)
        valid = self._obs_age >= p.delay_steps
        self._obs_age = (self._obs_age + 1).clamp_max(len(self._observation))
        return YAMObservation(*torch.where(valid, delayed, 0.).split((3, 1, 1, 3), dim=-1), valid)

    def impedance_step(self, position: Tensor, velocity: Tensor, command_position: Tensor,
                       command_velocity: Tensor, load: Tensor) -> tuple[Tensor, Tensor]:
        """Advance a 2 kg effective endpoint with 220 N/m and 42 N s/m impedance."""
        self._command[self._cmd_index] = torch.cat((command_position, command_velocity), dim=-1)
        index = (self._cmd_index - self.params.delay_steps.squeeze(-1)) % len(self._command)
        target, target_velocity = self._command[index, self._batch, self._arm].split(3, dim=-1)
        self._cmd_index = (self._cmd_index + 1) % len(self._command)
        ready = self._cmd_age >= self.params.delay_steps
        self._cmd_age = (self._cmd_age + 1).clamp_max(len(self._command))
        target = torch.where(ready, target, position)
        target_velocity = torch.where(ready, target_velocity, 0.)
        acceleration = (220. * (target - position) + 42. * (target_velocity - velocity) + load) / 2.
        velocity = velocity + DT * acceleration
        return position + DT * velocity, velocity
