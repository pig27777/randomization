"""Episode-randomized current observations from the YAM motor effort buffers."""

import math
from typing import NamedTuple

import torch


class JointCurrentParams(NamedTuple):
    gain: torch.Tensor
    deadzone: torch.Tensor                  # N·m on the arm, N at a rail
    coulomb: torch.Tensor
    pose_bias: torch.Tensor
    cutoff_hz: torch.Tensor
    noise: torch.Tensor
    delay_steps: torch.Tensor
    opening_deadzone_m: torch.Tensor

    @classmethod
    def sample(cls, like, *, generator=None):
        def draw(low, high, size=7):
            return low + (high - low) * torch.rand(
                (*like.shape[:-1], size), device=like.device, dtype=like.dtype, generator=generator)

        # Nominal lever lengths express the 40:1 current deadzone in joint torque units.
        lever = like.new_tensor([.25, .30, .25, .12, .09, .055])
        gain = torch.cat((draw(.75, 1.25, 6), draw(.80, 1.20, 1)), dim=-1)
        deadzone = torch.cat((lever * draw(.5, 2., 6), draw(.05, .25, 1)), dim=-1)
        bias = torch.cat((lever * draw(-.4, .4, 6), torch.zeros_like(deadzone[..., :1])), dim=-1)
        noise = torch.cat((lever * draw(.04, .15, 6), draw(.01, .04, 1)), dim=-1)
        delays = torch.randint(5, 26, like.shape, device=like.device, generator=generator)
        return cls(gain, deadzone, .5 * deadzone, bias, draw(4., 15.), noise, delays,
                   draw(.0005, .002, 1))


class JointCurrentObservation(NamedTuple):
    delta_effort: torch.Tensor
    tracking_error: torch.Tensor
    opening_m: torch.Tensor


class JointCurrentEstimatorFilter:
    """Track the delayed, tared seven motor channels per arm in torque/force units."""

    def __init__(self, params, *, dt=.002, tare_steps=50, generator=None):
        self.params, self.dt, self.tare_steps, self.generator = params, dt, tare_steps, generator
        like = params.gain
        self.filtered = torch.zeros_like(like)
        self.tare = torch.zeros_like(like)
        self.age = torch.zeros_like(params.delay_steps[..., :1])
        self.opening = torch.zeros_like(params.opening_deadzone_m)
        self.ring = like.new_zeros((26, *like.shape))
        self.index = 0
        self.blend = -torch.expm1(-2. * math.pi * params.cutoff_hz * dt)
        self.channels = torch.arange(like.numel(), device=like.device)

    def reset(self, mask, params=None):
        select = mask[:, None, None]
        if params is not None:
            self.params = JointCurrentParams(*(torch.where(select, new, old)
                                               for new, old in zip(params, self.params, strict=True)))
            self.blend = -torch.expm1(-2. * math.pi * self.params.cutoff_hz * self.dt)
        for tensor in (self.filtered, self.tare, self.age, self.opening):
            tensor.masked_fill_(select, 0)
        self.ring.masked_fill_(select.unsqueeze(0), 0.)

    def step(self, motor_effort, q, dq, q_des, opening_m):
        p = self.params
        effort = motor_effort.sign() * (motor_effort.abs() - p.deadzone).clamp_min(0.)
        friction = p.coulomb * torch.tanh(dq / .015)
        noise = torch.randn(motor_effort.shape, dtype=motor_effort.dtype, device=motor_effort.device,
                            generator=self.generator)
        raw = p.gain * effort + friction + p.pose_bias * q.sin() + noise * p.noise
        self.filtered.lerp_(raw, self.blend)
        taring = self.age < self.tare_steps
        mean = self.tare + (self.filtered - self.tare) / (self.age + 1)
        self.tare.copy_(torch.where(taring, mean, self.tare))
        self.ring[self.index] = self.filtered - self.tare
        delays = (self.index - p.delay_steps.reshape(-1)) % len(self.ring)
        delayed = self.ring.flatten(1)[delays, self.channels].reshape_as(motor_effort)
        valid = self.age >= self.tare_steps + p.delay_steps
        backlash = self.opening.clamp(opening_m - p.opening_deadzone_m, opening_m + p.opening_deadzone_m)
        self.opening.copy_(torch.where(self.age == 0, opening_m, backlash))
        self.index = (self.index + 1) % len(self.ring)
        self.age.add_(1).clamp_max_(self.tare_steps + len(self.ring))
        return JointCurrentObservation(torch.where(valid, delayed, 0.), q - q_des, self.opening)
