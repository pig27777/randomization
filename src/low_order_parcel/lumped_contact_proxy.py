"""Batched, unilateral film capture and two independent slack ropes (SI)."""

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import NamedTuple

import torch
from torch import Tensor

EPS = 1e-8


class ContactState(IntEnum):
    """Channel indices in the [B, left/right, 6] contact tensor."""

    SLACK = 0
    CAPTURE = 1
    SLIP_CUM = 2
    HOLD_TIME = 3
    FEED_DEPTH = 4
    PEAK_T = 5


@dataclass(frozen=True)
class ProxyParams:
    """Shared quantities have shape [B, 1], pad quantities [B, 2]."""

    S_bag: Tensor
    s_max: Tensor
    alpha: Tensor
    k: Tensor
    c_d: Tensor
    mu_eff: Tensor
    beta: Tensor
    s_crit: Tensor
    F_min: Tensor
    F_lock: Tensor
    F_release: Tensor
    h_min: Tensor
    cos_theta_max: Tensor
    tau_gate: Tensor
    v_gather_max: Tensor
    L0: Tensor

    @classmethod
    def from_scalars(cls, values, *, dtype=torch.float32, device=None):
        """Create one environment from scalar quantities and two pad pairs."""
        return cls(**{field.name: torch.as_tensor(values[field.name], dtype=dtype, device=device).reshape(1, -1)
                      for field in fields(cls)})

    def validate(self):
        batch = self.S_bag.shape[0]
        for field in fields(self):
            value = getattr(self, field.name)
            shape = (batch, 2 if field.name in {"mu_eff", "L0"} else 1)
            if value.shape != shape or value.dtype != self.S_bag.dtype or value.device != self.S_bag.device:
                raise ValueError(f"{field.name} must have shape {shape} and the device and dtype of S_bag")
            if not value.is_floating_point() or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{field.name} must contain finite nonnegative floating-point values")
        if ((self.s_crit <= EPS) | (self.s_crit > self.s_max)).any():
            raise ValueError("s_crit must be positive and no larger than s_max")
        if ((self.F_release >= self.F_lock) | (self.F_min > self.F_lock)).any():
            raise ValueError("F_release < F_lock and F_min <= F_lock are required")
        if ((self.alpha > 1) | (self.cos_theta_max > 1) | (self.k <= 0)).any():
            raise ValueError("alpha and cos_theta_max must be within [0, 1], and k must be positive")


class ContactInputs(NamedTuple):
    pinch_position: Tensor  # [B, 2, 3], world frame
    bag_attachment: Tensor  # [B, 2, 3], world frame
    pinch_velocity: Tensor  # [B, 2, 3], m/s
    attachment_velocity: Tensor  # [B, 2, 3], m/s
    bag_com: Tensor  # [B, 3], world frame
    normal_force: Tensor  # [B, 2], compressive N
    inward_velocity: Tensor  # [B, 2], signed closing m/s
    feed_depth: Tensor  # [B, 2], m
    alignment_cos: Tensor  # [B, 2], projected pad and local fold normals
    S_bag_eff: Tensor | None = None  # [B, 1], optional step-local film budget, m


class ContactStep(NamedTuple):
    state: Tensor
    force: Tensor  # [B, 2, 3], on bag
    torque: Tensor  # [B, 2, 3], about bag COM
    tension: Tensor  # [B, 2], N
    slip_rate: Tensor  # [B, 2], m/s
    elastic_energy: Tensor  # [B, 2], J
    elastic_tension: Tensor  # [B, 2], N before adding damping


class LumpedContactProxy:
    def __init__(self, params: ProxyParams, dt: float, *, ablation_rigid_snap: bool = False):
        if not .001 <= dt <= .002:
            raise ValueError("The supported step is 1–2 ms")
        params.validate()
        self.params = params
        self.dt = dt
        self.ablation_rigid_snap = ablation_rigid_snap
        self.stiffness = params.k.clamp_min(1000.) if ablation_rigid_snap else params.k

    def initial_state(self) -> Tensor:
        return self.params.S_bag.new_zeros((self.params.S_bag.shape[0], 2, len(ContactState)))

    def validate_inputs(self, state: Tensor, inputs: ContactInputs) -> None:
        """Validate external data at ingestion, without synchronizing the step kernel."""
        batch = self.params.S_bag.shape[0]
        shapes = ((batch, 2, 6), *((batch, 2, 3),) * 4, (batch, 3), *((batch, 2),) * 4, (batch, 1))
        for name, value, shape in zip(("state", *inputs._fields), (state, *inputs), shapes, strict=True):
            if name == "S_bag_eff" and value is None:
                continue
            if value.shape != shape or value.dtype != self.params.S_bag.dtype or value.device != self.params.S_bag.device:
                raise ValueError(f"{name} must have shape {shape} and the device and dtype of S_bag")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if inputs.S_bag_eff is not None and (inputs.S_bag_eff < 0.).any():
            raise ValueError("S_bag_eff must be nonnegative")

    def step(self, state: Tensor, inputs: ContactInputs) -> ContactStep:
        p, dt = self.params, self.dt
        budget = p.S_bag if inputs.S_bag_eff is None else inputs.S_bag_eff
        slack, captured, slip_cum, hold, _, peak = state.unbind(dim=-1)
        active = captured > .5
        normal = inputs.normal_force.clamp_min(0.)
        depth = inputs.feed_depth.clamp_min(0.)
        pressed = normal >= p.F_min
        hold = torch.where(pressed, hold + dt, 0.)
        geometry = (depth >= p.h_min) & (inputs.alignment_cos >= p.cos_theta_max)

        separation = inputs.pinch_position - inputs.bag_attachment
        distance = torch.linalg.vector_norm(separation, dim=-1)
        direction = separation / distance.clamp_min(EPS).unsqueeze(-1)
        length_rate = ((inputs.pinch_velocity - inputs.attachment_velocity) * direction).sum(dim=-1)
        beyond_rest = distance - p.L0 - slip_cum
        strain = beyond_rest.clamp_min(0.)
        trial_tension = torch.where(active & (beyond_rest >= 0.),
                                    (p.k * strain + p.c_d * length_rate).clamp_min(0.), 0.)

        free_fraction = (1. - slack.sum(dim=1, keepdim=True) / budget.clamp_min(EPS)).clamp_min(0.)
        gather_speed = torch.minimum(inputs.inward_velocity.clamp_min(0.), p.v_gather_max)
        gathered = torch.where(pressed & (hold >= p.tau_gate), p.alpha * gather_speed * free_fraction * dt, 0.)
        available = torch.minimum(slack + gathered, p.s_max)
        # Withdrawal consumes physical reserve. Apply the global clamp before withdrawal so
        # simultaneous pad motion cannot withdraw more film than was ever available.
        total = available.sum(dim=1, keepdim=True)
        available = available * (budget / total.clamp_min(EPS)).clamp(max=1.)
        withdrawn = torch.minimum(available, dt * p.beta * (trial_tension - p.mu_eff * normal).clamp_min(0.))
        slack_next = available - withdrawn
        rate = withdrawn / dt
        release = (slack_next <= EPS) | (normal < p.F_release)
        lock = geometry & (slack_next >= p.s_crit) & (normal >= p.F_lock)
        captured_next = torch.where(active, ~release, lock)
        slip_next = slip_cum + withdrawn

        # The ablation flag is model-wide and static across all environments and steps.
        if self.ablation_rigid_snap:
            captured_next = torch.where(active, normal >= p.F_release,
                                        geometry & (distance <= p.L0 + p.h_min) & (normal >= p.F_lock))
            slack_next = torch.zeros_like(slack)
            slip_next = slip_cum
            rate = torch.zeros_like(rate)

        remaining = distance - p.L0 - slip_next
        extension = remaining.clamp_min(0.)
        tension = torch.where(captured_next & (remaining >= 0.),
                              (self.stiffness * extension + p.c_d * (length_rate - rate)).clamp_min(0.), 0.)
        force = tension.unsqueeze(-1) * direction
        torque = torch.linalg.cross(inputs.bag_attachment - inputs.bag_com.unsqueeze(1), force)
        new_peak = torch.where(captured_next, torch.maximum(torch.where(active, peak, 0.), tension), 0.)
        new_state = torch.stack((slack_next, captured_next.to(state.dtype), slip_next, hold, depth, new_peak), dim=-1)
        energy = torch.where(captured_next, .5 * self.stiffness * extension.square(), 0.)
        elastic_tension = torch.where(captured_next, self.stiffness * extension, 0.)
        return ContactStep(new_state, force, torque, tension, rate, energy, elastic_tension)
