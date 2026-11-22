"""Power-conjugate film-neck damping at the pad/anchor contact port."""

from typing import NamedTuple

import torch
from torch import Tensor

from .lumped_contact_proxy import ContactState, EPS
from .pad_torsion import rotate


class NeckDampingParams(NamedTuple):
    coefficient_Ns_m: Tensor | float = 0.
    characteristic_m: Tensor | float = .050


class NeckDampingStep(NamedTuple):
    force: Tensor
    pad_couple: Tensor
    anchor_power: Tensor
    pad_force_power: Tensor
    pad_couple_power: Tensor
    dissipation_J: Tensor


class NeckDamping:
    """A transverse damper using one mobility matrix for all simultaneous pads."""

    def __init__(self, like, mu, dt, params):
        self.dt = dt
        self.mu = mu
        self.c, self.length = (
            torch.as_tensor(value, device=like.device, dtype=like.dtype).expand_as(like)
            for value in params
        )
        if not (self.c.isfinite() & (self.c >= 0.) & self.length.isfinite() & (self.length > 0.)).all():
            raise ValueError("Expected nonnegative neck damping and positive characteristic length")
        self.enabled = bool((self.c > 0.).any())
        arms = mu.shape[1]
        ids = torch.arange(arms, device=like.device).repeat_interleave(2)
        # In the macro wrist, rotations are prescribed and each translating wrist is 0.25 kg.
        self.wrist_mobility = 4. * (ids[:, None] == ids)

    def relative(self, tip, anchor, pad_velocity, anchor_velocity, jaw_rate):
        r = anchor - tip
        length = r.norm(dim=-1, keepdim=True)
        axis = r / length.clamp_min(EPS)
        relative = anchor_velocity - pad_velocity - torch.linalg.cross(jaw_rate, r)
        transverse = relative - (relative * axis).sum(dim=-1, keepdim=True) * axis
        speed = transverse.norm(dim=-1, keepdim=True)
        return r, length.squeeze(-1), transverse / speed.clamp_min(EPS), speed.squeeze(-1)

    def resolve(self, relative, prev, current, normal, motion, pad_velocity, anchor_velocity, jaw_rate,
                articulated_mobility=None):
        r, length, direction, speed = relative
        span = (length - current[..., ContactState.SLACK] - current[..., ContactState.SLIP_CUM]).clamp_min(0.)
        coefficient = self.c / (1. + (span / self.length).square())
        active = ((prev[..., ContactState.CAPTURE] == 1.) & (current[..., ContactState.CAPTURE] == 1.)
                  & (length > EPS) & (speed > EPS))
        radius = self.mu.flatten(1) * normal.clamp_min(0.)
        magnitude = torch.where(active, torch.minimum(coefficient * speed, radius), 0.)
        if articulated_mobility is None:
            articulated_mobility = (direction @ direction.transpose(1, 2)) * self.wrist_mobility
        motion = motion + articulated_mobility
        projected_delta = self.dt * (motion @ magnitude.unsqueeze(-1)).squeeze(-1)
        # One common factor prevents concurrent pads from pushing through a slower pad's zero crossing.
        scale = torch.where(active & (projected_delta > speed),
                            speed / projected_delta.clamp_min(EPS), 1.).amin(dim=1, keepdim=True)
        magnitude = magnitude * scale
        force = -magnitude.unsqueeze(-1) * direction
        pad_couple = -torch.linalg.cross(r, force)
        post_speed = speed - scale * projected_delta
        loss = self.dt * magnitude * (.5 * (speed + post_speed))
        return NeckDampingStep(force, pad_couple, (force * anchor_velocity).sum(dim=-1),
                               (force * pad_velocity).sum(dim=-1), -(pad_couple * jaw_rate).sum(dim=-1),
                               loss)

    def payload(self, model, state, weights, tip, anchor, pad_velocity, anchor_velocity,
                prev, current, normal, jaw_rate=None, port_mobility=None):
        if jaw_rate is None:
            jaw_rate = torch.zeros_like(tip)
        relative = self.relative(tip, anchor, pad_velocity, anchor_velocity, jaw_rate)
        direction = relative[2]
        # Adjoint of the existing anchor kinematics in the actual transverse directions.
        n, a, b, length = model._normal(state.pos)
        height = weights @ model.half_thickness(state.d_surf).unsqueeze(-1)
        projected = direction - (direction * n[:, None]).sum(dim=-1, keepdim=True) * n[:, None]
        adjoint = torch.where(length[:, None] > EPS, height * projected / length[:, None].clamp_min(EPS), 0.)
        ga = torch.linalg.cross(b[:, None], adjoint)
        gb = torch.linalg.cross(adjoint, a[:, None])
        jacobian = weights.unsqueeze(-1) * direction.unsqueeze(-2) + torch.stack((-ga, -gb, gb, ga), dim=-2)
        mobility = (jacobian * model.inv_mass[:, None, :, None]).flatten(2) @ jacobian.flatten(2).transpose(1, 2)
        articulated = None if port_mobility is None else port_mobility(relative[0], direction)
        return self.resolve(relative, prev, current, normal, mobility, pad_velocity, anchor_velocity, jaw_rate,
                            articulated)

    def rigid(self, tip, anchor, pad_velocity, anchor_velocity, prev, current, normal,
              mass, mounts, inverse_q, inverse_inertia, jaw_rate=None, port_mobility=None):
        if jaw_rate is None:
            jaw_rate = torch.zeros_like(tip)
        relative = self.relative(tip, anchor, pad_velocity, anchor_velocity, jaw_rate)
        direction = relative[2]
        lever = rotate(inverse_q[:, None], torch.linalg.cross(mounts, direction))
        mobility = ((direction @ direction.transpose(1, 2)) / mass[:, :, None]
                    + (lever @ inverse_inertia) @ lever.transpose(1, 2))
        articulated = None if port_mobility is None else port_mobility(relative[0], direction)
        return self.resolve(relative, prev, current, normal, mobility, pad_velocity, anchor_velocity, jaw_rate,
                            articulated)
