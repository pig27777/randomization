"""Work-conjugate finite-pad torsion, with no world-relative damping."""

from typing import NamedTuple

import torch
from torch import Tensor

from .lumped_contact_proxy import ContactState, EPS


def rotate(q: Tensor, points: Tensor) -> Tensor:
    """Rotate vectors by a unit quaternion in wxyz order."""
    qv = q[..., 1:].expand_as(points)
    cross = 2. * torch.linalg.cross(qv, points)
    return points + q[..., :1] * cross + torch.linalg.cross(qv, cross)


def _orientation(frame: Tensor) -> Tensor:
    """Use the largest quaternion component as the pivot, including at 180 degrees."""
    (xx, yx, zx), (xy, yy, zy), (xz, yz, zz) = (col.unbind(dim=1) for col in frame.unbind(dim=2))
    squares = torch.stack((1. + xx + yy + zz, 1. + xx - yy - zz,
                           1. - xx + yy - zz, 1. - xx - yy + zz), dim=1)
    w, x, y, z = squares.unbind(dim=1)
    wx, wy, wz = zy - yz, xz - zx, yx - xy
    xy, xz, yz = xy + yx, xz + zx, yz + zy
    candidates = torch.stack((torch.stack((w, wx, wy, wz), dim=1),
                              torch.stack((wx, x, xy, xz), dim=1),
                              torch.stack((wy, xy, y, yz), dim=1),
                              torch.stack((wz, xz, yz, z), dim=1)), dim=1)
    index = squares.argmax(dim=1).view(-1, 1, 1).expand(-1, 1, 4)
    return torch.nn.functional.normalize(candidates.gather(1, index).squeeze(1), dim=1)


def rotation_quaternion(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat(((angle / 2.).cos(), .5 * torch.sinc(angle / (2. * torch.pi)) * vector), dim=-1)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quaternion_product(a, b):
    av, bv = a[..., 1:], b[..., 1:]
    return torch.cat((a[..., :1] * b[..., :1] - (av * bv).sum(dim=-1, keepdim=True),
                      a[..., :1] * bv + b[..., :1] * av + torch.linalg.cross(av, bv)), dim=-1)


def rotation_vector(q):
    q = torch.where(q[..., :1] < 0., -q, q)
    sine = q[..., 1:].norm(dim=-1, keepdim=True)
    return q[..., 1:] * (2. * torch.atan2(sine, q[..., :1]) / sine.clamp_min(EPS))


class PadTorsionParams(NamedTuple):
    radius_m: Tensor | float = .012
    damping_ratio: Tensor | float = .80
    neck_length_m: Tensor | float = .035
    stiffness_Nm_rad: Tensor | float = 2.4


class TorsionState(NamedTuple):
    reference: Tensor
    energy: Tensor
    jaw_work: Tensor
    body_work: Tensor
    damping: Tensor
    slip_loss: Tensor
    integration_loss: Tensor
    slip_angle: Tensor
    capacity: Tensor
    torque: Tensor


class PadTorsion:
    def __init__(self, like, mu, dt, params):
        self.dt = dt
        self.mu = mu
        self.radius, self.zeta, self.length, self.stiffness = (
            torch.as_tensor(value, device=like.device, dtype=like.dtype).expand_as(like)
            for value in params)
        if not ((self.radius >= 0.) & (self.zeta >= 0.) & (self.length > 0.) & (self.stiffness > 0.) &
                torch.stack((self.radius, self.zeta, self.length, self.stiffness)).isfinite().all(dim=0)).all():
            raise ValueError("Invalid pad radius or membrane-neck parameters")
        arms = mu.shape[1]
        self.identity = like.new_tensor((1., 0., 0., 0.)).expand(len(like), arms, 4)
        self.zero = like.new_zeros((len(like), arms, 1))
        self.zero_angular = like.new_zeros((len(like), arms, 3))
        self.eye = torch.eye(3, device=like.device, dtype=like.dtype)

    def initial_state(self):
        return TorsionState(self.identity, *(self.zero for _ in range(8)), self.zero_angular)

    @staticmethod
    def _adjugate(matrix):
        a, b, c = matrix.unbind(dim=1)
        adj = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a), torch.linalg.cross(a, b)), dim=-1)
        return adj, (a * adj[:, :, 0]).sum(dim=-1, keepdim=True)

    def block_kinematics(self, payload, state):
        mass = payload.mass.unsqueeze(-1)
        r = state.pos - payload.com(state)[:, None]
        second = (mass * r).transpose(-1, -2) @ r
        inertia = second.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * self.eye - second
        adj, det = self._adjugate(inertia)
        valid = det > (inertia.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).pow(3) * 1e-8)
        inverse = torch.where(valid[..., None], adj / torch.where(valid, det, 1.)[..., None], 0.)
        momentum = (mass * torch.linalg.cross(r, state.vel - payload.com_velocity(state)[:, None])).sum(dim=1)
        omega = (inverse @ momentum.unsqueeze(-1)).squeeze(-1)
        return r, inverse, omega, valid

    def couples(self, payload, state, frame, old, prior, current, normal, separation, base_load, jaw, jaw_rate):
        r, inverse, omega, valid = self.block_kinematics(payload, state)
        base_moment = torch.linalg.cross(r, base_load).sum(dim=1)
        omega = omega + self.dt * (inverse @ base_moment.unsqueeze(-1)).squeeze(-1)
        torque, angular_acceleration, next_state = self.step(
            _orientation(frame), inverse, omega, valid, old, prior, current, normal, separation, jaw, jaw_rate)
        block_force = payload.mass.unsqueeze(-1) * torch.linalg.cross(angular_acceleration[:, None], r)
        # Remove finite-precision COM residual without changing the moment about COM.
        block_force = block_force - payload.weights.unsqueeze(-1) * block_force.sum(dim=1, keepdim=True)
        return torque, block_force, next_state

    def step(self, orientation, inverse, omega, valid, old, prior, current, normal, separation, jaw, jaw_rate):
        if not old:
            old = self.initial_state()
        jaw = self.identity if jaw is None else jaw
        jaw_rate = self.zero_angular if jaw_rate is None else jaw_rate
        previously_held = (prior[..., ContactState.CAPTURE] > .5).any(dim=-1, keepdim=True)
        captured = current[..., ContactState.CAPTURE] > .5
        held = captured.any(dim=-1, keepdim=True) & (self.radius[:, None] > 0.)
        free_span = (separation.norm(dim=-1) - current[..., ContactState.SLACK]
                     - current[..., ContactState.SLIP_CUM]).clamp_min(0.)
        neck = (free_span * captured).sum(dim=-1, keepdim=True) / captured.sum(dim=-1, keepdim=True).clamp_min(1)
        k = self.stiffness[:, None] / (1. + (neck / self.length[:, None]).square())
        capacity = self.radius[:, None] * (self.mu * normal.clamp_min(0.) * captured).sum(dim=-1, keepdim=True)
        relative = quaternion_product(conjugate(jaw), orientation[:, None].expand_as(jaw))
        # Both arms operate on the same momentum. The Frobenius norm bounds
        # the directional angular mobility for the dashpot impulse limit.
        mobility = inverse.flatten(1).norm(dim=1, keepdim=True).clamp_min(EPS)
        torques, rows = [], []
        for arm in range(self.mu.shape[1]):
            active = held[:, arm] & previously_held[:, arm] & valid
            cap = torch.where(active, capacity[:, arm], 0.)
            stiff = k[:, arm]
            rel, qj, rate_j = relative[:, arm], jaw[:, arm], jaw_rate[:, arm]
            energy_before = old.energy[:, arm]
            velocity = omega - rate_j
            error = rotate(qj, rotation_vector(quaternion_product(rel, conjugate(old.reference[:, arm]))))
            angle_limit = torch.minimum((2. * energy_before / stiff).clamp_min(0.).sqrt(), cap / stiff)
            angle = error * (angle_limit / error.norm(dim=-1, keepdim=True).clamp_min(EPS)).clamp_max(1.)
            c = torch.minimum(2. * self.zeta * (stiff / mobility).sqrt(), (self.dt * mobility).reciprocal())
            c = torch.minimum(c, cap / (velocity.norm(dim=-1, keepdim=True)
                                        + self.dt * mobility * cap).clamp_min(EPS))
            gamma = c + self.dt * stiff
            system = self.eye + (self.dt * gamma).unsqueeze(-1) * inverse
            adj, det = self._adjugate(system)
            trial = -(adj @ (stiff * angle + gamma * velocity).unsqueeze(-1)).squeeze(-1) / det
            torque = trial * (cap / trial.norm(dim=-1, keepdim=True).clamp_min(EPS)).clamp_max(1.)
            change = self.dt * (inverse @ torque.unsqueeze(-1)).squeeze(-1)
            model_rate = velocity + change
            delta = self.dt * model_rate
            slip = (system @ (torque - trial).unsqueeze(-1)).squeeze(-1) / stiff
            proposed = angle + delta - slip
            end = proposed * ((cap / stiff) / proposed.norm(dim=-1, keepdim=True).clamp_min(EPS)).clamp_max(1.)
            energy = .5 * stiff * end.square().sum(dim=-1, keepdim=True)
            next_rel = quaternion_product(rotation_quaternion(rotate(conjugate(qj), delta)), rel)
            reference = quaternion_product(rotation_quaternion(rotate(conjugate(qj), -end)), next_rel)
            reference = torch.nn.functional.normalize(reference, dim=-1, eps=EPS)
            reference = torch.where(active, reference, torch.where(held[:, arm], rel, self.identity[:, arm]))
            jaw_work = self.dt * (torque * rate_j).sum(dim=-1, keepdim=True)
            body_work = self.dt * (torque * (omega + .5 * change)).sum(dim=-1, keepdim=True)
            damping = self.dt * c * model_rate.square().sum(dim=-1, keepdim=True)
            slip_loss = -(torque * slip).sum(dim=-1, keepdim=True)
            projection_loss = energy_before + .5 * stiff * (
                proposed.square().sum(dim=-1, keepdim=True) - angle.square().sum(dim=-1, keepdim=True)) - energy
            integration = .5 * stiff * (proposed - angle).square().sum(dim=-1, keepdim=True)
            midpoint_loss = .5 * self.dt * (torque * change).sum(dim=-1, keepdim=True)
            rows.append(TorsionState(reference, energy, jaw_work, body_work, damping, slip_loss,
                                     torch.where(active, projection_loss + integration + midpoint_loss,
                                                 energy_before),
                                     torch.where(active, slip.norm(dim=-1, keepdim=True), 0.), capacity[:, arm], torque))
            omega = omega + change
            torques.append(torque)
        torque = torch.stack(torques, dim=1)
        acceleration = (inverse @ torque.sum(dim=1).unsqueeze(-1)).squeeze(-1)
        return torque, acceleration, TorsionState(*(torch.stack(field, dim=1) for field in zip(*rows)))
