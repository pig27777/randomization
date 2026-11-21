"""Four material points, XPBD constraints and unilateral surface compression (SI)."""

from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
from torch import Tensor

DT = .002
GRAVITY = 9.81
EPS = 1e-8
CORE_STIFFNESS = 10000.


@dataclass(frozen=True)
class PayloadParamsX:
    bag_half_extents: Tensor
    fill_ratio: Tensor
    total_mass: Tensor
    mass_weights: Tensor
    alpha_bend: Tensor
    theta_max: Tensor
    alpha_stretch: Tensor
    slide_slack_max: Tensor
    internal_damping: Tensor
    max_indent_ratio: Tensor
    k_soft: Tensor
    hardening_gamma: Tensor
    c_comp: Tensor
    tau_rebound: Tensor
    table_friction: Tensor
    lateral_crush_k: Tensor

    @classmethod
    def from_presets(cls, config, names, *, device=None, dtype=torch.float32, **overrides):
        rows = [config["defaults"] | config["presets"][name] for name in names]
        values = {field.name: torch.tensor([row[field.name] for row in rows], device=device,
                                           dtype=dtype).reshape(len(rows), -1) for field in fields(cls)}
        return cls(**(values | overrides))

    @staticmethod
    def cavity_slack(half_extents, fill_ratio):
        return .70 * (1. - fill_ratio.pow(1. / 3.)) * half_extents[:, :2].amin(dim=1, keepdim=True)

    def fill_slack_capacity(self):
        """Available perimeter for an envelope scaled isotropically by fill."""
        fraction = self.fill_ratio.pow(1. / 3.)
        section_perimeter = 4. * (self.bag_half_extents[:, :1] + self.bag_half_extents[:, 2:3])
        return section_perimeter * (1. - fraction) / fraction.clamp_min(EPS)

    @classmethod
    def sample(cls, config, batch, *, device=None, generator=None):
        values = {}
        for field in fields(cls):
            name = field.name
            kind, lower, upper = config["distributions"][name]
            width = {"bag_half_extents": 3, "mass_weights": 4}.get(name, 1)
            u = torch.rand((batch, width), generator=generator, device=device, dtype=torch.float32)
            lo, hi = u.new_tensor(lower), u.new_tensor(upper)
            if kind == "dirichlet_with_floor" and lower == 1.0:
                exponentials = -u.clamp(min=EPS, max=1. - 1e-7).log()
                values[name] = hi + (1. - width * hi) * exponentials / exponentials.sum(dim=1, keepdim=True)
            elif kind == "log_uniform":
                values[name] = torch.exp(torch.lerp(lo.log(), hi.log(), u))
            elif kind == "uniform":
                values[name] = torch.lerp(lo, hi, u)
            else:
                raise ValueError(f"Unsupported distribution for {name}: {kind}")
        values["slide_slack_max"] *= cls.cavity_slack(values["bag_half_extents"], values["fill_ratio"])
        return cls(**values)

    def validate(self):
        like = self.total_mass
        if like.ndim != 2 or not len(like) or like.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use a nonempty float32 or float64 batch with total_mass [B,1]")
        if like.is_cuda and like.dtype != torch.float32:
            raise ValueError("CUDA payload batches use float32")
        for field in fields(self):
            value = getattr(self, field.name)
            width = {"bag_half_extents": 3, "mass_weights": 4}.get(field.name, 1)
            if value.shape != (len(like), width) or value.dtype != like.dtype or value.device != like.device:
                raise ValueError(f"{field.name} must have shape [B,{width}] and match total_mass")
            if not (torch.isfinite(value) & (value >= 0.)).all():
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if not ((like > 0.).all() and (self.bag_half_extents > 0.).all() and (self.tau_rebound > 0.).all()):
            raise ValueError("Mass, dimensions, and rebound time must be positive")
        if not ((self.mass_weights >= .05).all() and
                torch.allclose(self.mass_weights.sum(dim=1), torch.ones_like(like[:, 0]), atol=1e-6, rtol=0.)):
            raise ValueError("Mass weights must add up to one and be at least 0.05")
        if ((self.max_indent_ratio >= 1.) | (self.fill_ratio > 1.) |
                (self.theta_max > torch.pi) | (self.internal_damping > 1.)).any():
            raise ValueError("Indentation must leave positive thickness, ratios must be within [0,1], and angles within [0,π]")


class PayloadState(NamedTuple):
    pos: Tensor     # [B, 4, 3]
    vel: Tensor     # [B, 4, 3]
    d_surf: Tensor  # [B, 4]


class PayloadStep(NamedTuple):
    state: PayloadState
    support_N: Tensor  # [B, 4]
    clearance: Tensor  # [B, 4]


class SurfaceStep(NamedTuple):
    depth: Tensor          # [B, 4]
    pad_force_N: Tensor    # [B, pads], upward on the pad
    block_force_N: Tensor  # [B, 4], downward on the payload
    core_force_N: Tensor   # [B, pads], included in pad_force_N


class UnifiedPayloadTemplate:
    def __init__(self, params: PayloadParamsX, dt: float = DT):
        if dt != DT:
            raise ValueError("The payload template uses a 2 ms step")
        params.validate()
        self.params = params
        self.dt = dt
        self.weights = params.mass_weights / params.mass_weights.sum(dim=1, keepdim=True)
        self.mass = params.total_mass * self.weights
        self.inv_mass = self.mass.reciprocal()
        self.half_height = params.bag_half_extents[:, 2:3]
        self.depth_max = 2. * self.half_height * params.max_indent_ratio
        self.slide = torch.minimum(params.slide_slack_max, params.cavity_slack(params.bag_half_extents, params.fill_ratio))
        self.rebound = torch.exp(-dt / params.tau_rebound)
        self.rest = params.total_mass.new_tensor([[-.5, -.5, 0.], [.5, -.5, 0.], [-.5, .5, 0.], [.5, .5, 0.]])
        self.rest = self.rest * params.bag_half_extents[:, None]
        self.up = self.rest.new_tensor([0., 0., 1.])
        self.right = self.rest.new_tensor([1., 0., 0.])
        self.bend_sign = self.rest.new_tensor([.5, -.5, -.5, .5])
        self.bend_delta = self.inv_mass * self.bend_sign
        self.bend_inverse_mass = self.inv_mass.sum(dim=1, keepdim=True) * .25
        self.bend_alpha = params.alpha_bend / dt**2
        self.stretch_alpha = params.alpha_stretch / dt**2
        self.diagonal = (self.rest[:, 3] - self.rest[:, 0]).norm(dim=-1, keepdim=True)
        self.bend_limit = self.diagonal * torch.sin(.5 * params.theta_max)
        # Each color contains two disjoint edges, so their corrections can be applied together.
        self.edge_groups = []
        for order in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
            indices = torch.tensor(order, device=self.rest.device)
            mass = self.inv_mass.index_select(1, indices).reshape(-1, 2, 2)
            pair = self.rest.index_select(1, indices).reshape(-1, 2, 2, 3)
            length = (pair[:, :, 0] - pair[:, :, 1]).norm(dim=-1)
            self.edge_groups.append((indices, indices.argsort(), mass, mass.sum(dim=-1), length))
        # Fixed SI scales for the privileged observation, in parameter order.
        scales = [[.19, .15, .08], 1., 2., 1., .05, 1.35, .01, .04, 1., .78, 500., 15., 25., 1.5, 1., 15000.]
        self.standard_params = torch.cat([getattr(params, f.name) / self.rest.new_tensor(scale)
                                         for f, scale in zip(fields(params), scales, strict=True)], dim=1)

    def initial_state(self):
        pos = self.rest + (self.half_height * self.up)[:, None]
        return PayloadState(pos, torch.zeros_like(pos), torch.zeros_like(pos[:, :, 2]))

    def com(self, state):
        return (state.pos * self.weights.unsqueeze(-1)).sum(dim=1)

    def com_velocity(self, state):
        return (state.vel * self.weights.unsqueeze(-1)).sum(dim=1)

    def bend_angle(self, state):
        offset = .5 * ((state.pos[:, 0] + state.pos[:, 3]) - (state.pos[:, 1] + state.pos[:, 2]))
        return 2. * torch.asin((offset.norm(dim=-1, keepdim=True) / self.diagonal).clamp(0., 1.))

    def half_thickness(self, depth):
        return self.half_height - .5 * depth

    def _edges(self, pos, group, multipliers):
        order, inverse, mass, mass_sum, rest = group
        pair = pos.index_select(1, order).reshape(-1, 2, 2, 3)
        delta = pair[:, :, 0] - pair[:, :, 1]
        length = delta.norm(dim=-1)
        error = length - rest
        c = error.sign() * (error.abs() - self.slide).clamp_min(0.)
        active = (self.slide == 0.) | (c != 0.)
        dl = (-c - self.stretch_alpha * multipliers) / (mass_sum + self.stretch_alpha)
        dl = torch.where(active, dl, 0.)
        correction = delta * (dl / length.clamp_min(EPS)).unsqueeze(-1)
        pair = torch.stack((pair[:, :, 0] + mass[:, :, :1] * correction,
                            pair[:, :, 1] - mass[:, :, 1:] * correction), dim=2)
        return pair.reshape_as(pos).index_select(1, inverse), torch.where(active, multipliers + dl, 0.)

    def _bend(self, pos, multiplier):
        delta = .5 * ((pos[:, 0] + pos[:, 3]) - (pos[:, 1] + pos[:, 2]))
        length = delta.norm(dim=-1, keepdim=True)
        soft = (-length - self.bend_alpha * multiplier) / (self.bend_inverse_mass + self.bend_alpha)
        after_soft = length + self.bend_inverse_mass * soft
        hard = -(after_soft - self.bend_limit).clamp_min(0.) / self.bend_inverse_mass
        direction = delta / length.clamp_min(EPS)
        return pos + ((soft + hard) * direction)[:, None] * self.bend_delta[:, :, None], multiplier + soft

    def _damp(self, pos, vel, group):
        order, inverse, mass, mass_sum, _ = group
        pair = pos.index_select(1, order).reshape(-1, 2, 2, 3)
        speed = vel.index_select(1, order).reshape_as(pair)
        direction = torch.nn.functional.normalize(pair[:, :, 0] - pair[:, :, 1], dim=-1, eps=EPS)
        relative = ((speed[:, :, 0] - speed[:, :, 1]) * direction).sum(dim=-1)
        impulse = (relative * self.params.internal_damping / mass_sum).unsqueeze(-1) * direction
        speed = torch.stack((speed[:, :, 0] - mass[:, :, :1] * impulse,
                             speed[:, :, 1] + mass[:, :, 1:] * impulse), dim=2)
        return speed.reshape_as(vel).index_select(1, inverse)

    def step(self, state: PayloadState, force: Tensor, depth: Tensor | None = None) -> PayloadStep:
        """Advance under world-frame per-block force [B,4,3] and optional new indentation."""
        if depth is None:
            depth = state.d_surf * self.rebound
        dt = self.dt
        half = self.half_thickness(depth)
        pos = state.pos + dt * state.vel + dt**2 * (force * self.inv_mass.unsqueeze(-1) - GRAVITY * self.up)
        edge_lambda = [torch.zeros_like(group[-1]) for group in self.edge_groups]
        bend_lambda = torch.zeros_like(self.half_height)
        normal_shift = torch.zeros_like(half)
        for _ in range(4):
            for i, group in enumerate(self.edge_groups):
                pos, edge_lambda[i] = self._edges(pos, group, edge_lambda[i])
            pos, bend_lambda = self._bend(pos, bend_lambda)
            shift = (half - pos[:, :, 2]).clamp_min(0.)
            pos = torch.cat((pos[:, :, :2], torch.maximum(pos[:, :, 2], half).unsqueeze(-1)), dim=-1)
            normal_shift = normal_shift + shift

        vel = (pos - state.pos) / dt
        for group in self.edge_groups:
            vel = self._damp(pos, vel, group)
        # Internal impulses may recreate inward normal speed at a supported block.
        floor = pos[:, :, 2] <= half
        normal_dv = torch.where(floor, -vel[:, :, 2].clamp_max(0.), 0.)
        speed_xy = vel[:, :, :2].norm(dim=-1)
        normal_dv_total = normal_shift / dt + normal_dv
        friction = (self.params.table_friction * normal_dv_total / speed_xy.clamp_min(EPS)).clamp_max(1.)
        vel = torch.cat((vel[:, :, :2] * (1. - friction.unsqueeze(-1)),
                         (vel[:, :, 2] + normal_dv).unsqueeze(-1)), dim=-1)
        return PayloadStep(PayloadState(pos, vel, depth), self.mass * normal_dv_total / dt, pos[:, :, 2] - half)

    def _normal(self, pos):
        a = pos[:, 3] - pos[:, 0]
        b = pos[:, 2] - pos[:, 1]
        cross = torch.linalg.cross(a, b)
        length = cross.norm(dim=-1, keepdim=True)
        normal = torch.where(length > EPS, cross / length.clamp_min(EPS), self.up)
        return normal, a, b, length

    def frame(self, state):
        normal, a, b, _ = self._normal(state.pos)
        x = a - b
        x = torch.where(x.norm(dim=-1, keepdim=True) > EPS,
                        torch.nn.functional.normalize(x, dim=-1, eps=EPS), self.right)
        return torch.stack((x, torch.linalg.cross(normal, x), normal), dim=-1)

    def weights_at(self, xy):
        """Bilinear weights for material coordinates on the block-center grid."""
        u, v = (xy / (.5 * self.params.bag_half_extents[:, None, :2])).clamp(-1., 1.).unbind(dim=-1)
        u, v = .5 * (u + 1.), .5 * (v + 1.)
        return torch.stack(((1. - u) * (1. - v), u * (1. - v), (1. - u) * v, u * v), dim=-1)

    @staticmethod
    def _surface_demand(tip_depth, weights, engaged):
        valid = engaged.unsqueeze(-1) & (weights > 0.)
        raw = torch.where(valid, tip_depth.unsqueeze(-1) * weights, -torch.inf)
        demand = torch.where(valid.any(dim=1), raw.amax(dim=1), 0.)
        return raw, demand

    def surface_contact(self, state, tip_depth, weights, engaged, end_demand=None):
        """Update under top pads using an optional predicted end-step block demand."""
        raw, demand = self._surface_demand(tip_depth, weights, engaged)
        if end_demand is None:
            end_demand = demand
        free = state.d_surf * self.rebound
        depth = torch.maximum(free, torch.minimum(end_demand, self.depth_max))
        p = self.params
        loading = (end_demand >= free) & (end_demand > 0.)
        old = state.d_surf
        # Secant of the compression potential between the two indentation states.
        ratio = (depth.square() + old.square()) / self.depth_max.clamp_min(EPS).square()
        elastic = p.k_soft * .5 * (depth + old) * (1. + .5 * p.hardening_gamma * ratio)
        # Unloading viscosity matches the free rebound at zero pad reaction.
        rebound_factor = (1. - (old - depth).clamp_min(0.) / (old - free).clamp_min(EPS)).clamp_min(0.)
        compress = loading * (elastic * rebound_factor + p.c_comp * ((depth - old) / self.dt).clamp_min(0.))
        # The core begins where the compressed top reaches its minimum height above the bottom.
        start_core = (demand - self.depth_max).clamp_min(0.)
        end_core = (end_demand - self.depth_max).clamp_min(0.)
        core = .5 * CORE_STIFFNESS * (start_core + end_core)
        sweep = (end_demand - demand).abs()
        core_secant = .5 * CORE_STIFFNESS * (end_core.square() - start_core.square()).abs()
        core = torch.where(sweep > EPS, core_secant / sweep.clamp_min(EPS), core)
        owners = ((raw == demand.unsqueeze(1)) & engaged.unsqueeze(-1)).to(raw.dtype)
        share = owners / owners.sum(dim=1, keepdim=True).clamp_min(1.)
        # Pull the block strain force back through its weighted pad displacement.
        pad_gradient = share * weights
        pad_force = (pad_gradient @ (compress + core).unsqueeze(-1)).squeeze(-1)
        block_force = (weights.transpose(1, 2) @ pad_force.unsqueeze(-1)).squeeze(-1)
        return SurfaceStep(depth, pad_force, block_force, (pad_gradient @ core.unsqueeze(-1)).squeeze(-1))

    def predict_surface_demand(self, state, pads_end, active):
        """Extrapolate the moving contact geometry without entering the floor."""
        pos = state.pos + self.dt * state.vel
        z = torch.maximum(pos[..., 2], self.half_thickness(state.d_surf))
        predicted = state._replace(pos=torch.cat((pos[..., :2], z.unsqueeze(-1)), dim=-1))
        frame = self.frame(predicted)
        local = (pads_end - predicted.pos.mean(dim=1, keepdim=True)) @ frame
        weights = self.weights_at(local[..., :2])
        separation = ((pads_end - weights @ predicted.pos) * frame[:, None, :, 2]).sum(dim=-1)
        return self._surface_demand(self.half_height - separation, weights, active)[1]

    def surface_core_energy(self, tip_depth, weights, engaged):
        """Recoverable energy of the unilateral core penalty at a boundary pose."""
        _, demand = self._surface_demand(tip_depth, weights, engaged)
        return .5 * CORE_STIFFNESS * (demand - self.depth_max).clamp_min(0.).square()

    def contact_geometry(self, depth, weights, feed, cosine, slack, *, pad_height=.020, pad_radius=.005):
        indentation = torch.bmm(weights, depth.unsqueeze(-1)).squeeze(-1)
        # The top segment of the cross-section gets longer if its two ends indent unevenly.
        width = 2. * self.params.bag_half_extents[:, :1]
        asymmetry = indentation[:, :1] - indentation[:, 1:]
        top_growth = torch.hypot(width, asymmetry) - width
        perimeter_drop = (indentation.sum(dim=1, keepdim=True) - top_growth).clamp_min(0.)
        return (feed + indentation.clamp_max(pad_height),
                (cosine + indentation / (indentation + pad_radius)).clamp(0., 1.),
                slack + torch.minimum(indentation.mean(dim=1, keepdim=True), perimeter_drop))

    def anchors(self, state, weights, depth_rate):
        normal, a, b, length = self._normal(state.pos)
        da = state.vel[:, 3] - state.vel[:, 0]
        db = state.vel[:, 2] - state.vel[:, 1]
        cross_rate = torch.linalg.cross(da, b) + torch.linalg.cross(a, db)
        normal_rate = (cross_rate - (cross_rate * normal).sum(dim=-1, keepdim=True) * normal) / length.clamp_min(EPS)
        normal_rate = torch.where(length > EPS, normal_rate, 0.)
        height = torch.bmm(weights, self.half_thickness(state.d_surf).unsqueeze(-1))
        height_rate = -.5 * torch.bmm(weights, depth_rate.unsqueeze(-1))
        return (torch.bmm(weights, state.pos) + height * normal[:, None],
                torch.bmm(weights, state.vel) + height * normal_rate[:, None] + height_rate * normal[:, None])

    def anchor_inverse_mass(self, state, weights, separation):
        """Absolute row sums of the axial mobility for all ropes on this payload."""
        direction = torch.nn.functional.normalize(separation, dim=-1, eps=EPS)
        normal, a, b, length = self._normal(state.pos)
        h = weights @ self.half_thickness(state.d_surf).unsqueeze(-1)
        projected = direction - (direction * normal[:, None]).sum(dim=-1, keepdim=True) * normal[:, None]
        adjoint = torch.where(length[:, None] > EPS, h * projected / length[:, None].clamp_min(EPS), 0.)
        ga = torch.linalg.cross(b[:, None].expand_as(adjoint), adjoint)
        gb = torch.linalg.cross(adjoint, a[:, None].expand_as(adjoint))
        jacobian = weights.unsqueeze(-1) * direction.unsqueeze(-2) + torch.stack((-ga, -gb, gb, ga), dim=-2)
        mobility = (jacobian * self.inv_mass[:, None, :, None]).flatten(2) @ jacobian.flatten(2).transpose(1, 2)
        return mobility.abs().sum(dim=-1)

    def pad_stop(self, state, weights, axis, pad_velocity, margin):
        """A compliant zero-restitution stop with no stored penetration spring."""
        gap = (margin / .0005).clamp(0., 1.)
        engage = gap.square() * (3. - 2. * gap)
        speed = ((pad_velocity - weights @ state.vel) * axis).sum(dim=-1).clamp_min(0.)
        # The row bound includes other pads pushing on the same blocks.
        row = (weights @ (weights.sum(dim=1) * self.inv_mass).unsqueeze(-1)).squeeze(-1)
        critical = 2. * torch.sqrt(.5 * self.params.lateral_crush_k / row.clamp_min(EPS))
        damping = critical * engage
        return damping * speed / (1. + self.dt * damping * row)

    def distribute_anchor_forces(self, state, weights, force):
        normal, a, b, length = self._normal(state.pos)
        height = torch.bmm(weights, self.half_thickness(state.d_surf).unsqueeze(-1))
        equivalent = (force * height).sum(dim=1)
        adjoint = (equivalent - (equivalent * normal).sum(dim=-1, keepdim=True) * normal) / length.clamp_min(EPS)
        adjoint = torch.where(length > EPS, adjoint, 0.)
        grad_a = torch.linalg.cross(b, adjoint)
        grad_b = torch.linalg.cross(adjoint, a)
        pos_force = torch.bmm(weights.transpose(1, 2), force) + torch.stack((-grad_a, -grad_b, grad_b, grad_a), dim=1)
        depth_force = -.5 * torch.bmm(weights.transpose(1, 2), (force * normal[:, None]).sum(dim=-1, keepdim=True))
        return pos_force, depth_force.squeeze(-1)

    def get_privileged_state(self, state: PayloadState) -> Tensor:
        pos = (state.pos - self.com(state)[:, None]) / self.params.bag_half_extents[:, None]
        vel = state.vel - self.com_velocity(state)[:, None]
        support = state.pos[:, :, 2] <= self.half_thickness(state.d_surf)
        return torch.cat((self.standard_params, pos.flatten(1), vel.flatten(1), self.bend_angle(state) / torch.pi,
                          state.d_surf / (2. * self.half_height), support.to(pos.dtype)), dim=1)
