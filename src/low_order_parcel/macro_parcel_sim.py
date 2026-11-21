"""Rigid or segmented parcel on a plane, loaded by two independent film ropes."""

import math
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from .lumped_contact_proxy import ContactInputs, EPS, LumpedContactProxy, ProxyParams
from .unified_payload import GRAVITY, PayloadState, UnifiedPayloadTemplate

PROBE_ACTION_CHANGE_S = 2.65


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


class ParcelState(NamedTuple):
    com: Tensor
    orientation: Tensor
    velocity: Tensor
    angular_velocity: Tensor
    contacts: Tensor


class PayloadParcelState(NamedTuple):
    com: Tensor
    orientation: Tensor
    velocity: Tensor
    angular_velocity: Tensor
    contacts: Tensor
    indentation_power: Tensor    # [B, 1], rope work rate on the prescribed indentation
    side_axis: Tensor            # [B, 3], horizontal approach direction latched at first push
    side_origin: Tensor          # [B, 3], parcel COM at first side contact
    payload: PayloadState


class ParcelAction(NamedTuple):
    pinch_position: Tensor  # [B, 2, 3]
    pinch_velocity: Tensor  # [B, 2, 3]
    normal_force: Tensor  # [B, 2]
    inward_velocity: Tensor  # [B, 2]
    feed_depth: Tensor  # [B, 2]
    alignment_cos: Tensor  # [B, 2]
    external_force: Tensor  # [B, 3], applied at the COM
    direct_press_N: Tensor  # [B, 1], downward resultant from the gripper


class ParcelStep(NamedTuple):
    state: ParcelState | PayloadParcelState
    wrist_wrench: Tensor  # [B, 6], world frame, about pad midpoint
    support_N: Tensor  # [B, 1]
    table_clearance: Tensor  # [B, 1], minimum box-to-plane separation
    tension: Tensor  # [B, 2]
    slip_rate: Tensor  # [B, 2]
    elastic_energy: Tensor  # [B, 2]
    elastic_pull: Tensor  # [B], vertical component of the spring forces before adding damping


class ProbeTrace(NamedTuple):
    elapsed_s: float
    channels: dict[str, np.ndarray | float]


class MacroParcelSim:
    def __init__(self, proxy: LumpedContactProxy, mass_kg: Tensor | None = None,
                 com_offset: Tensor | None = None, *, half_extents=(.075, .10, .016),
                 table_friction: float = .40, payload: UnifiedPayloadTemplate | None = None):
        """Mass and COM offset are [B,1] kg and [B,3] m in the parcel frame."""
        like = proxy.params.S_bag
        batch = like.shape[0]
        self.proxy = proxy
        self.dt = proxy.dt
        self.payload = payload
        self.up = like.new_tensor((0., 0., 1.)).expand(batch, 3)
        if payload is not None:
            if mass_kg is not None or com_offset is not None:
                raise ValueError("With a payload template, mass and COM come from PayloadParamsX")
            self.mass = payload.params.total_mass
            if (self.mass.shape != like.shape or self.mass.device != like.device or
                    self.mass.dtype != like.dtype or payload.dt != self.dt):
                raise ValueError("Payload and contact proxy must have the same batch, device, dtype and step")
            self.half_extents = payload.params.bag_half_extents
            # The calibrated mount coordinates are fixed material points on both sides.
            xy = like.new_tensor([[-.024, 0.], [.024, 0.]]).expand(batch, -1, -1)
            self.payload_mount_weights = payload.weights_at(xy)
            self.base_slack = torch.minimum(like, payload.params.fill_slack_capacity())
            return
        if mass_kg is None or com_offset is None:
            raise ValueError("Provide mass and COM offset for the rigid parcel")
        if mass_kg.shape != (batch, 1) or com_offset.shape != (batch, 3):
            raise ValueError("Expected mass [B,1] and COM offset [B,3]")
        if any(value.dtype != like.dtype or value.device != like.device for value in (mass_kg, com_offset)):
            raise ValueError("Mass and COM offset must match the contact parameter dtype and device")
        if (mass_kg <= 0).any() or not torch.isfinite(mass_kg).all() or not torch.isfinite(com_offset).all():
            raise ValueError("The parcel must have a finite positive mass and a finite COM offset")
        self.mass = mass_kg
        self.offset = com_offset
        self.half_extents = like.new_tensor(half_extents)
        if (self.half_extents <= 0).any() or (com_offset.abs() >= self.half_extents).any() or table_friction < 0.:
            raise ValueError("The parcel must enclose its COM and have nonnegative table friction")
        self.mu_table = table_friction
        hx, hy, hz = half_extents
        box_center_moi = like.new_tensor((hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2)) / 3.
        # Apply the parallel axis theorem to approximate a mass distribution centred on the offset COM.
        # Its inertia is the homogeneous-box inertia plus an eccentric payload's offset contribution.
        offset_sq = com_offset.unsqueeze(1) * com_offset.unsqueeze(2)
        box_moi = torch.diag(box_center_moi) + (offset_sq.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
                                               * torch.eye(3, dtype=like.dtype, device=like.device) - offset_sq)
        self.inertia = mass_kg.unsqueeze(-1) * box_moi
        self.inv_inertia = torch.linalg.inv(self.inertia)
        self.points = like.new_tensor([(x, y, -hz) for x in (-hx, hx) for y in (-hy, hy)]) - com_offset[:, None, :]
        self.mounts = like.new_tensor([(-.024, 0., hz), (.024, 0., hz)]) - com_offset[:, None, :]

    @classmethod
    def from_nominal(cls, nominal, dt, *, slack_m=None, com_offset=(0., 0., 0.),
                     half_extents=(.075, .10, .016)):
        values = dict(nominal["proxy"])
        if slack_m is not None:
            values["S_bag"] = slack_m
        params = ProxyParams.from_scalars(values, dtype=torch.float64)
        return cls(LumpedContactProxy(params, dt), params.S_bag.new_tensor([[nominal["mass_kg"]]]),
                   params.S_bag.new_tensor([com_offset]), half_extents=half_extents)

    def initial_state(self) -> ParcelState | PayloadParcelState:
        zero = torch.zeros_like(self.up)
        quat = torch.cat((torch.ones_like(zero[:, :1]), zero), dim=1)
        if self.payload is not None:
            state = self.payload.initial_state()
            com = self.payload.com(state)
            return PayloadParcelState(com, quat, zero, zero, self.proxy.initial_state(),
                                      torch.zeros_like(self.mass), zero, com, state)
        com = self.up * self.half_extents[2] + self.offset
        return ParcelState(com, quat, zero, zero, self.proxy.initial_state())

    def step(self, state: ParcelState | PayloadParcelState, action: ParcelAction) -> ParcelStep:
        if self.payload is not None:
            return self._step_payload(state, action)
        dt, q = self.dt, state.orientation
        mount_arm = rotate(q[:, None, :], self.mounts)
        attachment = state.com[:, None, :] + mount_arm
        attachment_velocity = state.velocity[:, None, :] + torch.linalg.cross(state.angular_velocity[:, None, :], mount_arm)
        contacts = self.proxy.step(state.contacts, ContactInputs(
            action.pinch_position, attachment, action.pinch_velocity,
            attachment_velocity, state.com, action.normal_force, action.inward_velocity,
            action.feed_depth, action.alignment_cos))

        force = contacts.force.sum(dim=1) + action.external_force - action.direct_press_N * self.up
        angular_momentum = self._inertia_action(q, self.inertia, state.angular_velocity)
        torque = contacts.torque.sum(dim=1) - torch.linalg.cross(state.angular_velocity, angular_momentum)
        velocity = state.velocity + dt * (force / self.mass - GRAVITY * self.up)
        omega = state.angular_velocity + dt * self._inertia_action(q, self.inv_inertia, torque)

        points = rotate(q[:, None, :], self.points)
        height = points[:, :, 2] + state.com[:, 2:3]
        clearance = height.amin(dim=1, keepdim=True)
        on_edge = height <= clearance + 1e-7
        arm = (points * on_edge.unsqueeze(-1)).sum(dim=1) / on_edge.sum(dim=1, keepdim=True)
        normal_eff = self._effective_inverse_mass(q, arm, self.up)
        point_velocity = velocity + torch.linalg.cross(omega, arm)
        impulse_n = (-clearance / dt - point_velocity[:, 2:3]).clamp_min(0.) / normal_eff
        dv, dw = self._impulse(q, arm, impulse_n * self.up)
        velocity, omega = velocity + dv, omega + dw

        point_velocity = velocity + torch.linalg.cross(omega, arm)
        tangential = point_velocity - point_velocity[:, 2:3] * self.up
        speed = torch.linalg.vector_norm(tangential, dim=1, keepdim=True)
        direction = tangential / speed.clamp_min(EPS)
        tangential_eff = self._effective_inverse_mass(q, arm, direction)
        impulse_t = torch.minimum(speed / tangential_eff, self.mu_table * impulse_n)
        dv, dw = self._impulse(q, arm, -impulse_t * direction)
        velocity, omega = velocity + dv, omega + dw

        qv = q[:, 1:]
        q_dot = torch.cat((-(omega * qv).sum(dim=1, keepdim=True), q[:, :1] * omega + torch.linalg.cross(omega, qv)), dim=1)
        orientation = torch.nn.functional.normalize(q + .5 * dt * q_dot, dim=1)
        com = state.com + velocity * dt
        bottom = rotate(orientation[:, None, :], self.points)[:, :, 2].amin(dim=1, keepdim=True)
        clearance_next = com[:, 2:3] + bottom
        com = com + (-clearance_next).clamp_min(0.) * self.up

        # The sensed loads are opposite to the forces transmitted to the parcel.
        # Direct symmetric top-press forces have no moment about the pad midpoint.
        pad_arm = action.pinch_position - action.pinch_position.mean(dim=1, keepdim=True)
        wrist_force = -contacts.force.sum(dim=1) + action.direct_press_N * self.up
        wrist_torque = torch.linalg.cross(pad_arm, -contacts.force).sum(dim=1)
        separation = action.pinch_position - attachment
        cos_vertical = separation[:, :, 2] / torch.linalg.vector_norm(separation, dim=-1).clamp_min(EPS)
        elastic_pull = (contacts.elastic_tension * cos_vertical).sum(dim=1)
        return ParcelStep(ParcelState(com, orientation, velocity, omega, contacts.state),
                          torch.cat((wrist_force, wrist_torque), dim=1), impulse_n / dt, clearance_next.clamp_min(0.),
                          contacts.tension, contacts.slip_rate, contacts.elastic_energy, elastic_pull)

    def _step_payload(self, state: PayloadParcelState, action: ParcelAction) -> ParcelStep:
        payload = self.payload
        current = state.payload
        weights = self.payload_mount_weights
        frame = payload.frame(current)
        tip = action.pinch_position
        midpoint = tip.mean(dim=1, keepdim=True)
        local_tip = torch.bmm(tip - current.pos.mean(dim=1, keepdim=True), frame)
        tip_weights = payload.weights_at(local_tip[:, :, :2])
        side_axis, side_origin, side_load, side_at_pad = self._side_contact(state, action, frame, local_tip, tip_weights)
        normal = frame[:, :, 2]
        center = torch.bmm(tip_weights, current.pos)
        normal_gap = ((tip - center) * normal[:, None]).sum(dim=-1)
        tip_depth = payload.half_height - normal_gap
        in_footprint = (local_tip[:, :, :2].abs() <= self.half_extents[:, None, :2]).all(dim=-1)
        previous_gap = normal_gap - self.dt * (action.pinch_velocity * normal[:, None]).sum(dim=-1)
        previous_indent = torch.bmm(tip_weights, current.d_surf.unsqueeze(-1)).squeeze(-1)
        top = ((previous_gap >= 0.) | (previous_indent > EPS) |
               (action.direct_press_N > 0.) | (tip[:, :, 2] <= 0.))
        surface = payload.surface_contact(current, tip_depth, tip_weights, in_footprint & top & ~side_at_pad)
        depth_rate = (surface.depth - current.d_surf) / self.dt
        current = current._replace(d_surf=surface.depth)
        anchor, anchor_vel = payload.anchors(current, weights, depth_rate)
        push_displacement = ((state.com - side_origin) * side_axis).sum(dim=1, keepdim=True)
        net_feed = (action.feed_depth - push_displacement).clamp_min(0.)
        feed, alignment, budget = payload.contact_geometry(current.d_surf, weights, net_feed,
                                                           action.alignment_cos, self.base_slack)

        jaw = tip[:, 1] - tip[:, 0]
        span = jaw.norm(dim=1, keepdim=True)
        axis = jaw / span.clamp_min(EPS)
        local_axis = torch.bmm(axis.unsqueeze(1), frame).squeeze(1)
        width = 2. * (local_axis[:, :2].abs() * self.half_extents[:, :2]).sum(dim=1, keepdim=True)
        half = torch.bmm(tip_weights, payload.half_thickness(current.d_surf).unsqueeze(-1)).squeeze(-1)
        z_in_slab = (local_tip[:, :, 2].abs() < half).all(dim=1, keepdim=True)
        bridge = in_footprint.any(dim=1, keepdim=True) & z_in_slab & (span > EPS)
        crush = .5 * payload.params.lateral_crush_k * (width - span).clamp_min(0.) * bridge
        jaw_load = crush[:, :, None] * torch.stack((axis, -axis), dim=1)

        contacts = self.proxy.step(state.contacts, ContactInputs(
            tip, anchor, action.pinch_velocity, anchor_vel, state.com,
            action.normal_force + crush, action.inward_velocity, feed, alignment, budget))
        force, depth_force = payload.distribute_anchor_forces(current, weights, contacts.force)
        indentation_power = (depth_force * depth_rate).sum(dim=1, keepdim=True)
        surface_vertical = (surface.pad_force_N * normal[:, 2:3]).clamp_min(0.)
        direct = (.5 * action.direct_press_N - surface_vertical).clamp_min(0.)
        extra_press = torch.bmm(tip_weights.transpose(1, 2), direct.unsqueeze(-1))
        force = (force + action.external_force[:, None] * payload.weights.unsqueeze(-1)
                 + torch.bmm(tip_weights.transpose(1, 2), jaw_load + side_load)
                 - surface.block_force_N.unsqueeze(-1) * normal[:, None] - extra_press * self.up[:, None])
        advanced = payload.step(state.payload, force, surface.depth)

        next_payload = advanced.state
        orientation = _orientation(payload.frame(next_payload))
        old = state.orientation
        delta_vector = old[:, :1] * orientation[:, 1:] - orientation[:, :1] * old[:, 1:]
        delta_vector += torch.linalg.cross(old[:, 1:], orientation[:, 1:])
        omega = torch.where((old * orientation).sum(dim=1, keepdim=True) >= 0., delta_vector, -delta_vector)
        next_state = PayloadParcelState(payload.com(next_payload), orientation, payload.com_velocity(next_payload),
                                        2. * omega / self.dt, contacts.state, indentation_power,
                                        side_axis, side_origin, next_payload)
        pad_reaction = (-contacts.force - jaw_load - side_load + surface.pad_force_N.unsqueeze(-1) * normal[:, None]
                        + direct.unsqueeze(-1) * self.up[:, None])
        wrench = torch.cat((pad_reaction.sum(dim=1), torch.linalg.cross(tip - midpoint, pad_reaction).sum(dim=1)), dim=1)
        line = tip - anchor
        vertical = line[:, :, 2] / line.norm(dim=-1).clamp_min(EPS)
        elastic_pull = (contacts.elastic_tension * vertical).sum(dim=1)
        return ParcelStep(next_state, wrench, advanced.support_N.sum(dim=1, keepdim=True),
                          advanced.clearance.amin(dim=1, keepdim=True), contacts.tension,
                          contacts.slip_rate, contacts.elastic_energy, elastic_pull)

    def _side_contact(self, state, action, frame, local_tip, weights):
        """Unilateral side-wall pressure and translation relative to the first entry."""
        payload = self.payload
        velocity = action.pinch_velocity.mean(dim=1)
        horizontal = velocity - velocity[:, 2:3] * self.up
        speed = horizontal.norm(dim=1, keepdim=True)
        retreat = ((horizontal * state.side_axis).sum(dim=1, keepdim=True) < -1e-4)
        retreat = retreat & (action.feed_depth <= 0.).all(dim=1, keepdim=True)
        latched = (state.side_axis.abs().sum(dim=1, keepdim=True) > 0.) & ~retreat
        candidate = horizontal / speed.clamp_min(EPS)
        direction = torch.where(latched, state.side_axis, candidate)
        transverse = torch.linalg.cross(self.up, direction)
        axes = torch.bmm(torch.stack((direction, transverse), dim=1), frame)
        extents = (axes[:, :, :2].abs() * self.half_extents[:, None, :2]).sum(dim=2)
        longitudinal, lateral = torch.bmm(local_tip, axes.transpose(1, 2)).unbind(dim=2)
        penetration = longitudinal + extents[:, :1]
        thickness = payload.half_thickness(state.payload.d_surf)
        pad_half = torch.bmm(weights, thickness.unsqueeze(-1)).squeeze(-1)
        supported = (state.payload.pos[:, :, 2] <= thickness + EPS).any(dim=1, keepdim=True)
        slab = (local_tip[:, :, 2].abs() < pad_half) & (lateral.abs() < extents[:, 1:])
        near_side = supported & slab & (longitudinal < 0.) & (penetration >= 0.)
        moving_in = (speed > velocity[:, 2:3].abs()) & (speed > 1e-4) & ~retreat
        entering = moving_in & near_side.any(dim=1, keepdim=True)
        active = latched | entering
        side = near_side & active
        axis = torch.where(active, direction, 0.)
        origin = torch.where(latched, state.side_origin, state.com)
        stiffness = .5 * payload.params.lateral_crush_k
        # Implicit scalar compliance limits the kick when the effective mass is small.
        gain = stiffness / (1. + 2. * self.dt**2 * stiffness / self.mass)
        load = gain * penetration.clamp_min(0.) * side
        return axis, origin, load[:, :, None] * axis[:, None], side

    def get_privileged_state(self, state: PayloadParcelState) -> Tensor:
        if self.payload is None:
            raise ValueError("Privileged payload state requires the template")
        return self.payload.get_privileged_state(state.payload)

    @staticmethod
    def _inertia_action(q, inertia, world):
        inverse_q = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
        local = rotate(inverse_q, world)
        return rotate(q, torch.bmm(inertia, local.unsqueeze(-1)).squeeze(-1))

    def _effective_inverse_mass(self, q, arm, direction):
        torque = torch.linalg.cross(arm, direction)
        return self.mass.reciprocal() + (torque * self._inertia_action(q, self.inv_inertia, torque)).sum(dim=1, keepdim=True)

    def _impulse(self, q, arm, impulse):
        return impulse / self.mass, self._inertia_action(q, self.inv_inertia, torch.linalg.cross(arm, impulse))

    def run_top_press(self, pressure_N: float) -> ProbeTrace:
        return self._run_probe("top", pressure_N=pressure_N)

    def run_side_grasp_and_hover(self, *, feed_depth_m=.015531134179, grip_N=6., duration_s=4.77) -> ProbeTrace:
        return self._run_probe("side", feed_depth_m=feed_depth_m, grip_N=grip_N, duration_s=duration_s)

    def run_controlled_release(self, *, grip_N=6., opening_span_m=.014, opening_speed_mps=.020) -> ProbeTrace:
        return self._run_probe("release", grip_N=grip_N, opening_span_m=opening_span_m,
                               opening_speed_mps=opening_speed_mps)

    def run_suspended_load_ramp(self, *, rate_N_s=5., grip_N=6., axis="vertical") -> ProbeTrace:
        if axis not in ("vertical", "tangential"):
            raise ValueError("axis must be vertical or tangential")
        return self._run_probe("ramp", rate_N_s=rate_N_s, grip_N=grip_N, axis=axis)

    def probe_actions(self, mode, *, feed_depth_m=.015531134179, grip_N=6., duration_s=4.77, pressure_N=.5,
                      opening_span_m=.014, opening_speed_mps=.020, rate_N_s=5., axis="vertical"):
        """Build the action trajectory used by both alignment and batched timing."""
        dt = self.dt
        change_at = PROBE_ACTION_CHANGE_S
        end_s = {"top": 1.5, "side": duration_s, "release": change_at + opening_span_m / opening_speed_mps + .5,
                 "ramp": change_at + 15. / rate_N_s + .25}[mode]
        times = torch.arange(math.ceil(end_s / dt), dtype=self.mass.dtype, device=self.mass.device)[:, None, None] * dt
        close_s = .36
        lift_start, lift_end = .50, 1.80
        depth = (feed_depth_m + self.mass.new_tensor([.00055, -.00055])).clamp_min(0.)
        p = self.proxy.params
        full_gather = (times / close_s).clamp(0., 1.)
        gap0 = .002
        gap = gap0 + (1. - full_gather) * depth.sum()
        # Partial insertion transmits only the portion of the close command supported by the fold.
        normal = grip_N * (.10 + .90 * (depth / .015).clamp(0., 1.).pow(8))
        normal = normal.expand(len(times), self.mass.shape[0], 2)
        inward = (times < close_s) * ((depth - .0042).clamp_min(0.) / close_s)
        applied = torch.zeros_like(times)
        press = torch.zeros_like(times)
        alignment_cos = torch.ones_like(normal)
        if mode == "top":
            # The test holds pressure while the pads sweep inward, then lifts away.
            press = pressure_N * (times < lift_start)
            normal = (press / 2.).expand_as(normal)
            alignment_cos = torch.zeros_like(normal)
            depth = torch.zeros_like(depth)
        if mode == "release":
            opening = (times - change_at).clamp_min(0.) * opening_speed_mps
            gap = gap + opening
            normal = normal * (1. - opening / opening_span_m).clamp_min(0.)
        if mode == "ramp":
            applied = ((times - change_at).clamp_min(0.) * rate_N_s).clamp_max(15.)

        shoulder_sq = (.024 - gap0 / 2.) ** 2
        hz = self.payload.half_height if self.payload is not None else self.half_extents[2]
        start_z = 2. * hz + (p.L0.square() - shoulder_sq).clamp_min(0.).sqrt()
        lift_progress = ((times - lift_start) / (lift_end - lift_start)).clamp(0., 1.)
        lift = .5 * (1. - torch.cos(math.pi * lift_progress))
        # Reserve travel for the load-ramp probe so a swinging, one-sided parcel
        # can detach without touching the table, as in the archived clear-ramp setup.
        lift_height = .25 if mode == "ramp" else .15
        lift_m = lift_height + self.mass * GRAVITY / (2. * p.k)
        z = start_z + lift * lift_m
        x = (gap / 2. * self.mass.new_tensor([-1., 1.])).expand(len(times), self.mass.shape[0], 2)
        pad_position = torch.stack((x, torch.zeros_like(z), z), dim=-1)
        pad_velocity = torch.cat(((pad_position[1:] - pad_position[:-1]) / dt, torch.zeros_like(pad_position[:1])))
        unit_load = self.mass.new_tensor((0., 0., -1.) if axis == "vertical" else (0., 1., 0.))
        batches = self.mass.shape[0]
        actions = ParcelAction(pad_position, pad_velocity, normal, inward.expand(-1, batches, 2),
                               depth.expand(len(times), batches, 2), alignment_cos,
                               (applied * unit_load).expand(-1, batches, -1), press.expand(-1, batches, -1))
        return times, gap, applied, actions

    def _run_probe(self, mode, **options):
        start = perf_counter()
        state = self.initial_state()
        times, gap, applied, actions = self.probe_actions(mode, **options)
        clearance, support, com, speed, tilt, wrench, tension, contacts, rates, energy, elastic_pull = (
            [] for _ in range(11))
        for i in range(len(times)):
            result = self.step(state, ParcelAction(*(channel[i] for channel in actions)))
            state = result.state
            clearance.append(result.table_clearance)
            support.append(result.support_N)
            com.append(state.com)
            speed.append(state.velocity)
            tilt.append(state.orientation)
            wrench.append(result.wrist_wrench)
            tension.append(result.tension)
            contacts.append(state.contacts)
            rates.append(result.slip_rate)
            energy.append(result.elastic_energy)
            elastic_pull.append(result.elastic_pull)

        channels = dict(time_s=times[:, 0, 0], gap_m=gap[:, 0, 0], external_load_N=applied[:, 0, 0],
                        normal_force_N=actions.normal_force[:, 0], pad_position_m=actions.pinch_position[:, 0],
                        table_clearance_m=torch.stack(clearance)[:, 0, 0], support_N=torch.stack(support)[:, 0, 0],
                        com_m=torch.stack(com)[:, 0], velocity_m_s=torch.stack(speed)[:, 0],
                        orientation=torch.stack(tilt)[:, 0], wrench=torch.stack(wrench)[:, 0],
                        tension_N=torch.stack(tension)[:, 0], contact_state=torch.stack(contacts)[:, 0],
                        slip_rate_mps=torch.stack(rates)[:, 0], elastic_energy_J=torch.stack(energy)[:, 0],
                        elastic_pull_N=torch.stack(elastic_pull)[:, 0])
        traces = {key: value.detach().cpu().numpy() for key, value in channels.items()}
        traces["action_change_s"] = PROBE_ACTION_CHANGE_S
        return ProbeTrace(perf_counter() - start, traces)
