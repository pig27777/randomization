"""Two frozen finger proxies acting on one four-block parcel (SI, 2 ms)."""

from typing import NamedTuple

import torch
from torch import Tensor

from .lumped_contact_proxy import ContactInputs, ContactState, EPS, LumpedContactProxy
from .macro_parcel_sim import ParcelAction
from .neck_damping import NeckDamping, NeckDampingParams
from .pad_torsion import PadTorsion, PadTorsionParams, TorsionState
from .unified_payload import DT, PayloadState, UnifiedPayloadTemplate


class DualArmAction(NamedTuple):
    arm_A: ParcelAction
    arm_B: ParcelAction
    enabled: Tensor  # [B,2] bool, one column per arm
    side_feed: Tensor | None = None  # [B,2] bool, planned side feed can clear the wall


class DualArmState(NamedTuple):
    payload: PayloadState
    contacts_A: Tensor  # [B,2,6]
    contacts_B: Tensor  # [B,2,6]
    torn_flag: Tensor  # [B,arm,finger] bool, latched until reset
    side_axis: Tensor  # [B,arm,3]
    side_origin: Tensor  # [B,arm,3]
    contact_reference: Tensor | None = None  # [B,arm,finger]
    torsion: TorsionState | tuple[()] = ()


class DualArmStep(NamedTuple):
    state: DualArmState
    wrist_wrench: Tensor  # [B,arm,6], world frame, about the pad midpoint
    tension: Tensor  # [B,arm,finger], N
    support_N: Tensor  # [B,1]
    table_clearance: Tensor  # [B,1]
    S_bag_total_eff: Tensor  # [B,1], m
    S_bag_available: Tensor  # [B,arm], before the simultaneous proxy step
    net_feed: Tensor  # [B,arm,finger], before adding indentation relief
    press_N: Tensor  # [B,arm], world-vertical top contact resultant


class DualArmParcelSim:
    def __init__(self, proxy_A: LumpedContactProxy, proxy_B: LumpedContactProxy,
                 payload: UnifiedPayloadTemplate, *, F_half: Tensor | float = 0.,
                 T_tear: Tensor | float | None = None, mount_xy: Tensor | None = None,
                 generator: torch.Generator | None = None, torsion: PadTorsionParams | None = None,
                 neck_damping: NeckDampingParams | None = None):
        like = payload.params.total_mass
        if like.dtype != torch.float32 or any(
                proxy.dt != DT or proxy.params.S_bag.shape != like.shape
                or proxy.params.S_bag.dtype != like.dtype or proxy.params.S_bag.device != like.device
                for proxy in (proxy_A, proxy_B)):
            raise ValueError("Use matching float32 proxies and payload on the same device at 2 ms")
        self.proxy_A, self.proxy_B, self.payload, self.dt = proxy_A, proxy_B, payload, DT
        self.mass = like
        self.base_slack = torch.minimum(
            payload.params.fill_slack_capacity(),
            torch.minimum(proxy_A.params.S_bag, proxy_B.params.S_bag))
        self.F_half = self._batch_parameter(F_half, "F_half", 0., float("inf"))
        self.generator = generator
        self.T_tear = self.sample_tear() if T_tear is None else self._batch_parameter(
            T_tear, "T_tear", 20., 60.)
        if mount_xy is None:
            mount_xy = like.new_tensor([[[-.024, 0.], [.024, 0.]],
                                         [[-.024, .04], [.024, .04]]]).expand(len(like), -1, -1, -1)
        if (mount_xy.shape != (len(like), 2, 2, 2) or mount_xy.dtype != like.dtype
                or mount_xy.device != like.device or not torch.isfinite(mount_xy).all()):
            raise ValueError("mount_xy must be finite [B,arm,finger,xy] in payload material coordinates")
        self.mount_weights = payload.weights_at(mount_xy.flatten(1, 2))
        self.min_press = torch.stack((proxy_A.params.F_min, proxy_B.params.F_min), dim=1)
        self.nominal_reference = torch.stack((proxy_A.params.L0, proxy_B.params.L0), dim=1)
        mu = torch.stack((proxy_A.params.mu_eff, proxy_B.params.mu_eff), dim=1)
        self.torsion = None if torsion is None else PadTorsion(like, mu, self.dt, torsion)
        self.neck = None if neck_damping is None else NeckDamping(like, mu, self.dt, neck_damping)

    def _batch_parameter(self, value, name, low, high):
        value = torch.as_tensor(value, dtype=self.mass.dtype, device=self.mass.device)
        if value.ndim == 0:
            value = value.expand_as(self.mass)
        if value.shape != self.mass.shape or not (value.isfinite() & (value >= low) & (value <= high)).all():
            raise ValueError(f"{name} must be a scalar or [B,1] in [{low}, {high}]")
        return value

    def sample_tear(self) -> Tensor:
        return 20. + 40. * torch.rand(self.mass.shape, device=self.mass.device, dtype=self.mass.dtype,
                                      generator=self.generator)

    def initial_state(self) -> DualArmState:
        payload = self.payload.initial_state()
        origin = self.payload.com(payload)[:, None].expand(-1, 2, -1)
        torsion = () if self.torsion is None else self.torsion.initial_state()
        return DualArmState(payload, self.proxy_A.initial_state(), self.proxy_B.initial_state(),
                            self.mass.new_zeros((len(self.mass), 2, 2), dtype=torch.bool),
                            torch.zeros_like(origin), origin, self.nominal_reference.clone(), torsion)

    def reset(self, state: DualArmState, reset_mask: Tensor, *, resample_tear: bool = True) -> DualArmState:
        """Reset selected environments at an episode boundary. Pass False for fixed corners."""
        fresh = self.initial_state()

        def select(new, old):
            mask = reset_mask.reshape(-1, *([1] * (old.ndim - 1)))
            return torch.where(mask, new, old)

        if resample_tear:
            self.T_tear = select(self.sample_tear(), self.T_tear)
        reference = self.nominal_reference if state.contact_reference is None else state.contact_reference
        torsion = fresh.torsion
        if torsion and state.torsion:
            torsion = TorsionState(*(select(n, o) for n, o in zip(torsion, state.torsion, strict=True)))
        return DualArmState(PayloadState(*(select(n, o) for n, o in zip(fresh.payload, state.payload))),
                            *(select(n, o) for n, o in zip(fresh[1:6], state[1:6], strict=True)),
                            select(fresh.contact_reference, reference), torsion)

    def get_privileged_state(self, state: DualArmState) -> Tensor:
        return self.payload.get_privileged_state(state.payload)

    def _side_contact(self, state, center, local_tip, frame, tip_weights, pad_velocity, feed, enabled):
        """Use unilateral side pressure both on the table and in the air."""
        payload = self.payload
        velocity = pad_velocity.mean(dim=2)
        horizontal = velocity[..., :2]
        speed = horizontal.norm(dim=-1, keepdim=True)
        retreat = ((horizontal * state.side_axis[..., :2]).sum(dim=-1, keepdim=True) < -1e-4)
        retreat = retreat & (feed <= 0.).all(dim=-1, keepdim=True)
        latched = (state.side_axis.abs().sum(dim=-1, keepdim=True) > 0.) & ~retreat
        candidate = torch.cat((horizontal / speed.clamp_min(EPS), torch.zeros_like(speed)), dim=-1)
        direction = torch.where(latched, state.side_axis, candidate)
        transverse = torch.linalg.cross(payload.up.expand_as(direction), direction)
        axes = torch.matmul(torch.stack((direction, transverse), dim=2), frame[:, None])
        extents = (axes[..., :2].abs() * payload.params.bag_half_extents[:, None, None, :2]).sum(dim=-1)
        longitudinal, lateral = (local_tip @ axes.transpose(-1, -2)).unbind(dim=-1)
        penetration = longitudinal + extents[..., :1]
        half = (tip_weights @ payload.half_thickness(state.payload.d_surf)[:, None, :, None]).squeeze(-1)
        clip = torch.minimum(half - local_tip[..., 2].abs(), extents[..., 1:] - lateral.abs())
        margin = torch.minimum(torch.minimum(clip, -longitudinal), penetration)
        near_side = margin >= -1e-6
        moving_in = (speed > velocity[..., 2:].abs()) & (speed > 1e-4) & ~retreat
        active = (latched | (moving_in & near_side.any(dim=-1, keepdim=True))) & enabled
        side = near_side & active
        axis = torch.where(active, direction, 0.)
        origin = torch.where(latched & enabled, state.side_origin, center[:, None])
        pad_axis = axis[:, :, None].expand_as(pad_velocity)
        load = payload.pad_stop(state.payload, tip_weights.flatten(1, 2), pad_axis.flatten(1, 2),
                                pad_velocity.flatten(1, 2), torch.where(side, margin, 0.).flatten(1))
        low_approach = enabled & moving_in & (clip > 0.) & (longitudinal < 0.)
        depth = torch.where(active | low_approach, penetration.clamp_min(0.), torch.inf)
        return axis, origin, load.unflatten(1, (2, 2)).unsqueeze(-1) * pad_axis, side, depth

    def step(self, state: DualArmState, action: DualArmAction, *,
             jaw_orientation: Tensor | None = None, jaw_angular_velocity: Tensor | None = None,
             pad_reaction_out: Tensor | None = None, neck_port_mobility=None,
             map_top_pad_load: bool = False, proxy_normal_out: Tensor | None = None) -> DualArmStep:
        """World jaw pose/rate, with optional [B,arm,pad,xyz] articulation reaction buffer."""
        payload, current = self.payload, state.payload
        arm_A, arm_B, enabled, side_feed = action
        enabled = enabled.unsqueeze(-1)
        tip, vel, normal, inward, feed, align, external, direct = (
            torch.stack(pair, dim=1) for pair in zip(arm_A, arm_B))
        pad_enabled = enabled & ~state.torn_flag
        direct = torch.where(enabled, direct, 0.)
        frame = payload.frame(current)
        center = payload.com(current)
        local_tip = (tip - current.pos.mean(dim=1)[:, None, None]) @ frame[:, None]
        tip_weights = payload.weights_at(local_tip.flatten(1, 2)[..., :2]).unflatten(1, (2, 2))
        side_axis, side_origin, side_load, side_pad, inward_margin = self._side_contact(
            state, center, local_tip, frame, tip_weights, vel, feed, enabled)
        up = frame[:, :, 2]
        pad_center = tip_weights @ current.pos[:, None]
        normal_gap = ((tip - pad_center) * up[:, None, None]).sum(dim=-1)
        tip_depth = payload.half_height[:, None] - normal_gap
        in_footprint = (local_tip[..., :2].abs() <= payload.params.bag_half_extents[:, None, None, :2]).all(dim=-1)
        previous_gap = normal_gap - self.dt * (vel * up[:, None, None]).sum(dim=-1)
        previous_indent = (tip_weights @ current.d_surf[:, None, :, None]).squeeze(-1)
        top = ((previous_gap >= 0.) | (previous_indent > EPS) | (direct > 0.) | (tip[..., 2] <= 0.))
        engaged = in_footprint & top & ~side_pad & enabled
        demand = payload.predict_surface_demand(current, (tip + self.dt * vel).flatten(1, 2), engaged.flatten(1))
        surface = payload.surface_contact(current, tip_depth.flatten(1), tip_weights.flatten(1, 2),
                                          engaged.flatten(1), demand)
        depth_rate = (surface.depth - current.d_surf) / self.dt
        compressed = current._replace(d_surf=surface.depth)
        anchors, anchor_vel = payload.anchors(compressed, self.mount_weights, depth_rate)
        anchors, anchor_vel = (item.unflatten(1, (2, 2)) for item in (anchors, anchor_vel))
        displacement = ((center[:, None] - side_origin) * side_axis).sum(dim=-1, keepdim=True)
        net_feed = torch.minimum((feed - displacement).clamp_min(0.), inward_margin)
        feed_A, align_A, budget_A = payload.contact_geometry(
            surface.depth, self.mount_weights[:, :2], net_feed[:, 0], align[:, 0], self.base_slack)
        feed_B, align_B, budget_B = payload.contact_geometry(
            surface.depth, self.mount_weights[:, 2:], net_feed[:, 1], align[:, 1], self.base_slack)
        # Both arms estimate reserve from the same deformed film. Counting both would duplicate relief.
        estimates = torch.cat((budget_A, budget_B), dim=1)
        budget = torch.where(enabled.squeeze(-1), estimates, self.base_slack).amax(dim=1, keepdim=True)

        jaw = tip[:, :, 1] - tip[:, :, 0]
        span = jaw.norm(dim=-1, keepdim=True)
        axis = jaw / span.clamp_min(EPS)
        local_axis = (axis @ frame)[..., :2]
        width = 2. * (local_axis.abs() * payload.params.bag_half_extents[:, None, :2]).sum(-1, keepdim=True)
        half = (tip_weights @ payload.half_thickness(surface.depth)[:, None, :, None]).squeeze(-1)
        # Pads on the entry wall are still outside the parcel. Applying a transverse
        # crush to that same contact would drive the blocks apart across the jaw.
        footprint = (payload.params.bag_half_extents[:, None, None, :2] -
                     local_tip[..., :2].abs()).amin(-1).amax(-1, keepdim=True)
        slab = (half - local_tip[..., 2].abs()).amin(-1, keepdim=True)
        gap = torch.minimum(width - span, torch.minimum(slab, footprint))
        jaw_margin = torch.where(~side_pad.any(dim=-1, keepdim=True) & (span > EPS) & enabled, gap, 0.)
        jaw_axis = torch.stack((axis, -axis), dim=2)
        load = payload.pad_stop(compressed, tip_weights.flatten(1, 2), jaw_axis.flatten(1, 2),
                                vel.flatten(1, 2), jaw_margin.expand_as(normal).flatten(1)).unflatten(1, (2, 2))
        available_force = normal.clamp_min(0.)
        load = available_force * torch.tanh(load / available_force.clamp_min(EPS))
        jaw_load = load.unsqueeze(-1) * jaw_axis
        surface_pad = surface.pad_force_N.unflatten(1, (2, 2))
        surface_vertical = (surface_pad * up[:, None, 2:3]).clamp_min(0.)
        extra_press = (.5 * direct - surface_vertical).clamp_min(0.)
        press = (surface_vertical + extra_press).sum(dim=-1, keepdim=True)
        # The side wall/closing load provides compression when feeding horizontally.
        side = side_axis.abs().any(dim=-1, keepdim=True)
        if side_feed is not None:
            side = side | side_feed.unsqueeze(-1)
        normal = torch.where(pad_enabled, normal, 0.)
        if map_top_pad_load:
            local_indent = (tip_weights @ surface.depth[:, None, :, None]).squeeze(-1)
            closing = (inward > 0.).all(dim=-1, keepdim=True)
            soft = payload.params.max_indent_ratio.unsqueeze(-1) > 0.
            top_loaded = closing & soft & ~side & engaged & pad_enabled & (local_indent > EPS)
            # The measured top normal is already included in the wrist reaction.
            # Use it only as proxy traction capacity, after computing the jaw load.
            normal = torch.maximum(normal, torch.where(top_loaded, surface_vertical, 0.))
        if proxy_normal_out is not None:
            proxy_normal_out.copy_(normal)
        compression = torch.where(side, normal, press)
        excess = (compression - self.min_press).clamp_min(0.)
        half_force = self.F_half[:, None]
        denominator = torch.where(half_force > 0., excess + half_force, 1.)
        gain = torch.where(half_force > 0., excess / denominator, 1.)
        effective_inward = torch.where(pad_enabled, gain * inward, 0.)

        prior = torch.stack((state.contacts_A, state.contacts_B), dim=1)
        prior = torch.where(pad_enabled.unsqueeze(-1), prior, 0.)
        held = prior[..., ContactState.SLACK].sum(dim=-1)
        available = (budget - held.flip(dims=(1,))).clamp_min(0.)
        inputs_A = ContactInputs(tip[:, 0], anchors[:, 0], vel[:, 0], anchor_vel[:, 0], center,
                                 normal[:, 0], effective_inward[:, 0], feed_A, align_A, available[:, :1])
        inputs_B = ContactInputs(tip[:, 1], anchors[:, 1], vel[:, 1], anchor_vel[:, 1], center,
                                 normal[:, 1], effective_inward[:, 1], feed_B, align_B, available[:, 1:])
        reference = self.nominal_reference if state.contact_reference is None else state.contact_reference
        separation = torch.where(pad_enabled.unsqueeze(-1), tip - anchors, 0.).flatten(1, 2)
        inverse_mass = payload.anchor_inverse_mass(compressed, self.mount_weights, separation).unflatten(1, (2, 2))
        result_A, _ = self.proxy_A.step_with_reference(prior[:, 0], inputs_A, reference[:, 0], inverse_mass[:, 0])
        result_B, _ = self.proxy_B.step_with_reference(prior[:, 1], inputs_B, reference[:, 1], inverse_mass[:, 1])
        claim_A = (result_A.state[..., ContactState.SLACK] + self.dt * result_A.slip_rate).sum(dim=1)
        claim_B = (result_B.state[..., ContactState.SLACK] + self.dt * result_B.slip_rate).sum(dim=1)
        claims = torch.stack((claim_A, claim_B), dim=1)

        # Simultaneous claims include film that would be withdrawn by slip in this step.
        # Reserve existing slack first. Recompute capture and tension with the resolved cap.
        retained = torch.minimum(claims, held)
        growth = claims - retained
        retained = retained * (budget / retained.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_max(1.)
        free = (budget - retained.sum(dim=1, keepdim=True)).clamp_min(0.)
        reserved = retained + growth * (free / growth.sum(dim=1, keepdim=True).clamp_min(EPS)).clamp_max(1.)
        capped = torch.where(claims.sum(dim=1, keepdim=True) > budget,
                             torch.minimum(available, reserved), available)
        result_A, reference_A = self.proxy_A.step_with_reference(
            prior[:, 0], inputs_A._replace(S_bag_eff=capped[:, :1]), reference[:, 0], inverse_mass[:, 0])
        result_B, reference_B = self.proxy_B.step_with_reference(
            prior[:, 1], inputs_B._replace(S_bag_eff=capped[:, 1:]), reference[:, 1], inverse_mass[:, 1])
        tension = torch.stack((result_A.tension, result_B.tension), dim=1)
        torn = state.torn_flag | (tension > self.T_tear[:, None])
        live = pad_enabled & ~torn
        next_contacts = torch.stack((result_A.state, result_B.state), dim=1)
        next_contacts = torch.where(live.unsqueeze(-1), next_contacts, 0.)
        tension = torch.where(live, tension, 0.)
        rope_force = torch.stack((result_A.force, result_B.force), dim=1)
        rope_force = torch.where(live.unsqueeze(-1), rope_force, 0.)
        neck_torque = None
        if self.neck is not None and self.neck.enabled:
            pad_rate = None if jaw_angular_velocity is None else jaw_angular_velocity.repeat_interleave(2, dim=1)
            neck = self.neck.payload(payload, compressed, self.mount_weights, tip.flatten(1, 2),
                                     anchors.flatten(1, 2), vel.flatten(1, 2), anchor_vel.flatten(1, 2),
                                     prior.flatten(1, 2), next_contacts.flatten(1, 2), normal.flatten(1), pad_rate,
                                     neck_port_mobility)
            rope_force = rope_force + neck.force.unflatten(1, (2, 2))
            neck_torque = neck.pad_couple.unflatten(1, (2, 2)).sum(dim=2)
        rope_load, _ = payload.distribute_anchor_forces(compressed, self.mount_weights, rope_force.flatten(1, 2))
        up = up[:, None]
        contact_load = jaw_load + side_load - extra_press.unsqueeze(-1) * payload.up
        block_load = tip_weights.flatten(1, 2).transpose(1, 2).bmm(contact_load.flatten(1, 2))
        external = torch.where(enabled, external, 0.).sum(dim=1)
        force = (rope_load + block_load + external[:, None] * payload.weights.unsqueeze(-1)
                 - surface.block_force_N.unsqueeze(-1) * up)
        torsion = ()
        if self.torsion is not None:
            pad_torque, couple, torsion = self.torsion.couples(
                payload, current, frame, state.torsion, prior, next_contacts, normal, tip - anchors, force,
                jaw_orientation, jaw_angular_velocity)
            force = force + couple
        advanced = payload.step(current, force, surface.depth)

        reaction = (-rope_force - contact_load + surface_pad.unsqueeze(-1) * up[:, None])
        if pad_reaction_out is not None:
            pad_reaction_out.copy_(reaction)
        torque = torch.linalg.cross(tip - tip.mean(dim=2, keepdim=True), reaction).sum(dim=2)
        if neck_torque is not None:
            torque = torque + neck_torque
        if self.torsion is not None:
            torque = torque - pad_torque
        wrench = torch.cat((reaction.sum(dim=2), torque), dim=-1)
        reference_next = torch.where(live, torch.stack((reference_A, reference_B), dim=1), self.nominal_reference)
        next_state = DualArmState(advanced.state, next_contacts[:, 0], next_contacts[:, 1], torn,
                                  side_axis, side_origin, reference_next, torsion)
        return DualArmStep(next_state, wrench, tension, advanced.support_N.sum(dim=1, keepdim=True),
                           advanced.clearance.amin(dim=1, keepdim=True), budget, available, net_feed,
                           press.squeeze(-1))
