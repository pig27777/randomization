"""Record 2 ms storage, boundary work, and measured losses for the frozen parcel."""
import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from audit_phase_diagrams import side_action, top_action
from audit_yam import StepForces, cases as yam_cases, scene as yam_scene
from test_unified_payload import STAGE2, model, pe_proxy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.low_order_parcel.lumped_contact_proxy import ContactState, LumpedContactProxy
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction, rotate
from src.low_order_parcel.neck_damping import NeckDamping, NeckDampingParams
from src.low_order_parcel.pad_torsion import PadTorsion, PadTorsionParams, conjugate, quaternion_product, rotation_vector
from src.low_order_parcel.unified_payload import DT, GRAVITY

LIMIT_J = -1e-4


class ObservePayload(StepForces):
    def __init__(self, sim):
        super().__init__(sim)
        p = sim.payload
        self.damp_step, self.edges_step, self.bend_step = p._damp, p._edges, p._bend
        p._damp, p._edges, p._bend = self.damp, self.edges, self.bend
        self.neck_result = None
        if sim.neck is not None and sim.neck.enabled:
            self.neck_step = sim.neck.payload
            sim.neck.payload = self.observe_neck

    def observe_neck(self, *args):
        self.neck_result = self.neck_step(*args)
        return self.neck_result

    def kinetic(self, v):
        return .5 * (self.sim.payload.mass.unsqueeze(-1) * v.square()).sum()

    def damp(self, pos, velocity, group):
        after = self.damp_step(pos, velocity, group)
        self.damping += float(self.kinetic(velocity) - self.kinetic(after))
        self.velocity_after_damping = after
        return after

    def edges(self, pos, group, multiplier):
        indices, _, _, _, rest = group
        pairs = pos.index_select(1, indices).reshape(-1, 2, 2, 3)
        separation = (pairs[:, :, 0] - pairs[:, :, 1]).norm(dim=-1)
        if group is self.sim.payload.edge_groups[-1]:
            self.edge_stop |= bool((((separation - rest).abs() > self.sim.payload.slide + 1e-8) &
                                    (self.sim.payload.slide > 0.)).any())
        return self.edges_step(pos, group, multiplier)

    def bend(self, pos, multiplier):
        p = self.sim.payload
        length = (.5 * (pos[:, 0] + pos[:, 3] - pos[:, 1] - pos[:, 2])).norm(dim=1, keepdim=True)
        new_pos, new_multiplier = self.bend_step(pos, multiplier)
        self.hard_bend |= bool((length + p.bend_inverse_mass * (new_multiplier - multiplier) > p.bend_limit + 1e-8).any())
        return new_pos, new_multiplier

    def advance(self, state, force, depth):
        self.damping = 0.
        self.edge_stop = self.hard_bend = False
        return super().advance(state, force, depth)

    def storage(self, state):
        p = self.sim.payload
        pos, _, depth = state
        grav = GRAVITY * (p.mass * pos[:, :, 2]).sum()
        p_ind = p.params
        ratio = depth / p.depth_max.clamp_min(1e-8)
        indent = (p_ind.k_soft * depth.square() * (.5 + .25 * p_ind.hardening_gamma * ratio.square())).sum()
        elastic = 0.
        if float(p_ind.alpha_stretch) > 0.:
            for indices, _, _, _, rest in p.edge_groups:
                pair = pos.index_select(1, indices).reshape(-1, 2, 2, 3)
                error = ((pair[:, :, 0] - pair[:, :, 1]).norm(dim=-1) - rest).abs()
                elastic += float(.5 * (error - p.slide).clamp_min(0.).square().sum() / p_ind.alpha_stretch)
        if float(p_ind.alpha_bend) > 0.:
            bend = .5 * (pos[:, 0] + pos[:, 3] - pos[:, 1] - pos[:, 2])
            elastic += float(.5 * bend.square().sum() / p_ind.alpha_bend)
        return dict(kinetic_J=float(self.kinetic(state.vel)), gravity_J=float(grav),
                    indent_J=float(indent), xpbd_storage_J=elastic)

    def ledger(self, old, action, result, end_tip, old_store):
        p = self.sim.payload
        _, jaw, top, _ = self.reaction(old, action, result, self.neck_result)
        step = result.state
        inc = end_tip - action.pinch_position
        left, right = (self.contact.force * inc).sum(dim=2)[0].tolist()
        block_displacement = step.payload.pos - old.payload.pos
        external = float((self.sim.payload.weights.unsqueeze(-1) * action.external_force[:, None] * block_displacement).sum())
        side = float((self.side * inc).sum())
        jaw_work = float((jaw * inc).sum())
        top_work = float(-(top * inc).sum())
        end_store = self.storage(step.payload)
        frame = p.frame(step.payload)
        local = (end_tip - step.payload.pos.mean(dim=1, keepdim=True)) @ frame
        weights = p.weights_at(local[..., :2])
        gap = ((end_tip - weights @ step.payload.pos) * frame[:, None, :, 2]).sum(dim=-1)
        footprint = (local[..., :2].abs() <= p.params.bag_half_extents[:, None, :2]).all(dim=-1)
        indentation = (weights @ step.payload.d_surf.unsqueeze(-1)).squeeze(-1)
        upper = ((gap >= 0.) | (indentation > 1e-8) |
                 (action.direct_press_N > 0.) | (end_tip[..., 2] <= 0.))
        side_end = self.side_step(step, action._replace(pinch_position=end_tip), frame, local, weights)[3]
        engaged = footprint & upper & ~side_end
        end_store["core_J"] = float(p.surface_core_energy(p.half_height - gap, weights, engaged).sum())
        anchors, _ = p.anchors(step.payload, self.sim.payload_mount_weights, torch.zeros_like(step.payload.d_surf))
        end_store["tether_J"] = tether_storage(self.sim, step, end_tip, anchors)
        xy = self.velocity_after_damping[:, :, :2]
        table = float(.5 * (p.mass.unsqueeze(-1) * (xy.square() - step.payload.vel[:, :, :2].square())).sum())
        floor_kinetic = float(.5 * (p.mass * (self.velocity_after_damping[:, :, 2].square() -
                                              step.payload.vel[:, :, 2].square())).sum())
        indent_growing = (step.payload.d_surf - old.payload.d_surf).clamp_min(0.) / DT
        indent_loss = float((p.params.c_comp * indent_growing.square()).sum() * DT)
        indent_work = float(step.indentation_power.sum() * DT)
        details = dict(evaluated_tether_J=float(result.elastic_energy.sum()),
                       max_elastic_tension_N=float(self.contact.elastic_tension.max()),
                       max_effective_normal_N=float(self.inputs.normal_force.max()),
                       max_anchor_speed_mps=float(self.inputs.attachment_velocity.norm(dim=2).max()))
        return end_store, details | complete_row(self.sim, old, result, old_store, end_store, left, right,
                                        external, top_work, side, jaw_work, indent_work, self.damping,
                                        table, floor_kinetic, indent_loss, tether_damping(self.contact, self.inputs, result),
                                        self.edge_stop, self.hard_bend,
                                        neck_terms(self.neck_result, inc))


def tether_storage(sim, state, pads, anchors):
    separation = (pads - anchors).norm(dim=-1)
    active = state.contacts[:, :, ContactState.CAPTURE] > .5
    reference = sim.proxy.params.L0 if state.contact_reference is None else state.contact_reference
    extension = ((separation - state.contacts[:, :, ContactState.SLIP_CUM]) - reference).clamp_min(0.)
    return float((.5 * sim.proxy.stiffness * extension.square() * active).sum())


def tether_damping(contact, inputs, result):
    difference = inputs.pinch_position - inputs.bag_attachment
    direction = difference / difference.norm(dim=2, keepdim=True).clamp_min(1e-8)
    speed = ((inputs.pinch_velocity - inputs.attachment_velocity) * direction).sum(dim=2)
    damping_force = result.tension - contact.elastic_tension
    return float((damping_force * (speed - result.slip_rate)).sum() * DT)


def neck_terms(step, pad_displacement):
    values = (0., 0., 0., 0., 0.) if step is None else (
        float((step.force * pad_displacement).sum()),
        float(step.pad_couple_power.sum() * DT),
        float(step.anchor_power.sum() * DT),
        float(step.dissipation_J.sum()),
        float((step.pad_force_power + step.pad_couple_power - step.anchor_power).sum() * DT),
    )
    return dict(zip(("neck_pad_force_work_J", "neck_pad_couple_work_J", "neck_anchor_work_J",
                     "neck_discrete_dissipation_J", "neck_contact_dissipation_J"), values, strict=True))


def complete_row(sim, old, result, before, after, left, right, external, top, side, jaw, indent_work, xpbd_damp,
                 friction, floor_kinetic, indent_loss, tendon_damp, edge_stop, hard_bend, neck=None):
    start = old.contacts[:, :, ContactState.CAPTURE] > .5
    state = result.state
    end = state.contacts[:, :, ContactState.CAPTURE] > .5
    events = []
    for value, label in (
        ((~start & end).any(), "capture"),
        ((start & ~end).any(), "release"),
        (result.slip_rate.gt(0.).any(), "slip"),
        (side != 0., "side_contact"),
        (edge_stop, "edge_stop"),
        (hard_bend, "hard_bend"),
    ):
        if value:
            events.append(label)
    if sim.payload is not None:
        change = state.payload.d_surf - old.payload.d_surf
        if (change < -1e-8).any():
            events.append("rebound")
        if (change > 1e-8).any():
            events.append("compression")
    torsion = state.torsion
    twist = {}
    angular_work = angular_loss = 0.
    if torsion:
        angular_work = float(torsion.jaw_work.sum())
        body_work = float(torsion.body_work.sum())
        rotational_damping = float(torsion.damping.sum())
        rotational_slip = float(torsion.slip_loss.sum())
        rotational_integration = float(torsion.integration_loss.sum())
        angular_loss = rotational_damping + rotational_slip + rotational_integration
        after["torsion_J"] = float(torsion.energy.sum())
        twist = dict(torsion_jaw_work_J=angular_work, torsion_body_work_J=body_work,
                     torsion_damping_J=rotational_damping, torsion_slip_loss_J=rotational_slip,
                     torsion_integration_loss_J=rotational_integration, torsion_slip_rad=float(torsion.slip_angle.sum()),
                     torsion_net_dissipation_J=angular_work - body_work - after["torsion_J"] + before.get("torsion_J", 0.))
    neck = neck if neck is not None else neck_terms(None, None)
    work = (left + right + external + top + side + jaw + angular_work
            + neck["neck_pad_force_work_J"] + neck["neck_pad_couple_work_J"])
    delta = sum(after[key] - before.get(key, 0.) for key in after)
    slip = float((result.tension * result.slip_rate).sum() * DT)
    known = (xpbd_damp + friction + floor_kinetic + indent_loss + slip + tendon_damp + angular_loss
             + neck["neck_discrete_dissipation_J"])
    return dict(work_J=work, delta_storage_J=delta, net_dissipation_J=work - delta,
                accounted_loss_J=known, closure_residual_J=work - delta - known,
                tether_left_work_J=left, tether_right_work_J=right, body_external_work_J=external,
                top_work_J=top, side_work_J=side, jaw_work_J=jaw, indent_transfer_J=indent_work,
                xpbd_damping_J=xpbd_damp, table_friction_J=friction, floor_velocity_loss_J=floor_kinetic,
                compression_damping_J=indent_loss, tether_damping_J=tendon_damp, slip_work_J=slip, events="+".join(events),
                tension_N=float(result.tension.max()), **neck, **twist, **after)


def rigid_storage(sim, state):
    return dict(kinetic_J=rigid_kinetic(sim, state.orientation, state.velocity, state.angular_velocity),
                gravity_J=float((sim.mass * GRAVITY * state.com[:, 2:]).sum()),
                indent_J=0., xpbd_storage_J=0.)


def rigid_kinetic(sim, orientation, velocity, angular_rate):
    momentum = sim._inertia_action(orientation, sim.inertia, angular_rate)
    return float(.5 * ((sim.mass * velocity.square()).sum() + (angular_rate * momentum).sum()))


def enable_neck(sim, coefficient):
    if coefficient:
        sim.neck = NeckDamping(sim.mass, sim.proxy.params.mu_eff[:, None], DT, NeckDampingParams(coefficient))


def run_rigid(mode, finite_pad=False, neck_coefficient=0.):
    sim = MacroParcelSim.from_nominal(STAGE2["materials"]["pe"]["nominal"], DT)
    if finite_pad:
        sim.torsion = PadTorsion(sim.mass, sim.proxy.params.mu_eff[:, None], DT, PadTorsionParams())
    enable_neck(sim, neck_coefficient)
    neck = None
    if sim.neck is not None:
        calculate = sim.neck.rigid

        def observe_neck(*args):
            nonlocal neck
            neck = calculate(*args)
            return neck

        sim.neck.rigid = observe_neck
    observed, impulse = sim.proxy.step_with_reference, sim._impulse
    impulses = []
    contact = boundary = None

    def capture(state, inputs, reference, inverse_mass=None):
        nonlocal contact, boundary
        contact, next_reference = observed(state, inputs, reference, inverse_mass)
        boundary = inputs
        return contact, next_reference

    def record_impulse(*args):
        result = impulse(*args)
        impulses.append(result)
        return result

    sim.proxy.step_with_reference, sim._impulse = capture, record_impulse
    _, _, _, actions = sim.probe_actions(mode, axis="tangential" if mode == "ramp" else "vertical")
    state = sim.initial_state()
    row = rigid_storage(sim, state) | {"tether_J": 0.}
    rows = []
    for i in range(len(actions.pinch_position)):
        old = state
        action = ParcelAction(*(a[i] for a in actions))
        impulses.clear()
        out = sim.step(old, action)
        state = out.state
        inc = action.pinch_velocity * DT
        end = rigid_storage(sim, state)
        attachment = state.com[:, None] + rotate(state.orientation[:, None], sim.mounts)
        end["tether_J"] = tether_storage(sim, state, action.pinch_position + inc, attachment)
        ext = float(((action.external_force - action.direct_press_N * sim.up) * (state.com - old.com)).sum())
        left, right = (contact.force * inc).sum(dim=2)[0].tolist()
        (ndv, ndw), (fdv, fdw), (cdv, cdw) = impulses
        post_friction = rigid_kinetic(sim, old.orientation, state.velocity-cdv, state.angular_velocity-cdw)
        pre_friction = rigid_kinetic(sim, old.orientation, state.velocity-cdv-fdv, state.angular_velocity-cdw-fdw)
        pre_floor = rigid_kinetic(sim, old.orientation, state.velocity-cdv-fdv-ndv,
                                  state.angular_velocity-cdw-fdw-ndw)
        post_correction = rigid_kinetic(sim, old.orientation, state.velocity, state.angular_velocity)
        normal_loss = pre_floor-pre_friction + post_friction-post_correction
        rows.append(dict(time_s=(i + 1) * DT, evaluated_tether_J=float(out.elastic_energy.sum()),
                         max_elastic_tension_N=float(contact.elastic_tension.max()),
                         max_effective_normal_N=float(boundary.normal_force.max()),
                         max_anchor_speed_mps=float(boundary.attachment_velocity.norm(dim=2).max()),
                         **complete_row(sim, old, out, row, end, left, right, ext, 0., 0., 0., 0.,
                                        0., pre_friction-post_friction, normal_loss, 0.,
                                        tether_damping(contact, boundary, out), False, False, neck_terms(neck, inc))))
        row = end
    return rows


def bench(name):
    preset, n = {
        "book-top": ("rigid_box", 2300),
        "clothes-rebound": ("folded_clothes", 2300),
        "pillow-rebound": ("fluffy_pillow", 2300),
        "side-push": ("rigid_box", 245),
        "loose-stop": ("loose_shifting", 2385),
        "bend-limit": ("loose_shifting", 2385),
        "overload": ("rigid_box", None),
        "release": ("rigid_box", None),
    }[name]
    like = torch.zeros(1, 1, dtype=torch.float64)
    settings = {}
    if name == "side-push":
        settings = dict(total_mass=like + .065, table_friction=like + .06)
    elif name == "bend-limit":
        settings = dict(theta_max=like + .025)
    elif name in ("book-top", "clothes-rebound", "pillow-rebound"):
        settings = dict(total_mass=like + .24, bag_half_extents=like.new_tensor([[.085, .105, .05]]))
    sim = MacroParcelSim(pe_proxy(1), payload=model(preset, dtype=like.dtype, **settings))
    explicit = None
    if name in ("loose-stop", "bend-limit", "overload", "release"):
        mode = {"loose-stop": "side", "bend-limit": "side", "overload": "ramp", "release": "release"}[name]
        _, _, _, explicit = sim.probe_actions(mode, axis="tangential")
        n = len(explicit.pinch_position)
    return sim, sim.initial_state(), n, explicit


@torch.inference_mode()
def run_payload(name, yam=None, finite_pad=False, neck_coefficient=0.):
    jaw = jaw_rate = None
    if yam is None:
        sim, state, n, explicit = bench(name)
    else:
        robot = yam_scene(yam_cases()[name])
        sim, state = robot.sim.parcel, robot.sim.state
        # Scene construction puts an observer on the parcel for immediate
        # robot-side reactions. Restore its wrapped methods before replay.
        tap = robot.sim.boundary
        sim.proxy.step_with_reference, sim.payload.surface_contact = tap.proxy_step, tap.surface_step
        sim._side_contact, sim.payload.step = tap.side_step, tap.payload_step
        sim._jaw_contact = tap.jaw_step
        with np.load(yam) as data:
            end_reference = data["clearance_m"]
            time = data["time_s"]
            explicit = ParcelAction(*(torch.as_tensor(data[f"action_{f}"])[:, None] for f in ParcelAction._fields))
        if finite_pad:
            jaw, jaw_rate = replay_wrist(robot.sim, yam.with_name("trajectory.npz"), time, sim.mass)
        n = len(explicit.pinch_position)
    if finite_pad:
        sim.torsion = PadTorsion(sim.mass, sim.proxy.params.mu_eff[:, None], DT, PadTorsionParams())
    enable_neck(sim, neck_coefficient)
    ledger = ObservePayload(sim)
    storage = ledger.storage(state.payload) | {"tether_J": 0.}
    previous = None
    z = torch.full_like(sim.mass, .115)
    feedback = torch.zeros_like(z)
    press_z = z
    rows = []
    for i in range(n):
        if explicit is not None:
            a = ParcelAction(*(value[i] for value in explicit))
            if name in ("loose-stop", "bend-limit"):
                t = min(1., max(0., (i * DT - .55) / .70))
                next_t = min(1., max(0., ((i + 1) * DT - .55) / .70))
                amplitude = .02 if name == "loose-stop" else .06
                offset = amplitude * (1. - math.cos(2. * math.pi * t))
                next_offset = amplitude * (1. - math.cos(2. * math.pi * next_t))
                tilt_v = (next_offset - offset) / DT
                side = sim.up[:, None] * sim.up.new_tensor([[[-1.], [1.]]])
                a = a._replace(pinch_position=a.pinch_position + offset * side,
                               pinch_velocity=a.pinch_velocity + tilt_v * side)
        elif name == "side-push":
            a = side_action(i, .024 + torch.zeros_like(z), previous)
        else:
            a, z, press_z = top_action(i, sim.mass, z, press_z, feedback, previous)
        commanded_tip = a.pinch_position
        if explicit is None and previous is not None:
            # The feedback probe constructs the next command from the previous
            # one. Integrate along that actual segment, not its extrapolation.
            a = a._replace(pinch_position=previous)
        out = sim.step(state, a, jaw_orientation=None if jaw is None else jaw[i],
                       jaw_angular_velocity=None if jaw_rate is None else jaw_rate[i])
        if explicit is None:
            end_tip = commanded_tip
        elif yam is None:
            end_tip = a.pinch_position + DT * a.pinch_velocity
        else:
            end_tip = explicit.pinch_position[min(i + 1, n - 1)]
            if not finite_pad and not neck_coefficient:
                np.testing.assert_allclose(out.table_clearance.numpy().reshape(-1), end_reference[i], atol=1e-10)
        storage, row = ledger.ledger(state, a, out, end_tip, storage)
        rows.append(dict(time_s=(i + 1) * DT, **row))
        state, previous, feedback = out.state, commanded_tip, out.wrist_wrench[:, 2:3]
    return rows


def replay_wrist(robot, path, time, like):
    """Reconstruct the archived wrist frame; use both endpoint poses for angular work."""
    with np.load(path) as saved:
        sample_time = saved["time"]
        positions = saved["qpos"]
    endpoints = np.append(time, time[-1] + DT)
    qpos = np.stack([np.interp(endpoints, sample_time, q) for q in positions.T], axis=1)
    wrist = robot.m.body("link_6").id
    data = mujoco.MjData(robot.m)
    quats = np.empty((len(qpos), 4))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(robot.m, data)
        quats[i] = data.xquat[wrist]
    oriented = like.new_tensor(quats)[:, None]
    rate = rotation_vector(quaternion_product(oriented[1:], conjugate(oriented[:-1]))) / DT
    return oriented[:-1], rate


def static_capture(neck_coefficient=0.):
    sim = MacroParcelSim(pe_proxy(1), payload=model("rigid_box", dtype=torch.float64,
                          total_mass=torch.tensor([[1.5]], dtype=torch.float64)), torsion=PadTorsionParams())
    old = sim.initial_state()
    old.contacts[:, :, ContactState.SLACK] = .003
    enable_neck(sim, neck_coefficient)
    tap = ObservePayload(sim)
    anchors, _ = sim.payload.anchors(old.payload, sim.payload_mount_weights, torch.zeros_like(old.payload.d_surf))
    tip = anchors + (sim.proxy.params.L0 + .010).unsqueeze(-1) * sim.up[:, None]
    zeros = torch.zeros_like(sim.proxy.params.L0)
    action = ParcelAction(tip, torch.zeros_like(tip), zeros + 6., zeros, zeros + .010, zeros + 1.,
                          torch.zeros_like(old.com), torch.zeros_like(sim.mass))
    before = tap.storage(old.payload) | {"tether_J": 0.}
    verdicts = {}
    for name, inputs in (("static-no-capture", action._replace(normal_force=zeros)), ("static-capture", action)):
        step = sim.step(old, inputs)
        _, row = tap.ledger(old, inputs, step, tip, before)
        assert row["torsion_J"] == row["evaluated_tether_J"] == row["neck_discrete_dissipation_J"] == 0.
        verdicts[name] = [dict(time_s=DT, **row)]
    return verdicts


def torsion_fixture(grip, radius=.012):
    """Hold a preloaded 1.5 kg book with centered mounts and an eccentric payload."""
    mass = torch.tensor([[1.5]])
    base = pe_proxy(1).params
    params = replace(type(base)(**{k: v.float() for k, v in vars(base).items()}),
                     k=mass.new_full((1, 1), 1800.), c_d=mass.new_full((1, 1), 10.),
                     L0=mass.new_full((1, 2), .004))
    sim = MacroParcelSim(LumpedContactProxy(params, DT), payload=model(
        "rigid_box", dtype=mass.dtype, total_mass=mass, mass_weights=mass.new_tensor([[.4, .4, .1, .1]])),
        torsion=PadTorsionParams(radius_m=radius))
    state = sim.initial_state()
    p = sim.payload
    positions = state.payload.pos + .30 * p.up
    payload_state = state.payload._replace(pos=positions)
    state = state._replace(payload=payload_state, com=p.com(payload_state))
    state.contacts[..., ContactState.CAPTURE] = 1.
    state.contacts[..., ContactState.SLACK] = .003
    tip, _ = p.anchors(state.payload, sim.payload_mount_weights, torch.zeros_like(state.payload.d_surf))
    tip = tip + (params.L0 + .5 * mass * GRAVITY / params.k).unsqueeze(-1) * sim.up[:, None]
    zero = torch.zeros_like(params.L0)
    inputs = ParcelAction(tip, torch.zeros_like(tip), zero + grip, zero, zero, zero + 1.,
                          torch.zeros_like(state.com), torch.zeros_like(mass))
    return sim, state, inputs


@torch.inference_mode()
def torsion_episode(grip, steps=1000, neck_coefficient=0.):
    sim, state, inputs = torsion_fixture(grip)
    enable_neck(sim, neck_coefficient)
    observe = ObservePayload(sim)
    storage = observe.storage(state.payload)
    anchors, _ = sim.payload.anchors(state.payload, sim.payload_mount_weights, state.payload.d_surf)
    storage["tether_J"] = tether_storage(sim, state, inputs.pinch_position, anchors)
    rows = []
    for i in range(steps):
        result = sim.step(state, inputs)
        storage, row = observe.ledger(state, inputs, result, inputs.pinch_position, storage)
        state = result.state
        frame = sim.payload.frame(state.payload)
        roll = torch.atan2(frame[:, 2, 1], frame[:, 2, 2])
        rows.append(dict(time_s=(i + 1) * DT, **row, roll_rad=float(roll),
                         roll_speed_rad_s=float(state.angular_velocity[:, 0]),
                         com_speed_mps=float(state.velocity.norm()),
                         clearance_m=float(result.table_clearance.min()),
                         captured=int(state.contacts[..., ContactState.CAPTURE].sum()),
                         torque_limit_Nm=float(state.torsion.capacity.sum())))
    return rows


def c6_neck_episode(coefficient):
    from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
    episode = CornerSuite(DomainRandomizer()).select(torch.tensor([5]))
    sim = MacroParcelSim(LumpedContactProxy(episode.proxy, DT), payload=model(
        "rigid_box", dtype=torch.float32, total_mass=episode.payload.total_mass, table_friction=episode.payload.table_friction),
        torsion=episode.torsion, neck_damping=NeckDampingParams(coefficient))
    old = sim.initial_state()
    payload = old.payload._replace(pos=old.payload.pos + .20 * sim.payload.up,
                                   vel=old.payload.vel + old.com.new_tensor([.12, 0., 0.]))
    old = old._replace(payload=payload, com=sim.payload.com(payload))
    old.contacts[..., ContactState.CAPTURE] = 1.
    old.contacts[..., ContactState.SLACK] = .005
    anchors, _ = sim.payload.anchors(payload, sim.payload_mount_weights, torch.zeros_like(payload.d_surf))
    length = episode.proxy.L0 + .5 * (episode.payload.total_mass * GRAVITY / episode.proxy.k)
    tip = anchors + length.unsqueeze(-1) * sim.up[:, None]
    zero = torch.zeros_like(length)
    cmd = ParcelAction(tip, torch.zeros_like(tip), zero + 12., zero, zero, zero + 1.,
                       torch.zeros_like(old.com), torch.zeros_like(sim.mass))
    observer = ObservePayload(sim)
    store = (observer.storage(payload) |
             {"tether_J": tether_storage(sim, old, tip, anchors), "torsion_J": 0.})
    rows = []
    for i in range(2250):
        out = sim.step(old, cmd)
        store, row = observer.ledger(old, cmd, out, tip, store)
        old = out.state
        rows.append(dict(time_s=(i + 1) * DT, **row,
                         com_speed_mps=float(old.velocity.norm()), clearance_m=float(out.table_clearance.min()),
                         captured=int(old.contacts[..., ContactState.CAPTURE].sum())))
    return rows


def write_case(output, name, rows):
    with (output / f"{name}.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    min_row = min(rows, key=lambda row: row["net_dissipation_J"])
    anomalies = [row for row in rows if row["net_dissipation_J"] < LIMIT_J]
    boundary = {name: {k: event[k] for k in ("time_s", "events", "work_J", "evaluated_tether_J",
                 "net_dissipation_J", "kinetic_J", "tether_J", "tension_N", "max_elastic_tension_N",
                 "max_effective_normal_N", "max_anchor_speed_mps")}
                for name in ("capture", "release", "side_contact", "edge_stop", "hard_bend", "rebound")
                if (event := min((r for r in rows if name in r["events"].split("+")),
                                  key=lambda r: r["net_dissipation_J"], default=None)) is not None}
    return dict(minimum_step_J=min_row["net_dissipation_J"], time_s=min_row["time_s"], events=min_row["events"],
                negative_steps=len(anomalies), peak_tension_N=max(row["tension_N"] for row in rows),
                event_frames={k: sum(k in r["events"].split("+") for r in rows)
                              for k in ("capture", "release", "rebound", "edge_stop", "hard_bend", "side_contact", "slip")},
                cumulative_dissipation_J=sum(row["net_dissipation_J"] for row in rows),
                events_below_threshold={name: sum(name in r["events"].split("+") for r in anomalies)
                                        for name in boundary}, event_worst_step=boundary, passed=not anomalies)


@torch.inference_mode()
def main(output, yam_roots, focus, finite_pad=False, neck_coefficient=0.):
    output.mkdir(parents=True, exist_ok=False)
    cases = static_capture(neck_coefficient)
    cases["torsion-hold"] = torsion_episode(90., neck_coefficient=neck_coefficient)
    cases["torsion-slip"] = torsion_episode(18., neck_coefficient=neck_coefficient)
    if neck_coefficient:
        cases["c6-neck"] = c6_neck_episode(neck_coefficient)
    if not focus:
        for name in ("book-top", "clothes-rebound", "pillow-rebound", "side-push", "loose-stop", "bend-limit",
                     "overload", "release"):
            cases[name] = run_payload(name, finite_pad=finite_pad, neck_coefficient=neck_coefficient)
        for mode in ("ramp", "release"):
            cases[f"stage2-{mode}"] = run_rigid(mode, finite_pad, neck_coefficient)
    for yam_root in yam_roots:
        for name in yam_cases():
            path = yam_root / name / "boundary.npz"
            if path.is_file():
                cases[f"yam-{name}"] = run_payload(name, path, finite_pad, neck_coefficient)
    summary = {}
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    for name, rows in cases.items():
        summary[name] = write_case(output, name, rows)
        time = [row["time_s"] for row in rows]
        axis = axes[int(name.startswith("yam-"))]
        axis.plot(time, np.cumsum([row["net_dissipation_J"] for row in rows]), label=name)
        print(name, json.dumps({k: v for k, v in summary[name].items() if k != "event_worst_step"}), flush=True)
    for ax, title in zip(axes, ("Controlled probes", "Articulated YAM boundary replay"), strict=True):
        ax.set(xlabel="Time (s)", ylabel="Integrated W − ΔE (J)", title=title, yscale="symlog")
        if ax.lines:
            ax.legend(fontsize=8, ncols=3)
    fig.savefig(output / "cumulative-dissipation.png", dpi=170)
    (output / "energy-summary.json").write_text(json.dumps(
        dict(threshold_J=LIMIT_J, finite_pad=finite_pad, neck_coefficient_Ns_m=neck_coefficient, cases=summary), indent=2) + "\n")
    return all(case["passed"] for case in summary.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--yam-root", type=Path, action="append", default=[])
    parser.add_argument("--capture-focus", action="store_true", help="keep the zero-motion capture and YAM replays")
    parser.add_argument("--finite-pad", action="store_true", help="enable torsion in historical scenarios as well")
    parser.add_argument("--neck-coefficient", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(0 if main(args.output, args.yam_root, args.capture_focus, args.finite_pad, args.neck_coefficient) else 1)
