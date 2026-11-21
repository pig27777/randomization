"""Record 2 ms storage, boundary work, and measured losses for the frozen parcel."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_phase_diagrams import side_action, top_action
from audit_yam import StepForces, cases as yam_cases, scene as yam_scene
from test_unified_payload import STAGE2, model, pe_proxy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction, rotate
from src.low_order_parcel.unified_payload import DT, GRAVITY

LIMIT_J = -1e-4


class ObservePayload(StepForces):
    def __init__(self, sim):
        super().__init__(sim)
        p = sim.payload
        self.damp_step, self.edges_step, self.bend_step = p._damp, p._edges, p._bend
        p._damp, p._edges, p._bend = self.damp, self.edges, self.bend

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
        _, jaw, top, _ = self.reaction(old, action, result)
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
                                        self.edge_stop, self.hard_bend)


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


def complete_row(sim, old, result, before, after, left, right, external, top, side, jaw, indent_work, xpbd_damp,
                 friction, floor_kinetic, indent_loss, tendon_damp, edge_stop, hard_bend):
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
    work = left + right + external + top + side + jaw
    delta = sum(after[key] - before.get(key, 0.) for key in after)
    slip = float((result.tension * result.slip_rate).sum() * DT)
    known = xpbd_damp + friction + floor_kinetic + indent_loss + slip + tendon_damp
    return dict(work_J=work, delta_storage_J=delta, net_dissipation_J=work - delta,
                accounted_loss_J=known, closure_residual_J=work - delta - known,
                tether_left_work_J=left, tether_right_work_J=right, body_external_work_J=external,
                top_work_J=top, side_work_J=side, jaw_work_J=jaw, indent_transfer_J=indent_work,
                xpbd_damping_J=xpbd_damp, table_friction_J=friction, floor_velocity_loss_J=floor_kinetic,
                compression_damping_J=indent_loss, tether_damping_J=tendon_damp, slip_work_J=slip, events="+".join(events),
                tension_N=float(result.tension.max()), **after)


def rigid_storage(sim, state):
    return dict(kinetic_J=rigid_kinetic(sim, state.orientation, state.velocity, state.angular_velocity),
                gravity_J=float((sim.mass * GRAVITY * state.com[:, 2:]).sum()),
                indent_J=0., xpbd_storage_J=0.)


def rigid_kinetic(sim, orientation, velocity, angular_rate):
    momentum = sim._inertia_action(orientation, sim.inertia, angular_rate)
    return float(.5 * ((sim.mass * velocity.square()).sum() + (angular_rate * momentum).sum()))


def run_rigid(mode):
    sim = MacroParcelSim.from_nominal(STAGE2["materials"]["pe"]["nominal"], DT)
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
                                        tether_damping(contact, boundary, out), False, False)))
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
def run_payload(name, yam=None):
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
            explicit = ParcelAction(*(torch.as_tensor(data[f"action_{f}"])[:, None] for f in ParcelAction._fields))
        n = len(explicit.pinch_position)
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
        out = sim.step(state, a)
        if explicit is None:
            end_tip = commanded_tip
        elif yam is None:
            end_tip = a.pinch_position + DT * a.pinch_velocity
        else:
            end_tip = explicit.pinch_position[min(i + 1, n - 1)]
            np.testing.assert_allclose(out.table_clearance.numpy().reshape(-1), end_reference[i], atol=1e-10)
        storage, row = ledger.ledger(state, a, out, end_tip, storage)
        rows.append(dict(time_s=(i + 1) * DT, **row))
        state, previous, feedback = out.state, commanded_tip, out.wrist_wrench[:, 2:3]
    return rows


def static_capture():
    sim = MacroParcelSim(pe_proxy(1), payload=model("rigid_box", dtype=torch.float64,
                          total_mass=torch.tensor([[1.5]], dtype=torch.float64)))
    old = sim.initial_state()
    old.contacts[:, :, ContactState.SLACK] = .003
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
        verdicts[name] = [dict(time_s=DT, **row)]
    return verdicts


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
def main(output, yam_roots, focus):
    output.mkdir(parents=True, exist_ok=False)
    cases = static_capture()
    if not focus:
        for name in ("book-top", "clothes-rebound", "pillow-rebound", "side-push", "loose-stop", "bend-limit",
                     "overload", "release"):
            cases[name] = run_payload(name)
        for mode in ("ramp", "release"):
            cases[f"stage2-{mode}"] = run_rigid(mode)
    for yam_root in yam_roots:
        for name in yam_cases():
            path = yam_root / name / "boundary.npz"
            if path.is_file():
                cases[f"yam-{name}"] = run_payload(name, path)
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
    (output / "energy-summary.json").write_text(json.dumps(dict(threshold_J=LIMIT_J, cases=summary), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--yam-root", type=Path, action="append", default=[])
    parser.add_argument("--capture-focus", action="store_true", help="keep the zero-motion capture and YAM replays")
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.output, args.yam_root, args.capture_focus)
