"""Execute and preserve individual PE benchmark trials with predeclared gates."""
import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback

import mujoco
import numpy as np

from pe_physics import (BASE, FIXTURES, MAX_PENETRATION, MAX_STRETCH, PAIRS, ROOT,
                        FlexBagBackend, save_json)
from bag_sim.wrap import panel_pairs, triangle_box_intersections, winding_number

GRAVITY = (0., 0., -9.81)
IDENTITY = (1., 0., 0., 0.)
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


@dataclass(frozen=True)
class Trial:
    case: str
    duration: float = .5
    dt: float = BASE.timestep
    iterations: int = BASE.iterations
    height: float = 0.
    gravity: tuple = GRAVITY
    table: bool = True
    plane_table: bool = False
    coupon: bool = False
    layers: int = 1
    axis: int = 0
    mass: float = 0.
    size: tuple = (.10, .07, .017)
    offset: tuple = (0., 0.)
    self_collision: bool = True
    self_contact_time: float = BASE.contact_time
    yaw: float = 0.
    probe_x: float = 0.
    gentle_probe: bool = False
    fold_angles: tuple = ()
    camber: float = 0.
    tool_mu: float = .45
    motion_time_scale: float = 1.
    vbd_extensional_scale: float = 1.
    initial_tools: dict = field(default_factory=dict)

    def fixed_nodes(self, film):
        if self.case == "B05-D":
            x = film["reference"][:, 0]
            return np.flatnonzero(np.abs(x) >= np.abs(x).max() - .001)
        if self.case.startswith(("B02", "B03-A", "B03-C")):
            coord = film["reference"][:, self.axis]
            width = .0101 if self.case == "B03-A" else 1e-9
            return np.flatnonzero(coord <= coord.min() + width)
        return np.empty(0, dtype=int)

    def loaded_nodes(self, film):
        if self.case.startswith("B02"):
            coord = film["reference"][:, self.axis]
            return np.flatnonzero(coord >= coord.max() - 1e-9)
        return np.empty(0, dtype=int)

    def clamped_nodes(self, film):
        fixed = self.fixed_nodes(film)
        result = {"fixed": fixed} if len(fixed) else {}
        if self.case == "B03-C":
            p = film["reference"][:, self.axis]
            result["moving"] = np.flatnonzero(p >= p.max() - 1e-9)
        return result


def phase_one(dt):
    trials = [
        Trial("B01", 1.2, height=.04),
        Trial("B02-X", .64, gravity=(0., 0., 0.), table=False, coupon=True),
        Trial("B02-Y", .64, gravity=(0., 0., 0.), table=False, coupon=True, axis=1),
        Trial("B03-A", .55, table=False, coupon=True),
        Trial("B03-B", .55, height=.046, coupon=True,
              initial_tools={"obstacle": (0., 0., .015)}),
        Trial("B03-C", .42, gravity=(0., 0., 0.), table=False, coupon=True, camber=.0002),
        Trial("B04-PLATES", .60, initial_tools={"platen": (0., 0., .011)}),
        Trial("B04-DOUBLE", .56, height=.0005, coupon=True, layers=2,
              initial_tools={"platen": (0., 0., .010)}),
        Trial("B04-U", .60, height=.0006, coupon=True, fold_angles=(0, 0, 0, 60, 120, 180, 180, 180),
              initial_tools={"platen": (0., 0., .025)}),
        Trial("B04-ACCORDION", .60, height=.0006, coupon=True,
              fold_angles=(0, 60, 120, 180, 180, 120, 60, 0), initial_tools={"platen": (0., 0., .043)}),
        Trial("B05-A-200", .5, mass=.20, offset=(-.035, 0.)),
        Trial("B05-A-350", .5, mass=.35, size=(.115, .075, .022), offset=(.035, 0.)),
        Trial("B05-B", .72, mass=.20, offset=(-.035, 0.), yaw=np.deg2rad(15), table=False,
              initial_tools={"platen": (0., 0., -.005)}),
        Trial("B05-C", .55, mass=.20, table=False, initial_tools={"platen": (0., 0., -.005)}),
        Trial("B05-D", .70, mass=.35, offset=(.035, 0.)),
    ]
    return trials if dt is None else [replace(trial, dt=dt) for trial in trials]


def phase_two(dt):
    probes = (-.065, 0., .065)
    rows = []
    for state, x in enumerate((None, *probes)):
        configuration = dict(mass=.20 if x is not None else 0., offset=(x or 0., 0.))
        for probe, coordinate in enumerate(probes):
            rows.append(Trial(f"B06-S{state}-P{probe}", .90, probe_x=coordinate,
                              initial_tools={"contact_probe": (coordinate, 0., .041)}, **configuration))
        if state:
            for control in ("HEIGHT", "FORCE"):
                rows.append(Trial(f"B07-{control}-S{state}", 2.0,
                                  initial_tools={"contact_probe": (-.10, 0., .041)}, **configuration))
    return rows if dt is None else [replace(trial, dt=dt) for trial in rows]


def ramp(t, start, end):
    s = np.clip((t - start) / (end - start), 0., 1.)
    return s * s * (3. - 2. * s), 6. * s * (1. - s) / (end - start)


def action(trial, time, feedback=None):
    time /= trial.motion_time_scale
    tools = {name: (np.array(pos), np.zeros(3), IDENTITY) for name, pos in trial.initial_tools.items()}
    load = np.zeros(3)
    clamps = {}
    if trial.case.startswith("B02"):
        ends = np.array([0., .08, .16, .24, .32, .40, .48, .56, .64])
        width = .04 if trial.axis == 0 else .08
        load[trial.axis] = width * BASE.membrane * np.interp(
            time, ends, [0., .01, .01, .02, .02, .03, .03, 0., 0.])
    if trial.case == "B03-C":
        s, _ = ramp(time, .06, .30)
        clamps["moving"] = np.array([-s * .006, 0., 0.])
    if trial.case.startswith("B04"):
        close, dc = ramp(time, .10, .28)
        release, dr = ramp(time, .41, .51)
        target = {"B04-DOUBLE": .00585, "B04-U": .0061, "B04-ACCORDION": .0067}.get(trial.case, .00565)
        p, _, quat = tools["platen"]
        travel = p[2] - target
        p[2] -= travel * (close - release)
        tools["platen"] = (p, np.array([0., 0., -travel * (dc - dr)]), quat)
    if trial.case in ("B05-B", "B05-C", "B05-D"):
        progress, speed = ramp(time, .22, .52)
        if trial.case == "B05-D":
            clamps["fixed"] = np.array([0., 0., .065 * progress])
        elif trial.case == "B05-B":
            angle = np.deg2rad(25) * progress
            tools["platen"] = (np.array([0., 0., -.005]), np.zeros(3),
                               (np.cos(angle / 2), 0., np.sin(angle / 2), 0.))
        else:
            tools["platen"] = (np.array([.065 * progress, 0., -.005]),
                               np.array([.065 * speed, 0., 0.]), IDENTITY)
    if trial.case.startswith(("B06", "B07")):
        press, speed = ramp(time, .25, .65)
        unload, up_speed = ramp(time, trial.duration - .18, trial.duration)
        x, vx = trial.probe_x, 0.
        z, vz = .041 - .0305 * (press - unload), -.0305 * (speed - up_speed)
        if trial.case.startswith("B07"):
            sweep, sweep_speed = ramp(time, .68, 1.75)
            x, vx = -.10 + .20 * sweep, .20 * sweep_speed
            if "FORCE" in trial.case and feedback is not None and .40 <= time < 1.82:
                z, vz = float(np.clip(feedback.height + feedback.speed * (time - feedback.time), .0105, .041)), feedback.speed
        if trial.gentle_probe:
            press, speed = ramp(time, .20, .65)
            z = .031 - .006 * (press - unload)
            vz = -.006 * (speed - up_speed)
        if feedback is not None and feedback.stop_time is not None:
            s, ds = ramp(time, feedback.stop_time, feedback.stop_time + .10)
            x, vx = feedback.stop_x, 0.
            rest = .031 if trial.gentle_probe else .041
            z, vz = feedback.stop_height + (rest - feedback.stop_height) * s, (rest - feedback.stop_height) * ds
        tools["contact_probe"] = (np.array([x, 0., z]), np.array([vx, 0., vz]), IDENTITY)
    if trial.motion_time_scale != 1.:
        tools = {name: (p, v / trial.motion_time_scale, q) for name, (p, v, q) in tools.items()}
    return dict(tools=tools, load=load, clamps=clamps)


class MeshMetrics:
    def __init__(self, film, closed):
        self.faces = film["faces"]
        self.edges = film["edges"]
        self.masses = film["masses"]
        q = film["reference"]
        self.length = np.linalg.norm(q[self.edges[:, 1]] - q[self.edges[:, 0]], axis=1)
        vec = q[self.faces[:, 1:]] - q[self.faces[:, :1]]
        a, b = vec.transpose(1, 0, 2)
        la = np.linalg.norm(a, axis=1)
        projection = np.einsum("ij,ij->i", a, b) / la
        height = np.linalg.norm(np.cross(a, b), axis=1) / la
        self.area = .5 * la * height
        coordinates = np.swapaxes(np.array([[la, projection], [np.zeros(len(la)), height]]), 0, 2)
        self.inv_rest = np.linalg.inv(coordinates.transpose(0, 2, 1))
        i, j = np.triu_indices(len(self.faces), k=1)
        a, b = self.faces[i], self.faces[j]
        disjoint = ~np.any(a[:, :, None] == b[:, None, :], axis=(1, 2))
        self.nonadjacent = (i[disjoint], j[disjoint])
        if closed:
            self.layers = panel_pairs(q, self.faces, film["panels"], [0], [3])
        elif np.unique(film["panels"]).size == 2:
            self.layers = np.arange(len(q)).reshape(2, -1)[::-1].T
        else:
            self.layers = np.empty((0, 2), dtype=int)

    def intersections(self, points):
        """Nonadjacent triangle crossings from interior edge/triangle intersections."""
        a, b = self.nonadjacent
        triangles = points[self.faces]
        lo, hi = triangles.min(axis=1), triangles.max(axis=1)
        overlap = np.all((hi[a] >= lo[b]) & (hi[b] >= lo[a]), axis=1)
        first, second = triangles[a[overlap]], triangles[b[overlap]]
        return int(np.count_nonzero(crosses(first, second) | crosses(second, first)))

    def measure(self, q, v, gravity):
        edge = np.linalg.norm(q[self.edges[:, 1]] - q[self.edges[:, 0]], axis=1) / self.length
        basis = np.swapaxes(q[self.faces[:, 1:]] - q[self.faces[:, :1]], 1, 2)
        gradient = basis @ self.inv_rest
        metric = gradient.transpose(0, 2, 1) @ gradient
        stretch = np.sqrt(np.maximum(0., np.linalg.eigvalsh(metric)))
        strain = .5 * (metric - np.eye(2))
        young, nu = BASE.membrane, BASE.poisson
        membrane = self.area @ (young / (2 * (1 + nu)) * (strain**2).sum(axis=(1, 2)) +
                                .5 * young * nu / (1. - nu**2) * np.trace(strain, axis1=1, axis2=2)**2)
        separation = q[self.layers[:, 0], 2] - q[self.layers[:, 1], 2]
        return dict(edge_stretch=edge, principal_stretch=stretch,
                    com=np.average(q, weights=self.masses, axis=0),
                    kinetic=float(.5 * np.sum(self.masses[:, None] * v**2)),
                    gravity_energy=float(-self.masses @ (q @ gravity)),
                    stvk_energy=float(membrane),
                    rms_speed=float(np.sqrt(np.average(np.sum(v**2, axis=1), weights=self.masses))),
                    layer_min=float(separation.min()) if len(separation) else np.nan,
                    layer_inverted_fraction=float(np.mean(separation < -1e-6)) if len(separation) else np.nan)


def crosses(edges, triangles):
    # Coplanar overlap and shared edges are intentionally outside this test.
    start = edges
    direction = np.roll(edges, -1, axis=1) - start
    u = (triangles[:, 1] - triangles[:, 0])[:, None]
    v = (triangles[:, 2] - triangles[:, 0])[:, None]
    h = np.cross(direction, v)
    det = np.sum(u * h, axis=2)
    valid = np.abs(det) > 1e-15
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    delta = start - triangles[:, :1]
    s = inv * np.sum(delta * h, axis=2)
    r = np.cross(delta, u)
    t = inv * np.sum(direction * r, axis=2)
    along = inv * np.sum(v * r, axis=2)
    eps = 1e-7
    return np.any(valid & (s >= -eps) & (t >= -eps) & (s + t <= 1. + eps) &
                  (along > eps) & (along < 1. - eps), axis=1)


def payload_geometry(q, pose, size, faces):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, pose[3:])
    frame = rotation.reshape(3, 3)
    half = np.asarray(size) / 2
    corners = (CORNERS * half) @ frame.T + pose[:3]
    inside = np.abs(winding_number(q, faces, corners)) >= .5
    # Exact SAT on the box with a 1 µm inset avoids classifying exact tangency as traversal.
    intersect = triangle_box_intersections(q, faces, half - 1e-6, pose[:3], frame)
    return dict(payload_outside=int(np.count_nonzero(~inside)), payload_intersections=int(intersect.sum()),
                payload_floor=float(corners[:, 2].min()))


def sample(backend, trial, metric, time, commanded, rest_gauge):
    q, v = backend.get_bag_state()
    pose, velocity = backend.get_payload_state()
    if not all(np.isfinite(array).all() for array in (q, v, pose, velocity)):
        raise FloatingPointError(f"nonfinite state at {time:.8f} s")
    result = dict(time=time, q=q.copy(), v=v.copy(), payload=pose, payload_velocity=velocity,
                  fixtures=backend.get_fixture_state(), load=commanded["load"],
                  contact_time=backend.contact_time if hasattr(backend, "contact_time") else max(0., time - trial.dt))
    result["fixture_target"] = np.array([commanded["tools"].get(n, (np.full(3, np.nan),))[0] for n in FIXTURES])
    result.update(metric.measure(q, v, trial.gravity), self_crossings=metric.intersections(q),
                  table_depth=max(0., -float(q[:, 2].min())) if trial.table else 0.)
    contacts = backend.get_contact_wrench()
    if not all(np.isfinite(value).all() for value in contacts.values()):
        raise FloatingPointError(f"nonfinite contact reaction at {time:.8f} s")
    result.update(contacts, **backend.get_physics_diagnostics(q))
    if trial.mass:
        result.update(payload_geometry(q, pose, trial.size, metric.faces))
    else:
        result.update(payload_outside=0, payload_intersections=0, payload_floor=np.nan)
    film = backend.envelope
    ends = trial.loaded_nodes(film)
    fixed = trial.fixed_nodes(film)
    result["strain"] = ((q[ends, trial.axis].mean() - q[fixed, trial.axis].mean()) / rest_gauge - 1.
                        if len(ends) else np.nan)
    result["tip_z"] = float(q[np.isclose(film["reference"][:, 0], .04), 2].mean()) if trial.coupon else np.nan
    return result


def evaluate(trial, outcome, data, elapsed):
    result = dict(case=trial.case, status=outcome, runtime_s=elapsed, samples=len(data.get("time", [])))
    if not result["samples"]:
        return result
    time = data["time"]
    tail = time >= trial.duration - .10 - .5 * trial.dt
    energy = data["kinetic"] + data["gravity_energy"] + data["stvk_energy"] + data["bending_energy"]
    last_time = float(time[-1])
    result.update(final_time_s=last_time, maximum_edge_ratio=float(data["edge_stretch"].max()),
                  edge_ratio_p95=float(np.quantile(data["edge_stretch"], .95)),
                  edge_ratio_p99=float(np.quantile(data["edge_stretch"], .99)),
                  max_principal_ratio=float(data["principal_stretch"].max()),
                  crossing_pairs=int(data["self_crossings"].max()),
                  payload_intersecting_triangles=int(data["payload_intersections"].max()),
                  payload_outside_corners=int(data["payload_outside"].max()),
                  max_table_depth_mm=float(1000 * data["table_depth"].max()),
                  max_contact_depth_mm=float(1000 * data["pair_depth"].max()),
                  peak_self_force_N=float(data["pair_normal"][:, PAIRS.index("film_self")].max()),
                  max_self_active_pairs=int(data["pairs"][:, PAIRS.index("film_self")].max()),
                  self_peak_depth_mm=1000 * float(data["pair_depth"][:, PAIRS.index("film_self")].max()),
                  any_bypass=bool(np.any(data["pairs"][:, PAIRS.index("payload_fixture")])),
                  contact_overflow=int(data["contact_overflow"].max()),
                  warnings=int(data["warnings"].max()), energy_peak_growth_J=float(energy.max() - energy[0]),
                  final_rms_mm_s=1000 * float(data["rms_speed"][-1]))
    crossed = result["crossing_pairs"] > 0
    violations = []
    if crossed:
        violations.append("nonadjacent_film_crossing")
    if data["edge_stretch"].max() > MAX_STRETCH:
        violations.append("edge_ratio_over_1.05")
    if max(data["pair_depth"].max(), data["table_depth"].max()) > MAX_PENETRATION:
        violations.append("penetration_over_1mm")
    if np.any(data["payload_outside"]) or np.any(data["payload_intersections"]):
        violations.append("payload_surface_crossing")
    if result["any_bypass"]:
        violations.append("fixture_payload_bypass")
    if result["warnings"] or result["contact_overflow"]:
        violations.append("solver_warning_or_overflow")
    if trial.case == "B01":
        quiet = np.maximum.accumulate(data["rms_speed"][::-1])[::-1] <= .005
        result["settling_time_s"] = float(time[np.argmax(quiet)]) if quiet.any() else None
        result["tail_rms_mm_s"] = 1000 * float(data["rms_speed"][tail].max()) if tail.any() else None
        if not tail.any() or data["rms_speed"][tail].max() > .005:
            violations.append("not_settled")
        if energy.max() - energy[0] > .05 * abs(energy[0]):
            violations.append("energy_growth")
    if trial.case.startswith("B02"):
        target = data["load"][:, trial.axis]
        response = data["clamp_force"][:, trial.axis]
        result.update(peak_strain=float(np.nanmax(data["strain"])),
                      residual_strain=float(data["strain"][-1]), peak_load_N=float(target.max()),
                      gauge_response=[dict(force_N=float(response[i]), strain=float(data["strain"][i]))
                                      for i in (np.argmin(abs(time - t)) for t in (.15, .31, .47, .64))])
        if last_time >= .64 - trial.dt:
            check = np.array([np.argmin(abs(time - t)) for t in (.15, .31, .47)])
            mismatch = np.abs(response[check] + target[check]) / target[check]
            if (not np.all(np.diff(data["strain"][check]) > 0)
                    or mismatch.max() > .05 or abs(result["residual_strain"]) > .002):
                violations.append("coupon_force_balance_or_residual")
    if trial.case == "B03-A":
        result.update(tip_final_mm=1000 * float(data["tip_z"][-1]),
                      tip_min_mm=1000 * float(data["tip_z"].min()))
    if trial.case.startswith("B04") and result["peak_self_force_N"] <= 0.:
        violations.append("no_effective_self_contact")
    if trial.case.startswith("B05-A"):
        weight = (trial.mass + float(data["masses"].sum())) * 9.81
        error = abs(float(data["wrench"][tail, len(FIXTURES), 2].mean()) + weight) / weight if tail.any() else None
        result["support_error_fraction"] = error
        if error is None or error > .05:
            violations.append("support_error")
        if np.any(data["pairs"][:, PAIRS.index("payload_table")]):
            violations.append("direct_payload_table")
    if trial.case == "B05-D":
        result["payload_lift_mm"] = float(1000 * (data["payload_floor"].max() - data["payload_floor"][0]))
        if result["payload_lift_mm"] < 20.:
            violations.append("boundary_lift_under_20mm")
    if trial.case.startswith(("B06", "B07")):
        z = data["fixtures"][:, 0, 2]
        force = data["wrench"][:, 0, 2]
        displacement = data["fixture_target"][0, 0, 2] - z
        loaded = np.flatnonzero(force > .05)
        result["probe_peak_normal_N"] = float(force.max())
        if force.max() > 5.:
            violations.append("probe_force_limit_exceeded")
        result["contact_onset_mm"] = float(displacement[loaded[0]] * 1000.) if len(loaded) else None
        result["payload_shift_mm"] = (float(1000 * np.linalg.norm(data["payload"][-1, :3] - data["payload"][0, :3]))
                                      if trial.mass else None)
        if trial.case.startswith("B06"):
            turnaround = int(np.argmax(displacement))
            for name, idx in (("loading", loaded[loaded < turnaround]), ("unloading", loaded[loaded >= turnaround])):
                span = np.ptp(displacement[idx]) if len(idx) else 0.
                result[f"{name}_stiffness_N_m"] = (float(np.polyfit(displacement[idx], force[idx], 1)[0])
                                                     if len(idx) >= 3 and span >= .001 else None)
            result["probe_net_work_J"] = float(np.trapezoid(force, displacement))
    result["violations"] = violations
    if outcome == "complete":
        result["status"] = "physics_gate_failure" if violations else "pilot_pass"
    return result


@dataclass
class ProbeFeedback:
    height: float = .041
    speed: float = 0.
    time: float = 0.
    stop_x: float = 0.
    monitored_peak: float = 0.
    stop_time: float | None = None
    stop_height: float = .041

    def update(self, time, commanded, normal, trigger_force=5.):
        self.monitored_peak = max(self.monitored_peak, normal)
        if self.stop_time is not None:
            return
        x, _, z = commanded["tools"]["contact_probe"][0]
        if normal >= trigger_force:
            self.stop_time, self.stop_height, self.stop_x = time, float(z), float(x)
        else:
            self.speed = float(np.clip(.012 * (normal - .50), -.03, .03))
            self.height, self.time = float(z), time


def run_trial(factory, trial, seed, output):
    print(f"{trial.case}: start", flush=True)
    rows = []
    probe_microsteps = []
    elapsed = perf_counter()
    outcome = "complete"
    error = None
    backend = None
    feedback = None
    try:
        backend = factory.reset(trial, seed)
        film = backend.envelope
        metric = MeshMetrics(film, closed=not trial.coupon)
        loaded = trial.loaded_nodes(film)
        fixed = trial.fixed_nodes(film)
        rest_gauge = (film["reference"][loaded, trial.axis].mean() -
                      film["reference"][fixed, trial.axis].mean()) if len(loaded) else 1.
        interval = max(1, round(.002 / trial.dt))
        steps = round(trial.duration / trial.dt)
        feedback = ProbeFeedback() if trial.case.startswith(("B06", "B07")) else None
        frame = action(trial, 0.)
        rows.append(sample(backend, trial, metric, 0., frame, rest_gauge))
        for step in range(1, steps + 1):
            frame = action(trial, (step - 1) * trial.dt, feedback)
            backend.step(frame)
            time = step * trial.dt
            if trial.gentle_probe and .20 <= (step - 1) * trial.dt <= .73:
                normal = backend.probe_normal_reaction()
                feedback.update(time, frame, normal, trigger_force=4.)
                probe_microsteps.append((time, frame["tools"]["contact_probe"][0][2], normal))
            triggered = trial.gentle_probe and feedback.stop_time == time
            if step % interval == 0 or step == steps or triggered:
                row = sample(backend, trial, metric, step * trial.dt, frame, rest_gauge)
                rows.append(row)
                if feedback is not None and not trial.gentle_probe:
                    feedback.update(step * trial.dt, frame, row["wrench"][0, 2])
                if row["warnings"] or row["contact_overflow"]:
                    raise FloatingPointError(f"solver warning/overflow at {step * trial.dt:.8f} s")
    except (ValueError, RuntimeError, FloatingPointError, NotImplementedError) as exc:
        error = traceback.format_exc()
        if isinstance(exc, NotImplementedError):
            outcome = "unsupported_backend_functionality"
        else:
            outcome = "numerical_failure" if backend is not None else "setup_failure"
    data = {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    if probe_microsteps:
        data["probe_microsteps"] = np.asarray(probe_microsteps)
    if backend is not None:
        film = backend.envelope
        data.update({key: film[key] for key in ("reference", "current", "faces", "edges", "panels", "masses")})
        if hasattr(backend, "xml"):
            xml = "\n".join(line.rstrip() for line in backend.xml.splitlines()) + "\n"
            (output / f"{trial.case}.xml").write_text(xml)
    result = evaluate(trial, outcome, data, perf_counter() - elapsed)
    result.update(config=asdict(trial), seed=seed)
    if backend is not None and feedback is not None:
        result["probe_force_limit_trigger_s"] = feedback.stop_time
        result["probe_monitored_peak_normal_N"] = float(feedback.monitored_peak)
        if trial.gentle_probe and feedback.monitored_peak > 5. and "probe_force_limit_exceeded" not in result["violations"]:
            result["violations"].append("probe_force_limit_exceeded")
            if result["status"] == "pilot_pass":
                result["status"] = "physics_gate_failure"
    if backend is not None:
        result["backend_config"] = backend.metadata
    if error:
        (output / f"{trial.case}.error.log").write_text(error)
        result["error"] = error.strip().splitlines()[-1]
    print(f"{trial.case}: {result['status']} {result.get('violations', [])}", flush=True)
    return data, result


def run_suite(args):
    factory = FlexBagBackend
    if args.backend == "vbd":
        from pe_physics_vbd import VBDBagBackend
        factory = VBDBagBackend
    timestep = args.dt if args.dt is not None else .00005 if args.backend == "vbd" else None
    if args.phase == "probe":
        if args.foundation is None:
            raise SystemExit("Probe phase requires a completed phase-one summary for this backend.")
        foundation = json.loads(args.foundation.read_text())
        expected = {trial.case for trial in phase_one(None)}
        evidence = json.loads(args.foundation.with_name("manifest.json").read_text())
        passed = {row["case"] for row in foundation
                  if row["status"] == "pilot_pass" and row.get("backend", evidence.get("backend")) == args.backend}
        if not expected.issubset(passed):
            raise SystemExit("Probe phase held at the Phase 1 physics gate.")
    trials = phase_one(timestep) if args.phase == "foundation" else phase_two(timestep)
    if args.phase == "probe-diagnostic":
        triad = {"B06-S0-P1", "B06-S2-P0", "B06-S2-P1"}
        trials = [trial for trial in trials if trial.case in triad]
    if args.cases:
        cases = set(args.cases.split(","))
        if cases - {trial.case for trial in trials}:
            raise SystemExit(f"Unknown trial: {cases - {trial.case for trial in trials}}")
        trials = [trial for trial in trials if trial.case in cases]
    if args.vbd_extensional_scale != 1.:
        if args.backend != "vbd" or any(not trial.case.startswith("B02") for trial in trials):
            raise SystemExit("The VBD membrane calibration applies to B02 material trials only.")
        trials = [replace(trial, vbd_extensional_scale=args.vbd_extensional_scale) for trial in trials]
    if args.gentle_probe:
        if args.backend != "flex" or len(trials) != 1 or trials[0].case != "B06-S2-P1":
            raise SystemExit("Gentle probe replay requires Flex B06-S2-P1 only.")
        trials = [replace(trials[0], gentle_probe=True, duration=.83,
                          initial_tools={"contact_probe": (0., 0., .031)})]
    if args.plane_table:
        if args.backend != "flex" or len(trials) != 1 or trials[0].case != "B04-U":
            raise SystemExit("The plane-table diagnostic requires Flex B04-U only.")
        trials = [replace(trials[0], plane_table=True)]
    if args.motion_time_scale != 1.:
        if len(trials) != 1 or trials[0].case not in ("B04-U", "B05-D"):
            raise SystemExit("Motion-time contrasts require B04-U or B05-D individually.")
        trials = [replace(trial, motion_time_scale=args.motion_time_scale,
                          duration=trial.duration * args.motion_time_scale) for trial in trials]
    if args.duration is not None:
        if args.duration <= 0. or args.duration < (args.dt or BASE.timestep):
            raise SystemExit("Duration must cover at least one positive physics step.")
        trials = [replace(trial, duration=min(args.duration, trial.duration)) for trial in trials]
    args.output.mkdir(parents=True, exist_ok=False)
    from importlib.metadata import version
    packages = ["mujoco", "mujoco-warp", "warp-lang", "numpy"]
    if args.backend == "vbd":
        packages.append("newton")
    save_json(args.output / "manifest.json", dict(machine=platform.node(), platform=platform.platform(),
              cpu=platform.processor(), started_at=datetime.now().astimezone().isoformat(),
              backend=args.backend, phase=args.phase, smoke=args.duration is not None, environment={p: version(p) for p in packages},
              revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [*Path(__file__).parent.glob("pe_physics*.py"),
                                      ROOT / "bag_sim/physical_parcel.py", ROOT / "bag_sim/newton_bag.py"]},
              solver_force_phase="solver contact/forces before integration, geometry after integration",
              pair_labels=PAIRS, wrench_labels=(*FIXTURES, "table", "payload"),
              probe_microstep_columns=("time_s", "commanded_height_m", "vertical_reaction_N"), seed=args.seed))
    save_json(args.output / f"{args.backend}_config.json", dict(base=asdict(BASE),
              thresholds=dict(edge_ratio=MAX_STRETCH, penetration_m=MAX_PENETRATION, tail_rms_speed_m_s=.005),
              trials=[asdict(t) for t in trials]))
    results = []
    arrays = {}
    for trial in trials:
        data, result = run_trial(factory, trial, args.seed, args.output)
        if args.duration is not None and result["status"] in ("pilot_pass", "physics_gate_failure"):
            result["status"] = "smoke_only"
        if args.phase == "probe-diagnostic":
            result["diagnostic_only"] = True
        arrays.update({f"{trial.case}/{k}": v for k, v in data.items()})
        results.append(result)
        # Keep data from completed cases even if a later solver process is interrupted.
        path = args.output / "raw_trajectories.npz"
        temporary = path.with_name("raw_trajectories.pending.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        save_json(args.output / "summary.json", results)
        headers = list(dict.fromkeys(k for r in results for k in r if k not in ("config", "gauge_response", "backend_config")))
        with (args.output / "trial_results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("flex", "vbd"), default="flex")
    parser.add_argument("--phase", choices=("foundation", "probe", "probe-diagnostic"), default="foundation")
    parser.add_argument("--foundation", type=Path, help="Passing Phase 1 summary.json from the same backend")
    parser.add_argument("--cases", help="Comma-separated condition labels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--vbd-extensional-scale", type=float, default=1.)
    parser.add_argument("--gentle-probe", action="store_true")
    parser.add_argument("--plane-table", action="store_true")
    parser.add_argument("--motion-time-scale", type=float, default=1.)
    parser.add_argument("--duration", type=float, help="Cap each condition for an entry smoke test")
    args = parser.parse_args()
    if args.dt is not None and (not np.isfinite(args.dt) or args.dt <= 0):
        parser.error("--dt must be positive and finite")
    if not np.isfinite(args.motion_time_scale) or args.motion_time_scale <= 0:
        parser.error("--motion-time-scale must be positive and finite")
    if not np.isfinite(args.vbd_extensional_scale) or args.vbd_extensional_scale <= 0:
        parser.error("--vbd-extensional-scale must be positive and finite")
    run_suite(args)


if __name__ == "__main__":
    main()
