"""Paired probe histories and held-out grasp trials on the PE Flex fixture."""
import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from math import isfinite
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bag_mvp import (DT, PAD_HALF, PROBE_D, PROBE_K, Grasp, SmokeFlex,
                     jaw_core_overlap, make_trial, probe_command, quaternion)
from bag_sim.physical_parcel import Parcel
from bag_sim.wrap import point_triangle_distance, triangle_box_intersections, winding_number
from pe_physics import FIXTURES, MAX_STRETCH, PAD, PAIRS, load_envelope, save_json, vec
from pe_physics_protocol import CORNERS, ramp

STATES = {"Left": -.055, "Center": 0., "Right": .055}
TRIES = {"Try_Center": (-.010, .020), "Try_Left": (-.065, .020), "Try_Scan": (-.095, .190)}
SEED_ROOT = 20261126
SIZE = (.088, .068, .024)
AUDIT_DT = .005
SHELF_Y = .015
TRY_DURATION = 1.84
EFFORT_CAP = 2.
FORCE_LIMIT = 3.5
CLOSE_DWELL = .58
BASE_HOLD_START = 1.74
JAW_TARGET = 4.
JAW_FEEDBACK_DT = .0005

# Public bag-frame jaw midpoint, closing axis, and outward approach vector.
# The pad frame uses x=closing, z=approach, y=z×x.
GRASPS = {
    "g1": ((-.055, 0., .019), (0., 1., 0.), (0., 0., 1.), .116),
    "g2": ((0., 0., .019), (0., 1., 0.), (0., 0., 1.), .116),
    "g3": ((.055, 0., .019), (0., 1., 0.), (0., 0., 1.), .116),
    "g4": ((-.055, .030, .014), (0., 0., 1.), (0., 1., 0.), .046),
    "g5": ((0., .030, .014), (0., 0., 1.), (0., 1., 0.), .046),
    "g6": ((.055, .030, .014), (0., 0., 1.), (0., 1., 0.), .046),
    "g7": ((.166, .050, .005), (0., 0., 1.), (0., 1., 0.), .030),
    "g8": ((-.166, .050, .005), (0., 0., 1.), (0., 1., 0.), .030),
    "g9": ((.028, .034, .015), (0., -.156, .988), (0., .988, .156), .048),
}
CHANNELS = tuple(f"{field}_{axis}" for field in ("position_m", "velocity_m_s", "error_m")
                 for axis in "xyz") + ("applied_z_N", "reaction_x_N", "reaction_y_N", "reaction_z_N",
                                       "command_x_m", "command_y_m", "command_z_m", "time_s")
EFFORT = slice(9, 13)


@dataclass(frozen=True)
class Job:
    experiment: str
    state: str
    action: str
    replicate: int
    phase: str
    mass: float = .20

    @property
    def seed(self):
        exp = {"A": 1, "B": 2}[self.experiment] + (10 if self.phase == "pilot" else 0)
        return int(np.random.SeedSequence([SEED_ROOT, exp, list(STATES).index(self.state),
                                           self.replicate]).generate_state(1)[0])

    @property
    def id(self):
        return f"{self.experiment}-{self.state}-{self.replicate:02d}-{self.seed}"

    @property
    def split(self):
        if self.experiment == "A":
            return "cv"
        return "selection" if self.replicate < 4 else "evaluation"


@lru_cache
def public_envelope(cfg):
    return load_envelope(replace(cfg, payload_size=(SIZE[0] + .110, *SIZE[1:]),
                                 payload_offset_xy=(0., 0.)))


class PairedFlex(SmokeFlex):
    def __init__(self, job):
        self.job = job
        rng = np.random.default_rng(job.seed)
        self.xy = rng.uniform(-.005, .005, size=2)
        self.yaw = np.deg2rad(rng.uniform(-2., 2.))
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        self.shift = np.r_[self.xy, 0.]
        if job.experiment == "A":
            spec = make_trial("try-center", DT)
            start, _ = TRIES[job.action]
            spec = replace(spec, duration=TRY_DURATION, initial_tools={"contact_probe": (start, 0., .045)})
            variant = "baseline"
        else:
            spec = make_trial("through", DT, "near-seam")
            spec = replace(spec, duration=spec.duration + CLOSE_DWELL)
            variant = "near-seam"
        super().__init__(replace(spec, mass=job.mass), variant=variant)
        self.core_q = self.m.joint("payload_free").qposadr[0]
        self.rest_lengths = np.linalg.norm(np.diff(
            self.envelope["reference"][self.envelope["edges"]], axis=1)[:, 0], axis=1)
        self.audit_collision_mask()

    def load_envelope(self, cfg):
        envelope = public_envelope(cfg).copy()
        envelope["current"] = self.transform(envelope["current"])
        self.public_shape_sha = sha256(public_envelope(cfg)["current"].tobytes()).hexdigest()
        envelope["payload_origin"] = self.transform(envelope["payload_origin"] + [STATES[self.job.state], 0., 0.])
        return envelope

    def transform(self, points):
        return points @ self.rotation.T + self.shift

    def fixture_xml(self, xml):
        tree = ET.fromstring(super().fixture_xml(xml))
        world = tree.find("worldbody")
        world.find("flexcomp").set("rgba", ".72 .56 .36 1")
        core = world.find("body[@name='payload_0']")
        core.set("quat", vec(quaternion(self.rotation)))
        core.find("geom").set("rgba", ".85 .50 .20 0")
        if self.job.experiment == "B":
            h = (.40 + SHELF_Y) / 2
            world.find("geom[@name='table']").attrib.update(
                type="box", pos=vec([0., SHELF_Y - h, -.009]), size=vec([.40, h, .009]))
            ET.SubElement(world, "geom", name="floor", type="plane", size=".5 .5 .03",
                          pos="0 0 -.20", contype="1", conaffinity="10")
            for name in ("left_pad", "right_pad"):
                world.find(f"body[@name='{name}']/geom").set("solref", ".002 1")
        return ET.tostring(tree, encoding="unicode")

    def audit_collision_mask(self):
        core = self.m.geom("payload_geom_0").id
        tools = [self.m.geom(name).id for name in FIXTURES]
        ct, ca = self.m.geom_contype, self.m.geom_conaffinity
        if self.m.neq or self.m.npair or np.any((ct[tools] & ca[core]) | (ct[core] & ca[tools])):
            raise RuntimeError("Fixture/core collision exclusion or absence of constraints violated")
        if not all((ct[g] & self.m.flex_conaffinity[0]) or (self.m.flex_contype[0] & ca[g])
                   for g in [core, *tools]):
            raise RuntimeError("Expected native fixture/film and core/film contacts are disabled")
        self.core_geom = core
        self.fixture_geoms = frozenset(tools)

    def direct_fixture_contact(self):
        # The shared generic payload_fixture counter also groups payload
        # contact with an additional floor. Audit the actual loaded pair.
        for contact in self.d.contact:
            if contact.efc_address < 0:
                continue
            first, second = contact.geom
            if ((first == self.core_geom and second in self.fixture_geoms)
                    or (second == self.core_geom and first in self.fixture_geoms)):
                return True
        return False


class PublicPoseGrasp(Grasp):
    def __init__(self, backend, edge):
        super().__init__(backend, edge)
        # The reference exposed-underface variant supplies the faster
        # force-feedback closure, with a longer common lift for loose film.
        self.variant = "exposed-underface"
        self.hold_start = BASE_HOLD_START
        self.release = self.hold_start + 1.
        self.measured_hold_start = self.hold_start + CLOSE_DWELL
        self.measured_release = self.release + CLOSE_DWELL

    def aim(self):
        mid, axis, outward, self.open_gap = GRASPS[self.backend.job.action]
        self.origin = self.backend.transform(np.array(mid))
        self.axis = self.backend.rotation @ (np.array(axis) / np.linalg.norm(axis))
        approach = self.backend.rotation @ (np.array(outward) / np.linalg.norm(outward))
        lateral = np.cross(approach, self.axis)
        self.orientation = quaternion(np.column_stack((self.axis, lateral, approach)))
        self.reach = self.origin + .045 * approach
        self.min_gap = .00025
        self.jaw_gap = self.open_gap

    def command(self, time, dt):
        # Use the same additional pre-lift closing window for every pose.
        virtual_time = time - min(max(0., time - .87), CLOSE_DWELL)
        # Re-center the reference controller's 8 N error around the common
        # 4 N target. Logged reactions remain the measured native forces.
        force = self.force
        self.force = force + (8. - JAW_TARGET)
        try:
            moves = super().command(virtual_time, dt)
        finally:
            self.force = force
        lift, speed = ramp(virtual_time, .87, BASE_HOLD_START)
        extra_position, extra_velocity = [0., 0., .115 * lift], [0., 0., .115 * speed]
        return {name: (position + extra_position, velocity + extra_velocity, orientation)
                for name, (position, velocity, orientation) in moves.items()}

    def refresh_force(self):
        left, right = self.backend.tool_v[2:4]
        constraint = self.backend.d.qfrc_constraint
        self.force = np.maximum(0., [-constraint[left:left + 3] @ self.axis,
                                     constraint[right:right + 3] @ self.axis])


def audit(backend, grasp):
    d, env = backend.d, backend.envelope
    if not all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.qacc)):
        return {"invalid": ["nonfinite"]}
    if any(w.number for w in d.warning):
        return {"invalid": ["solver_warning"]}
    q, _ = backend.get_bag_state()
    core = d.qpos[backend.core_q:backend.core_q + 7]
    rotation = np.empty((3, 3))
    mujoco.mju_quat2Mat(rotation.ravel(), core[3:])
    half = np.array(backend.spec.size) / 2
    corners = CORNERS * half @ rotation.T + core[:3]
    local = (q - core[:3]) @ rotation
    tris = local[env["faces"]]
    inset = half - .002
    near = np.all((tris.min(axis=1) < inset) & (tris.max(axis=1) > -inset), axis=1)
    inside = int(triangle_box_intersections(local, env["faces"][near], inset, np.zeros(3)).sum())
    outside = np.abs(winding_number(q, env["faces"], np.vstack((core[:3], corners)))) < .5
    exposed = max(point_triangle_distance(corners[outside[1:]], q, env["faces"]), default=0.)
    length = np.linalg.norm(np.diff(q[env["edges"]], axis=1)[:, 0], axis=1)
    stretch = float(np.max(length / backend.rest_lengths))
    invalid = []
    if inside:
        invalid.append("film_inset_core")
    if outside[0] or exposed > .002:
        invalid.append("core_exposed")
    if stretch > MAX_STRETCH:
        invalid.append("film_edge_ratio")
    if grasp and grasp.origin is not None:
        fixtures = backend.get_fixture_state()
        if jaw_core_overlap(core[None], fixtures[None], backend.spec.size):
            invalid.append("jaw_inset_core")
        table = backend.m.geom("table")
        table_pose = np.r_[table.pos, 1., 0., 0., 0.]
        if jaw_core_overlap(table_pose[None], fixtures[None], 2 * table.size):
            invalid.append("jaw_inset_table")
    return dict(invalid=invalid, clearance=float(corners[:, 2].min()),
                exposed_m=float(exposed), inset_triangles=inside, edge_ratio=stretch)


@lru_cache
def probe_schedule(name):
    times = np.arange(round(TRY_DURATION / DT) + 1) * DT
    _, _, z = probe_command(times)
    start, span = TRIES[name]
    progress, speed = ramp(times, .62, 1.54)
    return np.column_stack((start + span * progress, span * speed, z))


def jaw_forces(backend, grasp):
    if grasp.origin is None:
        return np.zeros(2), np.zeros(2, dtype=bool)
    contact = backend.get_contact_wrench()
    forces = contact["wrench"][2:4, :3] @ grasp.axis
    film = contact["pairs"][[PAIRS.index("film_left_pad"), PAIRS.index("film_right_pad")]] > 0
    return np.maximum(0., [-forces[0], forces[1]]), film


def simulate(job):
    start = perf_counter()
    backend = PairedFlex(job)
    grasp = PublicPoseGrasp(backend, SHELF_Y) if job.experiment == "B" else None
    d = backend.d
    q, v = backend.tool_q[0], backend.tool_v[0]
    checks, history = [], []
    initial = np.r_[d.qpos[backend.film_qpos].ravel(), d.qpos[backend.core_q:backend.core_q + 7]]
    peak = 0.
    contact_peak = 0.
    invalid = []
    command = np.array([TRIES[job.action][0], 0., .045]) if not grasp else None
    schedule = probe_schedule(job.action) if not grasp else None
    steps = round(backend.spec.duration / DT)
    for i in range(steps + 1):
        time = i * DT
        if i % round(AUDIT_DT / DT) == 0:
            check = audit(backend, grasp)
            check["time"] = time
            if grasp:
                grasp.force, film = jaw_forces(backend, grasp)
                check.update(jaw_N=grasp.force.tolist(), bilateral=bool(np.all(film & (grasp.force > 0.))))
                if backend.direct_fixture_contact():
                    check["invalid"].append("direct_core_force")
                if not np.isfinite(grasp.force).all():
                    check["jaw_N"] = None
                    check["invalid"].append("nonfinite_force")
            checks.append(check)
            invalid = check["invalid"]
            if invalid:
                break
        if not grasp and i % round(.04 / DT) == 0:
            pos = d.qpos[q:q + 3]
            history.append(np.r_[pos, d.qvel[v:v + 3], command - pos, d.qfrc_applied[v + 2],
                                 d.qfrc_constraint[v:v + 3], command, time])
        if i == steps:
            break
        if grasp:
            if grasp.origin is not None and i % round(JAW_FEEDBACK_DT / DT) == 0:
                grasp.refresh_force()
                if not np.isfinite(grasp.force).all():
                    invalid = ["nonfinite_force"]
                    break
            backend.drive(grasp.command(time, DT), None)
        else:
            x, vx, z = schedule[i]
            command = np.array([x, 0., z])
            d.qpos[q:q + 2] = [x, 0.]
            d.qvel[v:v + 2] = [vx, 0.]
            effort = PROBE_K * (z - d.qpos[q + 2]) - PROBE_D * d.qvel[v + 2]
            if not isfinite(effort):
                invalid = ["nonfinite_effort"]
                break
            d.qfrc_applied[v + 2] = min(EFFORT_CAP, max(-EFFORT_CAP, effort))
            Parcel.step(backend)
            peak = max(peak, abs(float(d.qfrc_applied[v + 2])))
            reaction = float(d.qfrc_constraint[v + 2])
            contact_peak = max(contact_peak, abs(reaction))
            if not isfinite(reaction) or abs(reaction) > FORCE_LIMIT:
                invalid = ["nonfinite_force" if not isfinite(reaction) else "try_force_limit"]
                break

    summary = dict(experiment=job.experiment, state=job.state, action=job.action, split=job.split,
                   replicate=job.replicate, seed=job.seed, initial_state_id=job.id,
                   physics_validity=not invalid, invalid_reasons=invalid,
                   xy_m=backend.xy.tolist(), yaw_deg=float(np.rad2deg(backend.yaw)),
                   core_size_m=backend.spec.size, core_mass_kg=backend.spec.mass,
                   public_shape_sha256=backend.public_shape_sha,
                   initial_mesh_sha256=sha256(backend.envelope["current"].tobytes()).hexdigest(),
                   initial_qpos_sha256=sha256(initial.tobytes()).hexdigest(),
                   end_time_s=time, runtime_s=perf_counter() - start,
                   peak_edge_ratio=max(c.get("edge_ratio", 0.) for c in checks),
                   max_exposed_m=max(c.get("exposed_m", 0.) for c in checks))
    if not grasp:
        summary.update(peak_reaction_N=contact_peak, peak_effort_N=peak)
    else:
        h = [c for c in checks if grasp.measured_hold_start - 1e-8 <= c["time"] <= grasp.measured_release + 1e-8]
        full_hold = len(h) == round(1. / AUDIT_DT) + 1 and all("clearance" in c for c in h)
        floor = checks[0].get("clearance")
        clearance = min(c["clearance"] for c in h) - floor if full_hold else None
        bilateral = sum(c["bilateral"] for c in h) / len(h) if full_hold else None
        drop = h[-1]["clearance"] - checks[-1]["clearance"] if full_hold and "clearance" in checks[-1] else None
        success = not invalid and full_hold and clearance >= .010 and bilateral >= .95 and drop >= .010
        summary.update(success=bool(success) if not invalid else None, hold_min_lift_m=clearance,
                       bilateral_fraction=bilateral, release_drop_m=drop,
                       commanded_pose_world=None if grasp.origin is None else
                       np.r_[grasp.origin, grasp.orientation].tolist(),
                       peak_jaw_N=max((max(c.get("jaw_N") or [0.]) for c in checks), default=0.))
    return job, summary, np.asarray(history), checks


def run_job(args):
    job, path = args
    name = f"{job.id}_{job.action}"
    final = path / f"{name}.json"
    if final.exists():
        return name, "cached"
    job, summary, history, checks = simulate(job)
    if job.experiment == "A":
        np.save(path / f"{name}.npy", history)
    elif job.phase == "pilot":
        summary["audit"] = checks
    pending = path / f"{name}.json.tmp"
    save_json(pending, summary)
    pending.replace(final)
    status = ",".join(summary["invalid_reasons"]) or str(summary.get("success", "valid"))
    return name, f"{status} ({summary['runtime_s']:.1f}s)"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=("pilot", "formal", "heavy"), required=True)
    parser.add_argument("--experiment", choices=("A", "B", "both"), default="both")
    parser.add_argument("--backend", choices=("cpu-flex",), default="cpu-flex")
    parser.add_argument("--workers", type=int, default=min(22, len(os.sched_getaffinity(0))))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for experiment, actions, n in (("A", TRIES, 15), ("B", GRASPS, 8)):
        if args.experiment not in (experiment, "both") or (args.phase == "heavy" and experiment == "A"):
            continue
        if args.phase == "pilot":
            pairs = [(state, act) for state in STATES for act in actions] if experiment == "A" else [
                ("Center", act) for act in actions]
            jobs.extend(Job(experiment, state, act, 0, args.phase) for state, act in pairs)
        else:
            mass = .35 if args.phase == "heavy" else .20
            jobs.extend(Job(experiment, state, act, rep, args.phase, mass) for rep in range(n)
                        for state in STATES for act in actions)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, status in pool.imap_unordered(run_job, ((job, args.output) for job in jobs)):
            print(name, status, flush=True)


if __name__ == "__main__":
    main()
