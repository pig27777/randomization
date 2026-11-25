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

from bag_mvp import (DT, PAD_HALF, PROBE_D, PROBE_K, SmokeFlex,
                     jaw_core_overlap, make_trial, probe_command, quaternion)
from bag_sim.physical_parcel import Parcel
from bag_sim.wrap import point_triangle_distance, triangle_box_intersections, winding_number
from pe_physics import FIXTURES, MAX_STRETCH, PAD, PAIRS, load_envelope, save_json, vec
from pe_physics_protocol import CORNERS, ramp

STATES = {"Left": -.055, "Center": 0., "Right": .055}
TRIES = {"Try_Center": (-.010, .020), "Try_Left": (-.065, .020), "Try_Scan": (-.095, .190)}
SEED_ROOT = 20261126
B_REWORK_SEED_ROOT = 2026112610
SIZE = (.088, .068, .024)
AUDIT_DT = .005
SHELF_Y = .015
TRY_DURATION = 1.84
EFFORT_CAP = 2.
FORCE_LIMIT = 3.5
CLOSE_START = .80
LIFT_START = 1.82
LIFT_END = 2.72
HOLD_END = LIFT_END + 1.
SETDOWN_TIME = 1.14
SETDOWN_TRAVEL = .280
SUPPORT_PAUSE = .095
MAX_SUPPORT_PAUSES = 3
RELEASE_CLEARANCE = .042
RELEASE_SETTLE = .55
B_DURATION = HOLD_END + SETDOWN_TIME + MAX_SUPPORT_PAUSES * SUPPORT_PAUSE + .28 + RELEASE_SETTLE + AUDIT_DT
LIFT_DISTANCE = .185
JAW_TARGET = 7.5
JAW_FORCE_LIMIT = 9.
JAW_MASS = .20
JAW_FEEDBACK_DT = .0005

# Public bag-frame jaw midpoint, closing axis, and outward approach vector.
# The pad frame uses x=closing, z=approach, y=z×x.
GRASPS = {
    "g1": ((-.055, .023, .015), (.139, 0., .990), (0., 1., 0.), .092),
    "g2": ((0., .023, .015), (.139, 0., .990), (0., 1., 0.), .092),
    "g3": ((.055, .023, .015), (.139, 0., .990), (0., 1., 0.), .092),
    "g4": ((-.055, .030, .014), (0., 0., 1.), (0., 1., 0.), .088),
    "g5": ((0., .030, .014), (0., 0., 1.), (0., 1., 0.), .088),
    "g6": ((.055, .030, .014), (0., 0., 1.), (0., 1., 0.), .088),
    "g7": ((.156, .044, .005), (0., 0., 1.), (0., 1., 0.), .088),
    "g8": ((-.156, .044, .005), (0., 0., 1.), (0., 1., 0.), .088),
    "g9": ((.028, .035, .015), (0., -.070, .998), (0., .998, .070), .088),
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
        root = B_REWORK_SEED_ROOT if self.experiment == "B" and self.phase != "pilot" else SEED_ROOT
        return int(np.random.SeedSequence([root, exp, list(STATES).index(self.state),
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
            spec = replace(spec, duration=B_DURATION)
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
                world.find(f"body[@name='{name}']/inertial").set("mass", str(JAW_MASS))
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


class PublicPoseGrasp:
    def __init__(self, backend):
        self.backend = backend
        self.origin = None
        self.force = np.zeros(2)
        self.release_pose = None
        self.release_at = None
        self.load = 0.
        self.hold_load = None
        self.setdown = None
        self.support_pause = None
        self.support_waited = 0.
        self.next_support_depth = .003
        self.failed_support_pauses = 0
        self.min_span = 2 * PAD_HALF + .00025
        self.low_force_start = None
        self.reclose_locked = False
        self.names = ("left_pad", "right_pad")
        self.q = backend.tool_q[2:4, None] + np.arange(3)
        self.v = backend.tool_v[2:4, None] + np.arange(3)
        self.signs = np.array([-1., 1.])
        self.effort = np.zeros(2)

    def aim(self):
        mid, axis, outward, self.open_gap = GRASPS[self.backend.job.action]
        self.origin = self.backend.transform(np.array(mid))
        self.axis = self.backend.rotation @ (np.array(axis) / np.linalg.norm(axis))
        approach = self.backend.rotation @ (np.array(outward) / np.linalg.norm(outward))
        lateral = np.cross(approach, self.axis)
        frame = np.column_stack((self.axis, lateral, approach))
        self.orientation = quaternion(frame)
        pad_extent_y = np.abs(frame[1]) @ [PAD_HALF, *PAD / 2]
        # The support lip is public and fixed in world coordinates. Keep the
        # pad faces outside it when the bag pose brings a candidate inward.
        self.origin[1] = max(self.origin[1], SHELF_Y + pad_extent_y + .001)
        # Descend outside the nominal side envelope before inserting an open
        # horizontal pinch. A top approach has no horizontal component.
        self.standoff = self.origin + .125 * approach
        self.overhead = self.standoff.copy()
        self.overhead[2] = max(.090, self.standoff[2])
        self.approach = approach

    def lift_path(self, time):
        if time < HOLD_END:
            lift, rate = ramp(time, LIFT_START, LIFT_END)
            return LIFT_DISTANCE * lift, LIFT_DISTANCE * rate
        if self.hold_load is None:
            self.hold_load = self.load
            if self.hold_load < .45:
                self.release_at = HOLD_END
        if self.hold_load < .45:
            return LIFT_DISTANCE, 0.
        if self.setdown is None:
            progress, rate = ramp(time - self.support_waited, HOLD_END, HOLD_END + SETDOWN_TIME)
            depth = SETDOWN_TRAVEL * progress
            if (self.support_pause is None and depth >= self.next_support_depth
                    and self.load < .56 * self.hold_load
                    and self.failed_support_pauses < MAX_SUPPORT_PAUSES):
                self.support_pause = time, depth
            supported = False
            if self.support_pause is not None:
                pause_start, depth = self.support_pause
                if time < pause_start + SUPPORT_PAUSE:
                    return LIFT_DISTANCE - depth, 0.
                self.support_pause = None
                self.support_waited += time - pause_start
                supported = self.load < min(.45, .3 * self.hold_load)
                if not supported:
                    self.failed_support_pauses += 1
                    self.next_support_depth = depth + .012
                    return LIFT_DISTANCE - depth, 0.
            if supported or progress == 1.:
                self.setdown = time, LIFT_DISTANCE - depth, supported
                self.release_at = (round(time / AUDIT_DT) + 55) * AUDIT_DT
            else:
                return LIFT_DISTANCE - depth, -SETDOWN_TRAVEL * rate
        touch_time, touch_height, supported = self.setdown
        # Confirm support with the arm stationary, then back off before opening.
        # Feedback is the summed vertical pad reaction.
        backoff, rate = ramp(time, touch_time, touch_time + .25)
        clearance = RELEASE_CLEARANCE if supported else 0.
        return touch_height + clearance * backoff, clearance * rate

    def drive(self, time):
        backend = self.backend
        if time < .24:
            backend.drive({}, None)
            return
        if self.origin is None:
            self.aim()
        if time < CLOSE_START:
            descend, descend_speed = ramp(time, .24, .52)
            insert, insert_speed = ramp(time, .52, CLOSE_START)
            down = self.standoff - self.overhead
            forward = self.origin - self.standoff
            center = self.overhead + descend * down + insert * forward
            speed = descend_speed * down + insert_speed * forward
            offsets = self.signs * (PAD_HALF + self.open_gap / 2)
            backend.drive({name: (center + offset * self.axis, speed, self.orientation)
                           for name, offset in zip(self.names, offsets)}, None)
            return

        lift, lift_speed = self.lift_path(time)
        center = self.origin + [0., 0., lift]
        speed = np.array([0., 0., lift_speed])
        d = backend.d
        if self.release_at is not None and time >= self.release_at - 1e-10:
            if self.release_pose is None:
                self.release_pose = d.qpos[self.q].copy()
                self.effort[:] = 0.
            spread, spread_speed = ramp(time, self.release_at, self.release_at + .24)
            retreat, retreat_speed = ramp(time, self.release_at + .18, self.release_at + .44)
            open_pos = center + self.signs[:, None] * (PAD_HALF + self.open_gap / 2) * self.axis
            delta = open_pos - self.release_pose
            away = .075 * self.approach
            backend.drive({name: (start + spread * move + retreat * away,
                                  spread_speed * move + retreat_speed * away, self.orientation)
                           for name, start, move in zip(self.names, self.release_pose, delta)}, None)
            return

        # Prescribe the arm path and pad orientation. The closing-axis
        # positions and velocities remain dynamic under bounded motor effort.
        offset = (d.qpos[self.q] - center) @ self.axis
        normal_vel = d.qvel[self.v] @ self.axis
        target_vel = speed @ self.axis
        if offset[1] - offset[0] < self.min_span:
            offset = offset.mean() + self.signs * self.min_span / 2
            if normal_vel[1] < normal_vel[0]:
                normal_vel[:] = normal_vel.mean()
        tangent_speed = speed - target_vel * self.axis
        common = 850. * -offset.mean() + 20. * (target_vel - normal_vel.mean())
        inward_speed = -self.signs * (normal_vel - target_vel)
        grip = JAW_TARGET - 85. * inward_speed
        # Open under excess native reaction and prohibit inward travel at the
        # empty-tool hard stop. Neither branch reads the payload pose.
        over = self.force >= JAW_TARGET
        grip[over] = np.minimum(grip[over], JAW_TARGET - 2. * (self.force[over] - JAW_TARGET))
        self.effort = np.clip(common - self.signs * grip, -JAW_FORCE_LIMIT, JAW_FORCE_LIMIT)
        if offset[1] - offset[0] <= self.min_span + 1e-12 and self.effort[0] > self.effort[1]:
            self.effort[:] = self.effort.mean()
        backend.apply_action(dict(tools={
            name: (center + p * self.axis, tangent_speed + v * self.axis, self.orientation)
            for name, p, v in zip(self.names, offset, normal_vel)
        }, load=np.zeros(3), clamps={}))
        d.qfrc_applied[self.v] = self.effort[:, None] * self.axis
        Parcel.step(backend)

    def refresh_force(self, time):
        left, right = self.backend.tool_v[2:4]
        constraint = self.backend.d.qfrc_constraint
        self.force = np.maximum(0., [-constraint[left:left + 3] @ self.axis,
                                     constraint[right:right + 3] @ self.axis])
        measured_load = float(-(constraint[left + 2] + constraint[right + 2]))
        self.load += .025 * (measured_load - self.load)
        if LIFT_START <= time < HOLD_END and not self.reclose_locked:
            if self.force.max() >= 2.:
                self.low_force_start = None
            elif self.low_force_start is None:
                self.low_force_start = time
            elif time - self.low_force_start >= .015:
                # A lost grasp must not close again on the moving loose film.
                left_pos, right_pos = self.backend.d.qpos[self.q]
                self.min_span = max(self.min_span, (right_pos - left_pos) @ self.axis)
                self.reclose_locked = True


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
    grasp = PublicPoseGrasp(backend) if job.experiment == "B" else None
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
                check.update(jaw_N=grasp.force.tolist(), jaw_motor_N=grasp.effort.tolist(), load_N=grasp.load,
                             bilateral=bool(np.all(film & (grasp.force > 0.))),
                             reclose_locked=grasp.reclose_locked)
                if grasp.origin is not None:
                    check["gap_m"] = float(np.diff(d.qpos[grasp.q] @ grasp.axis)[0] - 2 * PAD_HALF)
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
        if i == steps or (grasp and grasp.release_at is not None
                          and time >= grasp.release_at + RELEASE_SETTLE - 1e-10):
            break
        if grasp:
            if grasp.origin is not None and i % round(JAW_FEEDBACK_DT / DT) == 0:
                grasp.refresh_force(time)
                if not np.isfinite(grasp.force).all():
                    invalid = ["nonfinite_force"]
                    break
            grasp.drive(time)
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
        h = [c for c in checks if LIFT_END - 1e-8 <= c["time"] <= HOLD_END + 1e-8]
        full_hold = len(h) == round(1. / AUDIT_DT) + 1 and all("clearance" in c for c in h)
        floor = checks[0].get("clearance")
        clearance = min(c["clearance"] for c in h) - floor if full_hold else None
        bilateral = sum(c["bilateral"] for c in h) / len(h) if full_hold else None
        release = next((c for c in checks if grasp.release_at is not None
                        and abs(c["time"] - grasp.release_at) < 1e-8), {})
        drop = (release["clearance"] - checks[-1]["clearance"]
                if "clearance" in release and "clearance" in checks[-1] else None)
        success = (not invalid and full_hold and clearance >= .010 and bilateral >= .95
                   and drop is not None and drop >= .010)
        closing = [c for c in checks if CLOSE_START <= c["time"] < LIFT_START]
        summary.update(success=bool(success) if not invalid else None, hold_min_lift_m=clearance,
                       bilateral_fraction=bilateral, release_drop_m=drop,
                       bilateral_close_fraction=sum(c["bilateral"] for c in closing) / len(closing) if closing else None,
                       release_time_s=grasp.release_at,
                       setdown_contact=None if grasp.setdown is None else bool(grasp.setdown[2]),
                       reclose_locked=grasp.reclose_locked,
                       hold_jaw_median_N=np.median([c["jaw_N"] for c in h], axis=0).tolist() if full_hold else None,
                       peak_jaw_motor_N=max(abs(f) for c in checks for f in c["jaw_motor_N"]),
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
    parser.add_argument("--pilot-actions", nargs="+", choices=tuple(GRASPS))
    args = parser.parse_args()
    if args.pilot_actions and (args.phase, args.experiment) != ("pilot", "B"):
        parser.error("--pilot-actions requires --phase pilot --experiment B")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for experiment, actions, n in (("A", TRIES, 15), ("B", GRASPS, 8)):
        if args.experiment not in (experiment, "both") or (args.phase == "heavy" and experiment == "A"):
            continue
        if args.phase == "pilot":
            pairs = list(zip(STATES, ("g4", "g5", "g6"))) if experiment == "B" else []
            pairs += [(state, act) for state in STATES for act in actions if experiment == "A"
                      or (state, act) in (("Left", "g1"), ("Center", "g2"), ("Right", "g3"),
                                          ("Center", "g7"), ("Center", "g8"), ("Right", "g9"))]
            jobs.extend(Job(experiment, state, act, 0, args.phase) for state, act in pairs
                        if not args.pilot_actions or act in args.pilot_actions)
        else:
            mass = .35 if args.phase == "heavy" else .20
            jobs.extend(Job(experiment, state, act, rep, args.phase, mass) for rep in range(n)
                        for state in STATES for act in actions)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, status in pool.imap_unordered(run_job, ((job, args.output) for job in jobs)):
            print(name, status, flush=True)


if __name__ == "__main__":
    main()
