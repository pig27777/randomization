"""Continuous Try and grasp collection on the audited PE Flex reference fixture."""
import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bag_mvp import PROBE_D, PROBE_K, PROBE_MASS, quaternion
from pe_physics import save_json, vec
from pe_physics_protocol import ramp

SOURCE = Path(__file__).resolve().parents[1] / "runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1/source"
sys.path.insert(0, str(SOURCE))
from rss_exp_ab_94d44457 import (AUDIT_DT, DT, EFFORT_CAP, FORCE_LIMIT, GRASPS, JAW_FEEDBACK_DT,
                                 JAW_TARGET, SHELF_Y, STATES, TRIES, TRY_DURATION, PairedFlex,
                                 Parcel, PublicPoseGrasp, audit, jaw_forces, probe_schedule)

TICK = .04
HOVER = (0., -.500, .230)
PROBE_END = 2.52
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
CANDIDATES = GRASPS | {
    "g10": ((.003, .030, .014), (0., 0., 1.), (0., 1., 0.), .046),
    "g11": ((.025, .034, .015), (0., -.125, .992), (0., .992, .125), .047),
    "g12": ((-.003, .032, .014), (0., 0., 1.), (0., 1., 0.), .046),
}
PROPOSAL = np.array([.045, .045, .045, .025, .20, .025, .16, .17, .08, .09, .06, .055])
FINAL_PROPOSAL = np.array([.065, .11, .125, .01, .17, .01, .10, .08, .02, .03, .16, .12])
SCAN_CLEARANCE_OFFSET = .003
FINAL_PHASES = ("pilot-5", "collect", "closed", "matched")
STAGE = {"pilot": 1, "collect": 2, "closed": 3, "matched": 4, "pilot-2": 5, "pilot-3": 6,
         "pilot-4": 7, "pilot-5": 8}
SUMMARIES = ("time_s", "clearance_m", "edge_ratio", "exposed_m", "inset_triangles",
             "left_normal_N", "right_normal_N", "parked_probe_N", "bilateral")


@dataclass(frozen=True)
class Job:
    state: str
    probe: str
    grasp: str | None
    replicate: int
    phase: str
    experiment: str = "B"
    mass: float = .20

    @property
    def seed(self):
        state = list(STATES).index(self.state) if self.phase != "matched" else 0
        return int(np.random.SeedSequence([43061126, STAGE[self.phase], state,
                                           self.replicate]).generate_state(1)[0])

    @property
    def id(self):
        return f"{self.phase}-{self.state}-{self.replicate:03d}-{self.seed}"


class FlexMacro(PairedFlex):
    def fixture_xml(self, xml):
        tree = ET.fromstring(super().fixture_xml(xml))
        probe = tree.find("worldbody/body[@name='contact_probe']")
        probe.set("pos", vec([TRIES[self.job.probe][0], 0., .045]))
        mass = probe.find("inertial")
        mass.set("mass", str(PROBE_MASS))
        mass.set("diaginertia", "8e-7 8e-7 8e-7")
        return ET.tostring(tree, encoding="unicode")

    def integration_state(self):
        state = np.empty(mujoco.mj_stateSize(self.m, STATE_SPEC))
        mujoco.mj_getState(self.m, self.d, state, STATE_SPEC)
        return state

    def restore(self, state):
        self.d = mujoco.MjData(self.m)
        mujoco.mj_setState(self.m, self.d, state, STATE_SPEC)
        mujoco.mj_forward(self.m, self.d)

    def visual(self, initial):
        points = self.envelope["current"] if initial else self.get_bag_state()[0]
        local = (points - self.shift) @ self.rotation
        low, high = local.min(axis=0), local.max(axis=0)
        return dict(center=self.transform((low + high) / 2).tolist(), rotation=self.rotation.tolist(),
                    half_extent=((high - low) / 2).tolist(), source="fixture_geometry")

    def candidates(self):
        result = []
        for mid, axis, outward, width in CANDIDATES.values():
            axis, outward = (self.rotation @ (np.array(u) / np.linalg.norm(u)) for u in (axis, outward))
            result.append(np.r_[self.transform(np.array(mid)), axis, np.cross(outward, axis), width,
                                JAW_TARGET])
        return np.asarray(result)


class CandidateGrasp(PublicPoseGrasp):
    def __init__(self, backend, candidate):
        super().__init__(backend, SHELF_Y)
        self.candidate = candidate

    def aim(self):
        self.origin = self.candidate[:3]
        self.axis, lateral = self.candidate[3:9].reshape(2, 3)
        outward = np.cross(self.axis, lateral)
        self.orientation = quaternion(np.column_stack((self.axis, lateral, outward)))
        self.open_gap = self.candidate[9]
        self.reach = self.origin + .045 * outward
        self.min_gap = .00025
        self.jaw_gap = self.open_gap


@lru_cache
def schedule(name, offset_m):
    reference = probe_schedule(name)
    first = np.column_stack((reference[:, :2], np.zeros((len(reference), 2)), reference[:, 2]))
    if offset_m:
        times = np.arange(len(first)) * DT
        press, _ = ramp(times, .18, .58)
        retract, _ = ramp(times, 1.62, 1.84)
        first[:, 4] += offset_m * (press - retract)
    t = np.arange(len(first), round(PROBE_END / DT) + 1) * DT
    progress, speed = ramp(t, TRY_DURATION, 2.32)
    x, _, _, _, z = first[-1]
    return np.vstack((first, np.column_stack((x + (HOVER[0] - x) * progress,
                                             (HOVER[0] - x) * speed,
                                             HOVER[1] * progress, HOVER[1] * speed,
                                             z + (HOVER[2] - z) * progress))))


def run_probe(backend):
    d = backend.d
    q, v = backend.tool_q[0], backend.tool_v[0]
    guarded = backend.job.phase in FINAL_PHASES
    offset = SCAN_CLEARANCE_OFFSET if guarded and backend.job.probe == "Try_Scan" else 0.
    commands = schedule(backend.job.probe, offset)
    history, invalid = [], []
    peak = 0.
    command = commands[0, [0, 2, 4]]
    for i, (x, vx, y, vy, z) in enumerate(commands):
        t = i * DT
        if i % round(AUDIT_DT / DT) == 0:
            invalid = audit(backend, None)["invalid"]
            if backend.direct_fixture_contact():
                invalid.append("direct_core_force")
            if invalid:
                break
        if i % round(TICK / DT) == 0:
            pos = d.qpos[q:q + 3]
            history.append(np.r_[pos, d.qvel[v:v + 3], command - pos, d.qfrc_applied[v + 2],
                                 d.qfrc_constraint[v:v + 3], command, t])
        if i == len(commands) - 1:
            break
        command = np.array([x, y, z])
        d.qpos[q:q + 2] = [x, y]
        d.qvel[v:v + 2] = [vx, vy]
        effort = PROBE_K * (z - d.qpos[q + 2]) - PROBE_D * d.qvel[v + 2]
        if not isfinite(effort):
            invalid = ["nonfinite_effort"]
            break
        downward_cap = EFFORT_CAP
        if guarded:
            previous_reaction = abs(float(d.qfrc_constraint[v + 2]))
            if previous_reaction > EFFORT_CAP:
                fraction = max(0., (FORCE_LIMIT - previous_reaction) / (FORCE_LIMIT - EFFORT_CAP))
                downward_cap *= fraction
        d.qfrc_applied[v + 2] = min(EFFORT_CAP, max(-downward_cap, effort))
        Parcel.step(backend)
        reaction = float(d.qfrc_constraint[v + 2])
        if not isfinite(reaction):
            invalid = ["nonfinite_force"]
            break
        peak = max(peak, abs(reaction))
        if peak > FORCE_LIMIT:
            invalid = ["try_force_limit"]
            break
    hover_error = float(np.linalg.norm(d.qpos[q:q + 3] - HOVER))
    if not invalid and (hover_error > .001 or np.linalg.norm(d.qvel[v:v+3]) > .010):
        invalid = ["hover_tracking"]
    return dict(invalid_reasons=invalid, peak_reaction_N=peak, end_time_s=t,
                hover_error_m=hover_error, clearance_offset_m=offset), np.asarray(history).reshape(-1, 17)


def run_grasp(backend, candidate):
    grasp = CandidateGrasp(backend, candidate)
    checks, trace = [], []
    invalid = []
    probe_v = backend.tool_v[0]
    probe_peak = 0.
    # Hold the probe safely at the neutral hover throughout grasp and release.
    parked = {"contact_probe": (np.array(HOVER), np.zeros(3), (1., 0., 0., 0.))}
    steps = round(backend.spec.duration / DT)
    for i in range(steps + 1):
        t = i * DT
        if i % round(AUDIT_DT / DT) == 0:
            check = audit(backend, grasp)
            grasp.force, film = jaw_forces(backend, grasp)
            invalid = check["invalid"]
            if backend.direct_fixture_contact():
                invalid.append("direct_core_force")
            if not np.isfinite(grasp.force).all():
                invalid.append("nonfinite_force")
            probe_force = float(np.linalg.norm(backend.d.qfrc_constraint[probe_v:probe_v + 3]))
            if not isfinite(probe_force):
                invalid.append("nonfinite_force")
            elif probe_force > .05:
                invalid.append("parked_probe_contact")
            probe_peak = max(probe_peak, probe_force)
            bilateral = bool(np.all(film & (grasp.force > 0)))
            checks.append([t, *(check.get(k, np.nan) for k in
                                  ("clearance", "edge_ratio", "exposed_m", "inset_triangles")),
                           *grasp.force, probe_force, bilateral])
            if invalid:
                break
        if i % round(TICK / DT) == 0 or i == steps:
            trace.append(np.r_[t, backend.get_payload_state()[0], backend.get_fixture_state()[2:4].ravel()])
        if i == steps:
            break
        if grasp.origin is not None and i % round(JAW_FEEDBACK_DT / DT) == 0:
            grasp.refresh_force()
            if not np.isfinite(grasp.force).all():
                invalid = ["nonfinite_force"]
                break
        backend.drive(parked | grasp.command(t, DT), None)
    checks = np.asarray(checks)
    h = checks[(checks[:, 0] >= grasp.measured_hold_start - 1e-8)
               & (checks[:, 0] <= grasp.measured_release + 1e-8)]
    full_hold = len(h) == round(1. / AUDIT_DT) + 1 and np.isfinite(h).all()
    lift = float(h[:, 1].min() - checks[0, 1]) if full_hold else None
    bilateral = float(h[:, -1].mean()) if full_hold else None
    drop = float(h[-1, 1] - checks[-1, 1]) if full_hold and np.isfinite(checks[-1, 1]) else None
    success = None if invalid else bool(full_hold and lift >= .010 and bilateral >= .95 and drop >= .010)
    summary = dict(task_success=success, physics_validity=not invalid, invalid_stage="grasp" if invalid else None,
                   invalid_reasons=invalid, hold_min_lift_m=lift, bilateral_fraction=bilateral,
                   release_drop_m=drop, peak_parked_probe_N=probe_peak, end_time_s=t)
    return summary, np.asarray(trace).reshape(-1, 22), checks


def make_initial(job):
    start = perf_counter()
    backend = FlexMacro(job)
    public = dict(v0=backend.visual(initial=True), public_shape_sha256=backend.public_shape_sha,
                  initial_mesh_sha256=sha256(backend.envelope["current"].tobytes()).hexdigest(),
                  candidates=backend.candidates().tolist())
    probe, history = run_probe(backend)
    state = None if probe["invalid_reasons"] else backend.integration_state()
    fingerprint = None if state is None else sha256(state.tobytes()).hexdigest()
    record = dict(episode_id=job.id, initial_state_id=job.id, probe_state_id=f"{job.id}-try",
                  state=job.state, seed=job.seed, probe=job.probe, grasp=job.grasp, **public,
                  phase=job.phase, replicate=job.replicate,
                  hover_xyz_m=HOVER,
                  probe_summary=probe, post_try_sha256=fingerprint,
                  v1=None if state is None else backend.visual(initial=False),
                  runtime_s=perf_counter() - start)
    return backend, record, history, state


def collect(args):
    job, out = args
    path = out / f"{job.id}.json"
    if path.exists():
        return job.id, "cached"
    start = perf_counter()
    b, row, history, state = make_initial(job)
    trace = np.empty((0, 22))
    audit_frames = np.empty((0, len(SUMMARIES)))
    if state is None:
        row.update(task_success=None, physics_validity=False, invalid_stage="probe",
                   invalid_reasons=row["probe_summary"]["invalid_reasons"])
    elif job.grasp is not None:
        candidate = np.asarray(row["candidates"])[list(CANDIDATES).index(job.grasp)]
        result, trace, audit_frames = run_grasp(b, candidate)
        row.update(result)
    row["runtime_s"] = perf_counter() - start
    np.savez_compressed(path.with_suffix(".npz"), history=history,
                        padding_mask=np.arange(round(PROBE_END / TICK) + 1) >= len(history),
                        post_try_state=np.empty(0) if state is None else state,
                        grasp_trajectory=trace, grasp_audit=audit_frames,
                        audit_channels=SUMMARIES)
    pending = path.with_suffix(".tmp")
    save_json(pending, row)
    pending.replace(path)
    return job.id, f"{row.get('task_success')} {row.get('invalid_reasons', [])} {row['runtime_s']:.1f}s"


def schedule_jobs(phase, n):
    rng = np.random.default_rng(11039 + (1 if phase == "pilot-2" else STAGE[phase]))
    if phase in ("pilot-5", "collect"):
        common = np.searchsorted(FINAL_PROPOSAL.cumsum(), (np.arange(n) + .5) / n)
        draws = {state: iter(rng.permutation(common)) for state in STATES}
    else:
        total = len(STATES) * n
        probabilities = FINAL_PROPOSAL if phase == "pilot-4" else PROPOSAL
        ids = np.searchsorted(probabilities.cumsum(), (np.arange(total) + .5) / total)
        rng.shuffle(ids)
        joined = iter(ids)
        draws = dict.fromkeys(STATES, joined)
    for rep in range(n):
        for s in STATES:
            g = list(CANDIDATES)[next(draws[s])] if phase.startswith("pilot") or phase == "collect" else None
            probe = "Try_Scan" if (rep % 2 or phase == "matched") else "Try_Left"
            yield Job(s, probe, g, rep, phase)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("phase", choices=STAGE)
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--gate", type=Path)
    args = p.parse_args()
    if args.phase == "collect":
        gate = json.loads(args.gate.read_text()) if args.gate else {}
        if not (gate.get("trials") == 30 and gate.get("entry_pass") and gate.get("phase") == "pilot-5"
                and gate.get("hover_xyz_m") == list(HOVER) and gate.get("proposal") == FINAL_PROPOSAL.tolist()):
            p.error("Formal collection requires a passing 30-episode pilot summary")
    args.output.mkdir(parents=True, exist_ok=True)
    n = 10 if args.phase.startswith("pilot") else {"collect": 150, "closed": 15, "matched": 1}[args.phase]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, status in pool.imap_unordered(collect, ((j, args.output) for j in schedule_jobs(args.phase, n))):
            print(name, status, flush=True)
    if args.phase.startswith("pilot") or args.phase == "collect":
        rows = [json.loads(f.read_text()) for f in args.output.glob("*.json")]
        rows = [r for r in rows if "initial_state_id" in r]
        valid = sum(r["physics_validity"] for r in rows)
        pos = sum(r["task_success"] is True for r in rows)
        summary = dict(trials=len(rows), valid=valid, positives=pos, valid_rate=valid/len(rows),
                       positive_rate=pos/valid if valid else None,
                       invalid_reasons=dict(Counter(k for r in rows for k in r["invalid_reasons"])))
        summary["entry_pass"] = bool(len(rows) == count_rows(args.phase)
                                     and summary["valid_rate"] >= .80 and valid and .15 <= pos/valid <= .50)
        summary["hover_xyz_m"] = HOVER
        summary["proposal"] = (FINAL_PROPOSAL if args.phase in ("pilot-4", "pilot-5", "collect") else PROPOSAL).tolist()
        summary["phase"] = args.phase
        save_json(args.output / "summary.json", summary)
        print(summary, flush=True)


def count_rows(phase):
    return 450 if phase == "collect" else 30


if __name__ == "__main__":
    main()
