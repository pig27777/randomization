"""Compare load ramps from one recorded, already suspended whole-bag state."""

import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
SOURCE = RUN.with_name("rss2027-stage1-kraft-load-ramp-clear-r1") / "recording"
CHECKPOINT = RUN / "held-state.npz"
sys.path[:0] = [str(RUN.parents[3] / "bag_model"), str(RUN.with_name("rss2027-stage1-bubble-free-release-r1"))]
import physical_parcel_yam as backend  # noqa: E402
from record_physical_parcel import penetration_peak, release_contacts  # noqa: E402
from record_support_withdrawal import FLOOR, SUPPORT, support_pose, world_with_movable_support  # noqa: E402
from rss2027_release_traction import inner_loads  # noqa: E402

FIELDS = ("time_s", "added_N", "actual_gap_m", "tension_N", "support_N", "left_inner_N", "right_inner_N",
          "left_slide_N", "right_slide_N", "both_sliding", "fingertip_contact_N", "penetration_mm",
          "added_along_closing_axis_N")
RATE = 5.


def extract_state():
    event = json.loads((SOURCE / "withdrawal.json").read_bytes())
    with np.load(SOURCE / "trial/trajectory.npz") as archive:
        indices = np.flatnonzero(np.isclose(archive["time"], event["time_s"], atol=1e-12, rtol=0.))
        if len(indices) != 1:
            raise ValueError("Support withdrawal must identify one recorded state")
        index = int(indices[0])
        with CHECKPOINT.open("xb") as stream:
            np.savez_compressed(stream, index=index, time_s=archive["time"][index], qpos=archive["qpos"][index],
                                state=archive["integration_state"][index], spec=archive["integration_state_spec"])


def held_scene():
    event = json.loads((SOURCE / "withdrawal.json").read_bytes())
    original = json.loads((SOURCE / "trial/result.json").read_bytes())
    if original["warnings"] or original["peaks"]["penetration_mm"] > 1. or event["support_N"] != 0.:
        raise ValueError("Source is not a numerically qualified, unloaded suspension")
    cfg = backend.Config(**original["config"])
    with patch.object(backend, "parcel_world", partial(world_with_movable_support, lower_floor=True)):
        robot = backend.scene(cfg)
    sim = robot.sim
    with np.load(CHECKPOINT) as archive:
        index, state = int(archive["index"]), archive["state"]
        mujoco.mj_setState(sim.m, sim.d, state, int(archive["spec"]))
        np.testing.assert_array_equal(sim.d.qpos, archive["qpos"])
        np.testing.assert_allclose(sim.d.time, [float(archive["time_s"]), event["time_s"]], atol=1e-12, rtol=0.)
    poses = [support_pose(sim, name) for name in (SUPPORT, FLOOR)]
    np.testing.assert_allclose(poses, event["before_m"], atol=1e-12, rtol=0.)
    np.testing.assert_allclose(sim.d.xfrc_applied, 0., atol=0., rtol=0.)
    for pose, target in zip(poses, event["after_m"], strict=True):
        pose[:] = target
    return robot, index, hashlib.sha256(state.tobytes()).hexdigest()


def jaw_axis(robot):
    left, right = robot.sim.d.geom_xpos[robot.pad_geom]
    inward = right - left
    return inward / np.linalg.norm(inward), (left + right) / 2.


def run(dt, limit, load_mode):
    robot, index, fingerprint = held_scene()
    sim = robot.sim
    sim.m.opt.timestep = dt
    control = sim.d.ctrl.copy()
    payload = int(sim.payload_bodies[0])
    start = float(sim.d.time)
    mujoco.mj_fwdPosition(sim.m, sim.d)
    closing, pad_center = jaw_axis(robot)
    down = np.array([0., 0., -1.])
    if load_mode == "jaw-tangent":
        outward = sim.com() - pad_center
        projected = outward - (outward @ closing) * closing
        if np.linalg.norm(projected) < 1e-6:
            raise ValueError("The held parcel does not define a tangential pull direction")
        direction = projected / np.linalg.norm(projected)
        np.testing.assert_allclose(direction @ closing, 0., atol=1e-12, rtol=0.)
    else:
        direction = down
    peak = loss_at = None
    rows = []
    outcome = "no_loss_before_limit"
    wall_start = time.perf_counter()
    for i in range(round(limit / dt)):
        elapsed = float(sim.d.time) - start
        load = RATE * elapsed
        gap = robot.gap()
        sim.d.xfrc_applied[payload, :3] = load * direction
        sim.step()
        measured = backend.measure(robot)
        support = measured["support_N"]
        grip = max(measured["contact_magnitude_each_N"])
        depth = measured["penetration_mm"]
        force = np.sum(measured["force_on_fingers_N"], axis=0)
        tension = float(force @ direction)
        normal, sliding = inner_loads(release_contacts(robot))
        bilateral = min(normal) > .001 and support <= .05
        sliding_now = i > 0 and bilateral and tension > .001 and (sliding >= .5 * normal).all()
        axial_load = float(load * (direction @ jaw_axis(robot)[0]))
        rows.append((elapsed, load, gap, tension, support, *normal, *sliding, sliding_now, grip, depth, axial_load))
        if i == 0 and (not bilateral or force @ down <= .001):
            raise ValueError("Restored state is missing bilateral suspension")
        if depth > 1. or sim.d.warning.number.any():
            if depth > 1.:
                peak = penetration_peak(robot, "held_load")
                peak["feedback_time_s"] = start + elapsed
            outcome = "native_penetration_or_warning"
            break
        if i % round(.002 / dt) == 0:
            failure, _health = backend.check_state(sim)
            if failure:
                outcome = failure
                break
        loss_at = (i if loss_at is None else loss_at) if grip <= .05 else None
        if loss_at is not None and (i - loss_at) * dt >= .05:
            outcome = "detached"
            break
    np.testing.assert_array_equal(sim.d.ctrl, control)
    backend.refresh(sim)
    failure, health = backend.check_state(sim)
    if failure:
        outcome = failure
    rows = np.asarray(rows)
    bounds = np.flatnonzero(np.diff(np.r_[False, rows[:, 9].astype(bool), False])).reshape(-1, 2)
    windows = [(int(a), float(rows[b - 1, 0] - rows[a, 0])) for a, b in bounds]
    sustained = next((a for a, duration in windows if duration >= .010 - 1e-9), None)

    def sample(i):
        return dict(zip(FIELDS, rows[i].tolist(), strict=True))

    loss = loss_at if outcome == "detached" and np.max(rows[:loss_at + 1, 4]) <= .05 else None
    result = dict(dt_s=dt, rate_N_s=RATE, source_frame=index, source_time_s=start, source_state_sha256=fingerprint,
                  force_direction=direction.tolist(), initial_closing_axis=closing.tolist(), load_mode=load_mode,
                  actuator_input=control.tolist(), physics_steps=len(rows), outcome=outcome,
                  wall_s=time.perf_counter() - wall_start, first=sample(0), end_health=health,
                  peak_penetration_mm=float(rows[:, 11].max()), deepest_invalid_contact=peak,
                  supported_force_peak_N=float(rows[:, 4].max()), loss=sample(loss) if loss is not None else None,
                  longest_bilateral_slide_s=max((duration for _a, duration in windows), default=0.),
                  sliding_onset=sample(sustained) if sustained is not None else None)
    return result, rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--direction", choices=("vertical", "jaw-tangent"), default="vertical")
    parser.add_argument("--step-us", type=float, nargs="+", choices=(12.5, 25, 50, 100), default=(100, 50))
    args = parser.parse_args()
    if args.extract:
        extract_state()
        return
    if args.output is None:
        parser.error("specify an output directory")
    args.output.mkdir(exist_ok=False)
    for step_us in args.step_us:
        dt = step_us / 1e6
        result, rows = run(dt, .02 if args.smoke else 2.4, args.direction)
        stem = args.output / ("dt-" + f"{step_us:g}".replace(".", "p") + "us")
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(stem.with_suffix(".npz"), columns=np.asarray(FIELDS), rows=rows)
        with stem.with_suffix(".csv").open("x", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(FIELDS)
            writer.writerows(rows[::round(.002 / dt)])
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
