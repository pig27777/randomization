"""Compare two gathering meshes at the same observed normal pressure and travel."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from record_local_top import protocol
from replay_ridge_mesh import CASE, DT, load_pair
from replay_ridge_window import aperture_observation, backend

TARGET_N = .5
PRELOAD_LIMIT_S = 1.25
CLOSURE_S = .20
SPEED_M_S = .008


def run(robot, dt, preload_limit_s, closure_s):
    sim = robot.sim
    m, d = sim.m, sim.d
    m.opt.timestep = dt
    robot.control_period_steps = round(.010 / dt)
    feedforward = d.qfrc_bias[robot.dofs] - d.qfrc_gravcomp[robot.dofs]
    robot.qtarget = d.ctrl[robot.act] - feedforward / m.actuator_gainprm[robot.act, 0]
    robot.kin.qpos[:] = d.qpos
    robot.kin.qpos[robot.qadr] = robot.qtarget
    mujoco.mj_fwdPosition(m, robot.kin)
    target = robot.kin.site_xpos[robot.site].copy()
    gap = 2. * d.ctrl[robot.gripper]
    press = protocol.NormalPress(TARGET_N, protocol.tips(robot).mean(axis=0)[2])
    load_start = None
    consecutive = 0
    stride = round(.020 / dt)
    closure_steps = round(closure_s / dt)
    preload_steps = round(preload_limit_s / dt)
    preload, samples = [], []
    outcome = "preload timeout"
    for step in range(preload_steps + closure_steps + 1):
        feedback = backend.measure(robot)
        down = max(0., sum(force[2] for force in feedback["force_on_fingers_N"]))
        support = feedback["direct_support_N"]
        if feedback["penetration_mm"] > 1. or d.warning.number.any():
            outcome = "numerical gate"
            break
        if step % stride == 0:
            failure, health = backend.check_state(sim)
            if failure or health["payload_geometry"]["intersecting_triangles"]:
                outcome = "geometry gate"
                break
        qualified = .4 <= down <= .6 and support <= .05
        tip_z = protocol.tips(robot).mean(axis=0)[2]
        target[2] = press.move(target[2], tip_z, down, support, dt, approach=False)
        if load_start is None:
            if step % stride == 0:
                preload.append(dict(time_s=float(d.time), pressure_N=down, direct_support_N=support))
                if step % (stride * 25) == 0:
                    print(f"{m.nflexvert} nodes at {step * dt:.3f} s preload, {down:.6f} N", flush=True)
            consecutive = consecutive + 1 if qualified and .4 <= press.filtered_N <= .6 else 0
            if consecutive >= round(press.hold_s / dt):
                load_start = step
            elif step >= preload_steps:
                break
        if load_start is not None:
            elapsed = step - load_start
            if elapsed % stride == 0:
                samples.append(aperture_observation(robot, feedback))
            if elapsed == closure_steps:
                outcome = "complete"
                break
            if qualified and .4 <= press.filtered_N <= .6:
                gap -= SPEED_M_S * dt
        robot.command(target, gap)
        sim.step()
    return dict(outcome=outcome, preload_s=load_start * dt if load_start is not None else None,
                preload=preload, samples=samples)


def fit_common(runs):
    if any(run["outcome"] != "complete" for run in runs):
        return None
    traces = []
    for trial in runs:
        rows = trial["samples"]
        eligible = [.4 <= row["pressure_N"] <= .6 and row["direct_support_N"] <= .05 for row in rows]
        spans = np.flatnonzero(np.diff(np.r_[False, eligible, False])).reshape(-1, 2)
        span = max(spans, key=lambda pair: pair[1] - pair[0], default=None)
        if span is None or span[1] - span[0] < 3:
            return None
        data = rows[span[0]:span[1]]
        x = rows[0]["actual_gap_m"] - np.asarray([row["actual_gap_m"] for row in data])
        if (np.diff(x) <= 0.).any():
            return None
        traces.append((x, data))
    low, high = max(x[0] for x, _ in traces), min(x[-1] for x, _ in traces)
    if high <= low:
        return None
    positions = np.linspace(low, high, 5)
    fits = []
    for x, data in traces:
        values = np.interp(positions, x, [row["extra_length_m"] for row in data])
        reference = np.interp(positions, x, [row["reference_excess_m"] for row in data])
        pressure = np.interp(positions, x, [row["pressure_N"] for row in data])
        times = np.interp((low, high), x, [row["time_s"] for row in data])
        duration = float(times[1] - times[0])
        growth = float(values[-1] - values[0])
        reference_growth = float(reference[-1] - reference[0])
        fits.append(dict(duration_s=duration, growth_m=growth, pressure_N=float(pressure.mean()),
                         slope_m_per_m=float(np.polyfit(positions, values, 1)[0]),
                         eligible=duration >= .020 - 1e-9 and growth >= .00025,
                         reference_growth_m=reference_growth,
                         reference_slope_m_per_m=float(np.polyfit(positions, reference, 1)[0]),
                         reference_eligible=duration >= .020 - 1e-9 and reference_growth >= .00025))
    changes = {}
    for field in ("growth_m", "slope_m_per_m", "pressure_N"):
        if fits[0][field]:
            changes[field] = fits[1][field] / fits[0][field] - 1.
    reference_changes = {}
    for field in ("reference_growth_m", "reference_slope_m_per_m"):
        if fits[0][field]:
            reference_changes[field] = fits[1][field] / fits[0][field] - 1.
    passed = (len(changes) == 3 and all(fit["eligible"] for fit in fits)
              and all(abs(change) <= .10 for change in changes.values()))
    reference_passed = (len(reference_changes) == 2 and all(fit["reference_eligible"] for fit in fits)
                        and all(abs(change) <= .10 for change in reference_changes.values())
                        and abs(changes["pressure_N"]) <= .10)
    return dict(common_travel_interval_m=[float(low), float(high)], fits=fits, relative_change=changes, passed=passed,
                reference_relative_change=reference_changes, reference_passed=reference_passed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--step-us", type=float, choices=(12.5, 25.), default=DT * 1e6)
    parser.add_argument("--preload-limit-s", type=float, default=PRELOAD_LIMIT_S)
    parser.add_argument("--closure-s", type=float, default=CLOSURE_S)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < .020 for value in (args.preload_limit_s, args.closure_s)):
        parser.error("Preload and closure budgets must be finite and at least 20 ms")
    case = args.case.resolve()
    dt = args.step_us / 1e6
    coarse, refined, *_ = load_pair(case=case, timestep=dt)
    runs = [run(robot, dt, args.preload_limit_s, args.closure_s) for robot in (coarse, refined)]
    comparison = fit_common(runs)
    result = dict(dt_s=dt, mesh_vertices=[coarse.sim.m.nflexvert, refined.sim.m.nflexvert],
                  pressure_target_N=TARGET_N, preload_limit_s=args.preload_limit_s, closure_s=args.closure_s,
                  runs=runs, comparison=comparison)
    output = args.output or case / "local-mesh-window/matched-pressure.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print([trial["outcome"] for trial in runs], comparison, flush=True)


if __name__ == "__main__":
    main()
