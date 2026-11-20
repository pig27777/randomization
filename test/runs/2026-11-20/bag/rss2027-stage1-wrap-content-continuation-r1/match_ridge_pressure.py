"""Compare two gathering meshes at the same observed normal pressure and travel."""

import json

import mujoco
import numpy as np

from record_local_top import protocol
from replay_ridge_mesh import DT, OUTPUT, load_pair
from replay_ridge_window import aperture_observation, backend

TARGET_N = .5
PRELOAD_LIMIT_S = 1.25
CLOSURE_S = .20
SPEED_M_S = .008


def run(robot):
    sim = robot.sim
    m, d = sim.m, sim.d
    m.opt.timestep = DT
    robot.control_period_steps = round(.010 / DT)
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
    stride = round(.020 / DT)
    closure_steps = round(CLOSURE_S / DT)
    preload_steps = round(PRELOAD_LIMIT_S / DT)
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
        target[2] = press.move(target[2], tip_z, down, support, DT, approach=False)
        if load_start is None:
            if step % stride == 0:
                preload.append(dict(time_s=float(d.time), pressure_N=down, direct_support_N=support))
            consecutive = consecutive + 1 if qualified and .4 <= press.filtered_N <= .6 else 0
            if consecutive >= round(press.hold_s / DT):
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
                gap -= SPEED_M_S * DT
        robot.command(target, gap)
        sim.step()
    return dict(outcome=outcome, preload_s=load_start * DT if load_start is not None else None,
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
        pressure = np.interp(positions, x, [row["pressure_N"] for row in data])
        times = np.interp((low, high), x, [row["time_s"] for row in data])
        duration = float(times[1] - times[0])
        growth = float(values[-1] - values[0])
        fits.append(dict(duration_s=duration, growth_m=growth, pressure_N=float(pressure.mean()),
                         slope_m_per_m=float(np.polyfit(positions, values, 1)[0]),
                         eligible=duration >= .020 - 1e-9 and growth >= .00025))
    changes = {}
    for field in ("growth_m", "slope_m_per_m", "pressure_N"):
        if fits[0][field]:
            changes[field] = fits[1][field] / fits[0][field] - 1.
    passed = (len(changes) == 3 and all(fit["eligible"] for fit in fits)
              and all(abs(change) <= .10 for change in changes.values()))
    return dict(common_travel_interval_m=[float(low), float(high)], fits=fits, relative_change=changes, passed=passed)


def main():
    coarse, refined, *_ = load_pair()
    runs = [run(robot) for robot in (coarse, refined)]
    comparison = fit_common(runs)
    result = dict(dt_s=DT, mesh_vertices=[coarse.sim.m.nflexvert, refined.sim.m.nflexvert],
                  pressure_target_N=TARGET_N, runs=runs, comparison=comparison)
    with (OUTPUT / "matched-pressure.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print([trial["outcome"] for trial in runs], comparison, flush=True)


if __name__ == "__main__":
    main()
