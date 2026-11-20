"""Repeat a measured YAM gathering window at two native time steps."""

import json
from unittest.mock import patch

import mujoco
import numpy as np

from record_local_top import RUN, at_top_center, backend
from refine_parcel import physical_parcel, prepared_wrap
from measure_surface_excess import aperture_box, areas
from rss2027_yam_capture_audit import aperture_material_areas

CASE = RUN / "fold-aligned-60-side-energy/sample-058/ridge-height"


def assemble():
    trial = CASE / "recording/trial"
    result = json.loads((trial / "result.json").read_bytes())
    height = json.loads((CASE / "height.json").read_bytes())
    cfg = backend.Config(**result["config"])
    with at_top_center(height["target_center_xy_m"]), \
            patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    return robot


def load_window(robot, comparison):
    trial = CASE / "recording/trial"
    trace = json.loads((trial / "trace.json").read_bytes())
    fit = comparison["fold_window"]
    with np.load(trial / "trajectory.npz") as archive:
        time = archive["time"]
        selected = np.flatnonzero((time >= fit["t_start_s"]) & (time <= fit["t_last_s"]))
        if len(selected) != fit["samples"]:
            raise ValueError("The recorded pressure window does not match its geometry samples")
        indices = np.r_[selected[0] - 1, selected]
        states = archive["integration_state"][indices]
        spec = int(archive["integration_state_spec"])
    model, data = robot.sim.m, robot.sim.d
    commands = []
    for state in states:
        mujoco.mj_setState(model, data, state, spec)
        commands.append(data.ctrl.copy())
    mujoco.mj_setState(model, data, states[0], spec)
    backend.refresh(robot.sim)
    measured = backend.measure(robot)
    np.testing.assert_allclose(measured["force_on_fingers_N"], trace[indices[0]]["force_on_fingers_N"],
                               atol=1e-9, rtol=0.)
    return time[indices], np.asarray(commands), states[0], spec


def aperture_observation(robot, measurement):
    sim, gap = robot.sim, robot.gap()
    model, data = sim.m, sim.d
    pad = model.geom_bodyid[robot.pad_geom[0]]
    center, axes, half = aperture_box(data, data.geom_xpos[robot.pad_geom], pad, gap)
    faces = sim.parcel.envelope["faces"]
    triangles = ((data.flexvert_xpos - center) @ axes)[faces]
    spatial, projection = areas(triangles, half) if gap > 0. else (0., 0.)
    reference = aperture_material_areas(triangles, sim.parcel.envelope["reference"][faces], half)[1] if gap > 0. else 0.
    width = 2. * half[0]
    pressure = max(0., sum(vector[2] for vector in measurement["force_on_fingers_N"]))
    return dict(time_s=float(data.time), pressure_N=pressure, direct_support_N=measurement["direct_support_N"],
                actual_gap_m=gap, extra_length_m=(spatial - projection) / width,
                reference_length_m=reference / width, reference_excess_m=(reference - projection) / width,
                penetration_mm=measurement["penetration_mm"])


def replay(robot, dt, time, command, state, spec):
    sim = robot.sim
    model, data = sim.m, sim.d
    model.opt.timestep = dt
    mujoco.mj_setState(model, data, state, spec)
    backend.refresh(sim)
    sample_steps = np.rint((time - time[0]) / dt).astype(int)
    samples, forces = [], []
    outcome = "complete"
    for step in range(sample_steps[-1] + 1):
        if step in sample_steps:
            backend.refresh(sim)
        measurement = backend.measure(robot)
        pressure = max(0., sum(vector[2] for vector in measurement["force_on_fingers_N"]))
        direct = measurement["direct_support_N"]
        depth = measurement["penetration_mm"]
        if depth > 1. or data.warning.number.any():
            outcome = "numerical gate"
            break
        if step >= sample_steps[1]:
            forces.append((pressure, direct))
        if step in sample_steps:
            failure, health = backend.check_state(sim)
            if failure or health["payload_geometry"]["intersecting_triangles"]:
                outcome = "geometry gate"
                break
            samples.append(aperture_observation(robot, measurement))
        if step == sample_steps[-1]:
            break
        data.ctrl[:] = [np.interp(data.time, time, channel) for channel in command.T]
        sim.step()
    fit = None
    if outcome == "complete":
        x = samples[1]["actual_gap_m"] - np.array([row["actual_gap_m"] for row in samples[1:]])
        y = np.array([row["extra_length_m"] for row in samples[1:]])
        values = np.asarray(forces)
        growth = float(y[-1] - y[0])
        in_band = (values[:, 0] >= .4) & (values[:, 0] <= .6) & (values[:, 1] <= .05)
        sampled_band = all(.4 <= row["pressure_N"] <= .6 and row["direct_support_N"] <= .05 for row in samples[1:])
        monotonic = bool(np.all(np.diff(x) > 0.))
        fit = dict(displacement_m=float(x[-1]), growth_m=growth,
                   pressure_span_N=[float(values[:, 0].min()), float(values[:, 0].max())],
                   mean_pressure_N=float(np.mean([row["pressure_N"] for row in samples[1:]])),
                   pressure_band_all_steps=bool(in_band.all()),
                   slope_m_per_m=float(np.polyfit(x, y, 1)[0]) if monotonic else None,
                   eligible=bool(sampled_band and monotonic and growth >= .00025))
    return dict(dt_s=dt, outcome=outcome, fit=fit, samples=samples)


def main():
    original = json.loads((CASE / "physical-comparison.json").read_bytes())
    first = assemble()
    dt = float(first.sim.m.opt.timestep)
    arguments = load_window(first, original)
    runs = [replay(robot, step, *arguments) for robot, step in ((first, dt), (assemble(), dt / 2.))]
    slopes = [run["fit"]["slope_m_per_m"] if run["fit"] else None for run in runs]
    original_slope = original["fold_window"]["slope_m_per_m"]
    relatives = [value / original_slope - 1. if value is not None else None for value in slopes]
    paired = slopes[1] / slopes[0] - 1. if all(slopes) else None
    changes = {}
    if all(run["fit"] and run["fit"]["eligible"] for run in runs):
        for key in ("growth_m", "displacement_m"):
            changes[key] = runs[1]["fit"][key] / runs[0]["fit"][key] - 1.
    passed = (paired is not None and abs(paired) <= .10 and abs(relatives[0]) <= .10
              and bool(changes) and all(abs(change) <= .10 for change in changes.values()))
    result = dict(original_slope_m_per_m=original_slope, changes_from_original=relatives,
                  relative_half_step=paired, half_step_geometry_changes=changes, passed=passed, runs=runs)
    with (CASE / "local-step-comparison.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(passed, relatives, paired, flush=True)


if __name__ == "__main__":
    main()
