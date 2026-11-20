"""Compare tangential loading after the refined PE bag established its own grasp."""

import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from replay_refined_pe import RUN, experiment

SLIDE = RUN.with_name("rss2027-stage1-pe-held-load-step-r1")
sys.path.insert(0, str(SLIDE))
from measure_aperture_flow import record_flow, summarize  # noqa: E402
from measure_tangent_flow import estimate_beta  # noqa: E402

RECORDING = RUN / "full-action/recording"
replay = experiment.replay
backend = experiment.backend


def held_scene():
    event = json.loads((RECORDING / "withdrawal.json").read_bytes())
    trial = RECORDING / "trial"
    cfg = backend.Config(**json.loads((trial / "result.json").read_bytes())["config"])
    trace = json.loads((trial / "trace.json").read_bytes())
    with (RECORDING / "load-curve.csv").open(newline="") as file:
        preload = [row for row in csv.DictReader(file) if float(row["t_s"]) <= event["time_s"]]
    if (not preload or float(preload[-1]["t_s"]) - float(preload[0]["t_s"]) < .20 - 1e-8
            or not all(row["bilateral_held"] == "True" and float(row["support_N"]) == 0. for row in preload)):
        raise ValueError("The recorded preloading interval lacks 0.20 s of bilateral support-free holding")
    with patch.object(backend, "parcel_world", partial(replay.world_with_movable_support, lower_floor=True)), \
            patch.object(experiment.physical_parcel, "wrap_generic", experiment.prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    sim = robot.sim
    with np.load(trial / "trajectory.npz") as archive:
        index, = np.flatnonzero(np.isclose(archive["time"], event["time_s"], atol=1e-12, rtol=0.))
        state, spec = archive["integration_state"][index], int(archive["integration_state_spec"])
        mujoco.mj_setState(sim.m, sim.d, state, spec)
        np.testing.assert_array_equal(sim.d.qpos, archive["qpos"][index])
    sample = trace[index]
    if (sample["t"] != event["time_s"] or sample["phase"] != "lift_hold" or event["support_N"] != 0.
            or max(frame["penetration_mm"] for frame in trace[:index + 1]) > 1.):
        raise ValueError("Archived state is not the sampled suspended prefix")
    backend.refresh(sim)
    failure, health = backend.check_state(sim)
    measured = backend.measure(robot)
    inner, _ = replay.inner_loads(replay.release_contacts(robot))
    if (failure or any(health["payload_geometry"][key] for key in ("outside_corners", "intersecting_triangles"))
            or measured["penetration_mm"] > 1. or measured["support_N"] != 0. or min(inner) <= .001):
        raise ValueError(f"Restored fine preload has no qualified bilateral hold: {health}, {inner}")
    np.testing.assert_allclose(measured["force_on_fingers_N"], sample["force_on_fingers_N"], atol=1e-8, rtol=0.)
    mujoco.mj_setState(sim.m, sim.d, state, spec)
    supports = [replay.support_pose(sim, name) for name in (replay.SUPPORT, replay.FLOOR)]
    np.testing.assert_array_equal(supports, event["before_m"])
    np.testing.assert_array_equal(sim.d.xfrc_applied, 0.)
    for support, position in zip(supports, event["after_m"], strict=True):
        support[:] = position
    return robot, int(index), hashlib.sha256(state.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    reference = json.loads((SLIDE / "tangent-prefix/dt-25us/summary.json").read_bytes())
    original = reference["native_load"]
    direction = np.asarray(original["force_direction"])
    dt = original["dt_s"]
    with patch.object(replay, "held_scene", held_scene):
        native, rows, transitions = record_flow(dt, .02 if args.smoke else 1.29, "jaw-tangent", force_vector=direction)
    if original["rate_N_s"] != native["rate_N_s"]:
        raise ValueError("The reference and refined runs use different load rates")
    flow = summarize(native, rows, transitions, force_ratio=reference["material_flow"]["candidate_force_ratio"])
    candidate = estimate_beta(native, flow)
    peaks = json.loads((RECORDING / "penetration-peaks.json").read_bytes())
    prehistory_peak = max(-1000. * row["distance_m"] for row in peaks
                          if row["feedback_time_s"] <= native["source_time_s"])
    initial = {key: [original["first"][key], native["first"][key]]
               for key in ("actual_gap_m", "left_inner_N", "right_inner_N", "tension_N")}
    active = rows[:, 1] > 0.
    axial = float(np.max(np.abs(rows[active, 12] / rows[active, 1])))
    record = dict(native_load=native, beta_candidate=candidate, material_flow=flow,
                  maximum_closing_axis_fraction=axial, coarse_and_fine_initial=initial,
                  prehistory_peak_penetration_mm=prehistory_peak, prehistory_passed=prehistory_peak <= 1.)
    (args.output / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "native-load.npz", columns=np.asarray(replay.FIELDS), rows=rows)
    print(native["outcome"], native["peak_penetration_mm"], native["longest_bilateral_slide_s"], candidate, flush=True)


if __name__ == "__main__":
    main()
