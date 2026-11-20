"""Hold the saved fine PE actuator inputs around the observed film self-contact."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from replay_refined_pe import RUN, configure, experiment

RECORDING = RUN / "full-action/recording"
CHECKPOINT_TIME = 8.960025
CHECKPOINT = RUN / "full-action/pre-failure-state.npz"


def extract_checkpoint():
    with np.load(RECORDING / "trial/trajectory.npz") as archive:
        times = archive["time"]
        indices = np.flatnonzero(np.isclose(times, CHECKPOINT_TIME, atol=1e-7, rtol=0.))
        if len(indices) != 1:
            raise ValueError("Expected exactly one fine PE checkpoint at the recorded time")
        index = int(indices[0])
        with CHECKPOINT.open("xb") as stream:
            np.savez_compressed(stream, index=index, time_s=times[index], qpos=archive["qpos"][index],
                                state=archive["integration_state"][index], spec=archive["integration_state_spec"])


def held_scene():
    trial = RECORDING / "trial"
    result = json.loads((trial / "result.json").read_bytes())
    trace = json.loads((trial / "trace.json").read_bytes())
    adapter = experiment.backend
    replay = experiment.replay
    cfg = adapter.Config(**result["config"])
    with patch.object(adapter, "parcel_world", partial(replay.world_with_movable_support, lower_floor=True)), \
            patch.object(experiment.physical_parcel, "wrap_generic", experiment.prepared_wrap(cfg.geometry)):
        robot = adapter.scene(cfg)
    with np.load(CHECKPOINT) as archive:
        index, time = int(archive["index"]), float(archive["time_s"])
        state, spec = archive["state"], int(archive["spec"])
        qpos = archive["qpos"]
    sim = robot.sim
    sample = trace[index]
    if sample["t"] != time or sample["phase"] != "lift_hold":
        raise ValueError("The stored integration state and suspension sample do not agree")
    mujoco.mj_setState(sim.m, sim.d, state, spec)
    np.testing.assert_array_equal(sim.d.qpos, qpos)
    adapter.refresh(sim)
    error, health = adapter.check_state(sim)
    force = adapter.measure(robot)
    geometry = health["payload_geometry"]
    if (error or geometry["outside_corners"] or geometry["intersecting_triangles"]
            or force["penetration_mm"] > 1. or force["support_N"] != 0.):
        raise ValueError(f"The designated checkpoint does not satisfy the native preload conditions: {error}")
    np.testing.assert_allclose(force["force_on_fingers_N"], sample["force_on_fingers_N"], atol=1e-8, rtol=0.)
    mujoco.mj_setState(sim.m, sim.d, state, spec)
    return robot, index, hashlib.sha256(state.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--shorten-dominant-contact", action="store_true")
    args = parser.parse_args()
    if args.extract:
        extract_checkpoint()
        return
    if args.output is None:
        parser.error("provide an output directory")
    args.output.mkdir(exist_ok=False)
    configure()
    replay = experiment.replay
    with np.load(CHECKPOINT) as archive:
        start = float(archive["time_s"])
    load_start = json.loads((RECORDING / "load-check.json").read_bytes())["load_start_s"]
    initial_load = replay.RATE * (start - load_start)
    for tag, dt in (("25", .000025), ("12p5", .0000125)):
        changed = None

        def probe_scene():
            nonlocal changed
            robot, frame, digest = held_scene()
            if args.shorten_dominant_contact:
                model = robot.sim.m
                name = "left_pad" if tag == "25" else "bag self-contact"
                response = model.geom_solref[model.geom("left_pad").id] if tag == "25" else model.flex_solref[0]
                np.testing.assert_allclose(response, [.001, 1.], atol=1e-12, rtol=0.)
                response[0] = .0005
                changed = dict(contact=name, original_time_s=.001, replay_time_s=.0005)
            return robot, frame, digest

        with patch.object(replay, "held_scene", probe_scene):
            result, rows = replay.run(dt, .09, "vertical", initial_load_N=initial_load)
        result["initial_load_N"] = initial_load
        if changed is not None:
            result["changed_contact_response"] = changed
        filename = args.output / f"dt-{tag}us"
        filename.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(filename.with_suffix(".npz"), columns=np.asarray(replay.FIELDS), rows=rows)
        print(tag, result["outcome"], result["peak_penetration_mm"], flush=True)


if __name__ == "__main__":
    main()
