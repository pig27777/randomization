"""Record a physical parcel trial through the shared YAM protocol."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import time
from unittest.mock import patch

import mujoco
import numpy as np

import physical_parcel_yam
from yam_platform.protocol import run


def penetration_peak(robot, phase):
    sim = robot.sim
    model, data = sim.m, sim.d
    flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "bag")
    contact = min((c for c in data.contact if c.efc_address >= 0 and flex in c.flex), key=lambda c: c.dist)
    faces = sim.parcel.envelope["faces"]
    sides = []
    for geom, flex_id, element, vertex in zip(contact.geom, contact.flex, contact.elem, contact.vert, strict=True):
        if geom >= 0:
            sides.append(dict(geom=model.geom(geom).name, position_m=data.geom_xpos[geom].tolist(),
                              rotation=data.geom_xmat[geom].reshape(3, 3).tolist()))
        else:
            nodes = faces[element].tolist() if element >= 0 else [int(vertex)] if vertex >= 0 else []
            sides.append(dict(flex=int(flex_id), element=int(element), vertex_ids=nodes,
                              vertices_m=data.flexvert_xpos[nodes].tolist()))
    return dict(feedback_time_s=float(data.time), phase=phase, distance_m=float(contact.dist),
                pos_m=contact.pos.tolist(), normal=contact.frame[:3].tolist(), sides=sides)


def release_contacts(robot, *, include_all_fingers=False):
    model, data = robot.sim.m, robot.sim.d
    flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "bag")
    pad_bodies = model.geom_bodyid[robot.pad_geom]
    if include_all_fingers:
        pads = {int(geom): i for i, group in enumerate(robot.all_finger_sets) for geom in group}
    else:
        pads = {int(geom): i for i, body in enumerate(pad_bodies)
                for geom in np.flatnonzero(model.geom_bodyid == body)}
    left, right = data.geom_xpos[robot.pad_geom]
    axis = right - left
    axis /= max(np.linalg.norm(axis), 1e-12)
    local_force = np.zeros(6)
    records = []
    for index, contact in enumerate(data.contact):
        address = contact.efc_address
        if address < 0 or flex not in contact.flex:
            continue
        film_slot = list(contact.flex).index(flex)
        pad_slot = 1 - film_slot
        geom = int(contact.geom[pad_slot])
        pad = pads.get(geom)
        if pad is None:
            continue
        mujoco.mj_contactForce(model, data, index, local_force)
        normal = (2 * pad_slot - 1) * contact.frame[:3]
        row = dict(pad=pad, element=int(contact.elem[film_slot]), vertex=int(contact.vert[film_slot]),
                   inner=bool((2 * pad - 1) * (normal @ axis) > .7), normal_on_pad=normal.tolist(),
                   pos_m=contact.pos.tolist(), force_contact_N=local_force[:3].tolist(),
                   friction=contact.friction[:2].tolist(),
                   tangent_constraint_velocity_m_s=data.efc_vel[address + 1:address + 3].tolist()
                   if contact.dim == 3 else [])
        if include_all_fingers:
            body = int(model.geom_bodyid[geom])
            world = (2 * pad_slot - 1) * (contact.frame.reshape(3, 3).T @ local_force[:3])
            row.update(geom=model.geom(geom).name, body=model.body(body).name,
                       standard_pad=bool(body == pad_bodies[pad]), force_on_finger_N=world.tolist())
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("cpu", "warp"), default="cpu")
    parser.add_argument("--record-penetration-peaks", action="store_true")
    parser.add_argument("--record-capture-forces", action="store_true")
    args = parser.parse_args()
    if args.record_capture_forces and args.engine != "cpu":
        parser.error("--record-capture-forces requires the CPU contact buffer")
    backend = (physical_parcel_yam if args.engine == "cpu" else
               import_module("physical_parcel_yam_warp"))
    cfg = backend.Config(**json.loads(args.config.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recorder_source.py").write_bytes(Path(__file__).read_bytes())
    last = 0.0
    previous_phase = None
    started = time.perf_counter()
    device_samples = []
    penetration_events = []
    release_samples = []
    capture_samples = []

    def observe(robot, row):
        nonlocal last, previous_phase
        if cfg.pickup_release_seconds and row["phase"] in ("lift_hold", "open"):
            release_samples.append(dict(t_s=row["t"], phase=row["phase"], contacts=release_contacts(robot)))
        now = time.perf_counter()
        if now - last >= 5 or row["phase"] != previous_phase:
            if args.engine == "warp":
                device_samples.append(dict(t_s=row["t"], **robot.sim.telemetry()))
            temporary = args.output / "progress.tmp"
            progress = dict(row, wall_s=now-started)
            if device_samples:
                progress["gpu"] = device_samples[-1]
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(args.output / "progress.json")
            last = now
        if row["phase"] != previous_phase:
            print(json.dumps(dict(time_s=row["t"], phase=row["phase"], wall_s=now-started)), flush=True)
            previous_phase = row["phase"]

    def observe_peak(robot, phase):
        penetration_events.append(penetration_peak(robot, phase))

    run_args = dict(on_sample=observe,
                    on_penetration_peak=observe_peak if args.record_penetration_peaks else None)
    if args.record_capture_forces:
        native_scene = backend.scene
        stride = max(1, round(.002 / cfg.dt))
        capture_phases = {"close", "lift", "lift_hold"}
        active_phases = {"approach", *capture_phases}

        def capture_scene(config):
            robot = native_scene(config)
            data = robot.sim.d
            native_step = robot.sim.step

            def capture_step():
                t = float(data.time)
                if previous_phase in active_phases and round(t / cfg.dt) % stride == 0:
                    # Read the same native contact buffer as the shared controller.
                    feedback = backend.measure(robot)
                    capture_samples.append(dict(t_s=t, gap_m=robot.gap(),
                                                command_gap_m=2. * float(data.ctrl[robot.gripper]),
                                                contacts=release_contacts(robot, include_all_fingers=True), **feedback))
                native_step()

            robot.sim.step = capture_step
            return robot

        with patch.object(backend, "scene", capture_scene):
            result = run(backend, cfg, args.output / "trial", **run_args)
        ranges = [*result["phases"], dict(phase=result["phase"],
                  start_s=result["phases"][-1]["end_s"] if result["phases"] else 0., end_s=result["sim_time_s"])]
        selected = []
        for interval in ranges:
            if interval["phase"] in capture_phases:
                selected.extend(dict(phase=interval["phase"], **row) for row in capture_samples
                                if interval["start_s"] <= row["t_s"] < interval["end_s"])
        if selected:
            (args.output / "capture-contacts.json").write_text(json.dumps(selected, separators=(",", ":")) + "\n")
    else:
        result = run(backend, cfg, args.output / "trial", **run_args)
    if args.record_penetration_peaks:
        (args.output / "penetration-peaks.json").write_text(json.dumps(penetration_events, indent=2) + "\n")
    if release_samples:
        (args.output / "release-contacts.json").write_text(json.dumps(release_samples, indent=2) + "\n")
    if args.engine == "warp":
        result["gpu_samples"] = device_samples
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
