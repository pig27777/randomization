"""Record a physical parcel trial through the shared YAM protocol."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import time

import mujoco

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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("cpu", "warp"), default="cpu")
    parser.add_argument("--record-penetration-peaks", action="store_true")
    args = parser.parse_args()
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

    def observe(robot, row):
        nonlocal last, previous_phase
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

    result = run(backend, cfg, args.output / "trial", on_sample=observe,
                 on_penetration_peak=observe_peak if args.record_penetration_peaks else None)
    if args.record_penetration_peaks:
        (args.output / "penetration-peaks.json").write_text(json.dumps(penetration_events, indent=2) + "\n")
    if args.engine == "warp":
        result["gpu_samples"] = device_samples
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
