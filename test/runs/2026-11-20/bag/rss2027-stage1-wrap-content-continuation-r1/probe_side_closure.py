"""Screen closed-finger contact geometry on the three settled 90% parcels."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from select_highfill_side import RUN
from select_side_flap import (backend, box, finger_frames, physical_parcel, prepared_wrap,
                              shifted_target, supplementary_finger_frames, swept_fingers)
from rss2027_yam_capture_audit import frame_contacts


def probe(source):
    folder = RUN / f"sample-{source['sample']:03}"
    result = json.loads((folder / "settle/summary.json").read_bytes())
    selection = source["selected"]
    cfg = backend.Config(**(result["config"] | dict(grasp_face="side", press_target_N=None,
                                                   side_start_distance=.08, side_height_offset=.014,
                                                   yaw=selection["yaw_deg"], tilt=selection["tilt_deg"])))
    shift, advance = np.asarray(selection["shift_m"]), selection["close_advance_m"]
    with shifted_target(shift), patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    model, data = robot.sim.m, robot.sim.d
    npar = robot.qadr[0]
    with np.load(folder / "settle/trial/trajectory.npz") as trajectory:
        final = trajectory["final_qpos"]
        data.qpos[:npar] = final[:npar]
        data.qpos[robot.fingers] = final[robot.fingers]
    mujoco.mj_fwdPosition(model, data)
    opening = robot.gap()
    pads = finger_frames(model, data) + list(supplementary_finger_frames(robot))
    obstacles = [box(model, data, name) for name in ("payload_geom_0", "table")]
    blockers = swept_fingers(pads, np.zeros(3), (advance + .002) * robot.approach, data.flexvert_xpos,
                             model.flex_elem.reshape(-1, 3), obstacles)
    solve = robot.solve(robot.initial_target + advance * robot.approach)
    if solve["position_mm"] > 1. or solve["rotation_deg"] > 1.:
        return dict(sample=source["sample"], selection=selection, actual_opening_m=opening,
                    open_blockers=blockers, pose=solve, rows=[])
    data.qpos[robot.qadr] = robot.qtarget
    obstacles = {model.geom(name).id for name in ("payload_geom_0", "table", "floor")
                 if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0}
    finger_geoms = set().union(*robot.all_finger_sets)
    rows = []
    for gap in np.r_[opening, np.arange(0., opening, .0005)[::-1]]:
        data.qpos[robot.fingers] = gap / 2.
        mujoco.mj_fwdPosition(model, data)
        axis = data.geom_xpos[robot.pad_geom[1]] - data.geom_xpos[robot.pad_geom[0]]
        axis /= np.linalg.norm(axis)
        contacts = frame_contacts(model, data, robot.all_finger_sets, axis)
        if not contacts:
            continue
        rigid = [model.geom(geom).name for row in data.contact if row.dist < 0.
                 for geom in row.geom if geom in obstacles and finger_geoms.intersection(row.geom)]
        best = [max((point["closing_alignment"] for point in contacts
                     if point["pad"] == pad and model.geom(point["geom"]).id in robot.pad_sets[pad]), default=None)
                for pad in (0, 1)]
        depth = -min(point["distance_m"] for point in contacts) * 1000.
        rows.append(dict(gap_m=float(gap), strongest_standard_alignments=best, peak_penetration_mm=depth,
                         rigid_contacts=sorted(set(rigid))))
    qualified = [row for row in rows if not blockers and not row["rigid_contacts"] and row["peak_penetration_mm"] <= 1.
                 and all(value is not None and value > .70 for value in row["strongest_standard_alignments"])]
    return dict(sample=source["sample"], selection=selection, actual_opening_m=opening, open_blockers=blockers, pose=solve,
                first_bilateral=next(iter(qualified), None), rows=rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=RUN / "side-open-screen.json")
    parser.add_argument("--output", type=Path, default=RUN / "side-closure-screen.json")
    args = parser.parse_args()
    source = json.loads(args.source.read_bytes())
    report = [probe(row) for row in source["samples"] if row["selected"] is not None]
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print([(row["sample"], row.get("first_bilateral")) for row in report], flush=True)


if __name__ == "__main__":
    main()
