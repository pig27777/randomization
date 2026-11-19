"""Estimate film/pad tangential motion from saved material nodes and native contacts."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from rss2027_micropatch import sustained_starts


def interpolate_velocity(contact_pos, points, velocity, triangle):
    xyz = points[triangle]
    basis = (xyz[1:] - xyz[0]).T
    uv = np.linalg.lstsq(basis, contact_pos - xyz[0], rcond=None)[0]
    weights = np.maximum(0., [1. - uv.sum(), *uv])
    return (weights / weights.sum()) @ velocity[triangle]


def tangential_speeds(model, data, groups, velocity, pad_velocity, faces):
    across = [[], []]
    for contact in data.contact:
        if contact.dist > 0. or 0 not in contact.flex:
            continue
        film_slot = list(contact.flex).index(0)
        pad_slot = 1 - film_slot
        geom = contact.geom[pad_slot]
        for pad, group in enumerate(groups):
            normal = contact.frame[:3]
            if geom not in group or (2 * pad - 1) * (2 * pad_slot - 1) * normal[0] <= .7:
                continue
            vertex = contact.vert[film_slot]
            element = contact.elem[film_slot]
            if vertex >= 0:
                film_velocity = velocity[vertex]
            elif element >= 0:
                film_velocity = interpolate_velocity(contact.pos, data.flexvert_xpos, velocity, faces[element])
            else:
                continue
            relative = pad_velocity[pad] - film_velocity
            across[pad].append(float(relative[2] - (relative @ normal) * normal[2]))
    return across


def audit(case, output):
    summary = json.loads((case / "summary.json").read_text())
    with (case / "curve.csv").open() as stream:
        curve = list(csv.DictReader(stream))
    with np.load(case / "coupon_nodes.npz") as archive:
        ref, faces, positions = (archive[name] for name in ("reference_m", "faces", "current_m"))
    if len(curve) != len(positions):
        raise ValueError("coupon frames do not match the force trace")
    frames = [i for i, row in enumerate(curve) if row["phase"] == "pull"]
    if summary["capture_t_s"] is None:
        raise ValueError("coupon never reached the bilateral load criterion")
    if len(frames) < 3:
        raise ValueError("not enough pull samples to reconstruct velocity")
    model = mujoco.MjModel.from_binary_path(str(case / "model.mjb"))
    data = mujoco.MjData(model)
    bodies = [model.body(name).id for name in ("left", "right")]
    origins = model.body_pos[bodies]
    pad_joints = model.body_jntadr[bodies, None] + np.arange(3)
    pad_qpos = model.jnt_qposadr[pad_joints]
    vertex_joints = model.body_jntadr[model.flex_vertbodyid, None] + np.arange(3)
    vertex_qpos = model.jnt_qposadr[vertex_joints]
    groups = [set(np.flatnonzero(model.geom_bodyid == body)) for body in bodies]
    records = [{key: float(value) for key, value in curve[i].items() if key != "phase"} for i in frames]
    time_s = np.array([row["t_s"] for row in records])
    half_gap = abs(origins[0, 0]) - np.array([row["gather_per_pad_m"] for row in records])
    pad_z = np.array([row["pad_z_m"] for row in records])
    tips = np.stack([np.column_stack((sign * half_gap, np.zeros(len(frames)), pad_z))
                     for sign in (-1, 1)], axis=1)
    first = np.searchsorted(time_s, time_s - .002, side="right") - 1
    last = np.searchsorted(time_s, time_s + .002)
    first = first.clip(min=0)
    last = last.clip(max=len(frames) - 1)
    rows = []
    for index, record in enumerate(records):
        t = time_s[index]
        if t < summary["capture_t_s"] or (summary["release_t_s"] is not None and t >= summary["release_t_s"]):
            continue
        a, b = first[index], last[index]
        interval = time_s[b] - time_s[a]
        film_speed = (positions[frames[b]] - positions[frames[a]]) / interval
        pad_speed = (tips[b] - tips[a]) / interval
        data.qpos[vertex_qpos] = positions[frames[index]] - ref
        data.qpos[pad_qpos] = tips[index] - origins
        mujoco.mj_forward(model, data)
        speeds = tangential_speeds(model, data, groups, film_speed, pad_speed, faces)
        median = [float(np.median(side)) if side else None for side in speeds]
        loaded = min(record["left_inner_N"], record["right_inner_N"], record["signed_pull_N"]) > .001
        threshold = .10 * float(pad_speed[:, 2].mean())
        sliding = loaded and all(value is not None and value > threshold for value in median)
        rows.append(dict(t_s=t, pull_N=record["signed_pull_N"],
                         left_tangent_speed_m_s=median[0], right_tangent_speed_m_s=median[1],
                         left_candidates=len(speeds[0]), right_candidates=len(speeds[1]),
                         loaded=loaded, sliding=sliding,
                         aperture_material_length_m=record["aperture_material_length_m"]))
    if not rows:
        raise ValueError("no samples within the capture-to-release interval")
    sustained = sustained_starts(np.array([row["t_s"] for row in rows]), [row["sliding"] for row in rows])
    result = dict(case=str(case), samples=len(rows), first_sustained_sliding=rows[sustained[0]] if sustained else None,
                  geometric_slip_candidate_N=summary["slip_candidate_force_N"], pull_start_s=float(time_s[0]))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "contact-sliding.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.case, args.output)


if __name__ == "__main__":
    main()
