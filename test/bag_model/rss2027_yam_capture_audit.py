"""Align saved YAM forces with film positions and reconstructed contact geometry."""

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bag_sim.wrap import triangle_box_intersections  # noqa: E402


def frame_contacts(model, data, groups, closing_axis):
    records = []
    for contact in data.contact:
        if contact.dist > 0 or 0 not in contact.flex:
            continue
        film_slot = list(contact.flex).index(0)
        pad_slot = 1 - film_slot
        geom = contact.geom[pad_slot]
        for pad, group in enumerate(groups):
            if geom not in group:
                continue
            normal = (2 * pad_slot - 1) * contact.frame[:3]
            alignment = float((2 * pad - 1) * normal @ closing_axis)
            records.append(dict(pad=pad, geom=model.geom(geom).name, inner=alignment > .7,
                                closing_alignment=alignment, position_m=contact.pos.tolist(),
                                distance_m=float(contact.dist), element=int(contact.elem[film_slot]),
                                vertex=int(contact.vert[film_slot])))
    return records


def clip_plane(polygon, axis, sign, limit):
    clipped = []
    previous = polygon[-1]
    previous_distance = limit - sign * previous[axis]
    for vertex in polygon:
        distance = limit - sign * vertex[axis]
        if (distance >= 0.) != (previous_distance >= 0.):
            fraction = previous_distance / (previous_distance - distance)
            clipped.append(previous + fraction * (vertex - previous))
        if distance >= 0.:
            clipped.append(vertex)
        previous, previous_distance = vertex, distance
    return np.asarray(clipped)


def aperture_material_areas(current, reference, half_extents):
    """Area within the gap and corresponding area in material coordinates."""
    areas = np.zeros(2)
    for triangle, rest in zip(current, reference, strict=True):
        polygon = np.column_stack((triangle, rest))
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
            polygon = clip_plane(polygon, axis, sign, half_extents[axis])
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            fan = (polygon[1:] - polygon[0]).reshape(-1, 2, 3)
            areas += .5 * np.linalg.norm(np.cross(fan[:-1], fan[1:]), axis=2).sum(axis=0)
    return areas


def audit(trial, output):
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    trace = json.loads((trial / "trace.json").read_text())
    result = json.loads((trial / "result.json").read_text())
    with np.load(trial / "trajectory.npz") as state:
        positions = state["qpos"]
        if not np.array_equal(state["time"], [row["t"] for row in trace]):
            raise ValueError("trajectory and force trace time samples do not match")
    if model.nflex != 1 or model.flex_dim[0] != 2:
        raise ValueError("capture audit expects one triangular bag surface")
    # qpos0 defines the material coordinates before an archived wrap is applied.
    mujoco.mj_forward(model, data)
    reference = data.flexvert_xpos.copy()
    pads = [model.geom(f"{side}_pad").id for side in ("left", "right")]
    groups = [set(np.flatnonzero(model.geom_bodyid == model.geom_bodyid[geom])) for geom in pads]
    faces = model.flex_elem.reshape(-1, 3)
    rows, locator = [], []
    for qpos, sample in zip(positions, trace, strict=True):
        if sample["phase"] not in ("close", "lift", "lift_hold"):
            continue
        data.qpos[:] = qpos
        # No sampled velocities are present. Only read geometry from this replay.
        mujoco.mj_forward(model, data)
        centers = data.geom_xpos[pads]
        axis = centers[1] - centers[0]
        axis /= np.linalg.norm(axis)
        contacts = frame_contacts(model, data, groups, axis)
        by_pad = [sum(c["pad"] == pad and c["inner"] for c in contacts) for pad in (0, 1)]
        rotation = data.geom_xmat[pads[0]].reshape(3, 3)
        # 20 mm wide, 40 mm high pads. The rail gap is between the inner faces.
        # The rectangular volume includes the rounded corners as an upper bound.
        opening = sample["gap"]
        half_extents = np.array([.010, opening / 2, .020])
        center = centers.mean(axis=0)
        intersecting = np.flatnonzero(triangle_box_intersections(
            data.flexvert_xpos, faces, half_extents, center, rotation
        )) if opening > 0 else np.empty(0, dtype=int)
        aperture_faces = faces[intersecting]
        local = (data.flexvert_xpos[aperture_faces] - center) @ rotation
        current_area, reference_area = aperture_material_areas(local, reference[aperture_faces], half_extents)
        rows.append(dict(t_s=sample["t"], phase=sample["phase"], actual_gap_m=opening,
                         commanded_gap_m=sample["command_gap"], left_inner_candidates=by_pad[0],
                         right_inner_candidates=by_pad[1], aperture_faces=len(intersecting),
                         aperture_area_m2=current_area, aperture_reference_area_m2=reference_area,
                         aperture_material_length_m=reference_area / (2 * half_extents[0]),
                         left_grip_N=sample["grip_each_N"][0], right_grip_N=sample["grip_each_N"][1],
                         tension_N=-sum(f[2] for f in sample["force_on_fingers_N"]),
                         support_N=sample["support_N"], com_z_m=sample["com"][2], pad_height_m=center[2]))
        locator.append(dict(t_s=sample["t"], aperture_faces=intersecting.tolist(), contacts=contacts))
    if not rows:
        raise ValueError("trajectory has no closure or lift samples")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "capture-audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "capture-locations.json").write_text(json.dumps(locator, indent=2) + "\n")
    lift = [row for row in rows if row["phase"] == "lift"]
    summary = dict(trial=str(trial), outcome=result["outcome"], sample_count=len(rows),
                   bilateral_inner_candidates=sum(row["left_inner_candidates"] > 0
                                                  and row["right_inner_candidates"] > 0 for row in rows),
                   peak_aperture_material_length_m=max(row["aperture_material_length_m"] for row in rows),
                   lift_support_min_N=min((row["support_N"] for row in lift), default=None),
                   lift_tension_peak_N=max((row["tension_N"] for row in lift), default=None),
                   lift_com_max_m=max((row["com_z_m"] for row in lift), default=None),
                   initial_com_z_m=result["initial_com"][2],
                   last_close=next((row for row in reversed(rows) if row["phase"] == "close"), None),
                   first_lift=lift[0] if lift else None)
    (output / "capture-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.trial, args.output)


if __name__ == "__main__":
    main()
