"""Measure the reference film crossing a YAM aperture between recorded poses."""

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, "test/bag_model")
from rss2027_yam_capture_audit import aperture_box, aperture_material_areas, clip_plane  # noqa: E402


def shared_material_area(local_a, local_b, reference, bounds_a, bounds_b):
    limits = np.r_[bounds_a, bounds_b]
    if (limits <= 0).any():
        return 0.
    area = 0.
    for initial, final, rest in zip(local_a, local_b, reference, strict=True):
        polygon = np.column_stack((initial, final, rest))
        for axis, limit in enumerate(limits):
            for sign in (-1, 1):
                polygon = clip_plane(polygon, axis, sign, limit)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            fan = polygon[1:, 6:] - polygon[0, 6:]
            area += .5 * np.linalg.norm(np.cross(fan[:-1], fan[1:]), axis=1).sum()
    return float(area)


def material_exchange(before, after, reference):
    a, bounds_a, total_a, row_a = before
    b, bounds_b, total_b, row_b = after
    shared = shared_material_area(a, b, reference, bounds_a, bounds_b)
    assert -1e-12 <= shared <= min(total_a, total_b) + 1e-12
    width = 2. * bounds_a[0]
    return dict(from_phase=row_a["phase"], to_phase=row_b["phase"],
                from_t_s=row_a["t"], to_t_s=row_b["t"], initial_m=total_a / width, final_m=total_b / width,
                shared_m=shared / width, entered_m=max(0., total_b - shared) / width,
                left_m=max(0., total_a - shared) / width)


def compare_poses(trial, follow_all_samples):
    m = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    d = mujoco.MjData(m)
    mujoco.mj_fwdPosition(m, d)
    faces = m.flex_elem.reshape(-1, 3)
    reference = d.flexvert_xpos[faces]
    pads = [m.geom(f"{name}_pad").id for name in ("left", "right")]
    trace = json.loads((trial / "trace.json").read_text())
    phases = [row["phase"] for row in trace]
    indices = [phases.index("close"), phases.index("lift")]
    if "lift_hold" in phases:
        indices.append(len(phases) - 1 - phases[::-1].index("lift_hold"))
    snapshots = {}
    sample_indices = range(indices[0], indices[-1] + 1) if follow_all_samples else indices
    with np.load(trial / "trajectory.npz") as trajectory:
        if not np.array_equal(trajectory["time"], [row["t"] for row in trace]):
            raise ValueError(f"Mismatched trajectory and force trace in {trial}")
        for index in sample_indices:
            sample = trace[index]
            d.qpos[:] = trajectory["qpos"][index]
            mujoco.mj_fwdPosition(m, d)
            center, rotation, extents = aperture_box(d, d.geom_xpos[pads], m.geom_bodyid[pads[0]], sample["gap"])
            local = ((d.flexvert_xpos - center) @ rotation)[faces]
            total = float(aperture_material_areas(local, reference, extents)[1]) if extents[1] > 0. else 0.
            if index in indices:
                identity = shared_material_area(local, local, reference, extents, extents)
                assert np.isclose(total, identity, atol=1e-12, rtol=1e-8)
            snapshots[index] = local, extents, total, sample
    transitions = [material_exchange(snapshots[a], snapshots[b], reference) for a, b in zip(indices, indices[1:])]
    curve = []
    if follow_all_samples:
        for i in sample_indices[1:]:
            row = material_exchange(snapshots[i - 1], snapshots[i], reference)
            pose = trace[i]
            row.update(actual_gap_m=pose["gap"], pull_N=-sum(force[2] for force in pose["force_on_fingers_N"]),
                       left_grip_N=pose["grip_each_N"][0], right_grip_N=pose["grip_each_N"][1],
                       support_N=pose["support_N"])
            curve.append(row)
        net = sum(row["entered_m"] - row["left_m"] for row in curve)
        assert np.isclose(net, curve[-1]["final_m"] - curve[0]["initial_m"], atol=1e-10, rtol=1e-8)
    return transitions, curve


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--curve-output", type=Path)
    args = parser.parse_args()
    points = np.array([[x, y, 0.] for x in (-1., 1.) for y in (-1., 1.)])
    square = points[[[0, 2, 3], [0, 3, 1]]]
    extents = np.array([.5, 2., 1.])
    assert np.isclose(shared_material_area(square, square - [.5, 0, 0], square, extents, extents), 1.)
    root = Path("test/runs")
    runs = {"empty": "2026-11-19/bag/rss2027-stage1-empty-yam-r4",
            "centered": "2026-11-20/bag/rss2027-stage1-pickup-baseline-cpu-r4",
            "offset_15_mm": "2026-11-20/bag/rss2027-stage1-pickup-baseline-cpu-r5",
            "compact_30_percent": "2026-11-20/bag/rss2027-stage1-pickup-baseline-cpu-r6"}
    result, curves = {}, []
    for name, run in runs.items():
        transitions, curve = compare_poses(root / run / "recording/trial", args.curve_output is not None)
        result[name] = transitions
        curves.extend({"case": name, **row} for row in curve)
    assert all(row["entered_m"] == row["left_m"] == 0. for row in result["empty"])
    if args.curve_output is not None:
        assert all(row["entered_m"] == row["left_m"] == 0. for row in curves if row["case"] == "empty")
        with args.curve_output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=curves[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(curves)
    print(json.dumps(dict(source_runs=runs, transitions=result), indent=2))


if __name__ == "__main__":
    main()
