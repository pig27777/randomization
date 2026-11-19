"""Measure surface area beyond planar projection inside the recorded aperture."""

import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
from rss2027_yam_capture_audit import aperture_box, clip_plane  # noqa: E402


def areas(current, half):
    spatial = projected = 0.
    for triangle in current:
        polygon = triangle
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
            polygon = clip_plane(polygon, axis, sign, half[axis])
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            fan = polygon[1:] - polygon[0]
            cross = np.cross(fan[:-1], fan[1:])
            spatial += .5 * np.linalg.norm(cross, axis=1).sum()
            projected += .5 * np.abs(cross[:, 2]).sum()
    return spatial, projected


def read_case(case):
    trial = case / "recording/trial"
    trace = {row["t"]: row for row in json.loads((trial / "trace.json").read_bytes())}
    with (case / "capture/capture-audit.csv").open(newline="") as stream:
        capture = {float(row["t_s"]): row for row in csv.DictReader(stream) if row["phase"] == "close"}
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    pads = [model.geom(name).id for name in ("left_pad", "right_pad")]
    left = model.geom_bodyid[pads[0]]
    faces = model.flex_elem.reshape(-1, 3)
    rows = []
    with np.load(trial / "trajectory.npz") as state:
        for time, qpos in zip(state["time"], state["qpos"], strict=True):
            if time not in capture:
                continue
            data.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, data)
            row = capture[time]
            gap = float(row["actual_gap_m"])
            center, rotation, half = aperture_box(data, data.geom_xpos[pads], left, gap)
            triangles = ((data.flexvert_xpos - center) @ rotation)[faces]
            candidates = np.all(triangles.max(axis=1) >= -half, axis=1)
            candidates &= np.all(triangles.min(axis=1) <= half, axis=1)
            actual, projected = areas(triangles[candidates], half) if gap > 0. else (0., 0.)
            np.testing.assert_allclose(actual, float(row["aperture_area_m2"]), atol=1e-12, rtol=0.)
            rows.append(dict(variant=case.name, t_s=float(time), actual_gap_m=gap, downforce_N=trace[time]["down_N"],
                             aperture_reference_length_m=float(row["aperture_material_length_m"]),
                             spatial_area_m2=float(actual), planar_projection_m2=float(projected),
                             extra_length_m=float((actual - projected) / (2 * half[0]))))
    if len(rows) != len(capture):
        raise ValueError(f"Incomplete closure replay for {case}")
    return rows


def main():
    flat = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    tilted = flat.copy()
    tilted[0, 2, 2] = 1.
    np.testing.assert_allclose(areas(flat, [2., 2., 2.]), [.5, .5])
    np.testing.assert_allclose(areas(tilted, [2., 2., 2.]), [np.sqrt(2.) / 2, .5])
    cases = [read_case(RUN / variant) for variant in ("coarse", "refined")]
    report = {rows[0]["variant"]: dict(frames=len(rows),
              peak_total=max(rows, key=lambda row: row["aperture_reference_length_m"]),
              peak_excess=max(rows, key=lambda row: row["extra_length_m"])) for rows in cases}
    with (RUN / "surface-excess.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with (RUN / "surface-excess.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cases[0][0], lineterminator="\n")
        writer.writeheader()
        for rows in cases:
            writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
