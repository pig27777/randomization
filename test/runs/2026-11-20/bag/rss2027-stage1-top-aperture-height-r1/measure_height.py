"""Find the axial distance from the membrane to the archived YAM aperture."""

import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
DATA = RUN.with_name("rss2027-stage1-reference-curves-r1") / "stage1_whole_bag_data.json"
sys.path.insert(0, str(ROOT / "test/bag_model"))
from rss2027_yam_capture_audit import aperture_box, clip_plane  # noqa: E402


def axial_distance(vertices, faces, half):
    faces = vertices[faces]
    candidates = np.all(faces[:, :, :2].max(axis=1) >= -half[:2], axis=1)
    candidates &= np.all(faces[:, :, :2].min(axis=1) <= half[:2], axis=1)
    spans = []
    for face in faces[candidates]:
        polygon = face
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
            polygon = clip_plane(polygon, axis, sign, half[axis])
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            z = polygon[:, 2]
            spans.append((float(z.min()), float(z.max())))
    if not spans or half[1] <= 0.:
        return dict(axial_gap_m=None, relation="no_lateral_coverage", membrane_z_max_m=None)
    depth = half[2]
    axial_gap, relation = min(((-depth - high, "below_tip") if high < -depth else
                               (low - depth, "above_pad") if low > depth else (0., "in_aperture"))
                              for low, high in spans)
    return dict(axial_gap_m=axial_gap, relation=relation, membrane_z_max_m=max(high for _, high in spans))


def replay(case, columns):
    trial = ROOT / case["trial"]
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    pads = [model.geom(name).id for name in ("left_pad", "right_pad")]
    left = model.geom_bodyid[pads[0]]
    faces = model.flex_elem.reshape(-1, 3)
    export = case["samples"]
    rows = []
    with np.load(trial / "trajectory.npz") as archive:
        np.testing.assert_array_equal(archive["time"], [row[columns["t_s"]] for row in export])
        for packed, position in zip(export, archive["qpos"], strict=True):
            phase = packed[columns["phase"]]
            if phase not in ("close", "lift") or (phase == "lift" and rows and rows[-1]["phase"] == "lift"):
                continue
            data.qpos[:] = position
            mujoco.mj_fwdPosition(model, data)
            gap = packed[columns["actual_gap_m"]]
            center, frame, half = aperture_box(data, data.geom_xpos[pads], left, gap)
            local = (data.flexvert_xpos - center) @ frame
            geometry = axial_distance(local, faces, half)
            material = packed[columns["aperture_reference_length_m"]]
            if (geometry["relation"] == "in_aperture") != (material > 1e-9):
                raise ValueError(f"Axial contact geometry disagrees with 3D clipping for {case['id']} at {packed[0]}")
            rows.append(dict(case=case["id"], phase=phase, t_s=packed[0], actual_gap_m=gap,
                             signed_pull_N=packed[columns["signed_pull_N"]],
                             aperture_reference_length_m=material,
                             aperture_tip_center_world_z_m=float(center[2] - half[2] * frame[2, 2]), **geometry))
    return rows


def main():
    stage = json.loads(DATA.read_bytes())
    columns = {name: i for i, name in enumerate(stage["sample_columns"])}
    report, records = {}, []
    for case in stage["cases"][:4]:
        rows = replay(case, columns)
        close = [row for row in rows if row["phase"] == "close"]
        ref = dict(close_frames=len(close), below_tip_frames=sum(row["relation"] == "below_tip" for row in close),
                   smallest_close_axial_gap_m=min(row["axial_gap_m"] for row in close if row["axial_gap_m"] is not None),
                   last_close=close[-1], first_lift=next(row for row in rows if row["phase"] == "lift"))
        report[case["id"]] = ref
        records.extend(rows)
    with (RUN / "summary.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with (RUN / "axial-clearance.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
