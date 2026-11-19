"""Compare the center line and width-integrated aperture at recorded PE events."""

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "test" / "bag_model"))
from rss2027_yam_capture_audit import aperture_material_areas  # noqa: E402


def measure(case):
    summary = json.loads((case / "summary.json").read_text())
    with (case / "curve.csv").open() as stream:
        curve = list(csv.DictReader(stream))
    with np.load(case / "coupon_nodes.npz") as states:
        reference, faces, current = (states[key] for key in ("reference_m", "faces", "current_m"))
    if len(curve) != len(current):
        raise ValueError("Recorded curve and material states have different lengths")
    model = mujoco.MjModel.from_binary_path(str(case / "model.mjb"))
    pad = model.geom("right_core").id
    pad_center_x = float(model.body_pos[model.geom_bodyid[pad], 0])
    mesh = model.geom_dataid[pad]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[pad])
    core_half_thickness = float(np.max(np.abs(vertices @ rotation.reshape(3, 3).T)[:, 0]))
    time_s = np.array([float(row["t_s"]) for row in curve])
    events = {}
    for event in ("capture", "traction_slip_candidate", "release"):
        stamp = summary[f"{event}_t_s"]
        if stamp is None:
            continue
        index = int(np.argmin(np.abs(stamp - time_s)))
        row = curve[index]
        if abs(time_s[index] - stamp) > 1e-9 or row["phase"] != "pull":
            raise ValueError(f"Could not align {event} with the recorded pull")
        half_gap = pad_center_x - core_half_thickness - float(row["gather_per_pad_m"])
        half_extents = np.array([half_gap, .010, .020])
        local = current[index][faces] - [0., 0., float(row["pad_z_m"])]
        possible = ((local.min(axis=1) <= half_extents) & (local.max(axis=1) >= -half_extents)).all(axis=1)
        area, material = aperture_material_areas(local[possible], reference[faces[possible]], half_extents)
        events[event] = dict(
            time_s=float(time_s[index]), center_line_material_m=float(row["aperture_material_length_m"]),
            center_line_fold_excess_m=float(row["fold_excess_length_m"]),
            width_averaged_material_m=float(material / (2 * half_extents[1])),
            width_averaged_deformed_length_m=float(area / (2 * half_extents[1])),
            pull_N=float(row["signed_pull_N"]),
            total_inner_N=float(row["left_inner_N"]) + float(row["right_inner_N"]),
        )
    fields = ("t_s", "pad_z_m", "signed_pull_N", "left_inner_N", "right_inner_N")
    pull = np.array([[float(row[key]) for key in fields] for row in curve if row["phase"] == "pull"])
    return dict(case=str(case), events=events), summary, pull


def force_windows(summary, pull, start, end):
    time, height, tension, left, right = pull.T
    loaded = (left > .001) & (right > .001) & (tension > .001)
    before_slip = (time >= start - 1e-9) & (time < end - 1e-9) & loaded
    z, force = height[before_slip], tension[before_slip]
    if len(z) < 2:
        raise ValueError("No common loaded window before the geometry-based slip candidate")
    slope, offset = np.polyfit(z, force, 1)
    residual = np.sum((force - slope * z - offset)**2)
    total = np.sum((force - force.mean())**2)
    slip = summary["traction_slip_candidate_t_s"]
    onset = int(np.argmin(np.abs(time - slip)))
    normal = left + right
    mu_candidate = tension[onset] / normal[onset]
    extra = tension - mu_candidate * normal
    after_onset = (time >= slip - 1e-9) & (time < summary["release_t_s"])
    overload = after_onset & loaded & (extra > .001)
    bounds = np.flatnonzero(np.diff(np.r_[False, overload, False])).reshape(-1, 2)
    longest = max((time[last - 1] - time[first] for first, last in bounds), default=0.)
    return dict(pre_slip_window_s=[start, end], loaded_sample_count=len(z),
                pre_slip_slope_N_per_m=float(slope), pre_slip_r_squared=float(1 - residual / total),
                traction_force_ratio=float(mu_candidate),
                overloaded_bilateral_sample_count=int(overload.sum()),
                longest_overloaded_bilateral_s=float(longest))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("traction_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = {level: measure(args.traction_run / level) for level in ("level-1", "level-2")}
    start = max(summary["capture_t_s"] for _, summary, _ in cases.values())
    end = min(summary["slip_candidate_t_s"] for _, summary, _ in cases.values())
    measurements = {}
    for level, (result, summary, curve) in cases.items():
        measurements[level] = result | {"force_windows": force_windows(summary, curve, start, end)}
    with args.output.open("x") as stream:
        stream.write(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    main()
