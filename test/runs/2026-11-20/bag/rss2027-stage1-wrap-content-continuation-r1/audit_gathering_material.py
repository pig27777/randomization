"""Measure folded geometry and reference membrane flow in the same window."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path[:0] = [str(RUN.with_name(name)) for name in ("rss2027-stage1-material-feed-r1",
                                                     "rss2027-stage1-randomized-top-local-mesh-r1")]
from measure_surface_excess import read_case  # noqa: E402
from replay import aperture_box, aperture_material_areas, material_exchange  # noqa: E402


def measure(case, comparison_key=None):
    trial = case / "recording/trial"
    comparison = json.loads((case / "physical-comparison.json").read_bytes())
    if comparison_key is not None:
        comparison = comparison[comparison_key]
    fit = comparison["fold_window"]
    if fit is None:
        raise ValueError(f"The original geometric qualification did not select a window in {case}")
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    faces = model.flex_elem.reshape(-1, 3)
    rest = data.flexvert_xpos[faces].copy()
    pads = [model.geom(f"{side}_pad").id for side in ("left", "right")]
    rows, exchanges = [], []
    sampled = {row["t_s"]: row for row in read_case(case) if fit["t_start_s"] <= row["t_s"] <= fit["t_last_s"]}
    previous = None
    with np.load(trial / "trajectory.npz") as trajectory:
        for time, qpos in zip(trajectory["time"], trajectory["qpos"], strict=True):
            if time not in sampled:
                continue
            row = sampled[time]
            data.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, data)
            center, axes, half = aperture_box(data, data.geom_xpos[pads], model.geom_bodyid[pads[0]], row["actual_gap_m"])
            triangles = ((data.flexvert_xpos - center) @ axes)[faces]
            spatial, reference = aperture_material_areas(triangles, rest, half)
            width = 2. * half[0]
            np.testing.assert_allclose(spatial, row["spatial_area_m2"], rtol=0., atol=1e-12)
            np.testing.assert_allclose(reference / width, row["aperture_reference_length_m"], rtol=0., atol=1e-12)
            current = triangles, half, reference, dict(t=time, phase="close")
            if previous is not None:
                exchanges.append(material_exchange(previous, current, rest))
            previous = current
            rows.append(row | dict(reference_area_m2=float(reference),
                                   reference_excess_m=float((reference - row["planar_projection_m2"]) / width)))
    if len(rows) != fit["samples"]:
        raise ValueError(f"The material audit did not reproduce every sample in {case}")
    start, end = rows[0], rows[-1]
    changes = [(end[key] - start[key]) / width for key in ("spatial_area_m2", "planar_projection_m2", "reference_area_m2")]
    entered, exited = (sum(row[key] for row in exchanges) for key in ("entered_m", "left_m"))
    np.testing.assert_allclose(changes[2], entered - exited, rtol=0., atol=1e-12)
    travel = start["actual_gap_m"] - np.asarray([row["actual_gap_m"] for row in rows])
    slope = np.polyfit(travel, [row["reference_excess_m"] for row in rows], 1)[0]
    return dict(space_change_m=changes[0], projection_change_m=changes[1], reference_change_m=changes[2],
                spatial_excess_growth_m=end["extra_length_m"] - start["extra_length_m"],
                reference_excess_growth_m=end["reference_excess_m"] - start["reference_excess_m"],
                entered_m=entered, exited_m=exited, reference_excess_slope_m_per_m=float(slope),
                samples=rows, exchanges=exchanges)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--comparison-key", help="key for the trial in a grouped physical comparison")
    args = parser.parse_args()
    report = measure(args.case, args.comparison_key)
    with (args.case / "gathering-material-balance.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
