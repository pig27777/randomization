"""Align whole-bag material withdrawal, supported load and fingertip release."""

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "test" / "bag_model"))
from rss2027_micropatch import sustained_starts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    run = parser.parse_args().run
    trial = run / "recording" / "trial"
    result = json.loads((trial / "result.json").read_text())
    trace = json.loads((trial / "trace.json").read_text())
    open_trace = [frame for frame in trace if frame["phase"] == "open"]
    with (run / "capture" / "capture-audit.csv").open() as stream:
        aperture = [row for row in csv.DictReader(stream) if row["phase"] == "open"]
    time = np.array([float(row["t_s"]) for row in aperture])
    if not len(time) or not np.array_equal(time, [row["t"] for row in open_trace]):
        raise ValueError("Release geometry and raw load times are missing or misaligned")
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    radius = float(model.flex_radius[0])
    fields = ("aperture_material_length_m", "pad_height_m", "com_z_m", "tension_N", "support_N",
              "left_grip_N", "right_grip_N")
    material, finger_z, com_z, tension, support, grip_left, grip_right = np.array(
        [[float(row[key]) for key in fields] for row in aperture]).T
    held = (grip_left > .001) & (grip_right > .001) & (tension > .001) & (support <= .05)
    drop = finger_z - com_z
    withdrew = np.maximum.accumulate(material) - material
    sliding = sustained_starts(time, held & (withdrew > radius) & (drop - np.minimum.accumulate(drop) > radius))
    no_contact = np.array([max(row["contact_magnitude_each_N"]) <= .05 for row in open_trace])
    last_contact = np.flatnonzero(~no_contact)
    detached = int(last_contact[-1] + 1) if len(last_contact) else 0
    report = dict(
        stage_outcome=result["outcome"], release_samples=len(time), numerical_ok=not result["warnings"]
        and result["peaks"]["penetration_mm"] <= 1.,
        opening_start_t_s=float(time[0]), first_open_sample=aperture[0],
        sliding_candidate=aperture[sliding[0]] if sliding else None,
        last_contact_sample=aperture[detached - 1] if detached else None,
        first_final_contact_loss_sample=aperture[detached] if detached < len(time) else None,
        support_max_N=float(support.max()), contact_radius_m=radius,
    )
    with (run / "release-check.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
