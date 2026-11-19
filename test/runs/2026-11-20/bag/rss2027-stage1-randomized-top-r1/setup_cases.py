"""Translate geometrically accepted randomized inner contours into YAM cases."""

import csv
import json
from pathlib import Path
import sys


RUN = Path(__file__).resolve().parent
WRAP = RUN.with_name("rss2027-stage1-randomized-wrap-r1")
sys.path[:0] = [str(WRAP), str(RUN.parents[3] / "bag_model")]
from check_wrapping import DESIGN, MESH  # noqa: E402
from physical_parcel_yam import Config, parcel_world  # noqa: E402


def main():
    with DESIGN.open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    with (WRAP / "wrapping.csv").open(newline="") as stream:
        valid_ids = {int(row["sample"]) for row in csv.DictReader(stream)
                     if row["contour"] == "inside_seams" and row["status"] == "enclosed"}
    for sample_id in (4, 127, 2):
        if sample_id not in valid_ids:
            raise ValueError(f"Sample {sample_id} did not pass its original inner-contour assembly")
        row = samples[sample_id - 1]
        config = dict(category="bendable", mode="rigid", grasp_face="top", geometry=MESH, bag_profile="pe",
                      bag_size=[float(row[f"inside_{axis}_m"]) for axis in "xy"],
                      payload_size=[float(row[f"payload_{axis}_m"]) for axis in "xyz"],
                      payload_offset_xy=[float(row[f"offset_{axis}_m"]) for axis in "xy"],
                      payload_mass=.20, dt=.0002, ccd_tolerance_m=1e-9, contact_time=.001,
                      press_target_N=.5, press_timeout_s=15., press_start_offset_m=.013,
                      close_gap_hold_s=.30, pickup_only=True, simulation_seconds=22., wall_limit=1200.)
        case = RUN / f"sample-{sample_id:03d}"
        case.mkdir(exist_ok=False)
        parcel, _ = parcel_world(Config(**config))
        geometry = parcel.metadata["geometry"]
        assert geometry["vertices"] == 142
        assert geometry["preparation"]["maximum_edge_strain"] <= .05
        inputs = dict(sample=sample_id, fill_fraction=float(row["fill_fraction"]),
                      seam_and_shape_inputs=row, geometry=geometry)
        for name, value in (("config.json", config), ("smoke.json", config | {"smoke_seconds": .02}),
                            ("inputs.json", inputs)):
            (case / name).write_text(json.dumps(value, indent=2) + "\n")
        print(sample_id, geometry["preparation"])


if __name__ == "__main__":
    main()
