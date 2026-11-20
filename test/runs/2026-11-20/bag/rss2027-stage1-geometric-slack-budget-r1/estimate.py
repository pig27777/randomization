"""Bound film area available beyond a complete rectangular enclosure."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-wrap-r1")))
from check_wrapping import DESIGN  # noqa: E402

PAD_WIDTH_M = .020


def enclosure_budget(cavity, box):
    film_area = 2. * np.prod(cavity)
    x, y, z = box
    payload_area = 2. * (x * y + x * z + y * z)
    return float(film_area), float(payload_area), float(film_area - payload_area)


def bounds(row, sample):
    cavity = np.array([float(row[f"inside_{axis}_m"]) for axis in "xy"])
    box = np.array([float(row[f"payload_{axis}_m"]) for axis in "xyz"])
    offset = np.array([float(row[f"offset_{axis}_m"]) for axis in "xy"])
    fill = float(row["fill_fraction"])
    np.testing.assert_allclose(fill, np.prod(box) / float(row["capacity_m3"]), atol=1e-12, rtol=0.)
    half_budget = (cavity - box[:2] - box[2]) / 2.
    sections = np.stack((half_budget + offset, half_budget - offset), axis=1)
    labels = tuple(f"slack_{axis}_{side}_m" for axis in "xy" for side in ("minus", "plus"))
    np.testing.assert_allclose(sections.ravel(), [float(row[k]) for k in labels], atol=1e-12, rtol=0.)
    area, payload, reserve = enclosure_budget(cavity, box)
    if reserve < 0. or sections.min() < -1e-12:
        raise ValueError(f"Invalid area or section geometry for sample {sample}")
    return dict(sample=sample, group=row["group"], fill_fraction=fill,
                cavity_x_m=cavity[0], cavity_y_m=cavity[1], payload_x_m=box[0], payload_y_m=box[1],
                payload_z_m=box[2], offset_x_m=offset[0], offset_y_m=offset[1],
                unfolded_inner_area_m2=area, minimum_box_enclosure_area_m2=payload,
                reserve_area_upper_m2=reserve, reserve_length_at_pad_width_upper_m=reserve / PAD_WIDTH_M,
                **dict(zip(labels, sections.ravel(), strict=True)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    _, _, square = enclosure_budget([.20, .20], [.15, .15, .05])
    np.testing.assert_allclose(square, 2. * .05**2, atol=1e-12, rtol=0.)
    with DESIGN.open(newline="") as stream:
        data = [bounds(row, i) for i, row in enumerate(csv.DictReader(stream), start=1)]
    fields = ("fill_fraction", "reserve_area_upper_m2", "reserve_length_at_pad_width_upper_m",
              "offset_x_m", "offset_y_m", "slack_x_minus_m", "slack_x_plus_m", "slack_y_minus_m", "slack_y_plus_m")
    groups = {}
    for name in dict.fromkeys(row["group"] for row in data):
        rows = [row for row in data if row["group"] == name]
        groups[name] = {"samples": len(rows)} | {field: [min(row[field] for row in rows),
                                                        max(row[field] for row in rows)] for field in fields}
    with (args.output / "slack-limits.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=data[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    info = dict(design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(), pad_width_m=PAD_WIDTH_M,
                assumptions="Unstretched two-sheet interior and an enclosed rigid rectangular solid.",
                definition="Reserve is the inner film area minus the area of the box; an upper bound for film "
                "area assigned solely to a fold outside the enclosure.", groups=groups)
    with (args.output / "slack-limits.json").open("x") as stream:
        stream.write(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
