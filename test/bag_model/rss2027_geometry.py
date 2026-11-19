"""Sample Stage 1 geometric slack bounds for a two-sheet rectangular bag."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

BAG_CUT_RANGES_M = ((.22, .50), (.15, .36))
SEAM_WIDTH_RANGE_M = (.003, .015)
FILL_FRACTIONS = (0.30, 0.60, 0.90)


def sample_geometry(rng, fill):
    cut = rng.uniform(*np.asarray(BAG_CUT_RANGES_M).T)
    seams = rng.uniform(*SEAM_WIDTH_RANGE_M, size=(2, 2))
    inner = cut - seams.sum(axis=1)
    length, width = inner
    optimum = length * width / (length + width + np.sqrt(length**2 - length * width + width**2))
    capacity = optimum * np.prod(inner - optimum)
    # Conditional on the drawn fill, accept heights with enough footprint.
    while True:
        height = rng.uniform(0., min(inner))
        maximum_footprint = inner - height
        fraction = fill * capacity / (height * np.prod(maximum_footprint))
        if fraction <= 1.:
            break
    split = rng.random()
    footprint = maximum_footprint * fraction ** np.array([split, 1 - split])
    allowance = inner - footprint - height
    offset = rng.uniform(-1, 1, size=2) * allowance / 2
    side_bounds = np.column_stack((allowance / 2 + offset, allowance / 2 - offset))
    return cut, seams, inner, capacity, np.r_[footprint, height], offset, side_bounds


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--samples", type=int, default=32, help="samples per fill group")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("at least two samples per fill group are required")
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    with (args.output / "geometry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("group", "fill_fraction", "cut_x_m", "cut_y_m",
                         "seam_x_minus_m", "seam_x_plus_m", "seam_y_minus_m", "seam_y_plus_m",
                         "inside_x_m", "inside_y_m", "capacity_m3", "payload_x_m", "payload_y_m", "payload_z_m",
                         "offset_x_m", "offset_y_m", "slack_x_minus_m", "slack_x_plus_m",
                         "slack_y_minus_m", "slack_y_plus_m"))
        for fixed in (*FILL_FRACTIONS, None):
            label = "continuous" if fixed is None else f"{fixed:.2f}"
            fills, bounds, cut_sizes = [], [], []
            for _ in range(args.samples):
                target = rng.uniform(*FILL_FRACTIONS[::2]) if fixed is None else fixed
                cut, seams, inner, capacity, payload, offset, slack = sample_geometry(rng, target)
                fill = float(np.prod(payload) / capacity)
                if not np.isclose(fill, target, atol=1e-12, rtol=0) or np.any(slack < 0):
                    raise RuntimeError(f"Geometry violated its prescribed fill or perimeter: {target=}, {fill=}, {slack=}")
                writer.writerow((label, fill, *cut, *seams.ravel(), *inner, capacity, *payload, *offset, *slack.ravel()))
                fills.append(fill)
                bounds.append(slack.ravel())
                cut_sizes.append(cut)
            cuts, sides = np.asarray(cut_sizes), np.asarray(bounds)
            if np.any(np.ptp(cuts, axis=0) <= 0) or np.any(np.ptp(sides, axis=0) <= 0):
                raise RuntimeError(f"Fixed geometry or slack found in {label} group")
            groups[label] = {"samples": args.samples, "fill_range": [min(fills), max(fills)],
                             "min_side_slack_m": sides.min(axis=0).tolist(),
                             "max_side_slack_m": sides.max(axis=0).tolist()}
    summary = {"hostname": platform.node(), "seed": args.seed, "cut_axis_ranges_m": BAG_CUT_RANGES_M,
               "seam_width_range_m": SEAM_WIDTH_RANGE_M, "groups": groups,
               "side_order": ("x_minus", "x_plus", "y_minus", "y_plus"),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
