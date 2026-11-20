"""Align the archived bag grid to each content's own folding edges."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from continue_wrap import ASSET_VERTICES, DESIGN, FACES, MIN_GAP, PANELS, RUN, incident_faces, measurements
from fit_strain_constraints import RADIUS_M, fit
from bag_sim.wrap import open_pairs, panel_pairs

ROOT = RUN.parents[4]
BASE_GRID_SIZE = np.asarray([.35, .25])
VERTEX_GRID = np.column_stack((-ASSET_VERTICES[:, 1], ASSET_VERTICES[:, 0]))
EDGES = np.unique(np.sort(np.stack((FACES, np.roll(FACES, -1, axis=1)), axis=-1).reshape(-1, 2), axis=1), axis=0)
ADJACENT = incident_faces()


def align_grid(cavity, offset, half):
    reference = np.zeros_like(ASSET_VERTICES)
    for axis in range(2):
        values, labels = np.unique(VERTEX_GRID[:, axis], return_inverse=True)
        edge_lo, edge_hi = offset[axis] + np.array([-half[axis], half[axis]])
        grid = np.r_[-cavity[axis] / 2., np.linspace(edge_lo, edge_hi, len(values) - 2), cavity[axis] / 2.]
        if np.min(np.diff(grid)) <= 0.:
            raise ValueError("Content folding edges are outside the original cavity")
        reference[:, axis] = grid[labels]
    return reference


def fold_surfaces(reference, offset, half, angle):
    local = reference[:, :2] - offset
    sign = np.sign(local)
    edge = np.clip(local, -half[:2], half[:2])
    excess = np.maximum(abs(local) - half[:2], 0.)
    largest, smallest = excess.max(axis=1), excess.min(axis=1)
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    # Two isometric facets share the diagonal of each corner ear.
    diagonal_xy = (c + np.sqrt(2.) * s**2) / (1. + s**2)
    diagonal_z = (c * diagonal_xy - 1.) / s
    shape = c * excess + (diagonal_xy - c) * smallest[:, None]
    height = half[2] - s * largest + (diagonal_z + s) * smallest
    depth = np.maximum(-height, 0.)
    direction = excess / np.maximum(np.linalg.norm(excess, axis=1, keepdims=True), 1e-12)
    shape += (1. - c) * depth[:, None] / s * direction
    upper = np.unique(FACES[PANELS == 0])
    lower = np.unique(FACES[PANELS == 3])
    side = np.zeros(len(reference))
    side[np.setdiff1d(upper, lower)] = 1.
    side[np.setdiff1d(lower, upper)] = -1.
    points = np.column_stack((edge + sign * shape, side * np.maximum(height, 0.)))
    open_pairs(points, panel_pairs(reference, FACES, PANELS, [0], [3]), MIN_GAP)
    return points


def sample_fold(index, shape):
    cavity = np.array([float(shape[f"inside_{axis}_m"]) for axis in "xy"])
    size = np.array([float(shape[f"payload_{axis}_m"]) for axis in "xyz"])
    offset = np.array([float(shape[f"offset_{axis}_m"]) for axis in "xy"])
    half = size / 2. + RADIUS_M
    reference = align_grid(cavity, offset, half + 5e-6)
    item = dict(half_size=half, position=np.zeros(3))
    rows = []
    best = None
    for angle in (86., 88., 90.):
        seed = fold_surfaces(reference, offset, half + 5e-6, angle)
        initial = measurements(reference, seed, EDGES, item)
        if initial["outside_corners"] or initial["intersecting_triangles"]:
            rows.append(dict(angle_deg=angle, initial=initial, fitted=None))
            continue
        points, checks = fit(seed, reference, half, EDGES, ADJACENT, 1500)
        final = checks[-1]
        rows.append(dict(angle_deg=angle, initial=initial, fitted=final))
        if final["enclosed"] and final["minimum_wall_gap_m"] >= MIN_GAP - 1e-9:
            if best is None or final["maximum_edge_strain"] < best[0]:
                best = final["maximum_edge_strain"], points, angle
    result = dict(sample=index, fill_fraction=float(shape["fill_fraction"]), source_geometry=shape, orientations=rows)
    return result, best, reference, cavity, size, offset


def record_case(output, result, best, reference, cavity, size, offset):
    _, points, angle = best
    case = output / f"sample-{result['sample']:03d}"
    case.mkdir(exist_ok=False)
    origin = np.r_[offset, RADIUS_M + .00015 - points[:, 2].min()]
    # Convert the physically sized grid to the archived asset's axis convention.
    scaled = reference[:, :2] * BASE_GRID_SIZE / cavity
    asset = case / "prepared_mesh.npz"
    vertices = np.column_stack((scaled[:, 1], -scaled[:, 0], np.zeros(len(reference))))
    np.savez_compressed(asset, vertices=vertices, reference_m=reference, current_m=points + origin,
                        faces=FACES, panel_ids=PANELS, payload_origin_m=origin,
                        payload_size_m=size, payload_offset_xy=offset)
    config = json.loads((RUN / "equilibrated-entry/sample-061/smoke.json").read_bytes())
    config.update(geometry=f"../{asset.resolve().relative_to(ROOT)}", bag_size=cavity.tolist(),
                  payload_size=size.tolist(), payload_offset_xy=offset.tolist())
    for name, seconds in (("smoke", .02), ("settle", .30)):
        (case / f"{name}.json").write_text(json.dumps(config | {"smoke_seconds": seconds}, indent=2) + "\n")
    return angle, asset.relative_to(output).as_posix()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--all", action="store_true", help="check all 32 original 90% samples")
    args = parser.parse_args()
    with DESIGN.open(newline="") as stream:
        entries = [(i, row) for i, row in enumerate(csv.DictReader(stream), 1) if row["group"] == "0.90"]
    keys = ("slack_x_minus_m", "slack_x_plus_m", "slack_y_minus_m", "slack_y_plus_m")
    ranked = sorted(entries, key=lambda item: min(float(item[1][key]) for key in keys), reverse=True)
    args.output.mkdir(exist_ok=False)
    results = []
    for index, row in ranked if args.all else ranked[:4]:
        result, best, *geometry = sample_fold(index, row)
        if best is not None:
            result["selected_angle_deg"], result["asset"] = record_case(args.output, result, best, *geometry)
        results.append(result)
        print(index, [item["fitted"] for item in result["orientations"]], flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
