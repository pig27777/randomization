"""Fit exact high-fill shapes under the existing geometric acceptance bounds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from continue_wrap import (DESIGN, FACES, MIN_GAP, PANELS, RUN, edge_targets, grow_near_contents,
                           incident_faces, measurements, panel_pairs, separating_planes)

SOURCES = {
    74: "box-clearance/sample-074/prepared_mesh.npz",
    90: "randomized-entry/sample-090/prepared_mesh.npz",
}
RADIUS_M = .00025


def project_edge_band(points, group, axis, corrections):
    a, b, length = group
    left, right = points[a] + corrections[0], points[b] + corrections[1]
    difference = right - left
    maximum, minimum = 1.04999 * length, .95001 * length
    norm = np.linalg.norm(difference, axis=1)
    bounded = difference * np.minimum(1., maximum / np.maximum(norm, 1e-15))[:, None]
    # A conservative halfspace enforces the lower length bound for this edge.
    beyond = np.einsum("ij,ij->i", bounded, axis) < minimum
    parallel = np.einsum("ij,ij->i", difference, axis)
    tangent = difference - parallel[:, None] * axis
    tangent_norm = np.linalg.norm(tangent, axis=1)
    available = np.sqrt(maximum**2 - minimum**2)
    clipped = tangent * np.minimum(1., available / np.maximum(tangent_norm, 1e-15))[:, None]
    bounded[beyond] = (minimum[:, None] * axis + clipped)[beyond]
    change = .5 * (difference - bounded)
    points[a], points[b] = left + change, right - change
    corrections[0][:], corrections[1][:] = -change, change


def fit(start, reference, half, edges, adjacent, max_rounds, *, wall_gap_m=MIN_GAP):
    normals, bound = separating_planes(start, half)
    directions = normals[adjacent].transpose(1, 0, 2)
    limits = np.where(adjacent >= 0, bound[adjacent], -np.inf).T
    plane_errors = np.zeros_like(directions)
    groups = edge_targets(reference, edges)
    directions_at_start = [start[b] - start[a] for a, b, _ in groups]
    axes = [edge / np.linalg.norm(edge, axis=1)[:, None] for edge in directions_at_start]
    edge_errors = [np.zeros((2, len(a), 3)) for a, _, _ in groups]
    front, back = panel_pairs(reference, FACES, PANELS, [0], [3]).T
    wall_error = np.zeros(len(front))
    item = dict(position=np.zeros(3), half_size=half, rotation=np.eye(3))

    def inspect(round_id):
        return dict(iteration=round_id, minimum_wall_gap_m=float(np.min(points[front, 2] - points[back, 2])),
                    **measurements(reference, points, edges, item))

    points = start.copy()
    rows = [inspect(0)]
    for i in range(1, max_rounds + 1):
        for group, axis, residual in zip(groups, axes, edge_errors, strict=True):
            project_edge_band(points, group, axis, residual)
        front_z, back_z = points[front, 2] + wall_error, points[back, 2] - wall_error
        adjust = .5 * np.maximum(wall_gap_m - (front_z - back_z), 0.)
        points[front, 2], points[back, 2] = front_z + adjust, back_z - adjust
        wall_error[:] = -adjust
        for direction, limit, residual in zip(directions, limits, plane_errors, strict=True):
            displaced = points + residual
            adjust = np.maximum(limit - np.einsum("ij,ij->i", displaced, direction), 0.)[:, None] * direction
            points = displaced + adjust
            residual[:] = -adjust
        if i % 250 == 0:
            result = inspect(i)
            rows.append(result)
            if result["enclosed"] and result["minimum_wall_gap_m"] >= wall_gap_m - 1e-9:
                break
    return points, rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    pairs = np.stack((FACES, np.roll(FACES, -1, axis=1)), axis=-1).reshape(-1, 2)
    edges = np.unique(np.sort(pairs, axis=1), axis=0)
    adjacent = incident_faces()
    results = []
    for sample, asset in SOURCES.items():
        with np.load(RUN / asset) as source:
            data = dict(source.items())
        reference = data["reference_m"]
        target = np.array([float(design[sample - 1][f"payload_{axis}_m"]) for axis in "xyz"])
        offset = np.array([float(design[sample - 1][f"offset_{axis}_m"]) for axis in "xy"])
        np.testing.assert_allclose(data["payload_offset_xy"], offset, atol=1e-12, rtol=0.)
        half = target / 2 + RADIUS_M
        local = data["current_m"] - data["payload_origin_m"]
        scale = target / data["payload_size_m"]
        np.testing.assert_allclose(scale, scale[0], atol=1e-12, rtol=0.)
        local_seed = grow_near_contents(local, data["payload_size_m"] / 2 + RADIUS_M, half, adjacent)
        for label, seed in (("global", local * scale), ("local", local_seed)):
            if seed is None:
                results.append(dict(sample=sample, seed=label, status="no_enclosed_seed"))
                continue
            points, checks = fit(seed, reference, half, edges, adjacent, 3000)
            final = checks[-1]
            accepted = final["enclosed"] and final["minimum_wall_gap_m"] >= MIN_GAP - 1e-9
            status = "target_passed" if accepted else "constraint_fit_failed"
            if accepted:
                origin = data["payload_origin_m"].copy()
                origin[2] = RADIUS_M + .00015 - points[:, 2].min()
                prepared = data | dict(current_m=points + origin, payload_origin_m=origin, payload_size_m=target)
                out = args.output / f"sample-{sample:03d}-{label}"
                out.mkdir(exist_ok=False)
                np.savez_compressed(out / "prepared_mesh.npz", **prepared)
            print(sample, label, status, final, flush=True)
            results.append(dict(sample=sample, seed=label, status=status,
                                fill_fraction=float(design[sample - 1]["fill_fraction"]), checks=checks))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
