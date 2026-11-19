"""Check randomized wraps with geometric projections along the box distance gradient."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-wrap-r1")))
from check_wrapping import DESIGN, MESH, ROOT  # noqa: E402
from bag_sim.legacy.contents import normalize_contents, signed_distance  # noqa: E402
from bag_sim.legacy.initialization import edge_colors  # noqa: E402
from bag_sim.wrap import open_pairs, panel_pairs, triangle_box_intersections, winding_number, wrap_generic  # noqa: E402

CASES = (4, 127, 41, 59, 87, 83)
RADIUS = .00025
CLEARANCE = RADIUS + .0003
ITERATIONS = 1600
ORDER = 6
SIGNS = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


def measurements(reference, mesh, edges, item):
    a, b = edges.T
    strains = np.linalg.norm(mesh[b] - mesh[a], axis=1) / np.linalg.norm(reference[b] - reference[a], axis=1) - 1
    maximum = int(np.argmax(abs(strains)))
    half, center = item["half_size"], item["position"]
    outside = int(np.count_nonzero(np.abs(winding_number(mesh, FACES, center + SIGNS * half)) < .5))
    intersecting = int(triangle_box_intersections(mesh, FACES, half, center).sum())
    return dict(maximum_edge_strain=float(abs(strains[maximum])), signed_max_strain=float(strains[maximum]),
                edges_over_5pct=int(np.count_nonzero(abs(strains) > .05)),
                outside_corners=outside, intersecting_triangles=intersecting,
                enclosed=bool(abs(strains[maximum]) <= .05 and not outside and not intersecting))


def project(reference, current, faces, pairs, edges, item, keep_enclosed):
    points = current.copy()
    lengths = [(a, b, np.linalg.norm(reference[b] - reference[a], axis=1))
               for a, b in (group.T for group in edge_colors(edges))]
    weights = np.array([[i / ORDER, j / ORDER, 1. - (i + j) / ORDER]
                        for i in range(ORDER + 1) for j in range(ORDER + 1 - i)])
    half, center = item["half_size"], item["position"]
    corners = center + SIGNS * half
    shortened = rejected = 0
    for _ in range(ITERATIONS):
        if keep_enclosed:
            previous = points.copy()
        for a, b, target in lengths:
            delta = points[b] - points[a]
            length = np.linalg.norm(delta, axis=1)
            adjust = .45 * ((length - target) / np.maximum(length, 1e-12))[:, None] * delta
            points[a] += adjust
            points[b] -= adjust
        locations = np.einsum("sv,fvc->fsc", weights, points[faces])
        distance, normal = signed_distance(locations.reshape(-1, 3), item)
        distance = distance.reshape(len(faces), -1)
        normal = normal.reshape(*distance.shape, 3)
        nearest = distance.argmin(axis=1)
        selected = np.flatnonzero(distance[np.arange(len(faces)), nearest] < CLEARANCE)
        bary = weights[nearest[selected]]
        depth = np.minimum(CLEARANCE - distance[selected, nearest[selected]], .003)
        direction = normal[selected, nearest[selected]]
        motion = .85 * (depth / np.sum(bary**2, axis=1))[:, None] * direction
        correction = np.zeros_like(points)
        count = np.zeros(len(points))
        np.add.at(correction, faces[selected].ravel(), (bary[:, :, None] * motion[:, None]).reshape(-1, 3))
        np.add.at(count, faces[selected].ravel(), (bary.ravel() > 0.))
        points += correction / np.maximum(count[:, None], 1.)
        open_pairs(points, pairs, 2 * RADIUS + .0002)
        if keep_enclosed:
            displacement = points - previous
            for halvings in range(9):
                if (not triangle_box_intersections(points, faces, half, center).any()
                        and np.all(np.abs(winding_number(points, faces, corners)) >= .5)):
                    shortened += halvings > 0
                    break
                displacement *= .5
                points = previous + displacement
            else:
                points = previous
                rejected += 1
    return points, shortened, rejected


def evaluate(sample, row, flat, panels, keep_enclosed):
    reference = flat.copy()
    reference[:, :2] *= [float(row[f"inside_{axis}_m"]) / original for axis, original in zip("xy", (.35, .25))]
    size = [float(row[f"payload_{axis}_m"]) for axis in "xyz"]
    offset = [float(row[f"offset_{axis}_m"]) for axis in "xy"]
    item = normalize_contents([dict(shape="box", size_m=size, mass_kg=.20)])[0]
    current, prep = wrap_generic(reference, reference.copy(), FACES, panels, [0], [3], item, RADIUS,
                                 box_face_clearance=True, content_offset_xy=offset)
    edges = np.unique(np.sort(np.concatenate((FACES[:, [0, 1]], FACES[:, [1, 2]], FACES[:, [2, 0]])), axis=1), axis=0)
    pairs = panel_pairs(reference, FACES, panels, [0], [3])
    before = measurements(reference, current, edges, item)
    np.testing.assert_allclose(before["maximum_edge_strain"], prep["maximum_edge_strain"], rtol=0, atol=1e-12)
    projected, shortened, rejected = project(reference, current, FACES, pairs, edges, item, keep_enclosed)
    after = measurements(reference, projected, edges, item)
    if keep_enclosed:
        after.update(shortened_steps=shortened, rejected_steps=rejected)
    return dict(sample=sample, fill_fraction=float(row["fill_fraction"]), original=before, projected=after)


with np.load(ROOT / "bag_sim" / MESH) as asset:
    FACES = asset["faces"]
    PANELS = asset["panel_ids"]
    ASSET_VERTICES = asset["vertices"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--keep-enclosed", action="store_true")
    args = parser.parse_args()
    flat = np.column_stack((-ASSET_VERTICES[:, 1], ASSET_VERTICES[:, 0], ASSET_VERTICES[:, 2]))
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    results = []
    for sample in CASES:
        result = evaluate(sample, design[sample - 1], flat, PANELS, args.keep_enclosed)
        results.append(result)
        print(json.dumps(result), flush=True)
    destination = RUN / ("preserved-enclosure.json" if args.keep_enclosed else "summary.json")
    with destination.open("x") as stream:
        stream.write(json.dumps(dict(iterations=ITERATIONS, barycentric_order=ORDER, clearance_m=CLEARANCE,
                                     results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
