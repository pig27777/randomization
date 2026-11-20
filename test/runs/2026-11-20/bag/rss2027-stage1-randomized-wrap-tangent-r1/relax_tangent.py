"""Relax the original wrapping along separating planes of the contents."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-volume-projection-r1")))
from project_volume import ASSET_VERTICES, CASES, DESIGN, FACES, PANELS, RADIUS, measurements  # noqa: E402
from bag_sim.legacy.contents import normalize_contents  # noqa: E402
from bag_sim.legacy.initialization import edge_colors  # noqa: E402
from bag_sim.wrap import open_pairs, panel_pairs, wrap_generic  # noqa: E402

ITERATIONS = 800
MIN_GAP = 2 * RADIUS + .0002
PLANE_MARGIN = .000010


def separating_planes(points, half):
    """Choose an exterior supporting halfspace containing each triangle."""
    tri = points[FACES]
    edge = np.roll(tri, -1, axis=1) - tri
    normal = np.cross(edge[:, 0], edge[:, 1])
    cross = np.cross(edge[:, :, None], np.eye(3)).reshape(len(tri), 9, 3)
    axes = np.concatenate((np.broadcast_to(np.eye(3), (len(tri), 3, 3)), normal[:, None], cross), axis=1)
    sizes = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(sizes[:, :, None], 1e-30)
    projection = np.einsum("tvi,tai->tva", tri, axes)
    height = abs(axes) @ half
    positive = projection.min(axis=1) - height
    negative = -projection.max(axis=1) - height
    margins = np.where(sizes > 1e-14, np.maximum(positive, negative), -np.inf)
    index = margins.argmax(axis=1)
    row = np.arange(len(tri))
    separation = margins[row, index]
    if separation.min() < -1e-12:
        raise ValueError("Starting triangle has no exterior halfspace")
    sign = np.where(positive[row, index] >= negative[row, index], 1., -1.)
    # A small available clearance keeps an exact box corner from ending up on
    # a membrane face, where its winding classification becomes ambiguous.
    distance = height[row, index] + np.minimum(separation, PLANE_MARGIN)
    return axes[row, index] * sign[:, None], distance


def incident_faces():
    incident = [[] for _ in ASSET_VERTICES]
    for i, face in enumerate(FACES):
        for vertex in face:
            incident[vertex].append(i)
    degree = max(map(len, incident))
    return np.array([faces + [-1] * (degree - len(faces)) for faces in incident])


def project_step(previous, proposed, normals, distances, adjacent):
    direction = normals[adjacent]
    bounds = np.where(adjacent >= 0, distances[adjacent], -np.inf)
    # Resolve each vertex's incident constraints without truncating the other
    # vertices' allowed tangential motion when one triangle reaches the box.
    for _ in range(3):
        for n, bound in zip(direction.transpose(1, 0, 2), bounds.T, strict=True):
            error = np.maximum(bound - np.einsum("ij,ij->i", proposed, n), 0.)
            proposed += error[:, None] * n
    velocity = proposed - previous
    margin = np.maximum(np.einsum("ni,nki->nk", previous, direction) - bounds, 0.)
    speed = -np.einsum("ni,nki->nk", velocity, direction)
    scale = np.divide(margin, speed, out=np.full_like(margin, np.inf), where=speed > 0.)
    scale = np.minimum(scale.min(axis=1), 1.)
    return previous + scale[:, None] * velocity


def edge_targets(reference, edges):
    return [(a, b, np.linalg.norm(reference[b] - reference[a], axis=1))
            for a, b in (group.T for group in edge_colors(edges))]


def relax_once(local, half, groups, pairs, adjacent, strain_band=0.):
    normals, distance = separating_planes(local, half)
    proposed = local.copy()
    for a, b, target in groups:
        delta = proposed[b] - proposed[a]
        length = np.linalg.norm(delta, axis=1)
        bounded = np.clip(length, target * (1. - strain_band), target * (1. + strain_band))
        correction = .45 * ((length - bounded) / np.maximum(length, 1e-12))[:, None] * delta
        proposed[a] += correction
        proposed[b] -= correction
    open_pairs(proposed, pairs, MIN_GAP)
    return project_step(local, proposed, normals, distance, adjacent)


def run_case(sample, row):
    reference = np.column_stack((-ASSET_VERTICES[:, 1], ASSET_VERTICES[:, 0], ASSET_VERTICES[:, 2]))
    reference[:, :2] *= [float(row[f"inside_{axis}_m"]) / original for axis, original in zip("xy", (.35, .25))]
    size = [float(row[f"payload_{axis}_m"]) for axis in "xyz"]
    offset = [float(row[f"offset_{axis}_m"]) for axis in "xy"]
    item = normalize_contents([dict(shape="box", size_m=size, mass_kg=.20)])[0]
    assembled, prep = wrap_generic(reference, reference.copy(), FACES, PANELS, [0], [3], item, RADIUS,
                                   box_face_clearance=True, content_offset_xy=offset)
    edges = np.unique(np.sort(np.concatenate((FACES[:, [0, 1]], FACES[:, [1, 2]], FACES[:, [2, 0]])), axis=1), axis=0)
    before = measurements(reference, assembled, edges, item)
    np.testing.assert_allclose(before["maximum_edge_strain"], prep["maximum_edge_strain"], rtol=0., atol=1e-12)
    before["minimum_wall_gap_m"] = prep["minimum_wall_gap_m"]
    pairs = panel_pairs(reference, FACES, PANELS, [0], [3])
    adjacent = incident_faces()
    center, half = item["position"], item["half_size"]
    local = assembled - center
    groups = edge_targets(reference, edges)
    best = local
    best_strain = before["maximum_edge_strain"]
    checkpoints = []
    for tick in range(1, ITERATIONS + 1):
        local = relax_once(local, half, groups, pairs, adjacent)
        if tick % 20:
            continue
        check = measurements(reference, local + center, edges, item)
        gap = float((local[pairs[:, 0], 2] - local[pairs[:, 1], 2]).min())
        maximum = check["maximum_edge_strain"]
        checkpoints.append(dict(iteration=tick, strain=maximum, minimum_wall_gap_m=gap))
        if check["outside_corners"] or check["intersecting_triangles"]:
            raise ValueError(f"Lost enclosure in sample {sample} at iteration {tick}: {check}")
        if gap >= MIN_GAP - 1e-9 and maximum < best_strain:
            best, best_strain = local, maximum
    after = measurements(reference, best + center, edges, item)
    after["minimum_wall_gap_m"] = float((best[pairs[:, 0], 2] - best[pairs[:, 1], 2]).min())
    result = dict(sample=sample, fill_fraction=float(row["fill_fraction"]), before=before, after=after,
                  best_sampled_strain=min(row["strain"] for row in checkpoints),
                  max_retained_node_displacement_m=float(np.linalg.norm(best + center - assembled, axis=1).max()),
                  checkpoints=checkpoints)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "summary.json")
    args = parser.parse_args()
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    results = []
    for sample in CASES:
        result = run_case(sample, design[sample - 1])
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "checkpoints"}), flush=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(dict(iterations=ITERATIONS, plane_margin_m=PLANE_MARGIN, results=results),
                                indent=2) + "\n")


if __name__ == "__main__":
    main()
