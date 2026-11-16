"""Join the archived kraft bag's two face sheets at their outer perimeter."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def edge_sealed_geometry(source):
    with np.load(source) as mesh:
        vertices = mesh["vertices"].copy()
        panels = mesh["panel_ids"]
        sheets = [mesh["faces"][panels == panel].copy() for panel in (0, 3)]
    vertices[:, 2] = 0
    boundaries = []
    for faces in sheets:
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = np.unique(unique[counts == 1])
        boundaries.append(set(boundary))
        # Flip corner diagonals that would otherwise become four-face edges
        # when the two sheets share the perimeter vertices.
        for a, b in unique[(counts == 2) & np.all(np.isin(unique, boundary), axis=1)]:
            adjacent = np.flatnonzero(np.any(faces == a, axis=1) & np.any(faces == b, axis=1))
            c, d = [int(next(v for v in faces[index] if v not in (a, b))) for index in adjacent]
            normal_z = np.cross(vertices[faces[adjacent[0], 1]] - vertices[faces[adjacent[0], 0]],
                                vertices[faces[adjacent[0], 2]] - vertices[faces[adjacent[0], 0]])[2]
            replacement = np.array([[c, d, a], [d, c, b]])
            triangles = vertices[replacement]
            reverse = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2] * normal_z < 0
            replacement[reverse] = replacement[reverse, ::-1]
            faces[adjacent] = replacement

    points, joined, seam_lookup = [], [], {}
    seam_vertices = []
    for faces, boundary in zip(sheets, boundaries):
        remap = {}
        for old in np.unique(faces):
            xy = tuple(np.round(vertices[old, :2], 12))
            if old in boundary and xy in seam_lookup:
                remap[old] = seam_lookup[xy]
                continue
            remap[old] = len(points)
            points.append(vertices[old])
            if old in boundary:
                seam_lookup[xy] = remap[old]
                seam_vertices.append(remap[old])
        joined.append(np.array([[remap[v] for v in face] for face in faces]))
    points = np.asarray(points)
    faces = np.concatenate(joined)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2) or len(np.unique(directed, axis=0)) != len(directed):
        raise ValueError("edge-sealed sheets must form a consistently oriented closed manifold")
    triangles = points[faces]
    area = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area <= 0):
        raise ValueError("edge-sealed geometry contains degenerate triangles")
    masses = np.zeros(len(points))
    np.add.at(masses, faces.ravel(), np.repeat(area * .12 / 3, 3))
    return dict(vertices=points, faces=faces,
                panel_ids=np.repeat([0, 3], [len(sheet) for sheet in joined]),
                masses_kg=masses, uv=(points[:, :2] - points[:, :2].min(0)) / np.ptp(points[:, :2], axis=0),
                seam_vertices=np.asarray(seam_vertices),
                source_sha256=np.asarray(hashlib.sha256(Path(source).read_bytes()).hexdigest()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "geometry/kraft_coarse_35.npz")
    parser.add_argument("--output", type=Path, default=HERE / "geometry/kraft_edge_sealed_35.npz")
    args = parser.parse_args()
    mesh = edge_sealed_geometry(args.source)
    np.savez_compressed(args.output, **mesh)
    print(f"{args.output}: {len(mesh['vertices'])} vertices, {len(mesh['faces'])} triangles, "
          f"{len(mesh['seam_vertices'])} shared perimeter vertices")


if __name__ == "__main__":
    main()
