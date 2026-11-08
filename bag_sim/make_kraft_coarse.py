"""Write a coarse gusseted kraft mailer mesh in the unified geometry format (round 2, F1).

Same body as geometry/kraft.npz (250 x 350 mm, front z = +4 mm, back z = -4 mm, 25 mm side gussets folded in
along a crease at z = 0), but the flap is left out and both ends are sealed: the cross-section loop of each end
row is closed by a fan to a centre vertex (a sealed mailer has its flap glued down). Panel ids follow kraft.npz:
0 front, 3 back, 1 / 4 right / left gusset (front half), 2 / 5 (back half), 6 end seals.

Usage: python -B make_kraft_coarse.py [--edge 0.05] [--out geometry/kraft_coarse_50.npz]
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WIDTH, LENGTH, GUSSET, HALF_GAP, MASS = 0.25, 0.35, 0.025, 0.004, 0.028037894582216283   # kraft.npz body


def _triangulate(points):
    """Incremental planar triangulation; no runtime geometry dependency for F6d."""
    n = len(points)
    span = np.ptp(points, axis=0).max() * 10
    c = points.mean(0)
    pts = np.vstack([points, c + [[-span, -span], [span, -span], [0, span]]])
    triangles = [(n, n + 1, n + 2)]
    for i in range(n):
        bad = []
        for t in triangles:
            a, b, d = pts[list(t)]
            mat = 2 * np.array([b - a, d - a])
            if abs(np.linalg.det(mat)) < 1e-15:
                continue
            center = np.linalg.solve(mat, [b @ b - a @ a, d @ d - a @ a])
            if (pts[i] - center) @ (pts[i] - center) <= (a - center) @ (a - center) + 1e-14:
                bad.append(t)
        edges = {}
        for t in bad:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e = tuple(sorted((a, b)))
                edges[e] = edges.get(e, 0) + 1
        triangles = [t for t in triangles if t not in bad]
        triangles += [(a, b, i) for (a, b), count in edges.items() if count == 1]
    return np.array([t for t in triangles if max(t) < n], dtype=np.int32)


def build(edge, width=WIDTH, length=LENGTH, gusset=GUSSET, mass=MASS, bottom_edge=None, content_length=0.):
    """Mesh with edges of about `edge`; width / length of the flat body, total mass (default: kraft.npz)."""
    WIDTH_, LENGTH_ = width, length
    nx, ny = max(2, round(WIDTH_ / edge)), max(2, round(LENGTH_ / edge))
    xs = np.linspace(-WIDTH_ / 2, WIDTH_ / 2, nx + 1)
    # cross-section loop (x, z, panel of the face to the next point): front left->right, right gusset crease,
    # back right->left, left gusset crease
    loop = [(x, HALF_GAP, 0) for x in xs[:-1]] + [(xs[-1], HALF_GAP, 1), (WIDTH_ / 2 - gusset, 0.0, 2)]
    loop += [(x, -HALF_GAP, 3) for x in xs[::-1][:-1]] + [(xs[0], -HALF_GAP, 4), (-WIDTH_ / 2 + gusset, 0.0, 5)]
    n = len(loop)
    ys = np.linspace(-LENGTH_ / 2, LENGTH_ / 2, ny + 1)
    verts = [(x, y, z) for y in ys for x, z, _ in loop]
    faces, pids = [], []
    for r in range(ny):
        for i in range(n):
            a, b = r * n + i, r * n + (i + 1) % n
            c, d = a + n, b + n
            p = loop[i][2]
            faces += [(a, b, d), (a, d, c)]
            pids += [p, p]
    for r, flip in ((0, True), (ny, False)):          # end seals
        verts.append((0.0, ys[r], 0.0))
        cen = len(verts) - 1
        for i in range(n):
            a, b = r * n + i, r * n + (i + 1) % n
            faces.append((a, cen, b) if flip else (a, b, cen))
            pids.append(6)
    v = np.array(verts, float)
    f = np.array(faces, np.int32)
    pids = np.array(pids, np.int32)
    if bottom_edge is not None:
        # Keep seams and both empty ends unchanged. Retriangulate only the bottom panel,
        # retaining every perimeter vertex so the flex surface stays closed.
        bottom = np.unique(f[pids == 3])
        keep = bottom[(np.abs(v[bottom, 0]) >= width / 2 - 1e-10) |
                      (np.abs(v[bottom, 1]) >= content_length / 2 + edge)]
        bx = np.linspace(-width / 2, width / 2, max(2, round(width / bottom_edge)) + 1)[1:-1]
        by = np.linspace(-content_length / 2, content_length / 2,
                         max(2, round(content_length / bottom_edge)) + 1)
        extra = np.array([(x, y, -HALF_GAP) for y in by for x in bx])
        inds = np.r_[keep, np.arange(len(v), len(v) + len(extra))]
        v = np.vstack([v, extra])
        tri = inds[_triangulate(v[inds, :2])]
        # Original bottom winding has outward normal -z.
        cross = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])[:, 2]
        tri[cross > 0] = tri[cross > 0, ::-1]
        f = np.vstack([f[pids != 3], tri])
        pids = np.r_[pids[pids != 3], np.full(len(tri), 3)]
        used, inverse = np.unique(f, return_inverse=True)
        v, f = v[used], inverse.reshape(-1, 3)
    # outward orientation (signed volume > 0)
    if np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() < 0:
        f = f[:, ::-1].copy()
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    m = np.zeros(len(v))
    np.add.at(m, f.ravel(), np.repeat(area / 3, 3))
    return dict(vertices=v, faces=f, panel_ids=pids, masses_kg=m * mass / m.sum(),
                uv=v[:, :2].copy())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = build(a.edge)
    out = a.out or str(HERE / "geometry" / f"kraft_coarse_{round(a.edge * 1e3)}.npz")
    np.savez(out, **g)
    print(out, len(g["vertices"]), "vertices", len(g["faces"]), "faces")


if __name__ == "__main__":
    main()
