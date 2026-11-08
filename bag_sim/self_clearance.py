"""Smallest vertex-triangle distance among the pairs VBD self contact evaluates.

VBD skips a vertex-triangle pair when a triangle vertex lies within `rings` topological rings of
the vertex (particle_topological_contact_filter_threshold, default 2). Its contact energy is a
log barrier below half the interaction distance, so a starting state closer than that launches
the walls. Usage from code: self_clearance(q, faces) -> (distance, vertex, triangle).
"""
import numpy as np


def _closest_point_on_triangles(p, a, b, c):
    """Closest points on many triangles to one point (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ap @ ab.T if ab.ndim == 1 else np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
    v, w = vb / denom, vc / denom
    out = a + ab * v[:, None] + ac * w[:, None]
    # Vertex and edge regions, applied from the most general to the most specific.
    t = np.clip(d1 / np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1e-30), 0, 1)
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out[edge_ab] = (a + ab * t[:, None])[edge_ab]
    t = np.clip(d2 / np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1e-30), 0, 1)
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out[edge_ac] = (a + ac * t[:, None])[edge_ac]
    t = np.clip((d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-30, (d4 - d3) + (d5 - d6), 1e-30), 0, 1)
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    out[edge_bc] = (b + (c - b) * t[:, None])[edge_bc]
    out[(d1 <= 0) & (d2 <= 0)] = a[(d1 <= 0) & (d2 <= 0)]
    out[(d3 >= 0) & (d4 <= d3)] = b[(d3 >= 0) & (d4 <= d3)]
    out[(d6 >= 0) & (d5 <= d6)] = c[(d6 >= 0) & (d5 <= d6)]
    return out


def rings_within(faces, count, rings=2):
    """For each vertex, the set of vertices within `rings` edges."""
    neighbours = [set() for _ in range(count)]
    for f in faces:
        for i in range(3):
            neighbours[f[i]].update((f[(i + 1) % 3], f[(i + 2) % 3]))
    near = [{v} | neighbours[v] for v in range(count)]
    for _ in range(rings - 1):
        near = [set().union(*(neighbours[u] for u in s)) | s for s in near]
    return near


def self_clearance(q, faces, rings=2):
    near = rings_within(faces, len(q), rings)
    a, b, c = q[faces[:, 0]], q[faces[:, 1]], q[faces[:, 2]]
    best = (np.inf, -1, -1)
    for v in range(len(q)):
        keep = ~np.array([bool(near[v] & {int(x), int(y), int(z)}) for x, y, z in faces])
        if not keep.any():
            continue
        closest = _closest_point_on_triangles(q[v], a[keep], b[keep], c[keep])
        d = np.linalg.norm(closest - q[v], axis=1)
        k = int(np.argmin(d))
        if d[k] < best[0]:
            best = (float(d[k]), v, int(np.flatnonzero(keep)[k]))
    return best
