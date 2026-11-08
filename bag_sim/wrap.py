"""Wrap a bag mesh around one content as initialization (numpy only, no simulator imports).

Used by demo_contents.py and proxy_eval (Newton) and by mj_loose.py (MuJoCo 3.17).
"""
import math

import numpy as np


def panel_pairs(rest, faces, panel_ids, front_ids, back_ids):
    """Front/back vertex pairs that coincide in xy in the reference, excluding shared seam vertices."""
    front = np.unique(faces[np.isin(panel_ids, front_ids)])
    back = np.unique(faces[np.isin(panel_ids, back_ids)])
    shared = np.intersect1d(front, back)
    front, back = np.setdiff1d(front, shared), np.setdiff1d(back, shared)
    if not len(front) or not len(back):
        return np.zeros((0, 2), int)
    edge = np.median(np.linalg.norm(rest[faces[:, 1]] - rest[faces[:, 0]], axis=1))
    d = np.linalg.norm(rest[front, None, :2] - rest[None, back, :2], axis=2)
    nearest = d.argmin(axis=1)
    keep = d[np.arange(len(front)), nearest] < .5 * edge
    return np.c_[front[keep], back[nearest[keep]]]


def open_pairs(q, pairs, gap):
    """Push each front/back pair apart vertically until they are at least `gap` apart."""
    if not len(pairs):
        return
    f, b = pairs.T
    deficit = np.maximum(gap - (q[f, 2] - q[b, 2]), 0)
    q[f, 2] += .5 * deficit
    q[b, 2] -= .5 * deficit


def wrap_generic(rest, start, faces, panel_ids, front_ids, back_ids, item, radius, iterations=1600, wall_gap=None):
    """Open any front/back bag around one content, as initialization.py does for the bubble mailer.

    Front-group vertices rise and back-group vertices drop by a plateau profile over the content,
    then alternating edge-length, content-distance and wall-gap projections relax the shape. The
    wall gap keeps every front/back pair (except shared seam vertices) at least `wall_gap` apart,
    so no wall pair starts inside the VBD self-contact barrier (half the interaction distance);
    near PE corners the plain projections left edges 0.018 mm apart. Edge targets come from the
    unloaded reference, which stays unchanged.
    """
    from initialization import edge_colors
    from contents import signed_distance

    q = start.copy()
    front = np.unique(faces[np.isin(panel_ids, front_ids)])
    back = np.setdiff1d(np.unique(faces[np.isin(panel_ids, back_ids)]), front)
    center = q[front, :2].mean(axis=0)
    outer = (q[front, :2].max(axis=0) - q[front, :2].min(axis=0)) / 2
    extents = item["half_size"][:2]
    if np.any(extents + .008 >= outer):
        raise ValueError("Content does not fit inside this bag with a usable seam allowance")
    gap = float(np.median(q[front, 2]) - np.median(q[back, 2]))
    lift = max(0., 2 * item["half_size"][2] + 2 * radius + .0015 - gap)
    plateau = extents + .003
    u = np.clip((np.abs(q[:, :2] - center) - plateau) / (outer - plateau), 0, 1)
    profile = np.prod(1 - u * u * (3 - 2 * u), axis=1)
    q[front, 2] += .5 * lift * profile[front]
    q[back, 2] -= .5 * lift * profile[back]
    near = np.linalg.norm(q[:, :2] - center, axis=1)
    top, bottom = q[front[np.argmin(near[front])], 2], q[back[np.argmin(near[back])], 2]
    item["position"] = np.array([center[0], center[1], .5 * (top + bottom)])
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    groups = edge_colors(edges)
    targets = [np.linalg.norm(rest[g[:, 1]] - rest[g[:, 0]], axis=1) for g in groups]
    floor = q[:, 2].min()
    pairs = panel_pairs(rest, faces, panel_ids, front_ids, back_ids)
    wall_gap = 2 * radius + .0002 if wall_gap is None else wall_gap
    for _ in range(iterations):
        for group, target in zip(groups, targets):
            a, b = group.T
            delta = q[b] - q[a]
            length = np.linalg.norm(delta, axis=1)
            correction = .45 * ((length - target) / np.maximum(length, 1e-12))[:, None] * delta
            q[a] += correction
            q[b] -= correction
        distance, normal = signed_distance(q, item)
        q += np.maximum(radius + .0003 - distance, 0)[:, None] * normal
        open_pairs(q, pairs, wall_gap)
        q[:, 2] = np.maximum(q[:, 2], floor)
    # Rest the lowest node just above the table, and move the content with it.
    shift = radius + .00015 - q[:, 2].min()
    q[:, 2] += shift
    item["position"][2] += shift
    strain = np.linalg.norm(q[edges[:, 1]] - q[edges[:, 0]], axis=1) / lengths - 1
    gaps = q[pairs[:, 0], 2] - q[pairs[:, 1], 2] if len(pairs) else np.array([np.inf])
    return q, dict(maximum_edge_strain=float(np.abs(strain).max()), minimum_wall_gap_m=float(gaps.min()),
                   method="generic front/back plateau + projections + wall gap")


def winding_number(v, f, pts):
    """Generalized winding number of points w.r.t. a triangle surface (Jacobson et al. 2013)."""
    a = v[f[:, 0]][None] - pts[:, None]
    b = v[f[:, 1]][None] - pts[:, None]
    c = v[f[:, 2]][None] - pts[:, None]
    la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
    det = np.einsum("pij,pij->pi", a, np.cross(b, c))
    div = (la * lb * lc + np.einsum("pij,pij->pi", a, b) * lc + np.einsum("pij,pij->pi", b, c) * la
           + np.einsum("pij,pij->pi", c, a) * lb)
    return np.arctan2(det, div).sum(1) / (2 * math.pi)


def point_triangle_distance(p, v, f):
    """Unsigned distance of each point p (n, 3) to the closest triangle of (v, f) (Ericson, Real-Time Collision
    Detection 5.1.5, vectorized over points x triangles)."""
    a, b, c = (v[f[:, i]][None] for i in range(3))
    p = p[:, None]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp, cp = p - b, p - c
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
    v_, w_ = vb / denom, vc / denom
    q = a + ab * v_[..., None] + ac * w_[..., None]                     # interior
    eps = 1e-30
    t_ab = np.clip(d1 / np.maximum(d1 - d3, eps), 0, 1)
    t_ac = np.clip(d2 / np.maximum(d2 - d6, eps), 0, 1)
    t_bc = np.clip((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), eps), 0, 1)
    edges = [a + ab * t_ab[..., None], a + ac * t_ac[..., None], b + (c - b) * t_bc[..., None]]
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    dist = np.where(inside, np.linalg.norm(p - q, axis=-1), np.inf)
    for e in edges:
        dist = np.minimum(dist, np.linalg.norm(p - e, axis=-1))
    return dist.min(1)
