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


def triangle_box_intersections(vertices, faces, half_size, center, rotation=None):
    """Return triangles entering a box interior using the 13 separating axes.

    Rotation maps box-local coordinates to world coordinates. Boundary-only
    contact is excluded, with a 1e-12 m numerical tolerance on normalized axes.
    """
    local = np.asarray(vertices) - center
    if rotation is not None:
        local = local @ rotation
    triangles = local[faces]
    edges = np.roll(triangles, -1, axis=1) - triangles
    normal = np.cross(edges[:, 0], edges[:, 1])
    cross_axes = np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :])
    axes = np.concatenate((np.broadcast_to(np.eye(3), (len(faces), 3, 3)),
                           normal[:, None], cross_axes.reshape(-1, 9, 3)), axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(lengths[:, :, None], 1e-30)
    projection = np.einsum("tvc,tac->tva", triangles, axes)
    extent = np.abs(axes) @ np.asarray(half_size)
    separated = ((projection.min(axis=1) >= extent - 1e-12)
                 | (projection.max(axis=1) <= -extent + 1e-12))
    return ~np.any(separated & (lengths > 1e-14), axis=1)


def _clear_box_sheet(local, faces, shared, half, clearance, sign):
    """Keep the triangle footprint over a box on its exterior z face."""
    xy = local[faces, :2]
    eye = np.eye(3)
    extent = half[:2]
    weights = [np.broadcast_to(eye, (len(faces), 3, 3))]
    valid = [np.all(np.abs(xy) <= extent, axis=2)]

    # Extremes of a clipped linear triangle occur at original vertices, box
    # corners, or triangle/rectangle edge intersections. Vertex-only projection
    # misses the latter two when the contents nearly fill the sheet.
    corners = np.array([[x, y] for x in (-1, 1) for y in (-1, 1)]) * extent
    u, v = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    denom = np.where(abs(det) > 1e-20, det, 1.)
    delta = corners[None] - xy[:, None, 0]
    b = (delta[:, :, 0] * v[:, None, 1] - delta[:, :, 1] * v[:, None, 0]) / denom[:, None]
    c = (u[:, None, 0] * delta[:, :, 1] - u[:, None, 1] * delta[:, :, 0]) / denom[:, None]
    bary = np.stack((1 - b - c, b, c), axis=2)
    weights.append(bary)
    valid.append(np.all(bary >= -1e-12, axis=2) & (abs(det[:, None]) > 1e-20))
    for axis in (0, 1):
        first = xy
        step = np.roll(xy, -1, axis=1) - first
        divisor = np.where(abs(step[:, :, axis]) > 1e-20, step[:, :, axis], 1.)
        for side in (-1, 1):
            t = (side * extent[axis] - first[:, :, axis]) / divisor
            other = first[:, :, 1-axis] + t * step[:, :, 1-axis]
            weights.append((1-t[:, :, None]) * eye + t[:, :, None] * np.roll(eye, -1, axis=0))
            valid.append((t >= 0) & (t <= 1) & (abs(other) <= extent[1-axis])
                         & (abs(step[:, :, axis]) > 1e-20))
    weights = np.maximum(np.concatenate(weights, axis=1), 0.)
    valid = np.concatenate(valid, axis=1)
    free_weights = weights * ~np.isin(faces, shared)[:, None, :]
    norm = np.sum(free_weights**2, axis=2)
    height = np.einsum('tki,ti->tk', weights, sign * local[faces, 2])
    deficit = np.where(valid & (norm > 1e-20), np.maximum(half[2] + clearance - height, 0.), 0.)
    selected = np.argmax(deficit / np.sqrt(np.maximum(norm, 1e-20)), axis=1)
    row = np.arange(len(faces))
    correction = deficit[row, selected, None] * free_weights[row, selected] / np.maximum(norm[row, selected, None], 1e-20)
    dz = np.zeros(len(local))
    np.maximum.at(dz, faces.ravel(), correction.ravel())
    local[:, 2] += sign * dz


def wrap_generic(rest, start, faces, panel_ids, front_ids, back_ids, item, radius, iterations=1600, wall_gap=None,
                 box_face_clearance=False):
    """Open any front/back bag around one content, as initialization.py does for the bubble mailer.

    Front-group vertices rise and back-group vertices drop by a plateau profile over the content,
    then alternating edge-length, content-distance and wall-gap projections relax the shape. The
    wall gap keeps every front/back pair (except shared seam vertices) at least `wall_gap` apart,
    so no wall pair starts inside the VBD self-contact barrier (half the interaction distance);
    near PE corners the plain projections left edges 0.018 mm apart. Edge targets come from the
    unloaded reference, which stays unchanged.
    """
    if __package__:
        from .legacy.initialization import edge_colors
        from .legacy.contents import signed_distance
    else:
        from initialization import edge_colors
        from contents import signed_distance

    if box_face_clearance and item["shape"] != "box":
        raise ValueError("box-face initialization requires a box payload")

    q = start.copy()
    front = np.unique(faces[np.isin(panel_ids, front_ids)])
    shared = np.intersect1d(front, np.unique(faces[np.isin(panel_ids, back_ids)]))
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
    seam_clearance = False
    if box_face_clearance and len(shared):
        initial = (q - item["position"]) @ item["rotation"]
        triangles = initial[faces, :2]
        overlap = np.all(triangles.max(axis=1) >= -item["half_size"][:2] - radius, axis=1)
        overlap &= np.all(triangles.min(axis=1) <= item["half_size"][:2] + radius, axis=1)
        sheets = np.isin(panel_ids, [*front_ids, *back_ids])
        seam_clearance = bool(np.any(overlap & sheets & np.any(np.isin(faces, shared), axis=1)))
        if seam_clearance:
            seam_reference = (rest[shared] - item["position"]) @ item["rotation"]
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
        if box_face_clearance:
            # A triangle can cut a box corner while all three vertices stay
            # outside. Keep every overlapping top/bottom triangle on its
            # exterior side, then let edge projection redistribute the slack.
            local = (q - item["position"]) @ item["rotation"]
            half = item["half_size"]
            if seam_clearance:
                # Initialize the sealed rim on the box's equatorial side wall.
                # Otherwise point-only box clearance permits the entire rim to
                # drift above/below the box and cut its corners.
                local[shared, :2] = seam_reference[:, :2]
                local[shared, 2] = 0.
            triangles = local[faces]
            overlap = np.all(triangles[:, :, :2].max(axis=1) >= -half[:2] - radius, axis=1)
            overlap &= np.all(triangles[:, :, :2].min(axis=1) <= half[:2] + radius, axis=1)
            for panels, sign in ((front_ids, 1), (back_ids, -1)):
                selected = faces[overlap & np.isin(panel_ids, panels)]
                if seam_clearance and len(selected):
                    # Shared seam vertices remain on the side wall; projecting
                    # them to top and bottom would overwrite the same node.
                    _clear_box_sheet(local, selected, shared, half, radius + .0003, sign)
                elif len(selected):
                    ids = np.unique(selected)
                    local[ids, 2] = sign * np.maximum(sign * local[ids, 2], half[2] + radius + .0003)
            q = local @ item["rotation"].T + item["position"]
        open_pairs(q, pairs, wall_gap)
        if not seam_clearance:
            # Box initialization is translated above the table only after both
            # sheets have cleared the solid; an early floor blocks the bottom.
            q[:, 2] = np.maximum(q[:, 2], floor)
    # Rest the lowest node just above the table, and move the content with it.
    shift = radius + .00015 - q[:, 2].min()
    q[:, 2] += shift
    item["position"][2] += shift
    strain = np.linalg.norm(q[edges[:, 1]] - q[edges[:, 0]], axis=1) / lengths - 1
    gaps = q[pairs[:, 0], 2] - q[pairs[:, 1], 2] if len(pairs) else np.array([np.inf])
    return q, dict(maximum_edge_strain=float(np.abs(strain).max()), minimum_wall_gap_m=float(gaps.min()),
                   method="generic front/back plateau + projections + wall gap",
                   box_face_clearance=box_face_clearance, box_seam_clearance=seam_clearance)


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
