"""Collision-free loading preparation; never changes material reference data."""
import numpy as np
from contents import signed_distance


def edge_colors(edges):
    used, groups = {}, []
    for a, b in edges:
        occupied = used.get(int(a), set()) | used.get(int(b), set())
        color = next(c for c in range(len(groups) + 1) if c not in occupied)
        if color == len(groups):
            groups.append([])
        groups[color].append((a, b))
        used.setdefault(int(a), set()).add(color)
        used.setdefault(int(b), set()).add(color)
    return [np.asarray(group, int) for group in groups]


def prepare_loaded_positions(rest, faces, grid, params, items, support_z, origin_xy):
    if not items:
        return rest.copy(), dict(iterations=0, maximum_edge_strain=0.)
    width, height = params['width_mm'] * .001, params['height_mm'] * .001
    radius, gap = params['radius_mm'] * .001, params['rest_gap_mm'] * .001
    xy = np.asarray(origin_xy)
    extents = np.array([np.abs(item['rotation'][:2, :2]) @ item['half_size'][:2]
                        + np.abs(item['xy_m']) for item in items]).max(axis=0)
    outer = np.array([width, height]) / 2
    if np.any(extents + .008 >= outer):
        raise ValueError('Contents do not fit inside this bag with a usable seam allowance')
    max_height = max(2 * item['half_size'][2] for item in items)
    lift = max(0., max_height + 2 * radius + .0015 - gap)
    plateau = extents + .003
    u = np.clip((np.abs(rest[:, :2] - xy) - plateau) / (outer - plateau), 0, 1)
    profile = np.prod(1 - u * u * (3 - 2 * u), axis=1)
    n = len(rest) // 2
    q = rest.copy()
    q[:, 2] += lift / 2
    q[:n, 2] += .5 * lift * profile[:n]
    q[n:, 2] -= .5 * lift * profile[n:]
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                               faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    groups = edge_colors(edges)
    group_lengths = [np.linalg.norm(rest[g[:, 1]] - rest[g[:, 0]], axis=1) for g in groups]
    # Alternating length and nonpenetration projections prepare a warm start.
    # MuJoCo subsequently computes forces against the unchanged empty-bag mesh.
    for iteration in range(1600):
        for group, target in zip(groups, group_lengths):
            a, b = group.T
            delta = q[b] - q[a]
            length = np.linalg.norm(delta, axis=1)
            correction = .45 * ((length - target) / np.maximum(length, 1e-12))[:, None] * delta
            q[a] += correction
            q[b] -= correction
        for item in items:
            distance, normal = signed_distance(q, item)
            q += np.maximum(radius + .0003 - distance, 0)[:, None] * normal
        q[:, 2] = np.maximum(q[:, 2], support_z + radius + .0001)
    strain = np.linalg.norm(q[edges[:, 1]] - q[edges[:, 0]], axis=1) / lengths - 1
    return q, dict(iterations=iteration + 1, maximum_edge_strain=float(np.abs(strain).max()),
                   mean_edge_strain=float(np.abs(strain).mean()),
                   method='distance/contact projections; empty material reference unchanged')
