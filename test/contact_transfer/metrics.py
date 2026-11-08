"""Independent, float64 measurements for the contact benchmark (SI units)."""
from __future__ import annotations

import numpy as np


def tet_response(q, tets, inverse_rest, materials):
    """Stable Neo-Hookean energy and elastic nodal forces used by installed VBD.

    Implemented independently of the GPU solve. mu/lambda are Lame parameters;
    the stable-NH volumetric coefficient is lambda + mu, not lambda alone.
    """
    p = np.asarray(q, dtype=float)[tets]
    ds = np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=2)
    f = ds @ inverse_rest
    j = np.linalg.det(f)
    volume = 1.0 / (6.0 * np.linalg.det(inverse_rest))
    mu, lam = materials[:, 0], materials[:, 1]
    bulk = lam + mu
    alpha = 1.0 + mu / bulk
    cof = np.stack([np.cross(f[:, :, 1], f[:, :, 2]),
                    np.cross(f[:, :, 2], f[:, :, 0]),
                    np.cross(f[:, :, 0], f[:, :, 1])], axis=2)
    stress = mu[:, None, None] * f + (bulk * (j - alpha))[:, None, None] * cof
    energy = np.sum(volume * (0.5 * mu * np.sum(f * f, axis=(1, 2)) + 0.5 * bulk * (j - alpha) ** 2))
    corner_force = -volume[:, None, None] * (stress @ inverse_rest.transpose(0, 2, 1)).transpose(0, 2, 1)
    force = np.zeros_like(q, dtype=float)
    for k in range(3):
        np.add.at(force, tets[:, k + 1], corner_force[:, k])
    np.add.at(force, tets[:, 0], -corner_force.sum(axis=1))
    return float(energy), force, j


def _clip(poly, axis, bound, sign):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
        if da >= -1e-14:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + da / (da - db) * (b - a))
    return np.array(out).reshape(-1, 2)


def pressure_weights(q, faces, half, xy=(0.0, 0.0)):
    """Integrate linear triangle shape functions over a rectangular footprint.

    This avoids picking a single nearest node for an off-grid small finger.
    Returns normalized nodal weights; uniform pressure is only a diagnostic
    reference, not assumed to equal the pressure under a rigid punch.
    """
    weights = np.zeros(len(q))
    for face in faces:
        tri = q[face, :2]
        poly = tri.copy()
        for axis in (0, 1):
            for bound, sign in ((xy[axis] - half[axis], 1), (xy[axis] + half[axis], -1)):
                if len(poly):
                    poly = _clip(poly, axis, bound, sign)
        if len(poly) < 3:
            continue
        system = np.vstack([tri.T, np.ones(3)])
        for k in range(1, len(poly) - 1):
            piece = poly[[0, k, k + 1]]
            ab, ac = piece[1] - piece[0], piece[2] - piece[0]
            area = abs(ab[0] * ac[1] - ab[1] * ac[0]) / 2
            bary = np.linalg.solve(system, np.r_[piece.mean(axis=0), 1.0])
            weights[face] += area * bary
    if weights.sum() <= 0:
        raise ValueError("load footprint does not intersect the top surface")
    return weights / weights.sum()


def surface_height(q, faces, xy):
    """Vertical ray intersection with deformed top triangles (NaN outside)."""
    height = np.full(len(xy), np.nan)
    for face in faces:
        tri = q[face]
        ab, ac = tri[1, :2] - tri[0, :2], tri[2, :2] - tri[0, :2]
        det = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(det) < 1e-14:
            continue
        d = xy - tri[0, :2]
        u = (d[:, 0] * ac[1] - d[:, 1] * ac[0]) / det
        v = (ab[0] * d[:, 1] - ab[1] * d[:, 0]) / det
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        z = tri[0, 2] + u * (tri[1, 2] - tri[0, 2]) + v * (tri[2, 2] - tri[0, 2])
        height[inside] = np.fmax(height[inside], z[inside])
    return height


def transform_points(p, pose):
    xyz, quat = pose[:3], pose[3:]
    v, w = quat[:3], quat[3]
    return p + 2 * np.cross(v, np.cross(v, p) + w * p) + xyz
