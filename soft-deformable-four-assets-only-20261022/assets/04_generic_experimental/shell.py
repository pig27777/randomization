"""Pocket topology and independently checked dihedral derivatives; SI units."""
from collections import Counter
import math
import numpy as np

def make_mesh(nx=12, ny=16, width=0.18, height=0.24, gap=0.004, seam_width=0.015):
    vertices, grid, faces, regions = ([], {}, [], [])
    for side in (0, 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                key = (side, i, j)
                if side and (j == 0 or i in (0, nx)):
                    grid[key] = grid[0, i, j]
                    continue
                x = width * (i / nx - 0.5)
                y = (1 if side == 0 else -1) * gap * 0.5 * math.sin(math.pi * i / nx) * j / ny
                grid[key] = len(vertices)
                vertices.append((x, y, 0.065 + height * j / ny))
        for j in range(ny):
            for i in range(nx):
                a, b, c, d = (grid[side, u, v] for u, v in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)))
                diagonal_ad = (i + j) % 2 == 0
                if j == 0 and i == 0:
                    diagonal_ad = True
                if j == 0 and i == nx - 1:
                    diagonal_ad = False
                triangles = [(a, b, d), (a, d, c)] if diagonal_ad else [(a, b, c), (b, d, c)]
                for tri in triangles:
                    faces.append(tri[::-1] if side == 0 else tri)
                    xmid = width * ((i + 0.5) / nx - 0.5)
                    regions.append(int(abs(xmid) >= width * 0.5 - seam_width - 1e-09))
    return (np.array(vertices, np.float32), np.array(faces, np.int32), grid, np.array(regions, np.int32))

def topology_report(vertices, faces):
    counts = Counter((tuple(sorted((int(a), int(b)))) for f in faces for a, b in zip(f, (f[1], f[2], f[0]))))
    area = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) * 0.5
    return dict(vertices=len(vertices), triangles=len(faces), edges=len(counts), boundary_edges=sum((v == 1 for v in counts.values())), nonmanifold_edges=sum((v > 2 for v in counts.values())), duplicate_faces=len(faces) - len({tuple(sorted(f)) for f in faces}), degenerate_faces=int(np.sum(area < 1e-12)), euler_characteristic=len(vertices) - len(counts) + len(faces))

def smoothstep(t, start, end):
    u = min(1.0, max(0.0, (t - start) / (end - start)))
    return (u * u * (3 - 2 * u), 6 * u * (1 - u) / (end - start))

def compute_bag_hinge_batch(q: np.ndarray, hinge_ids: np.ndarray, gradient=True):
    """
    向量化计算所有四顶点铰链的二面角 angle 与关于 4 个顶点的解析梯度 grad
    完全对应 elastic-physics.js 中的 function bagHinge(q, ids)
    """
    a = q[hinge_ids[:, 0]]
    b = q[hinge_ids[:, 1]]
    c = q[hinge_ids[:, 2]]
    d = q[hinge_ids[:, 3]]
    e = b - a
    len2 = np.sum(e * e, axis=1, keepdims=True)
    elen = np.sqrt(np.maximum(len2, 1e-18))
    n1 = np.cross(e, c - a)
    n2 = np.cross(d - a, e)
    nn1 = np.sum(n1 * n1, axis=1, keepdims=True)
    nn2 = np.sum(n2 * n2, axis=1, keepdims=True)
    valid = (elen[:, 0] > 1e-09) & (nn1[:, 0] > 1e-15) & (nn2[:, 0] > 1e-15)
    nn1_safe = np.where(nn1 > 1e-15, nn1, 1.0)
    nn2_safe = np.where(nn2 > 1e-15, nn2, 1.0)
    len2_safe = np.where(len2 > 1e-18, len2, 1.0)
    u = n1 / np.sqrt(nn1_safe)
    v = n2 / np.sqrt(nn2_safe)
    e_hat = e / elen
    sin_term = np.sum(e_hat * np.cross(u, v), axis=1)
    cos_term = np.sum(u * v, axis=1)
    angle = np.arctan2(sin_term, cos_term)
    if not gradient:return angle,None
    gc = n1 * (-elen / nn1_safe)
    gd = n2 * (-elen / nn2_safe)
    coef_bc = -np.sum((b - c) * e, axis=1, keepdims=True) / len2_safe
    coef_bd = -np.sum((b - d) * e, axis=1, keepdims=True) / len2_safe
    ga = gc * coef_bc + gd * coef_bd
    coef_ca = -np.sum((c - a) * e, axis=1, keepdims=True) / len2_safe
    coef_da = -np.sum((d - a) * e, axis=1, keepdims=True) / len2_safe
    gb = gc * coef_ca + gd * coef_da
    grad = np.stack([ga, gb, gc, gd], axis=1)
    grad[~valid] = 0.0
    return (angle, grad)
