"""One layered tetrahedral model for every soft item: shape generator plus per-tetrahedron layers.

Items differ only in shape parameters and layer materials:
  skin  - thin, stiffer: stands in for fabric
  core  - soft: stands in for filling
  gap   - very soft: stands in for air between folded layers
All values are authored, not measured. SI units; Newton 1.6 VBD.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton

LAYERS = {
    'skin': (3.0e4, 1.0e4, 1.0),
    'core': (1.5e3, 1.0e3, 1.0),
    'gap': (3.0e2, 1.0e2, 0.5),
}


@dataclass(frozen=True)
class SoftShape:
    kind: str  # slab | blob | pillow
    size: tuple[float, float, float]  # x, y, thickness (blob: diameters)
    cells: tuple[int, int, int]
    mass: float
    layers: tuple[str, ...] = ()  # slab only, bottom to top
    corner_taper: float = 0.0  # pillow: thickness at the corners as a fraction of the center
    roughness: float = 0.0  # blob: radial bumps
    air_pockets: float = 0.0  # fraction of interior cells set to 'gap'
    stiffness_sigma: float = 0.0  # log-normal variation of every tetrahedron
    materials: dict = field(default_factory=lambda: dict(LAYERS))
    seed: int = 0


PRESETS = {
    'folded_clothing': SoftShape('slab', (0.30, 0.25, 0.05), (12, 10, 5), 0.3, layers=('skin', 'gap', 'skin', 'gap', 'skin'),
                                 stiffness_sigma=0.2),
    'crumpled_clothing': SoftShape('blob', (0.26, 0.20, 0.12), (11, 9, 7), 0.3, roughness=0.15, air_pockets=0.3,
                                   stiffness_sigma=0.6),
    'pillow': SoftShape('pillow', (0.50, 0.35, 0.12), (14, 10, 5), 0.8, corner_taper=0.15, stiffness_sigma=0.2),
}


def _tets(n):
    """Five tetrahedra per hexahedral cell with alternating parity; returns (tets, cell id per tet)."""
    nx, ny, nz = n
    idx = lambda a, b, c: (a * ny + b) * nz + c
    tets, cell_of = [], []
    for a in range(nx - 1):
        for b in range(ny - 1):
            for c in range(nz - 1):
                v = [idx(a + i, b + j, c + k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
                pattern = ([(0, 1, 2, 4), (1, 3, 2, 7), (1, 4, 5, 7), (2, 7, 6, 4), (1, 2, 4, 7)] if (a + b + c) % 2 == 0
                           else [(0, 1, 3, 5), (0, 3, 2, 6), (0, 5, 4, 6), (3, 5, 6, 7), (0, 3, 5, 6)])
                for t in pattern:
                    tets.append([v[i] for i in t])
                    cell_of.append((a, b, c))
    return np.array(tets), np.array(cell_of)


def generate(shape: SoftShape):
    """Return node positions, tetrahedra and per-tet (k_mu, k_lambda, k_damp)."""
    rng = np.random.default_rng(shape.seed)
    nx, ny, nz = [c + 1 for c in shape.cells]
    u = np.linspace(-1, 1, nx)[:, None, None] * np.ones((nx, ny, nz))
    v = np.linspace(-1, 1, ny)[None, :, None] * np.ones((nx, ny, nz))
    w = np.linspace(0, 1, nz)[None, None, :] * np.ones((nx, ny, nz))
    sx, sy, st = shape.size
    if shape.kind == 'slab':
        pos = np.stack([u * sx / 2, v * sy / 2, w * st], -1)
    elif shape.kind == 'pillow':
        profile = np.sqrt(np.clip((1 - np.abs(u) ** 4) * (1 - np.abs(v) ** 4), 0, 1))
        thick = st * (shape.corner_taper + (1 - shape.corner_taper) * profile)
        base = (st - thick) / 2
        pos = np.stack([u * sx / 2, v * sy / 2, base + w * thick], -1)
    elif shape.kind == 'blob':
        ww = 2 * w - 1
        c = np.stack([u, v, ww], -1)
        # Rounded box (superquadric, exponent 4): flattens far fewer corner cells than a cube-to-sphere map.
        norm_inf = np.abs(c).max(-1, keepdims=True)
        norm_4 = (np.abs(c) ** 4).sum(-1, keepdims=True) ** 0.25
        s = c * np.where(norm_4 > 0, norm_inf / np.maximum(norm_4, 1e-9), 0)
        h = rng.normal(size=(6, 4))
        bump = 1 + shape.roughness * sum(np.sin((k + 1) * (s @ hk[:3]) + hk[3]) for k, hk in enumerate(h)) / np.sqrt(6)
        pos = s * np.array([sx, sy, st]) / 2 * bump[..., None]
    else:
        raise ValueError(shape.kind)
    pos = pos.reshape(-1, 3)
    pos[:, 2] -= pos[:, 2].min()
    tets, cells = _tets((nx, ny, nz))
    # positive orientation
    p = pos[tets]
    vol = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]

    labels = np.empty(len(tets), dtype=object)
    cx, cy, cz = shape.cells
    for t, (a, b, c) in enumerate(cells):
        if shape.kind == 'slab':
            labels[t] = shape.layers[c]
        else:
            boundary = a in (0, cx - 1) or b in (0, cy - 1) or c in (0, cz - 1)
            labels[t] = 'skin' if (boundary and shape.kind == 'pillow') else 'core'
    if shape.air_pockets > 0:
        interior_cells = sorted({tuple(c) for c, lab in zip(cells, labels) if lab == 'core'})
        pick = rng.random(len(interior_cells)) < shape.air_pockets
        pockets = {c for c, chosen in zip(interior_cells, pick) if chosen}
        for t, c in enumerate(cells):
            if tuple(c) in pockets:
                labels[t] = 'gap'
    mats = np.array([shape.materials[lab] for lab in labels], dtype=float)
    if shape.stiffness_sigma > 0:
        scale = np.exp(shape.stiffness_sigma * rng.normal(size=len(tets)))
        mats[:, :2] *= scale[:, None]
    volume = float(np.abs(vol).sum() / 6)
    return pos.astype(np.float32), tets, mats, dict(labels=labels, volume=volume)


def add_item(builder, shape: SoftShape, base_z=0.002, xy=(0.0, 0.0)):
    pos, tets, mats, meta = generate(shape)
    start = builder.particle_count
    tet_start = len(builder.tet_materials)
    builder.add_soft_mesh(pos=wp.vec3(xy[0], xy[1], base_z), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0),
                          vertices=pos.tolist(), indices=tets.ravel().tolist(), density=shape.mass / meta['volume'],
                          k_mu=1.0, k_lambda=1.0, k_damp=1.0, particle_radius=0.004)
    n_added = len(builder.tet_materials) - tet_start
    if n_added != len(tets):
        raise RuntimeError(f'expected {len(tets)} tetrahedra, builder added {n_added}')
    for k, row in enumerate(mats):
        builder.tet_materials[tet_start + k] = tuple(float(x) for x in row)
    return dict(start=start, end=builder.particle_count, cloth=False, faces=None, positions=pos + [xy[0], xy[1], base_z],
                labels=meta['labels'], n_tets=len(tets))
