"""Local top blocks on the v1 coarse material frame.

The coarse connections keep the calibrated macroscopic law. Top volume and mass
are removed from each coarse top body and assigned to fine blocks. Three native
slide joints per block provide implicit damping and tensile/shear stiffness;
only the additional compressive stiffness is applied explicitly. Block rotation
follows the coarse material frame, so no additional rotational weld is used.
"""
from dataclasses import dataclass
import itertools
import math

import numpy as np

from model import Segment, mass_layout

PITCH = .020
MODULUS_RATIO = 100.0     # resolve local compliance without replacing v1's calibrated coarse law
TENSION_RATIO = .5
SHEAR_RATIO = .5
COMPRESSION_FRACTION = .9
EXTENSION = .1


@dataclass
class SurfaceLink:
    parent: int
    child: int
    compression: float
    tension: float
    shear: float
    damping: tuple
    travel: tuple


def refine(desc):
    """Partition the same cuboid density into a coarse lower slab and a fine top slab.

    Original coarse body frames and connection points remain in place. Their
    collision boxes and mass centres move within those frames. Integrating the
    old piecewise density over each new box preserves all mass moments, including
    nonuniform sampled COMs, without zero-mass coarse frames for thin objects.
    """
    k = desc.kappa
    L, n = np.array(k.size), np.array(k.segments)
    thickness = min(.020, L[2] / 3)
    initial = list(desc.segments)
    W, shift = mass_layout(k)
    total_mass = sum(s.mass for s in initial)
    weights = [W.sum(axis=tuple(i for i in range(3) if i != a)) for a in range(3)]
    edges = [np.linspace(-L[a] / 2, L[a] / 2, n[a] + 1) for a in range(3)]

    def partition(lo, hi, frame):
        first, variance, fraction = [], [], 1.
        for a in range(3):
            left, right = np.maximum(lo[a], edges[a][:-1]), np.minimum(hi[a], edges[a][1:])
            right = np.maximum(right, left)
            density = weights[a] / (L[a] / n[a])
            m0 = np.sum(density * (right - left))
            m1 = np.sum(density * (right ** 2 - left ** 2) / 2)
            m2 = np.sum(density * (right ** 3 - left ** 3) / 3)
            first.append(m1 / m0)
            variance.append(m2 / m0 - (m1 / m0) ** 2)
            fraction *= m0
        mass = total_mass * fraction
        variance = np.array(variance) - shift ** 2
        inertia = mass * (variance.sum() - variance)
        if min(inertia) <= 0:
            raise ValueError("intra-cell COM shift cannot be represented by the refined surface inertia")
        return mass, np.array(first) + shift - frame, inertia

    for coarse in initial:
        lo, hi = coarse.pos - coarse.half, coarse.pos + coarse.half
        scale = (L[2] - thickness) / L[2]
        lo[2], hi[2] = -L[2] / 2 + (lo[2] + L[2] / 2) * scale, -L[2] / 2 + (hi[2] + L[2] / 2) * scale
        coarse.half = (hi - lo) / 2
        coarse.geom_offset = (hi + lo) / 2 - coarse.pos
        coarse.mass, coarse.com, coarse.inertia = partition(lo, hi, coarse.pos)

    count = np.ceil(L[:2] / PITCH).astype(int)
    tile_xy = L[:2] / count
    tile_half = np.r_[tile_xy / 2, thickness / 2]
    index = {s.idx: i for i, s in enumerate(initial)}
    for i, j in itertools.product(range(count[0]), range(count[1])):
        pos = np.r_[-L[:2] / 2 + (np.array([i, j]) + .5) * tile_xy, L[2] / 2 - thickness / 2]
        parent_ij = np.minimum(((pos[:2] + L[:2] / 2) / (L[:2] / n[:2])).astype(int), n[:2] - 1)
        parent = index[(*parent_ij, n[2] - 1)]
        mass, com, inertia = partition(pos - tile_half, pos + tile_half, pos)
        child = len(desc.segments)
        desc.segments.append(Segment((i, j, n[2]), pos, tile_half.copy(), mass, com, inertia))
        kn = MODULUS_RATIO * desc.internal["E_z"] * np.prod(tile_xy) / thickness
        kt, ks = TENSION_RATIO * kn, SHEAR_RATIO * kn
        damping = tuple(2 * k.damping_ratio * math.sqrt(value * mass) for value in (ks, ks, kn))
        desc.surface_links.append(SurfaceLink(parent, child, kn, kt, ks, damping,
                                              (*(.5 * tile_xy), COMPRESSION_FRACTION * thickness)))
    # Face/edge/vertex neighbours can overlap within their compression travel, as in v1.
    # More distant pieces retain self-collision. This includes fine/coarse neighbours.
    n_coarse = len(initial)
    for a, sa in enumerate(desc.segments):
        for b in range(max(a + 1, n_coarse), len(desc.segments)):
            sb = desc.segments[b]
            pa, pb = sa.pos + sa.geom_offset, sb.pos + sb.geom_offset
            if np.all(np.abs(pb - pa) <= sa.half + sb.half + 1e-9):
                desc.exclude.append((a, b))


class SurfaceState:
    def __init__(self, model, desc):
        self.links = desc.surface_links
        self.q = np.array([[model.joint(f"surface_{s.child}_{a}").qposadr[0] for a in "xyz"] for s in self.links], int)
        self.v = np.array([[model.joint(f"surface_{s.child}_{a}").dofadr[0] for a in "xyz"] for s in self.links], int)
        self.extra = np.array([s.compression - s.tension for s in self.links])
        self.travel = np.array([s.travel for s in self.links])

    def compression_force(self, data):
        return self.extra * np.maximum(-data.qpos[self.q[:, 2]], 0)

    def check(self, data):
        q = data.qpos[self.q]
        ratios = np.c_[np.abs(q[:, :2]) / self.travel[:, :2], -q[:, 2] / self.travel[:, 2], q[:, 2] / EXTENSION]
        where = np.unravel_index(np.argmax(ratios), ratios.shape)
        if ratios[where] > 1.2:
            return f"surface link {where[0]} dof {where[1]} {ratios[where]:.6f} x limit"
        return None
