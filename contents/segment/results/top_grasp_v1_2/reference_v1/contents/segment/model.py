"""Segment-rigid content model, engine-agnostic description layer.

kappa_content (whole-object responses, see contents/package_params.md) -> internal
moduli (E_y, E_z, D) via analytic estimate x calibration table -> a description of
rigid segments, 6-DOF elastic joints, collision shapes and the optional mobile mass.
Engine adapters (mj_adapter.py, newton_adapter.py) only translate this description.

Object frame: origin at the envelope center, x = length, y = width, z = thickness.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

G = 9.81
PAD = 0.040                      # finger pad side (test/content_model/README.md)
SEG_CHOICES = (1, 2, 3, 4, 6)
SEG_MAX = (6, 4, 3)

# Fixed ratios of the first version (segment_model.md, "关节力学"); not part of kappa.
TENSION_RATIO = 0.5              # axial tension stiffness / compression stiffness
SHEAR_RATIO = 0.5                # shear modulus / compression modulus of the same joint
TWIST_RATIO = 0.5                # twist stiffness / mean bending stiffness
INPLANE_BEND = 1.0 / 24.0
VSHEAR_BEND = 48.0               # vertical shear of in-plane joints >= this * k_bend / d^2        # in-plane bending = k_n * c^2 * this (half a solid section)
BEND_LOCK_RAD = 1.0              # per-joint bending limit
JOINT_LOCK_MAX = 0.45            # joint compression strain cap: adjacent overlap at the lock stays < 50 %
MOBILE_DAMPING_RATIO = 0.2

TABLE_PATH = Path(__file__).resolve().parent / "results" / "freeze_v1_newton" / "mapping_pass1" / "table.json"


@dataclass
class Kappa:
    category: str | None = None              # sampling label; not an independent physical parameter
    size: tuple = (0.30, 0.22, 0.04)
    mass: float = 0.4
    com_offset: tuple = (0.0, 0.0, 0.0)       # fraction of the envelope
    segments: tuple = (6, 4, 2)
    rigid: bool = False
    squeeze_stiffness_y: float = 0.5          # N/mm
    squeeze_stiffness_z: float = 0.3          # N/mm
    lock_strain_y: float = 0.25
    lock_strain_z: float = 0.20
    droop_ratio: float = 0.45
    damping_ratio: float = 0.5
    mobile: bool = False
    mobile_frac: float = 0.3
    mobile_range: float = 0.02                # m, < half the shortest side
    mobile_freq: float = 2.0                  # Hz
    surface_friction: float = 0.8             # package surface vs gripper pads (equivalent skin)
    skin_compliance: float = 0.0              # m/N per contact point, in series with the contact stiffness

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        for k in ("size", "com_offset", "segments"):
            d[k] = tuple(d[k])
        return Kappa(**d)


ANCHORS = {
    # book / box: joints locked
    "hard": Kappa(category="hard", size=(0.24, 0.17, 0.05), mass=0.8, segments=(4, 3, 2), rigid=True,
                  squeeze_stiffness_y=50.0, squeeze_stiffness_z=50.0, lock_strain_y=0.02,
                  lock_strain_z=0.02, droop_ratio=0.0),
    # folded towel / clothes. A thinner, wider stack (0.30 x 0.22 x 0.04) buckles in R1-y at ~8 % strain.
    # Length 0.24 keeps a 0.32 droop clear of the table after R2's 100 mm lift.
    "bendable": Kappa(category="bendable", size=(0.24, 0.16, 0.05), mass=0.29, segments=(6, 3, 2),
                      squeeze_stiffness_y=0.5, squeeze_stiffness_z=0.3, lock_strain_y=0.20,
                      lock_strain_z=0.20, droop_ratio=0.32),
    # pillow / crumpled clothes
    "compressible": Kappa(category="compressible", size=(0.35, 0.25, 0.12), mass=0.5, segments=(4, 4, 3),
                          # squeeze_z 0.08 N/mm collapsed under self weight to its lock (120 -> 80 mm)
                          squeeze_stiffness_y=0.1, squeeze_stiffness_z=0.2, lock_strain_y=0.25,
                          lock_strain_z=0.25, droop_ratio=0.15),
}

# T5 segment counts per anchor
T5_SEGMENTS = {
    "hard": [(2, 2, 1), (4, 3, 2), (6, 4, 3)],
    # ny = 2 puts a joint under the pad center with only 20 mm of each row gripped: the rows twist and droop
    # stops depending on D; nz = 2 caps the compressible lock_z at 0.2, below the 15 % secant end
    "bendable": [(4, 3, 2), (6, 3, 2), (6, 4, 3)],
    "compressible": [(3, 3, 3), (4, 4, 3), (6, 4, 3)],
}


@dataclass
class Segment:
    idx: tuple
    pos: np.ndarray          # center in object frame
    half: np.ndarray
    mass: float
    com: np.ndarray          # COM offset from the center (segment frame = object frame)
    inertia: np.ndarray      # principal inertia about the COM, axes = object axes


@dataclass
class Joint:
    a: int
    b: int
    axis: int                # 0/1/2: direction from a to b
    rest: float              # center distance
    k_comp: float            # base N/m along the axis; compaction energy and stops add declared nonlinear forces
    k_tens: float
    k_shear: tuple           # N/m along the two other axes (ascending order)
    k_bend: tuple            # uncompressed N*m/rad about the two other axes (ascending order)
    k_twist: float           # uncompressed N*m/rad; CPU bushing uses the same compaction gain as bending
    m_eff: float             # reduced mass of the pair, for damping
    i_eff: tuple             # reduced inertia about the two other axes and the joint axis
    damping_ratio: float
    lock_comp: float         # joint compression strain at lock
    lock_bend: float         # rad
    fixed: bool = False


@dataclass
class Mobile:
    parent: int
    rest: np.ndarray         # rest position relative to the parent's center
    mass: float
    stiffness: float         # total, isotropic
    damping: float
    range: float
    anchors: list = field(default_factory=list)   # segments it hangs from (springs spread over them)


@dataclass
class ContentDesc:
    kappa: Kappa
    segments: list
    joints: list
    exclude: list            # segment pairs that must not collide
    mobile: Mobile | None
    rigid: bool
    internal: dict = field(default_factory=dict)

    @property
    def cell(self):
        return np.array(self.kappa.size) / np.array(self.kappa.segments)


# ---------------------------------------------------------------- mass layout

def _axis_weights(n, c, max_slope=0.8):
    """Linear mass gradient along one axis giving COM fraction c; residual goes to intra-cell shift."""
    u = (np.arange(n) + 0.5) / n - 0.5
    s2 = float(np.sum(u * u))
    if n == 1 or s2 == 0:
        return np.ones(n), c
    lam = c * n / s2
    lam = float(np.clip(lam, -max_slope / np.max(np.abs(u)), max_slope / np.max(np.abs(u))))
    w = 1 + lam * u
    achieved = float(np.sum(w * u) / np.sum(w))
    return w, c - achieved


def mass_layout(k: Kappa):
    """Per-segment mass and intra-segment COM shift reproducing total mass and com_offset."""
    n = np.array(k.segments)
    L = np.array(k.size)
    cell = L / n
    weights, resid = [], []
    for ax in range(3):
        w, r = _axis_weights(n[ax], k.com_offset[ax])
        weights.append(w)
        # residual fraction of the envelope -> shift inside each cell, capped at 40 % of the half cell
        resid.append(float(np.clip(r * L[ax], -0.4 * cell[ax] / 2, 0.4 * cell[ax] / 2)))
    W = np.einsum("i,j,k->ijk", *weights)
    W = W / W.sum()
    return W, np.array(resid)


# ---------------------------------------------------------------- analytic estimates

def _cover(n, L, lo, hi):
    """Total size of the cells of a uniform split of [-L/2, L/2] overlapping [lo, hi]."""
    c = L / n
    edges = -L / 2 + c * np.arange(n + 1)
    hit = (edges[1:] > lo + 1e-9) & (edges[:-1] < hi - 1e-9)
    return c * max(int(hit.sum()), 1)


def analytic_internal(k: Kappa):
    """Rough internal moduli from the target responses (segment_model.md: series springs + cantilever)."""
    nx, ny, nz = k.segments
    Lx, Ly, Lz = k.size
    xcov = _cover(nx, Lx, -PAD / 2, PAD / 2)
    ycov = _cover(ny, Ly, -PAD / 2, PAD / 2)
    zcov = _cover(nz, Lz, -PAD / 2, PAD / 2)
    leff_y = Ly * (ny - 1) / ny if ny > 1 else Ly
    leff_z = Lz * (nz - 1) / nz if nz > 1 else Lz
    E_y = k.squeeze_stiffness_y * 1e3 * leff_y / (xcov * zcov)
    E_z = k.squeeze_stiffness_z * 1e3 * leff_z / (xcov * ycov)
    rho = max(k.droop_ratio, 1e-4)
    lf = max(Lx - PAD, 0.5 * Lx)
    w = k.mass * G / Lx
    D = w * lf ** 4 / (8 * rho * Lx) / Ly       # plate bending rigidity per unit width (N*m)
    return {"E_y": E_y, "E_z": E_z, "D": D}


# ---------------------------------------------------------------- calibration table

_TABLE_CACHE = {}


def load_table(path=TABLE_PATH):
    """Calibration table, cached per file and reloaded when the file changes."""
    path = Path(path)
    if not path.exists():
        return {}
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = json.loads(path.read_text())
    return _TABLE_CACHE[key]


def _table_key(k: Kappa):
    return "x".join(str(v) for v in k.segments)


_POINTS_CACHE = {}


def _points(entry):
    """Table points sorted by response, keeping a strictly increasing response axis (cached per entry)."""
    hit = _POINTS_CACHE.get(id(entry))
    if hit is not None and hit[0] is entry:
        return hit[1], hit[2]
    x, y = _points_uncached(entry)
    _POINTS_CACHE[id(entry)] = (entry, x, y)
    return x, y


def _points_uncached(entry):
    pts = sorted(entry, key=lambda p: p[0])
    x = np.log([max(p[0], 1e-9) for p in pts])
    y = np.log([p[1] for p in pts])
    keep = [0]
    for i in range(1, len(x)):
        if x[i] > x[keep[-1]] + 1e-6:
            keep.append(i)
    return x[keep], y[keep]


def _correction(entry, target):
    """Log-interpolated knob correction at the target response; constant outside the scanned range
    (callers check in_table() and must not rely on that extrapolation)."""
    if not entry:
        return 1.0
    x, y = _points(entry)
    if len(x) == 1:
        return float(np.exp(y[0]))
    return float(np.exp(np.interp(math.log(max(target, 1e-9)), x, y)))


def in_table(entry, target, slack=0.02):
    """Whether the target lies inside the scanned response range (log space, small slack)."""
    if not entry:
        return False
    x, _ = _points(entry)
    t = math.log(max(target, 1e-9))
    return bool(x[0] - slack <= t <= x[-1] + slack) and len(x) >= 2


_INDEX_CACHE = {}


def table_entry(k: Kappa, table, geometry=None):
    """Entry for this segment count: the named geometry, else the anchor geometry closest in log size."""
    key = _table_key(k)
    if geometry is not None and f"{geometry}:{key}" in table:
        return table[f"{geometry}:{key}"]
    idx = _INDEX_CACHE.get(id(table))
    if idx is None or idx[0] is not table:
        by_seg = {}
        for name, v in table.items():
            by_seg.setdefault(name.split(":")[-1], []).append((name.split(":")[0], v))
        idx = _INDEX_CACHE[id(table)] = (table, by_seg)
    cands = idx[1].get(key, [])
    if not cands:
        return {}
    def dist(anchor):
        a = ANCHORS.get(anchor)
        return 0.0 if a is None else float(np.sum(np.log(np.array(k.size) / np.array(a.size)) ** 2))
    return min(cands, key=lambda c: dist(c[0]))[1]


def internal_params(k: Kappa, table=None, geometry=None, strict=None):
    """Analytic moduli times the table correction for this segment count."""
    est = analytic_internal(k)
    if table is None:
        table = load_table()
    entry = table_entry(k, table, geometry)
    modern = bool(entry.get("mapping_version")) or (not entry and any(v.get("mapping_version") for v in table.values()))
    covered = {q: in_table(entry.get(q), v, slack=0 if modern else .02)
               for q, v in (("squeeze_y", k.squeeze_stiffness_y),
                            ("squeeze_z", k.squeeze_stiffness_z), ("droop", k.droop_ratio))}
    if strict is None:
        strict = modern
    if strict and not k.rigid and (not entry or not all(covered.values())):
        raise ValueError(f"target outside calibration range: {[q for q, ok in covered.items() if not ok]}")
    out = dict(est)
    for name, q, value in (("E_y", "squeeze_y", k.squeeze_stiffness_y),
                           ("E_z", "squeeze_z", k.squeeze_stiffness_z), ("D", "droop", k.droop_ratio)):
        # New mappings never extrapolate. strict=False is for range diagnostics
        # and calibration; uncovered coordinates retain the explicit analytic estimate.
        if not modern or covered[q]:
            out[name] *= _correction(entry.get(q), value)
    out["table"] = bool(entry)
    out["in_table"] = covered
    return out


# ---------------------------------------------------------------- build

def _box_inertia(m, half):
    a, b, c = 2 * np.asarray(half)
    return m / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def _bend_lower_length(rest, arm, theta):
    """Length of an edge spring on the compressed side when the pair bends by theta about the midpoint."""
    h = theta / 2
    pa = np.array([-rest / 2, -arm])
    pb = np.array([rest / 2, -arm])
    ra = np.array([[math.cos(-h), -math.sin(-h)], [math.sin(-h), math.cos(-h)]])
    rb = ra.T
    return float(np.linalg.norm(rb @ pb - ra @ pa))


def build(k: Kappa, internal: dict | None = None, omega_max: float = math.inf, table=None,
          geometry=None) -> ContentDesc:
    """kappa -> engine-agnostic description. omega_max (rad/s) is the platform's auto-lock limit."""
    n = np.array(k.segments)
    L = np.array(k.size)
    cell = L / n
    half = cell / 2
    if internal is None:
        internal = internal_params(k, table, geometry)
    W, shift = mass_layout(k)
    frac = k.mobile_frac if k.mobile else 0.0
    seg_mass_total = k.mass * (1 - frac)

    segs, index = [], {}
    for i, j, l in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        pos = -L / 2 + cell * (np.array([i, j, l]) + 0.5)
        m = seg_mass_total * W[i, j, l]
        inertia = _box_inertia(m, half) - m * np.array([shift[1] ** 2 + shift[2] ** 2,
                                                        shift[0] ** 2 + shift[2] ** 2,
                                                        shift[0] ** 2 + shift[1] ** 2])
        index[(i, j, l)] = len(segs)
        segs.append(Segment((i, j, l), pos, half.copy(), m, shift.copy(), inertia))

    E_y, E_z, D = internal["E_y"], internal["E_z"], internal["D"]
    lock = {1: k.lock_strain_y, 2: k.lock_strain_z}
    joints = []
    zl = (np.arange(n[2]) + 0.5) * cell[2] - L[2] / 2
    sum_z2 = float(np.sum(zl ** 2))
    for (i, j, l), ia in index.items():
        for ax in range(3):
            nb = [i, j, l]
            nb[ax] += 1
            if nb[ax] >= n[ax]:
                continue
            ib = index[tuple(nb)]
            o = [a for a in range(3) if a != ax]
            area = cell[o[0]] * cell[o[1]]
            d = cell[ax]
            if ax == 0:
                kn_e = E_y * area / d
                K_plane = D * L[1] / d                  # rotational stiffness of one joint plane
                if n[2] > 1:
                    kn = min(kn_e, 0.5 * K_plane / (n[1] * sum_z2))
                    k_by = (K_plane - n[1] * kn * sum_z2) / (n[1] * n[2])
                else:
                    kn = kn_e
                    k_by = K_plane / n[1]
                k_bz = kn * cell[1] ** 2 * INPLANE_BEND
                k_bend = (k_by, k_bz)                   # about y, about z
                # transverse shear of in-plane joints follows the in-plane modulus (tying it to E_z let the
                # ungripped rows slide down and the grip collapse on lifting); the vertical component is at
                # least VSHEAR_BEND * k_b / d^2 so hanging deflection stays bending-dominated and droop
                # remains controllable for soft-squeeze objects
                k_shear = (SHEAR_RATIO * E_y * area / d, max(SHEAR_RATIO * E_y * area / d, VSHEAR_BEND * k_by / d ** 2))
                lk = JOINT_LOCK_MAX                     # x is not squeezed in R1; cap only
            elif ax == 1:
                kn = E_y * area / d
                k_bx = min(D * cell[0] / (d * n[2]), kn * cell[2] ** 2 * INPLANE_BEND)
                k_bz = kn * cell[0] ** 2 * INPLANE_BEND
                k_bend = (k_bx, k_bz)                   # about x, about z
                k_shear = (SHEAR_RATIO * E_y * area / d, max(SHEAR_RATIO * E_y * area / d, VSHEAR_BEND * k_bx / d ** 2))
                lk = lock[1] * n[1] / (n[1] - 1)
            else:
                kn = E_z * area / d
                k_bend = (kn * cell[1] ** 2 * INPLANE_BEND, kn * cell[0] ** 2 * INPLANE_BEND)
                k_shear = (SHEAR_RATIO * E_z * area / d,) * 2
                lk = lock[2] * n[2] / (n[2] - 1)
            ma, mb = segs[ia].mass, segs[ib].mass
            m_eff = ma * mb / (ma + mb)
            ii = []
            for a_ in o + [ax]:
                arm2 = (d / 2) ** 2 if a_ != ax else 0.0
                Ia, Ib = segs[ia].inertia[a_] + ma * arm2, segs[ib].inertia[a_] + mb * arm2
                ii.append(Ia * Ib / (Ia + Ib))
            k_twist = TWIST_RATIO * 0.5 * (k_bend[0] + k_bend[1])
            omega = max(math.sqrt(max(kn, *k_shear) / m_eff),
                        math.sqrt(max(k_bend[0] / ii[0], k_bend[1] / ii[1], k_twist / ii[2])))
            joints.append(Joint(ia, ib, ax, d, kn, TENSION_RATIO * kn, k_shear, k_bend, k_twist,
                                m_eff, tuple(ii), k.damping_ratio, min(lk, JOINT_LOCK_MAX),
                                BEND_LOCK_RAD, fixed=k.rigid or omega > omega_max))

    # adjacent segments (26-neighborhood) never collide; others do (folding)
    exclude = []
    keys = list(index)
    for p, q in itertools.combinations(keys, 2):
        if max(abs(p[0] - q[0]), abs(p[1] - q[1]), abs(p[2] - q[2])) <= 1:
            exclude.append((index[p], index[q]))

    mobile = None
    if k.mobile:
        target = np.array(k.com_offset) * L
        dist = np.array([np.linalg.norm(s.pos - target) for s in segs])
        parent = int(np.argmin(dist))
        # hang the mass from up to 8 surrounding segments: on one light segment of a soft object it
        # collapsed the joints around that segment
        order = [int(i) for i in np.argsort(dist) if dist[i] >= 0.25 * float(np.min(cell))]
        anchors = order[:8] if not k.rigid else [parent]
        mm = k.mass * frac
        kk = mm * (2 * math.pi * k.mobile_freq) ** 2
        mobile = Mobile(parent, target - segs[parent].pos, mm, kk,
                        2 * MOBILE_DAMPING_RATIO * math.sqrt(kk * mm),
                        min(k.mobile_range, 0.5 * float(np.min(L)) - 1e-4), anchors)

    return ContentDesc(k, segs, joints, exclude, mobile, k.rigid, dict(internal))


def theory_checks(desc: ContentDesc):
    """Total mass, COM and inertia of the description against kappa (T0)."""
    k = desc.kappa
    masses = [s.mass for s in desc.segments]
    pts = [s.pos + s.com for s in desc.segments]
    if desc.mobile:
        masses.append(desc.mobile.mass)
        pts.append(desc.segments[desc.mobile.parent].pos + desc.mobile.rest)
    M = float(np.sum(masses))
    com = np.sum([m * p for m, p in zip(masses, pts)], axis=0) / M
    return {"mass": M, "mass_target": k.mass, "com": com.tolist(),
            "com_target": (np.array(k.com_offset) * np.array(k.size)).tolist()}
