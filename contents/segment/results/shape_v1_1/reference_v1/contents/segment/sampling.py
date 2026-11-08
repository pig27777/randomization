"""kappa_content sampling for the policy line (test/content_model/REPORT.md round 2, S2).

    rng = np.random.default_rng(0)
    k = sample_kappa(rng, "mixed")            # "hard" | "bendable" | "compressible" | "mixed"
    desc = build(k)                            # engine-agnostic description (model.build)
    # MuJoCo: mj_adapter.object_xml(desc, origin) / mj_adapter.scene(...)

Draws are uniform in the reparameterized theta space inside per-category bounds (log for positive
quantities, uniform for fractions), then rejected until every constraint in check_kappa() holds, so
sample_kappa only returns kappas that satisfy all of them. The bounds are round-2 working values, not a
physical distribution (contents/package_params.md, "扰动范围目前全部待定").
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from model import (ANCHORS, G, JOINT_LOCK_MAX, PAD, SEG_CHOICES, SEG_MAX, Kappa, analytic_internal,
                   build, internal_params, load_table, mass_layout)

CATEGORIES = ("hard", "bendable", "compressible")
NEAR_RIGID_DROOP = 0.05          # droop below this is treated as rigid (round-1 review)
LOCK_MIN = 0.12                  # secant interval 2 % .. 0.6 x lock needs some width
SELF_WEIGHT_FRAC = 0.5           # self-weight compression <= this x lock strain
BUCKLING_C = 1.0                 # buckling estimate factor (see buckling_force)

# per-category bounds: (low, high); log-uniform for the positive quantities
BOUNDS = {
    "hard": dict(Lx=(0.12, 0.40), Ly=(0.08, 0.25), Lz=(0.02, 0.15), density=(150, 1200)),
    "bendable": dict(Lx=(0.15, 0.40), Ly=(0.10, 0.30), Lz=(0.02, 0.08), density=(80, 400),
                     squeeze_y=(0.1, 2.0), squeeze_z=(0.1, 1.5), lock=(LOCK_MIN, 0.4), droop=(0.2, 0.6),
                     damping=(0.2, 1.0), nx=(3, 4, 6), ny=(3, 4), nz=(2, 3)),
    "compressible": dict(Lx=(0.15, 0.45), Ly=(0.12, 0.35), Lz=(0.06, 0.20), density=(30, 250),
                         squeeze_y=(0.03, 0.5), squeeze_z=(0.05, 0.6), lock=(0.15, 0.4),
                         droop=(NEAR_RIGID_DROOP, 0.35), damping=(0.2, 1.0), nx=(2, 3, 4, 6), ny=(3, 4),
                         nz=(2, 3)),
}
HARD_SEGMENTS = ((2, 2, 1), (4, 3, 2), (6, 4, 3))
SURFACE_FRICTION = (0.2, 1.0)    # PE film ~0.2-0.3 to kraft paper / cardboard
SKIN_COMPLIANCE = (1e-5, 1e-3)   # m/N per contact point (bubble-wrap-like layers); off for 70 %
SKIN_P = 0.3
MOBILE_P = 0.3


def _logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def lock_cap(n):
    """Largest object lock strain that keeps the joint strain (n / (n - 1) x object) <= JOINT_LOCK_MAX."""
    return JOINT_LOCK_MAX * (n - 1) / n if n > 1 else 0.0


def min_nx_for_droop(droop):
    """Softer (larger droop) needs more segments along x; tentative, see round-1 report."""
    return 2 if droop < 0.15 else 3 if droop < 0.35 else 4


def secant_end(lock):
    return min(0.15, 0.6 * lock)


def buckling_force(k: Kappa, internal):
    """Euler estimate for R1-y: a clamped strip of width PAD + one cell, plate rigidity D, length Ly."""
    nx = k.segments[0]
    w = PAD + k.size[0] / nx
    return BUCKLING_C * 4 * math.pi ** 2 * internal["D"] * w / k.size[1] ** 2


def self_weight_ratio(k: Kappa, internal):
    """Worst bottom z-joint compression under self weight, as a fraction of the joint lock strain.
    Uses the per-segment mass layout (COM offset) and puts the mobile mass on its column."""
    nx, ny, nz = k.segments
    if nz < 2:
        return 0.0
    W, _ = mass_layout(k)
    col = (1 - (k.mobile_frac if k.mobile else 0.0)) * k.mass * W[:, :, 1:].sum(axis=2)   # mass above layer 0
    if k.mobile:
        L = np.array(k.size)
        c = np.array(k.com_offset) * L + L / 2
        i, j = min(int(c[0] / L[0] * nx), nx - 1), min(int(c[1] / L[1] * ny), ny - 1)
        if int(c[2] / L[2] * nz) >= 1:
            col[i, j] += k.mobile_frac * k.mass
    area = (k.size[0] / nx) * (k.size[1] / ny)
    strain = col.max() * G / (internal["E_z"] * area)
    joint_lock = k.lock_strain_z * nz / (nz - 1)
    return float(strain / joint_lock)


def check_kappa(k: Kappa, category=None, table=None, require_table=True, first_only=False):
    """List of violated constraints (empty = valid). require_table=False skips the calibrated-range check
    (only for stability experiments before the table covers every segment count); first_only returns at
    the first violation (sampling speed)."""
    bad = _Violations(first_only)
    try:
        _check(k, category, table, require_table, bad)
    except _Stop:
        pass
    return list(bad)


class _Stop(Exception):
    pass


class _Violations(list):
    def __init__(self, first_only):
        super().__init__()
        self.first_only = first_only

    def append(self, item):
        super().append(item)
        if self.first_only:
            raise _Stop


def _check(k, category, table, require_table, bad):
    L = np.array(k.size)
    values = [*k.size, k.mass, *k.com_offset, k.squeeze_stiffness_y, k.squeeze_stiffness_z,
              k.lock_strain_y, k.lock_strain_z, k.droop_ratio, k.damping_ratio,
              k.mobile_frac, k.mobile_range, k.mobile_freq, k.surface_friction, k.skin_compliance]
    if not np.all(np.isfinite(values)):
        bad.append("non-finite kappa")
        return
    if np.any(L <= 0) or k.mass <= 0:
        bad.append("non-positive size or mass")
        return
    if any(n not in SEG_CHOICES or n > cap for n, cap in zip(k.segments, SEG_MAX)):
        bad.append("invalid segments")
        return
    if k.category not in (*CATEGORIES, None) or (category is not None and k.category != category):
        bad.append("category mismatch")
    if k.category is not None and k.rigid != (k.category == "hard"):
        bad.append("category and rigid flag mismatch")
    if min(k.squeeze_stiffness_y, k.squeeze_stiffness_z, k.damping_ratio) <= 0 or not 0 <= k.droop_ratio < 1:
        bad.append("invalid response or damping")
    if k.mobile and not (0 < k.mobile_frac < 1 and k.mobile_freq > 0):
        bad.append("invalid mobile mass or frequency")
    rho = k.mass / float(np.prod(L))
    if not 30 <= rho <= 2000:
        bad.append(f"density {rho:.0f} kg/m3 outside 30-2000")
    if k.mobile and not 0 < k.mobile_range < 0.5 * float(L.min()):
        bad.append("mobile_range outside (0, half the shortest side)")
    if any(abs(c) >= 0.5 for c in k.com_offset):
        bad.append("com_offset outside (-0.5, 0.5)")
    if not 0.05 <= k.surface_friction <= 1.5 or k.skin_compliance < 0:
        bad.append("surface_friction outside [0.05, 1.5] or negative skin_compliance")
    if k.rigid:
        return
    nx, ny, nz = k.segments
    if ny < 3:
        bad.append("ny < 3 (a seam under the pad center)")
    if nz < 2:
        bad.append("nz < 2 (no compliance in z)")
    if nx < min_nx_for_droop(k.droop_ratio):
        bad.append(f"nx {nx} < {min_nx_for_droop(k.droop_ratio)} for droop {k.droop_ratio:.2f}")
    if k.droop_ratio < NEAR_RIGID_DROOP:
        bad.append("droop below the near-rigid threshold (use rigid)")
    for name, lock, n in (("lock_strain_y", k.lock_strain_y, ny), ("lock_strain_z", k.lock_strain_z, nz)):
        if not LOCK_MIN <= lock <= lock_cap(n) + 1e-12:
            bad.append(f"{name} {lock:.3f} outside [{LOCK_MIN}, {lock_cap(n):.3f}] for n = {n}")
    # category validates the label; build() selects geometry by size, independently
    # of that label. Validation and construction must inspect the same table entry.
    internal = internal_params(k, table if table is not None else load_table(), strict=False)
    if not require_table:
        pass
    elif not internal["table"]:
        bad.append("no calibration table for these segments")
    else:
        for q, ok in internal["in_table"].items():
            if not ok:
                bad.append(f"{q} target outside the calibrated range")
    if self_weight_ratio(k, internal) > SELF_WEIGHT_FRAC:
        bad.append(f"self-weight compression {self_weight_ratio(k, internal):.2f} x lock_z > {SELF_WEIGHT_FRAC}")
    f_target = k.squeeze_stiffness_y * 1e3 * k.size[1] * secant_end(k.lock_strain_y)
    if f_target > buckling_force(k, internal):
        bad.append(f"squeeze force {f_target:.1f} N above the buckling estimate {buckling_force(k, internal):.1f} N")


def _draw(rng, category):
    b = BOUNDS[category]
    Lx = _logu(rng, *b["Lx"])
    Ly = min(_logu(rng, *b["Ly"]), Lx)                   # x is the long side
    Lz = min(_logu(rng, *b["Lz"]), Ly)
    mass = _logu(rng, *b["density"]) * Lx * Ly * Lz
    com = tuple(float(v) for v in rng.uniform(-0.15, 0.15, 3))
    mobile = bool(rng.random() < MOBILE_P)
    mob = dict(mobile=mobile, mobile_frac=float(rng.uniform(0.1, 0.5)),
               mobile_range=float(rng.uniform(0.2, 0.9) * 0.5 * min(Lx, Ly, Lz)),
               mobile_freq=_logu(rng, 1.0, 5.0),
               surface_friction=_logu(rng, *SURFACE_FRICTION),
               skin_compliance=_logu(rng, *SKIN_COMPLIANCE) if rng.random() < SKIN_P else 0.0,
               category=category)
    if category == "hard":
        segs = HARD_SEGMENTS[rng.integers(len(HARD_SEGMENTS))]
        return Kappa(size=(Lx, Ly, Lz), mass=mass, com_offset=com, segments=segs, rigid=True,
                     squeeze_stiffness_y=50.0, squeeze_stiffness_z=50.0, lock_strain_y=0.02, lock_strain_z=0.02,
                     droop_ratio=0.0, **mob)
    droop = float(rng.uniform(*b["droop"]))
    nx_ok = [n for n in b["nx"] if n >= min_nx_for_droop(droop)]
    segs = (int(rng.choice(nx_ok)), int(rng.choice(b["ny"])), int(rng.choice(b["nz"])))
    lock_y = float(rng.uniform(b["lock"][0], min(b["lock"][1], lock_cap(segs[1]))))
    lock_z = float(rng.uniform(b["lock"][0], min(b["lock"][1], lock_cap(segs[2])))) \
        if lock_cap(segs[2]) > b["lock"][0] else lock_cap(segs[2])
    return Kappa(size=(Lx, Ly, Lz), mass=mass, com_offset=com, segments=segs, rigid=False,
                   squeeze_stiffness_y=_logu(rng, *b["squeeze_y"]), squeeze_stiffness_z=_logu(rng, *b["squeeze_z"]),
                   lock_strain_y=lock_y, lock_strain_z=lock_z, droop_ratio=droop,
                   damping_ratio=_logu(rng, *b["damping"]), **mob)


def sample_kappa(rng: np.random.Generator, category: str = "mixed", table=None, max_tries=10000,
                 require_table=True) -> Kappa:
    """A kappa of the given category ("mixed" picks one uniformly) that satisfies check_kappa()."""
    table = table if table is not None else load_table()
    cat = CATEGORIES[rng.integers(3)] if category == "mixed" else category
    if cat not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; use one of {CATEGORIES} or 'mixed'")
    for _ in range(max_tries):
        k = _draw(rng, cat)
        # check with the table entry build() will use (closest anchor geometry), not the category's
        if not check_kappa(k, None, table, require_table, first_only=True):
            return k
    raise RuntimeError(f"no valid {cat} kappa in {max_tries} draws; the bounds and constraints do not overlap")


def acceptance(category, n=2000, seed=0):
    """Fraction of raw draws accepted and the most frequent rejection reasons (diagnostics)."""
    rng = np.random.default_rng(seed)
    table = load_table()
    reasons, ok = {}, 0
    for _ in range(n):
        bad = check_kappa(_draw(rng, category), None, table)
        ok += not bad
        for r in bad:
            key = r.split(" ")[0] + " " + " ".join(w for w in r.split(" ")[1:3] if not w[:1].isdigit())
            reasons[key] = reasons.get(key, 0) + 1
    return ok / n, sorted(reasons.items(), key=lambda kv: -kv[1])[:8]


def rj_internal_bounds(table, category):
    """Direct internal-modulus ranges covered by valid scans, in SI units.

    Reconstruct the actual scanned coefficients, not the whole-response targets.
    RJ uses these log-uniform ranges independently of the sampled response labels.
    """
    from calibrate import KNOBS, anchor_for
    values = {q: [] for q in ("E_y", "E_z", "D")}
    for key, entry in table.items():
        if not key.startswith(category + ":"):
            continue
        seg = tuple(map(int, key.split(":")[1].split("x")))
        k = anchor_for(category, seg)
        for knob, (name, field) in KNOBS.items():
            for measured, correction in entry.get(knob, []):
                values[name].append(analytic_internal(replace(k, **{field: measured}))[name] * correction)
    if any(len(v) < 2 or min(v) <= 0 or max(v) <= min(v) for v in values.values()):
        raise ValueError(f"insufficient calibration scans for RJ {category}")
    return {name: (min(v), max(v)) for name, v in values.items()}


def sample_rj(rng: np.random.Generator, category: str = "mixed", table=None):
    """RJ ContentDesc: same sampled geometry/mass as R3; direct log-random E_y/E_z/D.

    Call r3(desc.kappa, desc=desc) or scene(desc, ...). Passing only desc.kappa
    would rebuild the response-mapped R3 object instead of this RJ control.
    Hard objects keep the rigid switch. Soft joints use the same fixed ratios,
    limits and damping template, with no response-to-parameter table lookup.
    """
    table = load_table() if table is None else table
    from mj_adapter import OMEGA_MAX
    k = sample_kappa(rng, category, table)
    if k.rigid:
        return build(k, table=table, omega_max=OMEGA_MAX)
    bounds = rj_internal_bounds(table, k.category)
    internal = {name: _logu(rng, *limits) for name, limits in bounds.items()}
    return build(k, internal=internal | {"sampling": "RJ", "bounds": bounds}, omega_max=OMEGA_MAX)
