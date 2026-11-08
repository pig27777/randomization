"""Unified content model: one fixed topology, items differ only in behavior parameters X.

Structure (B): free rigid core + surface patches + one internal mobile mass.
Each patch slides along its face normal through three series elements:
  soft spring -> stiff spring -> slow Kelvin-Voigt element.
Each element stops at its own joint limit, which produces the lock-up.

All values in PRESETS are authored, not measured. SI units; targets MuJoCo 3.14.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

GRID = (5, 3, 1)  # surface cells along x, y, z
PATCH_HALF_T = 0.0015
K_MAX = 2e4  # stiffness above this is carried by the joint limit instead


@dataclass(frozen=True)
class ContentX:
    half: tuple[float, float, float]  # envelope half extents, x is the long axis
    p: float  # superquadric exponent of the y-z cross-section (2 = round)
    q: float  # superquadric exponent along x
    mass: float
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    eps_knee: float = 0.2  # strain and pressure where the soft element locks
    p_knee: float = 200.0
    eps_lock: float = 0.75  # strain and pressure where all elements lock
    p_lock: float = 20000.0
    hysteresis: float = 0.0  # joint dry friction as a fraction of the knee force
    creep_eps: float = 0.0  # extra strain under p_creep at long hold
    creep_tau: float = 2.0
    p_creep: float = 1000.0
    coupling: float = 0.1  # neighbor coupling as a fraction of the soft stiffness
    mobile_frac: float = 0.0
    mobile_range: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mobile_period: float = 1.0
    mobile_damping_ratio: float = 1.0
    mobile_friction: float = 0.0
    friction: float = 0.6


PRESETS = {
    "book": ContentX(half=(0.12, 0.085, 0.015), p=40, q=40, mass=0.5,
                     eps_knee=0.002, p_knee=5e3, eps_lock=0.004, p_lock=5e4,
                     coupling=1.0, friction=0.4),
    "clothing": ContentX(half=(0.15, 0.125, 0.025), p=4, q=4, mass=0.3,
                         eps_knee=0.2, p_knee=200, eps_lock=0.75, p_lock=2e4,
                         hysteresis=0.5, creep_eps=0.05, creep_tau=2.0, p_creep=1000,
                         mobile_frac=0.4, mobile_range=(0.04, 0.03, 0.01),
                         mobile_period=2.0, mobile_damping_ratio=1.0, mobile_friction=0.1,
                         friction=0.7),
    "bottle": ContentX(half=(0.11, 0.035, 0.035), p=2, q=8, mass=0.55,
                       eps_knee=0.002, p_knee=5e3, eps_lock=0.01, p_lock=1e5,
                       coupling=1.0, mobile_frac=0.75, mobile_range=(0.05, 0.01, 0.01),
                       mobile_period=0.7, mobile_damping_ratio=0.15, friction=0.5),
}


def _f(pt, x):
    a, b, c = x.half
    cross = abs(pt[1] / b) ** x.p + abs(pt[2] / c) ** x.p
    return cross ** (x.q / x.p) + abs(pt[0] / a) ** x.q


def _surface(x, axis, sign, u, v, others):
    """Distance along the face normal to the envelope, or None when the ray misses it."""
    pt = np.zeros(3)
    pt[others[0]], pt[others[1]] = u, v
    if _f(pt, x) >= 1:
        return None
    lo, hi = 0.0, x.half[axis]
    for _ in range(40):
        mid = (lo + hi) / 2
        pt[axis] = mid
        lo, hi = (mid, hi) if _f(pt, x) < 1 else (lo, mid)
    return sign * lo


def _element_params(x, axis, area):
    """Invert the two-slope target curve into series springs, gaps and creep element."""
    depth = x.half[axis]
    d1, dl = depth * x.eps_knee, depth * x.eps_lock
    f1, fl = x.p_knee * area, x.p_lock * area
    s1, s2 = d1 / f1, (dl - d1) / (fl - f1)
    k1, k2 = 1 / (s1 - s2), 1 / s2
    if k1 > K_MAX or k2 > K_MAX:
        k1, k2 = min(k1, K_MAX), min(k2, K_MAX)
        g1, g2 = d1, dl - d1
    else:
        g1, g2 = f1 / k1, fl / k2
    if x.creep_eps > 0:
        g3 = depth * x.creep_eps
        k3 = x.p_creep * area / g3
        c3 = k3 * x.creep_tau
        g3 *= 3
    else:
        k3, c3, g3 = 1e3, 1e1, 1e-6
    return dict(k1=k1, k2=k2, g1=g1, g2=g2, k3=k3, c3=c3, g3=g3,
                f=x.hysteresis * f1)


def _joint(name, axis_vec, k, damping, friction, gap):
    arm = max(1e-3, k / 1e6)
    damp = damping + 0.6 * np.sqrt(k * arm)
    return (f'<joint name="{name}" type="slide" axis="{axis_vec[0]} {axis_vec[1]} {axis_vec[2]}" '
            f'stiffness="{k:.6g}" damping="{damp:.6g}" frictionloss="{friction:.6g}" armature="{arm:.6g}" '
            f'limited="true" range="{-gap:.6g} 0.0003" solreflimit="-1e6 -500"/>')


def item_xml(x: ContentX, pos=(0.0, 0.0, 0.0), name="item"):
    """Return (body xml, tendon xml, active patch count). Only numbers change between items."""
    patches, by_face = [], {}
    n_active = 0
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        nu, nv = GRID[others[0]], GRID[others[1]]
        hu, hv = x.half[others[0]] / nu, x.half[others[1]] / nv
        for sign in (1, -1):
            for iu in range(nu):
                for iv in range(nv):
                    u = -x.half[others[0]] + hu * (2 * iu + 1)
                    v = -x.half[others[1]] + hv * (2 * iv + 1)
                    s = _surface(x, axis, sign, u, v, others)
                    active = s is not None
                    n_active += active
                    pos_p = np.zeros(3)
                    pos_p[others[0]], pos_p[others[1]] = u, v
                    pos_p[axis] = (s - sign * PATCH_HALF_T) if active else 0.0
                    size = np.zeros(3)
                    size[others[0]], size[others[1]] = hu * 1.03, hv * 1.03
                    # Side patches must not span the thickness, or they act as rigid posts under
                    # a press along the thinner axis. Larger faces own the shared edges.
                    for j, other in enumerate(others):
                        if other > axis:
                            scale = 1 - x.eps_lock
                            if other == 2:
                                scale *= 2.0 ** (-1.0 / x.p)
                            pos_p[other] *= scale
                            size[other] *= scale
                    size[axis] = PATCH_HALF_T
                    patches.append(dict(axis=axis, sign=sign, iu=iu, iv=iv, pos=pos_p, size=size,
                                        active=active, area=4 * hu * hv))
    patch_mass = 0.1 * x.mass / max(n_active, 1)
    mobile_mass = max(x.mobile_frac * x.mass, 1e-5)
    core_mass = x.mass - patch_mass * n_active - x.mobile_frac * x.mass

    # The core sits inside the envelope and inside the lock depth, so it backstops only.
    inset = 2.0 ** (-1.0 / x.p)
    core_half = [x.half[0] * (1 - x.eps_lock) - 2 * PATCH_HALF_T,
                 x.half[1] * inset * (1 - x.eps_lock) - 2 * PATCH_HALF_T,
                 x.half[2] * inset * (1 - x.eps_lock) - 2 * PATCH_HALF_T]
    core_half = [max(h, 1e-3) for h in core_half]
    a, b, c = core_half
    inertia = [core_mass / 3 * (b * b + c * c), core_mass / 3 * (a * a + c * c), core_mass / 3 * (a * a + b * b)]
    com = x.com_offset

    contact = f'contype="2" conaffinity="1" condim="3" friction="{x.friction} 0.005 0.0001" solref="-1e5 -300" solimp="0.95 0.99 0.0005"'
    out = [f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint name="{name}_free"/>',
           f'<inertial pos="{com[0]} {com[1]} {com[2]}" mass="{core_mass:.6g}" diaginertia="{inertia[0]:.6g} {inertia[1]:.6g} {inertia[2]:.6g}"/>',
           f'<geom name="{name}_core" type="box" size="{a:.6g} {b:.6g} {c:.6g}" mass="0" rgba=".35 .35 .4 1" {contact}/>']

    tendons = []
    for i, pt in enumerate(patches):
        axis_vec = np.zeros(3)
        axis_vec[pt["axis"]] = pt["sign"]
        el = _element_params(x, pt["axis"], pt["area"])
        tag = f'{name}_p{i}'
        pos_s = " ".join(f"{v:.6g}" for v in pt["pos"])
        hidden = "" if pt["active"] else ' contype="0" conaffinity="0"'
        geom_contact = contact if pt["active"] else f'contype="0" conaffinity="0" condim="3"'
        out.append(f'<body name="{tag}_a" pos="{pos_s}"><inertial pos="0 0 0" mass="1e-4" diaginertia="1e-8 1e-8 1e-8"/>'
                   + _joint(f'{tag}_soft', axis_vec, el["k1"], 0, el["f"], el["g1"]))
        out.append(f'<body name="{tag}_b"><inertial pos="0 0 0" mass="1e-4" diaginertia="1e-8 1e-8 1e-8"/>'
                   + _joint(f'{tag}_stiff', axis_vec, el["k2"], 0, el["f"], el["g2"]))
        out.append(f'<body name="{tag}_c">' + _joint(f'{tag}_slow', axis_vec, el["k3"], el["c3"], 0, el["g3"]))
        size_s = " ".join(f"{v:.6g}" for v in pt["size"])
        rgba = ".85 .55 .35 1" if pt["active"] else "0 0 0 0"
        out.append(f'<geom name="{tag}" type="box" size="{size_s}" mass="{patch_mass:.6g}" rgba="{rgba}" {geom_contact}/>')
        out.append('</body></body></body>')
        pt["k1"] = el["k1"]
        by_face.setdefault((pt["axis"], pt["sign"]), {})[(pt["iu"], pt["iv"])] = (f'{tag}_soft', pt)

    for cells in by_face.values():
        for (iu, iv), (joint, pt) in cells.items():
            for neighbor in ((iu + 1, iv), (iu, iv + 1)):
                if neighbor in cells:
                    other, other_pt = cells[neighbor]
                    if pt["active"] and other_pt["active"]:
                        kc = x.coupling * min(pt["k1"], K_MAX)
                        tendons.append(f'<fixed name="{joint}_{neighbor[0]}_{neighbor[1]}" stiffness="{kc:.6g}" '
                                       f'springlength="0 0"><joint joint="{joint}" coef="1"/><joint joint="{other}" coef="-1"/></fixed>')

    k_m = mobile_mass * (2 * np.pi / x.mobile_period) ** 2
    c_m = 2 * x.mobile_damping_ratio * np.sqrt(k_m * mobile_mass)
    out.append(f'<body name="{name}_mobile"><inertial pos="0 0 0" mass="{mobile_mass:.6g}" diaginertia="1e-6 1e-6 1e-6"/>')
    for i, axis_name in enumerate("xyz"):
        vec = ["0", "0", "0"]
        vec[i] = "1"
        r = max(x.mobile_range[i], 1e-6)
        out.append(f'<joint name="{name}_mob_{axis_name}" type="slide" axis="{" ".join(vec)}" stiffness="{k_m:.6g}" '
                   f'damping="{c_m:.6g}" frictionloss="{x.mobile_friction:.6g}" armature="1e-3" limited="true" '
                   f'range="{-r:.6g} {r:.6g}" solreflimit="0.002 1"/>')
    out.append('</body></body>')
    return "\n".join(out), "\n".join(tendons), n_active


def com_in_core(model, data, name="item"):
    """Whole-item center of mass, expressed in the core frame."""
    root = model.body(name).id
    total, moment = 0.0, np.zeros(3)
    for body in range(model.nbody):
        ancestor = body
        while ancestor not in (0, root):
            ancestor = model.body_parentid[ancestor]
        if ancestor == root:
            total += model.body_mass[body]
            moment += model.body_mass[body] * data.xipos[body]
    world = moment / total
    return data.xmat[root].reshape(3, 3).T @ (world - data.xpos[root])


def scene_header(dt=0.0005):
    return (f'<mujoco><option timestep="{dt}" integrator="implicitfast" solver="Newton" iterations="60" '
            f'tolerance="1e-7" cone="elliptic"/><size memory="300M"/>'
            '<visual><global offwidth="1280" offheight="800"/></visual>'
            '<asset><texture type="skybox" builtin="gradient" rgb1=".9 .9 .95" rgb2=".6 .65 .7" width="64" height="64"/></asset>')


TOOL_CONTACT = 'contype="1" conaffinity="2" condim="3" friction="0.8 0.005 0.0001" solref="-1e5 -300" solimp="0.95 0.99 0.0005"'
