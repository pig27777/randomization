"""MuJoCo 3.17 adapter: ContentDesc -> MJCF, plus scene (table, kinematic pads) and measurements.

Each segment is a free body. The CPU reference uses conservative six-DOF face bushings.
Legacy truss modes are kept to reproduce earlier rounds:
  center (symmetric part + compression-only part with the compression lock as tendon limit),
  2 + 2 edge tendons for the two bending axes (with the bending lock as tendon limit),
  2 + 2 shear tendons across the joint plane (their offsets also give twist).
MuJoCo trees cannot close loops, the tendons can. Fixed joints (auto-lock) become weld equalities;
a rigid object is a single body with one box geom per segment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from metrics import JointGeometry
from model import ContentDesc, PAD, _bend_lower_length
from bushing import Bushings, joint_damping

DT = 0.0005                      # round 2 (0.25 ms in round 1)
BUSHING_DT = 0.0004              # explicit pair forces, native stops: 0.4 ms; legacy DT is unchanged
OMEGA_MAX = 0.8 / DT             # auto-lock limit for explicit tendon springs
PAD_T = 0.004                    # pad half thickness = edge radius (4 mm rounded edges)
PAD_FRICTION = 0.8
TABLE_FRICTION = 0.5
SEG_FRICTION = 0.5
CONTACT_K = 3e5                  # N/m, static stiffness per contact point, independent of body mass
CONTACT_D = 0.99
CONTACT_DAMPRATIO = 1.0          # normal contact damping relative to critical
BEND_ARM = 0.9                   # edge tendons at 90 % of the half cell
SHEAR_OFFSET = 0.5               # shear tendons at 50 % of the half cell (gives twist)
SHEAR_LOCK_ARM = 0.5             # shear-lock diagonals end at 50 % of the half cell
SHEAR_LOCK_DOFS = {(1, 0), (1, 2)}  # truss: (joint axis, shear axis), y-joints, the squeeze direction of R3;
                                    # bushing locks every shear (edge fibres alone let shear bypass the lock)
JOINT_MODE = "bushing"           # CPU reference; raw XML cannot carry Python pair forces
DAMPED_GROUPS = ("center", "bend", "shear")   # truss tendon groups that carry damping
# Share Newton with bagged scenes (I-27); convergence normally stops before these upper limits.
SOLVER = 'solver="Newton" iterations="100" ls_iterations="20" tolerance="1e-8"'
LIMIT_SOLREF = "0.001 2"         # joint limits: 1 ms, overdamped, so a released lock does not snap
LIMIT_SOLIMP = "0.95 0.99 0.0005"
WELD_ON, WELD_RELEASE = 0.9, 0.8  # compaction hysteresis
COMPACTION_LOAD_TAU = .005      # reject a single solver/contact dropout, matching the grip force filter
COMPACTION_LOAD_MIN = .05       # external confinement must exceed the protocol's touch force (N)
COMPACTION_SOLREF = "0.002 2"   # softer than the 1 ms geometric stops: reduce rigid-cluster path jumps


def contact_k(mass, dt=DT):
    """MuJoCo's static contact stiffness is mass*d*k/(1-d) for solref (-k, -b); scale k by the body mass
    so every object sees about CONTACT_K, with critical damping of the normal direction. For light bodies
    at large steps the stiffness is capped so that omega*dt <= 0.5."""
    k = CONTACT_K * (1 - CONTACT_D) / (CONTACT_D * mass)
    k = min(k, (0.5 / dt) ** 2 / CONTACT_D)
    return k, 2 * CONTACT_DAMPRATIO * math.sqrt(k)


def contact_attr(mass, dt=DT):
    """Object geom contact parameters. Table uses solmix 0, so the object geom sets the pair."""
    k, b = contact_k(mass, dt)
    return f'solref="{-k:.6g} {-b:.6g}" solimp="{CONTACT_D} {CONTACT_D} 0.001" solmix="1"'


def pad_pair(pad_geom, obj_geom, mass, dt=DT, friction=PAD_FRICTION, skin=0.0):
    """Explicit pad-object pair: friction rows are velocity-level with time constant 1/(d*b); for heavy
    bodies the normal b gives ~8 ms and the object creeps along the pads, so friction gets b >= 1000,
    but the time constant stays >= 2.5 steps for stability. friction is the package surface friction;
    skin (m/N) adds an equivalent skin compliance in series with each contact point."""
    k, b = contact_k(mass, dt)
    if skin > 0:
        K = k * CONTACT_D * mass / (1 - CONTACT_D)            # static stiffness per point
        k *= 1.0 / (1.0 + K * skin)
        b = 2 * CONTACT_DAMPRATIO * math.sqrt(k)
    bf = min(max(b, 1000.0), 1 / (2.5 * CONTACT_D * dt))
    return (f'<pair geom1="{pad_geom}" geom2="{obj_geom}" condim="3" friction="{friction} {friction} 0.005 '
            f'0.0001 0.0001" solref="{-k:.6g} {-b:.6g}" solreffriction="{-k:.6g} {-bf:.6g}" '
            f'solimp="{CONTACT_D} {CONTACT_D} 0.001"/>')


PASSIVE = 'solref="0.01 1" solmix="0"'


def _v(x):
    return " ".join(f"{float(a):.8g}" for a in x)


def _e(ax, s=1.0):
    v = np.zeros(3)
    v[ax] = s
    return v


@dataclass
class TendonInfo:
    kind: str                    # center_sym / center_comp / bend / shear
    joint: int
    stiffness: float
    rest: float
    lower: float = -1.0          # limit (only center_comp and bend)
    cell: float = 0.0


@dataclass
class Built:
    xml: str
    desc: ContentDesc
    tendons: list = field(default_factory=list)
    realized: list = field(default_factory=list)   # per joint: realized comp / tens / twist
    n_fixed: int = 0
    bushings: list = field(default_factory=list)  # descriptor joint indices
    tendon_names: list = field(default_factory=list)


def tree_edges(desc):
    """Spanning tree: all z-joints, layer-0 x-joints, and a y-spine through the middle column of layer 0.
    Returns {child: (parent, joint index)} and the root segment."""
    nx, ny, nz = desc.kappa.segments
    idx = {s.idx: n for n, s in enumerate(desc.segments)}
    jmap = {(min(j.a, j.b), max(j.a, j.b)): ji for ji, j in enumerate(desc.joints)}
    i0, j0 = nx // 2, ny // 2
    root = idx[(i0, j0, 0)]
    parent = {}

    def link(c, p):
        parent[c] = (p, jmap[(min(c, p), max(c, p))])

    for j in range(ny):
        if j != j0:
            link(idx[(i0, j, 0)], idx[(i0, j - 1 if j > j0 else j + 1, 0)])
    for j in range(ny):
        for i in range(nx):
            if i != i0:
                link(idx[(i, j, 0)], idx[(i - 1 if i > i0 else i + 1, j, 0)])
    for j in range(ny):
        for i in range(nx):
            for k in range(1, nz):
                link(idx[(i, j, k)], idx[(i, j, k - 1)])
    return parent, root


def object_xml(desc: ContentDesc, origin, name="s", mode=None, omega_max=None, dt=DT, table_contact=True):
    """Bodies, tendons, excludes and equalities of one object placed with its envelope center at origin.

    mode (JOINT_MODE by default) selects how the 6-DOF elastic joints are realized:
      bushing        free segments, conservative face forces applied by Sim.step; native edge-fibre stops
      truss          free segments, a spatial-tendon truss per joint (CPU reference)
      truss_dofdamp  same, no tendon damping; damping moved to the free-joint dofs (absolute drag)
      weld           free segments, one soft weld equality per joint (solref stiffness/damping)
      tree           spanning tree of slide x3 + hinge x3 joints (native stiffness, damping, limits);
                     the remaining joints close loops with undamped tendon trusses
    """
    mode = mode or JOINT_MODE
    omega_max = OMEGA_MAX if omega_max is None else omega_max
    origin = np.asarray(origin, float)
    segs = desc.segments
    affinity = 3 if table_contact else 2
    contact = lambda mass: f'contype="2" conaffinity="{affinity}" condim="3" friction="{SEG_FRICTION} 0.005 0.0001" {contact_attr(mass, dt)}'
    tendons, excludes, equalities, infos, realized, infos_mob = [], [], [], [], [], []
    bushings = []
    tendon_names = []
    mob = desc.mobile

    def mobile_xml(pos):
        out = [f'<body name="mobile" pos="{_v(pos)}">',
               f'<inertial pos="0 0 0" mass="{mob.mass:.8g}" diaginertia="1e-7 1e-7 1e-7"/>']
        for a, an in enumerate("xyz"):
            out.append(f'<joint name="mob_{an}" type="slide" axis="{_v(_e(a))}" stiffness="{mob.stiffness:.8g}" '
                       f'damping="{mob.damping:.8g}" limited="true" range="{-mob.range:.8g} {mob.range:.8g}" '
                       f'solreflimit="0.002 1"/>')
        out.append('<geom type="sphere" size="0.008" contype="0" conaffinity="0" mass="0" rgba=".9 .2 .2 1"/></body>')
        return "\n".join(out)

    if desc.rigid:
        M = sum(s.mass for s in segs)
        com = sum(s.mass * (s.pos + s.com) for s in segs) / M
        I = np.zeros((3, 3))
        for s in segs:
            r = s.pos + s.com - com
            I += np.diag(s.inertia) + s.mass * (r @ r * np.eye(3) - np.outer(r, r))
        b = [f'<body name="obj" pos="{_v(origin)}"><freejoint name="obj_free"/>',
             f'<inertial pos="{_v(com)}" mass="{M:.8g}" fullinertia="{I[0,0]:.8g} {I[1,1]:.8g} {I[2,2]:.8g} '
             f'{I[0,1]:.8g} {I[0,2]:.8g} {I[1,2]:.8g}"/>']
        for i, s in enumerate(segs):
            b.append(f'<geom name="{name}{i}" type="box" pos="{_v(s.pos)}" size="{_v(s.half)}" mass="0" '
                     f'rgba=".45 .45 .55 1" {contact(M)}/>')
        if mob is not None:
            b.append(mobile_xml(segs[mob.parent].pos + mob.rest))
        b.append("</body>")
        return Built("\n".join(b), desc), "", "", ""

    sites = {}   # (body, pos tuple) -> site name

    def site(body, pos):
        key = (body, tuple(np.round(pos, 9)))
        if key not in sites:
            sites[key] = f"{name}{body}_t{len(sites)}"
        return sites[key]

    def tendon(tname, s1, s2, stiff, springlength, damping, lower=None, upper=None):
        lim = ""
        if lower is not None:
            lim = (f' limited="true" range="{lower:.8g} {upper:.8g}" solreflimit="{LIMIT_SOLREF}" '
                   f'solimplimit="{LIMIT_SOLIMP}"')
        tendons.append(f'<spatial name="{tname}" stiffness="{stiff:.8g}" springlength="{springlength}" '
                       f'damping="{damping:.8g}"{lim}><site site="{s1}"/><site site="{s2}"/></spatial>')
        tendon_names.append(tname)

    def too_stiff(j):
        """Auto-lock on the realized truss: edge tendons add k_bend / arm^2 of axial stiffness."""
        half = segs[j.a].half
        o = [a for a in range(3) if a != j.axis]
        axial = j.k_comp + sum(j.k_bend[bi] / (BEND_ARM * half[along]) ** 2
                               for bi, along in enumerate((o[1], o[0])))
        return math.sqrt(axial / j.m_eff) > omega_max

    def truss(ji, j, damped):
        ax, d = j.axis, j.rest
        o = [a for a in range(3) if a != ax]
        half = segs[j.a].half
        zeta = j.damping_ratio if damped else 0.0
        # which tendon groups carry damping: all, or a subset (MJWarp cost of implicit tendon damping)
        zb = zeta if "bend" in DAMPED_GROUPS else 0.0
        zs = zeta if "shear" in DAMPED_GROUPS else 0.0
        zc = zeta if "center" in DAMPED_GROUPS else 0.0
        ca, cb = f"{name}{j.a}_c", f"{name}{j.b}_c"
        bend_axial = 0.0
        for bi, (about, along) in enumerate(((o[0], o[1]), (o[1], o[0]))):
            arm = BEND_ARM * half[along]
            kb = j.k_bend[bi]
            kt = kb / (2 * arm * arm)
            ct = zb * math.sqrt(kb * j.i_eff[bi]) / (arm * arm)
            lower = min(_bend_lower_length(d, arm, j.lock_bend), d * (1 - j.lock_comp) - 1e-5)
            for sgn in (1, -1):
                off = _e(along, sgn * arm)
                tendon(f"{name}j{ji}_b{bi}{'p' if sgn > 0 else 'n'}", site(j.a, off), site(j.b, off), kt,
                       f"{d:.8g} {d:.8g}", ct, lower, 10 * d)
                infos.append(TendonInfo("bend", ji, kt, d, lower, d))
            bend_axial += 2 * kt
        k_sym = max(j.k_tens - bend_axial, 0.0)
        k_cmp = max(j.k_comp - k_sym - bend_axial, 0.0)
        m = j.m_eff
        tendon(f"{name}j{ji}_cs", ca, cb, k_sym, f"{d:.8g} {d:.8g}", 2 * zc * math.sqrt(k_sym * m))
        infos.append(TendonInfo("center_sym", ji, k_sym, d, cell=d))
        tendon(f"{name}j{ji}_cc", ca, cb, k_cmp, f"{d:.8g} {100 * d:.8g}",
               2 * zc * math.sqrt(max(k_cmp, 1e-9) * m), d * (1 - j.lock_comp), 10 * d)
        infos.append(TendonInfo("center_comp", ji, k_cmp, d, d * (1 - j.lock_comp), d))
        twist = 0.0
        for si, (along, off_ax) in enumerate(((o[0], o[1]), (o[1], o[0]))):
            ell = half[along]
            r = SHEAR_OFFSET * half[off_ax]
            ks = j.k_shear[si] / 2
            for sgn in (1, -1):
                pa = _e(ax, d / 2) - _e(along, ell) + _e(off_ax, sgn * r)
                pb = _e(ax, -d / 2) + _e(along, ell) + _e(off_ax, sgn * r)
                tendon(f"{name}j{ji}_s{si}{'p' if sgn > 0 else 'n'}", site(j.a, pa), site(j.b, pb), ks,
                       f"{2 * ell:.8g} {2 * ell:.8g}", 2 * zs * math.sqrt(ks * m))
                infos.append(TendonInfo("shear", ji, ks, 2 * ell))
            twist += 2 * ks * r * r
        shear_lock(ji, j)
        realized.append({"comp": k_sym + k_cmp + bend_axial, "tens": k_sym + bend_axial, "twist": twist})

    def shear_lock(ji, j, every=False):
        # crossed diagonals in an (axis, shear) plane with a length limit only (no stiffness); their lower
        # limit is their length at the compression lock, so the allowed shear shrinks as the joint
        # compresses and vanishes at the lock; to first order they ignore bending and twist
        ax, d = j.axis, j.rest
        half = segs[j.a].half
        for si, sa in enumerate([a for a in range(3) if a != ax] if SHEAR_LOCK_ARM else ()):
            if not every and (ax, sa) not in SHEAR_LOCK_DOFS:
                continue
            arm = SHEAR_LOCK_ARM * half[sa]
            low = math.sqrt((d * (1 - j.lock_comp)) ** 2 + (2 * arm) ** 2) - 1e-6
            for sgn in (1, -1):
                tendon(f"{name}j{ji}_l{si}{'p' if sgn > 0 else 'n'}", site(j.a, _e(sa, sgn * arm)),
                       site(j.b, _e(sa, -sgn * arm)), 0.0, "0 0", 0.0, low, 10 * d)
                infos.append(TendonInfo("shearlock", ji, 0.0, math.sqrt(d * d + 4 * arm * arm), low, d))

    def edge_stops(ji, j):
        # Compression lock, bending lock and compaction support as native zero-stiffness limits. Inner
        # fibres (lower limit) sit at the arm where bending by lock_bend about the face centre alone
        # shortens them to the compression-lock length: straight compression to the lock engages all
        # four and blocks bending; partial compression allows less bending. Outer fibres at arm d (upper
        # limit) reach their limit at the same pure-bending lock, so the joint cannot pivot about an
        # inner fibre past lock_bend; straight they only cap tension near 0.84 d.
        d, low, h = j.rest, j.rest * (1 - j.lock_comp), j.lock_bend / 2
        arm = (d * math.cos(h) - low) / (2 * math.sin(h))
        if arm <= 0:
            raise ValueError(f"joint {ji}: lock_comp {j.lock_comp:.3g} too small for lock_bend {j.lock_bend:.3g} rad")
        up = d * math.cos(h) + 2 * d * math.sin(h)
        for along in (a for a in range(3) if a != j.axis):
            for sgn in (1, -1):
                tag = f"{along}{'p' if sgn > 0 else 'n'}"
                off = _e(along, sgn * arm)
                tendon(f"{name}j{ji}_e{tag}", site(j.a, off), site(j.b, off), 0.0, "0 0", 0.0, low, 10 * d)
                infos.append(TendonInfo("stop", ji, 0.0, d, low, d))
                off = _e(along, sgn * d)
                tendon(f"{name}j{ji}_o{tag}", site(j.a, off), site(j.b, off), 0.0, "0 0", 0.0, 0.0, up)
                infos.append(TendonInfo("stop_upper", ji, 0.0, d, 0.0, d))

    def dampers(ji, j):
        # Connection damping as native tendon damping, which implicitfast integrates implicitly. At the rest
        # pose each symmetric pair damps exactly two modes and cancels its cross terms: the centre line the
        # axial mode; tendons along transverse axis j at axial offsets +-e the shear along j and the bending
        # about the third axis; tendons along the more damped shear axis at lateral offsets +-r half of that
        # shear and the twist. Tendon length 2 * span keeps them clear of crossing under shear and bending.
        ct, cr = joint_damping(j)
        ax, d = j.axis, j.rest
        o = [a for a in range(3) if a != ax]
        tendon(f"{name}j{ji}_da", f"{name}{j.a}_c", f"{name}{j.b}_c", 0.0, "0 0", ct[ax])
        infos.append(TendonInfo("damper", ji, 0.0, d))
        twist_along = max(o, key=lambda a: ct[a])
        for along, about in ((o[0], o[1]), (o[1], o[0])):
            share = ct[along] / (2 if along == twist_along else 1)
            pairs = [(_e(ax, math.sqrt(cr[about] / share)), "b")]
            if along == twist_along:
                pairs.append((_e(about, math.sqrt(cr[ax] / share)), "t"))
            for offset, tag in pairs:
                span = max(np.abs(offset).max(), d)
                for sgn in (1, -1):
                    pa = _e(ax, d / 2) + sgn * offset - _e(along, span)
                    pb = _e(ax, -d / 2) + sgn * offset + _e(along, span)
                    tendon(f"{name}j{ji}_d{tag}{along}{'p' if sgn > 0 else 'n'}", site(j.a, pa), site(j.b, pb),
                           0.0, "0 0", share / 2)
                    infos.append(TendonInfo("damper", ji, 0.0, 2 * span))

    def weld(j, soft):
        if not soft:
            equalities.append(f'<weld body1="{name}{j.a}" body2="{name}{j.b}" solref="0.001 1"/>')
            return
        k_ref = j.k_comp * (1 - CONTACT_D) / (CONTACT_D * j.m_eff)
        b_ref = 2 * j.damping_ratio * math.sqrt(k_ref)
        equalities.append(f'<weld body1="{name}{j.a}" body2="{name}{j.b}" solref="{-k_ref:.6g} {-b_ref:.6g}" '
                          f'solimp="{CONTACT_D} {CONTACT_D} 0.001"/>')

    # ---- joints
    parent, root = tree_edges(desc) if mode == "tree" else ({}, None)
    tree_joint = {ji: c for c, (p, ji) in parent.items()}
    joint_xml = {}            # child segment -> joint lines (tree)
    n_fixed = 0
    for ji, j in enumerate(desc.joints):
        if ji in tree_joint:
            c = tree_joint[ji]
            p = parent[c][0]
            joint_xml[c], dropped = _tree_joint_xml(name, c, p, j, segs, omega_max)
            n_fixed += dropped == 6
            realized.append(None)
            continue
        if j.fixed or (mode != "bushing" and too_stiff(j)):
            n_fixed += 1
            weld(j, soft=False)
            realized.append(None)
        elif mode == "weld":
            weld(j, soft=True)
            realized.append(None)
        elif mode == "bushing":
            bushings.append(ji)
            edge_stops(ji, j)
            dampers(ji, j)
            shear_lock(ji, j, every=True)
            # Compaction support across directions: Sim.step switches this weld on at the current pose.
            equalities.append(f'<weld name="{name}j{ji}_w" body1="{name}{j.a}" body2="{name}{j.b}" active="false" '
                              f'solref="{COMPACTION_SOLREF}" solimp="{LIMIT_SOLIMP}"/>')
            realized.append({"comp": j.k_comp, "tens": j.k_tens, "twist": j.k_twist})
        else:
            truss(ji, j, damped=(mode == "truss"))

    # ---- bodies (flat, or nested along the tree)
    rgba = {False: ".85 .6 .4 1", True: ".5 .5 .6 1"}
    per_body = {}
    for (body, pos), sn in sites.items():
        per_body.setdefault(body, []).append(f'<site name="{sn}" pos="{_v(pos)}" size="0.0005"/>')
    children = {}
    for c, (p, _) in parent.items():
        children.setdefault(p, []).append(c)

    def body_xml(i, pos_rel, free):
        s = segs[i]
        lines = [f'<body name="{name}{i}" pos="{_v(pos_rel)}">']
        if free:
            dd = ""
            if mode == "truss_dofdamp":
                c_lin = sum(2 * jj.damping_ratio * math.sqrt(jj.k_comp * jj.m_eff) for jj in desc.joints
                            if i in (jj.a, jj.b) and not jj.fixed) / 2
                dd = f' damping="{c_lin:.6g}"'
            lines.append(f'<freejoint name="{name}{i}_free"/>' if not dd else
                         f'<joint name="{name}{i}_free" type="free"{dd}/>')
        else:
            lines.extend(joint_xml.get(i, []))
        lines += [f'<inertial pos="{_v(s.com)}" mass="{s.mass:.8g}" diaginertia="{_v(s.inertia)}"/>',
                  f'<geom name="{name}{i}" type="box" size="{_v(s.half)}" mass="0" rgba="{rgba[sum(s.idx) % 2 == 1]}" {contact(s.mass)}/>',
                  f'<site name="{name}{i}_c" pos="0 0 0" size="0.001"/>', *per_body.get(i, [])]
        for c in children.get(i, []):
            lines.append(body_xml(c, segs[c].pos - s.pos, False))
        lines.append("</body>")
        return "\n".join(lines)

    if mode == "tree":
        bodies = [body_xml(root, origin + segs[root].pos, True)]
    else:
        bodies = [body_xml(i, origin + s.pos, True) for i, s in enumerate(segs)]

    if mob is not None:
        # free body hung from the anchor segments by spring tendons; n springs along spread directions
        # give about n*k_i/3 per axis, so each gets 3k/n; the range is a tendon length limit
        mpos = segs[mob.parent].pos + mob.rest
        bodies.append(f'<body name="mobile" pos="{_v(origin + mpos)}"><freejoint name="mobile_free"/>'
                      f'<inertial pos="0 0 0" mass="{mob.mass:.8g}" diaginertia="1e-6 1e-6 1e-6"/>'
                      '<site name="mobile_c" pos="0 0 0" size="0.001"/>'
                      '<geom type="sphere" size="0.008" contype="0" conaffinity="0" mass="0" rgba=".9 .2 .2 1"/></body>')
        n_a = len(mob.anchors)
        for ai in mob.anchors:
            L0 = float(np.linalg.norm(segs[ai].pos - mpos))
            ks, cs = 3 * mob.stiffness / n_a, 3 * mob.damping / n_a
            lo = max(L0 - mob.range, 0.2 * L0)
            tendons.append(f'<spatial name="mob_t{ai}" stiffness="{ks:.8g}" springlength="{L0:.8g} {L0:.8g}" '
                           f'damping="{cs:.8g}" limited="true" range="{lo:.8g} {L0 + mob.range:.8g}" '
                           f'solreflimit="0.002 1"><site site="mobile_c"/><site site="{name}{ai}_c"/></spatial>')
            infos_mob.append(TendonInfo("mobile", -1, ks, L0))
            tendon_names.append(f"mob_t{ai}")

    for a, b in desc.exclude:
        if parent.get(a, (None,))[0] != b and parent.get(b, (None,))[0] != a:   # parent-child pairs are excluded anyway
            excludes.append(f'<exclude body1="{name}{a}" body2="{name}{b}"/>')
    built = Built("\n".join(bodies), desc, infos + infos_mob, realized, n_fixed, bushings, tendon_names)
    return built, "\n".join(tendons), "\n".join(excludes), "\n".join(equalities)


def _tree_joint_xml(name, c, p, j, segs, omega_max):
    """Joint lines for child c under parent p: slide x3 + hinge x3 at the joint midpoint, each with the
    descriptor stiffness, damping (implicit dof damping) and limits; a DOF stiffer than omega_max is dropped
    (rigid in that DOF). Returns (lines, number of dropped DOFs)."""
    ax, d = j.axis, j.rest
    o = [a for a in range(3) if a != ax]
    sgn = 1.0 if segs[c].idx[ax] > segs[p].idx[ax] else -1.0
    mid = _e(ax, -sgn * d / 2)                      # joint point in the child frame
    zeta = j.damping_ratio
    lines, dropped = [], 0
    dofs = [("c", "slide", _e(ax, sgn), j.k_comp, j.m_eff, (-j.lock_comp * d, d))]
    dofs += [(f"s{i}", "slide", _e(a), j.k_shear[i], j.m_eff, None) for i, a in enumerate(o)]
    dofs += [(f"b{i}", "hinge", _e(a), j.k_bend[i], j.i_eff[i], (-j.lock_bend, j.lock_bend)) for i, a in enumerate(o)]
    dofs += [("t", "hinge", _e(ax), j.k_twist, j.i_eff[2], None)]
    for tag, typ, axis, k, inertia, rng in dofs:
        if math.sqrt(k / inertia) > omega_max:
            dropped += 1
            continue
        lim = (f' limited="true" range="{rng[0]:.8g} {rng[1]:.8g}" solreflimit="{LIMIT_SOLREF}" '
               'solimplimit="0.95 0.99 0.0005"') if rng else ""
        lines.append(f'<joint name="{name}{c}_{tag}" type="{typ}" pos="{_v(mid)}" axis="{_v(axis)}" '
                     f'stiffness="{k:.8g}" damping="{2 * zeta * math.sqrt(k * inertia):.8g}"{lim}/>')
    return lines, dropped


def object_mjcf(desc: ContentDesc, origin, pad_geoms=(), dt=DT, name="s", mode=None):
    """MJCF pieces to embed one object in a caller's scene (policy line):
    {"worldbody": bodies, "tendon": ..., "contact": excludes + pad pairs, "equality": ...}.
    pad_geoms: names of the caller's gripper pad geoms; each gets an explicit contact pair with every
    object geom using the package surface friction (give them contype=conaffinity=0 so no default
    contact is generated). Use timestep dt with mj_adapter.SOLVER and integrator implicitfast."""
    built, tendons, excludes, equalities = object_xml(desc, origin, name=name, mode=mode,
                                                   omega_max=0.8 / dt, dt=dt)
    if built.bushings:
        raise ValueError("CPU bushings require scene() + Sim.step(); raw XML export needs an explicit "
                         "legacy mode='truss' (different mechanics, not the corrected model)")
    segs = desc.segments
    M = sum(s.mass for s in segs)
    pairs = [pad_pair(pg, f"{name}{i}", M if desc.rigid else s.mass, dt, desc.kappa.surface_friction,
                      desc.kappa.skin_compliance) for pg in pad_geoms for i, s in enumerate(segs)]
    return {"worldbody": built.xml, "tendon": tendons, "contact": "\n".join([excludes, *pairs]),
            "equality": equalities}


def pad_xml(pname, normal, pos):
    """Kinematic 40 x 40 mm pad, 8 mm thick, all edges rounded with 4 mm radius (box core + 4 capsules).
    Mocap bodies carry no velocity, so friction could not drag the object; the pad is a heavy body on three
    slide joints whose qpos/qvel are prescribed every step (Sim.set_pad)."""
    u, v = [a for a in range(3) if a != normal]
    hc = PAD / 2 - PAD_T
    size = np.zeros(3)
    size[u] = size[v] = hc
    size[normal] = PAD_T
    c = 'contype="0" conaffinity="0"'
    g = [f'<body name="{pname}" pos="{_v(pos)}" gravcomp="1">',
         '<inertial pos="0 0 0" mass="100" diaginertia="1 1 1"/>',
         *[f'<joint name="{pname}_{a}" type="slide" axis="{_v(_e(i))}"/>' for i, a in enumerate("xyz")],
         f'<geom name="{pname}_core" type="box" size="{_v(size)}" rgba=".2 .6 .9 .6" {c}/>']
    for along, across in ((u, v), (v, u)):
        for sgn in (1, -1):
            p0 = _e(across, sgn * hc) - _e(along, hc)
            p1 = _e(across, sgn * hc) + _e(along, hc)
            g.append(f'<geom name="{pname}_e{along}{sgn}" type="capsule" fromto="{_v(p0)} {_v(p1)}" size="{PAD_T}" '
                     f'rgba=".2 .6 .9 .6" {c}/>')
    g.append("</body>")
    return "\n".join(g)


def pad_geom_names(pname, normal):
    u, v = [a for a in range(3) if a != normal]
    return [f"{pname}_core"] + [f"{pname}_e{along}{sgn}" for along, _ in ((u, v), (v, u)) for sgn in (1, -1)]


def scene(desc: ContentDesc, origin, pads=(), table=True, table_xmin=-0.6, gravity=True, dt=None, mode=None,
          *, worldbody="", contact="", equality="", tendon="", pad_contact=True, table_contact=True, solver=None):
    """Build content plus MJCF child fragments (without enclosing section tags).

    pads: (name, normal_axis, position); table top is z = 0. Contact switches disable only direct
    content-pad/content-table contact, leaving caller contacts intact. solver is Newton/CG/PGS,
    with the iteration limits and tolerance from SOLVER. External tendons are appended and are
    excluded from content elastic-energy measurements.
    """
    mode = mode or JOINT_MODE
    dt = (BUSHING_DT if mode == "bushing" else DT) if dt is None else dt
    built, tendons, excludes, equalities = object_xml(desc, origin, mode=mode, omega_max=0.8 / dt, dt=dt,
                                                   table_contact=table_contact)
    solver_options = SOLVER
    if solver is not None:
        if solver not in ("Newton", "CG", "PGS"):
            raise ValueError(f"unknown MuJoCo solver: {solver!r}")
        solver_options = f'solver="{solver}" ' + SOLVER.split(maxsplit=1)[1]
    tx0, tx1 = table_xmin, 0.9
    table_geom = (f'<geom name="table" type="box" pos="{(tx0 + tx1) / 2:.6g} 0 -0.025" size="{(tx1 - tx0) / 2:.6g} 0.5 0.025" '
                  f'contype="1" conaffinity="0" condim="3" friction="{TABLE_FRICTION} 0.005 0.0001" {PASSIVE} rgba=".8 .8 .75 1"/>'
                  if table else "")
    g = "0 0 -9.81" if gravity else "0 0 0"
    xml = f"""<mujoco model="segment_content">
<option timestep="{dt}" integrator="implicitfast" {solver_options} cone="elliptic" gravity="{g}"><flag energy="enable"/></option>
<size memory="512M"/>
<visual><global offwidth="1280" offheight="800"/></visual>
<worldbody>
<light pos="0 0 2" dir="0 0 -1"/>
{table_geom}
{"".join(pad_xml(*p) for p in pads)}
{built.xml}
{worldbody}
</worldbody>
<contact>{excludes}
{_pairs(desc, pads, dt) if pad_contact else ""}
{contact}</contact>
<tendon>{tendons}
{tendon}</tendon>
<equality>{equalities}
{equality}</equality>
</mujoco>"""
    built.xml = xml
    return built


def _pairs(desc, pads, dt=DT):
    segs = desc.segments
    M = sum(s.mass for s in segs)
    out = []
    for pname, normal, _ in pads:
        for pg in pad_geom_names(pname, normal):
            for i, s in enumerate(segs):
                out.append(pad_pair(pg, f"s{i}", M if desc.rigid else s.mass, dt,
                                    desc.kappa.surface_friction, desc.kappa.skin_compliance))
    return "\n".join(out)


class Sim:
    """Compiled scene with index caches and measurements."""

    def __init__(self, built: Built, pad_names=()):
        self.built = built
        self.desc = built.desc
        self.m = mujoco.MjModel.from_xml_string(built.xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.pad_names = list(pad_names)
        self.pad_q = {p: np.array([m.jnt_qposadr[m.joint(f"{p}_{a}").id] for a in "xyz"]) for p in self.pad_names}
        self.pad_v = {p: np.array([m.jnt_dofadr[m.joint(f"{p}_{a}").id] for a in "xyz"]) for p in self.pad_names}
        self.pad_p0 = {p: m.body_pos[m.body(p).id].copy() for p in self.pad_names}
        self.pad_geoms = {p: {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(p).id} for p in self.pad_names}
        self._pad_geom_arr = {p: np.array(sorted(g)) for p, g in self.pad_geoms.items()}
        self._pad_geom_mask = {p: np.isin(np.arange(m.ngeom), list(g)) for p, g in self.pad_geoms.items()}
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if self.desc.rigid:
            self.seg_bodies = [m.body("obj").id]
        else:
            self.seg_bodies = [m.body(f"s{i}").id for i in range(len(self.desc.segments))]
        self.obj_bodies = list(self.seg_bodies) + ([m.body("mobile").id] if self.desc.mobile else [])
        self.obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.obj_bodies}
        self._obj_geom_mask = np.array([g in self.obj_geoms for g in range(m.ngeom)])
        self.mass = float(sum(m.body_mass[b] for b in self.obj_bodies))
        self.tinfo = built.tendons
        assert len(self.tinfo) == len(built.tendon_names), "content tendon metadata mismatch"
        self.content_tendons = np.array([m.tendon(name).id for name in built.tendon_names], int)
        self.t_kind = np.array([t.kind for t in self.tinfo]) if self.tinfo else np.array([])
        self.t_k = np.array([t.stiffness for t in self.tinfo])
        self.t_rest = np.array([t.rest for t in self.tinfo])
        self.t_lower = np.array([t.lower for t in self.tinfo])
        self.i_cc = np.where(self.t_kind == "center_comp")[0] if self.tinfo else np.array([], int)
        self.i_bend = np.where(self.t_kind == "bend")[0] if self.tinfo else np.array([], int)
        self._spring_joints = [jn for jn in range(m.njnt) if m.jnt_stiffness[jn] > 0
                               and m.jnt_type[jn] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.geom = JointGeometry(self.desc, BEND_ARM)
        self.bushings = Bushings(self.desc, self.seg_bodies, built.bushings) if built.bushings else None
        self.welds = np.array([m.equality(f"sj{ji}_w").id for ji in built.bushings], int)
        self._seg_qadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.seg_bodies], int)
        self.external_compaction_load = 0.0
        self.external_compaction_load_raw = 0.0
        mujoco.mj_forward(self.m, self.d)
        self._f6 = np.zeros(6)

    # ---------- kinematics
    def step(self):
        if self.bushings is None:
            mujoco.mj_step(self.m, self.d)
            return
        mujoco.mj_step1(self.m, self.d)
        external = self.d.xfrc_applied.copy()
        self.bushings.apply(self.m, self.d)
        mujoco.mj_step2(self.m, self.d)
        self.d.xfrc_applied[:] = external
        self.switch_welds(external)

    def compaction_load(self, external):
        """Opposing external normal loads, excluding internal contacts and constraint reactions.

        A one-sided support or uniform acceleration is not confinement. The difference between summed
        force magnitudes and their resultant is rotation invariant and vanishes for parallel loads.
        """
        forces = external[self.seg_bodies, :3]
        total, resultant = np.linalg.norm(forces, axis=1).sum(), forces.sum(0)
        contacts = self.d.contact
        inside = self._obj_geom_mask[contacts.geom]
        mask = (inside[:, 0] != inside[:, 1]) & (contacts.efc_address >= 0)
        if mask.any():
            fn = np.maximum(self.d.efc_force[contacts.efc_address[mask]], 0)
            signed = np.where(inside[mask, 1], fn, -fn)
            total += fn.sum()
            resultant += (signed[:, None] * contacts.frame[mask, :3]).sum(0)
        return max(0.0, .5 * (total - np.linalg.norm(resultant)))

    def switch_welds(self, external):
        """Lock connections with a segment compacted across them (WELD_ON / WELD_RELEASE hysteresis),
        at the relative pose of the new qpos; the change takes effect in the next step."""
        rho = self.bushings.orthogonal_compaction
        self.external_compaction_load_raw = self.compaction_load(external)
        fraction = self.m.opt.timestep / (COMPACTION_LOAD_TAU + self.m.opt.timestep)
        self.external_compaction_load += fraction * (self.external_compaction_load_raw - self.external_compaction_load)
        on = self.d.eq_active[self.welds].astype(bool)
        # Strain alone can be held by the welds themselves, especially under biaxial compression.
        # Once confinement is gone, stored elastic energy must be able to restore the block.
        want = np.where(on, rho >= WELD_RELEASE, rho >= WELD_ON) & (self.external_compaction_load > COMPACTION_LOAD_MIN)
        for i in np.flatnonzero(want & ~on):
            qa = self.d.qpos[self._seg_qadr[self.bushings.node_a[i]]:][:7]
            qb = self.d.qpos[self._seg_qadr[self.bushings.node_b[i]]:][:7]
            inv, rel, pos = np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_negQuat(inv, qa[3:])
            mujoco.mju_mulQuat(rel, inv, qb[3:])
            mujoco.mju_rotVecQuat(pos, qb[:3] - qa[:3], inv)
            self.m.eq_data[self.welds[i], 3:10] = np.r_[pos, rel]
        self.d.eq_active[self.welds] = want

    def set_pad(self, p, pos, vel):
        self.d.qpos[self.pad_q[p]] = np.asarray(pos) - self.pad_p0[p]
        self.d.qvel[self.pad_v[p]] = vel

    def pad_pos(self, p):
        return self.pad_p0[p] + self.d.qpos[self.pad_q[p]]

    def com(self):
        x = self.d.xipos[self.obj_bodies]
        w = self.m.body_mass[self.obj_bodies]
        return (w[:, None] * x).sum(0) / w.sum()

    def seg_pos(self):
        return self.d.xpos[self.seg_bodies].copy()

    def seg_speed(self):
        """Linear speed of each segment body origin (world)."""
        out = np.zeros(len(self.seg_bodies))
        v6 = np.zeros(6)
        for i, b in enumerate(self.seg_bodies):
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            out[i] = np.linalg.norm(v6[3:])
        return out

    def point_world(self, body, local):
        return self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ local

    def point_local(self, body, world):
        return self.d.xmat[body].reshape(3, 3).T @ (world - self.d.xpos[body])

    # ---------- forces
    def contact_forces(self):
        """World force exerted on each pad by the object, and on the object by the table / pads."""
        m, d = self.m, self.d
        pad_f = {p: np.zeros(3) for p in self.pad_names}
        table_on_obj = np.zeros(3)
        pad_contacts = 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            mujoco.mj_contactForce(m, d, i, self._f6)
            f2 = c.frame.reshape(3, 3).T @ self._f6[:3]      # force on geom2 from geom1
            for p, gs in self.pad_geoms.items():
                if g1 in gs and g2 in self.obj_geoms:
                    pad_f[p] -= f2
                    pad_contacts += 1
                elif g2 in gs and g1 in self.obj_geoms:
                    pad_f[p] += f2
                    pad_contacts += 1
            if g1 == self.table_geom and g2 in self.obj_geoms:
                table_on_obj += f2
            elif g2 == self.table_geom and g1 in self.obj_geoms:
                table_on_obj -= f2
        return pad_f, table_on_obj, pad_contacts

    def pad_normal_force(self, p, axis):
        """Sum of contact normal forces on pad p projected on world axis (fast, every step)."""
        d = self.d
        n = d.ncon
        if n == 0:
            return 0.0
        geom = d.contact.geom[:n]
        gs = self._pad_geom_mask[p]
        mask = gs[geom[:, 0]] | gs[geom[:, 1]]
        if not mask.any():
            return 0.0
        fn = d.efc_force[d.contact.efc_address[:n][mask]]
        return float(np.sum(fn * np.abs(d.contact.frame[:n, axis][mask])))

    def grip_anchors(self):
        """Material point on a most-loaded contact body; resolve 1% load ties by body id."""
        grouped = {}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            mujoco.mj_contactForce(self.m, self.d, i, self._f6)
            force = max(float(self._f6[0]), 0.0)
            if force <= 1e-8:
                continue
            for p, geoms in self.pad_geoms.items():
                other = c.geom2 if c.geom1 in geoms else c.geom1 if c.geom2 in geoms else -1
                if other not in self.obj_geoms:
                    continue
                b = int(self.m.geom_bodyid[other])
                total, point = grouped.get((p, b), (0.0, np.zeros(3)))
                grouped[p, b] = (total + force, point + force * c.pos)
        anchors = []
        for p in self.pad_names:
            candidates = [(b, f, point) for (pad, b), (f, point) in grouped.items() if pad == p]
            if candidates:
                peak = max(v[1] for v in candidates)
                # Equal load sharing is common across layers. A raw argmax can
                # switch reference layers under tiny timestep/solver differences.
                b, f, point = min((v for v in candidates if v[1] >= .99 * peak), key=lambda v: v[0])
                point = point / f
                anchors.append((p, b, self.point_local(b, point), point - self.pad_pos(p)))
        return anchors

    def contact_slip(self, anchors, normal):
        """Tangential material displacement at the captured pad contact points (metres)."""
        if not anchors or len(anchors) != len(self.pad_names):
            return None
        slips = []
        for p, b, local, offset in anchors:
            delta = self.point_world(b, local) - self.pad_pos(p) - offset
            delta[normal] = 0
            slips.append(float(np.linalg.norm(delta)))
        return max(slips)

    # ---------- energy and failure
    def elastic_energy(self):
        """Tendon springs (deadband-aware) plus hinge/slide joint springs (tree joints, rigid-body mobile)."""
        e = 0.0
        if self.bushings is not None:
            e += self.bushings.energy(self.d)
        if len(self.t_k):
            L = self.d.ten_length[self.content_tendons]
            dev = L - self.t_rest
            dev = np.where(self.t_kind == "center_comp", np.minimum(dev, 0.0), dev)
            e += float(0.5 * np.sum(self.t_k * dev * dev))
        m = self.m
        for jn in self._spring_joints:
            q = self.d.qpos[m.jnt_qposadr[jn]] - m.qpos_spring[m.jnt_qposadr[jn]]
            e += 0.5 * m.jnt_stiffness[jn] * q * q
        return e

    def kinetic_energy(self):
        mujoco.mj_energyVel(self.m, self.d)
        return float(self.d.energy[1])

    def seg_velocity(self):
        out = np.zeros((len(self.seg_bodies), 3))
        v6 = np.zeros(6)
        for i, b in enumerate(self.seg_bodies):
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
            out[i] = v6[3:]
        return out

    def failure(self, pad_vel=np.zeros(3), gripped=False):
        """README failure modes from segment poses (independent of how joints are realized)."""
        d = self.d
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return "non-finite state"
        for warning in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC):
            if d.warning[warning].number:
                return f"MuJoCo unstable state: {warning.name} (may have auto-reset)"
        if not self.desc.rigid:
            f = (self.bushings.check(d) if self.bushings is not None else
                 self.geom.check(d.xpos[self.seg_bodies], d.xmat[self.seg_bodies].reshape(-1, 3, 3)))
            if f:
                return f
        return None

    def velocity_diagnostic(self, pad_vel=np.zeros(3), gripped=False):
        """Speed is a diagnostic trigger, never a failure by itself (frozen README)."""
        v = self.seg_velocity()
        pad_speed = np.linalg.norm(v - (pad_vel if gripped else 0.0), axis=1)
        masses = np.array([self.m.body_mass[b] for b in self.seg_bodies])
        momentum = np.sum(masses[:, None] * v, axis=0)
        if self.desc.mobile:
            mobile_v = np.zeros(6)
            b = self.m.body("mobile").id
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, mobile_v, 0)
            momentum += self.m.body_mass[b] * mobile_v[3:]
        com_v = momentum / self.mass
        com_speed = np.linalg.norm(v - com_v, axis=1)
        return {"pad_relative_m_s": float(pad_speed.max()), "com_relative_m_s": float(com_speed.max())}

    def max_relative_angle_deg(self):
        if self.desc.rigid:
            return 0.0
        q = self.d.xquat
        out = 0.0
        for j in self.desc.joints:
            qa, qb = q[self.seg_bodies[j.a]], q[self.seg_bodies[j.b]]
            dot = abs(float(np.dot(qa, qb)))
            out = max(out, 2 * math.degrees(math.acos(min(1.0, dot))))
        return out

    def max_strain(self):
        if self.desc.rigid:
            return 0.0
        return self.geom.max_strain(self.d.xpos[self.seg_bodies])
