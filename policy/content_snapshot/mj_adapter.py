"""MuJoCo 3.17 adapter: ContentDesc -> MJCF, plus scene (table, kinematic pads) and measurements.

Each segment is a free body. The 6-DOF elastic joint of a pair is a truss of spatial tendons:
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

DT = 0.00025
OMEGA_MAX = 0.8 / DT             # auto-lock limit for explicit tendon springs
PAD_T = 0.004                    # pad half thickness = edge radius (4 mm rounded edges)
PAD_FRICTION = 0.8
TABLE_FRICTION = 0.5
SEG_FRICTION = 0.5
CONTACT_K = 3e5                  # N/m, static stiffness per contact point, independent of body mass
CONTACT_D = 0.99
BEND_ARM = 0.9                   # edge tendons at 90 % of the half cell
SHEAR_OFFSET = 0.5               # shear tendons at 50 % of the half cell (gives twist)


def contact_k(mass):
    """MuJoCo's static contact stiffness is mass*d*k/(1-d) for solref (-k, -b); scale k by the body mass
    so every object sees about CONTACT_K, with critical damping of the normal direction."""
    k = CONTACT_K * (1 - CONTACT_D) / (CONTACT_D * mass)
    return k, 2 * math.sqrt(k)


def contact_attr(mass):
    """Object geom contact parameters. Table uses solmix 0, so the object geom sets the pair."""
    k, b = contact_k(mass)
    return f'solref="{-k:.6g} {-b:.6g}" solimp="{CONTACT_D} {CONTACT_D} 0.001" solmix="1"'


def pad_pair(pad_geom, obj_geom, mass):
    """Explicit pad-object pair: friction rows are velocity-level with time constant 1/(d*b); for heavy
    bodies the normal b gives ~8 ms and the object creeps along the pads, so friction gets b >= 1000."""
    k, b = contact_k(mass)
    return (f'<pair geom1="{pad_geom}" geom2="{obj_geom}" condim="3" friction="{PAD_FRICTION} {PAD_FRICTION} 0.005 '
            f'0.0001 0.0001" solref="{-k:.6g} {-b:.6g}" solreffriction="{-k:.6g} {-max(b, 1000.0):.6g}" '
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


def object_xml(desc: ContentDesc, origin, name="s"):
    """Bodies, tendons, excludes and equalities of one object placed with its envelope center at origin."""
    origin = np.asarray(origin, float)
    k = desc.kappa
    segs = desc.segments
    contact = lambda mass: f'contype="2" conaffinity="3" condim="3" friction="{SEG_FRICTION} 0.005 0.0001" {contact_attr(mass)}'
    bodies, tendons, excludes, equalities, infos, realized = [], [], [], [], [], []
    infos_mob = []
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

    rgba = {False: ".85 .6 .4 1", True: ".5 .5 .6 1"}
    for i, s in enumerate(segs):
        b = [f'<body name="{name}{i}" pos="{_v(origin + s.pos)}"><freejoint name="{name}{i}_free"/>',
             f'<inertial pos="{_v(s.com)}" mass="{s.mass:.8g}" diaginertia="{_v(s.inertia)}"/>',
             f'<geom name="{name}{i}" type="box" size="{_v(s.half)}" mass="0" rgba="{rgba[sum(s.idx) % 2 == 1]}" {contact(s.mass)}/>',
             f'<site name="{name}{i}_c" pos="0 0 0" size="0.001"/>']
        b.append("</body>")
        bodies.append("\n".join(b))
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

    sites = {}   # (body, pos tuple) -> site name, appended to bodies lazily

    def site(body, pos):
        key = (body, tuple(np.round(pos, 9)))
        if key not in sites:
            sites[key] = f"{name}{body}_t{len(sites)}"
        return sites[key]

    def tendon(tname, s1, s2, stiff, springlength, damping, lower=None, upper=None):
        lim = ""
        if lower is not None:
            lim = f' limited="true" range="{lower:.8g} {upper:.8g}" solreflimit="0.001 1" solimplimit="0.95 0.99 0.0005"'
        tendons.append(f'<spatial name="{tname}" stiffness="{stiff:.8g}" springlength="{springlength}" '
                       f'damping="{damping:.8g}"{lim}><site site="{s1}"/><site site="{s2}"/></spatial>')

    def too_stiff(j):
        """Auto-lock on the realized truss: edge tendons add k_bend / arm^2 of axial stiffness."""
        half = segs[j.a].half
        o = [a for a in range(3) if a != j.axis]
        axial = j.k_comp + sum(j.k_bend[bi] / (BEND_ARM * half[along]) ** 2
                               for bi, along in enumerate((o[1], o[0])))
        return math.sqrt(axial / j.m_eff) > OMEGA_MAX

    n_fixed = 0
    for ji, j in enumerate(desc.joints):
        if j.fixed or too_stiff(j):
            n_fixed += 1
            equalities.append(f'<weld body1="{name}{j.a}" body2="{name}{j.b}" solref="0.001 1"/>')
            realized.append(None)
            continue
        ax, d = j.axis, j.rest
        o = [a for a in range(3) if a != ax]
        half = segs[j.a].half
        zeta = j.damping_ratio
        ca, cb = f"{name}{j.a}_c", f"{name}{j.b}_c"
        # bending: about o[0] uses edge tendons offset along o[1], and vice versa
        bend_axial = 0.0
        for bi, (about, along) in enumerate(((o[0], o[1]), (o[1], o[0]))):
            arm = BEND_ARM * half[along]
            kb = j.k_bend[bi]
            kt = kb / (2 * arm * arm)
            ct = zeta * math.sqrt(kb * j.i_eff[bi]) / (arm * arm)
            lower = min(_bend_lower_length(d, arm, j.lock_bend), d * (1 - j.lock_comp) - 1e-5)
            for sgn in (1, -1):
                off = _e(along, sgn * arm)
                tn = f"{name}j{ji}_b{bi}{'p' if sgn > 0 else 'n'}"
                tendon(tn, site(j.a, off), site(j.b, off), kt, f"{d:.8g} {d:.8g}", ct, lower, 10 * d)
                infos.append(TendonInfo("bend", ji, kt, d, lower, d))
            bend_axial += 2 * kt
        k_sym = max(j.k_tens - bend_axial, 0.0)
        k_cmp = max(j.k_comp - k_sym - bend_axial, 0.0)
        m = j.m_eff
        tendon(f"{name}j{ji}_cs", ca, cb, k_sym, f"{d:.8g} {d:.8g}", 2 * zeta * math.sqrt(k_sym * m))
        infos.append(TendonInfo("center_sym", ji, k_sym, d, cell=d))
        tendon(f"{name}j{ji}_cc", ca, cb, k_cmp, f"{d:.8g} {100 * d:.8g}",
               2 * zeta * math.sqrt(max(k_cmp, 1e-9) * m), d * (1 - j.lock_comp), 10 * d)
        infos.append(TendonInfo("center_comp", ji, k_cmp, d, d * (1 - j.lock_comp), d))
        # shear along o[si]: two tendons offset along the other axis
        twist = 0.0
        for si, (along, off_ax) in enumerate(((o[0], o[1]), (o[1], o[0]))):
            ell = half[along]
            r = SHEAR_OFFSET * half[off_ax]
            ks = j.k_shear[si] / 2
            for sgn in (1, -1):
                pa = _e(ax, d / 2) - _e(along, ell) + _e(off_ax, sgn * r)
                pb = _e(ax, -d / 2) + _e(along, ell) + _e(off_ax, sgn * r)
                tn = f"{name}j{ji}_s{si}{'p' if sgn > 0 else 'n'}"
                tendon(tn, site(j.a, pa), site(j.b, pb), ks, f"{2 * ell:.8g} {2 * ell:.8g}",
                       2 * zeta * math.sqrt(ks * m))
                infos.append(TendonInfo("shear", ji, ks, 2 * ell))
            twist += 2 * ks * r * r
        realized.append({"comp": k_sym + k_cmp + bend_axial, "tens": k_sym + bend_axial, "twist": twist})

    mob_xml = [t for t in tendons if 'name="mob_t' in t]
    tendons = [t for t in tendons if 'name="mob_t' not in t] + mob_xml
    # insert sites into their bodies
    per_body = {}
    for (body, pos), sn in sites.items():
        per_body.setdefault(body, []).append(f'<site name="{sn}" pos="{_v(pos)}" size="0.0005"/>')
    for body, lines in per_body.items():
        center = f'<site name="{name}{body}_c" pos="0 0 0" size="0.001"/>'
        bodies[body] = bodies[body].replace(center, center + "\n" + "\n".join(lines), 1)
    for a, b in desc.exclude:
        excludes.append(f'<exclude body1="{name}{a}" body2="{name}{b}"/>')
    built = Built("\n".join(bodies), desc, infos + infos_mob, realized, n_fixed)
    return built, "\n".join(tendons), "\n".join(excludes), "\n".join(equalities)


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


def scene(desc: ContentDesc, origin, pads=(), table=True, table_xmin=-0.6, gravity=True, dt=DT):
    """pads: list of (name, normal_axis, position). Table top is z = 0."""
    built, tendons, excludes, equalities = object_xml(desc, origin)
    tx0, tx1 = table_xmin, 0.9
    table_geom = (f'<geom name="table" type="box" pos="{(tx0 + tx1) / 2:.6g} 0 -0.025" size="{(tx1 - tx0) / 2:.6g} 0.5 0.025" '
                  f'contype="1" conaffinity="0" condim="3" friction="{TABLE_FRICTION} 0.005 0.0001" {PASSIVE} rgba=".8 .8 .75 1"/>'
                  if table else "")
    g = "0 0 -9.81" if gravity else "0 0 0"
    xml = f"""<mujoco model="segment_content">
<option timestep="{dt}" integrator="implicitfast" solver="Newton" iterations="50" ls_iterations="20"
        tolerance="1e-9" cone="elliptic" gravity="{g}"><flag energy="enable"/></option>
<size memory="512M"/>
<visual><global offwidth="1280" offheight="800"/></visual>
<worldbody>
<light pos="0 0 2" dir="0 0 -1"/>
{table_geom}
{"".join(pad_xml(*p) for p in pads)}
{built.xml}
</worldbody>
<contact>{excludes}
{_pairs(desc, pads)}</contact>
<tendon>{tendons}</tendon>
<equality>{equalities}</equality>
</mujoco>"""
    built.xml = xml
    return built


def _pairs(desc, pads):
    segs = desc.segments
    M = sum(s.mass for s in segs)
    out = []
    for pname, normal, _ in pads:
        for pg in pad_geom_names(pname, normal):
            for i, s in enumerate(segs):
                out.append(pad_pair(pg, f"s{i}", M if desc.rigid else s.mass))
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
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if self.desc.rigid:
            self.seg_bodies = [m.body("obj").id]
        else:
            self.seg_bodies = [m.body(f"s{i}").id for i in range(len(self.desc.segments))]
        self.obj_bodies = list(self.seg_bodies) + ([m.body("mobile").id] if self.desc.mobile else [])
        self.obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.obj_bodies}
        self.mass = float(sum(m.body_mass[b] for b in self.obj_bodies))
        self.tinfo = built.tendons
        assert len(self.tinfo) == m.ntendon or self.desc.rigid
        self.t_kind = np.array([t.kind for t in self.tinfo]) if self.tinfo else np.array([])
        self.t_k = np.array([t.stiffness for t in self.tinfo])
        self.t_rest = np.array([t.rest for t in self.tinfo])
        self.t_lower = np.array([t.lower for t in self.tinfo])
        self.i_cc = np.where(self.t_kind == "center_comp")[0] if self.tinfo else np.array([], int)
        self.i_bend = np.where(self.t_kind == "bend")[0] if self.tinfo else np.array([], int)
        self.mob_qpos = ([m.jnt_qposadr[m.joint(f"mob_{a}").id] for a in "xyz"]
                         if self.desc.mobile and self.desc.rigid else [])
        self.geom = JointGeometry(self.desc, BEND_ARM)
        mujoco.mj_forward(self.m, self.d)
        self._f6 = np.zeros(6)

    # ---------- kinematics
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
        gs = self._pad_geom_arr[p]
        mask = np.isin(geom[:, 0], gs) | np.isin(geom[:, 1], gs)
        if not mask.any():
            return 0.0
        fn = d.efc_force[d.contact.efc_address[:n][mask]]
        return float(np.sum(fn * np.abs(d.contact.frame[:n, axis][mask])))

    # ---------- energy and failure
    def elastic_energy(self):
        if self.desc.rigid:
            e = 0.0
        else:
            L = self.d.ten_length
            dev = L - self.t_rest
            comp = self.t_kind == "center_comp"
            dev = np.where(comp, np.minimum(dev, 0.0), dev)
            e = float(0.5 * np.sum(self.t_k * dev * dev))
        if self.mob_qpos:
            q = self.d.qpos[self.mob_qpos]
            e += 0.5 * self.desc.mobile.stiffness * float(q @ q)
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
        if not self.desc.rigid:
            f = self.geom.check(d.xpos[self.seg_bodies], d.xmat[self.seg_bodies].reshape(-1, 3, 3))
            if f:
                return f
        v = self.seg_velocity() - (pad_vel if gripped else 0.0)
        sp = np.linalg.norm(v, axis=1)
        if sp.max() > 1.0:
            return f"flung: segment speed {sp.max():.2f} m/s relative to the pads"
        return None

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
