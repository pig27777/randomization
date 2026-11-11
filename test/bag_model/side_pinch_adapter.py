"""Side-pinch adapter for the unified YAM grasp platform (test/flex3d), four-part bag template.

Contents: frozen v1.3 segmented body, read-only through test/flex3d/segmented.py (all faces compress
through the released bushings, no plate stand-ins). Film: 20 taut-only spatial tendons from two sites on
the real palm (link_6, at the platform pinch frame +-10 mm along the finger width) to the box corners and
edge midpoints, shortened at the pinch; reactions load the arm. Pinch rule (r2): film at the grip point and
gather >= GATHER_MIN -> ropes engage; press depth > DEPTH_CRIT -> the face-layer segments between the
fingers are also pinned to the pinch frame. Each group releases on its own overload (pinch_coeff x current
grip force). Rope slack is assigned per film loop (four_part_pack_demo.loop_slack). Hold damping (relative contents-palm
velocity, only while a rope carries load) and closing-consumes-slack are on by default, switchable for comparison.

STATUS 2026-11-11: written and statically checked only. Running it needs (1) the YAM assets that
policy/yam_probe_robot.py loads, and (2) an optional approach direction in grasp_platform.attach_robot /
run, described in test/runs/2026-11-11/bag/side-pinch-demo-r2/REPORT.md. scene() refuses to start
without (2) instead of running a top-down pinch under a side-pinch name.

Usage after both are in place (repository root):
  python -B test/flex3d/angles.py --backend side_pinch_adapter --dt .0004 \
    --model-options '{"size":[0.3,0.2,0.04],"mass":0.5,"slack":[0.04,0.04],"face":"y"}' --output <new dir>
(test/bag_model must be on PYTHONPATH; category "hard" is not offered by the angles.py CLI and needs
angles.execute() with a dict.)
"""
from dataclasses import dataclass, replace
import inspect
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "test/flex3d"), str(HERE)]
import segmented as SEG  # noqa: E402  (also activates the frozen v1.3 package)
from grasp_platform import attach_robot, measure_contacts  # noqa: E402
import mj_adapter as A  # noqa: E402
from model import ANCHORS, build  # noqa: E402
from pinch import fragments  # noqa: E402
from four_part_pack_demo import BoxSurface, effective_slack, hold_damping_coef, loop_slack  # noqa: E402

GATHER_MIN, DEPTH_COEF, DEPTH_CRIT = 0.008, 1.0, 0.005   # r1/r2 initial values, unchanged
LOAD_TAU = 0.005
FREE = 10.0
ROPE_SOLREF = "0.005 1"           # no tendon damping; dissipation only in the limit constraint
SITE_HALF = 0.010                 # rope sites +-10 mm along the 20 mm finger width
PINCH_Z = 0.1347                  # platform pinch frame on link_6 (grasp_platform.MuJoCoSim)
CENTER = np.array([-.25, .04])    # platform pick position


@dataclass
class Config(SEG.Config):
    slack: tuple = (0.0, 0.0)          # bag outline margin along the contents length x / width y [m]
    face: str = "y"                    # "y": long side facing the arm (y min), "x": short side (x min)
    fixed_block: bool = False          # static block against the opposite face
    # default on (user feedback 2026-11-11: a pinched bag barely swings and hangs close to the fingers);
    # switches kept for comparison
    hold_damping: bool = True          # damp contents-palm relative velocity while a rope carries load
    close_consumes_slack: bool = True  # fingers take up (w - g)/2 of the slack of the loop they close along
    film_bend_k: float = 0.0           # film bending stiffness [N m/rad]; ~0 plastic film, >0 paper / padded mailers

    def __post_init__(self):
        super().__post_init__()
        if self.face not in ("x", "y"):
            raise ValueError("face must be 'x' or 'y'")
        s = np.asarray(self.slack, float)
        if s.shape != (2,) or not np.isfinite(s).all() or (s < 0).any():
            raise ValueError("slack needs two finite nonnegative lengths [m]")
        self.slack = tuple(float(v) for v in s)


# ---------- pure geometry and rules (no simulator; checked by test_side_pinch_adapter.py)
def approach_axes(face):
    """Inward face normal n (approach), closing direction c, finger-width direction w (world)."""
    if face == "y":
        return np.array([0., 1, 0]), np.array([1., 0, 0]), np.array([0., 0, 1])
    return np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])


def box_anchors(size):
    """8 corners + 12 edge midpoints in box-corner coordinates: [(name, xyz)]; names as four_part_pack_demo."""
    out = []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if (sx != 0) + (sy != 0) + (sz != 0) >= 2:
                    p = (np.array([sx, sy, sz]) + 1) / 2 * np.asarray(size)
                    out.append(("a" + "".join("nmp"[v + 1] for v in (sx, sy, sz)), p))
    return out


def gather_amount(slack, closing, depth):
    """Gatherable film along the closing direction in contact coordinates: slack along c + DEPTH_COEF x depth."""
    return float(np.hypot(slack[0] * closing[0], slack[1] * closing[1])) + DEPTH_COEF * depth


def film_at(p_box, size, slack):
    """Bag outline = contents box grown by the slack in length and width, thickness unchanged (assumption)."""
    return bool(-slack[0] <= p_box[0] <= size[0] + slack[0] and -slack[1] <= p_box[1] <= size[1] + slack[1]
                and 0 <= p_box[2] <= size[2])


def decide(film, gather, depth):
    if not film or gather < GATHER_MIN:
        return "miss"
    return "rope+connect" if depth > DEPTH_CRIT else "rope"


def rope_limits(sites_box, site_of, anchors, size, slack):
    """Per rope: |site - nearest surface point q| + surface geodesic(q -> anchor) + slack of the film loops the
    path runs along (four_part_pack_demo.loop_slack); slack = (length loop, width loop) margins.
    sites_box: {site name: xyz box frame}; site_of: site name per anchor; anchors: [(name, xyz)]."""
    surf = BoxSurface(size[2], length=size[0], width=size[1])
    cache, out = {}, []
    for (nm, a), s in zip(anchors, site_of):
        if s not in cache:
            q, dist = surf.nearest(np.asarray(sites_box[s], float))
            cache[s] = (q, dist, surf.geodesic_loops(q, [p for _, p in anchors]))
        q, dist, geo = cache[s]
        length, path_l, path_w = geo[len(out)]
        out.append(dist + length + loop_slack(path_l, path_w, slack))
    return np.array(out)


def hold_damping_forces(seg_vel, seg_mass, palm_vel, c, loaded):
    """Per-segment force -c (m_i / M)(v_i - v_palm); exactly zero unless a rope carries load."""
    if not loaded:
        return np.zeros_like(seg_vel)
    w = np.asarray(seg_mass, float) / np.sum(seg_mass)
    return -c * w[:, None] * (np.asarray(seg_vel) - np.asarray(palm_vel))


def fit_frame(rest, current):
    """Rotation R and translation t with current ~ R rest + t (Kabsch); rest in the object frame."""
    pc, qc = rest.mean(0), current.mean(0)
    U, _, Vt = np.linalg.svd((rest - pc).T @ (current - qc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


# ---------- scene
def content_xml_parts(desc, cfg, size):
    """Anchor sites (inside the segment bodies), rope tendons, side-pinch connects and the optional block."""
    n, c, w = approach_axes(cfg.face)
    anchors = box_anchors(size)
    seg_rest = np.array([s.pos for s in desc.segments])
    sites, tendons = {}, []
    for nm, p in anchors:
        p_obj = p - np.asarray(size) / 2
        if desc.rigid:
            body, local = "obj", p_obj
        else:
            i = int(np.argmin(np.linalg.norm(seg_rest - p_obj, axis=1)))
            body, local = f"s{i}", p_obj - seg_rest[i]
        sites.setdefault(body, []).append(f'<site name="{nm}" pos="{local[0]:.6g} {local[1]:.6g} {local[2]:.6g}" size=".003"/>')
        dw, dc = p_obj @ w, p_obj @ c                  # relative to the grip point: face centre line, mid thickness
        side = "rope_ga" if (dw < -1e-9 or (abs(dw) < 1e-9 and dc < 0)) else "rope_gb"
        tendons.append(f'<spatial name="r{nm}" limited="true" range="0 {FREE}" solreflimit="{ROPE_SOLREF}" '
                       f'width=".0015" rgba=".1 .1 .1 1"><site site="{side}"/><site site="{nm}"/></spatial>')
    face = [] if desc.rigid else [i for i, s in enumerate(desc.segments)
                                  if abs((s.pos + np.asarray(size) / 2) @ n - s.half @ np.abs(n)) < 1e-9]
    connects = "".join(f'<connect name="side_{i}" body1="pinch_frame" body2="s{i}" anchor="0 0 0" active="false" '
                       'solref="0.005 1" solimp="0.99 0.99 0.001"/>' for i in face)
    block = ""
    if cfg.fixed_block:
        far = np.r_[CENTER, 0.0] + n * (np.asarray(size) @ np.abs(n) / 2 + .026)
        half = np.abs(c) * .25 + np.abs(n) * .025 + np.array([0, 0, .5 * size[2] + .02])
        block = (f'<geom name="side_block" type="box" pos="{far[0]:.6g} {far[1]:.6g} {half[2]:.6g}" '
                 f'size="{half[0]:.6g} {half[1]:.6g} {half[2]:.6g}" contype="2" conaffinity="1" rgba=".45 .45 .5 1"/>')
    return anchors, sites, "".join(tendons), connects, face, block


class RopeSim(SEG.SegmentSim):
    """Segment sim with anchor sites, palm rope sites and tendons inserted before compilation."""

    def __init__(self, built):
        root = ET.fromstring(built.xml)
        for body, items in built.rope_sites.items():
            node = root.find(f".//body[@name='{body}']")
            for s in items:
                node.append(ET.fromstring(s))
        link = root.find(".//body[@name='link_6']")
        for nm, x in (("rope_ga", -SITE_HALF), ("rope_gb", SITE_HALF)):   # local x = finger width (to verify)
            ET.SubElement(link, "site", name=nm, pos=f"{x} 0 {PINCH_Z}", size=".003")
        tendon = root.find("tendon")
        if tendon is None:
            tendon = ET.SubElement(root, "tendon")
        for t in ET.fromstring(f"<t>{built.rope_tendons}</t>"):
            tendon.append(t)
        built.xml = ET.tostring(root, encoding="unicode")
        super().__init__(built)
        m = self.m
        self.ropes = np.array([m.tendon(f"r{nm}").id for nm, _ in built.rope_anchors])
        self.rope_index = np.full(m.ntendon, -1)
        self.rope_index[self.ropes] = np.arange(len(self.ropes))
        self.rope_damping_c, self.rope_loaded, self.gate_toggles = 0.0, False, 0
        self.bend_k, self.q_pinch = 0.0, None
        self.palm_body = m.body("link_6").id

    def content_frame(self):
        d = self.d
        if self.desc.rigid:
            return d.xmat[self.seg_bodies[0]].reshape(3, 3), d.xpos[self.seg_bodies[0]].copy()
        return fit_frame(np.array([s.pos for s in self.desc.segments]), d.xpos[self.seg_bodies])

    def rope_tension(self):
        d, nf = self.d, self.d.nefc
        T = np.zeros(len(self.ropes))
        rows = np.flatnonzero(d.efc_type[:nf] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
        rows = rows[self.rope_index[d.efc_id[rows]] >= 0]
        T[self.rope_index[d.efc_id[rows]]] = d.efc_force[rows]
        return T

    def step(self):
        # The platform has already written its own xfrc_applied (gathering spring) for this step.
        if self.rope_damping_c:
            d, v6 = self.d, np.zeros(6)
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.palm_body, v6, 0)
            palm_v = v6[3:].copy()
            vel = []
            for b in self.seg_bodies:
                mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
                vel.append(v6[3:] + np.cross(v6[:3], d.xipos[b] - d.xpos[b]))
            f = hold_damping_forces(np.array(vel), self.m.body_mass[self.seg_bodies], palm_v,
                                    self.rope_damping_c, self.rope_loaded)
            d.xfrc_applied[self.seg_bodies, :3] += f
            d.xfrc_applied[self.palm_body, :3] -= f.sum(0)
        if self.bend_k and self.rope_loaded and self.q_pinch is not None:
            # film bending stiffness: -k * contents rotation since the pinch (fitted frame), reaction on the palm
            d, rv, qr = self.d, np.zeros(3), np.zeros(4)
            R = self.content_frame()[0]
            mujoco.mju_mat2Quat(qr, (R @ self.R_pinch.T).reshape(9))
            mujoco.mju_quat2Vel(rv, qr, 1.0)
            tau = -self.bend_k * rv
            share = self.m.body_mass[self.seg_bodies] / self.m.body_mass[self.seg_bodies].sum()
            d.xfrc_applied[self.seg_bodies, 3:] += share[:, None] * tau
            d.xfrc_applied[self.palm_body, 3:] -= tau
        super().step()
        loaded = bool(self.rope_tension().max(initial=0.0) > 0.0)
        self.gate_toggles += loaded != self.rope_loaded
        self.rope_loaded = loaded


def scene(cfg):
    if "approach" not in inspect.signature(attach_robot).parameters:
        raise RuntimeError("side pinch needs the optional approach direction in grasp_platform.attach_robot/run "
                           "(not landed; see side-pinch-demo-r2 REPORT, platform change section)")
    anchor = ANCHORS[cfg.category]
    kappa = replace(anchor, pinch_coeff=cfg.pinch_coeff,
                    size=cfg.size if cfg.size is not None else anchor.size,
                    mass=cfg.mass if cfg.mass is not None else anchor.mass)
    desc = build(kappa, omega_max=.8 / cfg.dt, geometry=cfg.category if cfg.size is not None else None)
    size = np.array(kappa.size)
    _, top_equality = fragments(desc)                 # keeps SegmentPinch's top-layer connects available
    anchors, sites, tendons, connects, face, block = content_xml_parts(desc, cfg, size)
    built = A.scene(desc, [*CENTER, .002 + size[2] / 2], dt=cfg.dt, equality=top_equality + connects,
                    worldbody=block)
    root = ET.fromstring(built.xml)
    table = root.find(".//geom[@name='table']")
    table.attrib.update(pos='0 0 -.025', size='1 .22 .025', contype='1', conaffinity='6',
                        friction='.8 .005 .0001', solref=f'{cfg.contact_time} 1', solimp='.99 .99 .001')
    built.xml = ET.tostring(root, encoding='unicode')
    built.cfg, built.rest_top = cfg, size[2] + .002
    built.rope_anchors, built.rope_sites, built.rope_tendons, built.face_segments = anchors, sites, tendons, face
    n, c, w = approach_axes(cfg.face)
    face_point = np.r_[CENTER, .002 + size[2] / 2] - n * (size @ np.abs(n) / 2)
    robot = attach_robot(cfg, built, RopeSim, CENTER, size[2] + .002, kappa.size, kappa.mass,
                         approach=dict(direction=n, closing=c, contact_point=face_point))
    robot.sim.pad_geoms = {side: robot.all_finger_sets[j] for j, side in enumerate(('left', 'right'))}
    robot.side = dict(n=n, c=c, w=w, size=size, open_w=0.0)
    if cfg.hold_damping:
        robot.sim.rope_damping_c = hold_damping_coef(kappa.mass, size, cfg.face)
    robot.sim.bend_k = cfg.film_bend_k
    return robot


def measure(robot):
    out = measure_contacts(robot, target_geoms=robot.sim.obj_geoms)
    if max(out["contact_magnitude_each_N"]) > .05:     # opening w when the fingertips meet the film/face
        robot.side["open_w"] = max(robot.side["open_w"], robot.gap())
    return out


class SidePinch:
    """Two link groups: ropes (film held) and connects (contents between the fingers)."""

    def __init__(self, robot, cfg):
        self.robot, self.cfg, self.sim = robot, cfg, robot.sim
        self.active, self.load, self.events = None, 0.0, []
        self.ropes_on, self.pins = False, []
        self.lp = dict(rope=0.0, pin=0.0)
        rest = np.array([s.pos for s in self.sim.desc.segments])
        self.rest = rest if not self.sim.desc.rigid else None

    def frame(self):
        d = self.sim.d
        if self.sim.desc.rigid:
            b = self.sim.seg_bodies[0]
            return d.xmat[b].reshape(3, 3), d.xpos[b].copy()
        return fit_frame(self.rest, d.xpos[self.sim.seg_bodies])

    def to_box(self, p):
        R, t = self.frame()
        return R.T @ (np.asarray(p) - t) + self.robot.side["size"] / 2

    def try_bind(self, tips):
        m, d, side, cfg = self.sim.m, self.sim.d, self.robot.side, self.cfg
        grip = np.asarray(tips).mean(0)
        pb = self.to_box(grip)
        size = side["size"]
        inside = np.minimum(pb, size - pb)
        depth = float(max(0.0, inside.min())) if (inside >= 0).all() else 0.0
        film = film_at(pb, size, cfg.slack)
        gather = gather_amount(cfg.slack, side["c"], depth)
        mode = decide(film, gather, depth)
        g = self.robot.gap()
        info = dict(t=float(d.time), event="decision", mode=mode, depth_mm=1e3 * depth, gather_mm=1e3 * gather,
                    film=film, grip_point_box_mm=(1e3 * pb).tolist(), open_w_mm=1e3 * side["open_w"], gap_mm=1e3 * g)
        if mode == "miss":
            self.events.append(info)
            return False
        sl = effective_slack(cfg.slack, side["c"], side["open_w"], g, cfg.close_consumes_slack)
        site_box = {s: self.to_box(d.site_xpos[m.site(s).id]) for s in ("rope_ga", "rope_gb")}
        site_of = [m.site(int(m.wrap_objid[m.tendon_adr[t]])).name for t in self.sim.ropes]
        lim = rope_limits(site_box, site_of, self.sim.built.rope_anchors, size, sl)
        m.tendon_range[self.sim.ropes, 1] = lim
        self.ropes_on = True
        self.sim.R_pinch, self.sim.q_pinch = self.sim.content_frame()[0], True
        info.update(slack_used_mm=(1e3 * sl).tolist(),
                    rope_limits_mm={nm: round(1e3 * v, 1) for (nm, _), v in zip(self.sim.built.rope_anchors, lim)})
        if mode == "rope+connect":
            touching = self.touching_face_segments()
            frame = m.body("pinch_frame").id
            for i in touching:
                e, b = m.equality(f"side_{i}").id, self.sim.seg_bodies[i]
                m.eq_data[e, :3] = d.xmat[frame].reshape(3, 3).T @ (grip - d.xpos[frame])
                m.eq_data[e, 3:6] = d.xmat[b].reshape(3, 3).T @ (grip - d.xpos[b])
                d.eq_active[e] = True
                self.pins.append(e)
            info["pinned_segments"] = touching
        self.active = list(self.pins) or [-1]
        self.events.append(info)
        return True

    def touching_face_segments(self):
        d, fingers = self.sim.d, set().union(*self.robot.all_finger_sets)
        seg_geom = {g: i for i, b in enumerate(self.sim.seg_bodies)
                    for g in range(self.sim.m.ngeom) if self.sim.m.geom_bodyid[g] == b}
        out = set()
        for c in d.contact[:d.ncon]:
            for g, h in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in fingers and h in seg_geom and seg_geom[h] in self.sim.built.face_segments:
                    out.add(seg_geom[h])
        return sorted(out)

    def group_forces(self):
        d, m = self.sim.d, self.sim.m
        f_pin = np.zeros(3)
        if self.pins:
            rows = np.flatnonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            rows = rows[np.isin(d.efc_id[rows], self.pins)]
            f_pin = d.efc_force[rows].reshape(-1, 3).sum(0)
        f_rope = np.zeros(3)
        if self.ropes_on:
            T = self.sim.rope_tension()
            a = d.site_xpos[[int(m.wrap_objid[m.tendon_adr[t] + 1]) for t in self.sim.ropes]]
            g = d.site_xpos[[int(m.wrap_objid[m.tendon_adr[t]]) for t in self.sim.ropes]]
            u = (a - g) / np.linalg.norm(a - g, axis=1, keepdims=True)
            f_rope = u.T @ T
        return dict(pin=float(np.linalg.norm(f_pin)), rope=float(np.linalg.norm(f_rope)))

    def check_load(self, dt, grip_force):
        """Each group against pinch_coeff x current grip force; True once nothing holds (platform drops its spring)."""
        F, limit = self.group_forces(), self.cfg.pinch_coeff * grip_force
        for g, on in (("pin", bool(self.pins)), ("rope", self.ropes_on)):
            if not on:
                continue
            self.lp[g] += dt / (LOAD_TAU + dt) * (F[g] - self.lp[g])
            if self.lp[g] > limit:
                self.release_group(g, "overload", filtered_force=self.lp[g], limit=limit)
        self.load = max(self.lp["pin"] if self.pins else 0.0, self.lp["rope"] if self.ropes_on else 0.0)
        return self.active is None

    def release_group(self, g, reason, **evidence):
        m, d = self.sim.m, self.sim.d
        if g == "pin" and self.pins:
            d.eq_active[self.pins] = False
            self.pins = []
        if g == "rope" and self.ropes_on:
            m.tendon_range[self.sim.ropes, 1] = FREE
            self.ropes_on = False
        self.events.append(dict(t=float(d.time), event=f"{g}_{reason}", **evidence))
        self.active = (list(self.pins) or [-1]) if (self.pins or self.ropes_on) else None

    def release(self, reason="open", **evidence):
        for g in ("pin", "rope"):
            self.release_group(g, reason, **evidence)


def attachment(robot, cfg):
    return SidePinch(robot, cfg)


def check_state(sim):
    failure, health = SEG.check_state(sim)
    T = sim.rope_tension()
    health.update(rope_tension_max_N=float(T.max(initial=0.0)), ropes_taut=int((T > 0).sum()),
                  rope_loaded=sim.rope_loaded, hold_damping_gate_toggles=sim.gate_toggles)
    return failure, health


def summarize(trace):
    out = SEG.summarize(trace)
    out.update({k: max((row[k] for row in trace), default=None)
                for k in ("rope_tension_max_N", "ropes_taut", "hold_damping_gate_toggles")})
    return out


def dimensions(sim):
    out = SEG.dimensions(sim)
    out.update(ropes=len(sim.ropes), face_segments=len(sim.built.face_segments))
    return out


def source_files():
    return {"segmented": ROOT / "test/flex3d/segmented.py", "rope_geometry": HERE / "four_part_pack_demo.py"}
