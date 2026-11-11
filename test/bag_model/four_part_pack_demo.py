"""Exploratory demo: one template, four parts -- contents, conforming surface, slack film, gripper.

Contents: 300 x 200 mm, 4 x 2 segments in one layer, bend + twist between neighbours, and a plate per
segment on the pressed face (top face for the top pinch, the gripped side face for the side pinch) sliding
along the face normal with a stiffening spring. Surface: friction only. Film: taut-only spatial tendons from
two gripper points to anchors on the contents, present from the start with a huge limit and shortened at
the pinch. Gripper: actuated palm with two force-limited fingers; no arm. Not a formal model.

Sets:
  top   r1 cases: palm comes down on the top face, fingers close along the width.
  side  r2 cases: palm approaches a side face horizontally, fingers close along that face's length.

r2 changes for both sets: no tendon damping (dissipation only in the limit constraint, solreflimit ratio
1); a pinch always activates the ropes, and a press deeper than DEPTH_CRIT additionally pins the plates
between the fingers; each link group has its own capacity (pinch coefficient x grip force).

Usage (repository root):
  MUJOCO_GL=egl FFMPEG=<ffmpeg> python -B test/bag_model/four_part_pack_demo.py --set side --output test/runs/<date>/bag/<run-id>
"""
import argparse
import heapq
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slack_rope_demo import FFMPEG, FPS, RATE, VH, VP9, VW, overview, smooth, write_video  # noqa: E402

L, W, H = 0.30, 0.20, 0.04       # H: thickness of the top-pinch (r1) contents; side cases set their own
NX, NY = 4, 2
SX, SY = L / NX, W / NY
HB, HP = 0.03, 0.01              # top set: segment body / top plate thickness; HP is also the side plate thickness
DT = 0.0005
PLATE_FRAC = 0.2                 # share of segment mass in its plate
TABLE_MU, PAD_MU = 0.5, 0.8      # as v1.3 (mj_adapter TABLE_FRICTION / PAD_FRICTION)
# r2: tendon damping acts on slack ropes too (r1 dragged an unpinched package 67-108 mm), so it is 0 and
# the only dissipation is the damping ratio of the limit constraint, active only while a rope is taut.
ROPE_DAMP, ROPE_SOLREF, FREE = 0.0, "0.005 1", 10.0
GATHER_MIN, DEPTH_COEF, DEPTH_CRIT, GRIP_F = 0.008, 1.0, 0.005, 15.0   # user's initial values
PRESS_F, OPEN, PAD_W, PAD_T, PAD_H = 20.0, 0.095, 0.02, 0.01, 0.06
LP_TAU = 0.005
# k, c: bend/twist stiffness [N m/rad] and damping between neighbours; k0 [N/m], dmax [m]: top plate spring
# F = k0 d / (1 - d/dmax). k0 = v1.3 anchor squeeze_stiffness_z / 8 segments, dmax = lock_strain_z * H.
# sq_y [N/m], lock_y: v1.3 anchor squeeze_stiffness_y and lock_strain_y, the same rule for the side plates
# (k0 = sq_y / plates on the face, dmax = lock_y * depth along the press).
CONTENTS = {
    "hard": None,
    "bendable": dict(k=0.25, c=0.0033, k0=37.5, dmax=0.008, sq_y=500.0, lock_y=0.20),
    "compressible": dict(k=0.5, c=0.0047, k0=25.0, dmax=0.010, sq_y=100.0, lock_y=0.25),
}
YAW_K, YAW_C, PLATE_C = 5.0, 0.01, 2.0
LOOSE = (0.04, 0.04)
# top set (r1): id, contents, slack (length x, width y) [m], grip xy [m], mass [kg], pinch coefficient, bend scale
TOP_CASES = [
    ("1", "hard", (0, 0), (0, 0), 0.5, 1.0, 1), ("2", "hard", LOOSE, (0, 0), 0.5, 1.0, 1),
    ("3", "bendable", (0, 0), (0, 0), 0.5, 1.0, 1), ("4", "bendable", LOOSE, (0, 0), 0.5, 1.0, 1),
    ("5", "compressible", (0, 0), (0, 0), 0.5, 1.0, 1), ("6", "hard", LOOSE, (0.06, 0), 0.5, 1.0, 1),
    ("7", "hard", LOOSE, (0, 0), 1.5, 0.5, 1), ("8", "hard", LOOSE, (L / 2 + 0.02, 0), 0.5, 1.0, 1),
    ("3k3", "bendable", (0, 0), (0, 0), 0.5, 1.0, 3), ("4k3", "bendable", LOOSE, (0, 0), 0.5, 1.0, 3),
]
# side set (r2): id, contents, thickness [m], mass [kg], slack (x, y) [m], gripped face ("y": long side at
# y = -W/2, "x": short side at x = -L/2), fixed block behind the opposite face
SIDE_CASES = [
    ("1", "hard", 0.04, 0.5, LOOSE, "y", False), ("2", "bendable", 0.04, 0.5, LOOSE, "y", False),
    ("3", "compressible", 0.12, 0.29, (0, 0), "y", False), ("4", "hard", 0.04, 0.5, (0, 0), "y", False),
    ("5", "compressible", 0.12, 0.29, (0, 0), "y", True), ("6", "hard", 0.04, 0.5, LOOSE, "x", False),
]
# schedules [s]
TOP_PHASES = [("set", 0.2), ("desc", 0.6), ("press", 0.3), ("close", 0.8), ("lift", 2.0), ("hold1", 2.0),
              ("move", 2.0), ("hold2", 3.0), ("lower", 2.0), ("open", 0.3), ("up", 0.5), ("rest", 2.0)]
SIDE_PHASES = [("set", 0.2), ("approach", 0.6), ("press", 0.3), ("close", 0.8), ("lift", 2.0), ("hold1", 2.0),
               ("move", 2.0), ("hold2", 3.0), ("lower", 2.0), ("open", 0.3), ("back", 0.5), ("rest", 2.2)]
LIFT, MOVE, Z0, Z_PRESS = 0.3, 0.5, 0.03, -0.05   # top: palm starts above the top; press target (force-limited)
GAP0, STROKE, MOVE_S, BACK = 0.03, 0.02, 0.3, 0.1  # side: start gap, stroke past the face, carry, back-off [m]
SEGS = [(i, j) for j in range(NY) for i in range(NX)]
TREE = {(1, 0): [(0, 0), (2, 0), (1, 1)], (0, 0): [(0, 1)], (2, 0): [(3, 0), (2, 1)], (3, 0): [(3, 1)]}
LOOPS = [((i, 1), (i + 1, 1)) for i in range(NX - 1)]       # x-edges closed by connect + Python springs
TOP_ANCHORS = [  # name, xy at the bottom outer edge, owner segment
    ("c00", (-L / 2, -W / 2), (0, 0)), ("c10", (L / 2, -W / 2), (3, 0)), ("c11", (L / 2, W / 2), (3, 1)),
    ("c01", (-L / 2, W / 2), (0, 1)), ("my0", (0, -W / 2), (1, 0)), ("my1", (0, W / 2), (1, 1)),
    ("mx0", (-L / 2, 0), (0, 0)), ("mx1", (L / 2, 0), (3, 0)),
]


def phases(spec):
    out, t = {}, 0.0
    for name, dur in spec:
        out[name] = (t, t + dur)
        t += dur
    return out, t


def seg_xy(s):
    return np.array([(s[0] - 1.5) * SX, (s[1] - 0.5) * SY])


def owner(x, y):
    """Segment holding footprint point (x, y); points on a seam go to the lower index (as r1)."""
    i = int(np.clip(np.ceil((x + L / 2) / SX - 1e-9) - 1, 0, NX - 1))
    j = int(np.clip(np.ceil((y + W / 2) / SY - 1e-9) - 1, 0, NY - 1))
    return i, j


def side_anchors(hc):
    """8 corners + 12 edge midpoints of the contents box: (name, xyz with z from the bottom, owner).
    Name letters give the position along x, y, z: n = minimum, m = middle, p = maximum."""
    out = []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if (sx != 0) + (sy != 0) + (sz != 0) < 2:
                    continue
                p = (sx * L / 2, sy * W / 2, (sz + 1) / 2 * hc)
                out.append(("a" + "".join("nmp"[v + 1] for v in (sx, sy, sz)), p, owner(p[0], p[1])))
    return out


def film_path(p, a):
    """Top set (r1): shortest film path from grip point p (xy, content frame) to bottom-edge anchor a (xy).

    p on the top face: unfold the side face holding a into the top plane, a -> a' = (ax, sy (W/2 + H)) for a
    long side, (sx (L/2 + H), ay) for a short side; d = min |p - a'| over the faces holding a (two for a corner).
    p outside the contents (flat bag margin): along the bottom layer, d = |p - a|.
    """
    p, a = np.asarray(p, float), np.asarray(a, float)
    if abs(p[0]) > L / 2 + 1e-9 or abs(p[1]) > W / 2 + 1e-9:
        return float(np.linalg.norm(p - a))
    cands = []
    if abs(abs(a[1]) - W / 2) < 1e-9:
        cands.append([a[0], np.sign(a[1]) * (W / 2 + H)])
    if abs(abs(a[0]) - L / 2) < 1e-9:
        cands.append([np.sign(a[0]) * (L / 2 + H), a[1]])
    return float(min(np.linalg.norm(p - np.array(c)) for c in cands))


def loop_slack(path_l, path_w, slack):
    """Slack of one rope from its surface path's split over the two loops: sqrt((sL uL)^2 + (sW uW)^2) with
    (uL, uW) the normalised (length-loop, width-loop) path parts."""
    n = np.hypot(path_l, path_w)
    return 0.0 if n < 1e-12 else float(np.hypot(slack[0] * path_l / n, slack[1] * path_w / n))


def effective_slack(slack, closing, w, g, consume):
    """Slack left for the ropes. With consume, fingers closing from opening w (first touch of film/contents)
    to gap g take up (w - g)/2 of the slack of the loop the closing direction runs along (slack = half the
    extra loop perimeter); the other loop keeps its slack."""
    s = np.array(slack, float)
    if consume:
        k = int(np.argmax(np.abs(np.asarray(closing)[:2])))
        s[k] = max(0.0, s[k] - max(w - g, 0.0) / 2)
    return s


def hold_damping_coef(mass, size, face):
    """Critical damping of the hanging pendulum, c = 2 M sqrt(g / l); l = distance from the grip point on the
    face to the contents centre at rest (half the size along the face normal)."""
    l = size[1] / 2 if face == "y" else size[0] / 2
    return 2 * mass * np.sqrt(9.81 / l)


def rope_slack(p, a, slack):
    """Slack along the rope direction u (p -> a): sqrt((sx ux)^2 + (sy uy)^2). The top set uses horizontal p, a
    (r1); the side set passes 3D points, so a rope along the thickness gets no slack (the outline has none)."""
    u = np.asarray(a, float) - np.asarray(p, float)
    u /= np.linalg.norm(u)
    return float(np.hypot(slack[0] * u[0], slack[1] * u[1]))


class BoxSurface:
    """Geodesics on the surface of an L x W x hc box (corner frame, coordinates in [0, dims]).

    Graph shortest path on a ~10 mm grid over all six faces; nodes on shared edges are merged, and nodes on
    one face within 4 grid steps are joined by straight segments, so in-face paths are nearly straight
    (direction error < 0.5 % in length). A source point is joined to every node of its face(s).
    """

    def __init__(self, hc, step=0.01, reach=4, length=L, width=W):
        self.dims = np.array([length, width, hc])
        self.key, self.xyz, self.adj, self.faces, self.nface = {}, [], [], [], []
        for ax in range(3):
            u, v = [k for k in range(3) if k != ax]
            nu, nv = (int(round(self.dims[k] / step)) for k in (u, v))
            for side in (0.0, self.dims[ax]):
                ids = np.zeros((nu + 1, nv + 1), int)
                for a in range(nu + 1):
                    for b in range(nv + 1):
                        p = np.zeros(3)
                        p[ax], p[u], p[v] = side, a * self.dims[u] / nu, b * self.dims[v] / nv
                        ids[a, b] = self._node(p)
                        self.nface[ids[a, b]].add((ax, side))
                for a in range(nu + 1):
                    for b in range(nv + 1):
                        for da in range(0, reach + 1):
                            for db in range(-reach, reach + 1):
                                if (da, db) <= (0, 0) or not (a + da <= nu and 0 <= b + db <= nv):
                                    continue
                                self._edge(ids[a, b], ids[a + da, b + db])
                self.faces.append((ax, side, ids.ravel()))

    def _node(self, p):
        k = tuple(np.round(p * 1e5).astype(int))
        if k not in self.key:
            self.key[k] = len(self.xyz)
            self.xyz.append(p)
            self.adj.append({})
            self.nface.append(set())
        return self.key[k]

    def _edge(self, i, j):
        w = float(np.linalg.norm(self.xyz[i] - self.xyz[j]))
        self.adj[i][j] = self.adj[j][i] = w

    def nearest(self, p):
        """Nearest surface point q and distance |p - q|."""
        q = np.clip(p, 0.0, self.dims)
        if (q != p).any():
            return q, float(np.linalg.norm(p - q))
        gaps = np.concatenate([p, self.dims - p])            # inside: project to the closest face
        k = int(np.argmin(gaps))
        q = p.copy()
        q[k % 3] = 0.0 if k < 3 else self.dims[k % 3]
        return q, float(gaps[k])

    def geodesic(self, q, targets):
        """Shortest surface path lengths from surface point q to each target point (grid nodes)."""
        return [g[0] for g in self.geodesic_loops(q, targets)]

    def geodesic_loops(self, q, targets):
        """Per target: (path length, length-loop part, width-loop part) of the shortest surface path from q.

        Film slack belongs to the two loops around the box, not to world axes. The loop around the long axis
        (cross-section width x thickness) takes the width direction on the top/bottom faces and the vertical
        on the long side faces; the loop around the short axis takes the length direction on the top/bottom
        faces and the vertical on the short side faces. Horizontal runs on a long side belong to the length
        loop, on a short side to the width loop. Each path segment is split into |dx|, |dy|, |dz| by that rule
        (a vertical run along a vertical box edge counts half to each loop)."""
        dist, prev = {}, {}
        q = np.asarray(q, float)
        qface = {(ax, side) for ax, side, _ in self.faces if abs(q[ax] - side) < 1e-9}
        start = [(float(np.linalg.norm(self.xyz[n] - q)), n)
                 for ax, side, ids in self.faces if abs(q[ax] - side) < 1e-9 for n in ids]
        best = {}
        for w, n in start:
            if w < best.get(n, np.inf):
                best[n] = w
                prev[n] = -1
        heap = [(w, n) for n, w in best.items()]
        heapq.heapify(heap)
        while heap:
            w, n = heapq.heappop(heap)
            if n in dist:
                continue
            dist[n] = w
            for m_, e in self.adj[n].items():
                if m_ not in dist and w + e < best.get(m_, np.inf):
                    best[m_] = w + e
                    prev[m_] = n
                    heapq.heappush(heap, (w + e, m_))
        out = []
        for t in targets:
            n = self.key[tuple(np.round(np.asarray(t) * 1e5).astype(int))]
            loops = np.zeros(2)
            while n != -1:
                p_ = prev[n]
                a, fa = (q, qface) if p_ == -1 else (self.xyz[p_], self.nface[p_])
                loops += self._loops(self.xyz[n] - a, fa & self.nface[n])
                n = p_
            out.append((dist[self.key[tuple(np.round(np.asarray(t) * 1e5).astype(int))]], *loops))
        return out

    @staticmethod
    def _loops(v, faces):
        """(length loop, width loop) share of segment v lying on the given faces."""
        dx, dy, dz = np.abs(v)
        long_side, short_side = any(ax == 1 for ax, _ in faces), any(ax == 0 for ax, _ in faces)
        share_l = 0.5 if (long_side and short_side) else (1.0 if short_side else 0.0)
        return np.array([dx + share_l * dz, dy + (1 - share_l) * dz])   # dz = 0 on the top/bottom faces


def layout(run):
    """Geometry of one case: approach n, closing c, finger-width w, grip point, plates, anchors, cameras."""
    P = CONTENTS[run.contents]
    lay = dict(P=P)
    if run.kind == "top":
        lay.update(n=np.array([0., 0, -1]), c=np.array([0., 1, 0]), w=np.array([1., 0, 0]),
                   grip0=np.array([run.grip[0], run.grip[1], H + Z0]), zc=HB / 2, hc=H,
                   plate_geoms=list(SEGS), movable=list(SEGS) if P else [],
                   k0=P["k0"] if P else 0.0, dmax=P["dmax"] if P else 0.0, gravcomp=1,
                   anchors=[(n, (xy[0], xy[1], 0.0), o) for n, xy, o in TOP_ANCHORS],
                   cams='<camera name="side" pos="0.30 -1.2 0.22" xyaxes="1 0 0 0 0 1" fovy="45"/>')
        lay["seg_geom"] = lambda s: (np.zeros(3), np.array([SX / 2, SY / 2, HB / 2]))
        lay["plate"] = lambda s: (np.array([0, 0, HB / 2 + HP / 2]), np.array([SX / 2, SY / 2, HP / 2]))
        return lay
    hc = run.Hc
    if run.face == "y":
        n, c = np.array([0., 1, 0]), np.array([1., 0, 0])
        face_segs, depth_dim, face_pt = [s for s in SEGS if s[1] == 0], W, np.array([0, -W / 2, hc / 2])
        side_cam = '<camera name="side" pos="-0.95 -0.08 0.25" xyaxes="0 -1 0 0 0 1" fovy="45"/>'
    else:
        n, c = np.array([1., 0, 0]), np.array([0., 1, 0])
        face_segs, depth_dim, face_pt = [s for s in SEGS if s[0] == 0], L, np.array([-L / 2, 0, hc / 2])
        side_cam = '<camera name="side" pos="0.10 -1.25 0.25" xyaxes="1 0 0 0 0 1" fovy="45"/>'
    grip0 = face_pt - GAP0 * n
    cam_pos = grip0 - 0.40 * n + 0.22 * c + np.array([0, 0, 0.30])
    ax_n = int(np.argmax(np.abs(n)))
    half = np.array([SX / 2, SY / 2, hc / 2])
    shrunk, pl = half.copy(), half.copy()
    shrunk[ax_n] -= HP / 2
    pl[ax_n] = HP / 2
    plated = face_segs if P else []
    lay.update(n=n, c=c, w=np.array([0., 0, 1]), grip0=grip0, zc=hc / 2, hc=hc, plate_geoms=plated, movable=plated,
               k0=P["sq_y"] / len(face_segs) if P else 0.0, dmax=P["lock_y"] * depth_dim if P else 0.0, gravcomp=0,
               anchors=side_anchors(hc),
               cams=side_cam + (f'<camera name="grip" mode="targetbody" target="palm" pos="{cam_pos[0]} {cam_pos[1]} '
                                f'{cam_pos[2]}" fovy="40"/>'))
    lay["seg_geom"] = lambda s: ((n * HP / 2, shrunk) if s in plated else (np.zeros(3), half))
    lay["plate"] = lambda s: (-n * (half[ax_n] - HP / 2), pl)
    return lay


def v3(a):
    return " ".join(f"{float(x):.6g}" for x in a)


def build_xml(run):
    lay, P = run.lay, run.lay["P"]
    m_seg = run.mass / len(SEGS)
    cgeom = f'contype="1" conaffinity="2" condim="3" friction="{TABLE_MU} 0.005 0.0001"'
    # Bodies behind a movable side plate do not touch the pads (the pad presses the plate up to dmax > HP).
    behind = lambda s: f'contype="1" conaffinity="{0 if run.kind == "side" and s in lay["movable"] else 2}" condim="3" ' \
                       f'friction="{TABLE_MU} 0.005 0.0001"'
    zc, hc = lay["zc"], lay["hc"]
    center = lambda s: np.array([*seg_xy(s), zc])
    masses = lambda s: (m_seg * (1 - PLATE_FRAC), m_seg * PLATE_FRAC) if s in lay["plate_geoms"] else (m_seg, 0.0)

    def sites(s, origin):
        return "".join(f'<site name="{nm}" pos="{v3(np.array(p) - origin)}" size="0.004"/>'
                       for nm, p, o in lay["anchors"] if o == s)

    if P is None:   # hard: one rigid body, plates fixed
        o = np.array([0, 0, zc])
        geoms = ""
        for s in SEGS:
            gp, gs = lay["seg_geom"](s)
            mb, mp = masses(s)
            geoms += (f'<geom name="g{s[0]}{s[1]}" type="box" size="{v3(gs)}" pos="{v3(center(s) - o + gp)}" mass="{mb}" '
                      f'{cgeom} rgba=".85 .55 .25 1"/>')
            if s in lay["plate_geoms"]:
                pp, ps = lay["plate"](s)
                geoms += (f'<geom name="pg{s[0]}{s[1]}" type="box" size="{v3(ps)}" pos="{v3(center(s) - o + pp)}" '
                          f'mass="{mp}" {cgeom} rgba=".95 .85 .4 1"/>')
        content = f'<body name="root" pos="{v3(o)}"><freejoint/>{geoms}' + "".join(
            f'<site name="{nm}" pos="{v3(np.array(p) - o)}" size="0.004"/>' for nm, p, _ in lay["anchors"]) + "</body>"
        loops, plates = "", ""
    else:
        k, c = P["k"], P["c"]

        def body(s, parent=None):
            cs = center(s)
            pos = cs - (center(parent) if parent else 0)
            name = f"s{s[0]}{s[1]}"
            if parent is None:
                joints = "<freejoint/>"
            else:
                face = (center(parent) - cs) / 2
                xedge = parent[1] == s[1]
                bend, twist = ("0 1 0", "1 0 0") if xedge else ("1 0 0", "0 1 0")
                fp = f'pos="{face[0]} {face[1]} 0"'
                joints = (f'<joint name="b{name}" type="hinge" axis="{bend}" {fp} stiffness="{k * run.bend_scale}" damping="{c}"/>'
                          f'<joint name="t{name}" type="hinge" axis="{twist}" {fp} stiffness="{k}" damping="{c}"/>')
            mb, mp = masses(s)
            gp, gs = lay["seg_geom"](s)
            plate = ""
            if s in lay["plate_geoms"]:
                pp, ps = lay["plate"](s)
                plate = (f'<body name="p{s[0]}{s[1]}" pos="{v3(pp)}" gravcomp="{lay["gravcomp"]}">'
                         f'<joint name="z{s[0]}{s[1]}" type="slide" axis="{v3(-lay["n"])}" range="{-lay["dmax"]} 0" '
                         f'damping="{PLATE_C}"/>'
                         f'<geom name="pg{s[0]}{s[1]}" type="box" size="{v3(ps)}" mass="{mp}" {cgeom} rgba=".95 .85 .4 1"/></body>')
            kids = "".join(body(ch, s) for ch in TREE.get(s, []))
            rgba = ".25 .55 .85 1" if s in ((1, 0), (2, 0), (1, 1), (2, 1)) else ".85 .55 .25 1"
            return (f'<body name="{name}" pos="{v3(pos)}">{joints}'
                    f'<geom name="g{s[0]}{s[1]}" type="box" size="{v3(gs)}" pos="{v3(gp)}" mass="{mb}" {behind(s)} rgba="{rgba}"/>'
                    f'{sites(s, cs)}{plate}{kids}</body>')

        content = body((1, 0))
        loops = "".join(f'<connect name="loop{a[0]}" body1="s{a[0]}{a[1]}" body2="s{b[0]}{b[1]}" '
                        f'anchor="{SX / 2} 0 0" solref="0.002 1"/>' for a, b in LOOPS)
        plates = "".join(f'<connect name="pin{i}{j}" body1="palm" body2="p{i}{j}" anchor="0 0 0" active="false" '
                         f'solref="0.005 1"/>' for (i, j) in lay["movable"])
    n, c, w, g0 = lay["n"], lay["c"], lay["w"], lay["grip0"]
    ref = g0 if run.kind == "side" else np.array([*run.grip, 0.0])   # rope goes to the nearer site along w, tie by c
    tendons = []
    for nm, p, _ in lay["anchors"]:
        dw, dc = (np.array(p) - ref) @ np.array([w, c]).T
        site = "ga" if (dw < -1e-9 or (abs(dw) < 1e-9 and dc < 0)) else "gb"
        tendons.append(f'<spatial name="r{nm}" limited="true" range="0 {FREE}" damping="{ROPE_DAMP}" '
                       f'solreflimit="{ROPE_SOLREF}" width="0.0015" rgba=".1 .1 .1 0">'
                       f'<site site="{site}"/><site site="{nm}"/></spatial>')
    box = lambda a_w, a_c, a_n: np.abs(w) * a_w + np.abs(c) * a_c + np.abs(n) * a_n
    finger = lambda nm, sgn: (
        f'<body name="f{nm}" pos="{v3(-n * PAD_H / 2 + c * sgn * (OPEN / 2 + PAD_T / 2))}" gravcomp="1">'
        f'<joint name="j{nm}" type="slide" axis="{v3(-sgn * c)}" range="0 0.06" damping="1"/>'
        f'<geom name="pad{nm}" type="box" size="{v3(box(PAD_W / 2, PAD_T / 2, PAD_H / 2))}" mass="0.05" contype="3" '
        f'conaffinity="2" condim="3" friction="{PAD_MU} 0.005 0.0001" rgba=".2 .2 .2 1"/></body>')
    pos_act = lambda nm, j, kp, kv, lo, hi: (f'<position name="{nm}" joint="{j}" kp="{kp}" kv="{kv}" '
                                            f'forcelimited="true" forcerange="{lo} {hi}"/>')
    if run.kind == "top":
        acts = (pos_act("ax", "px", 20000, 300, -500, 500) + pos_act("ay", "py", 20000, 300, -500, 500)
                + pos_act("az", "pz", 3000, 100, -PRESS_F, 500))
    else:   # approach axis is +x or +y: weak and force-limited while pushing; others stiff
        ia = int(np.argmax(np.abs(n)))
        acts = "".join(pos_act(a, j, 3000, 100, -500, PRESS_F) if k_ == ia else pos_act(a, j, 20000, 300, -500, 500)
                       for k_, (a, j) in enumerate((("ax", "px"), ("ay", "py"), ("az", "pz"))))
    block = ""
    if run.block:   # fixed block touching the face opposite the gripped one
        block = (f'<geom name="block" type="box" pos="{v3(n * (W / 2 + 0.026) + [0, 0, 0.08])}" size="0.25 0.025 0.08" '
                 f'contype="2" conaffinity="1" condim="3" friction="{TABLE_MU} 0.005 0.0001" rgba=".45 .45 .5 1"/>')
    palm_geom = (f'<geom type="box" pos="{v3(-n * (PAD_H + 0.01))}" size="{v3(box(0.02, 0.065, 0.01))}" mass="0.3" '
                 f'contype="0" conaffinity="0" rgba=".5 .5 .5 1"/>')
    xml = f"""<mujoco model="four_part_pack_demo">
<option timestep="{DT}" integrator="implicitfast" solver="Newton" iterations="50" tolerance="1e-8" cone="elliptic"/>
<visual><global offwidth="{VW}" offheight="{VH}"/><quality shadowsize="0"/><headlight ambient=".45 .45 .45" diffuse=".5 .5 .5"/></visual>
<worldbody>
<light pos="0.2 -1 1.5" dir="0 0.6 -1"/>
<geom name="table" type="plane" size="2 1 0.01" rgba=".8 .8 .75 1" contype="0" conaffinity="1" friction="{TABLE_MU} 0.005 0.0001"/>
{lay["cams"]}{block}
<body name="palm" pos="{v3(g0)}" gravcomp="1">
  <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/><joint name="pz" type="slide" axis="0 0 1"/>
  {palm_geom}
  <site name="ga" pos="{v3(-w * PAD_W / 2)}" size="0.003"/><site name="gb" pos="{v3(w * PAD_W / 2)}" size="0.003"/>
  {finger("l", -1)}{finger("r", 1)}
</body>
{content}
</worldbody>
<equality>{loops}{plates}</equality>
<tendon>{"".join(tendons)}</tendon>
<actuator>
{acts}
{pos_act("afl", "jl", 2000, 10, -GRIP_F, GRIP_F)}{pos_act("afr", "jr", 2000, 10, -GRIP_F, GRIP_F)}
</actuator>
</mujoco>"""
    return xml


class Run:
    def __init__(self, case, kind, render=True, hold_damping=True, consume_slack=True, bend_k=0.0, slack_scale=1.0):
        self.kind = kind
        self.bend_k = bend_k if kind == "side" else 0.0      # film bending stiffness [N m/rad], default 0
        # side set only: rope slack per film loop (always), hold damping and closing-consumes-slack (default on)
        self.hold_on, self.consume = kind == "side" and hold_damping, kind == "side" and consume_slack
        if kind == "top":
            self.cid, self.contents, self.slack, self.grip, self.mass, self.pinch, self.bend_scale = case
            self.Hc, self.face, self.block = H, "top", False
        else:
            self.cid, self.contents, self.Hc, self.mass, self.slack, self.face, self.block = case
            self.slack = tuple(slack_scale * v for v in self.slack)
            self.pinch, self.bend_scale = 1.0, 1
        self.ph, self.t_end = phases(TOP_PHASES if kind == "top" else SIDE_PHASES)
        self.t_dec = self.ph["close"][1]
        self.P = CONTENTS[self.contents]
        self.lay = layout(self)
        self.xml = build_xml(self)
        m = self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(m)
        self.palm = m.body("palm").id
        self.root = m.body("root" if self.P is None else "s10").id
        # box-corner coordinates of the root body origin
        self.root_off = (np.array([L / 2, W / 2, self.lay["zc"]]) if self.P is None
                         else np.array([L / 2 - SX / 2, W / 2 - SY / 2, self.lay["zc"]]))
        self.g_seg = np.array([m.geom(f"g{i}{j}").id for i, j in SEGS])
        self.b_seg = None if self.P is None else np.array([m.body(f"s{i}{j}").id for i, j in SEGS])
        self.content_geoms = set(self.g_seg) | {m.geom(f"pg{i}{j}").id for i, j in self.lay["plate_geoms"]}
        self.movable = list(self.lay["movable"])
        self.plate_geom_body = {m.geom(f"pg{i}{j}").id: (i, j) for i, j in self.movable}
        self.pads = {m.geom("padl").id, m.geom("padr").id}
        self.table = m.geom("table").id
        self.names = [nm for nm, _, _ in self.lay["anchors"]]
        self.ropes = np.array([m.tendon(f"r{nm}").id for nm in self.names])
        self.rope_sites = [(m.wrap_objid[m.tendon_adr[t]], m.wrap_objid[m.tendon_adr[t] + 1]) for t in self.ropes]
        self.site_grip = np.array([s[0] for s in self.rope_sites])
        self.site_anchor = np.array([s[1] for s in self.rope_sites])
        self.rope_index = np.full(m.ntendon, -1)
        self.rope_index[self.ropes] = np.arange(len(self.ropes))
        self.act = {nm: m.actuator(nm).id for nm in ("ax", "ay", "az", "afl", "afr")}
        self.ia = int(np.argmax(np.abs(self.lay["n"])))           # approach axis (side) / 2 = z (top)
        self.a_app = self.act[("ax", "ay", "az")[self.ia]]
        self.q_app = m.jnt_qposadr[m.joint(("px", "py", "pz")[self.ia]).id]
        self.qz = m.jnt_qposadr[m.joint("pz").id]
        self.fing_v = [m.jnt_dofadr[m.joint(j).id] for j in ("jl", "jr")]
        if self.P is not None:
            self.plate_q = {s: m.jnt_qposadr[m.joint(f"z{s[0]}{s[1]}").id] for s in self.movable}
            self.plate_v = np.array([m.jnt_dofadr[m.joint(f"z{s[0]}{s[1]}").id] for s in self.movable])
            self.plate_qa = np.array([self.plate_q[s] for s in self.movable])
            self.pin_eq = {s: m.equality(f"pin{s[0]}{s[1]}").id for s in self.movable}
            self.loop_b = [(m.body(f"s{a[0]}{a[1]}").id, m.body(f"s{b[0]}{b[1]}").id) for a, b in LOOPS]
            k, c = self.P["k"], self.P["c"]
            self.loop_k = np.array([k, k * self.bend_scale, YAW_K])   # twist (x), bend (y), yaw (z) for x-edges
            self.loop_c = np.array([c, c, YAW_C])
        self.renderer = mujoco.Renderer(m, VH, VW) if render else None
        self.cams = [m.camera(nm).id for nm in (("side",) if kind == "top" else ("side", "grip"))]
        self.mode, self.rope_on, self.pins, self.events = None, False, [], {}
        self.lp = dict(pin=0.0, rope=0.0)
        self.lp_peak = dict(pin=0.0, rope=0.0)
        self.capacity = None
        self.z_hold, self.touch = None, 0     # side put-down: palm height frozen at touchdown
        self.fing_q = [m.jnt_qposadr[m.joint(j).id] for j in ("jl", "jr")]
        self.open_w = 0.0                      # largest pad gap while a pad touches the contents, before the pinch
        self.c_bodies = np.array([b for b in range(m.nbody) if b and m.body_rootid[b] == m.body_rootid[self.root]])
        self.hold_c = hold_damping_coef(self.mass, np.array([L, W, self.Hc]), self.face) if self.hold_on else 0.0
        self.rope_loaded, self.gate_toggles, self.hold_F = False, 0, 0.0
        self.q_pinch, self.bend_T = None, 0.0      # contents orientation at the pinch; bending torque applied
        self._q = np.zeros((4, 4))

    # ---------- forces computed in Python (explicit)
    def gap(self):
        return OPEN - float(self.d.qpos[self.fing_q].sum())

    def apply_forces(self):
        m, d = self.m, self.d
        d.xfrc_applied[:] = 0.0
        self.hold_F = 0.0
        if self.hold_c and self.rope_loaded:
            # hold damping: -c (m_i/M)(v_i - v_palm) on every contents body, reaction on the palm; zero unless a
            # rope carried load in the previous step
            v6, f = np.zeros(6), np.zeros(3)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.palm, v6, 0)
            vp = v6[3:].copy()
            M = m.body_mass[self.c_bodies].sum()
            for b in self.c_bodies:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
                v = v6[3:] + np.cross(v6[:3], d.xipos[b] - d.xpos[b])
                fb = -self.hold_c * m.body_mass[b] / M * (v - vp)
                d.xfrc_applied[b, :3] += fb
                f += fb
            d.xfrc_applied[self.palm, :3] -= f
            self.hold_F = float(np.linalg.norm(f))
        self.bend_T = 0.0
        if self.bend_k and self.rope_loaded and self.q_pinch is not None:
            # film bending stiffness: torque -k * rotation of the contents (root body) since the pinch, reaction
            # on the palm (the palm does not rotate); only while a rope carries load
            qn, qr, rv = np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_negQuat(qn, self.q_pinch)
            mujoco.mju_mulQuat(qr, d.xquat[self.root], qn)       # world-frame rotation since the pinch
            mujoco.mju_quat2Vel(rv, qr, 1.0)
            tau = -self.bend_k * rv
            d.xfrc_applied[self.root, 3:] += tau
            d.xfrc_applied[self.palm, 3:] -= tau
            self.bend_T = float(np.linalg.norm(tau))
        if self.P is None:
            return
        dlt = np.clip(-d.qpos[self.plate_qa], 0.0, None)
        r = np.clip(dlt / self.lay["dmax"], 0.0, 0.98)
        d.qfrc_applied[self.plate_v] = self.lay["k0"] * dlt / (1 - r)
        qn, qr, rv = self._q[0], self._q[1], np.zeros(3)
        for a, b in self.loop_b:
            mujoco.mju_negQuat(qn, d.xquat[a])
            mujoco.mju_mulQuat(qr, qn, d.xquat[b])
            mujoco.mju_quat2Vel(rv, qr, 1.0)
            Ra = d.xmat[a].reshape(3, 3)
            w = Ra.T @ (d.cvel[b, :3] - d.cvel[a, :3])
            tau = Ra @ (-(self.loop_k * rv) - self.loop_c * w)
            d.xfrc_applied[b, 3:] += tau
            d.xfrc_applied[a, 3:] -= tau

    def ctrl(self, t):
        d, a, ph = self.d, self.act, self.ph
        if self.kind == "top":
            d.ctrl[a["ax"]] = MOVE * smooth((t - ph["move"][0]) / (ph["move"][1] - ph["move"][0]))
            d.ctrl[a["ay"]] = 0.0
            if t < self.t_dec:
                d.ctrl[a["az"]] = Z_PRESS * min(max((t - ph["desc"][0]) / (ph["desc"][1] - ph["desc"][0]), 0.0), 1.0)
            else:
                d.ctrl[a["az"]] = (self.z_ref + LIFT * (self._s("lift", t) - self._s("lower", t)) + 0.1 * self._s("up", t))
        else:
            tgt = np.zeros(3)
            if t < self.t_dec:
                u = (t - ph["approach"][0]) / (ph["approach"][1] - ph["approach"][0])
                tgt[self.ia] = (GAP0 + STROKE) * min(max(u, 0.0), 1.0)
            else:
                tgt[self.ia] = self.n_ref - BACK * self._s("back", t)
                tgt[2] = self.z_ref + LIFT * (self._s("lift", t) - self._s("lower", t))
                if self.z_hold is not None:
                    tgt[2] = self.z_hold
            tgt[0] += MOVE_S * self._s("move", t)
            d.ctrl[[a["ax"], a["ay"], a["az"]]] = tgt
        closing = ph["close"][0] <= t < ph["open"][0]
        # ramped target (~0.15 m/s); a step target drives the pads through each other
        d.ctrl[a["afl"]] = d.ctrl[a["afr"]] = 0.06 * min((t - ph["close"][0]) / 0.4, 1.0) if closing else 0.0

    def _s(self, name, t):
        t0, t1 = self.ph[name]
        return smooth((t - t0) / (t1 - t0))

    def pad_force(self):
        """Contact force of the contents on the two pads (world vector sum)."""
        f6, F = np.zeros(6), np.zeros(3)
        for i, c in enumerate(self.d.contact[:self.d.ncon]):
            if c.geom1 in self.pads and c.geom2 in self.content_geoms:
                sign = -1.0
            elif c.geom2 in self.pads and c.geom1 in self.content_geoms:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            F += sign * (c.frame.reshape(3, 3).T @ f6[:3])
        return F

    def pad_touch(self):
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in self.pads and c.geom2 in self.content_geoms) or (c.geom2 in self.pads and c.geom1 in self.content_geoms):
                return True
        return False

    def pad_plates(self):
        out = set()
        for c in self.d.contact[:self.d.ncon]:
            for g, h in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in self.pads and h in self.plate_geom_body:
                    out.add(self.plate_geom_body[h])
        return out

    def to_box(self, p):
        """World point -> contents box-corner frame (rest geometry of the root body)."""
        d = self.d
        R = d.xmat[self.root].reshape(3, 3)
        return R.T @ (p - d.xpos[self.root]) + self.root_off

    def decide(self):
        """Pinch rule (r2): gather = slack along the closing direction + DEPTH_COEF x press depth (actual plate
        travel of the plates touching the pads). Pinched if there is film at the grip point and gather >=
        GATHER_MIN; then the ropes always engage (film held), and a press deeper than DEPTH_CRIT also pins the
        plates between the fingers."""
        m, d = self.m, self.d
        plates = self.pad_plates()
        depth = float(max((-d.qpos[self.plate_q[s]] for s in plates), default=0.0))
        sx, sy = self.slack
        c = self.lay["c"]
        if self.kind == "top":
            gx, gy = self.grip
            film = abs(gx) <= L / 2 + sx and abs(gy) <= W / 2 + sy
        else:
            pb = self.to_box(d.xpos[self.palm])
            # outline: contents grown by the slack in length and width; the loop slack also reaches the
            # thickness now, so the outline top is raised by the larger slack (only used for "is there film")
            film = bool(-sx <= pb[0] <= L + sx and -sy <= pb[1] <= W + sy and 0 <= pb[2] <= self.Hc + max(sx, sy))
        gather = float(np.hypot(sx * c[0], sy * c[1])) + DEPTH_COEF * depth
        grip = float(np.mean(np.abs(d.actuator_force[[self.act["afl"], self.act["afr"]]])))
        fv = float(np.max(np.abs(d.qvel[self.fing_v])))
        self.z_ref = float(d.qpos[self.qz])
        self.n_ref = float(d.qpos[self.q_app])
        self.grip_force = grip
        self.capacity = self.pinch * grip
        info = dict(depth_mm=round(1e3 * depth, 2), gather_mm=round(1e3 * gather, 2), film=film,
                    grip_force_N=round(grip, 2), grip_reached=bool(grip >= 0.95 * GRIP_F and fv < 1e-3),
                    pad_plates=sorted(f"{i}{j}" for i, j in plates),
                    press_force_N=round(float(d.actuator_force[self.a_app]), 2))
        if self.kind == "side":
            info["grip_point_box_mm"] = [round(1e3 * v, 1) for v in pb]
            self.slack_used = effective_slack(self.slack, c, self.open_w, self.gap(), self.consume)
            info.update(open_w_mm=round(1e3 * self.open_w, 1), gap_mm=round(1e3 * self.gap(), 1),
                        slack_used_mm=[round(1e3 * v, 1) for v in self.slack_used])
            m.actuator_forcerange[self.a_app] = [-500, 500]    # hold the reached pose firmly from now on
        if not film or gather < GATHER_MIN:
            self.mode = "miss"
        else:
            self.mode = "rope"
            if depth > DEPTH_CRIT:
                self.mode = "rope+connect"
                for s in plates:
                    e, b = self.pin_eq[s], m.body(f"p{s[0]}{s[1]}").id
                    m.eq_data[e, 0:3] = 0.0
                    m.eq_data[e, 3:6] = d.xmat[b].reshape(3, 3).T @ (d.xpos[self.palm] - d.xpos[b])
                    d.eq_active[e] = 1
                    self.pins.append(e)
            lim = self.rope_limits()
            self.q_pinch = d.xquat[self.root].copy()
            m.tendon_range[self.ropes, 1] = lim
            m.tendon_rgba[self.ropes, 3] = 1.0
            self.rope_on = True
            info["rope_limits_mm"] = {nm: round(1e3 * v, 1) for nm, v in zip(self.names, lim)}
        info["mode"] = self.mode
        self.events["decision"] = info

    def rope_limits(self):
        m, d = self.m, self.d
        out = np.zeros(len(self.ropes))
        if self.kind == "top":     # r1 formula on the nominal grip point
            gx, gy = self.grip
            for k, (nm, p, _) in enumerate(self.lay["anchors"]):
                site = m.site(int(self.rope_sites[k][0])).name
                g = (gx + (-PAD_W / 2 if site == "ga" else PAD_W / 2), gy)
                out[k] = film_path(g, p[:2]) + rope_slack(g, p[:2], self.slack)
            return out
        # side: |site - nearest surface point q| + geodesic(q -> anchor) + slack of the film loops the path runs
        # along (loop_slack; consumed by the closing fingers when enabled)
        surf = BoxSurface(self.Hc)
        offs = np.array([L / 2, W / 2, 0.0])
        cache = {}
        for k, (nm, p, _) in enumerate(self.lay["anchors"]):
            sid = int(self.rope_sites[k][0])
            if sid not in cache:
                g = self.to_box(d.site_xpos[sid])
                q, dist = surf.nearest(g)
                geo = surf.geodesic_loops(q, [np.array(a) + offs for _, a, _ in self.lay["anchors"]])
                cache[sid] = (q, dist, geo)
            q, dist, geo = cache[sid]
            length, path_l, path_w = geo[k]
            out[k] = dist + length + loop_slack(path_l, path_w, self.slack_used)
        return out

    def release(self, group, why, t):
        m, d = self.m, self.d
        if group in ("pin", "all"):
            for e in self.pins:
                d.eq_active[e] = 0
            self.pins = []
        if group in ("rope", "all"):
            m.tendon_range[self.ropes, 1] = FREE
            m.tendon_rgba[self.ropes, 3] = 0.0
            self.rope_on = False
        self.events[why] = dict(t=round(t, 4), palm_rise_mm=round(1e3 * (d.qpos[self.qz] - self.z_ref), 1))

    def pin_force(self):
        d, n = self.d, self.d.nefc
        rows = np.where((d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(d.efc_id[:n], self.pins))[0]
        return float(np.linalg.norm(d.efc_force[rows].reshape(-1, 3).sum(0)))    # connect rows come in xyz triples

    def rope_dirs(self):
        """Unit vectors from the gripper point toward each anchor."""
        u = self.d.site_xpos[self.site_anchor] - self.d.site_xpos[self.site_grip]
        return u / np.linalg.norm(u, axis=1, keepdims=True)

    def rope_tension(self):
        d, n = self.d, self.d.nefc
        T = np.zeros(len(self.ropes))
        rows = np.where(d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)[0]
        T[self.rope_index[d.efc_id[rows]]] = d.efc_force[rows]
        return T

    def rope_net(self):
        """All rope forces on the contents: limit tension + tendon passive damping, as one vector norm."""
        fd = -self.m.tendon_damping[self.ropes] * self.d.ten_velocity[self.ropes]
        return float(np.linalg.norm(self.rope_dirs().T @ (self.rope_tension() + fd)))

    def table_contacts(self):
        n = 0
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 == self.table and c.geom2 in self.content_geoms) or (c.geom2 == self.table and c.geom1 in self.content_geoms):
                n += 1
        return n

    def seg_pos(self):
        return (self.d.geom_xpos[self.g_seg] if self.P is None else self.d.xpos[self.b_seg]).copy()

    def frame(self):
        imgs = []
        for cam in self.cams:
            self.renderer.update_scene(self.d, camera=cam)
            imgs.append(self.renderer.render().copy())
        return np.hstack(imgs)

    def run(self):
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        nsteps = int(round(self.t_end / DT))
        rec_every, frame_every = 10, int(round(1 / (FPS * 0.001))) * 2   # 5 ms records, 33 ms frames
        keys = ("t", "palm", "com", "seg", "R", "ncon", "F_pin", "F_rope", "lp_pin", "lp_rope", "taut", "rope_net",
                "slack_tension", "tension", "f_app", "root_pos", "hold_F", "loaded", "bend_T", "pad_F")
        rec = {k: [] for k in keys}
        frames, t_step, t_py, bad, capped = [], 0.0, 0.0, None, 0
        dec_step = int(round(self.t_dec / DT))
        for k in range(nsteps):
            t = k * DT
            p0 = time.perf_counter()
            if k == dec_step:
                self.decide()
            if (self.rope_on or self.pins) and t >= self.ph["lower"][1]:
                self.release("all", "put_down_release", t)
            if (self.kind == "side" and (self.rope_on or self.pins) and self.z_hold is None
                    and self.ph["lower"][0] <= t < self.ph["lower"][1]):
                # put down = lower until the contents has touched the table for 20 ms, then hold that height; a
                # contents turned on its side stands taller than at the pinch, and lowering the palm back to the
                # pinch height would drive the fingers into it
                self.touch = self.touch + 1 if self.table_contacts() else 0
                if self.touch * DT >= 0.02:
                    self.z_hold = float(d.qpos[self.qz])
                    self.events["touchdown"] = dict(t=round(t, 4), palm_rise_mm=round(1e3 * (self.z_hold - self.z_ref), 1))
            if self.kind == "side" and self.ph["approach"][0] <= t < self.t_dec and self.pad_touch():
                self.open_w = max(self.open_w, self.gap())
            self.ctrl(t)
            self.apply_forces()
            lim = m.tendon_range[self.ropes, 1].copy()     # limits this step is solved with
            p1 = time.perf_counter()
            mujoco.mj_step(m, d)
            p2 = time.perf_counter()
            loaded = bool(self.rope_on and self.rope_tension().max(initial=0.0) > 0.0)
            self.gate_toggles += loaded != self.rope_loaded
            self.rope_loaded = loaded
            F = dict(pin=self.pin_force() if self.pins else 0.0, rope=0.0)
            if self.rope_on:
                F["rope"] = float(np.linalg.norm(self.rope_dirs().T @ self.rope_tension()))
            for g, on in (("pin", bool(self.pins)), ("rope", self.rope_on)):
                if not on:
                    self.lp[g] = 0.0
                    continue
                self.lp[g] += DT / LP_TAU * (F[g] - self.lp[g])
                self.lp_peak[g] = max(self.lp_peak[g], self.lp[g])
                if self.lp[g] > self.capacity:
                    self.release(g, f"{g}_overload_release", t + DT)
            t_py += (p1 - p0) + (time.perf_counter() - p2)
            t_step += p2 - p1
            if d.solver_niter[0] >= m.opt.iterations:
                capped += 1
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                bad = round(t, 4)
                break
            if k % rec_every == 0:
                slack = d.ten_length[self.ropes] < lim - 1e-4
                tension = self.rope_tension()
                rec["t"].append(t + DT)
                rec["palm"].append(d.xpos[self.palm].copy())
                rec["com"].append(d.subtree_com[self.root].copy())
                rec["seg"].append(self.seg_pos())
                rec["R"].append(d.xmat[self.root].reshape(3, 3).copy())
                rec["ncon"].append(self.table_contacts())
                rec["F_pin"].append(F["pin"])
                rec["F_rope"].append(F["rope"])
                rec["lp_pin"].append(self.lp["pin"])
                rec["lp_rope"].append(self.lp["rope"])
                rec["taut"].append(int((~slack).sum()) if self.rope_on else 0)
                rec["rope_net"].append(self.rope_net())
                rec["slack_tension"].append(float(np.abs(tension[slack]).max()) if slack.any() else 0.0)
                rec["tension"].append(tension)
                rec["f_app"].append(float(d.actuator_force[self.a_app]))
                rec["root_pos"].append(d.xpos[self.root].copy())
                rec["hold_F"].append(self.hold_F)
                rec["loaded"].append(int(self.rope_loaded))
                rec["bend_T"].append(self.bend_T)
                rec["pad_F"].append(self.pad_force())
            if self.renderer is not None and k % frame_every == 0:
                frames.append(self.frame())
        if self.renderer is not None:
            self.renderer.close()
        warnings = {mujoco.mjtWarning(i).name: int(d.warning[i].number) for i in range(len(d.warning)) if d.warning[i].number}
        rec = {k: np.array(v) for k, v in rec.items()}
        return rec, frames, dict(steps=k + 1, step_s=t_step, py_s=t_py, nan_at=bad, warnings=warnings, capped=capped)


def elev(v):
    return np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))


def shape(seg):
    """seg: (..., 8, 3) segment centers in SEGS order -> pitch (long axis), roll (short axis), droop left/right [deg]."""
    s = seg.reshape(seg.shape[:-2] + (NY, NX, 3))
    c = s.mean(axis=-3)                       # per column i, averaged over rows
    pitch = elev(c[..., 3, :] - c[..., 0, :])
    roll = elev(s[..., 1, :, :].mean(-2) - s[..., 0, :, :].mean(-2))
    mid = elev(c[..., 2, :] - c[..., 1, :])
    left = elev(c[..., 1, :] - c[..., 0, :]) - mid
    right = mid - elev(c[..., 3, :] - c[..., 2, :])
    return pitch, roll, left, right


def metrics(run, rec, info):
    t, ph = rec["t"], run.ph
    win = lambda a, b: (t >= a) & (t < b)
    at = lambda ts: int(np.searchsorted(t, ts))
    out = dict(run.events["decision"])
    for k in ("pin_overload_release", "rope_overload_release", "touchdown", "put_down_release"):
        if k in run.events:
            out[k] = run.events[k]
    lo = None
    hold = int(round(0.05 / (10 * DT)))
    for j in np.where(t >= run.t_dec)[0]:
        if j + hold <= len(t) and (rec["ncon"][j:j + hold] == 0).all():
            lo = j
            break
    out["liftoff_palm_rise_mm"] = None if lo is None else round(1e3 * (rec["palm"][lo, 2] - rec["palm"][at(run.t_dec), 2]), 1)
    h = win(ph["hold1"][1] - 0.5, ph["hold1"][1])
    p, r, dl, dr = shape(rec["seg"][h])
    out.update(hang_pitch_deg=round(float(p.mean()), 1), hang_roll_deg=round(float(r.mean()), 1),
               hang_droop_deg=[round(float(dl.mean()), 1), round(float(dr.mean()), 1)],
               hang_airborne=bool((rec["ncon"][h] == 0).all()), hang_taut_ropes=int(np.median(rec["taut"][h])))
    tens = np.median(rec["tension"][h], axis=0)
    out["hang_rope_tension_N"] = {nm: round(float(v), 2) for nm, v in zip(run.names, tens) if v > 0.02}
    out["capacity_N"] = None if run.capacity is None else round(run.capacity, 2)
    for g in ("pin", "rope"):
        out[f"{g}_peak_over_capacity"] = round(run.lp_peak[g] / run.capacity, 3) if run.lp_peak[g] > 0 else None
    s = win(ph["hold2"][0], ph["hold2"][1])
    ts, sx = t[s] - ph["hold2"][0], rec["com"][s, 0] - rec["palm"][s, 0]
    sx = sx - sx[-int(1.0 / (10 * DT)):].mean()
    peaks = [j for j in range(1, len(sx) - 1) if abs(sx[j]) >= abs(sx[j - 1]) and abs(sx[j]) > abs(sx[j + 1])]
    out["swing_amp0_mm"] = round(1e3 * abs(sx[peaks[0]]), 1) if peaks else 0.0
    half = None
    if peaks and abs(sx[peaks[0]]) > 1e-3:
        a0 = abs(sx[peaks[0]])
        for j in peaks:
            if all(abs(sx[q]) <= a0 / 2 for q in peaks if q >= j):
                half = round(float(ts[j]), 2)
                break
    out["swing_half_time_s"] = half
    out["swing_amp_end_mm"] = round(1e3 * float(np.abs(sx[-int(0.5 / (10 * DT)):]).max()), 1)
    r_ = win(run.t_end - 0.5, run.t_end + 1)
    v = np.linalg.norm(np.diff(rec["seg"][r_], axis=0), axis=-1).max() / (10 * DT)
    p, r, dl, dr = shape(rec["seg"][r_])
    out.update(rest_still=bool(v < 0.005 and rec["ncon"][r_].min() > 0), rest_max_speed_mm_s=round(1e3 * float(v), 2),
               rest_pitch_roll_deg=[round(float(p.mean()), 1), round(float(r.mean()), 1)],
               rest_droop_deg=[round(float(dl.mean()), 1), round(float(dr.mean()), 1)])
    # fix-1 check: rope force on the contents while no rope is engaged, and how far the carry moved the contents
    off = rec["F_rope"] == 0
    carry = win(ph["lift"][0], ph["lower"][1])
    out["rope_force_unengaged_max_N"] = float(rec["rope_net"][off].max()) if off.any() else None
    out["rope_force_slack_rope_max_N"] = float(rec["slack_tension"].max())
    dxy = lambda a, b: round(1e3 * float(np.linalg.norm(rec["com"][at(b), :2] - rec["com"][at(a), :2])), 1)
    out["carry_contents_disp_mm"] = dxy(ph["lift"][0], ph["lower"][1])
    if run.kind == "side":
        a0 = ph["approach"][0]
        out["push_approach_mm"] = dxy(a0, ph["press"][1])
        out["push_close_mm"] = dxy(ph["press"][1], run.t_dec)
        out["push_total_mm"] = dxy(a0, run.t_dec)
        app = win(a0, ph["press"][1])
        out["press_force_max_N"] = round(float(rec["f_app"][app].max()), 2)
        out["press_force_limited"] = bool(rec["f_app"][app].max() >= 0.99 * PRESS_F)
        nf = np.einsum("tij,j->ti", rec["R"][h], -run.lay["n"])    # outward normal of the gripped face
        out["hang_face_normal_elev_deg"] = round(float(elev(nf).mean()), 1)
        Rh, ph_, pr = rec["R"][h], rec["palm"][h], rec["root_pos"][h]
        pb = np.einsum("tji,tj->ti", Rh, ph_ - pr) + run.root_off           # grip point in the box frame
        dims = np.array([L, W, run.Hc])
        out["hang_grip_to_surface_mm"] = round(1e3 * float(np.median(np.linalg.norm(pb - np.clip(pb, 0, dims), axis=1))), 1)
        out["hold_damping"] = dict(on=run.hold_on, c_N_s_m=round(run.hold_c, 2), gate_toggles=run.gate_toggles,
                                   force_max_N=round(float(rec["hold_F"].max()), 3),
                                   force_without_engaged_rope_max_N=float(rec["hold_F"][rec["F_rope"] == 0][
                                       rec["lp_rope"][rec["F_rope"] == 0] == 0].max(initial=0.0)),
                                   samples_force_while_gate_off=int(((rec["hold_F"] > 0) & (rec["loaded"] == 0)).sum()))
        out["consume_slack"] = run.consume
        out["film_bend_k_N_m_rad"] = run.bend_k
        out["hang_bend_torque_N_m"] = round(float(np.median(rec["bend_T"][h])), 3)
        out["hang_pad_contact_force_N"] = [round(float(v), 2) for v in np.median(rec["pad_F"][h], axis=0)]
        out["rest_face_normal_elev_deg"] = round(float(elev(np.einsum("tij,j->ti", rec["R"][r_], -run.lay["n"])).mean()), 1)
    out.update(nan_at_s=info["nan_at"], warnings=info["warnings"], solver_capped_steps=info["capped"],
               mean_step_us=round(1e6 * info["step_s"] / info["steps"], 2),
               mean_python_us=round(1e6 * info["py_s"] / info["steps"], 2))
    return out


def write_video_wide(frames, path):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                          "-video_size", f"{w}x{h}", "-framerate", RATE, "-i", "-", *VP9, str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def overview_side(paths, labels, out):
    """2 columns x 3 rows of the two-view case videos."""
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in paths:
        cmd += ["-i", str(p)]
    f = [f"[{i}:v]drawtext=fontfile={font}:text='{lab}':fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:x=6:y=6[v{i}]"
         for i, lab in enumerate(labels)]
    layout_ = "|".join(f"{(i % 2) * 2 * VW}_{(i // 2) * VH}" for i in range(len(paths)))
    f.append("".join(f"[v{i}]" for i in range(len(paths))) + f"xstack=inputs={len(paths)}:layout={layout_}[o]")
    cmd += ["-filter_complex", ";".join(f), "-map", "[o]", *VP9, str(out)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--set", choices=("top", "side"), default="top")
    ap.add_argument("--cases", nargs="*", help="subset of case ids")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-hold-damping", action="store_true", help="side set: switch the default hold damping off")
    ap.add_argument("--no-consume-slack", action="store_true", help="side set: closing fingers keep all the slack")
    ap.add_argument("--film-bend-k", type=float, default=0.0, help="side set: film bending stiffness [N m/rad]")
    ap.add_argument("--slack-scale", type=float, default=1.0, help="side set: scale every case's slack")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results, videos, labels = [], [], []
    for case in (TOP_CASES if args.set == "top" else SIDE_CASES):
        if args.cases and case[0] not in args.cases:
            continue
        run = Run(case, args.set, render=not args.no_video, hold_damping=not args.no_hold_damping,
                  consume_slack=not args.no_consume_slack, bend_k=args.film_bend_k, slack_scale=args.slack_scale)
        rec, frames, info = run.run()
        if args.set == "top":
            res = dict(case=case[0], contents=case[1], slack_mm=[round(1e3 * v) for v in case[2]],
                       grip_mm=[round(1e3 * v) for v in case[3]], mass_kg=case[4], pinch_coef=case[5], bend_scale=case[6])
        else:
            res = dict(case=case[0], contents=case[1], thickness_mm=round(1e3 * case[2]), mass_kg=case[3],
                       slack_mm=[round(1e3 * v) for v in run.slack], face={"y": "long", "x": "short"}[case[5]],
                       fixed_block=case[6], side_plate_k0_N_m=round(run.lay["k0"], 2),
                       side_plate_dmax_mm=round(1e3 * run.lay["dmax"], 1))
        res.update(metrics(run, rec, info))
        results.append(res)
        print(json.dumps(res), flush=True)
        np.savez_compressed(out / f"case{case[0]}_traj.npz", names=np.array(run.names), **rec)
        if case[0] == "3":
            (out / "case3_model.xml").write_text(run.xml)
        if frames:
            path = out / f"case{case[0]}.webm"
            if args.set == "top":
                write_video(frames, path)
                if case[6] == 1:
                    videos.append(path)
                    labels.append(f"{case[0]} {case[1]} slack {res['slack_mm'][0]} grip x {res['grip_mm'][0]}"
                                  + (f" {case[4]}kg c{case[5]}" if case[4] != 0.5 else ""))
            else:
                write_video_wide(frames, path)
                videos.append(path)
                labels.append(f"{case[0]} {case[1]} {res['thickness_mm']}mm slack {res['slack_mm'][0]} {res['face']} side"
                              + (" block" if case[6] else ""))
    if args.set == "top" and len(videos) == 8:
        overview(videos, labels, out / "overview_2x4.webm")
    if args.set == "side" and len(videos) == 6:
        overview_side(videos, labels, out / "overview_3x2.webm")
    summary = dict(host=platform.node(), mujoco=mujoco.__version__, dt=DT, set=args.set,
                   schedule_s={k: [round(a, 3), round(b, 3)] for k, (a, b) in phases(TOP_PHASES if args.set == "top" else SIDE_PHASES)[0].items()},
                   rule=dict(gather_min_m=GATHER_MIN, depth_coef=DEPTH_COEF, depth_crit_m=DEPTH_CRIT, grip_force_N=GRIP_F,
                             press_force_N=PRESS_F, rope_damping=ROPE_DAMP, rope_solreflimit=ROPE_SOLREF,
                             slack_by_film_loop=args.set == "side", hold_damping=args.set == "side" and not args.no_hold_damping,
                             consume_slack=args.set == "side" and not args.no_consume_slack,
                             film_bend_k=args.film_bend_k if args.set == "side" else 0.0,
                             slack_scale=args.slack_scale if args.set == "side" else 1.0),
                   cases=results,
                   mean_step_us=round(float(np.mean([r["mean_step_us"] for r in results])), 2),
                   mean_python_us=round(float(np.mean([r["mean_python_us"] for r in results])), 2))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("mean_step_us", "mean_python_us")}))


if __name__ == "__main__":
    main()
