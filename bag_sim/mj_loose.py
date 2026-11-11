"""Loose-bag representations for the segment content model, MuJoCo 3.17 (test/bag_model/REPORT.md round 1, C2).

Two cheap representations of a loosely packed mailer, both wrapped around one content description from
contents/segment (the content stays exactly as contents/segment/mj_adapter.scene builds it; it is only called):

  plates  thin box plates (cell ~50 mm) for the front and the back panel. Each panel is a comb tree of
          hinges (fold about the shared edge + twist, stiffness from the bending rigidity D, in-plane rigid);
          the remaining in-panel neighbours and the four perimeter seams are closed with site `connect`
          equalities. integrator implicitfast, as the content model.
  flex    the bag's reference mesh (geometry/<profile>.npz) as a dim-2 flex with membrane + bending
          elasticity; wrapped around the content with wrap.wrap_generic; the open mouth is sealed with
          connect equalities. MuJoCo 3.17 only accepts flex elasticity with integrator="discrete", so the
          whole scene (content included) runs with that integrator.

Contact layout (collision bits): table 1/0, content 2/2 (no direct table contact inside a bag), bag 2/7,
pads 4/0 (pads touch only the bag). Built through the scene() fragment hook of contents/segment (I-27). Bag geoms have priority 1, so every bag contact uses the bag's friction
(surface friction = pad friction = inner friction) and its contact stiffness BAG_K per point; with the bag
in series this gives the pads about the protocol's 40 N/mm.

Bag size: `size` (L, W) of the flat bag body, or from `fill`: L = (Lx + Lz) / fill, W = (Ly + Lz) / fill,
i.e. fill is the fraction of the flat bag length the content takes once the panels drape over its height
(about 0.95 is a full bag, C1's loose kraft case is 0.43). Long sides of bag and content along x.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE / "legacy", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from frozen_content import activate
CONTENT_RELEASE_SHA256 = activate()
from mj_adapter import BUSHING_DT, CONTACT_D, Sim, scene  # noqa: E402
from wrap import point_triangle_distance, winding_number  # noqa: E402

SOLVER = "Newton"
BAG_K = 2e4                 # N/m static contact stiffness per bag contact point
WRAP_SOFT_S, WRAP_STIFF_S = 0.25, 0.1      # plates: seams pulled in softly, then stiffened
SEAM_SOLREF_SOFT, SEAM_SOLREF = "0.03 1", "0.002 1"
FLEX_RADIUS = 0.001
WRAP_RADIUS = 0.003         # flex: radius used to wrap the mesh when the contact radius strains it too much
GUSSET = {"kraft": 0.025}  # side gusset depth of geometry/kraft.npz (panels 1, 2, 4, 5)
ESCAPE_SHRINK = 0.003       # content points this far inside the content surface are tested for containment


@dataclass
class LooseBag:
    kind: str = "plates"                 # plates | flex
    size: tuple | None = None            # (L, W) of the flat bag body; None: from fill
    fill: float = 0.45
    profile: str = "kraft"               # material values (friction, mass, D) and the flex mesh
    mass: float | None = None            # kg; None: profile mesh mass
    friction: float | None = None        # bag surface = pads = table = inner friction; None: profile
    bending: float | None = None         # D in N m; None: profile bending_rigidity_nm
    membrane: float = 1e3                # flex: E*t in N/m (softened, see REPORT round 1)
    cell: float = 0.05                   # plates: target cell size (m)
    thickness: float = 0.002             # plates: collision thickness (m)
    gusset: float | None = None          # plates: side gusset depth per panel (m); None: profile (kraft 25 mm)
    crease: float | None = None          # plates: gusset fold stiffness ratio; None: profile precreased ratio
    offset: tuple = (0.0, 0.0)           # content centre in the bag, fraction of the free half-slack
    selfcollide: str = "none"            # flex self collision
    mesh: str | None = None              # flex: geometry npz (relative to bag_sim/); None: the profile's mesh
    edge: float | None = None            # flex: generate the sealed kraft mesh at this edge length for the bag size
                                         # (make_kraft_coarse.py); overrides mesh
    radius: float = FLEX_RADIUS          # flex: collision radius (half the collision thickness of the film)
    solimp: float = CONTACT_D            # flex: contact impedance d (static stiffness m*k*d/(1-d), see _contact_solref)
    panel_contact: bool = True           # plates: front/back panel plates collide with each other
    curl: float = 0.2                    # plates: rest angle (rad) lifting the back panel's outer cells
    compress: float = 0.9                # plates: in-plane shortening allowed per joint (fraction of the cell);
                                         # 0 = in-plane rigid in compression too (the plan's literal form)
    compress_k: float = 2.0              # plates: N/m resisting that shortening (buckled film is nearly free)
    tolerance: float = 1e-8
    iterations: int = 100
    ls_iterations: int = 20
    bottom_edge: float | None = None     # F6d: coarser table-facing panel, empty ends retain edge
    pinch: bool = False                  # F7: inactive upper-film connect equalities

    def resolved(self, content_size):
        prof = json.loads((HERE / "profiles" / f"{self.profile}.json").read_text())
        p = prof["params"]
        with np.load(HERE / prof["geometry"]) as g:
            mesh_mass = float(g["masses_kg"].sum())
        Lx, Ly, Lz = content_size
        size = self.size or ((Lx + Lz) / self.fill, (Ly + Lz) / self.fill)
        # 8 mm seam allowance per side, as wrap.wrap_generic; the gussets take up the content height
        if size[0] < Lx + 0.016 or size[1] < Ly + 0.016:
            raise ValueError(f"bag {size} too small for content {content_size}")
        return replace(self, size=tuple(map(float, size)),
                       gusset=GUSSET.get(self.profile, 0.0) if self.gusset is None else self.gusset,
                       crease=p.get("precreased_stiffness_ratio", 1.0) if self.crease is None else self.crease,
                       mass=mesh_mass if self.mass is None else self.mass,
                       friction=p["friction"] if self.friction is None else self.friction,
                       bending=p["bending_rigidity_nm"] if self.bending is None else self.bending)


def _v(x):
    return " ".join(f"{float(a):.8g}" for a in x)


def _contact_solref(mass, dt, d=CONTACT_D):
    """Static stiffness BAG_K per point for a body of this mass (MuJoCo: mass*d*k/(1-d)), critically damped,
    capped at omega*dt <= 0.5 like contents/segment/mj_adapter.contact_k. For light flex vertices at large steps
    the cap leaves K far below BAG_K; a larger impedance d recovers it without raising k (round 2, F1)."""
    k = min(BAG_K * (1 - d) / (d * mass), (0.5 / dt) ** 2 / d)
    return f'solref="{-k:.6g} {-2 * math.sqrt(k):.6g}" solimp="{d} {d} 0.001"'


# ------------------------------------------------------------------ plates

def _plates_xml(bag: LooseBag, xy, z_top, dt):
    """Plate grid: uniform cells along x; along y the panel cells plus one gusset column of width
    bag.gusset on each side, joined to the panel by a pre-creased fold (bending x crease ratio)."""
    L, W = bag.size
    nx, ny_p = max(2, round(L / bag.cell)), max(2, round(W / bag.cell))
    wy = [W / ny_p] * ny_p
    if bag.gusset > 0:
        wy = [bag.gusset] + wy + [bag.gusset]
    ny = len(wy)
    ey = np.r_[0.0, np.cumsum(wy)] - sum(wy) / 2              # column edges along y
    cx, t = L / nx, bag.thickness
    area = L * sum(wy)
    mass = lambda j: bag.mass / 2 * cx * wy[j] / area
    crease = {(0, 1), (ny - 2, ny - 1)} if bag.gusset > 0 else set()

    def k_hinge(a, b):
        """Fold stiffness of the hinge between cells a and b: D * edge length / mean span."""
        if a[1] == b[1]:                                       # x-neighbours, fold line along y
            return bag.bending * wy[a[1]] / cx
        j = min(a[1], b[1])
        k = bag.bending * cx / (0.5 * (wy[j] + wy[j + 1]))
        return k * (bag.crease if (j, j + 1) in crease else 1.0)

    def damp(k, j):
        return 2 * 0.5 * math.sqrt(k * mass(j) * max(cx, wy[j]) ** 2 / 3)   # damping ratio 0.5 on one plate

    def geom(j):
        aff = 7 if bag.panel_contact else 5
        return (f'type="box" size="{cx / 2:.6g} {wy[j] / 2:.6g} {t / 2:.6g}" mass="{mass(j):.6g}" contype="2" '
                f'conaffinity="{aff}" condim="3" priority="1" friction="{bag.friction} 0.005 0.0001" '
                f'{_contact_solref(mass(j), dt)}')

    i0, j0 = nx // 2, ny // 2
    center = lambda i, j: np.array([-L / 2 + (i + .5) * cx, 0.5 * (ey[j] + ey[j + 1])])
    half = lambda i, j: np.array([cx / 2, wy[j] / 2])
    sites, eqs, excludes, ropes = {}, [], [], []

    def site(panel, i, j, local):
        key = (panel, i, j, tuple(np.round(local, 9)))
        if key not in sites:
            sites[key] = f"bag{panel}_{i}_{j}_s{len(sites)}"
        return sites[key]

    def parent_of(i, j):
        if j != j0:
            return i, j - 1 if j > j0 else j + 1
        if i != i0:
            return (i - 1 if i > i0 else i + 1), j
        return None

    children = {}
    for i in range(nx):
        for j in range(ny):
            pa = parent_of(i, j)
            if pa is not None:
                children.setdefault(pa, []).append((i, j))

    def body(panel, i, j, z):
        name = f"bag{panel}_{i}_{j}"
        pa = parent_of(i, j)
        if pa is None:
            pos = np.r_[center(i, j) + xy, z]
            joints = [f'<freejoint name="{name}_free"/>']
        else:
            d = np.r_[center(i, j) - center(*pa), 0.0]
            pos = d
            e = d / np.linalg.norm(d)                       # parent -> child, in plane
            fold = np.array([-e[1], e[0], 0.0])               # shared edge direction
            k = k_hinge(pa, (i, j))
            kt = bag.bending * (wy[j] / cx if abs(e[0]) > .5 else cx / (0.5 * (wy[j] + wy[pa[1]])))
            hinge = -e * np.abs(np.r_[half(i, j), 0] @ np.abs(e))   # shared edge midpoint in child frame
            # A perfectly flat margin on the table is a strut: pushed at its edge it jams instead of folding
            # (REPORT round 1). The back panel's outermost panel cells rest slightly curled up (rotation -curl
            # about ez x e lifts the child's outer edge).
            outer = (j in (1, ny - 2) and bag.gusset > 0 and pa[1] in (2, ny - 3)) or (pa[1] == j and i in (0, nx - 1))
            ref = -math.degrees(bag.curl) if panel == "B" and outer else 0.0       # MJCF angles are degrees
            joints = [f'<joint name="{name}_f" type="hinge" pos="{_v(hinge)}" axis="{_v(fold)}" stiffness="{k:.6g}" '
                      f'springref="{ref:.6g}" damping="{damp(k, j):.6g}"/>',
                      f'<joint name="{name}_t" type="hinge" pos="{_v(hinge)}" axis="{_v(e)}" stiffness="{kt:.6g}" '
                      f'damping="{damp(kt, j):.6g}"/>']
            if bag.compress > 0:
                # Film is stiff in tension but wrinkles in compression: the joint may only shorten (tension stop
                # at 0). Rigid in compression, the closed bag acted as a stiff tube (REPORT round 1).
                span = float(np.abs(np.r_[half(i, j) + half(*pa), 0]) @ np.abs(e))
                kc, mc = bag.compress_k, mass(j)
                joints.append(f'<joint name="{name}_c" type="slide" pos="{_v(hinge)}" axis="{_v(e)}" '
                              f'stiffness="{kc:.6g}" damping="{2 * math.sqrt(kc * mc):.6g}" limited="true" '
                              f'range="{-bag.compress * span:.6g} 0" solreflimit="{SEAM_SOLREF}"/>')
        out = [f'<body name="{name}" pos="{_v(pos)}">', *joints,
               f'<geom name="{name}" {geom(j)} rgba="{".72 .56 .36 1" if panel == "F" else ".62 .47 .3 1"}"/>']
        for (pi, pj, local), s in [((a[1], a[2], np.array(a[3])), s) for a, s in sites.items() if a[0] == panel]:
            if (pi, pj) == (i, j):
                out.append(f'<site name="{s}" pos="{_v(local)}" size="0.001"/>')
        for c in children.get((i, j), []):
            out.append(body(panel, *c, z))
        out.append("</body>")
        return "\n".join(out)

    # in-panel loops: x-neighbours off the spine row are not parent/child
    for panel in "FB":
        for i in range(nx - 1):
            for j in range(ny):
                if j == j0:
                    continue
                if bag.compress > 0:      # tension-only rope between the plate centres
                    a, b = site(panel, i, j, (0, 0, 0)), site(panel, i + 1, j, (0, 0, 0))
                    ropes.append(f'<spatial name="{a}_{b}" limited="true" range="0 {cx:.6g}" '
                                 f'solreflimit="{SEAM_SOLREF}"><site site="{a}"/><site site="{b}"/></spatial>')
                    continue
                a = site(panel, i, j, (cx / 2, 0, 0))
                b = site(panel, i + 1, j, (-cx / 2, 0, 0))
                eqs.append(f'<connect name="{a}_{b}" site1="{a}" site2="{b}" solref="{SEAM_SOLREF}"/>')
        # neighbours (incl. diagonal) never collide with each other
        for i in range(nx):
            for j in range(ny):
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = (i, j), (i + di, j + dj)
                        if b <= a or not (0 <= b[0] < nx and 0 <= b[1] < ny):
                            continue
                        if parent_of(*b) == a or parent_of(*a) == b:
                            continue
                        excludes.append(f'<exclude body1="bag{panel}_{i}_{j}" body2="bag{panel}_{b[0]}_{b[1]}"/>')
    # perimeter seams: front bottom face to back top face at each outer edge midpoint
    seams = []
    for i in range(nx):
        for j in range(ny):
            for n in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= i + n[0] < nx and 0 <= j + n[1] < ny:
                    continue
                edge = np.array(n) * half(i, j)
                a = site("F", i, j, (*edge, -t / 2))
                b = site("B", i, j, (*edge, t / 2))
                seams.append(f"{a}_{b}")
                if not excludes or excludes[-1] != (ex := f'<exclude body1="bagF_{i}_{j}" body2="bagB_{i}_{j}"/>'):
                    excludes.append(ex)          # seam partners: joined at the edge, faces may cross
                eqs.append(f'<connect name="{a}_{b}" site1="{a}" site2="{b}" solref="{SEAM_SOLREF_SOFT}"/>')
    xml = body("B", i0, j0, t / 2) + "\n" + body("F", i0, j0, z_top + t / 2)
    names = [f"bag{p}_{i}_{j}" for p in "FB" for i in range(nx) for j in range(ny)]
    return dict(xml=xml, eq="\n".join(eqs), exclude="\n".join(excludes), tendon="\n".join(ropes), bodies=names, seams=seams,
                grid=(nx, ny), cell=cx, widths=wy, integrator="implicitfast")


# ------------------------------------------------------------------ flex

def _flex_mesh(bag: LooseBag, content_size, center):
    """Reference mesh rotated so its length is along x (mouth at -x), scaled to the bag size, wrapped."""
    from contents import normalize_contents
    from wrap import wrap_generic
    prof = json.loads((HERE / "profiles" / f"{bag.profile}.json").read_text())
    p = prof["params"]
    if bag.edge:
        from make_kraft_coarse import build as kraft_mesh
        # gussets deep enough for the content height (kraft.npz: 25 mm for its 30 mm boxes)
        g = kraft_mesh(bag.edge, width=bag.size[1], length=bag.size[0], mass=bag.mass,
                       gusset=max(0.025, 0.5 * content_size[2] + 2 * bag.radius),
                       bottom_edge=bag.bottom_edge, content_length=content_size[0])
        v, faces, pid = np.array(g["vertices"], float), g["faces"], g["panel_ids"]
    else:
        with np.load(HERE / (bag.mesh or prof["geometry"])) as g:
            v, faces, pid = np.array(g["vertices"], float), np.array(g["faces"]), np.array(g["panel_ids"])
    front = np.unique(faces[np.isin(pid, p["self_contact_front_panel_ids"])])
    lo, hi = v[front, :2].min(0), v[front, :2].max(0)          # bag body (front panel) extent
    v[:, :2] -= (lo + hi) / 2
    body = hi - lo                                             # mesh (x = width, y = length)
    rest = np.c_[-v[:, 1] * bag.size[0] / body[1], v[:, 0] * bag.size[1] / body[0], v[:, 2]]   # mouth at -x
    item = normalize_contents([dict(shape="box", size_m=list(content_size), mass_kg=1.0)])[0]
    # wrap_generic centres the content on the front panel; shift the mesh so the content lands at `center`
    # The wrap is only the initial shape. Its wall gap and lift grow with 2 r; at r = 5 mm the anchor bags strain
    # 35 % (round 2), so they are wrapped at WRAP_RADIUS and the thicker contact layer settles in LooseSim.wrap().
    for r_wrap in dict.fromkeys((bag.radius, min(bag.radius, WRAP_RADIUS))):
        item = normalize_contents([dict(shape="box", size_m=list(content_size), mass_kg=1.0)])[0]
        q, report = wrap_generic(rest, rest.copy(), faces, pid, p["self_contact_front_panel_ids"],
                                 p["self_contact_back_panel_ids"], item, r_wrap)
        if report["maximum_edge_strain"] <= .05:
            break
    else:
        raise ValueError(f"wrapped flex strains edges by {report['maximum_edge_strain']:.3f}")
    report["wrap_radius"] = r_wrap
    shift = np.r_[np.asarray(center[:2]) - item["position"][:2], 0.0]
    q += shift
    z_content = float(item["position"][2])
    # mouth seal: front-panel boundary vertices to the nearest back/flap vertex
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    boundary = np.unique(u[c == 1])
    mouth = np.intersect1d(boundary, front)
    back = np.setdiff1d(np.unique(faces[np.isin(pid, p["self_contact_back_panel_ids"])]), front)
    seal = []
    if len(mouth):                      # closed meshes (make_kraft_coarse.py) have no mouth
        x_mouth = rest[mouth, 0].mean()
        near = back[np.abs(rest[back, 0] - x_mouth) < 1e-6]
        seal = [(int(a), int(near[np.argmin(np.linalg.norm(rest[near, 1:2] - rest[a, 1:2], axis=1))])) for a in mouth]
    return q, faces, z_content, seal, boundary, np.unique(faces[pid == 0])


def _flex_xml(bag: LooseBag, q, faces, seal, dt):
    E_t, D, nu = bag.membrane, bag.bending, 0.3
    t_eff = math.sqrt(12 * (1 - nu * nu) * D / E_t)            # match both E*t and D with one plate law
    m_vert = bag.mass / len(q)
    pts = " ".join(f"{x:.7g}" for x in q.ravel())
    el = " ".join(str(int(i)) for i in faces.ravel())
    xml = (f'<flexcomp name="bag" type="direct" dim="2" radius="{bag.radius}" point="{pts}" element="{el}" '
           f'mass="{bag.mass:.6g}" rgba=".72 .56 .36 1">\n'
           f'<elasticity young="{E_t / t_eff:.6g}" poisson="{nu}" thickness="{t_eff:.6g}" damping="1e-4" elastic2d="both"/>\n'
           f'<contact condim="3" priority="1" friction="{bag.friction} 0.005 0.0001" {_contact_solref(m_vert, dt, bag.solimp)} '
           f'selfcollide="{bag.selfcollide}" contype="2" conaffinity="7"/>\n</flexcomp>')
    eqs = [f'<connect name="seal{a}" body1="bag_{a}" body2="bag_{b}" anchor="0 0 0" solref="{SEAM_SOLREF}"/>'
           for a, b in seal]
    return dict(xml=xml, eq="\n".join(eqs), exclude="", bodies=[f"bag_{i}" for i in range(len(q))],
                integrator="discrete")


# ------------------------------------------------------------------ scene

@dataclass
class LooseBuilt:
    built: object            # contents/segment Built with the full scene xml
    bag: LooseBag
    meta: dict = field(default_factory=dict)


# Keep the legacy bag-model API while maintaining the pad geometry in one place.
sys.path.insert(0, str(ROOT / "experience"))
from yam_platform.geometry import fingertip_xml


def loose_scene(desc, bag: LooseBag, pads=(), dt=None, table_xmin=-0.6, content_xy=(0.0, 0.0), fingertips=()):
    """fingertips: (name, normal_axis, position) of v2 fingertips, used instead of the 40 mm pads."""
    """Scene with table, pads, the content (contents/segment scene) and a loose bag around it.
    Bag centre at content_xy - offset; the content rests on the back panel."""
    Lx, Ly, Lz = desc.kappa.size
    bag = bag.resolved((Lx, Ly, Lz))
    dt = BUSHING_DT if dt is None else dt
    L, W = bag.size
    slack = np.array([L - Lx - Lz, W - Ly - Lz]) / 2
    off = np.asarray(bag.offset) * slack
    bag_c = np.asarray(content_xy) - off
    if bag.kind == "plates":
        z0 = bag.thickness + 0.0005 + Lz / 2
        parts = _plates_xml(bag, bag_c, z0 + Lz / 2 + 0.001, dt)
        bag_xml = parts["xml"]
        origin = (*content_xy, z0)
    elif bag.kind == "flex":
        q, faces, z_c, seal, boundary, upper = _flex_mesh(bag, (Lx, Ly, Lz), (off[0], off[1], 0))
        q[:, :2] += bag_c
        parts = _flex_xml(bag, q, faces, seal, dt)
        parts.update(faces=faces, boundary=boundary, upper=upper)
        if bag.pinch:
            parts['xml'] += '\n<body name="pinch_frame" mocap="true" pos="0 0 1"/>'
            parts['eq'] += '\n' + '\n'.join(
                f'<connect name="film_pinch_{i}" body1="pinch_frame" body2="bag_{i}" anchor="0 0 0" '
                'active="false" solref="0.005 1" solimp="0.99 0.99 0.001"/>' for i in upper)
        bag_xml = parts["xml"]
        origin = (*content_xy, z_c)
    else:
        raise ValueError(f"unknown bag kind {bag.kind!r}")
    # contents/segment scene() hook (I-27): the bag goes in as MJCF fragments; the content touches neither the
    # table nor the pads directly; the bag scene uses the content model's own solver (Newton since v1)
    if fingertips:
        if bag.pinch:
            from top_grasp import fingertip_xml as frozen_tip
            bag_xml += '\n' + '\n'.join(frozen_tip(f[0], f[2]).replace(
                'contype="0" conaffinity="0"', 'contype="4" conaffinity="0"') for f in fingertips)
        else:
            bag_xml += "\n" + "\n".join(fingertip_xml(*f) for f in fingertips)
    built = scene(desc, origin, pads, table_xmin=table_xmin, dt=dt, worldbody=bag_xml, contact=parts["exclude"],
                  equality=parts["eq"], tendon=parts.get("tendon", ""), pad_contact=False, table_contact=False,
                  solver=SOLVER)
    xml = built.xml
    # Not covered by the hook: the pads collide with the bag only (bit 4), and MuJoCo 3.17 accepts flex
    # elasticity only with integrator="discrete"
    for pname, _, _ in pads:
        xml = re.sub(rf'(<geom name="{pname}_[^"]*"[^>]*?)contype="0" conaffinity="0"',
                     rf'\1contype="4" conaffinity="0" friction="{bag.friction} 0.005 0.0001"', xml)
    xml = xml.replace('integrator="implicitfast"', f'integrator="{parts["integrator"]}"')
    built.xml = xml
    meta = {k: v for k, v in parts.items() if k not in ("xml", "eq", "exclude", "tendon")}
    meta.update(bag_center=bag_c.tolist(), content_origin=list(map(float, origin)))
    return LooseBuilt(built, bag, meta)


class LooseSim(Sim):
    """contents/segment Sim (content bodies, bushings, welds, pads) plus the bag."""

    def __init__(self, lb: LooseBuilt, pad_names=()):
        super().__init__(lb.built, pad_names)
        self.bag = lb.bag
        self.meta = lb.meta
        m = self.m
        self.bag_bodies = np.array([m.body(n).id for n in lb.meta["bodies"]])
        self.bag_mass = float(m.body_mass[self.bag_bodies].sum())
        self.pkg_mass = self.mass + self.bag_mass
        self._seams = np.array([m.equality(n).id for n in lb.meta.get("seams", [])], int)
        self._content_pts = self._content_points()
        m.opt.tolerance = lb.bag.tolerance
        m.opt.iterations = lb.bag.iterations
        m.opt.ls_iterations = lb.bag.ls_iterations
        self.max_ncon = self.max_nefc = self.measured_steps = self.solver_iterations = 0
        self.step_wall_s = 0.

    def step(self):
        t0 = time.perf_counter()
        super().step()
        self.step_wall_s += time.perf_counter() - t0
        self.measured_steps += 1
        self.max_ncon = max(self.max_ncon, self.d.ncon)
        self.max_nefc = max(self.max_nefc, self.d.nefc)
        self.solver_iterations += int(self.d.solver_niter.max())

    # ---------- setup
    def wrap(self):
        """Close the seams around the content and come to rest (plates); short settle (flex).
        Afterwards time is reset to 0 and velocities zeroed: this is the package's initial state."""
        m, d = self.m, self.d
        if len(self._seams):
            stiff = np.array([float(x) for x in SEAM_SOLREF.split()])
            for _ in range(round(WRAP_SOFT_S / m.opt.timestep)):
                self.step()
            m.eq_solref[self._seams] = stiff
            for _ in range(round(WRAP_STIFF_S / m.opt.timestep)):
                self.step()
        else:
            for _ in range(round(0.1 / m.opt.timestep)):
                self.step()
        d.qvel[:] = 0
        d.time = 0.0
        mujoco.mj_forward(m, d)

    # ---------- package measurements
    def bag_points(self):
        return self.d.xpos[self.bag_bodies]

    def pad_normal_force(self, p, axis):
        # Flex contacts use geom=-1. It must not index the last (sometimes fingertip) geom.
        c = self.d.contact
        if not self.d.ncon:
            return 0.
        g = c.geom
        valid = g >= 0
        hit = (self._pad_geom_mask[p][np.maximum(g, 0)] & valid).any(axis=1)
        hit &= c.efc_address >= 0
        return float(np.sum(self.d.efc_force[c.efc_address[hit]] * np.abs(c.frame[hit, axis])))

    def pkg_com(self):
        w = np.r_[self.m.body_mass[self.obj_bodies], self.m.body_mass[self.bag_bodies]]
        x = np.r_[self.d.xipos[self.obj_bodies], self.d.xipos[self.bag_bodies]]
        return (w[:, None] * x).sum(0) / w.sum()

    def table_force_pkg(self):
        """Vertical force of the table on content + bag."""
        d = self.d
        n = d.ncon
        if n == 0:
            return 0.0
        geom = d.contact.geom[:n]
        mask = (geom[:, 0] == self.table_geom) | (geom[:, 1] == self.table_geom)
        if not mask.any():
            return 0.0
        fn = d.efc_force[d.contact.efc_address[:n][mask]]
        return float(np.sum(fn * np.abs(d.contact.frame[:n, 2][mask])))

    def footprint(self):
        """Top-view axis-aligned box of the package: (centre xy, size xy, height)."""
        v, _ = self.bag_surface()
        pad = self.bag.thickness / 2 if self.bag.kind == "plates" else self.bag.radius
        pts = np.vstack([v, self.content_world_points()])
        lo, hi = pts[:, :2].min(0) - pad, pts[:, :2].max(0) + pad
        return (lo + hi) / 2, hi - lo, float(pts[:, 2].max() + pad)

    def _content_points(self):
        """Segment corners pulled ESCAPE_SHRINK inside, plus centres: (body, local point) arrays."""
        segs = self.desc.segments
        bodies, pts = [], []
        for i, s in enumerate(segs):
            b = self.seg_bodies[0] if self.desc.rigid else self.seg_bodies[i]
            off = s.pos if self.desc.rigid else np.zeros(3)
            h = np.maximum(s.half - ESCAPE_SHRINK, 0.25 * s.half)
            for c in [np.zeros(3)] + [np.array([a, b_, c_]) * h for a in (-1, 1) for b_ in (-1, 1) for c_ in (-1, 1)]:
                bodies.append(b)
                pts.append(off + c)
        return np.array(bodies), np.array(pts)

    def content_world_points(self):
        b, p = self._content_pts
        R = self.d.xmat[b].reshape(-1, 3, 3)
        return self.d.xpos[b] + np.einsum("nij,nj->ni", R, p)

    def bag_surface(self):
        """Closed triangle surface of the bag (for containment)."""
        if self.bag.kind == "flex":
            v = self.d.flexvert_xpos.copy()
            f = self.meta["faces"]
            loop = self.meta["boundary"]
            if not len(loop):                  # closed mesh (make_kraft_coarse.py)
                return v, f
            c = v[loop].mean(0)
            # fan over the open mouth/flap boundary (order the loop by angle around its centre in the y-z plane)
            ang = np.arctan2(v[loop, 2] - c[2], v[loop, 1] - c[1])
            loop = loop[np.argsort(ang)]
            v = np.vstack([v, c])
            fan = np.c_[loop, np.roll(loop, -1), np.full(len(loop), len(v) - 1)]
            return v, np.vstack([f, fan])
        nx, ny = self.meta["grid"]
        cx, wy = self.meta["cell"], self.meta["widths"]
        corners = {}
        for p in "FB":
            acc = np.zeros((nx + 1, ny + 1, 3))
            cnt = np.zeros((nx + 1, ny + 1, 1))
            for i in range(nx):
                for j in range(ny):
                    b = self.m.body(f"bag{p}_{i}_{j}").id
                    R = self.d.xmat[b].reshape(3, 3)
                    for di in (0, 1):
                        for dj in (0, 1):
                            acc[i + di, j + dj] += self.d.xpos[b] + R @ [(di - .5) * cx, (dj - .5) * wy[j], 0]
                            cnt[i + di, j + dj] += 1
            corners[p] = acc / cnt
        Fv, Bv = corners["F"], corners["B"]
        edge = np.zeros((nx + 1, ny + 1), bool)
        edge[[0, -1], :] = edge[:, [0, -1]] = True
        Fv[edge] = Bv[edge] = 0.5 * (Fv[edge] + Bv[edge])
        idx = lambda p, i, j: (0 if p == "F" else (nx + 1) * (ny + 1)) + i * (ny + 1) + j
        tris = []
        for i in range(nx):
            for j in range(ny):
                a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
                tris += [[idx("F", *a), idx("F", *b), idx("F", *c)], [idx("F", *a), idx("F", *c), idx("F", *d)],
                         [idx("B", *a), idx("B", *c), idx("B", *b)], [idx("B", *a), idx("B", *d), idx("B", *c)]]
        return np.vstack([Fv.reshape(-1, 3), Bv.reshape(-1, 3)]), np.array(tris)

    def outside_fraction(self):
        """(centres outside, corner points outside) as fractions, by generalized winding number."""
        v, f = self.bag_surface()
        pts = self.content_world_points()
        w = winding_number(v, f, pts)
        inside = np.abs(w) >= 0.5
        b, _ = self._content_pts
        is_center = np.zeros(len(pts), bool)
        is_center[::9] = True
        return float(1 - inside[is_center].mean()), float(1 - inside[~is_center].mean())

    def penetration(self):
        """Largest penetration (m) between bag and content: bag vertices inside a content box, and content box
        corners outside the closed bag surface (distance to it). Small values are contact compliance; a film
        passed through the content shows up as a depth of the order of the segment size."""
        v, f = self.bag_surface()
        verts = self.bag_points() if self.bag.kind == "plates" else self.d.flexvert_xpos
        depth = 0.0
        for i, s in enumerate(self.desc.segments):
            b = self.seg_bodies[0] if self.desc.rigid else self.seg_bodies[i]
            off = s.pos if self.desc.rigid else np.zeros(3)
            R = self.d.xmat[b].reshape(3, 3)
            local = (verts - self.d.xpos[b]) @ R - off
            inside = np.min(s.half - np.abs(local), axis=1)
            depth = max(depth, float(inside.max()))
            corners = np.array([[a, b_, c] for a in (-1, 1) for b_ in (-1, 1) for c in (-1, 1)]) * s.half
            world = self.d.xpos[b] + (corners + off) @ R.T
            out = np.abs(winding_number(v, f, world)) < 0.5
            if out.any():
                depth = max(depth, float(point_triangle_distance(world[out], v, f).max()))
        return depth

    def bag_failure(self, pad_vel=np.zeros(3), gripped=False):
        """Content-model failure (README) plus bag evidence: non-finite bag, content centre outside the bag."""
        f = self.failure(pad_vel, gripped)
        if f is not None and not f.startswith("flung"):
            return f, None
        if not np.all(np.isfinite(self.bag_points())):
            return "non-finite bag state", None
        centers_out, corners_out = self.outside_fraction()
        if centers_out > 0:
            return f"content outside the bag ({centers_out:.0%} of segment centres)", None
        return None, f            # second value: speed diagnostic only (test/policy/README.md)
