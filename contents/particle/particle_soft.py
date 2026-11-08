"""Particle-bond soft content: nodes joined by bonds, wrapped in a contact-only flex surface.

Flat and crumpled items share the node lattice, the force law and the contact surface.
Flat items carry compression in columns along the local normal, with shear as a separate
tangential force. Crumpled items carry compression along every bond. Forces are computed
in Python and applied through qfrc_applied; history advances after each step.
All presets are authored, not measured. SI units; targets MuJoCo 3.14.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

RADIUS = 0.007
PATCH_FACES = 32
TOOL_CONTACT = ('contype="1" conaffinity="2" condim="3" friction="{mu} 0.005 0.0001" '
                'solref="-2e4 -60" solimp="0.95 0.99 0.0005"')


@dataclass(frozen=True)
class SoftX:
    mass: float = 0.3
    eps_knee: float = 0.2  # item-level strain and pressure
    p_knee: float = 200.0
    eps_lock: float = 0.65
    p_lock: float = 2e4
    hysteresis: float = 0.5  # friction as a fraction of the knee force
    creep_eps: float = 0.05
    creep_tau: float = 2.0
    p_creep: float = 1000.0
    k_in_plane: float = 400.0
    k_bend: float = 10.0
    k_shear: float = 20.0
    slide_range: float = 0.03
    loose_frac: float = 0.3  # crumpled only: bonds that carry no tension
    k_tension: float = 40.0  # crumpled only
    stiffness_sigma: float = 0.5  # crumpled only: log-normal spatial variation
    friction: float = 0.7
    seed: int = 0
    # closed-loop corrections written by calibrate.py
    g_knee: float = 1.0
    g_lock: float = 1.0
    g_creep: float = 1.0
    g_hyst: float = 1.0


class Bonds:
    """Struct of arrays for bond parameters and history."""

    def __init__(self):
        self.rows = []

    def add(self, i, j, L0, **kw):
        row = dict(i=i, j=j, L0=L0, kt=0.0, s1=0.0, s2=0.0, e1=1e9, e2=1e9, klock=0.0, f=0.0, k3=0.0,
                   tau=1.0, vmax=0.0, c=0.05)
        row.update(kw)
        self.rows.append(row)

    def finish(self):
        for key in self.rows[0]:
            dtype = np.int64 if key in ('i', 'j') else np.float64
            setattr(self, key, np.array([r[key] for r in self.rows], dtype=dtype))
        n = len(self.rows)
        self.v, self.s = np.zeros(n), np.zeros(n)
        self.kf = np.maximum(2 * self.s1, 1e-6)
        del self.rows

    def compression(self, delta):
        """Elastic-plus-slider push force for compression delta; records state for history."""
        e = np.maximum(delta - self.v, 0.0)
        elastic = (self.s1 * np.minimum(e, self.e1) + self.s2 * np.clip(e - self.e1, 0, np.maximum(self.e2 - self.e1, 0))
                   + self.klock * np.maximum(e - self.e2, 0))
        slip = np.clip(self.kf * (delta - self.s), -self.f, self.f)
        self._state = (delta, elastic, slip)
        return np.maximum(elastic + slip, 0.0)

    def advance(self, dt):
        delta, elastic, slip = self._state
        active = self.f > 0
        self.s[active] = (delta - slip / self.kf)[active]
        creeping = self.k3 > 0
        rate = (elastic[creeping] / self.k3[creeping] - self.v[creeping]) / self.tau[creeping]
        self.v[creeping] = np.clip(self.v[creeping] + rate * dt, 0.0, self.vmax[creeping])


def _law(F1, FL, e1, eL, klock=1500.0):
    return dict(s1=F1 / e1, s2=max(FL - F1, 0.0) / max(eL - e1, 1e-9), e1=e1, e2=eL, klock=klock)


def _creep(x, force, delta):
    if x.creep_eps <= 0:
        return {}
    delta = delta * x.g_creep
    return dict(k3=force / delta, tau=x.creep_tau, vmax=3 * delta)


def _lattice_surface(n, idx):
    """Triangles on the six faces of an n[0] x n[1] x n[2] lattice, grouped into tiles."""
    tiles = []
    for axis in range(3):
        u_axis, v_axis = [a for a in range(3) if a != axis]
        for level in (0, n[axis] - 1):
            tile = {}
            for u in range(n[u_axis] - 1):
                for v in range(n[v_axis] - 1):
                    corner = [0, 0, 0]
                    corner[axis] = level

                    def at(du, dv):
                        c = list(corner)
                        c[u_axis], c[v_axis] = u + du, v + dv
                        return idx(*c)
                    a, b, c, d = at(0, 0), at(1, 0), at(1, 1), at(0, 1)
                    key = (u // 4, v // 4)
                    tile.setdefault(key, []).extend([(a, b, c), (a, c, d)])
            tiles.extend(tile.values())
    return tiles


def build_flat(x: SoftX, size=(0.30, 0.25), thick=0.05, spacing=0.025, nz=3):
    nx, ny = int(round(size[0] / spacing)) + 1, int(round(size[1] / spacing)) + 1
    gap = (thick - 2 * RADIUS) / (nz - 1)
    idx = lambda a, b, c: a * ny * nz + b * nz + c
    pos = np.zeros((nx * ny * nz, 3))
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                pos[idx(a, b, c)] = (a * spacing - size[0] / 2, b * spacing - size[1] / 2, RADIUS + c * gap)

    area = spacing ** 2
    e_per_gap = lambda eps: eps * thick / (nz - 1)
    law = _law(x.p_knee * x.g_knee * area, x.p_lock * x.g_lock * area, e_per_gap(x.eps_knee), min(e_per_gap(x.eps_lock), gap * 0.95))
    creep = _creep(x, x.p_creep * area, e_per_gap(x.creep_eps))
    fric = x.hysteresis * x.g_hyst * x.p_knee * area

    bonds, cols = Bonds(), Bonds()
    nb = []
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                i = idx(a, b, c)
                for da, db in ((1, 0), (0, 1)):
                    if a + da < nx and b + db < ny:
                        bonds.add(i, idx(a + da, b + db, c), spacing, kt=x.k_in_plane, s1=x.k_in_plane * 0.05,
                                  s2=x.k_in_plane * 0.05, c=0.08)
                    if a + 2 * da < nx and b + 2 * db < ny:
                        bonds.add(i, idx(a + 2 * da, b + 2 * db, c), 2 * spacing, kt=x.k_bend, s1=x.k_bend,
                                  s2=x.k_bend, c=0.01)
                if a + 1 < nx and b + 1 < ny:
                    L = spacing * np.sqrt(2)
                    kw = dict(kt=x.k_in_plane * 0.3, s1=x.k_in_plane * 0.05, s2=x.k_in_plane * 0.05, c=0.03)
                    bonds.add(i, idx(a + 1, b + 1, c), L, **kw)
                    bonds.add(idx(a + 1, b, c), idx(a, b + 1, c), L, **kw)
                if c + 1 < nz:
                    cols.add(i, idx(a, b, c + 1), gap, kt=2.0, f=fric, c=0.03, **law, **creep)
                    nb.append((idx(max(a - 1, 0), b, c), idx(min(a + 1, nx - 1), b, c),
                               idx(a, max(b - 1, 0), c), idx(a, min(b + 1, ny - 1), c)))
    bonds.finish()
    cols.finish()
    cols.nb = np.array(nb)
    tiles = _lattice_surface((nx, ny, nz), idx)
    return pos, bonds, cols, tiles, dict(kind='flat', idx=(nx, ny, nz), area=size[0] * size[1], thick=thick)


def _smooth_noise(pos, rng, n_modes=12, wavelength=0.08):
    k = rng.normal(size=(n_modes, 3)) * 2 * np.pi / wavelength / np.sqrt(3)
    phase = rng.uniform(0, 2 * np.pi, n_modes)
    return np.sqrt(2 / n_modes) * np.cos(pos @ k.T + phase).sum(axis=1)


def build_blob(x: SoftX, radii=(0.13, 0.10, 0.07), grid=(10, 8, 6), spacing=0.025):
    """Map a lattice onto an irregular ellipsoid, so the surface triangles stay known."""
    rng = np.random.default_rng(x.seed)
    nx, ny, nz = grid
    idx = lambda a, b, c: a * ny * nz + b * nz + c
    harmonics = rng.normal(size=(6, 4))
    pos = np.zeros((nx * ny * nz, 3))
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                u, v, w = [2 * t / (n - 1) - 1 for t, n in ((a, nx), (b, ny), (c, nz))]
                # cube-to-sphere map
                s = np.array([u * np.sqrt(1 - v * v / 2 - w * w / 2 + v * v * w * w / 3),
                              v * np.sqrt(1 - u * u / 2 - w * w / 2 + u * u * w * w / 3),
                              w * np.sqrt(1 - u * u / 2 - v * v / 2 + u * u * v * v / 3)])
                bump = 1 + 0.12 * sum(np.sin((k + 1) * (h[0] * s[0] + h[1] * s[1] + h[2] * s[2]) + h[3])
                                      for k, h in enumerate(harmonics)) / np.sqrt(len(harmonics))
                pos[idx(a, b, c)] = s * np.array(radii) * bump + rng.uniform(-0.1, 0.1, 3) * spacing
    pos[:, 2] += RADIUS + 0.001 - pos[:, 2].min()

    distance = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    field = np.exp(x.stiffness_sigma * _smooth_noise(pos, rng))
    height = pos[:, 2].max() + RADIUS
    mean_spacing = np.median(np.sort(distance, axis=1)[:, 1:7])
    area_b = mean_spacing ** 2
    gaps = height / mean_spacing
    e_per = lambda eps: eps * height / gaps

    bonds = Bonds()
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            L = distance[i, j]
            if L > 1.7 * mean_spacing:
                continue
            f = np.sqrt(field[i] * field[j])
            law = _law(x.p_knee * x.g_knee * area_b * f, x.p_lock * x.g_lock * area_b * f,
                       e_per(x.eps_knee), min(e_per(x.eps_lock), 0.8 * L))
            creep = _creep(x, x.p_creep * area_b * f, e_per(x.creep_eps))
            kt = 0.0 if rng.random() < x.loose_frac else x.k_tension
            bonds.add(i, j, float(L), kt=kt, f=x.hysteresis * x.g_hyst * x.p_knee * area_b * f, c=0.02, **law, **creep)
    bonds.finish()
    tiles = _lattice_surface(grid, idx)
    return pos, bonds, None, tiles, dict(kind='blob', area=float(np.pi * radii[0] * radii[1]), thick=float(height), radii=radii)


def build_pillow(x: SoftX, size=(0.50, 0.35, 0.12), grid=(11, 8, 5), corner_taper=0.15, k_skin=600.0):
    """Lattice pillow: thick center, thin corners. Filling bonds carry the compression law in every
    direction; bonds between surface nodes add fabric-like tension."""
    nx, ny, nz = grid
    idx = lambda a, b, c: (a * ny + b) * nz + c
    pos = np.zeros((nx * ny * nz, 3))
    for a in range(nx):
        for b in range(ny):
            u, v = 2 * a / (nx - 1) - 1, 2 * b / (ny - 1) - 1
            profile = np.sqrt(max((1 - abs(u) ** 4) * (1 - abs(v) ** 4), 0.0))
            thick = size[2] * (corner_taper + (1 - corner_taper) * profile)
            for c in range(nz):
                pos[idx(a, b, c)] = (u * size[0] / 2, v * size[1] / 2, (size[2] - thick) / 2 + c / (nz - 1) * thick)
    pos[:, 2] += RADIUS + 0.001 - pos[:, 2].min()
    surface = np.zeros(len(pos), bool)
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                surface[idx(a, b, c)] = a in (0, nx - 1) or b in (0, ny - 1) or c in (0, nz - 1)
    spacing = float(np.mean([size[0] / (nx - 1), size[1] / (ny - 1)]))
    area_b = spacing ** 2
    height = size[2]
    e_per = lambda eps: eps * height / (nz - 1)
    bonds = Bonds()
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                i = idx(a, b, c)
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            a2, b2, c2 = a + da, b + db, c + dc
                            if not (0 <= a2 < nx and 0 <= b2 < ny and 0 <= c2 < nz):
                                continue
                            j = idx(a2, b2, c2)
                            if j <= i:
                                continue
                            L = float(np.linalg.norm(pos[j] - pos[i]))
                            share = abs(pos[j, 2] - pos[i, 2]) / L  # vertical bonds carry the thickness law
                            law = _law(max(x.p_knee * x.g_knee * area_b * share, 1e-4), max(x.p_lock * x.g_lock * area_b * share, 2e-4),
                                       e_per(x.eps_knee), min(e_per(x.eps_lock), 0.8 * L))
                            creep = _creep(x, max(x.p_creep * area_b * share, 1e-4), e_per(x.creep_eps))
                            kt = k_skin if (surface[i] and surface[j] and dc == 0) else x.k_tension
                            bonds.add(i, j, L, kt=kt, f=x.hysteresis * x.g_hyst * x.p_knee * area_b * share, c=0.02, **law, **creep)
    bonds.finish()
    tiles = _lattice_surface(grid, idx)
    return pos, bonds, None, tiles, dict(kind='pillow', area=size[0] * size[1], thick=height)


def _flex_xml(tiles, x):
    out = []
    for t, faces in enumerate(tiles):
        faces = np.array(faces)
        verts = np.unique(faces)
        lookup = {int(v): k for k, v in enumerate(verts)}
        elements = ' '.join(str(lookup[int(v)]) for v in faces.ravel())
        bodies = ' '.join(f'n{v}' for v in verts)
        zeros = ' '.join(['0'] * (3 * len(verts)))
        out.append(f'<flex name="skin{t}" dim="2" radius="{RADIUS}" rgba=".8 .5 .35 1" body="{bodies}" vertex="{zeros}" '
                   f'element="{elements}"><contact selfcollide="none" contype="2" conaffinity="1" condim="3" '
                   f'friction="{x.friction} 0.005 0.0001" solref="-2e4 -60" solimp="0.95 0.99 0.0005"/>'
                   '<edge damping="1e-12"/></flex>')
    return '<deformable>' + ''.join(out) + '</deformable>'


class Soft:
    """Nodes in a MuJoCo model plus the force law."""

    def __init__(self, built, x: SoftX, tools_xml, table_xml, offset=(0, 0, 0), dt=2e-4):
        pos, self.bonds, self.cols, tiles, self.meta = built
        self.x, self.dt = x, dt
        n = len(pos)
        m_node = x.mass / n
        nodes = []
        for i, p in enumerate(pos + np.asarray(offset)):
            nodes.append(f'<body name="n{i}" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}">'
                         '<joint type="slide" axis="1 0 0" damping="0.0015"/><joint type="slide" axis="0 1 0" damping="0.0015"/>'
                         '<joint type="slide" axis="0 0 1" damping="0.0015"/>'
                         f'<geom type="sphere" size="0.002" mass="{m_node:.6g}" contype="0" conaffinity="0" rgba=".5 .3 .2 1"/></body>')
        xml = (f'<mujoco><option timestep="{dt}" integrator="Euler" solver="CG" iterations="30" tolerance="1e-6" jacobian="sparse"/>'
               '<size memory="300M"/><visual><global offwidth="1280" offheight="800"/></visual><worldbody>'
               '<light pos=".4 -.4 1.2" dir="-.3 .3 -1"/>' + table_xml + ''.join(nodes) + tools_xml + '</worldbody>'
               + _flex_xml(tiles, x) + '</mujoco>')
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        first = self.model.body('n0').jntadr[0]
        self.q0 = self.model.jnt_qposadr[first]
        self.d0 = self.model.jnt_dofadr[first]
        self.n = n
        self.start = np.asarray(pos) + np.asarray(offset)

    def positions(self):
        return self.start + self.data.qpos[self.q0:self.q0 + 3 * self.n].reshape(-1, 3)

    def velocities(self):
        return self.data.qvel[self.d0:self.d0 + 3 * self.n].reshape(-1, 3)

    def _bond_forces(self, p, vel, forces):
        b = self.bonds
        d = p[b.j] - p[b.i]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        nrm = d / L[:, None]
        stretch = L - b.L0
        mag = (b.compression(np.maximum(-stretch, 0.0)) - b.kt * np.maximum(stretch, 0.0)
               - b.c * np.einsum('ij,ij->i', vel[b.j] - vel[b.i], nrm))
        np.add.at(forces, b.j, mag[:, None] * nrm)
        np.add.at(forces, b.i, -mag[:, None] * nrm)

    def _column_forces(self, p, vel, forces):
        """Compression acts along the local normal, so it cannot buckle a column; shear is tangential."""
        c = self.cols
        normal = np.cross(p[c.nb[:, 1]] - p[c.nb[:, 0]], p[c.nb[:, 3]] - p[c.nb[:, 2]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        d = p[c.j] - p[c.i]
        gap = np.einsum('ij,ij->i', d, normal)
        stretch = gap - c.L0
        rel = vel[c.j] - vel[c.i]
        mag = (c.compression(np.maximum(-stretch, 0.0)) - c.kt * np.maximum(stretch, 0.0)
               - c.c * np.einsum('ij,ij->i', rel, normal))
        tangent = d - gap[:, None] * normal
        rel_t = rel - np.einsum('ij,ij->i', rel, normal)[:, None] * normal
        length = np.linalg.norm(tangent, axis=1, keepdims=True)
        hard = 300.0 * np.maximum(length - self.x.slide_range, 0.0) / np.maximum(length, 1e-9)
        shear = -(self.x.k_shear + hard) * tangent - 0.02 * rel_t
        total = mag[:, None] * normal + shear
        np.add.at(forces, c.j, total)
        np.add.at(forces, c.i, -total)

    def apply(self):
        p, vel = self.positions(), self.velocities()
        forces = np.zeros_like(p)
        self._bond_forces(p, vel, forces)
        if self.cols is not None:
            self._column_forces(p, vel, forces)
        self.data.qfrc_applied[self.d0:self.d0 + 3 * self.n] = forces.ravel()

    def advance_history(self):
        self.bonds.advance(self.dt)
        if self.cols is not None:
            self.cols.advance(self.dt)

    def check(self):
        v = self.velocities()
        if not np.isfinite(v).all() or np.abs(v).max() > 30:
            raise FloatingPointError(f'unstable at t={self.data.time:.3f}')

    def com(self):
        return self.positions().mean(axis=0)


def tool_force(model, data, geoms):
    total = np.zeros(3)
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom[0] in geoms or c.geom[1] in geoms:
            mujoco.mj_contactForce(model, data, i, wrench)
            sign = -1.0 if c.geom[0] in geoms else 1.0
            total += sign * c.frame.reshape(3, 3).T @ wrench[:3]
    return total


def render(soft, path, azimuth=135, elevation=-25, distance=0.75, lookat=None):
    import os
    import struct
    import zlib
    os.environ.setdefault('MUJOCO_GL', 'egl')
    renderer = mujoco.Renderer(soft.model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = soft.com() if lookat is None else lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    renderer.update_scene(soft.data, cam)
    img = renderer.render().copy()
    renderer.close()
    h, w, _ = img.shape
    raw = b''.join(b'\0' + img[i].tobytes() for i in range(h))

    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff)
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                 + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
