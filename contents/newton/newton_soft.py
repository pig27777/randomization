"""Soft content experiments on Newton VBD: real cloth with self-contact versus tetrahedral soft body.

Usage: newton_soft.py <variant> <test> [--out DIR]
variant: cloth_folded | cloth_crumpled | tet_flat | tet_blob
test:    speed | squeeze | finger | top_pinch | edge_pinch | make_crumple
All material values are authored, not measured. SI units.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

HERE = Path(__file__).resolve().parent

MASS = 0.3
FRAME_DT = 0.01
SUBSTEPS = 5
ITERATIONS = 10
FRICTION_EPS = 1e-2
TAG = ''
CONTACT_KE = 3e3
GRIP_SQUEEZE = 0.25
TOP_GAP = 0.012
LIFT_STEP = 0.0002  # per frame, 2 cm/s
Z0 = 0.002
TABLE_Z = 0.2


# ---------- geometry ----------

def folded_sheet(layers=4, length=0.30, width=0.25, h=0.015, gap=0.006):
    """One continuous sheet folded back and forth; the bent rest shape stores the folds."""
    nu = int(round(length / h))
    path = []
    for layer in range(layers):
        z = layer * gap
        xs = np.linspace(-length / 2, length / 2, nu + 1)
        xs = xs if layer % 2 == 0 else xs[::-1]
        path.extend((x, z) for x in xs)
        if layer < layers - 1:
            side = xs[-1]
            sign = np.sign(side)
            for angle in (np.pi / 4, np.pi / 2, 3 * np.pi / 4):
                path.append((side + sign * gap / 2 * np.sin(angle), z + gap / 2 * (1 - np.cos(angle))))
    path = np.array(path)
    ny = int(round(width / h))
    ys = np.linspace(-width / 2, width / 2, ny + 1)
    verts = np.array([(px, y, pz) for px, pz in path for y in ys], dtype=np.float32)
    idx = lambda a, b: a * (ny + 1) + b
    faces = []
    for a in range(len(path) - 1):
        for b in range(ny):
            faces += [idx(a, b), idx(a + 1, b), idx(a + 1, b + 1), idx(a, b), idx(a + 1, b + 1), idx(a, b + 1)]
    area = _tri_area(verts, faces)
    return verts, faces, area


def flat_sheet(length=0.60, width=0.50, h=0.015):
    nx, ny = int(round(length / h)), int(round(width / h))
    verts = np.array([(a * h - length / 2, b * h - width / 2, 0.0) for a in range(nx + 1) for b in range(ny + 1)], dtype=np.float32)
    idx = lambda a, b: a * (ny + 1) + b
    faces = []
    for a in range(nx):
        for b in range(ny):
            faces += [idx(a, b), idx(a + 1, b), idx(a + 1, b + 1), idx(a, b), idx(a + 1, b + 1), idx(a, b + 1)]
    return verts, faces, _tri_area(verts, faces)


def _tri_area(verts, faces):
    t = verts[np.array(faces).reshape(-1, 3)]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def lattice_tets(positions, n):
    """Five tetrahedra per cell with alternating parity, so faces match between cells."""
    nx, ny, nz = n
    idx = lambda a, b, c: a * ny * nz + b * nz + c
    tets = []
    for a in range(nx - 1):
        for b in range(ny - 1):
            for c in range(nz - 1):
                v = [idx(a + i, b + j, c + k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
                if (a + b + c) % 2 == 0:
                    pattern = [(0, 1, 2, 4), (1, 3, 2, 7), (1, 4, 5, 7), (2, 7, 6, 4), (1, 2, 4, 7)]
                else:
                    pattern = [(0, 1, 3, 5), (0, 3, 2, 6), (0, 5, 4, 6), (3, 5, 6, 7), (0, 3, 5, 6)]
                for t in pattern:
                    p = positions[[v[i] for i in t]]
                    vol = np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0])
                    tets += [v[t[0]], v[t[1]], v[t[2]], v[t[3]]] if vol > 0 else [v[t[0]], v[t[2]], v[t[1]], v[t[3]]]
    return tets


def flat_lattice(size=(0.30, 0.25, 0.05), cell=0.025):
    n = (int(round(size[0] / cell)) + 1, int(round(size[1] / cell)) + 1, 3)
    pos = np.array([(a * cell - size[0] / 2, b * cell - size[1] / 2, c * size[2] / 2)
                    for a in range(n[0]) for b in range(n[1]) for c in range(n[2])], dtype=np.float32)
    return pos, lattice_tets(pos, n)


def blob_lattice(radii=(0.13, 0.10, 0.07), grid=(10, 8, 6), seed=0):
    rng = np.random.default_rng(seed)
    harmonics = rng.normal(size=(6, 4))
    nx, ny, nz = grid
    pos = np.zeros((nx * ny * nz, 3), dtype=np.float32)
    i = 0
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                u, v, w = [2 * t / (m - 1) - 1 for t, m in ((a, nx), (b, ny), (c, nz))]
                s = np.array([u * np.sqrt(1 - v * v / 2 - w * w / 2 + v * v * w * w / 3),
                              v * np.sqrt(1 - u * u / 2 - w * w / 2 + u * u * w * w / 3),
                              w * np.sqrt(1 - u * u / 2 - v * v / 2 + u * u * v * v / 3)])
                bump = 1 + 0.12 * sum(np.sin((k + 1) * (h[0] * s[0] + h[1] * s[1] + h[2] * s[2]) + h[3])
                                      for k, h in enumerate(harmonics)) / np.sqrt(6)
                pos[i] = s * np.array(radii) * bump
                i += 1
    pos[:, 2] -= pos[:, 2].min()
    return pos, lattice_tets(pos, grid)


# ---------- content ----------

def add_content(builder, variant, base_z, out):
    start = builder.particle_count
    info = dict(variant=variant)
    if variant == 'cloth_folded':
        verts, faces, area = folded_sheet()
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, base_z), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=verts.tolist(),
                               indices=faces, density=MASS / area, tri_ke=5e3, tri_ka=5e3, tri_kd=1e-1, edge_ke=0.05, edge_kd=1e-3,
                               particle_radius=0.003)
        info.update(cloth=True, faces=faces)
    elif variant in ('cloth_crumpled', 'crumple_source'):
        verts, faces, area = flat_sheet()
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, base_z), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=verts.tolist(),
                               indices=faces, density=MASS / area, tri_ke=5e3, tri_ka=5e3, tri_kd=1e-1, edge_ke=0.05, edge_kd=1e-3,
                               particle_radius=0.003)
        info.update(cloth=True, faces=faces)
    else:
        pos, tets = flat_lattice() if variant == 'tet_flat' else blob_lattice()
        volume = 0.3 * 0.25 * 0.05 if variant == 'tet_flat' else 4 / 3 * np.pi * 0.13 * 0.10 * 0.07
        builder.add_soft_mesh(pos=wp.vec3(0, 0, base_z), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=pos.tolist(),
                              indices=tets, density=MASS / volume, k_mu=1500.0, k_lambda=1000.0, k_damp=1.0, particle_radius=0.004)
        info.update(cloth=False, faces=None)
    info.update(start=start, end=builder.particle_count)
    return info


def make_solver(model, info):
    return newton.solvers.SolverVBD(
        model, iterations=ITERATIONS, friction_epsilon=FRICTION_EPS, rigid_compliant_alm=True, rigid_body_contact_buffer_size=256,
        rigid_body_particle_contact_buffer_size=2048,
        particle_enable_self_contact=bool(info['cloth']),
        particle_self_contact_radius=0.002, particle_self_contact_margin=0.005)


class Sim:
    def __init__(self, builder, info, joints=None):
        builder.color(include_bending=bool(info['cloth']))
        self.info = info
        self.model = builder.finalize()
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = CONTACT_KE * 1e-3
        self.model.soft_contact_mu = 0.7
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = make_solver(self.model, info)
        self.pipeline = newton.CollisionPipeline(self.model, broad_phase='nxn', soft_contact_gap=0.006,
                                                 enable_rigid_soft_full_surface_contact=True)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        if self.model.joint_count:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
            wp.copy(self.s1.body_q, self.s0.body_q)
        self.dt = FRAME_DT / SUBSTEPS
        self.targets = np.array(self.model.joint_q.numpy(), dtype=np.float32) if self.model.joint_dof_count else None
        self.frames = 0
        if self.targets is not None:
            self.apply_targets()

    def apply_targets(self):
        self.control.joint_target_q.assign(wp.array(self.targets, dtype=float, device=self.model.device))

    def frame(self, n=1):
        for _ in range(n):
            for _ in range(SUBSTEPS):
                self.s0.clear_forces()
                self.pipeline.collide(self.s0, self.contacts)
                self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
                self.s0, self.s1 = self.s1, self.s0
            self.frames += 1
        q = self.s0.particle_q.numpy()
        if not np.isfinite(q).all() or np.abs(q).max() > 20:
            raise FloatingPointError(f'unstable at frame {self.frames}')

    def q(self):
        return self.s0.particle_q.numpy()[self.info['start']:self.info['end']]

    def body_z(self, body):
        return float(self.s0.body_q.numpy()[body][2])

    def set_positions(self, pos):
        all_q = self.s0.particle_q.numpy()
        all_q[self.info['start']:self.info['end']] = pos
        self.s0.particle_q.assign(all_q)
        self.s1.particle_q.assign(all_q)


def save_state(sim, path, extra_boxes=()):
    np.savez_compressed(path, q=sim.q(), faces=np.array(sim.info['faces'] if sim.info['faces'] is not None else [], dtype=np.int32),
                        body_q=sim.s0.body_q.numpy() if sim.model.body_count else np.zeros((0, 7)), boxes=np.array(extra_boxes, dtype=np.float32))


# ---------- crumple generation ----------

def make_crumple(out, seed=0):
    rng = np.random.default_rng(seed)
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = add_content(builder, 'crumple_source', 0.5, out)
    sim = Sim(builder, info)
    q0 = sim.q().copy()
    # Hang from three nearby points, then drop: the sheet collapses into irregular folds.
    picks = np.argsort(np.linalg.norm(q0[:, :2] - rng.uniform(-0.08, 0.08, 2), axis=1))[[0, 40, 90]]
    flags = sim.model.particle_flags.numpy()
    for p in picks:
        flags[info['start'] + p] &= ~int(newton.ParticleFlags.ACTIVE)
    sim.model.particle_flags.assign(flags)
    sim.frame(150)
    for p in picks:
        flags[info['start'] + p] |= int(newton.ParticleFlags.ACTIVE)
    sim.model.particle_flags.assign(flags)
    sim.frame(250)
    q = sim.q()
    np.save(out / 'crumple_state.npy', q)
    save_state(sim, out / 'crumple_state_view.npz')
    return dict(height=float(q[:, 2].max()), footprint=[float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))], n=len(q))


def place_content(sim, variant, base_z, out):
    if variant == 'cloth_crumpled':
        q = np.load(out / 'crumple_state.npy')
        q = q - [q[:, 0].mean(), q[:, 1].mean(), q[:, 2].min() - base_z]
        sim.set_positions(q)


# ---------- tests ----------

def test_speed(variant, out):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = add_content(builder, variant, Z0, out)
    sim = Sim(builder, info)
    place_content(sim, variant, Z0, out)
    sim.frame(20)
    wp.synchronize()
    t0 = time.perf_counter()
    sim.frame(100)
    wp.synchronize()
    wall = time.perf_counter() - t0
    q = sim.q()
    save_state(sim, out / f'{variant}_settled.npz')
    return dict(particles=len(q), frames_per_s=100 / wall, realtime_factor=100 * FRAME_DT / wall, height=float(q[:, 2].max()),
                footprint=[float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))], max_v=float(np.abs(sim.s0.particle_qd.numpy()).max()))


GANTRY_KE = 5e4


def press_rig(builder, half, xy, top):
    link = builder.add_link(xform=wp.transform(wp.vec3(xy[0], xy[1], top), wp.quat_identity()), mass=0.5, label='press')
    cfg = newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.8)
    builder.add_shape_box(link, hx=half[0], hy=half[1], hz=0.01, cfg=cfg)
    joint = builder.add_joint_prismatic(parent=-1, child=link, axis=wp.vec3(0, 0, 1), parent_xform=wp.transform(wp.vec3(xy[0], xy[1], 0), wp.quat_identity()),
                                        target_ke=GANTRY_KE, target_kd=2e3, target_pos=top, label='press_z')
    builder.add_articulation([joint], label='press')
    builder.joint_q[0] = top
    return link


def press(variant, half, xy, out, f_cap=60.0, depth_cap=0.04, speed=0.02, hold=True, view=None):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = add_content(builder, variant, Z0, out)
    top = 0.32
    link = press_rig(builder, half, xy, top)
    sim = Sim(builder, info)
    place_content(sim, variant, Z0, out)
    sim.frame(60)
    surface = float(sim.q()[:, 2].max())
    # gantry approaches fast to just above the top of the content
    z = top
    while z > surface + 0.02:
        z -= 0.002
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
    sim.frame(30)
    # force is the drive reaction relative to the free baseline
    baseline = GANTRY_KE * (z - sim.body_z(link))

    def force():
        return GANTRY_KE * (sim.targets[0] - sim.body_z(link)) - baseline

    while force() < 0.2 and z > -0.05:
        z -= speed * FRAME_DT
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
    z0 = sim.body_z(link)
    load = []
    while True:
        z -= speed * FRAME_DT
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
        depth, f = z0 - sim.body_z(link), force()
        load.append((depth, f))
        if f >= f_cap or depth >= depth_cap:
            break
    load = np.array(load)
    res = dict(surface=surface, peak_force=float(load[-1, 1]), peak_depth=float(load[-1, 0]),
               force_at_depth={f'{mm}mm': float(np.interp(mm * 1e-3, load[:, 0], load[:, 1])) for mm in (2, 5, 8, 12, 16, 20, 30) if load[-1, 0] >= mm * 1e-3},
               largest_drop=float(np.max(np.maximum.accumulate(load[:, 1]) - load[:, 1])))
    if not hold:
        return res
    h = []
    for _ in range(40):
        sim.frame(5)
        h.append(force())
    if view:
        save_state(sim, view, [(link, half[0], half[1], 0.01)])
    unload = []
    while z < z0 + 0.008:
        z += speed * FRAME_DT
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
        unload.append((z0 - sim.body_z(link), force()))
    unload = np.array(unload)
    inside = unload[unload[:, 0] >= 0]
    area_l, area_u = np.trapezoid(load[:, 1], load[:, 0]), -np.trapezoid(inside[:, 1], inside[:, 0])
    touching = np.where(unload[:, 1] > 0.2)[0]
    res.update(hysteresis=float((area_l - area_u) / max(area_l, 1e-9)), residual_mm=float(unload[touching[-1], 0] * 1e3) if len(touching) else 0.0,
               relax=float(np.mean(h[-5:]) / max(np.mean(h[:2]), 1e-9)))
    return res


def test_squeeze(variant, out):
    return press(variant, (0.19, 0.16), (0, 0), out, view=out / f'{variant}_squeeze.npz')


def test_finger(variant, out):
    res = {}
    for name, half, xy in (('20mm', (.01, .01), (0, 0)), ('20mm side', (.01, .01), (.05, .04)), ('60mm', (.03, .03), (0, 0)), ('platen', (.19, .16), (0, 0))):
        try:
            r = press(variant, half, xy, out, f_cap=40.0, depth_cap=0.02, hold=False)
            res[name] = dict(force_at_depth=r['force_at_depth'], peak_force=r['peak_force'], peak_depth=r['peak_depth'], largest_drop=r['largest_drop'])
        except FloatingPointError as err:
            res[name] = dict(FAILED=str(err))
    return res


def jaw_rig(builder, x, y, z_start, axis, half, offset, ke=2e4):
    """Gantry on z with two fingers on a closing axis. Returns link ids."""
    gantry = builder.add_link(xform=wp.transform(wp.vec3(x, y, z_start), wp.quat_identity()), mass=0.1, label='gantry')
    a = builder.add_link(xform=wp.transform(wp.vec3(x, y, z_start) + wp.vec3(*axis) * offset, wp.quat_identity()), label='a')
    b = builder.add_link(xform=wp.transform(wp.vec3(x, y, z_start) - wp.vec3(*axis) * offset, wp.quat_identity()), label='b')
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=1.0)
    builder.add_shape_box(a, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    builder.add_shape_box(b, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    j0 = builder.add_joint_prismatic(parent=-1, child=gantry, axis=wp.vec3(0, 0, 1), parent_xform=wp.transform(wp.vec3(x, y, 0), wp.quat_identity()),
                                     target_ke=GANTRY_KE, target_kd=3e3, target_pos=z_start, label='gz')
    j1 = builder.add_joint_prismatic(parent=gantry, child=a, axis=wp.vec3(*axis), target_ke=ke, target_kd=100.0, target_pos=offset, label='ja')
    j2 = builder.add_joint_prismatic(parent=gantry, child=b, axis=-wp.vec3(*axis), target_ke=ke, target_kd=100.0, target_pos=offset, label='jb')
    builder.add_articulation([j0, j1, j2], label='gripper')
    builder.joint_q[-3:] = [z_start, offset, offset]
    return gantry, a, b


def lift_metrics(sim, before, gantry, target_z, weight, out_path, links, half):
    q = sim.q()
    rise = q[:, 2] - before[:, 2]
    reaction = GANTRY_KE * (target_z - sim.body_z(gantry))
    save_state(sim, out_path, [(l, *half) for l in links])
    return dict(mean_rise_mm=float(rise.mean() * 1e3), max_rise_mm=float(rise.max() * 1e3), share_up_30mm=float((rise > 0.03).mean()),
                share_up_80mm=float((rise > 0.08).mean()), min_height_mm=float(q[:, 2].min() * 1e3), max_height_mm=float(q[:, 2].max() * 1e3),
                reaction_N=float(reaction), weight_N=weight, max_v=float(np.abs(sim.s0.particle_qd.numpy()).max()))


def test_top_pinch(variant, out, lift=0.10, press_in=0.012):
    """Open jaws descend into the top surface, close, and rise."""
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = add_content(builder, variant, Z0, out)
    half, offset = (0.01, 0.02, 0.03), 0.05
    gantry, a, b = jaw_rig(builder, 0.0, 0.0, 0.36, (1, 0, 0), half, offset)
    sim = Sim(builder, info)
    place_content(sim, variant, Z0, out)
    sim.frame(60)
    q = sim.q()
    mask = (np.abs(q[:, 0]) < 0.07) & (np.abs(q[:, 1]) < 0.04)
    surface = float(q[mask, 2].max())
    z = 0.36
    goal = surface + half[2] - press_in
    while z > goal:
        z = max(z - 0.002, goal)
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
    sim.frame(30)
    closed = TOP_GAP / 2 + half[0]
    off = offset
    while off > closed:
        off = max(off - 0.0005, closed)
        sim.targets[1], sim.targets[2] = off, off
        sim.apply_targets()
        sim.frame(1)
    sim.frame(50)
    before = sim.q().copy()
    actual_gap = 2 * (float(sim.s0.body_q.numpy()[a][0])) - 2 * half[0]
    for _ in range(int(lift / LIFT_STEP)):
        z += LIFT_STEP
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
    sim.frame(100)
    res = lift_metrics(sim, before, gantry, z, MASS * 9.81, out / f'{variant}_top_pinch{TAG}.npz', (a, b), half)
    res.update(surface=surface, jaw_gap_before_lift_mm=actual_gap * 1e3)
    return res


def test_edge_pinch(variant, out, lift=0.10, overhang=0.05):
    """Item sits on a table with one edge overhanging; jaws close vertically on it and rise."""
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    info = add_content(builder, variant, TABLE_Z + Z0, out)
    half, offset = (0.02, 0.02, 0.005), 0.06
    if variant == 'cloth_crumpled':
        crumple = np.load(out / 'crumple_state.npy')
        x_max = float(np.ptp(crumple[:, 0]) / 2)
    elif variant == 'tet_blob':
        x_max = 0.13
    else:
        x_max = 0.15
    edge = x_max - overhang
    table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(edge - 0.5, 0, TABLE_Z - 0.1), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.1, cfg=table_cfg)
    grip_x, grip_y = x_max - 0.025, 0.07 if variant in ('cloth_folded', 'tet_flat') else 0.0
    gantry, a, b = jaw_rig(builder, grip_x, grip_y, TABLE_Z + 0.03, (0, 0, 1), half, offset)
    sim = Sim(builder, info)
    place_content(sim, variant, TABLE_Z + Z0, out)
    sim.frame(60)
    q = sim.q()
    mask = (np.abs(q[:, 0] - grip_x) < 0.03) & (np.abs(q[:, 1] - grip_y) < 0.03)
    if not mask.any():
        return dict(FAILED='no material under the jaws')
    thickness = float(q[mask, 2].max() - TABLE_Z)
    # Jaws centered on the local material, then closed to half its thickness.
    z = TABLE_Z + thickness / 2
    sim.targets[0] = z
    sim.apply_targets()
    sim.frame(40)
    closed = half[2] + max(thickness * GRIP_SQUEEZE, 0.002)
    off = offset
    while off > closed:
        off = max(off - 0.0005, closed)
        sim.targets[1], sim.targets[2] = off, off
        sim.apply_targets()
        sim.frame(1)
    sim.frame(50)
    before = sim.q().copy()
    for _ in range(int(lift / LIFT_STEP)):
        z += LIFT_STEP
        sim.targets[0] = z
        sim.apply_targets()
        sim.frame(1)
    sim.frame(100)
    res = lift_metrics(sim, before, gantry, z, MASS * 9.81, out / f'{variant}_edge_pinch{TAG}.npz', (a, b), half)
    q = sim.q()
    res.update(thickness=thickness, far_end_rise_mm=float((q[:, 2] - before[:, 2])[before[:, 0] < before[:, 0].min() + 0.05].mean() * 1e3))
    return res


def main():
    global SUBSTEPS, FRICTION_EPS, CONTACT_KE, GRIP_SQUEEZE, TOP_GAP, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument('variant')
    ap.add_argument('test')
    ap.add_argument('--out', default=str(HERE / 'results_newton'))
    ap.add_argument('--substeps', type=int, default=SUBSTEPS)
    ap.add_argument('--friction-eps', type=float, default=FRICTION_EPS)
    ap.add_argument('--contact-ke', type=float, default=CONTACT_KE)
    ap.add_argument('--grip-squeeze', type=float, default=GRIP_SQUEEZE)
    ap.add_argument('--tag', default='')
    ap.add_argument('--top-gap', type=float, default=TOP_GAP)
    args = ap.parse_args()
    SUBSTEPS, FRICTION_EPS, CONTACT_KE, GRIP_SQUEEZE = args.substeps, args.friction_eps, args.contact_ke, args.grip_squeeze
    TOP_GAP, TAG = args.top_gap, args.tag
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    wp.config.verbose = False
    t0 = time.perf_counter()
    try:
        if args.test == 'make_crumple':
            res = make_crumple(out)
        else:
            res = globals()[f'test_{args.test}'](args.variant, out)
    except FloatingPointError as err:
        res = dict(FAILED=str(err))
    res['wall_s'] = round(time.perf_counter() - t0, 1)
    (out / f'{args.variant}_{args.test}{args.tag}.json').write_text(json.dumps(res, indent=2))
    print(args.variant, args.test, args.tag, json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
