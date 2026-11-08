"""Part B of the membrane benchmark (README.md): grip a sponge through a closed bubble-film bag.

The sponge, pads, phases and pad motion are those of ../gripper_grasp/grasp.py, which stays
unchanged and is the no-bag control. Here a closed box membrane (bubble parameters, as in
../contact_transfer/benchmark.py --membrane bubble) surrounds the sponge and rests on the ground.
The pads close so the sponge's nominal squeeze equals the control's; the film stack (pad to film
radius plus film to sponge contact distance) is added to the closed pad gap.

Every substep integrates three impulses: rigid contact per shape, rigid contact on sponge nodes
(should vanish: the pads must not reach the sponge through the film) and particle contact on
sponge nodes (the bag's push). Momentum of the sponge and of the whole system closes each chain.
Results go to ../gripper_grasp/results/<tag>_<material>_off<offset>.json.

Example (Newton 1.7 interpreter):
  python -B bagged_grasp.py --offset-mm 5 --tag bag_b1_r1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp
import newton
from newton._src.solvers.vbd.particle_vbd_kernels import accumulate_self_contact_force_and_hessian

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'gripper_grasp'))
sys.path.insert(0, str(HERE.parent / 'contact_transfer'))
sys.path.insert(0, str(ROOT / 'contents' / 'newton'))
import unified_soft as us
from benchmark import MEMBRANES, _contact_probe, triangle_areas
from grasp import (CELLS, CLOSE, HOLD, LIFT, LIFT_M, MASS, MATERIALS, PAD, SETTLE, SIZE, SQUEEZE_HOLD, Grasp,
                   _integrate_contact_force, _place_pads, rounded_box)
from metrics import tet_response

SHEET = MEMBRANES['bubble']
MU_PAD, MU_BAG, MU_GROUND = .8, .5, .5  # pad-bag, bag-sponge, bag-ground


@wp.kernel
def _integrate_range(force: wp.array[wp.vec3], start: int, dt: float, slot: int, impulse: wp.array[wp.vec3d]):
    i = wp.tid()
    wp.atomic_add(impulse, slot, wp.vec3d(force[start + i]) * wp.float64(dt))


def box_surface(half, spacing):
    """Closed, shared-vertex triangle grid on an axis-aligned box centred at the origin."""
    half = np.asarray(half, float)
    n = [max(1, round(2 * h / spacing)) for h in half]
    verts, index, faces = [], {}, []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (1., -1.):
            grid = np.empty((n[u] + 1, n[v] + 1), int)
            for i in range(n[u] + 1):
                for j in range(n[v] + 1):
                    c = np.empty(3)
                    c[axis], c[u], c[v] = sign * half[axis], half[u] * (2 * i / n[u] - 1), half[v] * (2 * j / n[v] - 1)
                    key = tuple(np.round(c, 9))
                    if key not in index:
                        index[key] = len(verts)
                        verts.append(c)
                    grid[i, j] = index[key]
            outward = sign * (1 if (u, v) in ((1, 2), (2, 0), (0, 1)) else -1)
            for i in range(n[u]):
                for j in range(n[v]):
                    a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
                    faces.extend([(a, b, c), (a, c, d)] if outward > 0 else [(a, c, b), (a, d, c)])
    return np.array(verts), np.array(faces, np.int32)


def point_triangle_distance(p, tri):
    """Distance from each point p (n, 3) to the nearest of the triangles tri (m, 3, 3)."""
    a, b, c = (tri[None, :, k] for k in range(3))
    p = p[:, None]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    plane = np.einsum('nmk,nmk->nm', p - a, normal)
    foot = p - plane[..., None] * normal
    inside = np.ones(plane.shape, bool)
    for x, y in ((a, b), (b, c), (c, a)):
        inside &= np.einsum('nmk,nmk->nm', np.cross(y - x, foot - x), normal) >= 0
    best = np.where(inside, np.abs(plane), np.inf)
    for x, y in ((a, b), (b, c), (c, a)):
        e = y - x
        s = np.clip(np.einsum('nmk,nmk->nm', p - x, e) / np.einsum('nmk,nmk->nm', e, e), 0., 1.)
        best = np.minimum(best, np.linalg.norm(p - x - s[..., None] * e, axis=-1))
    return best.min(axis=1)


def inside_tets(p, corners):
    """Boolean per point: strictly inside at least one tetrahedron (corners (m, 4, 3))."""
    t = np.stack([corners[:, k] - corners[:, 0] for k in (1, 2, 3)], axis=2)
    inv = np.linalg.inv(t)
    hit = np.zeros(len(p), bool)
    for start in range(0, len(p), 256):
        local = np.einsum('mij,nmj->nmi', inv, p[start:start + 256, None] - corners[None, :, 0])
        hit[start:start + 256] = ((local > 1e-9).all(axis=2) & (local.sum(axis=2) < 1 - 1e-9)).any(axis=1)
    return hit


class BaggedGrasp:
    def __init__(self, material, offset, squeeze, dt, iterations, ke, kd, bag_ke, bag_kd, friction_eps, pad_radius):
        builder = newton.ModelBuilder(gravity=wp.vec3(0., 0., -9.81))
        # VBD averages shape and particle materials; shapes carry 2*target - particle so rigid contact gets ke/kd/mu.
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=2 * ke - bag_ke, kd=2 * kd - bag_kd,
                                                                     mu=MU_GROUND ** 2 / MU_BAG))
        gap = SHEET['distance'] + .0002
        bag_half = np.array(SIZE) / 2 + gap
        bottom = SHEET['radius'] + .0002
        base_z = bottom + gap  # sponge bottom
        shape = us.SoftShape('slab', SIZE, CELLS, MASS, layers=('core',) * CELLS[2], materials={'core': MATERIALS[material]})
        self.n_nodes = us.add_item(builder, shape, base_z=base_z)['end']
        # The sponge should only meet the bag; a film-thin radius makes any pad or ground force on it a bypass.
        for i in range(self.n_nodes):
            builder.particle_radius[i] = SHEET['radius']
        self.sponge_faces = np.asarray(builder.tri_indices, dtype=int).copy()
        verts, faces = box_surface(bag_half, SHEET['spacing'])
        verts[:, 2] += base_z + SIZE[2] / 2
        start = builder.particle_count
        builder.add_cloth_mesh(pos=wp.vec3(0., 0., 0.), rot=wp.quat_identity(), scale=1., vel=wp.vec3(),
            vertices=verts.tolist(), indices=faces.ravel().tolist(), density=SHEET['density'],
            tri_ke=SHEET['tri_ke'], tri_ka=SHEET['tri_ka'], tri_kd=SHEET['tri_kd'], edge_ke=SHEET['edge_ke'],
            edge_kd=1e-5, particle_radius=SHEET['radius'])
        self.bag = slice(start, builder.particle_count)
        self.bag_faces = faces + start
        self.pad_z = base_z + SIZE[2] / 2
        stack = SHEET['radius'] + SHEET['distance']
        self.open_y = bag_half[1] + PAD[1] + .015
        self.closed_y = SIZE[1] / 2 * (1 - squeeze) + stack + PAD[1]
        self.x = offset
        cfg = newton.ModelBuilder.ShapeConfig(ke=2 * ke - bag_ke, kd=2 * kd - bag_kd, mu=MU_PAD ** 2 / MU_BAG, density=0.)
        if pad_radius > 0:
            pad_mesh = rounded_box(PAD, pad_radius)
            pad_mesh.build_sdf(target_voxel_size=.0015)
        else:
            cfg.configure_sdf(force_sdf=True, target_voxel_size=.0015)
        self.pads, self.pad_shapes = [], []
        for side in (1, -1):
            body = builder.add_body(xform=wp.transform(wp.vec3(offset, side * self.open_y, self.pad_z), wp.quat_identity()),
                                    is_kinematic=True, label=f'pad{side:+d}')
            self.pads.append(body)
            self.pad_shapes.append(builder.add_shape_mesh(body, mesh=pad_mesh, cfg=cfg) if pad_radius > 0 else
                                   builder.add_shape_box(body, hx=PAD[0], hy=PAD[1], hz=PAD[2], cfg=cfg))
        builder.color(include_bending=True)
        self.model = m = builder.finalize()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = bag_ke, bag_kd, MU_BAG
        self.solver = newton.solvers.SolverVBD(m, iterations=iterations, rigid_compliant_alm=True, friction_epsilon=friction_eps,
            particle_enable_self_contact=True, particle_self_contact_margin=SHEET['distance'],
            particle_self_contact_gap=SHEET['distance'] / 2,
            rigid_body_particle_contact_buffer_size=8192, rigid_body_contact_buffer_size=256)
        self.pipeline = newton.CollisionPipeline(m, broad_phase='nxn', soft_contact_gap=.004,
                                                 enable_rigid_soft_full_surface_contact=True)
        self.contacts = self.pipeline.contacts()
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()
        self.substeps = 2 * max(1, int(np.ceil(.01 / dt / 2)))
        self.dt = .01 / self.substeps
        self.ids = wp.array(self.pads, dtype=int)
        start = [(offset, side * self.open_y, self.pad_z) for side in (1, -1)]
        self.p_from = wp.array(start, dtype=wp.vec3)
        self.p_to = wp.array(start, dtype=wp.vec3)
        self.q0 = m.particle_q.numpy().astype(float)
        self.tets = m.tet_indices.numpy()
        self.inv = m.tet_poses.numpy().astype(float)
        self.materials = m.tet_materials.numpy().astype(float)
        mass = m.particle_mass.numpy().astype(float)
        self.mass, self.bag_mass = mass[:self.n_nodes], mass[self.bag]
        self.probe_f = wp.zeros(m.particle_count, dtype=wp.vec3)
        self.self_f = wp.zeros_like(self.probe_f)
        self.self_h = wp.zeros(m.particle_count, dtype=wp.mat33)
        self.shape_f = wp.zeros(m.shape_count, dtype=wp.vec3)
        self.kinds = wp.zeros(3, dtype=int)
        self.body_prev_measure = wp.clone(m.body_q)
        self.contact_impulse = wp.zeros(m.shape_count, dtype=wp.vec3d)
        # 0: rigid contact on sponge nodes, 1: particle contact on sponge nodes, 2: particle contact on bag nodes
        self.node_impulse = wp.zeros(3, dtype=wp.vec3d)
        self.graph = None

    pad_targets = Grasp.pad_targets

    def contact_forces(self, q, prev):
        """Rigid (probe_f, shape_f) and particle (self_f) contact forces at positions q."""
        m, v, c = self.model, self.solver, self.contacts
        self.probe_f.zero_(); self.shape_f.zero_(); self.kinds.zero_(); self.self_f.zero_(); self.self_h.zero_()
        wp.launch(_contact_probe, dim=c.soft_contact_max, inputs=[c.soft_contact_count, c.soft_contact_indices,
            c.soft_contact_barycentric, q, v.particle_q_prev, m.particle_radius,
            v.body_particle_contact_penalty_k, v.body_particle_contact_material_kd, v.body_particle_contact_material_mu,
            v.friction_epsilon, m.shape_body, self.s0.body_q, self.body_prev_measure, self.s0.body_qd, m.body_com,
            c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal,
            m.shape_margin, self.dt, False, self.probe_f, self.shape_f, self.kinds])
        for color in range(len(m.particle_color_groups)):
            wp.launch(accumulate_self_contact_force_and_hessian, dim=v.particle_self_contact_evaluation_kernel_launch_size,
                inputs=[self.dt, color, v.particle_q_prev, q, m.particle_colors, m.tri_indices, m.edge_indices,
                    v.trimesh_collision_info, v.particle_self_contact_margin, m.soft_contact_ke, m.soft_contact_kd,
                    m.soft_contact_mu, v.friction_epsilon, v._self_contact_edge_edge_parallel_epsilon],
                outputs=[self.self_f, self.self_h])

    def substep(self, k):
        wp.launch(_place_pads, dim=2, inputs=[self.s0.body_q, self.s0.body_qd, self.ids, self.p_from, self.p_to,
                                              (k + 1) / self.substeps, self.dt * self.substeps])
        self.s0.clear_forces()
        self.pipeline.collide(self.s0, self.contacts)
        wp.copy(self.body_prev_measure, self.solver.body_q_prev)
        self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
        self.contact_forces(self.s1.particle_q, self.solver.particle_q_prev)
        wp.launch(_integrate_contact_force, dim=self.model.shape_count, inputs=[self.shape_f, self.dt, self.contact_impulse])
        n_bag = self.bag.stop - self.bag.start
        wp.launch(_integrate_range, dim=self.n_nodes, inputs=[self.probe_f, 0, self.dt, 0, self.node_impulse])
        wp.launch(_integrate_range, dim=self.n_nodes, inputs=[self.self_f, 0, self.dt, 1, self.node_impulse])
        wp.launch(_integrate_range, dim=n_bag, inputs=[self.self_f, self.bag.start, self.dt, 2, self.node_impulse])
        self.s0, self.s1 = self.s1, self.s0
        wp.copy(self.s0.body_q, self.s1.body_q)

    def advance(self, frame):
        self.p_from.assign(self.p_to.numpy())
        self.p_to.assign(np.array(self.pad_targets((frame + 1) * .01), np.float32))
        if self.graph is None:
            if frame == 0:
                for k in range(self.substeps):
                    self.substep(k)
                return
            with wp.ScopedCapture() as cap:
                for k in range(self.substeps):
                    self.substep(k)
            self.graph = cap.graph
        wp.capture_launch(self.graph)

    def measure(self, t):
        s, c = self.s0, self.contacts
        q_all = s.particle_q.numpy().astype(float)
        if not np.isfinite(q_all).all() or np.abs(q_all).max() > 2:
            raise FloatingPointError(f'diverged at {t:.2f}s')
        count = int(c.soft_contact_count.numpy()[0])
        if count > c.soft_contact_max:
            raise RuntimeError(f'contact buffer overflow {count} > {c.soft_contact_max}')
        q, bag = q_all[:self.n_nodes], q_all[self.bag]
        qd = s.particle_qd.numpy().astype(float)
        speed = np.linalg.norm(qd, axis=1) * 1e3
        jac = tet_response(q, self.tets, self.inv, self.materials)[2]
        pads = s.body_q.numpy()[self.pads].astype(float)
        # Geometric checks: film nodes inside the deformed sponge, sponge nodes inside a pad box.
        inside = inside_tets(bag, q[self.tets])
        depth = point_triangle_distance(bag[inside], q[self.sponge_faces]) if inside.any() else np.zeros(0)
        in_pad = sum(int((np.abs(q - p[:3]) < PAD).all(axis=1).sum()) for p in pads)
        bag_in_pad = sum(int((np.abs(bag - p[:3]) < PAD).all(axis=1).sum()) for p in pads)
        stretch = triangle_areas(q_all, self.bag_faces) / triangle_areas(self.q0, self.bag_faces) - 1
        return dict(t=round(t, 3), pad_z_mm=float(pads[0, 2] * 1e3),
                    pad_gap_mm=float((pads[0, 1] - pads[1, 1] - 2 * PAD[1]) * 1e3),
                    com_z_mm=float(self.mass @ q[:, 2] / self.mass.sum() * 1e3), min_z_mm=float(q[:, 2].min() * 1e3),
                    bag_min_z_mm=float(bag[:, 2].min() * 1e3),
                    max_speed_mm_s=float(speed[:self.bag.stop].max()), sponge_max_speed_mm_s=float(speed[:self.n_nodes].max()),
                    median_speed_mm_s=float(np.median(speed[:self.n_nodes])),
                    nodes_over_1mm_s=int((speed[:self.bag.stop] > 1).sum()),
                    min_J=float(jac.min()), inverted=int((jac <= 0).sum()),
                    bag_nodes_in_sponge=int(inside.sum()), bag_into_sponge_mm=float(depth.max() * 1e3) if len(depth) else 0.,
                    sponge_nodes_in_pads=in_pad, bag_nodes_in_pads=bag_in_pad,
                    bag_max_area_stretch=float(stretch.max()), contacts=count,
                    contact_impulse_Ns=self.contact_impulse.numpy().tolist(),
                    node_impulse_Ns=self.node_impulse.numpy().tolist(),
                    momentum_kg_m_s=(self.mass @ qd[:self.n_nodes]).tolist(),
                    bag_momentum_kg_m_s=(self.bag_mass @ qd[self.bag]).tolist(),
                    positions=q_all[:self.bag.stop].astype(np.float32))


def run(args):
    t0 = time.perf_counter()
    g = BaggedGrasp(args.material, args.offset_mm * 1e-3, args.squeeze, args.dt, args.iterations, args.ke, args.kd,
                    args.bag_ke, args.bag_kd, args.friction_eps, args.pad_radius_mm * 1e-3)
    total = SETTLE + CLOSE + SQUEEZE_HOLD + LIFT + HOLD
    rows, failure, previous = [], None, None
    for frame in range(round(total / .01)):
        g.advance(frame)
        if (frame + 1) % 5 == 0:
            try:
                row = g.measure((frame + 1) * .01)
            except (FloatingPointError, RuntimeError) as err:
                failure = str(err)
                break
            q = row.pop('positions')
            if previous is not None:
                row['max_speed_50ms_mm_s'] = float(np.linalg.norm(q - previous, axis=1).max() / .05 * 1e3)
            previous = q
            rows.append(row)
            if row['inverted']:
                failure = f"{row['inverted']} inverted tetrahedra at {row['t']} s"
                break
    wp.synchronize()
    weight, bag_weight = g.mass.sum() * 9.81, g.bag_mass.sum() * 9.81
    lift_start = SETTLE + CLOSE + SQUEEZE_HOLD
    before = [r for r in rows if SETTLE + CLOSE < r['t'] <= lift_start]
    hold = [r for r in rows if r['t'] > total - .5]
    summary = {}
    if before and len(hold) >= 2:
        T = hold[-1]['t'] - hold[0]['t']
        diff = lambda key: (np.asarray(hold[-1][key]) - np.asarray(hold[0][key])) / T
        shape_force, node_force = diff('contact_impulse_Ns'), diff('node_impulse_Ns')
        pads = shape_force[g.pad_shapes]
        grip = (pads[1, 1] - pads[0, 1]) / 2  # force on the particles: pad +1 pushes toward -y
        sponge_rate, bag_rate = diff('momentum_kg_m_s'), diff('bag_momentum_kg_m_s')
        gravity = np.array([0., 0., -1.])
        rise = np.mean([r['com_z_mm'] for r in hold]) - np.mean([r['com_z_mm'] for r in before])
        summary = dict(
            com_rise_mm=float(rise), carried_fraction=float(rise / (LIFT_M * 1e3)),
            pads_on_system_over_weight=float(pads[:, 2].sum() / (weight + bag_weight)),
            bag_on_sponge_over_weight=float(node_force[1, 2] / weight),
            rigid_on_sponge_N=node_force[0].tolist(),
            ground_N=float(shape_force[:, 2].sum() - pads[:, 2].sum()),
            system_momentum_balance_pct=float(np.linalg.norm(sponge_rate + bag_rate - shape_force.sum(axis=0)
                                                             - (weight + bag_weight) * gravity) / (weight + bag_weight) * 100),
            sponge_momentum_balance_pct=float(np.linalg.norm(sponge_rate - node_force[0] - node_force[1] - weight * gravity)
                                              / weight * 100),
            particle_contact_pair_imbalance_N=float(np.linalg.norm(node_force[1] + node_force[2])),
            squeeze_N=float(grip), squeeze_imbalance_pct=float(abs(pads[0, 1] + pads[1, 1]) / max(abs(grip), 1e-9) * 100),
            slip_mm_s=float((hold[0]['com_z_mm'] - hold[-1]['com_z_mm']) / T),
            max_speed_mm_s=float(np.mean([r['max_speed_mm_s'] for r in hold])),
            sponge_max_speed_mm_s=float(np.mean([r['sponge_max_speed_mm_s'] for r in hold])),
            median_speed_mm_s=float(np.mean([r['median_speed_mm_s'] for r in hold])),
            nodes_over_1mm_s=float(np.mean([r['nodes_over_1mm_s'] for r in hold])),
            max_speed_50ms_mm_s=float(np.mean([r['max_speed_50ms_mm_s'] for r in hold])),
            bag_max_area_stretch=max(r['bag_max_area_stretch'] for r in rows),
            min_J=min(r['min_J'] for r in rows), impulse_window_s=T,
            bag_into_sponge_mm=max(r['bag_into_sponge_mm'] for r in rows),
            bag_nodes_in_sponge=max(r['bag_nodes_in_sponge'] for r in rows),
            sponge_nodes_in_pads=max(r['sponge_nodes_in_pads'] for r in rows),
            bag_nodes_in_pads=max(r['bag_nodes_in_pads'] for r in rows),
            bag_mass_kg=float(g.bag_mass.sum()))
    checks = dict(finished=failure is None and bool(hold),
                  carried=summary.get('carried_fraction', 0) >= .9,
                  pads_carry_bag_and_sponge=abs(summary.get('pads_on_system_over_weight', 0) - 1) <= .05,
                  bag_carries_sponge=abs(summary.get('bag_on_sponge_over_weight', 0) - 1) <= .05,
                  momentum_balance=max(summary.get('system_momentum_balance_pct', 1e9),
                                       summary.get('sponge_momentum_balance_pct', 1e9)) <= 5,
                  squeeze_balanced=summary.get('squeeze_imbalance_pct', 1e9) <= 5,
                  no_slip=summary.get('slip_mm_s', 1e9) <= .1,
                  no_inversion=failure is None and summary.get('min_J', 0) > 0,
                  no_penetration=summary.get('bag_into_sponge_mm', 1e9) <= .05 and summary.get('sponge_nodes_in_pads', 1) == 0,
                  settled=summary.get('max_speed_mm_s', 1e9) <= 1)
    result = dict(args=vars(args), actual_dt_s=g.dt, nodes=g.n_nodes, bag_nodes=g.bag.stop - g.bag.start,
                  tets=int(g.model.tet_count), weight_N=weight, bag_weight_N=bag_weight, membrane=SHEET,
                  newton=newton.__version__, warp=wp.__version__, executable=sys.executable,
                  wall_s=round(time.perf_counter() - t0, 2), failure=failure, summary=summary, checks=checks,
                  passed=all(checks.values()), trace=rows)
    sources = [Path(__file__), HERE.parent / 'gripper_grasp' / 'grasp.py', HERE.parent / 'contact_transfer' / 'benchmark.py']
    result['project_sources_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out = HERE.parent / 'gripper_grasp' / 'results'
    out.mkdir(exist_ok=True)
    name = f'{args.tag}_{args.material}_off{args.offset_mm:g}'
    (out / f'{name}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(name=name, failure=failure, summary=summary, checks=checks, wall_s=result['wall_s'])), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--material', choices=sorted(MATERIALS), default='firm')
    ap.add_argument('--offset-mm', type=float, default=0., help='pad centre x offset; 5 puts it between sponge nodes')
    ap.add_argument('--squeeze', type=float, default=.3, help='nominal sponge squeeze, as in grasp.py')
    ap.add_argument('--dt', type=float, default=.00025)
    ap.add_argument('--iterations', type=int, default=10)
    ap.add_argument('--ke', type=float, default=3000., help='pad-bag and ground-bag contact ke')
    ap.add_argument('--kd', type=float, default=3.)
    ap.add_argument('--bag-ke', type=float, default=300., help='bag-sponge particle contact ke')
    ap.add_argument('--bag-kd', type=float, default=.3)
    ap.add_argument('--pad-radius-mm', type=float, default=4.)
    ap.add_argument('--friction-eps', type=float, default=1e-2, help='VBD friction regularisation velocity (m/s)')
    ap.add_argument('--tag', default='bag_b1')
    args = ap.parse_args()
    if not 0 < args.squeeze < 1 or args.dt <= 0 or args.iterations < 1:
        ap.error('0 < squeeze < 1, dt > 0, iterations >= 1')
    wp.config.quiet = True
    run(args)


if __name__ == '__main__':
    main()
