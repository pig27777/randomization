"""Position-controlled parallel gripper on a homogeneous sponge block (Newton VBD).

The block rests on the ground under gravity. Two kinematic pads close on its 80 mm
sides, squeeze, lift and hold. Pads take no part in the solve, which isolates the
contact path from rigid-soft coupling (see ../contact_transfer/REPORT_position_control.md).

Example (existing Newton 1.6.1 interpreter):
  python -B grasp.py --material soft --offset-mm 0 --tag base
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'contact_transfer'))
sys.path.insert(0, str(ROOT / 'contents' / 'newton'))
import unified_soft as us
from benchmark import _contact_probe
from metrics import tet_response

SIZE = (.12, .08, .05)  # m
CELLS = (12, 8, 5)  # 10 mm cells
MASS = .3
MATERIALS = {'soft': (1500., 1000., 1.), 'firm': (6000., 2000., 1.)}  # Lame mu, lambda, damping
PAD = (.02, .005, .02)  # half extents: 40 mm wide, 10 mm thick, 40 mm tall
MU = .8
# phase durations (s) and motion
SETTLE, CLOSE, SQUEEZE_HOLD, LIFT, HOLD = .3, .5, .3, 1., 1.
LIFT_M = .05


# Contact-zone refinement (mm): (start, end, spacing) per axis. 2.5 mm cells where the pads touch,
# around the pad side edges (x = +-20) and over the lower half where inversions started; ~10 mm elsewhere.
REFINED_AXES = ([(-60, -30, 10), (-30, 30, 2.5), (30, 60, 10)],
                [(-40, -27.5, 2.5), (-27.5, 27.5, 55 / 6), (27.5, 40, 2.5)],
                [(0, 25, 2.5), (25, 50, 6.25)])


def graded_axis(pieces):
    points = [np.linspace(a, b, round((b - a) / h) + 1)[:-1] for a, b, h in pieces]
    return np.r_[np.concatenate(points), pieces[-1][1]] * 1e-3


def add_refined_block(builder, material, base_z, radius):
    """Tensor-product graded hex lattice split into tets (same split as unified_soft)."""
    axes = [graded_axis(p) for p in REFINED_AXES]
    n = tuple(len(a) for a in axes)
    pos = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    tets, _ = us._tets(n)
    p = pos[tets]
    vol = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    mu, lam, damp = MATERIALS[material]
    start = builder.particle_count
    builder.add_soft_mesh(pos=wp.vec3(0., 0., base_z), rot=wp.quat_identity(), scale=1., vel=wp.vec3(0., 0., 0.),
                          vertices=pos.tolist(), indices=tets.ravel().tolist(), density=MASS / float(np.abs(vol).sum() / 6),
                          k_mu=mu, k_lambda=lam, k_damp=damp, particle_radius=radius)
    return builder.particle_count - start


def rounded_box(half, radius, n=12):
    """Watertight triangle mesh of a box with every edge and corner rounded to `radius`.

    A grid on the outer box surface is pulled onto the Minkowski sum of the inner box and a sphere.
    """
    half = np.asarray(half, float)
    inner = half - radius
    if (inner <= 0).any():
        raise ValueError('pad radius must be smaller than every pad half extent')
    verts, index, faces = [], {}, []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (1., -1.):
            grid = np.empty((n + 1, n + 1), int)
            for i in range(n + 1):
                for j in range(n + 1):
                    c = np.empty(3)
                    c[axis], c[u], c[v] = sign * half[axis], half[u] * (2 * i / n - 1), half[v] * (2 * j / n - 1)
                    core = np.clip(c, -inner, inner)
                    point = core + radius * (c - core) / np.linalg.norm(c - core)
                    key = tuple(np.round(point, 9))
                    if key not in index:
                        index[key] = len(verts)
                        verts.append(point)
                    grid[i, j] = index[key]
            for i in range(n):
                for j in range(n):
                    a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
                    # u x v points along +axis for even permutations, so flip the winding where needed
                    outward = sign * (1 if (u, v) in ((1, 2), (2, 0), (0, 1)) else -1)
                    tris = [(a, b, c), (a, c, d)] if outward > 0 else [(a, c, b), (a, d, c)]
                    faces.extend(t for t in tris if len(set(t)) == 3)
    return newton.Mesh(np.array(verts, np.float32), np.array(faces, np.int32).ravel())


@wp.kernel
def _place_pads(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], ids: wp.array[int],
                p_from: wp.array[wp.vec3], p_to: wp.array[wp.vec3], fraction: float, frame_dt: float):
    i = wp.tid()
    p = p_from[i] + fraction * (p_to[i] - p_from[i])
    body_q[ids[i]] = wp.transform(p, wp.quat_identity())
    body_qd[ids[i]] = wp.spatial_vector((p_to[i] - p_from[i]) / frame_dt, wp.vec3(0., 0., 0.))


@wp.kernel
def _integrate_contact_force(force: wp.array[wp.vec3], dt: float, impulse: wp.array[wp.vec3d]):
    i = wp.tid()
    impulse[i] = impulse[i] + wp.vec3d(force[i]) * wp.float64(dt)


class Grasp:
    def __init__(self, material, offset, squeeze, dt, iterations, ke, kd, friction_eps, self_contact=False, pad_radius=0.,
                 refined=False, contact=.004):
        builder = newton.ModelBuilder(gravity=wp.vec3(0., 0., -9.81))
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=MU))
        if refined:
            self.n_nodes = add_refined_block(builder, material, .002, contact)
        else:
            shape = us.SoftShape('slab', SIZE, CELLS, MASS, layers=('core',) * CELLS[2], materials={'core': MATERIALS[material]})
            self.n_nodes = us.add_item(builder, shape, base_z=.002)['end']
            # unified_soft uses a 4 mm particle radius; override it so the contact layer can be thinner than a cell.
            for i in range(self.n_nodes):
                builder.particle_radius[i] = contact
        self.open_y = SIZE[1] / 2 + PAD[1] + .015
        self.closed_y = SIZE[1] / 2 * (1 - squeeze) + PAD[1]
        self.pad_z = .002 + SIZE[2] / 2
        self.x = offset
        cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=MU, density=0.)
        if pad_radius > 0:
            # Mesh shapes carry their own SDF, built at the same voxel size as the box pads.
            pad_mesh = rounded_box(PAD, pad_radius)
            pad_mesh.build_sdf(target_voxel_size=.0015)
        else:
            cfg.configure_sdf(force_sdf=True, target_voxel_size=.0015)
        self.pads, self.pad_shapes = [], []
        for side in (1, -1):
            body = builder.add_body(xform=wp.transform(wp.vec3(offset, side * self.open_y, self.pad_z), wp.quat_identity()),
                                    is_kinematic=True, label=f'pad{side:+d}')
            self.pads.append(body)
            if pad_radius > 0:
                self.pad_shapes.append(builder.add_shape_mesh(body, mesh=pad_mesh, cfg=cfg))
            else:
                self.pad_shapes.append(builder.add_shape_box(body, hx=PAD[0], hy=PAD[1], hz=PAD[2], cfg=cfg))
        builder.color()
        self.model = m = builder.finalize()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = ke, kd, MU
        self.solver = newton.solvers.SolverVBD(m, iterations=iterations, rigid_compliant_alm=True, friction_epsilon=friction_eps,
            particle_enable_self_contact=self_contact, particle_self_contact_margin=.008, particle_self_contact_gap=.004,
            rigid_body_particle_contact_buffer_size=8192, rigid_body_contact_buffer_size=256)
        # Contact acts within particle radius + gap of the surface; keep that below the cell size or interior
        # layers get pushed too.
        self.pipeline = newton.CollisionPipeline(m, broad_phase='nxn', soft_contact_gap=contact,
                                                 enable_rigid_soft_full_surface_contact=True)
        self.contacts = self.pipeline.contacts()
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()
        self.substeps = 2 * max(1, int(np.ceil(.01 / dt / 2)))
        self.dt = .01 / self.substeps
        self.ids = wp.array(self.pads, dtype=int)
        start = [(offset, side * self.open_y, self.pad_z) for side in (1, -1)]
        self.p_from = wp.array(start, dtype=wp.vec3)
        self.p_to = wp.array(start, dtype=wp.vec3)
        self.tets = m.tet_indices.numpy()
        self.inv = m.tet_poses.numpy().astype(float)
        self.materials = m.tet_materials.numpy().astype(float)
        self.mass = m.particle_mass.numpy()[:self.n_nodes].astype(float)
        self.probe_f = wp.zeros(m.particle_count, dtype=wp.vec3)
        self.shape_f = wp.zeros(m.shape_count, dtype=wp.vec3)
        self.kinds = wp.zeros(3, dtype=int)
        self.body_prev_measure = wp.clone(m.body_q)
        self.contact_impulse = wp.zeros(m.shape_count, dtype=wp.vec3d)
        self.graph = None

    def pad_targets(self, t):
        """Pad centres at time t: close, hold, lift, hold."""
        def smooth(u):
            u = min(max(u, 0.), 1.)
            return u * u * (3 - 2 * u)
        close = smooth((t - SETTLE) / CLOSE)
        lift = smooth((t - SETTLE - CLOSE - SQUEEZE_HOLD) / LIFT)
        y = self.open_y + close * (self.closed_y - self.open_y)
        z = self.pad_z + lift * LIFT_M
        return [(self.x, y, z), (self.x, -y, z)]

    def substep(self, k):
        wp.launch(_place_pads, dim=2, inputs=[self.s0.body_q, self.s0.body_qd, self.ids, self.p_from, self.p_to,
                                              (k + 1) / self.substeps, self.dt * self.substeps])
        self.s0.clear_forces()
        self.pipeline.collide(self.s0, self.contacts)
        wp.copy(self.body_prev_measure, self.solver.body_q_prev)
        self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
        # Evaluate before history is reused by the next step. Integrate every substep,
        # rather than estimating the mean force from one sample per 50 ms.
        self.probe_f.zero_(); self.shape_f.zero_(); self.kinds.zero_()
        m, v, c = self.model, self.solver, self.contacts
        wp.launch(_contact_probe, dim=c.soft_contact_max, inputs=[c.soft_contact_count, c.soft_contact_indices,
            c.soft_contact_barycentric, self.s1.particle_q, v.particle_q_prev, m.particle_radius,
            v.body_particle_contact_penalty_k, v.body_particle_contact_material_kd, v.body_particle_contact_material_mu,
            v.friction_epsilon, m.shape_body, self.s0.body_q, self.body_prev_measure, self.s0.body_qd, m.body_com,
            c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal,
            m.shape_margin, self.dt, False, self.probe_f, self.shape_f, self.kinds])
        wp.launch(_integrate_contact_force, dim=m.shape_count,
                  inputs=[self.shape_f, self.dt, self.contact_impulse])
        self.s0, self.s1 = self.s1, self.s0
        # The solver does not write kinematic poses to the output state; keep the placed pose.
        wp.copy(self.s0.body_q, self.s1.body_q)

    def advance(self, frame):
        self.p_from.assign(self.p_to.numpy())
        self.p_to.assign(np.array(self.pad_targets((frame + 1) * .01), np.float32))
        if self.graph is None:
            # The first frame runs eagerly so kernels are compiled before capture.
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
        m, s, c, v = self.model, self.s0, self.contacts, self.solver
        q = s.particle_q.numpy().astype(float)[:self.n_nodes]
        if not np.isfinite(q).all() or np.abs(q).max() > 2:
            raise FloatingPointError(f'diverged at {t:.2f}s')
        count = int(c.soft_contact_count.numpy()[0])
        if count > c.soft_contact_max:
            raise RuntimeError(f'contact buffer overflow {count} > {c.soft_contact_max}')
        self.probe_f.zero_(); self.shape_f.zero_(); self.kinds.zero_()
        if count:
            wp.launch(_contact_probe, dim=count, inputs=[c.soft_contact_count, c.soft_contact_indices, c.soft_contact_barycentric,
                s.particle_q, v.particle_q_prev, m.particle_radius, v.body_particle_contact_penalty_k,
                v.body_particle_contact_material_kd, v.body_particle_contact_material_mu, v.friction_epsilon,
                m.shape_body, s.body_q, self.body_prev_measure, s.body_qd, m.body_com, c.soft_contact_shape,
                c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal, m.shape_margin, self.dt,
                False, self.probe_f, self.shape_f, self.kinds])
        shape_f = self.shape_f.numpy().astype(float)
        on_pad = [-shape_f[k] for k in self.pad_shapes]  # force the block exerts on each pad
        speed = np.linalg.norm(s.particle_qd.numpy()[:self.n_nodes], axis=1) * 1e3
        jac = tet_response(q, self.tets, self.inv, self.materials)[2]
        pads = s.body_q.numpy()[self.pads].astype(float)
        return dict(t=round(t, 3), pad_z_mm=float(pads[0, 2] * 1e3), pad_gap_mm=float((pads[0, 1] - pads[1, 1] - 2 * PAD[1]) * 1e3),
                    com_z_mm=float(self.mass @ q[:, 2] / self.mass.sum() * 1e3), min_z_mm=float(q[:, 2].min() * 1e3),
                    squeeze_N=[float(on_pad[0][1]), float(on_pad[1][1])], lift_N=-float(on_pad[0][2] + on_pad[1][2]),
                    max_speed_mm_s=float(speed.max()), median_speed_mm_s=float(np.median(speed)),
                    nodes_over_1mm_s=int((speed > 1).sum()), min_J=float(jac.min()), inverted=int((jac <= 0).sum()),
                    contacts=count, contact_kinds=self.kinds.numpy().tolist(),
                    contact_impulse_Ns=self.contact_impulse.numpy().tolist(),
                    momentum_kg_m_s=(self.mass @ s.particle_qd.numpy()[:self.n_nodes].astype(float)).tolist())


def run(args):
    t0 = time.perf_counter()
    if args.friction_zero_fix:
        from contact_fixes import enable_friction_zero_fix
        enable_friction_zero_fix()
    g = Grasp(args.material, args.offset_mm * 1e-3, args.squeeze, args.dt, args.iterations, args.ke, args.kd, args.friction_eps, args.self_contact,
              args.pad_radius_mm * 1e-3, args.refined, args.contact_mm * 1e-3)
    total = SETTLE + CLOSE + SQUEEZE_HOLD + LIFT + HOLD
    rows, failure = [], None
    for frame in range(round(total / .01)):
        g.advance(frame)
        if (frame + 1) % 5 == 0:
            try:
                rows.append(g.measure((frame + 1) * .01))
            except (FloatingPointError, RuntimeError) as err:
                failure = str(err)
                break
            if rows[-1]['inverted']:
                failure = f"{rows[-1]['inverted']} inverted tetrahedra at {rows[-1]['t']} s"
                break
    wp.synchronize()
    weight = MASS * 9.81
    lift_start = SETTLE + CLOSE + SQUEEZE_HOLD
    before = [r for r in rows if SETTLE + CLOSE < r['t'] <= lift_start]
    hold = [r for r in rows if r['t'] > total - .5]
    summary = {}
    if before and len(hold) >= 2:
        rise = np.mean([r['com_z_mm'] for r in hold]) - np.mean([r['com_z_mm'] for r in before])
        squeeze = np.array([r['squeeze_N'] for r in hold])
        grip = (squeeze[:, 0] - squeeze[:, 1]) / 2
        # Lift forces are judged only in the final hold: while the pads move, contact damping adds a
        # velocity-dependent term that is not weight.
        summary = dict(
            com_rise_mm=float(rise), carried_fraction=float(rise / (LIFT_M * 1e3)),
            lift_force_over_weight=float(np.mean([r['lift_N'] for r in hold]) / weight),
            squeeze_N=float(grip.mean()),
            squeeze_imbalance_pct=float(abs(np.mean(squeeze[:, 0] + squeeze[:, 1])) / max(grip.mean(), 1e-9) * 100),
            slip_mm_s=float((hold[0]['com_z_mm'] - hold[-1]['com_z_mm']) / (hold[-1]['t'] - hold[0]['t'])),
            lift_force_peak_to_peak_N=float(np.ptp([r['lift_N'] for r in hold])),
            max_speed_mm_s=float(np.mean([r['max_speed_mm_s'] for r in hold])),
            median_speed_mm_s=float(np.mean([r['median_speed_mm_s'] for r in hold])),
            nodes_over_1mm_s=float(np.mean([r['nodes_over_1mm_s'] for r in hold])),
            min_J=min(r['min_J'] for r in rows), min_z_hold_mm=float(np.mean([r['min_z_mm'] for r in hold])))
        duration = hold[-1]['t'] - hold[0]['t']
        impulse = np.asarray(hold[-1]['contact_impulse_Ns']) - np.asarray(hold[0]['contact_impulse_Ns'])
        momentum = np.asarray(hold[-1]['momentum_kg_m_s']) - np.asarray(hold[0]['momentum_kg_m_s'])
        pads_force = impulse[g.pad_shapes].sum(axis=0) / duration
        all_force = impulse.sum(axis=0) / duration
        balance = momentum / duration - all_force - np.array([0., 0., -weight])
        summary.update(instantaneous_lift_force_over_weight=summary['lift_force_over_weight'],
                       lift_force_over_weight=float(pads_force[2] / weight),
                       ground_force_over_weight=float((all_force[2] - pads_force[2]) / weight),
                       momentum_balance_pct=float(np.linalg.norm(balance) / weight * 100),
                       impulse_window_s=duration)
    checks = dict(finished=failure is None and bool(hold),
                  carried=summary.get('carried_fraction', 0) >= .9,
                  weight_carried_by_pads=abs(summary.get('lift_force_over_weight', 0) - 1) <= .05,
                  momentum_balance=summary.get('momentum_balance_pct', 1e9) <= 5,
                  squeeze_balanced=summary.get('squeeze_imbalance_pct', 1e9) <= 5,
                  no_slip=summary.get('slip_mm_s', 1e9) <= .1,
                  quiet=summary.get('median_speed_mm_s', 1e9) <= 1 and summary.get('max_speed_mm_s', 1e9) <= 10)
    result = dict(args=vars(args), actual_dt_s=g.dt, nodes=g.n_nodes, tets=int(g.model.tet_count), weight_N=weight,
                  newton=newton.__version__, executable=sys.executable, wall_s=round(time.perf_counter() - t0, 2),
                  failure=failure, summary=summary, checks=checks, passed=all(checks.values()), trace=rows)
    sources = [Path(__file__), HERE.parent / 'contact_transfer' / 'benchmark.py']
    if args.friction_zero_fix:
        sources.append(HERE.parent / 'contact_transfer' / 'contact_fixes.py')
    result['project_sources_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    name = f'{args.tag}_{args.material}_off{args.offset_mm:g}'
    (out / f'{name}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(name=name, failure=failure, summary=summary, checks=checks)), flush=True)


def record(args):
    """Simulate and save every frame for replay (replay.py). Keeps going after an inversion so the
    failure stays visible; stops only on divergence. Writes results/<tag>_<material>_off<offset>.npz."""
    if args.friction_zero_fix:
        from contact_fixes import enable_friction_zero_fix
        enable_friction_zero_fix()
    g = Grasp(args.material, args.offset_mm * 1e-3, args.squeeze, args.dt, args.iterations, args.ke, args.kd, args.friction_eps, args.self_contact,
              args.pad_radius_mm * 1e-3, args.refined, args.contact_mm * 1e-3)
    frames, pads, events = [], [], []
    for frame in range(round((SETTLE + CLOSE + SQUEEZE_HOLD + LIFT + HOLD) / .01)):
        g.advance(frame)
        q = g.s0.particle_q.numpy()[:g.n_nodes]
        if not np.isfinite(q).all() or np.abs(q).max() > 2:
            events.append(f'diverged at {(frame + 1) * .01:.2f} s')
            break
        frames.append(q.astype(np.float32))
        pads.append(g.s0.body_q.numpy()[g.pads].astype(np.float32))
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    name = f'{args.tag}_{args.material}_off{args.offset_mm:g}'
    np.savez_compressed(out / f'{name}.npz', frames=np.array(frames), pads=np.array(pads), tets=g.tets,
                        pad_half=np.array(PAD), dt_frame=.01, label=f'{args.material} squeeze {args.squeeze:g}',
                        events=np.array(events))
    print(json.dumps(dict(name=name, frames=len(frames), events=events)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--material', choices=sorted(MATERIALS), default='soft')
    ap.add_argument('--offset-mm', type=float, default=0., help='pad centre x offset; 5 puts it between nodes')
    ap.add_argument('--squeeze', type=float, default=.3, help='closed pad gap as a fraction of the block width removed')
    ap.add_argument('--dt', type=float, default=.001)
    ap.add_argument('--iterations', type=int, default=10)
    ap.add_argument('--ke', type=float, default=3000.)
    ap.add_argument('--kd', type=float, default=3.)
    ap.add_argument('--contact-mm', type=float, default=4., help='particle radius and soft contact gap (mm)')
    ap.add_argument('--refined', action='store_true', help='2.5 mm cells in the contact zone (REFINED_AXES)')
    ap.add_argument('--pad-radius-mm', type=float, default=0., help='round every pad edge and corner (0 = sharp box)')
    ap.add_argument('--self-contact', action='store_true', help='enable VBD particle self-contact (surface only)')
    ap.add_argument('--friction-eps', type=float, default=1e-2, help='VBD friction regularisation velocity (m/s)')
    ap.add_argument('--friction-zero-fix', action='store_true', help='project-local continuous zero-slip stiffness')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--record', action='store_true', help='save every frame for replay.py instead of checking')
    args = ap.parse_args()
    if not 0 < args.squeeze < 1 or args.dt <= 0 or args.iterations < 1:
        ap.error('0 < squeeze < 1, dt > 0, iterations >= 1')
    wp.config.quiet = True
    (record if args.record else run)(args)


if __name__ == '__main__':
    main()
