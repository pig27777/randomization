"""Step 0: can a rigid body press a tetrahedral soft body in Newton VBD?

A uniform tet slab on the ground carries a resting rigid block or the same weight applied directly to
the top nodes under it. Correct rigid-soft force transfer means the two sink alike, and the block rests
one particle radius above the nodes instead of floating higher or slipping between them.
  plate:      240 x 200 mm, 2 kg, over 81 top nodes
  finger:     20 mm cube, 2 N, centred on one top node
  finger_off: the same finger half a cell (12.5 mm) off in x and y, between four nodes
Repeated over contact settings.

Usage: rigid_tet_check.py [config ...] [--seconds S]
Writes results_unified/rigid_tet_check.json. Newton 1.6.1 (bubble_mailer venv). Values are authored.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import sys

NEWTON = Path(__file__).resolve().parents[2] / "contents/newton"
sys.path.insert(0, str(NEWTON))

import unified_soft as us

HERE = NEWTON
OUT = HERE / 'results_unified'

FRAME_DT = 0.01
SUBSTEPS = 5
ITERATIONS = 10
BASE_Z = 0.002
RADIUS = 0.004  # unified_soft.add_item particle radius
# half extents, mass, xy of each resting block
BLOCKS = {
    'plate': ((0.12, 0.10, 0.01), 2.0, (0.0, 0.0)),
    'finger': ((0.01, 0.01, 0.01), 2.0 / 9.81, (0.0, 0.0)),
    'finger_off': ((0.01, 0.01, 0.01), 2.0 / 9.81, (0.0125, 0.0125)),
}
SLAB = us.SoftShape('slab', (0.30, 0.25, 0.05), (12, 10, 4), 0.3, layers=('core',) * 4,
                    materials={'core': (1.5e3, 1.0e3, 1.0)})

# particle-side soft_contact_ke, rigid shape ke (VBD averages the two), solver options
CONFIGS = {
    # newton/newton_soft.py as it was when Newton was rejected
    'contents_legacy': dict(soft_ke=3e3, shape_ke=2.5e3, alm=True, buffer=2048, k_start=None),
    # bag_sim/newton_bag.py solver options (k_start is a no-op in 1.6.1 while the ramp beta is 0)
    'bag_style': dict(soft_ke=3e3, shape_ke=2.5e3, alm=None, buffer=4096, k_start=1e12),
    'ke_3e4': dict(soft_ke=3e4, shape_ke=3e4, alm=True, buffer=4096, k_start=None),
    'ke_3e5': dict(soft_ke=3e5, shape_ke=3e5, alm=True, buffer=4096, k_start=None),
    'ke_3e5_legacy_alm': dict(soft_ke=3e5, shape_ke=3e5, alm=False, buffer=4096, k_start=None),
    # full-surface contact adds edge and face contacts on top of the vertex ones
    'legacy_vertex_only': dict(soft_ke=3e3, shape_ke=2.5e3, alm=True, buffer=2048, k_start=None, full_surface=False),
    'ke_3e5_vertex_only': dict(soft_ke=3e5, shape_ke=3e5, alm=True, buffer=4096, k_start=None, full_surface=False),
    'ke_1e4_vertex_only': dict(soft_ke=1e4, shape_ke=1e4, alm=True, buffer=4096, k_start=None, full_surface=False),
    'ke_3e4_vertex_only': dict(soft_ke=3e4, shape_ke=3e4, alm=True, buffer=4096, k_start=None, full_surface=False),
    # more substeps raise the particle inertia term m/dt^2 relative to the contact stiffness
    'ke_3e4_vertex_only_sub20': dict(soft_ke=3e4, shape_ke=3e4, alm=True, buffer=4096, k_start=None, full_surface=False,
                                     substeps=20),
    'legacy_sub20': dict(soft_ke=3e3, shape_ke=2.5e3, alm=True, buffer=2048, k_start=None, substeps=20),
    'legacy_sub40': dict(soft_ke=3e3, shape_ke=2.5e3, alm=True, buffer=2048, k_start=None, substeps=40),
    'legacy_iter40': dict(soft_ke=3e3, shape_ke=2.5e3, alm=True, buffer=2048, k_start=None, iterations=40),
    'ke_3e4_sub20': dict(soft_ke=3e4, shape_ke=3e4, alm=True, buffer=4096, k_start=None, substeps=20),
}


def build(cfg, load):
    builder = newton.ModelBuilder(gravity=-9.81)
    shape_cfg = newton.ModelBuilder.ShapeConfig(ke=cfg['shape_ke'], kd=cfg['shape_ke'] * 1e-3, mu=0.8)
    builder.add_ground_plane(cfg=shape_cfg)
    info = us.add_item(builder, SLAB, base_z=BASE_Z)
    box = None
    if load in BLOCKS:
        half, mass, (x, y) = BLOCKS[load]
        top = BASE_Z + SLAB.size[2] + RADIUS
        box = builder.add_body(xform=wp.transform(wp.vec3(x, y, top + 0.001 + half[2]), wp.quat_identity()), label=load)
        box_cfg = newton.ModelBuilder.ShapeConfig(ke=cfg['shape_ke'], kd=cfg['shape_ke'] * 1e-3, mu=0.8,
                                                   density=float(mass / (8 * np.prod(half))))
        if cfg.get('full_surface', True):
            box_cfg.configure_sdf(force_sdf=True, target_voxel_size=0.002)
        builder.add_shape_box(box, hx=half[0], hy=half[1], hz=half[2], cfg=box_cfg)
    builder.color()
    model = builder.finalize()
    model.soft_contact_ke = cfg['soft_ke']
    model.soft_contact_kd = cfg['soft_ke'] * 1e-3
    model.soft_contact_mu = 0.8
    kwargs = dict(iterations=cfg.get('iterations', ITERATIONS), friction_epsilon=1e-2, rigid_body_contact_buffer_size=256,
                  rigid_body_particle_contact_buffer_size=cfg['buffer'])
    if cfg['alm'] is not None:
        kwargs['rigid_compliant_alm'] = cfg['alm']
    if cfg['k_start'] is not None:
        kwargs['rigid_contact_k_start'] = cfg['k_start']
    solver = newton.solvers.SolverVBD(model, **kwargs)
    pipeline = newton.CollisionPipeline(model, broad_phase='nxn', soft_contact_gap=0.006,
                                        enable_rigid_soft_full_surface_contact=cfg.get('full_surface', True))
    return model, solver, pipeline, info, box


def nodes_under(q0, load):
    """Top nodes under the block; for a block between nodes, the four around it."""
    half, _, (x, y) = BLOCKS[load]
    top = np.where(q0[:, 2] > q0[:, 2].max() - 1e-4)[0]
    dx, dy = np.abs(q0[top, 0] - x), np.abs(q0[top, 1] - y)
    inside = top[(dx <= half[0] + 1e-6) & (dy <= half[1] + 1e-6)]
    return inside if len(inside) else top[np.argsort(dx ** 2 + dy ** 2)[:4]]


def run(cfg, load, seconds):
    model, solver, pipeline, info, box = build(cfg, load)
    s0, s1, control, contacts = model.state(), model.state(), model.control(), pipeline.contacts()
    sl = slice(info['start'], info['end'])
    q0 = s0.particle_q.numpy()[sl]
    block = load.removeprefix('direct_') if load != 'none' else 'plate'
    under = nodes_under(q0, block)
    half = BLOCKS[block][0]
    force = None
    if load.startswith('direct_'):
        f = np.zeros((model.particle_count, 3), np.float32)
        f[info['start'] + under, 2] = -BLOCKS[block][1] * 9.81 / len(under)
        force = wp.array(f, dtype=wp.vec3, device=model.device)
    substeps = cfg.get('substeps', SUBSTEPS)
    dt = FRAME_DT / substeps
    frames = int(round(seconds / FRAME_DT))
    trace = []
    t0 = time.time()
    for frame in range(frames):
        for _ in range(substeps):
            s0.clear_forces()
            if force is not None:
                wp.copy(s0.particle_f, force)
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        q = s0.particle_q.numpy()[sl]
        if not np.isfinite(q).all() or np.abs(q).max() > 5:
            return dict(FAILED=f'unstable at frame {frame}')
        if frame % 10 == 9:
            row = dict(t=round((frame + 1) * FRAME_DT, 2), surface_z=float(q[under, 2].mean()))
            if box is not None:
                row['box_bottom_z'] = float(s0.body_q.numpy()[box][2] - half[2])
                row['box_vz'] = float(s0.body_qd.numpy()[box][2])
            trace.append(row)
    q = s0.particle_q.numpy()[sl]
    out = dict(surface_z_mm=float(q[under, 2].mean()) * 1e3, min_z_mm=float(q[:, 2].min()) * 1e3, n_under=len(under),
               wall_s=time.time() - t0, last_0p5s_surface_drift_mm=(trace[-6]['surface_z'] - trace[-1]['surface_z']) * 1e3)
    if box is not None:
        bottom = trace[-1]['box_bottom_z']
        out.update(box_bottom_z_mm=bottom * 1e3, box_vz_mm_s=trace[-1]['box_vz'] * 1e3,
                   # contact sits one particle radius above the nodes: about 0 = resting, < 0 floating above
                   # the contact shell, > 0 inside it; > RADIUS means the block face is below the nodes
                   box_into_shell_mm=(float(q[under, 2].mean()) + RADIUS - bottom) * 1e3,
                   last_0p5s_box_drift_mm=(trace[-6]['box_bottom_z'] - bottom) * 1e3)
    if load == 'none':
        out['q_by_block'] = {b: round(float(q[nodes_under(q0, b), 2].mean()) * 1e3, 3) for b in BLOCKS}
    out['trace'] = trace
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='*', default=list(CONFIGS))
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--blocks', nargs='+', default=list(BLOCKS))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / 'rigid_tet_check.json'
    results = json.loads(path.read_text()) if path.is_file() else {}
    for name in args.configs:
        rest_run = run(CONFIGS[name], 'none', args.seconds)
        res = dict(none=rest_run)
        summary = {}
        for block in args.blocks:
            res[block] = run(CONFIGS[name], block, args.seconds)
            res['direct_' + block] = run(CONFIGS[name], 'direct_' + block, args.seconds)
            if 'FAILED' in rest_run or 'FAILED' in res[block] or 'FAILED' in res['direct_' + block]:
                summary[block] = 'FAILED'
                continue
            # rest height of the same nodes from an unloaded run
            model_q = rest_run['q_by_block'][block]
            summary[block] = dict(direct_sink_mm=round(model_q - res['direct_' + block]['surface_z_mm'], 3),
                                  block_sink_mm=round(model_q - res[block]['surface_z_mm'], 3),
                                  block_into_shell_mm=res[block]['box_into_shell_mm'],
                                  block_vz_mm_s=res[block]['box_vz_mm_s'])
        res['summary'] = summary
        res['config'] = CONFIGS[name]
        results[name] = res
        print(name, json.dumps(summary), flush=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
