"""Which nodes chatter under a kinematic finger held between slab nodes, and which contacts touch them.

Builds test/contact_transfer/benchmark.py's Experiment (position drive), runs it to steady load, then
records every substep for a short window: per-node velocity, and the soft-contact records
(corners, barycentric weights, normal, penetration) touching the fastest nodes.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = NEWTON
sys.path.insert(0, str(HERE.parents[1] / 'test' / 'contact_transfer'))
import warp as wp  # noqa: E402


NEWTON = Path(__file__).resolve().parents[2] / "contents/newton"
sys.path.insert(0, str(NEWTON))

import benchmark as bm  # noqa: E402

wp.config.quiet = True
case = sys.argv[1] if len(sys.argv) > 1 else 'finger_off'
cfg = bm.Config(case=case, dt=.001, iterations=10, ke=3000., kd=3., friction=0., drive='position', tool_z_mm=57.102, seconds=2.)
exp = bm.Experiment(cfg)
for frame in range(150):
    exp.advance(frame)
wp.synchronize()

m, c = exp.model, exp.contacts
n_tet = exp.n_tet_nodes
window = 12
vel, pos = [], []
for k in range(window):
    exp.substep(k % exp.substeps)
    vel.append(exp.s0.particle_qd.numpy()[:n_tet].astype(float))
    pos.append(exp.s0.particle_q.numpy()[:n_tet].astype(float))
vel, pos = np.array(vel), np.array(pos)
speed = np.linalg.norm(vel, axis=2)
fast = np.argsort(speed.max(axis=0))[::-1][:6]

count = int(c.soft_contact_count.numpy()[0])
corners = c.soft_contact_indices.numpy()[:count]
bary = c.soft_contact_barycentric.numpy()[:count]
normal = c.soft_contact_normal.numpy()[:count]
body_pos = c.soft_contact_body_pos.numpy()[:count]
shape = c.soft_contact_shape.numpy()[:count]
tool_pose = exp.s0.body_q.numpy()[exp.body]
q = exp.s0.particle_q.numpy()
radius = m.particle_radius.numpy()

report = dict(case=case, tool_xyz_mm=(tool_pose[:3] * 1e3).round(3).tolist(), contacts=count, fast_nodes=[])
for i in fast:
    rest = exp.q0[i]
    row = dict(node=int(i), rest_xyz_mm=(rest * 1e3).round(2).tolist(),
               max_speed_mm_s=round(float(speed[:, i].max() * 1e3), 1),
               velocity_mm_s_per_substep=(vel[:, i] * 1e3).round(1).tolist(),
               z_mm_per_substep=(pos[:, i, 2] * 1e3).round(3).tolist(), records=[])
    for r in np.flatnonzero((corners == i).any(axis=1)):
        ids = [int(v) for v in corners[r] if v >= 0]
        w = bary[r][:len(ids)]
        x = sum(wj * q[v] for wj, v in zip(w, ids))
        bx = body_pos[r] + tool_pose[:3] if shape[r] == exp.tool_shape else body_pos[r]  # tool is unrotated
        pen = -(float(normal[r] @ (x - bx)) - max(radius[v] for v in ids))
        row['records'].append(dict(kind=['vertex', 'edge', 'face'][len(ids) - 1], corners=ids,
                                   weight_of_node=round(float(w[ids.index(int(i))]), 3),
                                   normal=normal[r].round(3).tolist(), penetration_mm=round(pen * 1e3, 3),
                                   shape='tool' if shape[r] == exp.tool_shape else int(shape[r])))
    report['fast_nodes'].append(row)

out = HERE / 'results_unified' / 'ef_probe'
out.mkdir(parents=True, exist_ok=True)
(out / f'node_probe_{case}.json').write_text(json.dumps(report, indent=1, default=float))
print(json.dumps(report, indent=1, default=float))
