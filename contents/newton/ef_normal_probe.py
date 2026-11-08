"""Track, substep by substep, the contact records on one chattering corner node under the held finger_off tool.

Prints the node's position and, for each record touching it, the normal and penetration at that substep,
to see whether normals or activation flip between substeps. Uses test/contact_transfer/benchmark.py's
Experiment with a kinematic (position-driven) tool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'test' / 'contact_transfer'))
import warp as wp  # noqa: E402

import benchmark as bm  # noqa: E402

wp.config.quiet = True
NODE = int(sys.argv[1]) if len(sys.argv) > 1 else 359
cfg = bm.Config(case='finger_off', dt=.001, iterations=10, ke=3000., kd=3., friction=0., drive='position', tool_z_mm=57.102, seconds=2.)
exp = bm.Experiment(cfg)
for frame in range(150):
    exp.advance(frame)
m, c = exp.model, exp.contacts
radius = m.particle_radius.numpy()
for k in range(9):
    exp.substep(k % exp.substeps)
    q = exp.s0.particle_q.numpy()
    v = exp.s0.particle_qd.numpy()[NODE]
    tool = exp.s0.body_q.numpy()[exp.body][:3]
    n = int(c.soft_contact_count.numpy()[0])
    corners = c.soft_contact_indices.numpy()[:n]
    bary = c.soft_contact_barycentric.numpy()[:n]
    normal = c.soft_contact_normal.numpy()[:n]
    bpos = c.soft_contact_body_pos.numpy()[:n]
    rows = []
    for r in np.flatnonzero((corners == NODE).any(axis=1)):
        ids = [int(i) for i in corners[r] if i >= 0]
        x = sum(w * q[i] for w, i in zip(bary[r][:len(ids)], ids))
        pen = -(float(normal[r] @ (x - (bpos[r] + tool))) - radius[ids[0]])
        w_node = float(bary[r][ids.index(NODE)])
        rows.append(f"{['V', 'E', 'F'][len(ids) - 1]}w{w_node:.2f} n=({normal[r][0]:+.2f},{normal[r][1]:+.2f},{normal[r][2]:+.2f}) pen={pen * 1e3:+.2f}")
    print(f"substep {k}: node z={q[NODE, 2] * 1e3:.3f} mm  v=({v[0] * 1e3:+.0f},{v[1] * 1e3:+.0f},{v[2] * 1e3:+.0f}) mm/s")
    for row in rows:
        print('    ', row)
