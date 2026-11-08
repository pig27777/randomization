"""Audit the existing centroid experiment without changing its contact rule or validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

import benchmark as bm

sys.path.insert(0, str(bm.ROOT / 'contents' / 'newton'))
from ef_centroid_check import CentroidExperiment


def contact_snapshot(exp):
    c, s = exp.contacts, exp.s0
    n = int(c.soft_contact_count.numpy()[0])
    ids = c.soft_contact_indices.numpy()[:n]
    bary = c.soft_contact_barycentric.numpy()[:n].astype(float)
    q = s.particle_q.numpy().astype(float)
    valid = ids >= 0
    x = (q[np.maximum(ids, 0)] * (bary * valid)[..., None]).sum(axis=1)
    pose = s.body_q.numpy()[exp.body].astype(float)
    bx = bm.transform_points(c.soft_contact_body_pos.numpy()[:n], pose)
    normal = c.soft_contact_normal.numpy()[:n].astype(float)
    pen = exp.model.particle_radius.numpy()[ids[:, 0]] - np.sum(normal * (x - bx), axis=1)
    ke = exp.solver.body_particle_contact_penalty_k.numpy()[:n]
    active = pen > 0
    count = valid.sum(axis=1)
    centroid = np.divide(valid, count[:, None])
    pinned = (count > 1) & (np.abs(bary - centroid).max(axis=1) < 1e-5)
    return dict(keys=[tuple(map(int, row)) for row in ids], pinned=pinned, active=active,
                contact_energy_J=float(np.sum(.5 * ke * np.maximum(pen, 0) ** 2)),
                reaction_N=float(np.sum(ke * np.maximum(pen, 0) * -normal[:, 2])),
                max_penetration_mm=float(max(0., pen.max()) * 1e3),
                pinned_records=int(pinned.sum()), active_pinned=int((pinned & active).sum()))


def run(case):
    cfg = bm.Config(case=case, dt=.001, seconds=2., drive='position',
                    tool_z_mm=60.148 if case == 'plate' else 57.102)
    CentroidExperiment.footprint_only = 1
    exp = CentroidExperiment(cfg)
    for frame in range(150):
        exp.advance(frame)
    exp.z_from.assign(exp.z_to.numpy())
    rows, colors = [], []
    apply = exp.solver._penetration_free_truncation

    def audited_apply(*args):
        before = exp.s0.particle_q.numpy()
        apply(*args)
        after = exp.s0.particle_q.numpy()
        snap = contact_snapshot(exp)
        colors.append(dict(step=len(rows), update_index=len(colors),
                           max_update_mm=float(np.linalg.norm(after - before, axis=1).max() * 1e3),
                           **{key: value for key, value in snap.items() if key not in ('keys', 'pinned', 'active')}))

    exp.solver._penetration_free_truncation = audited_apply
    previous = None
    for step in range(12):
        exp.substep(step % exp.substeps)
        snap = contact_snapshot(exp)
        current = {key: (bool(pin), bool(active)) for key, pin, active in zip(snap['keys'], snap['pinned'], snap['active'])}
        common = current.keys() & previous.keys() if previous is not None else set()
        rows.append(dict(step=step, condition_switches=sum(current[k][0] != previous[k][0] for k in common),
                         activation_switches=sum(current[k][1] != previous[k][1] for k in common),
                         **{key: value for key, value in snap.items() if key not in ('keys', 'pinned', 'active')}))
        previous = current
    result = dict(case=case, newton=bm.newton.__version__, color_groups=len(exp.model.particle_color_groups),
                  substeps=rows, color_updates=colors)
    path = Path(__file__).parent / 'results' / f'repair_centroid_audit_{case}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(case=case, condition_switches=sum(r['condition_switches'] for r in rows),
                         activation_switches=sum(r['activation_switches'] for r in rows),
                         pinned_range=[min(r['pinned_records'] for r in rows), max(r['pinned_records'] for r in rows)],
                         max_color_update_mm=max(r['max_update_mm'] for r in colors))))


if __name__ == '__main__':
    wp.config.quiet = True
    for case in ('plate', 'finger_off'):
        run(case)
