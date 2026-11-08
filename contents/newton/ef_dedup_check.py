"""Test: are degenerate edge/face contacts (collapsed onto one vertex) behind the rigid-tet chatter?

Full-surface contact emits edge and face records whose barycentric weight can sit entirely on one
vertex. Such a record duplicates that vertex's own contact (same point, same normal), so the vertex
sees the contact stiffness several times over, switching on and off together near the threshold.

This wraps test/contact_transfer/benchmark.py's Experiment without editing it: after every collision
pass, a kernel disables edge/face records whose largest weight >= THRESHOLD by moving their rigid
contact point 1 m back along the normal, so they never penetrate (no force, no Hessian, either side).
Results go to results_unified/ef_dedup/.

Usage: ef_dedup_check.py [run ...]   runs: see RUNS
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'test' / 'contact_transfer'))
import warp as wp  # noqa: E402

import benchmark as bm  # noqa: E402

OUT = HERE / 'results_unified' / 'ef_dedup'
THRESHOLD = .99


@wp.kernel
def _disable_degenerate(count: wp.array[int], corners: wp.array[wp.vec3i], bary: wp.array[wp.vec3],
                        normal: wp.array[wp.vec3], shape: wp.array[int], shape_body: wp.array[int],
                        body_q: wp.array[wp.transform], threshold: float, body_pos: wp.array[wp.vec3],
                        disabled: wp.array[int]):
    k = wp.tid()
    if k >= count[0] or corners[k][1] < 0:
        return
    b = bary[k]
    if wp.max(b[0], wp.max(b[1], b[2])) < threshold:
        return
    body = shape_body[shape[k]]
    n_local = normal[k]
    if body >= 0:
        n_local = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[body]), normal[k])
    body_pos[k] = body_pos[k] - n_local
    wp.atomic_add(disabled, 0, 1)


class DedupExperiment(bm.Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.disabled = wp.zeros(1, dtype=int)
        collide = self.pipeline.collide
        m = self.model

        def collide_dedup(state, contacts):
            collide(state, contacts)
            self.disabled.zero_()
            wp.launch(_disable_degenerate, dim=contacts.soft_contact_max,
                      inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
                              contacts.soft_contact_normal, contacts.soft_contact_shape, m.shape_body, state.body_q, THRESHOLD,
                              contacts.soft_contact_body_pos, self.disabled])

        self.pipeline.collide = collide_dedup


POS = dict(drive='position', seconds=2.)
RUNS = {
    # kinematic tool held at the benchmark's settled heights: contact path only
    'pos_finger_off': dict(case='finger_off', tool_z_mm=57.102, **POS),
    'pos_finger': dict(case='finger', tool_z_mm=57.102, **POS),
    'pos_plate': dict(case='plate', tool_z_mm=60.148, **POS),
    # formal force-driven, vertically guided runs (compare guided_base_* in test/contact_transfer/REPORT.md)
    'guided_finger_off': dict(case='finger_off'),
    'guided_finger': dict(case='finger'),
    'guided_plate': dict(case='plate'),
}
BASE = bm.Config(dt=.001, iterations=10, ke=3000., kd=3., friction=0.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', default=list(RUNS))
    ap.add_argument('--off', action='store_true', help='same runs without the filter (control)')
    args = ap.parse_args()
    wp.config.quiet = True
    if not args.off:
        bm.Experiment = DedupExperiment  # run() looks the class up in its module
    for name in args.runs:
        bm.run(replace(BASE, tag=('raw_' if args.off else 'dedup_') + name, **RUNS[name]), OUT)


if __name__ == '__main__':
    main()
