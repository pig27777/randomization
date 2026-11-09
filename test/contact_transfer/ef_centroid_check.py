"""Test: does a jumping contact point on flat edges/faces drive the rigid-tet chatter?

Under a flat tool, a soft edge or face parallel to the tool surface is equally deep everywhere, so the
single deepest-point contact record jumps to whichever corner is momentarily higher and loads it with
almost all the force; that corner sinks, the point jumps back, and so on.

This wraps test/contact_transfer/benchmark.py's Experiment without editing it: after every collision
pass, edge records nearly perpendicular to the contact normal get their point pinned to the edge
midpoint, and face records nearly parallel to the tool surface to the face centroid. The rigid point
lies on the same tool plane, so penetration becomes the corner average and varies smoothly.
Results go to results_unified/ef_centroid/.

Usage: ef_centroid_check.py [run ...] [--off]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = NEWTON
sys.path.insert(0, str(HERE.parents[1] / 'test' / 'contact_transfer'))
import warp as wp  # noqa: E402


NEWTON = Path(__file__).resolve().parents[2] / "contents/newton"
sys.path.insert(0, str(NEWTON))

import benchmark as bm  # noqa: E402
from ef_dedup_check import BASE, RUNS  # noqa: E402

OUT = HERE / 'results_unified' / 'ef_centroid'
PARALLEL_COS = .95  # face: |face normal . contact normal| above this; edge: |edge dir . contact normal| below 1 - this


@wp.func
def _under_tool(x: wp.vec3, X_wb: wp.transform, half: wp.vec3) -> bool:
    """Whether x projects inside the tool box's bottom face (box half extents in the body frame)."""
    p = wp.transform_point(wp.transform_inverse(X_wb), x)
    return wp.abs(p[0]) <= half[0] and wp.abs(p[1]) <= half[1]


@wp.kernel
def _pin_parallel(count: wp.array[int], corners: wp.array[wp.vec3i], q: wp.array[wp.vec3],
                  normal: wp.array[wp.vec3], cos_min: float, footprint_only: int, shape: wp.array[int],
                  shape_body: wp.array[int], body_q: wp.array[wp.transform], half: wp.vec3,
                  bary: wp.array[wp.vec3], pinned: wp.array[int]):
    k = wp.tid()
    if k >= count[0]:
        return
    c = corners[k]
    if c[1] < 0:
        return
    n = normal[k]
    body = shape_body[shape[k]]
    X_wb = wp.transform_identity()
    if body >= 0:
        X_wb = body_q[body]
    if c[2] < 0:
        e = wp.normalize(q[c[1]] - q[c[0]])
        mid = .5 * (q[c[0]] + q[c[1]])
        if wp.abs(wp.dot(e, n)) < 1. - cos_min and (footprint_only == 0 or _under_tool(mid, X_wb, half)):
            bary[k] = wp.vec3(.5, .5, 0.)
            wp.atomic_add(pinned, 0, 1)
    else:
        f = wp.normalize(wp.cross(q[c[1]] - q[c[0]], q[c[2]] - q[c[0]]))
        centroid = (q[c[0]] + q[c[1]] + q[c[2]]) / 3.
        if wp.abs(wp.dot(f, n)) > cos_min and (footprint_only == 0 or _under_tool(centroid, X_wb, half)):
            bary[k] = wp.vec3(1. / 3., 1. / 3., 1. / 3.)
            wp.atomic_add(pinned, 1, 1)


class CentroidExperiment(bm.Experiment):
    footprint_only = 0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pinned = wp.zeros(2, dtype=int)
        collide = self.pipeline.collide

        def collide_pinned(state, contacts):
            collide(state, contacts)
            self.pinned.zero_()
            wp.launch(_pin_parallel, dim=contacts.soft_contact_max,
                      inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, state.particle_q,
                              contacts.soft_contact_normal, PARALLEL_COS, self.footprint_only, contacts.soft_contact_shape,
                              self.model.shape_body, state.body_q, wp.vec3(*self.half),
                              contacts.soft_contact_barycentric, self.pinned])

        self.pipeline.collide = collide_pinned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', default=list(RUNS))
    ap.add_argument('--footprint', action='store_true', help='pin only where the pinned point lies under the tool face')
    args = ap.parse_args()
    wp.config.quiet = True
    CentroidExperiment.footprint_only = int(args.footprint)
    bm.Experiment = CentroidExperiment  # run() looks the class up in its module
    for name in args.runs:
        bm.run(replace(BASE, tag=('centroid_fp_' if args.footprint else 'centroid_') + name, **RUNS[name]), OUT)


if __name__ == '__main__':
    main()
