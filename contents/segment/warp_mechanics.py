"""Warp candidate for frozen v1.2 face forces (CPU validation comes first).

One thread owns an environment. Ordered reductions preserve the reference's
all-A then all-B accumulation and avoid nondeterministic floating-point atomics.
This module is opt-in until the frozen-reference equivalence gate passes.
"""
from __future__ import annotations

import numpy as np
import warp as wp

import bushing

wp.set_module_options({"enable_backward": False, "enable_fp_fma": False})


@wp.func
def norm3(v: wp.vec3d):
    return wp.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


@wp.func
def rotation_log(a: wp.vec4d, b: wp.vec4d):
    w = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    av, bv = wp.vec3d(a[1], a[2], a[3]), wp.vec3d(b[1], b[2], b[3])
    v = a[0]*bv - b[0]*av - wp.cross(av, bv)
    sign = wp.float64(1.)
    if w < wp.float64(0.):
        sign = wp.float64(-1.)
    v, w = v*sign, w*sign
    s = norm3(v)
    scale = wp.float64(2.)
    if s > wp.float64(1.e-12):
        scale = wp.float64(2.) * wp.atan2(s, w) / s
    return v * scale


@wp.func
def log_jacobian_transpose(phi: wp.vec3d, moment: wp.vec3d):
    theta = norm3(phi)
    a = wp.float64(1./12.)
    if theta > wp.float64(1.e-4):
        a = (wp.float64(1.) - wp.float64(.5)*theta/wp.tan(wp.float64(.5)*theta)) / (theta*theta)
    first = wp.cross(phi, moment)
    return moment + wp.float64(.5)*first + a*wp.cross(phi, first)


@wp.kernel
def face_forces(
    a: wp.array(dtype=int), b: wp.array(dtype=int),
    node_a: wp.array(dtype=int), node_b: wp.array(dtype=int),
    axis: wp.array(dtype=int), arm: wp.array(dtype=wp.vec3d),
    kt: wp.array(dtype=wp.vec3d), kr: wp.array(dtype=wp.vec3d),
    tension: wp.array(dtype=wp.float64), travel: wp.array(dtype=wp.float64),
    weight_a: wp.array(dtype=wp.float64), weight_b: wp.array(dtype=wp.float64),
    centered: bool,
    xpos: wp.array2d(dtype=wp.vec3d), xmat: wp.array2d(dtype=wp.mat33d),
    xquat: wp.array2d(dtype=wp.vec4d), xipos: wp.array2d(dtype=wp.vec3d),
    xfrc: wp.array3d(dtype=wp.float64),
    forces: wp.array2d(dtype=wp.vec3d),
    torques_a: wp.array2d(dtype=wp.vec3d), torques_b: wp.array2d(dtype=wp.vec3d),
    compact: wp.array2d(dtype=wp.float64), node: wp.array2d(dtype=wp.vec3d),
    rho: wp.array2d(dtype=wp.float64),
):
    env = wp.tid()
    n = a.shape[0]
    for i in range(node.shape[1]):
        node[env, i] = wp.vec3d(wp.float64(0.))
    for i in range(n):
        ai = a[i]
        bi = b[i]
        ax = axis[i]
        ra = xmat[env, ai]
        rb = xmat[env, bi]
        aa = ra @ arm[i]
        ab = rb @ (-arm[i])
        delta = xpos[env, bi] + ab - xpos[env, ai] - aa
        u = wp.transpose(ra) @ delta
        phi = rotation_log(xquat[env, ai], xquat[env, bi])
        stiffness = kt[i]
        if u[ax] > wp.float64(0.):
            stiffness[ax] = tension[i]
        local_f = wp.cw_mul(stiffness, u)
        moment = wp.cw_mul(kr[i], phi)
        f = ra @ local_f
        torque = ra @ log_jacobian_transpose(phi, moment)
        ta = torque + wp.cross(aa + delta, f)
        tb = -torque - wp.cross(ab, f)
        if not centered:
            ta += wp.cross(xpos[env, ai] - xipos[env, ai], f)
            tb += wp.cross(xpos[env, bi] - xipos[env, bi], -f)
        forces[env, i] = f
        torques_a[env, i] = ta
        torques_b[env, i] = tb
        compact[env, i] = wp.max(-u[ax]/travel[i], wp.float64(0.))
    for i in range(n):
        ni = node_a[i]
        axis_a = axis[i]
        value = node[env, ni]
        value[axis_a] += compact[env, i]*weight_a[i]
        node[env, ni] = value
    for i in range(n):
        nj = node_b[i]
        axis_b = axis[i]
        value_b = node[env, nj]
        value_b[axis_b] += compact[env, i]*weight_b[i]
        node[env, nj] = value_b
    for i in range(n):
        va = node[env, node_a[i]]
        vb = node[env, node_b[i]]
        axis_rho = axis[i]
        across_a = va[0] + va[1] + va[2] - va[axis_rho]
        across_b = vb[0] + vb[1] + vb[2] - vb[axis_rho]
        rho[env, i] = wp.max(across_a, across_b)
    # np.add.at reference order: A forces, B forces, A torques, B torques.
    for i in range(n):
        for c in range(3):
            xfrc[env, a[i], c] += forces[env, i][c]
    for i in range(n):
        for c in range(3):
            xfrc[env, b[i], c] -= forces[env, i][c]
    for i in range(n):
        for c in range(3):
            xfrc[env, a[i], c+3] += torques_a[env, i][c]
    for i in range(n):
        for c in range(3):
            xfrc[env, b[i], c+3] += torques_b[env, i][c]


class WarpBushings(bushing.Bushings):
    """CPU zero-copy adapter; validation explicitly opts into this candidate."""
    def __init__(self, *args):
        super().__init__(*args)
        if any((bushing.ANGLE_STOP_GAIN, bushing.COMPRESSION_STOP_GAIN, bushing.COMPACTION_BEND_GAIN)):
            raise ValueError("Warp candidate requires the frozen v1.2 zero penalty gains")
        self._inputs = [wp.array(value, dtype=dtype, device="cpu") for value, dtype in (
            (self.a, int), (self.b, int), (self.node_a, int), (self.node_b, int),
            (self.axis, int), (self.ra, wp.vec3d), (self.kt, wp.vec3d), (self.kr, wp.vec3d),
            (self.tension, wp.float64), (self.compression_travel, wp.float64),
            (self.weight_a, wp.float64), (self.weight_b, wp.float64))]
        self._forces = [wp.zeros((1, self.n), dtype=wp.vec3d, device="cpu") for _ in range(3)]
        self._compact = wp.zeros((1, self.n), dtype=wp.float64, device="cpu")
        self._node = wp.zeros((1, self.node_count), dtype=wp.vec3d, device="cpu")
        self._rho = wp.zeros((1, self.n), dtype=wp.float64, device="cpu")
        self._views = None

    def apply(self, m, d):
        if not self.n:
            return
        if self._views is None:
            arrays = ((d.xpos, wp.vec3d), (d.xmat.reshape(-1, 3, 3), wp.mat33d),
                      (d.xquat, wp.vec4d), (d.xipos, wp.vec3d), (d.xfrc_applied, wp.float64))
            self._views = [wp.array(array[None], dtype=dtype, device="cpu", copy=False)
                           for array, dtype in arrays]
            self.last_wrenches = tuple(value.numpy()[0] for value in self._forces)
            self.orthogonal_compaction = self._rho.numpy()[0]
        wp.launch(face_forces, dim=1, device="cpu",
                  inputs=[*self._inputs, self.centered, *self._views, *self._forces,
                          self._compact, self._node, self._rho])
