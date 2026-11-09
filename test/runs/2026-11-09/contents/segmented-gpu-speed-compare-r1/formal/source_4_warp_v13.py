"""Warp operators for the v1.3 GPU backend (separate from the v1.2 weld-era warp_* files).

Reused unchanged from S2: warp_effects.compaction (compaction welds and their load filter) and the
face-force mathematics of warp_mechanics.face_forces. New here:
  face_forces_env  the same face forces with per-environment material parameters (heterogeneous kappa);
  failure_metrics  Bushings.check ratios and finiteness;
  pack_contacts    per-environment contact lists with per-row efc addresses, dim and dist;
  readouts         contact_reactions, pad_normal_force maxima, table contacts and COM;
  pinch_*          v1.3 top pinch: top contacts, eligibility with the fingertip band, point-connection
                   binding and the resultant connection force.
The 5 ms low-pass release is evaluated on the host in float64 (LoadFilter), exactly as pinch.py does.
All material and pinch arithmetic is float64; MJWarp state is float32 and is promoted first.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from warp_mechanics import log_jacobian_transpose, norm3, rotation_log

wp.set_module_options({"enable_backward": False, "enable_fp_fma": False})

BAND_SAMPLES = wp.constant(41)             # pinch.PinchAttachment.covered defaults
BAND_ACROSS = wp.constant(9)


@wp.kernel
def face_forces_env(
    a: wp.array(dtype=int), b: wp.array(dtype=int),
    node_a: wp.array(dtype=int), node_b: wp.array(dtype=int), axis: wp.array(dtype=int),
    weight_a: wp.array(dtype=wp.float64), weight_b: wp.array(dtype=wp.float64),
    arm: wp.array2d(dtype=wp.vec3d), kt: wp.array2d(dtype=wp.vec3d), kr: wp.array2d(dtype=wp.vec3d),
    tension: wp.array2d(dtype=wp.float64), travel: wp.array2d(dtype=wp.float64),
    centered: bool,
    xpos: wp.array2d(dtype=wp.vec3d), xmat: wp.array2d(dtype=wp.mat33d),
    xquat: wp.array2d(dtype=wp.vec4d), xipos: wp.array2d(dtype=wp.vec3d),
    xfrc: wp.array3d(dtype=wp.float64),
    forces: wp.array2d(dtype=wp.vec3d),
    torques_a: wp.array2d(dtype=wp.vec3d), torques_b: wp.array2d(dtype=wp.vec3d),
    compact: wp.array2d(dtype=wp.float64), node: wp.array2d(dtype=wp.vec3d),
    rho: wp.array2d(dtype=wp.float64),
):
    """warp_mechanics.face_forces with [env, connection] parameters; identical operation order."""
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
        aa = ra @ arm[env, i]
        ab = rb @ (-arm[env, i])
        delta = xpos[env, bi] + ab - xpos[env, ai] - aa
        u = wp.transpose(ra) @ delta
        phi = rotation_log(xquat[env, ai], xquat[env, bi])
        stiffness = kt[env, i]
        if u[ax] > wp.float64(0.):
            stiffness[ax] = tension[env, i]
        local_f = wp.cw_mul(stiffness, u)
        moment = wp.cw_mul(kr[env, i], phi)
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
        compact[env, i] = wp.max(-u[ax]/travel[env, i], wp.float64(0.))
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


@wp.kernel
def failure_metrics(
    a: wp.array(dtype=int), b: wp.array(dtype=int), axis: wp.array(dtype=int),
    arm: wp.array2d(dtype=wp.vec3d), compression_limit: wp.array2d(dtype=wp.float64),
    bend_limit: wp.array2d(dtype=wp.vec3d),
    xpos: wp.array2d(dtype=wp.vec3d), xmat: wp.array2d(dtype=wp.mat33d), xquat: wp.array2d(dtype=wp.vec4d),
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
    out: wp.array2d(dtype=wp.float64),
):
    """[max overlap strain, max compression / limit, max |phi| / bending limit, finite qpos and qvel]."""
    env = wp.tid()
    finite = wp.float64(1.)
    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[env, i]):
            finite = wp.float64(0.)
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[env, i]):
            finite = wp.float64(0.)
    strain_max = wp.float64(-1.e300)
    ratio_max = wp.float64(-1.e300)
    bend_max = wp.float64(0.)
    for i in range(a.shape[0]):
        ra = xmat[env, a[i]]
        rb = xmat[env, b[i]]
        delta = xpos[env, b[i]] + rb @ (-arm[env, i]) - xpos[env, a[i]] - ra @ arm[env, i]
        u = wp.transpose(ra) @ delta
        ax = axis[i]
        strain = -u[ax]/(wp.float64(2.)*arm[env, i][ax])
        strain_max = wp.max(strain_max, strain)
        ratio_max = wp.max(ratio_max, strain/compression_limit[env, i])
        phi = rotation_log(xquat[env, a[i]], xquat[env, b[i]])
        limit = bend_limit[env, i]
        for c in range(3):
            if c != ax:
                bend_max = wp.max(bend_max, wp.abs(phi[c])/limit[c])
    out[env, 0] = strain_max
    out[env, 1] = ratio_max
    out[env, 2] = bend_max
    out[env, 3] = finite


@wp.kernel
def pack_contacts(
    nacon: wp.array(dtype=int), world: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    frame: wp.array(dtype=wp.mat33), address: wp.array2d(dtype=int), dim: wp.array(dtype=int),
    dist: wp.array(dtype=float), counters: wp.array(dtype=int), overflow: wp.array(dtype=int),
    out_geom: wp.array3d(dtype=int), out_frame: wp.array3d(dtype=wp.float64),
    out_address: wp.array3d(dtype=int), out_dist: wp.array2d(dtype=wp.float64),
):
    """Per-environment contact lists; rows beyond dim get address -1 (MJWarp rows need not be adjacent)."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    env = world[i]
    if env < 0 or env >= counters.shape[0]:
        return
    slot = wp.atomic_add(counters, env, 1)
    if slot >= out_address.shape[1]:
        wp.atomic_add(overflow, 0, 1)
        return
    g = geom[i]
    out_geom[env, slot, 0] = g[0]
    out_geom[env, slot, 1] = g[1]
    r = frame[i]
    for row in range(3):
        for col in range(3):
            out_frame[env, slot, 3*row+col] = wp.float64(r[row, col])
    for row in range(3):
        out_address[env, slot, row] = -1
        if row < dim[i]:
            out_address[env, slot, row] = address[i, row]
    out_dist[env, slot] = wp.float64(dist[i])


@wp.kernel
def readouts(
    count: wp.array(dtype=int), contact_geom: wp.array3d(dtype=int), contact_frame: wp.array3d(dtype=wp.float64),
    contact_address: wp.array3d(dtype=int), contact_dist: wp.array2d(dtype=wp.float64),
    efc_force: wp.array2d(dtype=wp.float64), pad_of_geom: wp.array(dtype=int), content: wp.array(dtype=bool),
    table_geom: int, bodies: wp.array(dtype=int), mass: wp.array2d(dtype=wp.float64),
    xipos: wp.array2d(dtype=wp.vec3d), qpos: wp.array2d(dtype=wp.float64), pad_qadr: wp.array2d(dtype=int),
    out: wp.array2d(dtype=wp.float64),
):
    """Per environment: 0-5 force on pads A, B by the object; 6-8 table force on the object;
    9 table contact count (efc rows, dist <= 0); 10-11 max over world axes of the summed |normal| force per pad;
    12-14 object COM; 15-20 pad joint positions A, B (x, y, z slides)."""
    env = wp.tid()
    fa = wp.vec3d(wp.float64(0.))
    fb = wp.vec3d(wp.float64(0.))
    table = wp.vec3d(wp.float64(0.))
    touching = wp.float64(0.)
    normal_a = wp.vec3d(wp.float64(0.))
    normal_b = wp.vec3d(wp.float64(0.))
    for c in range(count[env]):
        g1 = contact_geom[env, c, 0]
        g2 = contact_geom[env, c, 1]
        local = wp.vec3d(wp.float64(0.))
        for row in range(3):
            address = contact_address[env, c, row]
            if address >= 0:
                local[row] = efc_force[env, address]
        on2 = wp.vec3d(wp.float64(0.))
        for col in range(3):
            for k in range(3):
                on2[col] += contact_frame[env, c, 3*k+col]*local[k]
        p1 = pad_of_geom[g1]
        p2 = pad_of_geom[g2]
        if p1 == 0 and content[g2]:
            fa -= on2
        if p2 == 0 and content[g1]:
            fa += on2
        if p1 == 1 and content[g2]:
            fb -= on2
        if p2 == 1 and content[g1]:
            fb += on2
        if g1 == table_geom and content[g2]:
            table += on2
        if g2 == table_geom and content[g1]:
            table -= on2
        if contact_address[env, c, 0] >= 0 and contact_dist[env, c] <= wp.float64(0.):
            if (g1 == table_geom and content[g2]) or (g2 == table_geom and content[g1]):
                touching += wp.float64(1.)
        if contact_address[env, c, 0] >= 0:
            fn = efc_force[env, contact_address[env, c, 0]]
            normal = wp.vec3d(wp.abs(contact_frame[env, c, 0]), wp.abs(contact_frame[env, c, 1]),
                              wp.abs(contact_frame[env, c, 2]))
            if p1 == 0 or p2 == 0:
                normal_a += fn*normal
            if p1 == 1 or p2 == 1:
                normal_b += fn*normal
    total = wp.float64(0.)
    weighted = wp.vec3d(wp.float64(0.))
    for i in range(bodies.shape[0]):
        weighted += mass[env, i]*xipos[env, bodies[i]]
        total += mass[env, i]
    com = weighted/total
    for c in range(3):
        out[env, c] = fa[c]
        out[env, 3+c] = fb[c]
        out[env, 6+c] = table[c]
        out[env, 12+c] = com[c]
        out[env, 15+c] = qpos[env, pad_qadr[0, c]]
        out[env, 18+c] = qpos[env, pad_qadr[1, c]]
    out[env, 9] = touching
    out[env, 10] = wp.max(normal_a[0], wp.max(normal_a[1], normal_a[2]))
    out[env, 11] = wp.max(normal_b[0], wp.max(normal_b[1], normal_b[2]))


@wp.kernel
def promote_qpos(source: wp.array2d(dtype=float), target: wp.array2d(dtype=wp.float64)):
    env, i = wp.tid()
    target[env, i] = wp.float64(source[env, i])


@wp.kernel
def pinch_top_contacts(
    top: wp.array(dtype=int), bodies: wp.array(dtype=int), geom_body: wp.array(dtype=int),
    pad_of_geom: wp.array(dtype=int),
    count: wp.array(dtype=int), contact_geom: wp.array3d(dtype=int), contact_frame: wp.array3d(dtype=wp.float64),
    contact_address: wp.array3d(dtype=int), efc_force: wp.array2d(dtype=wp.float64),
    xmat: wp.array2d(dtype=wp.mat33d), contacts: wp.array3d(dtype=int),
):
    """PinchAttachment.top_contacts: top segments touched by each fingertip on an upward-facing surface."""
    env = wp.tid()
    for pad in range(2):
        for i in range(top.shape[0]):
            contacts[env, pad, i] = 0
    for c in range(count[env]):
        address = contact_address[env, c, 0]
        if address < 0:
            continue
        if efc_force[env, address] <= wp.float64(.05):
            continue
        g1 = contact_geom[env, c, 0]
        g2 = contact_geom[env, c, 1]
        for pad in range(2):
            other = int(-1)
            sign = wp.float64(1.)
            if pad_of_geom[g1] == pad:
                other = g2
                sign = wp.float64(-1.)
            elif pad_of_geom[g2] == pad:
                other = g1
            if other < 0:
                continue
            body = geom_body[other]
            normal = sign*wp.vec3d(contact_frame[env, c, 0], contact_frame[env, c, 1], contact_frame[env, c, 2])
            for i in range(top.shape[0]):
                if bodies[top[i]] == body:
                    r = xmat[env, body]
                    z = wp.vec3d(r[0, 2], r[1, 2], r[2, 2])
                    if wp.dot(normal, z) > wp.float64(.5):
                        contacts[env, pad, i] = 1


@wp.kernel
def pinch_candidate(
    request: wp.array(dtype=int), top: wp.array(dtype=int), bodies: wp.array(dtype=int),
    half: wp.array2d(dtype=wp.vec3d), contacts: wp.array3d(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3d), xmat: wp.array2d(dtype=wp.mat33d),
    tips: wp.array2d(dtype=wp.vec3d), original_top: wp.array(dtype=wp.float64),
    depth_min: wp.float64, tip_width: wp.float64,
    covered: wp.array2d(dtype=int), eligible: wp.array(dtype=int),
):
    """PinchAttachment.candidate (callers pass only unbound, unbroken, non-rigid environments):
    depth, both fingertips on the top patch near the pinch, and the fingertip band coverage."""
    env = wp.tid()
    eligible[env] = 0
    for i in range(top.shape[0]):
        covered[env, i] = 0
    if request[env] == 0:
        return
    for p in range(2):
        if original_top[env]-tips[env, p][2] < depth_min-wp.float64(1.e-12):
            return
    midpoint = (tips[env, 0]+tips[env, 1])/wp.float64(2.)
    d = tips[env, 0]-tips[env, 1]
    span = wp.sqrt(d[0]*d[0]+d[1]*d[1])/wp.float64(2.)+wp.float64(.020)
    for p in range(2):
        any_contact = int(0)
        in_span = int(0)
        for i in range(top.shape[0]):
            if contacts[env, p, i] != 0:
                any_contact = 1
                offset = xpos[env, bodies[top[i]]]-midpoint
                h = half[env, top[i]]
                if wp.sqrt(offset[0]*offset[0]+offset[1]*offset[1]) <= span+wp.sqrt(h[0]*h[0]+h[1]*h[1]):
                    in_span = 1
        if any_contact == 0 or in_span == 0:
            return
    along = wp.vec2d(tips[env, 1][0]-tips[env, 0][0], tips[env, 1][1]-tips[env, 0][1])
    length = wp.max(wp.sqrt(along[0]*along[0]+along[1]*along[1]), wp.float64(1.e-12))
    normal = wp.vec2d(-along[1]/length, along[0]/length)
    found = int(0)
    for i in range(top.shape[0]):
        body = bodies[top[i]]
        if xpos[env, body][2] > midpoint[2]:
            continue
        r = xmat[env, body]
        h = half[env, top[i]]
        hit = int(0)
        for s in range(BAND_SAMPLES):
            # numpy.linspace: index * step + start, with the endpoint set exactly.
            fs = wp.float64(s)*(wp.float64(1.)/wp.float64(BAND_SAMPLES-1))
            if s == BAND_SAMPLES-1:
                fs = wp.float64(1.)
            for q in range(BAND_ACROSS):
                fq = wp.float64(q)*(wp.float64(1.)/wp.float64(BAND_ACROSS-1))+wp.float64(-.5)
                if q == BAND_ACROSS-1:
                    fq = wp.float64(.5)
                x = tips[env, 0][0]+fs*along[0]+fq*tip_width*normal[0]
                y = tips[env, 0][1]+fs*along[1]+fq*tip_width*normal[1]
                rel = wp.vec3d(x, y, midpoint[2])-xpos[env, body]
                local = wp.transpose(r) @ rel
                if wp.abs(local[0]) <= h[0]+wp.float64(1.e-9) and wp.abs(local[1]) <= h[1]+wp.float64(1.e-9):
                    hit = 1
        if hit != 0:
            covered[env, i] = 1
            found = 1
    eligible[env] = found


@wp.kernel
def pinch_bind(
    bind: wp.array(dtype=int), top: wp.array(dtype=int), bodies: wp.array(dtype=int),
    connects: wp.array(dtype=int), covered: wp.array2d(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3d), xmat: wp.array2d(dtype=wp.mat33d), point: wp.array(dtype=wp.vec3d),
    eq_data: wp.array3d(dtype=wp.float64), active: wp.array2d(dtype=bool),
):
    """PinchAttachment.bind_segments: pin the pinch point of every covered top segment, no teleport."""
    env = wp.tid()
    if bind[env] == 0:
        return
    for i in range(top.shape[0]):
        if covered[env, i] == 0:
            continue
        body = bodies[top[i]]
        eq = connects[i]
        local = wp.transpose(xmat[env, body]) @ (point[env]-xpos[env, body])
        for c in range(3):
            eq_data[env, eq, c] = wp.float64(0.)
            eq_data[env, eq, c+3] = local[c]
        active[env, eq] = True


@wp.kernel
def pinch_release(release: wp.array(dtype=int), connects: wp.array(dtype=int), active: wp.array2d(dtype=bool)):
    env = wp.tid()
    if release[env] == 0:
        return
    for i in range(connects.shape[0]):
        active[env, connects[i]] = False


@wp.kernel
def pinch_reaction(
    nefc: wp.array(dtype=int), equality_type: int, types: wp.array2d(dtype=int), ids: wp.array2d(dtype=int),
    efc_force: wp.array2d(dtype=wp.float64), is_connect: wp.array(dtype=int), active: wp.array2d(dtype=bool),
    out: wp.array2d(dtype=wp.float64),
):
    """PinchAttachment.force_vector: sum of the active point-connection rows, in row order;
    out[env] = [fx, fy, fz, rows]."""
    env = wp.tid()
    f = wp.vec3d(wp.float64(0.))
    count = int(0)
    for i in range(nefc[env]):
        if types[env, i] == equality_type:
            eq = ids[env, i]
            if is_connect[eq] != 0 and active[env, eq]:
                f[count % 3] += efc_force[env, i]
                count += 1
    for c in range(3):
        out[env, c] = f[c]
    out[env, 3] = wp.float64(count)


class LoadFilter:
    """Host float64 copy of pinch.PinchAttachment.check_load's 5 ms low-pass and release test."""
    def __init__(self, nworld, tau):
        self.tau, self.load = tau, np.zeros(nworld)

    def reset(self, mask):
        self.load[mask] = 0.

    def update(self, dt, force_norm, limit, active):
        self.load = np.where(active, self.load + dt/(self.tau + dt)*(force_norm - self.load), self.load)
        return active & (self.load > limit)
