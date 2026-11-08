"""Double-precision Warp operators for v1.2 compaction and top-pinch effects.

All arrays have an environment dimension. CPU adapters and the future MJWarp
adapter share these kernels; no floating-point atomics are used.
"""
import warp as wp

from warp_mechanics import norm3

wp.set_module_options({"enable_backward": False, "enable_fp_fma": False})


@wp.func
def quat_multiply(a: wp.vec4d, b: wp.vec4d):
    return wp.vec4d(a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
                    a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                    a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])


@wp.func
def quat_rotate(q: wp.vec4d, v: wp.vec3d):
    xyz = wp.vec3d(q[1], q[2], q[3])
    first = wp.cross(xyz, v)
    return v + wp.float64(2.)*(q[0]*first + wp.cross(xyz, first))


@wp.func
def write_pose(eq: int, qa: wp.vec4d, qb: wp.vec4d, delta: wp.vec3d,
               env: int, eq_data: wp.array3d(dtype=wp.float64)):
    inv = wp.vec4d(qa[0], -qa[1], -qa[2], -qa[3])
    rel = quat_multiply(inv, qb)
    pos = quat_rotate(inv, delta)
    for c in range(3):
        eq_data[env, eq, c+3] = pos[c]
    for c in range(4):
        eq_data[env, eq, c+6] = rel[c]


@wp.kernel
def compaction(
    bodies: wp.array(dtype=int), obj_geom: wp.array(dtype=bool),
    welds: wp.array(dtype=int), node_a: wp.array(dtype=int), node_b: wp.array(dtype=int),
    qadr: wp.array(dtype=int), external: wp.array3d(dtype=wp.float64),
    contact_geom: wp.array3d(dtype=int), contact_frame: wp.array3d(dtype=wp.float64),
    contact_address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=wp.float64),
    rho: wp.array2d(dtype=wp.float64), qpos: wp.array2d(dtype=wp.float64),
    dt: wp.float64, tau: wp.float64, on_threshold: wp.float64,
    release_threshold: wp.float64, load_threshold: wp.float64,
    filtered: wp.array(dtype=wp.float64), raw: wp.array(dtype=wp.float64),
    active: wp.array2d(dtype=bool), eq_data: wp.array3d(dtype=wp.float64),
):
    env = wp.tid()
    total = wp.float64(0.)
    resultant = wp.vec3d(wp.float64(0.))
    for i in range(bodies.shape[0]):
        body = bodies[i]
        f = wp.vec3d(external[env, body, 0], external[env, body, 1], external[env, body, 2])
        total += norm3(f)
        resultant += f
    contact_total = wp.float64(0.)
    contact_resultant = wp.vec3d(wp.float64(0.))
    for i in range(contact_address.shape[1]):
        g1 = contact_geom[env, i, 0]
        g2 = contact_geom[env, i, 1]
        address = contact_address[env, i]
        if obj_geom[g1] != obj_geom[g2] and address >= 0:
            fn = wp.max(efc_force[env, address], wp.float64(0.))
            signed = -fn
            if obj_geom[g2]:
                signed = fn
            normal = wp.vec3d(contact_frame[env, i, 0], contact_frame[env, i, 1], contact_frame[env, i, 2])
            contact_total += fn
            contact_resultant += signed*normal
    total += contact_total
    resultant += contact_resultant
    load = wp.max(wp.float64(0.), wp.float64(.5)*(total-norm3(resultant)))
    raw[env] = load
    filtered[env] += dt/(tau+dt)*(load-filtered[env])
    for i in range(welds.shape[0]):
        eq = welds[i]
        was_on = active[env, eq]
        threshold = on_threshold
        if was_on:
            threshold = release_threshold
        want = rho[env, i] >= threshold and filtered[env] > load_threshold
        if want and not was_on:
            aa = qadr[node_a[i]]
            ab = qadr[node_b[i]]
            qa = wp.vec4d(qpos[env, aa+3], qpos[env, aa+4], qpos[env, aa+5], qpos[env, aa+6])
            qb = wp.vec4d(qpos[env, ab+3], qpos[env, ab+4], qpos[env, ab+5], qpos[env, ab+6])
            delta = wp.vec3d(qpos[env, ab]-qpos[env, aa], qpos[env, ab+1]-qpos[env, aa+1],
                             qpos[env, ab+2]-qpos[env, aa+2])
            write_pose(eq, qa, qb, delta, env, eq_data)
        active[env, eq] = want


@wp.kernel
def pinch_candidate(
    top: wp.array(dtype=int), bodies: wp.array(dtype=int), half: wp.array(dtype=wp.vec3d),
    geom_body: wp.array(dtype=int), pad_geom: wp.array2d(dtype=bool),
    contact_geom: wp.array3d(dtype=int), contact_frame: wp.array3d(dtype=wp.float64),
    contact_address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=wp.float64),
    xpos: wp.array2d(dtype=wp.vec3d), xmat: wp.array2d(dtype=wp.mat33d),
    tips: wp.array2d(dtype=wp.vec3d), original_top: wp.float64,
    active: int, broken: bool, depth_min: wp.float64,
    contacts: wp.array3d(dtype=int), selected: wp.array(dtype=int),
):
    env = wp.tid()
    selected[env] = -1
    for pad in range(2):
        for i in range(top.shape[0]):
            contacts[env, pad, i] = 0
    for c in range(contact_address.shape[1]):
        address = contact_address[env, c]
        if address < 0:
            continue
        if efc_force[env, address] <= wp.float64(.05):
            continue
        g1 = contact_geom[env, c, 0]
        g2 = contact_geom[env, c, 1]
        for pad in range(2):
            other = int(-1)
            sign = wp.float64(1.)
            if pad_geom[pad, g1]:
                other = g2
                sign = wp.float64(-1.)
            elif pad_geom[pad, g2]:
                other = g1
            if other < 0:
                continue
            body = geom_body[other]
            normal = sign*wp.vec3d(contact_frame[env, c, 0], contact_frame[env, c, 1], contact_frame[env, c, 2])
            for i in range(top.shape[0]):
                if bodies[top[i]] == body:
                    orientation = xmat[env, body]
                    z = wp.vec3d(orientation[0, 2], orientation[1, 2], orientation[2, 2])
                    if wp.dot(normal, z) > wp.float64(.5):
                        contacts[env, pad, i] = 1
    if active >= 0 or broken:
        return
    for p in range(2):
        if original_top-tips[env, p][2] < depth_min-wp.float64(1.e-12):
            return
    midpoint = (tips[env, 0]+tips[env, 1])/wp.float64(2.)
    tip_delta = tips[env, 0]-tips[env, 1]
    span = wp.sqrt(tip_delta[0]*tip_delta[0]+tip_delta[1]*tip_delta[1])/wp.float64(2.)+wp.float64(.020)
    for p in range(2):
        in_span = bool(False)
        for i in range(top.shape[0]):
            if contacts[env, p, i] != 0:
                body = bodies[top[i]]
                offset = xpos[env, body]-midpoint
                h = half[top[i]]
                distance = wp.sqrt(offset[0]*offset[0]+offset[1]*offset[1])
                radius = wp.sqrt(h[0]*h[0]+h[1]*h[1])
                if distance <= span+radius:
                    in_span = True
        if not in_span:
            return
    nearest = wp.float64(1.e100)
    for i in range(top.shape[0]):
        body = bodies[top[i]]
        if xpos[env, body][2] <= midpoint[2]:
            distance = norm3(xpos[env, body]-midpoint)
            if distance < nearest or (distance == nearest and top[i] < selected[env]):
                nearest = distance
                selected[env] = top[i]


@wp.kernel
def bind_pose(body: int, eq: int, mocap: int,
              xpos: wp.array2d(dtype=wp.vec3d), xquat: wp.array2d(dtype=wp.vec4d),
              mocap_pos: wp.array2d(dtype=wp.vec3d), mocap_quat: wp.array2d(dtype=wp.vec4d),
              active: wp.array2d(dtype=bool), eq_data: wp.array3d(dtype=wp.float64)):
    env = wp.tid()
    write_pose(eq, mocap_quat[env, mocap], xquat[env, body],
               xpos[env, body]-mocap_pos[env, mocap], env, eq_data)
    active[env, eq] = True


@wp.kernel
def deactivate(eq: int, active: wp.array2d(dtype=bool)):
    env = wp.tid()
    active[env, eq] = False


@wp.kernel
def reaction(eq: int, equality_type: int, types: wp.array2d(dtype=int),
             ids: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=wp.float64),
             force: wp.array(dtype=wp.vec3d), counts: wp.array(dtype=int)):
    env = wp.tid()
    f = wp.vec3d(wp.float64(0.))
    count = int(0)
    for i in range(types.shape[1]):
        if types[env, i] == equality_type and ids[env, i] == eq:
            if count < 3:
                f[count] = efc_force[env, i]
            count += 1
    force[env] = f
    counts[env] = count


@wp.kernel
def overload(dt: wp.float64, grip: wp.float64, coefficient: wp.float64,
             duration: wp.float64, force: wp.vec3d, previous: wp.float64, peak: wp.float64,
             values: wp.array2d(dtype=wp.float64), detached: wp.array(dtype=bool)):
    env = wp.tid()
    magnitude = norm3(force)
    limit = coefficient*grip
    elapsed = wp.float64(0.)
    if magnitude > limit:
        elapsed = previous+dt
    values[env, 0] = elapsed
    values[env, 1] = wp.max(peak, magnitude)
    values[env, 2] = magnitude
    values[env, 3] = limit
    detached[env] = elapsed >= duration-wp.float64(1.e-12)


@wp.kernel
def spring_force(gap: wp.float64, reference: wp.float64, armed: bool,
                 gather: wp.float64, stiffness: wp.float64, output: wp.array(dtype=wp.float64)):
    env = wp.tid()
    force = wp.float64(0.)
    if armed:
        force = stiffness*wp.max(reference-gather-gap, wp.float64(0.))
    output[env] = force
