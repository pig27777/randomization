"""Device-only sampling, engagement and elastic return mapping.

Units are SI. Force buffers are COM-centered, world-frame force then torque.
No rope dashpot is used: the slip solve brackets the capacity-dependent residual.
"""
import warp as wp
from mujoco_warp._src.support import contact_force_fn

IDLE = 0
ENGAGED = 1
SLIP = 2
DETACHMENT = 3
RUPTURE = 4


@wp.struct
class Parameters:
    size: wp.vec3
    mass: float
    weights: wp.vec3
    com: wp.vec3
    bend: float
    twist: float
    damping: float
    compression: float
    quadratic: float
    slack: float
    slack_y: float
    budget: wp.vec4
    lift_slack: float
    rope_k: float
    mu_s: float
    mu_d: float
    table_mu: float
    strength: float
    gather: float
    escape: float
    threshold: float
    efficiency: float
    slip_damping: float


@wp.struct
class State:
    mode: int
    gathered: float
    captured: float
    slip: float
    closed: int
    reason: int
    rng: wp.uint32
    rest: wp.vec3
    tension: wp.vec3
    anchor: wp.vec3
    probability: float


@wp.kernel
def sample(seed: int, episode: int, rigid: int, prototype: int,
           bounds: wp.array(dtype=wp.vec2), params: wp.array(dtype=Parameters),
           states: wp.array(dtype=State)):
    e = wp.tid()
    rng = wp.rand_init(seed, e + episode * params.shape[0])
    p = Parameters()
    p.size = wp.vec3(wp.randf(rng, bounds[0][0], bounds[0][1]),
                     wp.randf(rng, bounds[1][0], bounds[1][1]),
                     wp.randf(rng, bounds[2][0], bounds[2][1]))
    p.mass = wp.randf(rng, bounds[3][0], bounds[3][1])
    a = wp.randf(rng, bounds[9][0], bounds[9][1])
    b = wp.randf(rng, bounds[9][0], bounds[9][1])
    c = wp.randf(rng, bounds[9][0], bounds[9][1])
    p.weights = wp.vec3(a,b,c)/(a+b+c)
    p.com = wp.vec3(wp.randf(rng,bounds[10][0],bounds[10][1])*p.size[0],
                    wp.randf(rng,bounds[10][0],bounds[10][1])*p.size[1],0.0)
    proto = prototype
    if proto < 0:
        proto = 2 + e % 4
        if rigid == 1:
            proto = e % 2
    p.slack = 0.01
    if proto % 2 == 1:
        p.slack = wp.randf(rng,bounds[4][0],bounds[4][1])
    p.slack_y = p.slack*wp.randf(rng,bounds[17][0],bounds[17][1])
    # Target cantilever tip deflection under an end force equal to its weight.
    sag = wp.randf(rng,bounds[11][0],bounds[11][1])
    if proto >= 4:
        sag = wp.randf(rng,bounds[12][0],bounds[12][1])
    arm = p.size[0]/3.0
    p.bend = p.mass*9.81*(5.0*arm*arm)/sag
    p.twist = p.bend*wp.randf(rng,bounds[13][0],bounds[13][1])
    p.damping = wp.randf(rng,bounds[14][0],bounds[14][1])
    target_depth = wp.randf(rng,bounds[15][0],bounds[15][1])
    if proto >= 4:
        target_depth = wp.randf(rng,bounds[16][0],bounds[16][1])
    # F(20N target depth) = 20, half linear and half quadratic.
    p.compression = 10.0/target_depth
    p.quadratic = 10.0/(target_depth*target_depth)
    p.lift_slack = p.slack*p.size[2]
    p.rope_k = wp.randf(rng,bounds[5][0],bounds[5][1])
    p.mu_s = wp.randf(rng,bounds[6][0],bounds[6][1])
    p.mu_d = p.mu_s*.8
    p.table_mu = wp.randf(rng,bounds[7][0],bounds[7][1])
    p.strength = wp.randf(rng,bounds[8][0],bounds[8][1])
    # Four non-overlapping regional allocations sum to the surplus perimeter.
    material = (p.slack*p.size[0]+p.slack_y*p.size[1])*2.0
    p.budget = material*wp.vec4(.05,.30,.25,.40)
    if rigid == 1 and proto % 2 == 0:
        p.budget = material*wp.vec4(0.0,.35,.25,.40)
    p.gather = .9
    p.escape = .2
    p.threshold = .001
    p.efficiency = .9
    p.slip_damping = 2.0
    params[e] = p
    s = State()
    s.rng = rng
    states[e] = s


@wp.func
def box_inertia(m: float, size: wp.vec3):
    return m/12.0*wp.vec3(size[1]*size[1]+size[2]*size[2],
                         size[0]*size[0]+size[2]*size[2],
                         size[0]*size[0]+size[1]*size[1])


@wp.kernel
def configure(rigid: int, table: int, ids: wp.array2d(dtype=int), params: wp.array(dtype=Parameters),
              body_pos: wp.array2d(dtype=wp.vec3), body_ipos: wp.array2d(dtype=wp.vec3),
              body_mass: wp.array2d(dtype=float), body_inertia: wp.array2d(dtype=wp.vec3),
              geom_size: wp.array2d(dtype=wp.vec3), geom_friction: wp.array2d(dtype=wp.vec3),
              joint_pos: wp.array2d(dtype=wp.vec3), stiffness: wp.array2d(dtype=float), damping: wp.array2d(dtype=float),
              qpos0: wp.array2d(dtype=float), qpos: wp.array2d(dtype=float)):
    e = wp.tid()
    p = params[e]
    n = 3
    if rigid == 1:
        n = 1
    cell = wp.vec3(p.size[0]/float(n),p.size[1],p.size[2])
    geom_friction[e,table] = wp.vec3(p.table_mu,.005,.0001)
    for i in range(n):
        b = ids[i,0]
        g = ids[i,1]
        mass = p.mass*p.weights[i]
        if rigid == 1:
            mass = p.mass
        body_mass[e,b] = mass*.9
        body_inertia[e,b] = box_inertia(mass*.9,wp.vec3(cell[0],cell[1],cell[2]*.8))
        shift = p.com
        if rigid == 0:
            shift[0] -= cell[0]*(p.weights[2]-p.weights[0])
        body_ipos[e,b] = shift
        body_ipos[e,ids[i,2]] = shift
        geom_size[e,g] = wp.vec3(cell[0]*.49,cell[1]*.5,cell[2]*.4)
        geom_friction[e,g] = wp.vec3(p.table_mu,.005,.0001)
        if i > 0:
            body_pos[e,b] = wp.vec3(cell[0],0.0,0.0)
            for j in range(3):
                joint = ids[i,4]+j
                dof = ids[i,5]+j
                k = p.bend
                if j == 0:
                    k = p.twist
                joint_pos[e,joint] = wp.vec3(-cell[0]*.5,0.0,0.0)
                stiffness[e,joint] = k
                damping[e,dof] = 2.0*p.damping*wp.sqrt(k*mass*cell[0]*cell[0]/12.0)
        cap = ids[i,2]
        cg = ids[i,3]
        body_pos[e,cap] = wp.vec3(0.0,0.0,cell[2]*.5)
        body_mass[e,cap] = mass*.1
        body_inertia[e,cap] = box_inertia(mass*.1,wp.vec3(cell[0],cell[1],cell[2]*.2))
        geom_size[e,cg] = wp.vec3(cell[0]*.49,cell[1]*.5,cell[2]*.1)
        geom_friction[e,cg] = wp.vec3(p.table_mu,.005,.0001)
        if rigid == 0:
            stiffness[e,ids[i,6]] = p.compression
            damping[e,ids[i,7]] = 2.0*p.damping*wp.sqrt(p.compression*mass*.1)
    root = ids[0,8]
    qpos0[e,root] = -p.size[0]/3.0
    if rigid == 1:
        qpos0[e,root] = 0.0
    qpos0[e,root+2] = p.size[2]*.4+.001
    for j in range(qpos.shape[1]):
        qpos[e,j] = qpos0[e,j]


@wp.kernel
def contacts(cone: int, nacon: wp.array(dtype=int), worldid: wp.array(dtype=int),
             geom: wp.array(dtype=wp.vec2i), frame: wp.array(dtype=wp.mat33),
             friction: wp.array(dtype=wp.types.vector(length=5,dtype=float)),
             dim: wp.array(dtype=int), address: wp.array2d(dtype=int), adhesion: wp.array(dtype=float),
             efc: wp.array2d(dtype=float), njmax: int, left: int, right: int,
             normal: wp.array2d(dtype=float)):
    c = wp.tid()
    if c < nacon[0]:
        g = geom[c]
        side = -1
        if g[0] == left or g[1] == left:
            side = 0
        if g[0] == right or g[1] == right:
            side = 1
        if side >= 0:
            e = worldid[c]
            f = contact_force_fn(cone,frame,friction,dim,address,adhesion,efc,njmax,nacon,e,c,False)
            wp.atomic_add(normal,e,side,wp.max(f[0],0.0))


@wp.func
def anchor_position(e: int, i: int, rigid: int, ids: wp.array2d(dtype=int),
                    p: Parameters, xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33)):
    b = ids[i,0]
    local = wp.vec3(0.0,0.0,0.0)
    if rigid == 1:
        b = ids[0,0]
        local = wp.vec3(float(i-1)*p.size[0]/3.0,0.0,0.0)
    return xpos[e,b]+xmat[e,b]*local


@wp.func
def tensions(lengths: wp.vec3, rest: wp.vec3, slip: float, k: float):
    return k*wp.vec3(wp.max(lengths[0]-rest[0]-slip,0.0),
                     wp.max(lengths[1]-rest[1]-slip,0.0),
                     wp.max(lengths[2]-rest[2]-slip,0.0))


@wp.kernel
def dynamics(dt: float, rigid: int, level: int, cart: int, region: int,
             close: int, release: int, explicit_bind: int, gather_speed: float,
             params: wp.array(dtype=Parameters), states: wp.array(dtype=State),
             ids: wp.array2d(dtype=int), normal: wp.array2d(dtype=float), gap: wp.array(dtype=float),
             xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
             xipos: wp.array2d(dtype=wp.vec3), qpos: wp.array2d(dtype=float),
             qfrc: wp.array2d(dtype=float), forces: wp.array2d(dtype=wp.spatial_vector)):
    e = wp.tid()
    p = params[e]
    s = states[e]
    if rigid == 0:
        for i in range(3):
            compression = wp.max(-qpos[e,ids[i,9]],0.0)
            qfrc[e,ids[i,7]] += p.quadratic*compression*compression
    if level == 0:
        return
    grip = xpos[e,cart] + xmat[e,cart]*s.anchor
    lengths = wp.vec3(0.0)
    for i in range(3):
        lengths[i] = wp.length(anchor_position(e,i,rigid,ids,p,xpos,xmat)-grip)
    if release == 1:
        s.mode = 0
        s.gathered = 0.0
        s.captured = 0.0
        s.slip = 0.0
    if level == 1 and explicit_bind == 1 and s.mode == 0:
        s.mode = 1
        s.captured = .1
        s.rest = lengths+wp.vec3(p.lift_slack)
    if level == 2:
        budget = p.budget[region]
        touching = normal[e,0] > .05 and normal[e,1] > .05
        access = lengths[1] < p.size[0]+p.size[2]
        speed = wp.max(gather_speed,0.0)
        efficiency = 0.0
        if touching and access:
            efficiency = p.efficiency
        rate = p.gather*efficiency*speed*(budget-s.gathered)/(budget+1.e-8)
        rate -= p.escape*wp.max(-gather_speed,0.0)*s.gathered/(budget+1.e-8)
        s.gathered = wp.clamp(s.gathered+dt*rate,0.0,budget)
        if close == 1 and s.closed == 0 and s.mode == 0:
            s.reason = 0
            s.probability = 0.0
            if budget <= p.threshold:
                s.reason = 1
            elif not touching or not access or gap[e] > 2.0*s.gathered+.004:
                s.reason = 3
            elif s.gathered <= p.threshold:
                s.reason = 2
            else:
                s.probability = p.efficiency*(1.0-wp.exp(-500.0*(s.gathered-p.threshold)))
                draw = wp.randf(s.rng)
                if draw < s.probability:
                    s.mode = 1
                    s.captured = wp.min(s.gathered,.02)
                    s.slip = 0.0
                    s.rest = lengths+wp.vec3(p.lift_slack)
                else:
                    s.reason = 4
        s.closed = close
    s.tension = wp.vec3(0.0)
    if s.mode == 1 or s.mode == 2:
        trial = tensions(lengths,s.rest,s.slip,p.rope_k)
        drive = trial[0]+trial[1]+trial[2]
        if level == 2:
            n = 2.0*wp.min(normal[e,0],normal[e,1])
            phi = wp.clamp((s.captured-s.slip)/.005,0.0,1.0)
            capacity = p.mu_s*n*p.efficiency*phi
            if drive > capacity:
                old = s.slip
                lo = old
                hi = s.captured
                # Smallest positive return-map root. At exhaustion detach.
                for j in range(24):
                    mid = (lo+hi)*.5
                    t = tensions(lengths,s.rest,mid,p.rope_k)
                    cap = p.mu_d*n*p.efficiency*wp.clamp((s.captured-mid)/.005,0.0,1.0)
                    residual = t[0]+t[1]+t[2]-cap-p.slip_damping*(mid-old)/dt
                    if residual > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s.slip = hi
                s.mode = 2
            elif s.mode == 2:
                s.mode = 1
            if s.slip >= s.captured-1.e-7:
                s.mode = 3
        s.tension = tensions(lengths,s.rest,s.slip,p.rope_k)
        if wp.max(s.tension[0],wp.max(s.tension[1],s.tension[2])) > p.strength:
            s.mode = 4
        if s.mode >= 3:
            s.tension = wp.vec3(0.0)
        for i in range(3):
            point = anchor_position(e,i,rigid,ids,p,xpos,xmat)
            f = -s.tension[i]*(point-grip)/wp.max(lengths[i],1.e-8)
            b = ids[i,0]
            if rigid == 1:
                b = ids[0,0]
            torque = wp.cross(point-xipos[e,b],f)
            reaction = wp.cross(grip-xipos[e,cart],-f)
            forces[e,b] += wp.spatial_vector(f,torque)
            forces[e,cart] += wp.spatial_vector(-f,reaction)
    states[e] = s
