"""Device-side Yam_B control, measurement, history and randomization."""

import warp as wp

wp.set_module_options({"enable_backward": False})

Vec6 = wp.types.vector(length=6, dtype=wp.float32)
Mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.struct
class Geometry:
    joints: wp.array[int]
    pos: wp.array[int]
    dof: wp.array[int]
    actuators: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    speed: wp.array[float]
    gravity_factor: wp.array[float]
    gravity_bodies: wp.array2d[int]
    gravity: wp.vec3
    site: int
    max_gap: float
    min_drive_gap: float
    motor_Nm_per_N: float


@wp.struct
class Motor:
    kp: float
    kd: float
    limit: float
    response: float
    gain: float
    bias: float
    noise: float
    lowpass: float
    quantum: float


@wp.struct
class Commands:
    requested: wp.array2d[float]
    target: wp.array2d[float]
    velocity: wp.array2d[float]
    desired_velocity: wp.array2d[float]
    feedforward: wp.array2d[float]
    action: wp.array2d[float]


@wp.struct
class Actuators:
    tuning: wp.array2d[Motor]
    command: wp.array2d[float]
    applied: wp.array2d[float]
    feedback: wp.array2d[float]


@wp.func
def coordinates(values: wp.array2d[float], index: wp.array[int], env: int, channel: int):
    result = values[env, index[channel]]
    if channel == 6:
        result += values[env, index[7]]
    return result


@wp.func
def safe(value: float, fallback: float):
    if wp.isfinite(value):
        return value
    return fallback


@wp.func
def damped_solve(jac: Mat66, error: Vec6, damping: float):
    # Solve J J^T + lambda^2 I via Cholesky. Damping keeps every pivot
    # positive at singular poses, including an exactly straight wrist.
    gram = jac * wp.transpose(jac)
    lower = Mat66()
    rhs = Vec6()
    solution = Vec6()
    for i in range(6):
        for j in range(i + 1):
            value = gram[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            if i == j:
                lower[i, j] = wp.sqrt(value + damping * damping)
            else:
                lower[i, j] = value / lower[j, j]
        value = error[i]
        for j in range(i):
            value -= lower[i, j] * rhs[j]
        rhs[i] = value / lower[i, i]
    for ii in range(6):
        i = 5 - ii
        value = rhs[i]
        for jj in range(ii):
            j = 5 - jj
            value -= lower[j, i] * solution[j]
        solution[i] = value / lower[i, i]
    return wp.transpose(jac) * solution


@wp.kernel
def cartesian_targets(robot: Geometry, cmds: Commands, actions: wp.array2d[float], period: float,
                      scales: wp.vec2, workspace_lo: wp.vec3, workspace_hi: wp.vec3, damping: float,
                      qpos: wp.array2d[float], sites: wp.array2d[wp.vec3],
                      anchor: wp.array2d[wp.vec3], axis: wp.array2d[wp.vec3]):
    env = wp.tid()
    point = sites[env, robot.site]
    error = Vec6()
    jac = Mat66()
    for i in range(7):
        fallback = float(0.0)
        if i == 6:
            fallback = 2.0 * cmds.target[env, i] / robot.max_gap - 1.0
        cmds.action[env, i] = wp.clamp(safe(actions[env, i], fallback), -1.0, 1.0)
        cmds.velocity[env, i] = 0.0
        cmds.feedforward[env, i] = 0.0
    for i in range(3):
        target = wp.clamp(point[i] + scales[0] * cmds.action[env, i], workspace_lo[i], workspace_hi[i])
        error[i] = 4.0 * (target - point[i])
        error[i + 3] = scales[1] * cmds.action[env, i + 3]
    for j in range(6):
        angular = axis[env, robot.joints[j]]
        linear = wp.cross(angular, point - anchor[env, robot.joints[j]])
        for i in range(3):
            jac[i, j] = 4.0 * linear[i]
            jac[i + 3, j] = angular[i]
    delta = damped_solve(jac, error, damping)
    for j in range(6):
        limit = robot.speed[j] * period
        q = qpos[env, robot.pos[j]]
        cmds.requested[env, j] = wp.clamp(q + wp.clamp(delta[j], -limit, limit), robot.lower[j], robot.upper[j])
    cmds.requested[env, 6] = 0.5 * robot.max_gap * (cmds.action[env, 6] + 1.0)


@wp.kernel
def joint_targets(robot: Geometry, cmds: Commands, position: wp.array2d[float], velocity: wp.array2d[float],
                  feedforward: wp.array2d[float], policy_period: float):
    env, j = wp.tid()
    target = wp.clamp(safe(position[env, j], cmds.target[env, j]), robot.lower[j], robot.upper[j])
    cmds.requested[env, j] = target
    cmds.velocity[env, j] = wp.clamp(safe(velocity[env, j], 0.0), -robot.speed[j], robot.speed[j])
    cmds.feedforward[env, j] = safe(feedforward[env, j], 0.0)
    cmds.action[env, j] = wp.clamp((target - cmds.target[env, j]) / (robot.speed[j] * policy_period), -1.0, 1.0)
    if j == 6:
        cmds.action[env, j] = 2.0 * target / robot.max_gap - 1.0


@wp.kernel
def slew_targets(robot: Geometry, cmds: Commands, period: float):
    env, j = wp.tid()
    before = cmds.target[env, j]
    change = robot.speed[j] * period
    target = before + wp.clamp(cmds.requested[env, j] - before, -change, change)
    cmds.target[env, j] = target
    velocity = cmds.velocity[env, j]
    if (target <= robot.lower[j] and velocity < 0.0) or (target >= robot.upper[j] and velocity > 0.0):
        velocity = 0.0
    cmds.desired_velocity[env, j] = velocity


@wp.kernel
def motor_control(robot: Geometry, cmds: Commands, motors: Actuators, period: float,
                  qpos: wp.array2d[float], qvel: wp.array2d[float], ctrl: wp.array2d[float],
                  mass: wp.array2d[float], com: wp.array2d[wp.vec3], anchor: wp.array2d[wp.vec3],
                  axis: wp.array2d[wp.vec3]):
    env, j = wp.tid()
    tuning = motors.tuning[env, j]
    q = coordinates(qpos, robot.pos, env, j)
    dq = coordinates(qvel, robot.dof, env, j)
    command = cmds.target[env, j]
    gravity = float(0.0)
    if j < 6:
        joint = robot.joints[j]
        for b in range(robot.gravity_bodies.shape[1]):
            body = robot.gravity_bodies[j, b]
            if body >= 0:
                lever = com[env, body] - anchor[env, joint]
                weight = mass[env % mass.shape[0], body] * robot.gravity
                gravity -= wp.dot(axis[env, joint], wp.cross(lever, weight))
        gravity *= robot.gravity_factor[j]
    else:
        # Closed command requests a small preload behind the mating pad plane.
        command += robot.min_drive_gap * (1.0 - command / robot.max_gap)
    tau = tuning.kp * (command - q) + tuning.kd * (cmds.desired_velocity[env, j] - dq)
    tau += gravity + cmds.feedforward[env, j]
    motors.command[env, j] = tau
    limited = wp.clamp(tau, -tuning.limit, tuning.limit)
    if tuning.response > 0.0:
        prev = motors.applied[env, j]
        limited = prev + (1.0 - wp.exp(-period / tuning.response)) * (limited - prev)
    motors.applied[env, j] = limited
    ctrl[env, robot.actuators[j]] = limited


@wp.kernel
def read_motors(robot: Geometry, motors: Actuators, period: float, seed: int,
                qpos: wp.array2d[float], qvel: wp.array2d[float], force: wp.array2d[float],
                clock: wp.array[int], time: wp.array[float], sensor: wp.array4d[float],
                sensor_time: wp.array2d[float]):
    env, j = wp.tid()
    tick = clock[env]
    slot = tick % sensor.shape[1]
    effort = force[env, robot.actuators[j]]
    if j == 6:
        effort *= robot.motor_Nm_per_N
    tuning = motors.tuning[env, j]
    if tuning.lowpass > 0.0:
        previous = motors.feedback[env, j]
        effort = previous + (1.0 - wp.exp(-period / tuning.lowpass)) * (effort - previous)
    motors.feedback[env, j] = effort
    rng = wp.rand_init(seed + 1823 * j, env + tick * sensor.shape[0])
    effort = tuning.gain * effort + tuning.bias + tuning.noise * wp.randn(rng)
    if tuning.quantum > 0.0:
        effort = wp.round(effort / tuning.quantum) * tuning.quantum
    q = coordinates(qpos, robot.pos, env, j)
    if j == 6:
        q = wp.clamp(q, 0.0, robot.max_gap)
    sensor[env, slot, 0, j] = q
    sensor[env, slot, 1, j] = coordinates(qvel, robot.dof, env, j)
    sensor[env, slot, 2, j] = effort
    if j == 0:
        sensor_time[env, slot] = time[env]


@wp.kernel
def advance_clock(clock: wp.array[int]):
    env = wp.tid()
    clock[env] += 1


@wp.kernel
def observation(cmds: Commands, clock: wp.array[int], delay: wp.array[int], sensor: wp.array4d[float],
                sensor_time: wp.array2d[float], obs: wp.array2d[float], sample_time: wp.array[float]):
    env = wp.tid()
    tick = wp.max(0, clock[env] - 1 - delay[env])
    slot = tick % sensor.shape[1]
    for k in range(3):
        for j in range(6):
            obs[env, 6 * k + j] = sensor[env, slot, k, j]
        obs[env, 24 + k] = sensor[env, slot, k, 6]
    for j in range(6):
        obs[env, 18 + j] = cmds.target[env, j]
    obs[env, 27] = cmds.target[env, 6]
    sample_time[env] = sensor_time[env, slot]


@wp.kernel
def reset_controls(robot: Geometry, cmds: Commands, motors: Actuators, qpos: wp.array2d[float],
                   initial: wp.array2d[float], mask: wp.array[bool], clock: wp.array[int],
                   sensor: wp.array4d[float], sensor_time: wp.array2d[float], count: wp.array[int], try_count: wp.array[int],
                   try_active: wp.array[bool]):
    env = wp.tid()
    if mask[env]:
        for k in range(initial.shape[1]):
            qpos[env, k] = initial[env % initial.shape[0], k]
        for j in range(7):
            q = coordinates(qpos, robot.pos, env, j)
            cmds.requested[env, j] = q
            cmds.target[env, j] = q
            cmds.velocity[env, j] = 0.0
            cmds.desired_velocity[env, j] = 0.0
            cmds.feedforward[env, j] = 0.0
            cmds.action[env, j] = 0.0
            motors.command[env, j] = 0.0
            motors.applied[env, j] = 0.0
            motors.feedback[env, j] = 0.0
            sensor[env, 0, 0, j] = q
            sensor[env, 0, 1, j] = 0.0
            sensor[env, 0, 2, j] = 0.0
        cmds.action[env, 6] = 2.0 * cmds.target[env, 6] / robot.max_gap - 1.0
        clock[env] = 1
        sensor_time[env, 0] = 0.0
        count[env] = 0
        try_count[env] = 0
        try_active[env] = False


@wp.func
def sample_range(rng: wp.uint32, bounds: wp.vec2):
    return bounds[0] + wp.randf(rng) * (bounds[1] - bounds[0])


@wp.kernel
def randomize_controls(mask: wp.array[bool], seed: int, period: float, nominal: wp.array[wp.vec3],
                       ranges: wp.array[wp.vec2], tuning: wp.array2d[Motor], delay: wp.array[int]):
    env, j = wp.tid()
    if mask[env]:
        rng = wp.rand_init(seed, env * 7 + j)
        limits = nominal[j]
        motor = Motor()
        motor.kp = limits[0] * sample_range(rng, ranges[0])
        motor.kd = limits[1] * sample_range(rng, ranges[1])
        if j < 6:
            motor.kp = wp.min(motor.kp, 500.0)
            motor.kd = wp.min(motor.kd, 5.0)
        motor.limit = limits[2] * sample_range(rng, ranges[2])
        if j == 6:
            motor.kp = limits[0] * sample_range(rng, ranges[3])
            motor.kd = limits[1] * sample_range(rng, ranges[3])
            motor.limit = limits[2] * sample_range(rng, ranges[4])
        motor.gain = sample_range(rng, ranges[5])
        motor.bias = sample_range(rng, ranges[6])
        motor.noise = sample_range(rng, ranges[7])
        motor.response = sample_range(rng, ranges[8])
        motor.lowpass = sample_range(rng, ranges[9])
        motor.quantum = sample_range(rng, ranges[10])
        tuning[env, j] = motor
        if j == 0:
            delay[env] = int(wp.round(sample_range(rng, ranges[11]) / period))


@wp.kernel
def randomize_links(mask: wp.array[bool], seed: int, addresses: wp.array[int], bounds: wp.vec2,
                    mass_0: wp.array[float], inertia_0: wp.array[wp.vec3],
                    mass: wp.array2d[float], inertia: wp.array2d[wp.vec3]):
    env, index = wp.tid()
    if mask[env]:
        body = addresses[index]
        rng = wp.rand_init(seed, env * addresses.shape[0] + index)
        scale = sample_range(rng, bounds)
        mass[env, body] = scale * mass_0[body]
        inertia[env, body] = scale * inertia_0[body]


@wp.kernel
def randomize_dofs(mask: wp.array[bool], seed: int, robot: Geometry, friction_range: wp.vec2,
                   armature_range: wp.vec2, friction_0: wp.array[float], armature_0: wp.array[float],
                   friction: wp.array2d[float], armature: wp.array2d[float]):
    env, j = wp.tid()
    if mask[env]:
        dof = robot.dof[j]
        rng = wp.rand_init(seed, env * 6 + j)
        friction[env, dof] = friction_0[dof] * sample_range(rng, friction_range)
        armature[env, dof] = armature_0[dof] * sample_range(rng, armature_range)


@wp.kernel
def randomize_pads(mask: wp.array[bool], seed: int, pads: wp.array[int], bounds: wp.vec2,
                   nominal: wp.array[wp.vec3], friction: wp.array2d[wp.vec3]):
    env, index = wp.tid()
    if mask[env]:
        geom = pads[index]
        rng = wp.rand_init(seed, env * pads.shape[0] + index)
        friction[env, geom] = nominal[geom] * sample_range(rng, bounds)
