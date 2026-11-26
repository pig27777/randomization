"""Batched Yam_B safety and force feedback on the MJWarp device."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.types import OverflowType, Stage, vec5
from yam_platform.yam_warp import SimYAMInterface
from yam_platform.yam_warp_kernels import (Actuators, Commands, Geometry, Mat66, Motor,
                                           Vec6, coordinates, damped_solve)

from .environment import MotionStop
from .yam_kinematics import descendants

wp.set_module_options({"enable_backward": False})
SOLVER_LIMIT_FLAGS = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)


@wp.func
def overflow_stop_reason(flags: int):
    if (flags & ~wp.static(SOLVER_LIMIT_FLAGS)) != 0:
        return 1
    if flags != 0:
        return 11
    return 0


@wp.func
def flex_endpoint(geom: int, flex: int):
    # Rigid contacts reuse slots without clearing flex metadata in MJWarp 4.0.1.
    return geom < 0 and flex >= 0


@wp.kernel
def apply_pad_film_contact(ncon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
                           flex: wp.array(dtype=wp.vec2i), pads: wp.array(dtype=int),
                           ref: wp.vec2, imp: vec5, solref: wp.array(dtype=wp.vec2),
                           solreffriction: wp.array(dtype=wp.vec2), solimp: wp.array(dtype=vec5)):
    i = wp.tid()
    if i < ncon[0]:
        pair, bag = geom[i], flex[i]
        if (((pair[0] == pads[0] or pair[0] == pads[1]) and flex_endpoint(pair[1], bag[1])) or
                ((pair[1] == pads[0] or pair[1] == pads[1]) and flex_endpoint(pair[0], bag[0]))):
            solref[i] = ref
            solreffriction[i] = ref
            solimp[i] = imp


def install_pad_film_contact(model, pads, profile):
    time_constant, dmin, dmax, width = profile
    reference = wp.vec2(time_constant, 1.)
    impedance = vec5(dmin, dmax, width, .5, 2.)

    def filter_contacts(_model, data):
        c = data.contact
        wp.launch(apply_pad_film_contact, data.naconmax,
                  [data.nacon, c.geom, c.flex, pads, reference, impedance,
                   c.solref, c.solreffriction, c.solimp], device=pads.device)

    model.callback.contactfilter = filter_contacts


@wp.func
def arm_group(groups: wp.array(dtype=int), geom: int):
    if geom >= 0:
        return groups[geom]
    return 0


@wp.func
def is_arm(group: int):
    return group == 1 or group == 2


@wp.func
def film_surface_slot(slots: wp.array(dtype=int), geom: int):
    if geom >= 0:
        return slots[geom]
    return 0


@wp.kernel
def find_bad_contact(
    ncon: wp.array(dtype=int), world: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    flex: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float), groups: wp.array(dtype=int),
    film_slots: wp.array(dtype=int), table: int, depth: float, bad: wp.array(dtype=int),
    static_film_min: wp.array2d(dtype=float),
):
    i = wp.tid()
    if i >= ncon[0]:
        return
    pair = geom[i]
    a, b = arm_group(groups, pair[0]), arm_group(groups, pair[1])
    if (pair[0] == table and b == 2) or (a == 2 and pair[1] == table):
        return
    f = flex[i]
    allowance = depth
    slot = int(0)
    if flex_endpoint(pair[0], f[0]):
        slot = film_surface_slot(film_slots, pair[1])
    if flex_endpoint(pair[1], f[1]):
        slot = film_surface_slot(film_slots, pair[0])
    if slot < 0:
        allowance = .002
    elif slot > 0:
        allowance = .005
        wp.atomic_min(static_film_min, world[i], slot - 1, dist[i])
    if dist[i] < -allowance or (dist[i] < -.0005 and (
            (is_arm(a) and (b == 3 or is_arm(b))) or (is_arm(b) and a == 3))):
        wp.atomic_min(bad, world[i], i)


@wp.kernel
def measure_edges(points: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
                  rest: wp.array(dtype=float), frame_peak: wp.array(dtype=float)):
    env, i = wp.tid()
    edge = edges[i]
    ratio = wp.length(points[env, edge[0]] - points[env, edge[1]]) / rest[i]
    wp.atomic_max(frame_peak, env, ratio)


@wp.kernel
def guard_substep(
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), overflow: wp.array(dtype=int),
    ncon: wp.array(dtype=int), bad: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    flex: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float), groups: wp.array(dtype=int),
    stamp: wp.array(dtype=float), stretch: wp.array(dtype=float), max_stretch: float,
    stop: wp.array(dtype=int), stop_pair: wp.array2d(dtype=int), stop_value: wp.array2d(dtype=float),
):
    env = wp.tid()
    if stop[env] != 0:
        return
    reason = overflow_stop_reason(overflow[env])
    if ncon[0] > dist.shape[0]:
        reason = 1
    for j in range(qpos.shape[1]):
        if not wp.isfinite(qpos[env, j]):
            reason = 2
    for j in range(qvel.shape[1]):
        if not wp.isfinite(qvel[env, j]):
            reason = 2
    i = bad[env]
    if reason == 0 and i < dist.shape[0]:
        pair, f = geom[i], flex[i]
        a, b = arm_group(groups, pair[0]), arm_group(groups, pair[1])
        reason = 3
        if is_arm(a) or is_arm(b):
            reason = 4
        if (is_arm(a) and (b == 3 or is_arm(b))) or (is_arm(b) and a == 3):
            reason = 5
        stop_pair[env, 0] = pair[0]
        stop_pair[env, 1] = pair[1]
        stop_pair[env, 2] = f[0]
        stop_pair[env, 3] = f[1]
        stop_value[env, 0] = dist[i]
        stop_value[env, 1] = stamp[env]
    if reason == 0 and stretch[env] > max_stretch:
        reason = 6
    stop[env] = reason


@wp.kernel
def sum_contact_forces(
    ncon: wp.array(dtype=int), world: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    flex: wp.array(dtype=wp.vec2i), address: wp.array2d(dtype=int), dist: wp.array(dtype=float),
    force: wp.array2d(dtype=float), groups: wp.array(dtype=int), table: int, pads: wp.array(dtype=int),
    payload: int, load: wp.array2d(dtype=float), direct_index: wp.array(dtype=int),
    nonfinite: wp.array(dtype=int),
):
    i = wp.tid()
    if i >= ncon[0]:
        return
    env, row = world[i], address[i, 0]
    if row < 0 or row >= force.shape[1]:
        return
    n = force[env, row]  # PE and the default Yam model use an elliptic friction cone.
    if not wp.isfinite(n):
        wp.atomic_max(nonfinite, env, 1)
        return
    n = wp.max(0., n)
    pair = geom[i]
    a, b = arm_group(groups, pair[0]), arm_group(groups, pair[1])
    if not (is_arm(a) or is_arm(b)) or (pair[0] == table and b == 2) or (a == 2 and pair[1] == table):
        return
    wp.atomic_add(load, env, 0, n)
    f = flex[i]
    for j in range(2):
        if ((pair[0] == pads[j] and (flex_endpoint(pair[1], f[1]) or payload < 0)) or
                (pair[1] == pads[j] and (flex_endpoint(pair[0], f[0]) or payload < 0))):
            wp.atomic_add(load, env, 1 + j, n)
            wp.atomic_min(load, env, 3, dist[i])
    if payload >= 0 and n > 0. and (pair[0] == payload or pair[1] == payload):
        wp.atomic_max(load, env, 4, n)
        wp.atomic_min(direct_index, env, i)


@wp.kernel
def control_after_solve(
    robot: Geometry, cmd: Commands, qpos: wp.array2d(dtype=float), sites: wp.array2d(dtype=wp.vec3),
    anchor: wp.array2d(dtype=wp.vec3), axis: wp.array2d(dtype=wp.vec3), stamp: wp.array(dtype=float),
    damping: float, load: wp.array2d(dtype=float), limits: wp.array2d(dtype=float),
    safe_q: wp.array2d(dtype=float), safe_jac: wp.array(dtype=Mat66), safe_known: wp.array(dtype=int),
    unload_q: wp.array2d(dtype=float),
    probe_event: wp.array2d(dtype=float), jaw_guard: wp.array(dtype=int),
    peaks: wp.array2d(dtype=float), counts: wp.array2d(dtype=int),
    direct_index: wp.array(dtype=int), nonfinite: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
    dist: wp.array(dtype=float), overflow: wp.array(dtype=int), stop: wp.array(dtype=int),
    stop_pair: wp.array2d(dtype=int), stop_value: wp.array2d(dtype=float),
):
    env = wp.tid()
    total, left, right = load[env, 0], load[env, 1], load[env, 2]
    pad_force = left + right
    peak_pad = wp.max(left, right)
    for j in range(3):
        peaks[env, j] = wp.max(peaks[env, j], load[env, j])
    peaks[env, 3] = wp.min(peaks[env, 3], load[env, 3])
    peaks[env, 4] = wp.max(peaks[env, 4], load[env, 4])
    if stop[env] != 0:
        return
    if overflow[env] != 0:
        stop[env] = overflow_stop_reason(overflow[env])
    elif nonfinite[env] != 0:
        stop[env] = 2
    elif load[env, 4] > 0.:
        stop[env] = 9
        i = direct_index[env]
        pair = geom[i]
        stop_pair[env, 0] = pair[0]
        stop_pair[env, 1] = pair[1]
        stop_value[env, 0] = dist[i]
        stop_value[env, 1] = stamp[env]
        stop_value[env, 2] = load[env, 4]
    elif total > limits[env, 0] or (limits[env, 1] > 0. and peak_pad > limits[env, 1]):
        stop[env] = 8
    if stop[env] != 0:
        return

    threshold = limits[env, 2]
    if threshold > 0. and probe_event[env, 0] == 0.:
        if pad_force < .8:
            jac = Mat66()
            for j in range(6):
                safe_q[env, j] = coordinates(qpos, robot.pos, env, j)
                angular = axis[env, robot.joints[j]]
                linear = wp.cross(angular, sites[env, robot.site] - anchor[env, robot.joints[j]])
                for k in range(3):
                    jac[k, j] = 4. * linear[k]
                    jac[k + 3, j] = angular[k]
            safe_jac[env] = jac
            safe_known[env] = 1
        elif pad_force >= threshold:
            if safe_known[env] == 0:
                stop[env] = 10
                return
            probe_event[env, 0] = pad_force
            probe_event[env, 1] = stamp[env]
            probe_event[env, 2] = load[env, 3]
            error = Vec6()
            error[2] = 4. * .003
            delta = damped_solve(safe_jac[env], error, damping)
            for j in range(6):
                goal = wp.clamp(safe_q[env, j] + delta[j], robot.lower[j], robot.upper[j])
                unload_q[env, j] = goal
                cmd.requested[env, j] = goal
                cmd.target[env, j] = goal
                cmd.velocity[env, j] = 0.
                cmd.desired_velocity[env, j] = 0.
                cmd.feedforward[env, j] = 0.
            # Release pad squeeze during the vertical unload. Keeping the old
            # jaw setpoint reintroduces the nominal mating-plane preload.
            gap = wp.min(robot.max_gap, coordinates(qpos, robot.pos, env, 6) + .004)
            cmd.requested[env, 6] = gap
            cmd.target[env, 6] = gap
            cmd.velocity[env, 6] = 0.
            cmd.desired_velocity[env, 6] = 0.
            cmd.feedforward[env, 6] = 0.

    jaw_guard[env] = 0
    if limits[env, 1] > 0.:
        gap = coordinates(qpos, robot.pos, env, 6)
        if peak_pad > 11. or (peak_pad >= 8. and cmd.requested[env, 6] <= gap):
            goal = gap
            counts[env, 0] += 1
            if peak_pad > 11.:
                goal = wp.min(robot.max_gap, gap + .0004)
                counts[env, 1] += 1
            goal = wp.max(goal, cmd.requested[env, 6])
            cmd.target[env, 6] = goal
            cmd.velocity[env, 6] = 0.
            cmd.desired_velocity[env, 6] = 0.
            cmd.feedforward[env, 6] = 0.
            jaw_guard[env] = 1


@wp.kernel
def apply_guarded_motor(
    robot: Geometry, cmd: Commands, motors: Actuators, qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float), mass: wp.array2d(dtype=float),
    com: wp.array2d(dtype=wp.vec3), anchor: wp.array2d(dtype=wp.vec3), axis: wp.array2d(dtype=wp.vec3),
    limits: wp.array2d(dtype=float), probe_event: wp.array2d(dtype=float), jaw_guard: wp.array(dtype=int),
    stop: wp.array(dtype=int),
):
    env, j = wp.tid()
    index = robot.actuators[j]
    if stop[env] != 0 or limits[env, 0] == 0.:
        ctrl[env, index] = 0.
        motors.applied[env, j] = 0.
    elif ((limits[env, 2] > 0. and probe_event[env, 0] > 0.) or (j == 6 and jaw_guard[env] != 0)):
        tuning = motors.tuning[env, j]
        q = coordinates(qpos, robot.pos, env, j)
        dq = coordinates(qvel, robot.dof, env, j)
        tau = tuning.kp * (cmd.target[env, j] - q) - tuning.kd * dq
        if j < 6:
            joint = robot.joints[j]
            gravity = float(0.)
            for k in range(robot.gravity_bodies.shape[1]):
                body = robot.gravity_bodies[j, k]
                if body >= 0:
                    lever = com[env, body] - anchor[env, joint]
                    weight = mass[env % mass.shape[0], body] * robot.gravity
                    gravity -= wp.dot(axis[env, joint], wp.cross(lever, weight))
            tau += gravity * robot.gravity_factor[j]
        elif jaw_guard[env] != 0:
            # Freezing the jaw must also remove any remaining closing motor effort.
            tau = wp.max(0., tau)
        effort = wp.clamp(tau, -tuning.limit, tuning.limit)
        ctrl[env, index] = effort
        motors.applied[env, j] = effort
        motors.command[env, j] = tau


@wp.kernel
def set_effort(tuning: wp.array2d(dtype=Motor), efforts: wp.array(dtype=float), gear: float):
    env = wp.tid()
    motor = tuning[env, 6]
    motor.limit = efforts[env] / gear
    tuning[env, 6] = motor


@wp.kernel
def clear_idle_guard(mask: wp.array(dtype=wp.bool), reason: wp.array(dtype=int),
                     jaw: wp.array(dtype=int), probe: wp.array2d(dtype=float)):
    env = wp.tid()
    if mask[env]:
        reason[env] = 0
        jaw[env] = 0
        for i in range(3):
            probe[env, i] = 0.


@wp.kernel
def clear_probe_event(event: wp.array2d(dtype=float), clear: wp.array(dtype=int)):
    env = wp.tid()
    if clear[env] != 0:
        for i in range(3):
            event[env, i] = 0.


STOP_REASONS = {
    1: "solver_overflow", 2: "nonfinite_physics", 3: "excessive_contact_penetration",
    4: "excessive_contact_penetration", 5: "workcell_collision", 6: "flex_overstretch",
    8: "contact_force_limit", 9: "robot_payload_direct_force", 10: "missing_safe_probe_pose",
    11: "solver_nonconvergence",
}


class GPUDrive:
    name = "mjwarp_cuda"

    def __init__(self, scene, cfg, seed, device, motor_Nm_per_N, limits, *, worlds=1,
                 randomized_payload=False, contact_capacity=None, constraint_capacity=None):
        self.gear = motor_Nm_per_N
        self.flex = scene.is_flex
        self.limits = limits
        contacts, constraints = (1024, 4096) if self.flex else (128, 256)
        contacts = contacts if contact_capacity is None else contact_capacity
        constraints = constraints if constraint_capacity is None else constraint_capacity
        if min(contacts, constraints) < 1:
            raise ValueError("Contact and constraint capacities must be positive")
        # MJWarp allocates a global contact pool, but njmax and its sparse budget are per world.
        capacities = dict(naconmax=contacts * worlds, njmax=constraints)
        if self.flex:
            capacities["njmax_nnz"] = 32768
        sim = self.sim = SimYAMInterface(worlds, config=cfg, seed=seed, device=device,
                                         add_scene=scene.add_scene, compile_model=scene.compile_model,
                                         initialize=scene.initialize, capacities=capacities)
        self.model = m = sim.model
        self.randomized_payload = randomized_payload
        self.payload_body = m.body(scene.payload_body).id
        self.payload_geom = m.geom(scene.payload_geom).id
        self.payload_mass = m.body_mass[self.payload_body]
        if randomized_payload:
            # Only payload dynamics vary. Batch the model fields and their cached inertial terms.
            model_fields = ("body_mass", "body_inertia", "geom_friction", "body_subtreemass",
                            "dof_invweight0", "body_invweight0", "actuator_acc0")
            with wp.ScopedDevice(sim.device):
                for name in model_fields:
                    shared = getattr(sim.gpu_model, name)
                    data = np.repeat(shared.numpy(), worlds, axis=0)
                    setattr(sim.gpu_model, name, wp.array(data, dtype=shared.dtype))
        self.initial = sim.initial.numpy()[0].copy()
        self.data = mujoco.MjData(m)
        d = sim.data
        arm = descendants(m, "arm")
        base = m.body("arm").id
        names = [m.geom(i).name for i in range(m.ngeom)]
        groups = [2 if body == base else 1 if body in arm else
                  3 if names[i].startswith(("table", "bin_", "Yam_A_")) else 0
                  for i, body in enumerate(m.geom_bodyid)]
        self.static_film_surfaces = tuple(name for name in names if self.flex and (
            name in ("table", "table_flex_support") or name.startswith("bin_wall_")))
        # Positive slots index the distance diagnostics. The negative slot is the pad allowance.
        static_slots = {name: i for i, name in enumerate(self.static_film_surfaces, 1)}
        film_slots = [static_slots.get(name, -1 if self.flex and name in ("left_pad", "right_pad") else 0)
                      for name in names]

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=sim.device)

        self.reason = zeros(worlds, int)
        self.load = zeros((worlds, 5))
        self.peaks = zeros((worlds, 5))
        self.edge_peak = zeros(worlds)
        self.static_film_min = wp.full((worlds, len(static_slots)), np.inf, dtype=float, device=sim.device)
        self.stop_pair = zeros((worlds, 4), int)
        self.stop_value = zeros((worlds, 3))
        self.probe_event = zeros((worlds, 3))
        self.unload_q = zeros((worlds, 6))
        self.grasp_clamp_counts = zeros((worlds, 2), int)
        self.force_limits = zeros((worlds, 3))
        self.safe_known = zeros(worlds, int)
        safe_q = zeros((worlds, 6))
        safe_jac = zeros(worlds, Mat66)
        bad = zeros(worlds, int)
        direct = zeros(worlds, int)
        nonfinite = zeros(worlds, int)
        jaw_guard = self.jaw_guard = zeros(worlds, int)
        c = d.contact
        geom_groups = wp.array(groups, dtype=int, device=sim.device)
        surface_slots = wp.array(film_slots, dtype=int, device=sim.device)
        table = m.geom("table").id
        payload = m.geom(scene.payload_geom).id if self.flex else -1
        if self.flex:
            edge_array = scene.parcel.envelope["edges"]
            x = scene.parcel.envelope["reference"]
            rest = np.linalg.norm(x[edge_array[:, 0]] - x[edge_array[:, 1]], axis=1)
            edge_inputs = [d.flexvert_xpos, wp.array(edge_array, dtype=wp.vec2i, device=sim.device),
                           wp.array(rest, dtype=float, device=sim.device), self.edge_peak]

        def guard(gpu_model, data):
            bad.fill_(d.naconmax)
            wp.launch(find_bad_contact, d.naconmax, [d.nacon, c.worldid, c.geom, c.flex, c.dist,
                      geom_groups, surface_slots, table,
                      limits.flex_penetration_m if self.flex else limits.penetration_m, bad,
                      self.static_film_min], device=sim.device)
            if self.flex:
                wp.launch(measure_edges, (worlds, len(edge_array)), edge_inputs, device=sim.device)
            wp.launch(guard_substep, worlds, [d.qpos, d.qvel, d.overflow, d.nacon, bad, c.geom, c.flex,
                      c.dist, geom_groups, d.time, self.edge_peak, limits.flex_edge_stretch, self.reason,
                      self.stop_pair, self.stop_value], device=sim.device)
            wp.launch(apply_guarded_motor, (worlds, 7), [sim.robot, sim.cmds, sim.motors, d.qpos, d.qvel,
                      d.ctrl, gpu_model.body_mass, d.xipos, d.xanchor, d.xaxis, self.force_limits,
                      self.probe_event, jaw_guard, self.reason], device=sim.device)

        def after_solve(_model, _data, stage):
            if stage != Stage.ACC:
                return
            self.load.zero_()
            nonfinite.zero_()
            direct.fill_(d.naconmax)
            wp.launch(sum_contact_forces, d.naconmax, [d.nacon, c.worldid, c.geom, c.flex, c.efc_address,
                      c.dist, d.efc.force, geom_groups, table, sim.pad_ids, payload, self.load, direct,
                      nonfinite], device=sim.device)
            wp.launch(control_after_solve, worlds, [sim.robot, sim.cmds, d.qpos, d.site_xpos, d.xanchor,
                      d.xaxis, d.time, cfg["ik_damping"], self.load, self.force_limits, safe_q, safe_jac, self.safe_known,
                      self.unload_q, self.probe_event, jaw_guard, self.peaks, self.grasp_clamp_counts,
                      direct, nonfinite, c.geom, c.dist, d.overflow, self.reason, self.stop_pair, self.stop_value],
                      device=sim.device)

        sim.gpu_model.callback.sensor = after_solve
        if self.flex:
            install_pad_film_contact(sim.gpu_model, sim.pad_ids, scene.pad_film_contact)
            scene.couple(sim, guard)
        else:
            sim.before_solve = guard

    def read_batch(self):
        sim = self.sim
        mappings = {
            "time": sim.data.time, "qpos": sim.data.qpos, "qvel": sim.data.qvel,
            "observation": sim.obs, "command": sim.sample_action, "timestamp": sim.measurement_timestamp,
            "frame_load": self.load, "frame_peak": self.peaks, "frame_edge_peak": self.edge_peak,
            "frame_static_film_min_m": self.static_film_min,
            "frame_reason": self.reason, "probe_event": self.probe_event, "unload_q": self.unload_q,
            "clamp_counts": self.grasp_clamp_counts, "frame_stop_pair": self.stop_pair,
            "frame_stop_value": self.stop_value,
        }
        return {field: array.numpy() for field, array in mappings.items()}

    def populate_frame(self, target, frame, world):
        # Reconstruct the final-frame geometry from the integrated state.
        # Contacts, solver rows, flex Jacobians and substep forces stay on device.
        host = target.data
        host.time = float(frame["time"][world])
        host.qpos[:] = frame["qpos"][world]
        host.qvel[:] = frame["qvel"][world]
        for field in ("observation", "command", "timestamp", "frame_load", "frame_peak",
                      "frame_edge_peak", "frame_static_film_min_m", "frame_reason", "clamp_counts",
                      "frame_stop_pair", "frame_stop_value"):
            setattr(target, field, frame[field][world])
        target.static_film_surfaces = self.static_film_surfaces
        if np.isfinite(host.qpos).all() and np.isfinite(host.qvel).all():
            mujoco.mj_kinematics(self.model, host)
            mujoco.mj_comPos(self.model, host)
            mujoco.mj_camlight(self.model, host)
            if self.flex:
                mujoco.mj_flex(self.model, host)
        elif not target.frame_reason:
            target.frame_reason = 2
        target.unload_target = frame["unload_q"][world]
        load, time, distance = frame["probe_event"][world].tolist()
        if target.current_probe_cutoff and load and target.probe_trigger_event is None:
            target.probe_trigger_event = dict(load_N=load, time_s=time, pad_film_dist_m=distance)

    def _read(self):
        self.populate_frame(self, self.read_batch(), 0)

    def reset_batch(self, q, payload=None):
        if np.shape(q) != (self.sim.worlds, self.model.nq):
            raise ValueError("Batch reset requires a complete qpos for every world")
        self.sim.reset(initial_qpos=q)
        if payload is not None:
            values = np.asarray(payload, dtype=np.float32)
            if not self.randomized_payload or values.shape != (self.sim.worlds, 2):
                raise ValueError("Expected mass and friction for every randomized world")
            mass = np.broadcast_to(self.model.body_mass, (self.sim.worlds, self.model.nbody)).copy()
            inertia = np.broadcast_to(self.model.body_inertia, (self.sim.worlds, self.model.nbody, 3)).copy()
            friction = np.broadcast_to(self.model.geom_friction, (self.sim.worlds, self.model.ngeom, 3)).copy()
            mass[:, self.payload_body] = values[:, 0]
            inertia[:, self.payload_body] *= (values[:, 0] / self.model.body_mass[self.payload_body])[:, None]
            friction[:, self.payload_geom, 0] = values[:, 1]
            gpu_model = self.sim.gpu_model
            gpu_model.body_mass.assign(mass.astype(np.float32))
            gpu_model.body_inertia.assign(inertia.astype(np.float32))
            gpu_model.geom_friction.assign(friction.astype(np.float32))
            with wp.ScopedDevice(self.sim.device):
                mjw.set_const(gpu_model, self.sim.data)
        for array in (self.reason, self.load, self.peaks, self.edge_peak, self.jaw_guard, self.force_limits,
                      self.probe_event, self.grasp_clamp_counts, self.stop_value, self.safe_known):
            array.zero_()
        self.static_film_min.fill_(np.inf)
        self.stop_pair.fill_(-1)
        self.probe_trigger_event = None
        self.current_probe_cutoff = 0.
        self.stopped = False
        self.current_effort = None

    def park_worlds(self, worlds):
        sim = self.sim
        mask = np.zeros(sim.worlds, dtype=bool)
        mask[list(worlds)] = True
        mask = wp.array(mask, dtype=wp.bool, device=sim.device)
        sim.reset(mask=mask)
        wp.launch(clear_idle_guard, sim.worlds, [mask, self.reason, self.jaw_guard, self.probe_event],
                  device=sim.device)
        self.current_effort = None

    def reset(self, q, payload=None):
        if payload is not None:
            self.payload_mass = payload[0]
        self.reset_batch(np.asarray(q)[None, :], None if payload is None else [payload])
        self._read()

    def set_probe_cutoff(self, threshold):
        self.probe_event.zero_()
        self.current_probe_cutoff = threshold
        if threshold:
            self.probe_trigger_event = None

    def probe_unload_target(self):
        return self.unload_target

    def penetration_event(self):
        return self.contact_stop_event(self)

    def contact_stop_event(self, frame):
        if frame.frame_reason not in (3, 4, 5, 9):
            return None
        a, b, flex_a, flex_b = frame.frame_stop_pair
        depth, stamp, direct_force = frame.frame_stop_value
        return dict(geom=[self.model.geom(int(i)).name if i >= 0 else None for i in (a, b)],
                    flex=[int(f) if g < 0 else -1 for g, f in ((a, flex_a), (b, flex_b))],
                    separation_m=float(depth), time_s=float(stamp),
                    direct_force_N=float(direct_force))

    def stop(self):
        self.stopped = True
        self.sim.data.ctrl.zero_()
        self.sim.motors.applied.zero_()

    def step_batch(self, commands, efforts, limits, *, clear_probe=None):
        sim = self.sim
        efforts = np.asarray(efforts, dtype=np.float32)
        limits = np.asarray(limits, dtype=np.float32)
        if efforts.shape != (sim.worlds,) or limits.shape != (sim.worlds, 3):
            raise ValueError("Per-world effort and force limits must match the batch size")
        if not np.array_equal(efforts, self.current_effort):
            wp.launch(set_effort, sim.worlds, [sim.motors.tuning,
                      wp.array(efforts, dtype=float, device=sim.device), self.gear], device=sim.device)
            self.current_effort = efforts.copy()
        if clear_probe is not None:
            wp.launch(clear_probe_event, sim.worlds, [self.probe_event,
                      wp.array(clear_probe, dtype=int, device=sim.device)], device=sim.device)
        self.force_limits.assign(limits)
        self.peaks.zero_()
        self.edge_peak.zero_()
        self.static_film_min.fill_(np.inf)
        sim.send_joint_command(commands)
        sim.step()

    def tick(self, command, effort, _check, phase):
        if self.stopped:
            raise RuntimeError("Drive is stopped until reset")
        limit = 3.5 if self.flex and phase == "probe" else (
            self.limits.probe_contact_N if phase in ("probe", "reset") else self.limits.grasp_contact_N)
        pad_limit = 25. if self.flex and phase == "grasp" else 0.
        self.step_batch(command[None, :], [effort], [[limit, pad_limit, self.current_probe_cutoff]])
        self._read()
        self.check_stop(self.frame_reason, phase)

    @staticmethod
    def check_stop(reason, phase):
        if reason:
            raise MotionStop(STOP_REASONS[reason], physics_valid=reason in (5, 8),
                             unsafe=reason in (5, 8, 10) or phase == "probe" and reason in (4, 6))
