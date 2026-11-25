"""Attach the macro safety and effort cap to the existing MJWarp Yam_B interface."""

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.types import Stage, vec5
from yam_platform.yam_warp import SimYAMInterface
from yam_platform.yam_warp_kernels import Commands, Geometry, Motor, coordinates

from .environment import MotionStop
from .yam_kinematics import descendants


@wp.kernel
def limit_gripper(tuning: wp.array2d(dtype=Motor), gap_force: float):
    motor = tuning[0, 6]
    motor.limit = gap_force
    tuning[0, 6] = motor


@wp.kernel
def apply_pad_film_contact(ncon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
                           flex: wp.array(dtype=wp.vec2i), pads: wp.array(dtype=int),
                           ref: wp.vec2, imp: vec5, solref: wp.array(dtype=wp.vec2),
                           solreffriction: wp.array(dtype=wp.vec2), solimp: wp.array(dtype=vec5)):
    i = wp.tid()
    if i < ncon[0]:
        pair, bag = geom[i], flex[i]
        if (((pair[0] == pads[0] or pair[0] == pads[1]) and bag[1] >= 0) or
                ((pair[1] == pads[0] or pair[1] == pads[1]) and bag[0] >= 0)):
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


@wp.kernel
def check_substep(
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), overflow: wp.array(dtype=int),
    nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float),
    group: wp.array(dtype=int), table: int, max_depth: float,
    points: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i), rest_length: wp.array(dtype=float),
    max_stretch: float, ctrl: wp.array2d(dtype=float), stop: wp.array(dtype=int),
    flex: wp.array(dtype=wp.vec2i), stamp: wp.array(dtype=float),
    stop_pair: wp.array(dtype=int), stop_value: wp.array(dtype=float),
    pads: wp.array(dtype=int),
):
    if stop[0] == 0:
        reason = int(0)
        if overflow[0] != 0:
            reason = 1
        for j in range(qpos.shape[1]):
            if not wp.isfinite(qpos[0, j]):
                reason = 2
        for j in range(qvel.shape[1]):
            if not wp.isfinite(qvel[0, j]):
                reason = 2
        contact_with_arm = bool(False)
        for i in range(wp.min(nacon[0], dist.shape[0])):
            pair = geom[i]
            a = int(0)
            b = int(0)
            if pair[0] >= 0:
                a = group[pair[0]]
            if pair[1] >= 0:
                b = group[pair[1]]
            base_contact = (pair[0] == table and b == 2) or (pair[1] == table and a == 2)
            if not base_contact:
                active = a == 1 or a == 2 or b == 1 or b == 2
                contact_with_arm = contact_with_arm or (active and dist[i] < 0.)
                f = flex[i]
                pad_film = ((pair[0] == pads[0] or pair[0] == pads[1]) and f[1] >= 0) or (
                    (pair[1] == pads[0] or pair[1] == pads[1]) and f[0] >= 0)
                if reason == 0 and not pad_film and dist[i] < -max_depth:
                    reason = 3
                    if active:
                        reason = 4
                    f = flex[i]
                    stop_pair[0] = pair[0]
                    stop_pair[1] = pair[1]
                    stop_pair[2] = f[0]
                    stop_pair[3] = f[1]
                    stop_value[0] = dist[i]
                    stop_value[1] = stamp[0]
                if reason == 0 and active and dist[i] < -0.0005:
                    if a == 3 or b == 3 or ((a == 1 or a == 2) and (b == 1 or b == 2)):
                        reason = 5
        if reason == 0:
            for i in range(edges.shape[0]):
                edge = edges[i]
                if wp.length(points[0, edge[0]] - points[0, edge[1]]) > max_stretch * rest_length[i]:
                    reason = 6
                    if contact_with_arm:
                        reason = 7
        stop[0] = reason
    if stop[0] != 0:
        for j in range(ctrl.shape[1]):
            ctrl[0, j] = 0.


@wp.kernel
def check_solved_force(
    nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), address: wp.array2d(dtype=int),
    force: wp.array2d(dtype=float), group: wp.array(dtype=int), table: int, limit: wp.array(dtype=float),
    stop: wp.array(dtype=int), peak: wp.array(dtype=float), ctrl: wp.array2d(dtype=float),
    cutoff: wp.array(dtype=float), trigger: wp.array(dtype=int), trigger_load: wp.array(dtype=float), payload: int,
    dist: wp.array(dtype=float), stamp: wp.array(dtype=float), stop_pair: wp.array(dtype=int),
    stop_value: wp.array(dtype=float),
    pads: wp.array(dtype=int), flex: wp.array(dtype=wp.vec2i), pad_limit: wp.array(dtype=float),
    pad_stats: wp.array(dtype=float), probe_force: wp.array(dtype=float),
):
    if stop[0] == 0:
        total = float(0.)
        left, right = float(0.), float(0.)
        pad_separation = float(0.)
        for i in range(wp.min(nacon[0], geom.shape[0])):
            pair = geom[i]
            row = address[i, 0]
            if row >= 0 and row < force.shape[1]:
                n = force[0, row]
                if not wp.isfinite(n):
                    stop[0] = 2
                a, b = int(0), int(0)
                if pair[0] >= 0:
                    a = group[pair[0]]
                if pair[1] >= 0:
                    b = group[pair[1]]
                base_contact = (pair[0] == table and b == 2) or (pair[1] == table and a == 2)
                if not base_contact and (a == 1 or a == 2 or b == 1 or b == 2):
                    total += wp.max(n, 0.)
                    f = flex[i]
                    if (pair[0] == pads[0] and f[1] >= 0) or (pair[1] == pads[0] and f[0] >= 0):
                        left += wp.max(n, 0.)
                        pad_separation = wp.min(pad_separation, dist[i])
                    if (pair[0] == pads[1] and f[1] >= 0) or (pair[1] == pads[1] and f[0] >= 0):
                        right += wp.max(n, 0.)
                        pad_separation = wp.min(pad_separation, dist[i])
                    if stop[0] == 0 and payload >= 0 and n > 1.e-6 and (pair[0] == payload or pair[1] == payload):
                        stop[0] = 9
                        stop_pair[0] = pair[0]
                        stop_pair[1] = pair[1]
                        stop_value[0] = dist[i]
                        stop_value[1] = stamp[0]
                        stop_value[2] = n
        peak[0] = wp.max(peak[0], total)
        probe_force[0] = total
        pad_stats[0] = wp.max(pad_stats[0], left)
        pad_stats[1] = wp.max(pad_stats[1], right)
        pad_stats[2] = wp.min(pad_stats[2], pad_separation)
        if stop[0] == 0 and cutoff[0] > 0. and trigger[0] == 0 and total >= cutoff[0]:
            trigger[0] = 1
            trigger_load[0] = total
            trigger_load[1] = stamp[0]
            trigger_load[2] = pad_separation
        if stop[0] == 0 and (total > limit[0] or (pad_limit[0] > 0. and wp.max(left, right) > pad_limit[0])):
            stop[0] = 8
    if stop[0] != 0:
        for j in range(ctrl.shape[1]):
            ctrl[0, j] = 0.


@wp.kernel
def hold_probe_targets(robot: Geometry, cmds: Commands, qpos: wp.array2d(dtype=float),
                       trigger: wp.array(dtype=int), cutoff: wp.array(dtype=float),
                       force: wp.array(dtype=float), unload_q: wp.array(dtype=float)):
    if cutoff[0] > 0.:
        if trigger[0] == 0 and force[0] <= .5 * cutoff[0]:
            for i in range(6):
                unload_q[i] = coordinates(qpos, robot.pos, 0, i)
        if trigger[0] == 1:
            for i in range(7):
                goal = coordinates(qpos, robot.pos, 0, i)
                if i < 6:
                    goal = unload_q[i]
                cmds.requested[0, i] = goal
                cmds.target[0, i] = goal
                cmds.velocity[0, i] = 0.
                cmds.desired_velocity[0, i] = 0.
                cmds.feedforward[0, i] = 0.
            trigger[0] = 2


class GPUDrive:
    name = "mjwarp_cuda"

    def __init__(self, scene, cfg, seed, device, motor_Nm_per_N, limits):
        self.gear = motor_Nm_per_N
        capacities = dict(nconmax=1024, njmax=4096, njmax_nnz=32768) if scene.is_flex else dict(
            nconmax=128, njmax=256)
        sim = self.sim = SimYAMInterface(1, config=cfg, seed=seed, device=device,
                                         add_scene=scene.add_scene, compile_model=scene.compile_model,
                                         initialize=scene.initialize, capacities=capacities)
        self.model = m = sim.model
        self.initial = sim.initial.numpy()[0].copy()
        self.data = mujoco.MjData(m)
        d = sim.data
        arm = descendants(m, "arm")
        base = m.body("arm").id
        groups = np.array([2 if body == base else 1 if body in arm else
                           3 if m.geom(i).name.startswith(("table", "bin_", "Yam_A_")) else 0
                           for i, body in enumerate(m.geom_bodyid)], dtype=np.int32)
        if scene.is_flex:
            edge_array = scene.parcel.envelope["edges"]
            x = scene.parcel.envelope["reference"]
            rest = np.linalg.norm(x[edge_array[:, 0]] - x[edge_array[:, 1]], axis=1)
        else:
            edge_array = np.empty((0, 2), dtype=np.int32)
            rest = np.empty(0)
        self.reason = wp.zeros(1, dtype=int, device=sim.device)
        self.peak_contact = wp.zeros(1, dtype=float, device=sim.device)
        self.force_limit = wp.zeros(1, dtype=float, device=sim.device)
        self.probe_cutoff = wp.zeros(1, dtype=float, device=sim.device)
        self.probe_force = wp.zeros(1, dtype=float, device=sim.device)
        self.unload_q = wp.zeros(6, dtype=float, device=sim.device)
        self.flex = scene.is_flex
        self.pad_limit = wp.zeros(1, dtype=float, device=sim.device)
        self.pad_stats = wp.zeros(3, dtype=float, device=sim.device)
        self.probe_trigger = wp.zeros(1, dtype=int, device=sim.device)
        self.probe_trigger_load = wp.zeros(3, dtype=float, device=sim.device)
        self.stop_pair = wp.full(4, -1, dtype=int, device=sim.device)
        self.stop_value = wp.zeros(3, dtype=float, device=sim.device)
        self.limits = limits
        geom_groups = wp.array(groups, dtype=int, device=sim.device)
        table = m.geom("table").id
        args = [d.qpos, d.qvel, d.overflow, d.nacon, d.contact.geom, d.contact.dist,
                geom_groups, table,
                limits.flex_penetration_m if scene.is_flex else limits.penetration_m,
                d.flexvert_xpos, wp.array(edge_array, dtype=wp.vec2i, device=sim.device),
                wp.array(rest, dtype=float, device=sim.device), limits.flex_edge_stretch, d.ctrl, self.reason,
                d.contact.flex, d.time, self.stop_pair, self.stop_value, sim.pad_ids]
        payload = m.geom(scene.payload_geom).id if scene.is_flex else -1

        def guard(_model, _data):
            wp.launch(check_substep, 1, args, device=sim.device)

        def after_solve(_model, _data, stage):
            if stage == Stage.ACC:
                wp.launch(check_solved_force, 1, [d.nacon, d.contact.geom, d.contact.efc_address, d.efc.force,
                          geom_groups, table, self.force_limit, self.reason, self.peak_contact, d.ctrl,
                          self.probe_cutoff, self.probe_trigger, self.probe_trigger_load, payload,
                          d.contact.dist, d.time, self.stop_pair, self.stop_value,
                          sim.pad_ids, d.contact.flex, self.pad_limit, self.pad_stats, self.probe_force],
                          device=sim.device)
                wp.launch(hold_probe_targets, 1, [sim.robot, sim.cmds, d.qpos, self.probe_trigger,
                          self.probe_cutoff, self.probe_force, self.unload_q], device=sim.device)

        sim.gpu_model.callback.sensor = after_solve
        if scene.is_flex:
            install_pad_film_contact(sim.gpu_model, sim.pad_ids, scene.pad_film_contact)
            scene.couple(sim, guard)
        else:
            sim.before_solve = guard

    def _read(self):
        import mujoco_warp as mjw

        mjw.get_data_into(self.data, self.model, self.sim.data, world_id=0)
        self.observation = self.sim.obs.numpy()[0]
        sample = self.sim.get_command_history_sample()
        self.command = sample["previous_action"].numpy()[0]
        self.timestamp = float(sample["measurement_timestamp"].numpy()[0])

    def reset(self, q):
        self.sim.reset(initial_qpos=q)
        self.reason.zero_()
        self.peak_contact.zero_()
        self.set_probe_cutoff(0.)
        self.probe_trigger_event = None
        self.pad_limit.zero_()
        self.pad_stats.zero_()
        self.current_pad_limit = 0.
        self.stop_pair.fill_(-1)
        self.stop_value.zero_()
        self.current_force_limit = None
        self.stopped = False
        self.current_effort = None
        self._read()

    def set_probe_cutoff(self, threshold):
        self.probe_trigger.zero_()
        self.probe_trigger_load.zero_()
        self.probe_cutoff.fill_(threshold)
        self.current_probe_cutoff = threshold
        if threshold:
            self.probe_trigger_event = None
            self.unload_q.assign(np.asarray(self.observation[:6], dtype=np.float32))

    def penetration_event(self):
        if int(self.reason.numpy()[0]) not in (3, 4, 9):
            return None
        a, b, flex_a, flex_b = self.stop_pair.numpy()
        depth, stamp, direct_force = self.stop_value.numpy()
        return dict(geom=[self.model.geom(int(i)).name if i >= 0 else None for i in (a, b)],
                    flex=[int(flex_a), int(flex_b)], separation_m=float(depth), time_s=float(stamp),
                    direct_force_N=float(direct_force))

    def stop(self):
        self.stopped = True
        self.sim.data.ctrl.zero_()
        self.sim.motors.applied.zero_()

    def tick(self, command, effort, check, phase):
        if self.stopped:
            raise RuntimeError("Drive is stopped until reset")
        sim = self.sim
        if effort != self.current_effort:
            wp.launch(limit_gripper, 1, [sim.motors.tuning, effort / self.gear], device=sim.device)
            self.current_effort = effort
        probing = self.flex and phase == "probe"
        pad_limit = 25. if self.flex and phase == "grasp" else 0.
        if pad_limit != self.current_pad_limit:
            self.pad_limit.fill_(pad_limit)
            self.current_pad_limit = pad_limit
        limit = 3.5 if probing else (self.limits.probe_contact_N if phase in ("probe", "reset")
                                     else self.limits.grasp_contact_N)
        if limit != self.current_force_limit:
            self.force_limit.fill_(limit)
            self.current_force_limit = limit
        sim.send_joint_command(command)
        self.pad_stats.zero_()
        sim.step()
        self._read()
        if self.current_probe_cutoff and self.probe_trigger_event is None:
            load, time, distance = self.probe_trigger_load.numpy().tolist()
            if load:
                self.probe_trigger_event = dict(load_N=load, time_s=time, pad_film_dist_m=distance)
        reason = int(self.reason.numpy()[0])
        invalid = {
            1: "solver_overflow", 2: "nonfinite_physics", 3: "excessive_contact_penetration",
            4: "excessive_contact_penetration", 6: "flex_overstretch", 7: "flex_overstretch",
            9: "robot_payload_direct_force",
        }
        if reason in invalid:
            raise MotionStop(invalid[reason], physics_valid=False, unsafe=phase == "probe" and reason in (4, 7))
        if reason == 8:
            raise MotionStop("contact_force_limit", unsafe=True)
        if reason == 5:
            raise MotionStop("workcell_collision", unsafe=True)
        if (not np.isfinite(self.observation).all() or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qacc).all()):
            raise MotionStop("nonfinite_physics", physics_valid=False)
        check()
