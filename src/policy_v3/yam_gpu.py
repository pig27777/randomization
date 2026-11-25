"""Attach the macro safety and effort cap to the existing MJWarp Yam_B interface."""

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.types import Stage
from yam_platform.yam_warp import SimYAMInterface
from yam_platform.yam_warp_kernels import Motor

from .environment import MotionStop
from .yam_kinematics import descendants


@wp.kernel
def limit_gripper(tuning: wp.array2d(dtype=Motor), gap_force: float):
    motor = tuning[0, 6]
    motor.limit = gap_force
    tuning[0, 6] = motor


@wp.kernel
def check_substep(
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), overflow: wp.array(dtype=int),
    nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float),
    group: wp.array(dtype=int), table: int, max_depth: float,
    points: wp.array2d(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i), rest_length: wp.array(dtype=float),
    max_stretch: float, ctrl: wp.array2d(dtype=float), stop: wp.array(dtype=int),
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
                if reason == 0 and dist[i] < -max_depth:
                    reason = 3
                    if active:
                        reason = 4
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
):
    if stop[0] == 0:
        total = float(0.)
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
        peak[0] = wp.max(peak[0], total)
        if stop[0] == 0 and total > limit[0]:
            stop[0] = 8
    if stop[0] != 0:
        for j in range(ctrl.shape[1]):
            ctrl[0, j] = 0.


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
        self.limits = limits
        geom_groups = wp.array(groups, dtype=int, device=sim.device)
        table = m.geom("table").id
        args = [d.qpos, d.qvel, d.overflow, d.nacon, d.contact.geom, d.contact.dist,
                geom_groups, table,
                limits.flex_penetration_m if scene.is_flex else limits.penetration_m,
                d.flexvert_xpos, wp.array(edge_array, dtype=wp.vec2i, device=sim.device),
                wp.array(rest, dtype=float, device=sim.device), limits.flex_edge_stretch, d.ctrl, self.reason]

        def guard(_model, _data):
            wp.launch(check_substep, 1, args, device=sim.device)

        def after_solve(_model, _data, stage):
            if stage == Stage.ACC:
                wp.launch(check_solved_force, 1, [d.nacon, d.contact.geom, d.contact.efc_address, d.efc.force,
                          geom_groups, table, self.force_limit, self.reason, self.peak_contact, d.ctrl],
                          device=sim.device)

        sim.gpu_model.callback.sensor = after_solve
        if scene.is_flex:
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
        self.current_force_limit = None
        self.stopped = False
        self.current_effort = None
        self._read()

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
        limit = self.limits.probe_contact_N if phase in ("probe", "reset") else self.limits.grasp_contact_N
        if limit != self.current_force_limit:
            self.force_limit.fill_(limit)
            self.current_force_limit = limit
        sim.send_joint_command(command)
        sim.step()
        self._read()
        reason = int(self.reason.numpy()[0])
        invalid = {
            1: "solver_overflow", 2: "nonfinite_physics", 3: "excessive_contact_penetration",
            4: "excessive_contact_penetration", 6: "flex_overstretch", 7: "flex_overstretch",
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
