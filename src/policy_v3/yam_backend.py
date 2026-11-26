"""Shared macro motion for native Yam_B dynamics and the PE Flex adapter."""

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np

from bag_sim.wrap import triangle_box_intersections, winding_number
from yam_platform.robot import rotation_error
from yam_platform.yam_model import arm_indices, build_yam_model, initial_state, load_config

from .environment import MotionStop
from .geometry import VisualObservation, rotation_6d, rotation_matrix
from .perception import SimDepthCamera
from .yam_kinematics import WorkcellKinematics, descendants
from .yam_scene import BIN, CORNERS, PICK, FlexScene, RigidScene, add_workcell


@dataclass(frozen=True)
class SafetyLimits:
    probe_contact_N: float = 85.
    grasp_contact_N: float = 130.
    penetration_m: float = .003
    flex_penetration_m: float = .001
    flex_edge_stretch: float = 1.12


class YamBackend:
    """CPU rigid smoke and GPU Yam_B use the same kinematics and task checks."""

    def __init__(self, seed, *, scene="rigid", device="cpu", scripted=False, camera=False, limits=SafetyLimits(),
                 worlds=1, randomized_payload=False, flex_solver="nominal", solver_options=None,
                 contact_capacity=None, constraint_capacity=None):
        self.rng = np.random.default_rng(seed)
        self.scripted = scripted
        self.limits = limits
        if scene not in ("rigid", "flex"):
            raise ValueError(f"Unsupported Yam scene {scene}")
        spec = self.scene = FlexScene(flex_solver, **(solver_options or {})) if scene == "flex" else RigidScene()
        if randomized_payload and not spec.is_flex:
            raise ValueError("Per-world payload randomization is available for GPU Flex")
        if spec.is_flex and device == "cpu":
            raise ValueError("The PE Flex policy adapter requires MJWarp CUDA")
        cfg = self.config = load_config("nominal", spec.config)
        self.workspace_lower = np.array(cfg["workspace_lower_m"])
        self.workspace_upper = np.array(cfg["workspace_upper_m"])
        self.hover_joints = np.array(cfg["ready_joints_rad"])
        self.hover_gap = cfg["ready_gap_m"]
        self.max_gap = cfg["gripper_max_gap_m"]
        self.speed = np.array([*cfg["joint_target_speed_rad_s"], cfg["gripper_speed_m_s"]])
        self.motor_Nm_per_N = cfg["gripper_sdk_stroke_m"] / cfg["gripper_motor_travel_rad"]
        self.effort_cap_Nm = cfg["gripper_force_limit_N"] * self.motor_Nm_per_N
        self.probe_effort_Nm = 15. * self.motor_Nm_per_N
        self.grasp_width_range = (.060, self.max_gap)
        self.grasp_effort_range = (16. * self.motor_Nm_per_N, self.effort_cap_Nm)
        self.probe_displacement_m = np.array([.029, .029, .038])
        self.probe_cutoff_N = 1.80 if spec.is_flex else None
        self.probe_contact_speed = .16
        self.guard_full_probe_search = spec.is_flex
        self.grasp_cruise_scale = .65
        self.grasp_approach_scale = .65
        self.compact_grasp = False
        self.common_sampling_region = (PICK + [0, 0, .052], np.array([.045, .039, .024]), np.eye(3))
        _, empty = build_yam_model(cfg, add_scene=add_workcell)
        self.kinematics = WorkcellKinematics(empty, self.workspace_lower, self.workspace_upper)
        _, self.hover_rotation = self.kinematics.pose(self.hover_joints)
        if device == "cpu":
            _, model = build_yam_model(cfg, add_scene=spec.add_scene)
            self.drive = CPURigidDrive(model, cfg, self.motor_Nm_per_N)
        else:
            from .yam_gpu import GPUDrive
            self.drive = GPUDrive(spec, cfg, seed, device, self.motor_Nm_per_N, limits, worlds=worlds,
                                  randomized_payload=randomized_payload, contact_capacity=contact_capacity,
                                  constraint_capacity=constraint_capacity)
        self.randomized_payload = randomized_payload
        self.model = model = self.drive.model
        _, self.qadr, self.dofs = arm_indices(model)
        self.site = model.site("grasp_site").id
        self.payload_id = model.body(spec.payload_body).id
        self.payload_geom = model.geom(spec.payload_geom).id
        self.arm_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, list(descendants(model, "arm")))))
        self.base_geoms = set(np.flatnonzero(model.geom_bodyid == model.body("arm").id))
        self.obstacles = {g for g in range(model.ngeom)
                          if model.geom(g).name.startswith(("table", "bin_", "Yam_A_"))}
        self.table = model.geom("table").id
        self.pads = (model.geom("left_pad").id, model.geom("right_pad").id)
        self.force = np.empty(6)
        self.optics = None
        if camera or spec.is_flex:
            if spec.is_flex:
                obj_type = mujoco.mjtObj.mjOBJ_FLEX
                obj_id = mujoco.mj_name2id(model, obj_type, "bag")
            else:
                obj_type, obj_id = mujoco.mjtObj.mjOBJ_GEOM, self.payload_geom
            self.optics = SimDepthCamera(model, obj_type, obj_id)
        if spec.is_flex:
            a, b = self.scene.parcel.envelope["edges"].T
            self.flex_edges = (a, b)
            rest = self.scene.parcel.envelope["reference"]
            self.flex_rest_length = np.linalg.norm(rest[a] - rest[b], axis=1)

    @property
    def joint_pos(self):
        return self.drive.observation[:6]

    def reset(self, *, seed=None):
        rng = self.rng if seed is None else np.random.default_rng(seed)
        if self.randomized_payload:
            q, randomization = self.scene.randomized_episode(self.model, self.drive.initial, rng)
            self.drive.reset(q, payload=(randomization["payload_mass_kg"], randomization["payload_friction"]))
        else:
            q, randomization = self.scene.episode(self.model, self.drive.initial, rng, self.scripted)
            self.drive.reset(q)
        self.last_command = np.r_[self.hover_joints, self.hover_gap]
        self.measured_ticks = 0
        self.episode_probe_peak_N = 0.
        self.motion_label = "reset"
        self.grasp_plan_cache = {}
        for _ in range(13 if self.scene.is_flex else 8):
            self.tick(self.last_command, self.probe_effort_Nm, "reset")
        self.original = self.drive.data.xpos[self.payload_id].copy()
        if not self.at_hover():
            raise RuntimeError("Reset failed to reach the common neutral pose")
        return self.observe(), self.private_state(), dict(initial_state=self.state_identity(), randomization_params={
            **randomization, "scene": "flex" if self.scene.is_flex else "rigid",
            "backend": self.drive.name, "safety_limits": asdict(self.limits),
        })

    def state_identity(self):
        return np.r_[self.drive.data.qpos, self.drive.data.qvel]

    def payload_box_pose(self):
        # The box half extents use the authored body frame. MuJoCo may
        # reorient a compiled mesh geom into its inertia principal axes.
        data = self.drive.data
        return data.xpos[self.payload_id], data.xmat[self.payload_id].reshape(3, 3)

    def flex_core_counts(self):
        if not np.isfinite(self.drive.data.qpos).all():
            return None
        center, rotation = self.payload_box_pose()
        points = self.drive.data.flexvert_xpos
        faces = self.scene.parcel.envelope["faces"]
        half = self.scene.half
        corners = CORNERS * half @ rotation.T + center
        inset = triangle_box_intersections(points, faces, half - .002, center, rotation)
        outside = np.abs(winding_number(points, faces, corners)) < .5
        return int(inset.sum()), int(outside.sum())

    def private_state(self):
        d = self.drive.data
        p = self.payload_id
        return np.r_[d.xpos[p], d.xquat[p], d.qvel[self.model.joint("payload_free").dofadr[0]
                                                    + np.arange(6)],
                     getattr(self.drive, "payload_mass", self.model.body_mass[p])].astype(np.float32)

    def observe(self):
        data = self.drive.data
        if self.optics is not None:
            return self.optics.observe(data)
        return VisualObservation(data.geom_xpos[self.payload_geom],
                                 data.geom_xmat[self.payload_geom].reshape(3, 3),
                                 self.scene.half, "fixture_geometry")

    def close(self):
        if self.optics is not None:
            self.optics.close()

    def stop(self):
        self.drive.stop()

    def at_hover(self):
        obs = self.drive.observation
        p, r = self.kinematics.pose(obs[:6])
        goal, target_rot = self.kinematics.pose(self.hover_joints)
        return bool(np.max(np.abs(obs[:6] - self.hover_joints)) < .018
                    and np.max(np.abs(obs[6:12])) < .075
                    and abs(obs[24] - self.hover_gap) < .005
                    and abs(obs[25]) < .020 and np.linalg.norm(p - goal) < .005
                    and np.linalg.norm(rotation_error(r, target_rot)) < .035
                    and not any(self.contact_loads(check=False)))

    def settle_hover(self, effort, record):
        self.motion_label = "neutral_hover"
        for _ in range(28):
            self.tick(np.r_[self.hover_joints, self.hover_gap], effort, "probe")
            record("neutral_hover", 1.)
            if self.at_hover():
                break

    def contact_loads(self, *, check, phase=None):
        if hasattr(self.drive, "frame_load"):
            return tuple(self.drive.frame_load[:3])
        m, d = self.model, self.drive.data
        load = 0.
        pads = [0., 0.]
        max_depth = self.limits.flex_penetration_m if self.scene.is_flex else self.limits.penetration_m
        for i, c in enumerate(d.contact):
            a, b = c.geom
            active = a in self.arm_geoms or b in self.arm_geoms
            base_table = ((a == self.table and b in self.base_geoms) or
                          (b == self.table and a in self.base_geoms))
            if base_table:
                continue
            pad_film = self.scene.is_flex and ((a in self.pads and c.flex[1] >= 0)
                                               or (b in self.pads and c.flex[0] >= 0))
            if check and not pad_film and c.dist < -max_depth:
                raise MotionStop("excessive_contact_penetration", physics_valid=False, unsafe=active and phase == "probe")
            if not active:
                continue
            if check and c.dist < -.0005 and (a in self.obstacles or b in self.obstacles
                                              or (a in self.arm_geoms and b in self.arm_geoms)):
                raise MotionStop("workcell_collision", unsafe=True)
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, self.force)
            n = max(0., self.force[0])
            load += n
            if self.payload_geom in (a, b) or (self.scene.is_flex and max(c.flex) >= 0):
                for j, pad in enumerate(self.pads):
                    if pad in (a, b):
                        pads[j] += n
        if check:
            limit = 3.5 if self.scene.is_flex and phase == "probe" else (
                self.limits.probe_contact_N if phase in ("reset", "probe") else self.limits.grasp_contact_N)
            if load > limit or (self.scene.is_flex and phase == "grasp" and max(pads) > 25.):
                raise MotionStop("contact_force_limit", unsafe=True)
            if self.scene.is_flex:
                a, b = self.flex_edges
                stretch = np.linalg.norm(d.flexvert_xpos[a] - d.flexvert_xpos[b], axis=1) / self.flex_rest_length
                if stretch.max() > self.limits.flex_edge_stretch:
                    raise MotionStop("flex_overstretch", physics_valid=False, unsafe=phase == "probe" and load > 0.)
        return load, *pads

    def tick(self, command, effort, phase):
        stop = None
        try:
            self.drive.tick(command, effort, lambda: self.contact_loads(check=True, phase=phase), phase)
        except MotionStop as error:
            stop = error
        if self.scene.is_flex:
            if phase == "probe" and hasattr(self.drive, "frame_peak"):
                self.episode_probe_peak_N = max(self.episode_probe_peak_N, float(self.drive.frame_peak[0]))
            self.frame_core_counts = self.flex_core_counts()
            if self.frame_core_counts and any(self.frame_core_counts):
                if stop is not None:
                    stop.physics_valid = False
                else:
                    stop = MotionStop("film_core_geometry", physics_valid=False, unsafe=phase == "probe")
        if stop is not None:
            raise stop
        self.last_command = command
        self.measured_ticks += 1

    def follow(self, plan, effort, *, record=None, stage="grasp", contact_cutoff_N=None):
        before = self.last_command.copy()
        targets = []
        cruise_stages = ("pregrasp", "transport", "retreat", "release", "place") if self.compact_grasp else (
            "pregrasp", "transport", "retreat")
        for q, gap, label in plan:
            contact_search = contact_cutoff_N is not None and label != "probe_approach"
            speed = self.probe_contact_speed if contact_search else .65
            if stage == "grasp" and label in cruise_stages:
                speed = self.grasp_cruise_scale
            if stage == "grasp" and label == "approach":
                speed = self.grasp_approach_scale
            nominal_step = speed * .04 * self.speed
            after = np.r_[q, gap]
            # The device has already latched this unload target.
            count = 1 if label == "probe_unload" else max(1, math.ceil(np.max(np.abs(after - before) / nominal_step)))
            targets.extend((before + a * (after - before), label) for a in np.arange(1, count + 1) / count)
            before = after
        if contact_cutoff_N is not None:
            self.drive.set_probe_cutoff(contact_cutoff_N)
        try:
            for step, (command, label) in enumerate(targets, 1):
                self.motion_label = label
                self.tick(command, effort, stage)
                if record is not None:
                    record(label, step / len(targets))
                if contact_cutoff_N is not None and self.drive.probe_trigger_event is not None:
                    self.last_command = np.r_[self.joint_pos, self.drive.observation[24]]
                    return self.drive.probe_trigger_event["load_N"]
        finally:
            if contact_cutoff_N is not None:
                self.drive.set_probe_cutoff(0.)
        return None

    def history_sample(self, initial, progress):
        obs = self.drive.observation
        if not np.isfinite(obs).all():
            return None
        q, dq, eff, q_cmd = obs[:24].reshape(4, 6)
        pos, rot, velocity = self.kinematics.pose(q, dq)
        target, target_rot = self.kinematics.pose(q_cmd)
        r = initial.rotation
        frame = np.r_[q, dq, q_cmd, q_cmd - q, eff, obs[24:27], obs[27] - obs[24],
                      (pos - initial.center) @ r, (target - pos) @ r, velocity @ r,
                      rotation_error(rot, target_rot) @ r, self.drive.command, progress]
        return frame, float(self.drive.timestamp), pos

    def grasp_plan(self, grasp):
        key = (self.compact_grasp, *grasp)
        if key in self.grasp_plan_cache:
            return self.grasp_plan_cache[key]
        p, r = grasp[:3], rotation_matrix(grasp[3:9])
        gap, effort = grasp[9:]
        if not (0. < gap <= self.max_gap and 0. < effort <= self.effort_cap_Nm):
            return None
        pre = p - .045 * r[:, 2]
        lift = p.copy()
        lift[2] = max(.185, p[2] + .132)
        corridor = np.array([-.11, p[1], lift[2]])
        basket = np.array([-.11, -.045, lift[2]])
        place = basket.copy()
        place[2] = .065
        release_gap = min(self.hover_gap, max(.048, gap)) if self.compact_grasp else self.hover_gap
        transit = [] if self.compact_grasp else [(corridor, self.hover_rotation, 0., "transport")]
        plan = self.kinematics.plan(self.hover_joints, self.hover_gap, [
            (pre, r, gap, "pregrasp"), (p, r, gap, "approach"), (p, r, 0., "close"),
            (lift, r, 0., "lift"), *transit,
            (basket, self.hover_rotation, 0., "transport"), (place, self.hover_rotation, 0., "place"),
            (place, self.hover_rotation, release_gap, "release"),
            (basket, self.hover_rotation, release_gap, "retreat"),
        ])
        if plan is not None:
            self.grasp_plan_cache[key] = plan
        return plan

    def scripted_grasp(self):
        return np.r_[PICK[:2], .052, rotation_6d(self.hover_rotation), self.hover_gap, self.effort_cap_Nm]

    def execute_grasp(self, grasp, *, screen_lift=False):
        plan = self.grasp_plan(grasp)
        if plan is None:
            return dict(task_success=False, failure_reason="grasp_path_rejected")
        peak_z = self.drive.data.xpos[self.payload_id, 2]
        carry_y = self.original[1]
        dual_frames = 0
        peak_gripper_effort = 0.
        min_rim_clearance = None

        def log(stage, _progress):
            nonlocal peak_z, carry_y, dual_frames, peak_gripper_effort, min_rim_clearance
            peak_gripper_effort = max(peak_gripper_effort, abs(float(self.drive.observation[26])))
            x = self.drive.data.xpos[self.payload_id]
            if stage == "lift":
                peak_z = max(peak_z, float(x[2]))
                dual_frames += min(self.contact_loads(check=False)[1:]) > 1.
            if stage == "transport":
                carry_y = max(carry_y, float(x[1]))
                if self.scene.is_flex:
                    center, rotation = self.payload_box_pose()
                    core_bottom = (CORNERS * self.scene.half @ rotation.T + center)[:, 2].min()
                    film_bottom = self.drive.data.flexvert_xpos[:, 2].min()
                    clearance = float(min(core_bottom, film_bottom) - BIN["rim_height_m"])
                    min_rim_clearance = clearance if min_rim_clearance is None else min(
                        min_rim_clearance, clearance)

        effort = float(grasp[-1])
        for label in ("pregrasp", "approach", "close", "lift", "transport", "place", "release", "retreat"):
            self.follow([step for step in plan if step[2] == label], effort, record=log)
            if label in ("close", "release"):
                hold_frames = 4 if self.compact_grasp and label == "release" else 8
                for _ in range(hold_frames):
                    self.tick(self.last_command, effort, "grasp")
                    log(label, 1.)
            if label == "lift":
                rise = float(peak_z - self.original[2])
                lifted = rise >= .04 and dual_frames >= 5
                if screen_lift and not lifted:
                    return dict(task_success=False, failure_reason="not_lifted", lifted=False,
                                peak_payload_lift_m=rise, lifting_dual_contact_frames=int(dual_frames))

        self.motion_label = "basket_settle"
        positions = [self.drive.data.xpos[self.payload_id].copy()]
        displacement = []
        for _ in range(math.ceil(BIN["settled_duration_s"] / .04)):
            self.tick(self.last_command, effort, "grasp")
            positions.append(self.drive.data.xpos[self.payload_id].copy())
            displacement.append(np.linalg.norm(positions[-1] - positions[-2]) / .04)
        data = self.drive.data
        center = np.array(BIN["center_m"])
        hxhy = np.array(BIN["inner_half_xy_m"]) - BIN["wall_thickness_m"] / 2
        xyz = positions[-1]
        core_center, core_rotation = self.payload_box_pose()
        solid = CORNERS * self.scene.half @ core_rotation.T + core_center
        outer = data.flexvert_xpos if self.scene.is_flex else solid
        inside = all(np.all(abs(points[:, :2] - center[:2]) < hxhy)
                     and points[:, 2].max() < BIN["rim_height_m"] for points in (outer, solid))
        released = max(self.contact_loads(check=False)[1:]) < .05
        still = max(displacement) < BIN["settled_speed_m_s"]
        carried = carry_y - self.original[1] >= .08
        clear_rim = not self.scene.is_flex or (min_rim_clearance is not None
                                              and min_rim_clearance >= BIN["clearance_over_rim_m"])
        success = lifted and carried and clear_rim and inside and released and still
        reason = next((key for key, ok in (("not_lifted", lifted), ("lost_in_transport", carried),
                                          ("insufficient_rim_clearance", clear_rim), ("outside_bin", inside),
                                          ("release_failed", released), ("not_settled", still)) if not ok), None)
        return dict(task_success=bool(success), failure_reason=reason, lifted=bool(lifted), carried=bool(carried),
                    inside=bool(inside), released=bool(released), still=bool(still),
                    peak_payload_lift_m=rise, lifting_dual_contact_frames=int(dual_frames),
                    final_payload_world=xyz.tolist(), peak_gripper_motor_effort_Nm=peak_gripper_effort,
                    min_transport_clearance_m=min_rim_clearance)


class CPURigidDrive:
    """Native MuJoCo rigid-only smoke driver with the Yam_B nominal motor contract."""

    name = "mujoco_cpu"

    def __init__(self, model, cfg, motor_Nm_per_N):
        self.model = model
        self.data = initial_state(model, cfg)
        self.initial = self.data.qpos.copy()
        joints, self.qadr, self.dofs = arm_indices(model)
        self.gear = motor_Nm_per_N
        self.preload = 2 * model.jnt_range[joints[6], 0]
        self.kp = np.array([*cfg["kp"], cfg["gripper_kp_N_m"]])
        self.kd = np.array([*cfg["kd"], cfg["gripper_kd_N_s_m"]])
        self.caps = np.array([*cfg["torque_limit_Nm"], cfg["gripper_force_limit_N"]], dtype=float)
        self.speed = np.array([*cfg["joint_target_speed_rad_s"], cfg["gripper_speed_m_s"]])
        self.gravity_factor = np.array(cfg["gravity_factor"])
        self.max_gap = cfg["gripper_max_gap_m"]
        self.dt = cfg["physics_dt_s"]
        self.motor_steps = round(1. / (self.dt * cfg["actuator_hz"]))
        self.command_steps = round(1. / (self.dt * cfg["command_hz"]))
        text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "gravity_bodies")
        gravity_text = model.text_data[model.text_adr[text_id]:].split(b"\x00", 1)[0].decode()
        gravity_ids = [model.body(name).id for name in gravity_text.split()]
        mask = np.array([[body in descendants(model, model.body(model.jnt_bodyid[j]).name) for body in gravity_ids]
                         for j in joints[:6]])
        self.links = np.array(gravity_ids)
        self.weights = mask * model.body_mass[self.links][None]
        self.joints = joints[:6]

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.target = np.r_[q[self.qadr[:6]], q[self.qadr[6:]].sum()]
        self.command = np.r_[np.zeros(6), 2 * self.target[-1] / self.max_gap - 1]
        mujoco.mj_forward(self.model, self.data)
        self.stopped = False
        self._read()

    def _read(self):
        d = self.data
        self.timestamp = float(d.time)
        self.observation = np.r_[d.qpos[self.qadr[:6]], d.qvel[self.dofs[:6]], d.actuator_force[:6], self.target[:6],
                                 d.qpos[self.qadr[6:]].sum(), d.qvel[self.dofs[6:]].sum(),
                                 d.actuator_force[6] * self.gear, self.target[-1]]

    def stop(self):
        self.stopped = True
        self.data.ctrl[:] = 0.

    def tick(self, command, effort, check, phase):
        if self.stopped:
            raise RuntimeError("Drive is stopped until reset")
        m, d = self.model, self.data
        self.command = np.r_[np.clip((command[:6] - self.target[:6]) / (.04 * self.speed[:6]), -1, 1),
                             2 * command[-1] / self.max_gap - 1]
        caps = self.caps.copy()
        caps[-1] = effort / self.gear
        slew = self.dt * self.command_steps * self.speed
        start = d.time
        for i in range(round(.04 / self.dt)):
            if i % self.command_steps == 0:
                self.target += np.clip(command - self.target, -slew, slew)
            mujoco.mj_step1(m, d)
            if i % self.motor_steps == 0:
                q = np.r_[d.qpos[self.qadr[:6]], d.qpos[self.qadr[6:]].sum()]
                dq = np.r_[d.qvel[self.dofs[:6]], d.qvel[self.dofs[6:]].sum()]
                target = self.target.copy()
                target[-1] += self.preload * (1 - target[-1] / self.max_gap)
                levers = d.xipos[self.links][None] - d.xanchor[self.joints, None]
                forces = self.weights[..., None] * m.opt.gravity
                gravity = -np.einsum("ij,ij->i", np.cross(levers, forces).sum(axis=1), d.xaxis[self.joints])
                torque = self.kp * (target - q) - self.kd * dq
                torque[:6] += gravity * self.gravity_factor
                d.ctrl[:] = np.clip(torque, -caps, caps)
            mujoco.mj_step2(m, d)
            if (d.time <= start or any(d.warning[index].number for index in (
                    mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
                    mujoco.mjtWarning.mjWARN_CNSTRFULL, mujoco.mjtWarning.mjWARN_CONTACTFULL))):
                raise MotionStop("solver_warning", physics_valid=False)
            if d.ncon:
                check()
        self._read()
        if not np.isfinite(self.observation).all():
            raise MotionStop("nonfinite_physics", physics_valid=False)
