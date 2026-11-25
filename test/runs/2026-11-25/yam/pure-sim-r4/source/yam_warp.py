"""GPU-resident SimYAMInterface for MuJoCo Warp 3.17."""

from math import ceil

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from . import yam_warp_kernels as kernels
from .yam_history import History
from .yam_interface import observation_fields
from .yam_model import ARM_JOINTS, arm_indices, build_yam_model, initial_state, load_config


CONTROL_RANGES = (
    "kp_scale", "kd_scale", "effort_limit_scale", "gripper_gain_scale", "gripper_effort_limit_scale",
    "feedback_gain", "feedback_bias_Nm", "feedback_noise_std_Nm", "actuator_response_s",
    "feedback_lowpass_s", "feedback_quantization_Nm", "observation_delay_s",
)
BATCH_FIELDS = (
    "body_mass", "body_inertia", "body_subtreemass", "dof_frictionloss", "dof_armature", "geom_friction",
    "dof_invweight0", "body_invweight0", "actuator_acc0",
)
HARD_EFFORT_CAPS = np.array([28, 28, 28, 10, 10, 10, 50])


class SimYAMInterface:
    """GPU control and history, with scene assembly and substep coupling hooks.

    Every call uses the current Warp stream. Returned device views are live.
    Explicit snapshot/diagnostic methods are the only host-side physics reads.
    """

    def __init__(self, worlds=1, *, config=None, randomization=False, randomization_config=None,
                 seed=0, try_capacity=0, device="cuda:0", add_scene=None, before_solve=None,
                 capacities=None, capture=True, compile_model=mujoco.MjModel.from_xml_string, initialize=None):
        if worlds < 1 or try_capacity < 0:
            raise ValueError("worlds must be positive and try_capacity nonnegative")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("SimYAMInterface requires CUDA")
        self.worlds = worlds
        self.seed = seed
        self.reset_count = 0
        self.before_solve = before_solve
        self.capture = capture
        self.graph = None
        self.config = cfg = load_config("nominal", config)
        self.ranges = load_config("randomization", randomization_config)
        self.randomization = randomization
        self._check_config(cfg)
        self.dt = cfg["physics_dt_s"]
        self.substeps, self.motor_stride, self.command_stride, self.sensor_stride = (
            round(1.0 / (self.dt * cfg[f"{name}_hz"]))
            for name in ("policy", "actuator", "command", "observation")
        )
        self.policy_period = self.substeps * self.dt
        self.sensor_period = self.sensor_stride * self.dt
        self.xml, self.model = build_yam_model(cfg, add_scene=add_scene, compile_model=compile_model)
        data = initial_state(self.model, cfg)
        if initialize is not None:
            initialize(self.model, data)
        m = self.model
        with wp.ScopedDevice(self.device):
            self.gpu_model = mjw.put_model(m, {name: worlds for name in BATCH_FIELDS} if randomization else None)
            self.data = mjw.put_data(m, data, nworld=worlds, **(capacities or {}))
            self.initial = wp.array(data.qpos[None, :], dtype=float)
            self.robot = self._geometry()
            self.cmds = kernels.Commands()
            for item in ("requested", "target", "velocity", "desired_velocity", "feedforward", "action"):
                setattr(self.cmds, item, wp.zeros((worlds, 7), dtype=float))
            self.motors = kernels.Actuators()
            self.motors.tuning = wp.empty((worlds, 7), dtype=kernels.Motor)
            for item in ("command", "applied", "feedback"):
                setattr(self.motors, item, wp.zeros((worlds, 7), dtype=float))
            self.nominal = wp.array(np.array([
                [*cfg["kp"], cfg["gripper_kp_N_m"]],
                [*cfg["kd"], cfg["gripper_kd_N_s_m"]],
                [*cfg["torque_limit_Nm"], cfg["gripper_force_limit_N"]],
            ]).T, dtype=wp.vec3)
            self.mass = wp.array(m.body_mass, dtype=float)
            self.inertia = wp.array(m.body_inertia, dtype=wp.vec3)
            self.friction = wp.array(m.dof_frictionloss, dtype=float)
            self.armature = wp.array(m.dof_armature, dtype=float)
            self.pad_friction = wp.array(m.geom_friction, dtype=wp.vec3)
            self.body_ids = wp.array(self.body_ids, dtype=int)
            self.pad_ids = wp.array([m.geom(f"{side}_pad").id for side in ("left", "right")], dtype=int)
            enabled = [(1., 1.)] * 6 + [(0., 0.)] * 6
            if randomization:
                enabled = [self.ranges[key] for key in CONTROL_RANGES]
            self.control_ranges = wp.array(enabled, dtype=wp.vec2)
            slots = 1 + ceil(enabled[-1][1] / self.sensor_period)
            self.sensor = wp.zeros((worlds, slots, 3, 7), dtype=float)
            self.sensor_time = wp.zeros((worlds, slots), dtype=float)
            self.clock = wp.zeros(worlds, dtype=int)
            self.delay = wp.zeros_like(self.clock)
            self.obs = wp.zeros((worlds, 28), dtype=float)
            self.measurement_timestamp = wp.zeros(worlds, dtype=float)
            self.sample_action = wp.zeros((worlds, 7), dtype=float)
            self.sample_velocity = wp.zeros((worlds, 6), dtype=float)
            self.zero = wp.zeros((worlds, 7), dtype=float)
            self.all_worlds = wp.full(worlds, True, dtype=bool)
            self.history = History(worlds, cfg["history_length"], try_capacity, self.device)
        self.reset()

    def _check_config(self, cfg):
        clocks = np.array([cfg["physics_dt_s"], *(cfg[f"{key}_hz"] for key in
                          ("policy", "actuator", "command", "observation"))])
        if not np.isfinite(clocks).all() or (clocks <= 0).any():
            raise ValueError("Expected finite positive clock rates and timestep")
        for name, cap in (("kp", 500), ("kd", 5), ("gravity_factor", np.inf),
                          ("joint_frictionloss_Nm", np.inf)):
            values = np.asarray(cfg[name])
            if (values.shape != (6,) or not np.isfinite(values).all() or
                    (values < 0).any() or (values > cap).any()):
                raise ValueError(f"Invalid six-joint {name} configuration")
        rates = [1 / (cfg["physics_dt_s"] * cfg[f"{key}_hz"])
                 for key in ("policy", "actuator", "command", "observation")]
        if any(rate < 1 or abs(rate - round(rate)) > 1e-8 for rate in rates):
            raise ValueError("Every rate must divide the physics clock exactly")
        p, a, c, o = map(round, rates)
        if any(p % stride for stride in (a, c, o)) or c % a:
            raise ValueError("Each update clock must divide a policy period; commands align with actuator ticks")
        caps = np.r_[cfg["torque_limit_Nm"], cfg["gripper_force_limit_N"]]
        speed = np.asarray(cfg["joint_target_speed_rad_s"])
        if (caps.shape != (7,) or speed.shape != (6,) or not np.isfinite(caps).all()
                or (caps <= 0).any() or (caps > HARD_EFFORT_CAPS).any()
                or not np.isfinite(speed).all() or (speed <= 0).any()
                or (speed > [10, 10, 10, 30, 30, 30]).any()):
            raise ValueError("Expected six positive joint speeds and efforts within the SDK envelopes")
        if (not 0 < cfg["ik_damping"] or not 0 < cfg["gripper_max_gap_m"] <= .095
                or not 0 <= cfg["ready_gap_m"] <= cfg["gripper_max_gap_m"]
                or not 0 < cfg["gripper_speed_m_s"] <= 30 * cfg["gripper_sdk_stroke_m"] / cfg["gripper_motor_travel_rad"]
                or cfg["history_length"] < 1):
            raise ValueError("Invalid IK, history or gripper configuration")
        for name, bounds in self.ranges.items():
            lo, hi = bounds
            if not np.isfinite(bounds).all() or lo > hi:
                raise ValueError(f"Invalid DR interval {name}")
            if (name != "feedback_bias_Nm" and lo < 0) or (name.endswith("_scale") and lo == 0):
                raise ValueError(f"Invalid lower bound for {name}")
            if name.endswith("effort_limit_scale") and hi > 1:
                raise ValueError(f"{name} exceeds configured hardware cap")

    def _geometry(self):
        m, cfg = self.model, self.config
        joints, pos, dof = arm_indices(m)
        robot = kernels.Geometry()
        robot.joints = wp.array(joints, dtype=int)
        robot.pos = wp.array(pos, dtype=int)
        robot.dof = wp.array(dof, dtype=int)
        robot.actuators = wp.array([m.actuator(n).id for n in (*ARM_JOINTS, "gripper")], dtype=int)
        limits = m.jnt_range[joints[:6]]
        robot.lower = wp.array(np.r_[limits[:, 0], 0.], dtype=float)
        robot.upper = wp.array(np.r_[limits[:, 1], cfg["gripper_max_gap_m"]], dtype=float)
        robot.speed = wp.array([*cfg["joint_target_speed_rad_s"], cfg["gripper_speed_m_s"]], dtype=float)
        robot.gravity_factor = wp.array(cfg["gravity_factor"], dtype=float)
        robot.gravity = wp.vec3(*m.opt.gravity)
        robot.site = m.site("grasp_site").id
        robot.max_gap = cfg["gripper_max_gap_m"]
        robot.min_drive_gap = 2 * m.jnt_range[joints[6], 0]
        robot.motor_Nm_per_N = cfg["gripper_sdk_stroke_m"] / cfg["gripper_motor_travel_rad"]

        ancestor = np.zeros((m.nbody, m.nbody), dtype=bool)
        for child in range(1, m.nbody):
            parent = child
            while parent:
                ancestor[parent, child] = True
                parent = m.body_parentid[parent]
        self.body_ids = np.flatnonzero(ancestor[m.body("arm").id] & (m.body_mass > 0))
        text_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, "gravity_bodies")
        text = m.text_data[m.text_adr[text_id]:].split(b"\x00", 1)[0].decode()
        compensated = [m.body(name).id for name in text.split()]
        gravity_ids = np.full((6, len(compensated)), -1, dtype=np.int32)
        for row, jnt in enumerate(joints[:6]):
            contributing = [b for b in compensated if ancestor[m.jnt_bodyid[jnt], b]]
            gravity_ids[row, :len(contributing)] = contributing
        robot.gravity_bodies = wp.array(gravity_ids, dtype=int)
        return robot

    def _launch(self, kernel, shape, *args):
        wp.launch(kernel, shape, list(args), device=self.device)

    def _mask(self, mask):
        if mask is None:
            return self.all_worlds
        if mask.shape != (self.worlds,) or mask.dtype != wp.bool or mask.device != self.device:
            raise ValueError("reset mask must be a device bool array of shape [N]")
        return mask

    def reset(self, mask=None, *, initial_qpos=None):
        """Reset selected worlds, optionally with a full scene configuration per world."""
        mask = self._mask(mask)
        initial = self.initial if initial_qpos is None else self._input(initial_qpos, self.model.nq)
        self.reset_count += 1
        seed = self.seed + 65537 * self.reset_count
        m, d = self.gpu_model, self.data
        with wp.ScopedDevice(self.device):
            mjw.reset_data(m, d, mask)
            self._launch(kernels.randomize_controls, (self.worlds, 7), mask, seed, self.sensor_period,
                         self.nominal, self.control_ranges, self.motors.tuning, self.delay)
            if self.randomization:
                vec = lambda key: wp.vec2(*self.ranges[key])
                self._launch(kernels.randomize_links, (self.worlds, self.body_ids.size), mask, seed + 11,
                             self.body_ids, vec("link_mass_scale"), self.mass, self.inertia, m.body_mass, m.body_inertia)
                self._launch(kernels.randomize_dofs, (self.worlds, 6), mask, seed + 29, self.robot,
                             vec("joint_friction_scale"), vec("armature_scale"), self.friction, self.armature,
                             m.dof_frictionloss, m.dof_armature)
                self._launch(kernels.randomize_pads, (self.worlds, 2), mask, seed + 43, self.pad_ids,
                             vec("fingertip_friction_scale"), self.pad_friction, m.geom_friction)
                mjw.set_const(m, d, restore=False)
            self._launch(kernels.reset_controls, self.worlds, self.robot, self.cmds, self.motors, d.qpos, initial,
                         mask, self.clock, self.sensor, self.sensor_time, self.history.count, self.history.try_count, self.history.try_active)
            self._observe()
            mjw.kinematics(m, d)

    def _input(self, value, width):
        if not isinstance(value, wp.array):
            value = np.asarray(value, dtype=np.float32)
            if value.shape == (width,) and self.worlds == 1:
                value = value[None]
            value = wp.array(value, dtype=float, device=self.device)
        if value.shape != (self.worlds, width) or value.dtype != wp.float32 or value.device != self.device:
            raise ValueError(f"Expected float32 control [{self.worlds}, {width}] on {self.device}")
        return value

    def send_joint_command(self, position, *, velocity=None, feedforward=None):
        """Stage six target angles and full gap, with optional six rates/torques.

        Optional device rate/torque arguments are [N,7]; channel 7 is the
        equivalent gap velocity/force. Default is zero in all channels.
        """
        self._launch(kernels.joint_targets, (self.worlds, 7), self.robot, self.cmds, self._input(position, 7),
                     self.zero if velocity is None else self._input(velocity, 7),
                     self.zero if feedforward is None else self._input(feedforward, 7), self.policy_period)

    def send_cartesian_action(self, action):
        """Stage world-frame XYZ / rotation-vector increment and absolute gap action in [-1,1]."""
        d, cfg = self.data, self.config
        self._launch(kernels.cartesian_targets, self.worlds, self.robot, self.cmds, self._input(action, 7),
                     self.policy_period, wp.vec2(cfg["translation_scale_m"], cfg["rotation_scale_rad"]),
                     wp.vec3(*cfg["workspace_lower_m"]), wp.vec3(*cfg["workspace_upper_m"]), cfg["ik_damping"],
                     d.qpos, d.site_xpos, d.xanchor, d.xaxis)

    def _observe(self):
        self._launch(kernels.observation, self.worlds, self.cmds, self.clock, self.delay,
                     self.sensor, self.sensor_time, self.obs, self.measurement_timestamp)
        wp.copy(self.sample_action, self.cmds.action)
        wp.copy(self.sample_velocity, self.cmds.desired_velocity[:, :6])

    def _policy_step(self):
        m, d = self.gpu_model, self.data
        for step in range(self.substeps):
            if step % self.command_stride == 0:
                self._launch(kernels.slew_targets, (self.worlds, 7), self.robot, self.cmds, self.dt * self.command_stride)
            mjw.step1(m, d)
            if step % self.motor_stride == 0:
                self._launch(kernels.motor_control, (self.worlds, 7), self.robot, self.cmds, self.motors,
                             self.dt * self.motor_stride, d.qpos, d.qvel, d.ctrl, m.body_mass, d.xipos, d.xanchor, d.xaxis)
            if self.before_solve is not None:
                self.before_solve(m, d)
            mjw.step2(m, d)
            if (step + 1) % self.sensor_stride == 0:
                self._launch(kernels.read_motors, (self.worlds, 7), self.robot, self.motors,
                             self.sensor_period, self.seed, d.qpos, d.qvel, d.actuator_force, self.clock, d.time,
                             self.sensor, self.sensor_time)
                self._launch(kernels.advance_clock, self.worlds, self.clock)
        mjw.kinematics(m, d)
        self._observe()
        self.history.record(self.obs, self.sample_action, self.measurement_timestamp)

    def step(self):
        """Advance one policy period using the currently staged command."""
        with wp.ScopedDevice(self.device):
            if not self.capture:
                self._policy_step()
                return
            if self.graph is None:
                with wp.ScopedCapture() as capture:
                    self._policy_step()
                self.graph = capture.graph
            wp.capture_launch(self.graph)

    def get_observation(self, *, include_effort=True):
        return observation_fields(self.obs, include_effort=include_effort)

    def get_command_history_sample(self):
        return dict(previous_action=self.sample_action, joint_vel_cmd=self.sample_velocity,
                    timestamp=self.data.time, measurement_timestamp=self.measurement_timestamp)

    def get_diagnostics(self):
        return dict(tau_command=self.motors.command[:, :6], tau_actuator=self.motors.applied[:, :6],
                    tau_feedback=self.obs[:, 12:18], qfrc_constraint=self.data.qfrc_constraint,
                    gripper_force_command_N=self.motors.command[:, 6:7],
                    gripper_force_applied_N=self.motors.applied[:, 6:7])

    def snapshot(self, world=0):
        """Copy one world for offline contact, video and solver diagnostics."""
        data = mujoco.MjData(self.model)
        mjw.get_data_into(data, self.model, self.data, world_id=world)
        return data

    def contact_torque(self, world=0, *, snapshot=None):
        """Contact rows projected onto the six arm joints, in Nm."""
        m = self.model
        d = self.snapshot(world) if snapshot is None else snapshot
        torque = np.zeros(m.nv)
        if d.nefc:
            first_contact_type = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
            contact_force = np.where(d.efc_type >= first_contact_type, d.efc_force, 0.)
            mujoco.mj_mulJacTVec(m, d, torque, contact_force)
        _, _, dof = arm_indices(m)
        return torque[dof[:6]]
