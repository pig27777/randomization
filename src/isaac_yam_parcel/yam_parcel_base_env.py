"""Direct 25 Hz ABC joint control and 2 ms PhysX/parcel coupling."""

import math

import torch
from torch.utils._pytree import tree_map
from isaaclab import sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab_physx.physics import PhysxCfg

from src.low_order_parcel.domain_randomizer import DomainRandomizer
from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import ParcelAction
from src.low_order_parcel.unified_payload import DT
from src.low_order_parcel.virtual_rgb import VirtualRGBProjector

from .joint_current_filter import JointCurrentEstimatorFilter, JointCurrentParams
from .yam_scene import abc_scene_cfg, load_scene


JOINT_CHANNELS = 7
OBS_PER_ARM = 57


def yam_parcel_cfg(num_envs, *, dual_arm=False, seed=17, use_newton_actuators=False):
    scene = abc_scene_cfg(num_envs, dual_arm=dual_arm)
    scene.arm_a.class_type = "src.isaac_yam_parcel.yam_load_port:RailLoadedYam"
    if dual_arm:
        scene.arm_b.class_type = scene.arm_a.class_type
    arms = 2 if dual_arm else 1
    return DirectRLEnvCfg(
        sim=sim_utils.SimulationCfg(device="cuda:0", dt=DT, render_interval=20, physics=PhysxCfg(),
                                    use_newton_actuators=use_newton_actuators),
        scene=scene, decimation=20, seed=seed, episode_length_s=18.,
        action_space=arms * JOINT_CHANNELS, observation_space=arms * OBS_PER_ARM,
        state_space=arms * OBS_PER_ARM + 54 + 24,
    )


class YAMParcelBaseEnv(DirectRLEnv):
    arm_count = 1

    def __init__(self, cfg, *, episode=None, curriculum_alpha=1.):
        from .yam_load_port import YamLoadPort

        if cfg.action_space != self.arm_count * JOINT_CHANNELS or cfg.sim.dt != DT or cfg.decimation != 20:
            raise ValueError("Use 7 targets per arm, 2 ms physics, and 20:1 decimation.")
        super().__init__(cfg)
        if self._physics_handles_decimation:
            raise ValueError("The co-simulation must be called on each 2 ms substep.")
        self.render_enabled = self.sim.has_gui
        self.scene_definition = scene = load_scene()
        self.robots = tuple(self.scene.articulations.values())
        origin = self.scene.env_origins
        self.ports = tuple(YamLoadPort(robot, scene, origin) for robot in self.robots)
        self.randomizer = DomainRandomizer(self.device, cfg.seed)
        self.curriculum_alpha = curriculum_alpha
        self.fixed_episode = episode is not None
        self.episode = episode if self.fixed_episode else self.randomizer.sample(
            self.num_envs, curriculum_alpha=curriculum_alpha)
        self.parcel_sim = self.episode.make_sim()
        self.parcel_state = self.episode.table_state(self.parcel_sim)
        self.q_des = [robot.data.default_joint_pos.torch.clone() for robot in self.robots]
        self.step_limit = origin.new_tensor([.04] * 6 + [.008])
        self.joint_limits = [robot.data.soft_joint_pos_limits.torch for robot in self.robots]
        self.enabled = torch.arange(2, device=self.device).expand(self.num_envs, -1) < self.arm_count
        self.reaction = origin.new_zeros((self.num_envs, 2, 2, 3))
        self.empty_action = ParcelAction(self.reaction[:, 1].clone(), self.reaction[:, 1].clone(),
                                         *(origin.new_zeros((self.num_envs, 2)) for _ in range(4)),
                                         torch.zeros_like(origin), origin.new_zeros((self.num_envs, 1)))
        self.identity = origin.new_tensor([1., 0., 0., 0.]).expand(self.num_envs, -1)
        self.side = torch.zeros_like(self.enabled)
        self.feed_axis = torch.zeros_like(origin[:, None].expand(-1, 2, -1))
        self.side_feed = origin.new_zeros((self.num_envs, 2, 1))
        self.have_history = torch.zeros_like(self.enabled)
        self.hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.longest_hold = torch.zeros_like(self.hold)
        self.ever_captured = torch.zeros_like(self.hold, dtype=torch.bool)
        self.ever_airborne = torch.zeros_like(self.ever_captured)
        self.current_filter = JointCurrentEstimatorFilter(self._current_params(), generator=self.randomizer.rng)
        self.sensed = origin.new_zeros((self.num_envs, self.arm_count, 15))
        self.visual = origin.new_zeros((self.num_envs, self.arm_count, 21))
        self.height = origin.new_full((self.num_envs, self.arm_count), .045)
        size_x, size_y = scene["table"]["size_m"][:2]
        bounds = origin.new_tensor([-size_x / 2., size_x / 2., -size_y / 2., size_y / 2.]).expand(self.num_envs, -1)
        horizontal_fov = 2. * math.atan(4. / 3. * math.tan(math.radians(scene["top_camera"]["vertical_fov_deg"]) / 2.))
        self.projectors = [VirtualRGBProjector(
            self.parcel_sim.payload, port.frames.top_camera(self.num_envs), self.episode.rgb,
            table_bounds=bounds, top_horizontal_fov=horizontal_fov, generator=self.randomizer.rng)
            for port in self.ports]
        self.result = None
        self.markers = self._make_markers() if self.sim.has_gui else None

    def _current_params(self):
        like = self.scene.env_origins.new_empty((self.num_envs, self.arm_count, JOINT_CHANNELS))
        return JointCurrentParams.sample(like, generator=self.randomizer.rng)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        mask = torch.zeros_like(self.ever_captured)
        mask[env_ids] = True
        for i, robot in enumerate(self.robots):
            q = robot.data.default_joint_pos.torch[env_ids].clone()
            if not self.fixed_episode:
                q[:, :6] += .01 * torch.rand(q[:, :6].shape, device=self.device, generator=self.randomizer.rng) - .005
            self.q_des[i][env_ids] = q
            robot.rail_load[env_ids] = 0.
            robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        if not self.fixed_episode:
            self.episode = self.randomizer.resample(self.episode, mask, curriculum_alpha=self.curriculum_alpha)
            self.parcel_sim = self.episode.make_sim()
        fresh = self.episode.table_state(self.parcel_sim)

        def reset_tensor(new, old):
            return torch.where(mask.reshape(-1, *([1] * (old.ndim - 1))), new, old)

        self.parcel_state = tree_map(reset_tensor, fresh, self.parcel_state)
        for projector in self.projectors:
            projector.payload = self.parcel_sim.payload
            projector.reset(mask, params=self.episode.rgb)
        self.current_filter.reset(mask, self._current_params())
        for buffer in (self.hold, self.longest_hold, self.ever_captured, self.ever_airborne,
                       self.side, self.feed_axis, self.side_feed, self.have_history, self.sensed, self.visual):
            buffer[env_ids] = 0
        self.height[env_ids] = .045
        self.actions[env_ids] = 0.

    @staticmethod
    def motor_channels(dof_values, robot):
        rails = dof_values.index_select(1, robot.rail_indices)
        return torch.cat((dof_values[:, :6], rails.mean(dim=-1, keepdim=True)), dim=-1)

    def _pre_physics_step(self, actions):
        self.actions.copy_(actions.clamp(-1., 1.))
        for i, (robot, target, limits) in enumerate(zip(self.robots, self.q_des, self.joint_limits, strict=True)):
            q = self.motor_channels(robot.data.joint_pos.torch, robot)
            command = q + self.actions[:, i * JOINT_CHANNELS:(i + 1) * JOINT_CHANNELS] * self.step_limit
            target[:, :6] = command[:, :6].clamp(limits[:, :6, 0], limits[:, :6, 1])
            rail = command[:, 6:7].clamp(limits[:, robot.rail_ids, 0].amax(dim=-1, keepdim=True),
                                         limits[:, robot.rail_ids, 1].amin(dim=-1, keepdim=True))
            target[:, robot.rail_ids] = rail

    def _parcel_action(self, i, robot, port_state, dq, motor):
        """Construct physical skin-contact inputs from the pad and parcel geometry."""
        pads, velocities = port_state.pads, port_state.pad_velocity
        point = pads.mean(dim=1)
        velocity = velocities.mean(dim=1)
        payload = self.parcel_sim.payload
        parcel = self.parcel_state.payload
        frame = payload.frame(parcel)
        body = (point - parcel.pos.mean(dim=1)).unsqueeze(1) @ frame
        local_velocity = velocity.unsqueeze(1) @ frame
        horizontal = local_velocity[:, 0, :2]
        speed = horizontal.norm(dim=-1, keepdim=True)
        local_dir = horizontal / speed.clamp_min(1e-6)
        direction = (frame[:, :, :2] @ local_dir.unsqueeze(-1)).squeeze(-1)
        footprint = payload.params.bag_half_extents[:, :2]
        rim_margin = (footprint - body[:, 0, :2].abs()).amin(dim=-1, keepdim=True)
        height = payload.half_thickness(parcel.d_surf).amax(dim=1, keepdim=True)
        near_rim = (rim_margin >= -.015) & (rim_margin < .016)
        moves_in = (body[:, 0, :2] * horizontal).sum(dim=-1, keepdim=True) < 0.
        side_allowed = self.episode.clearance_m.amax(dim=1, keepdim=True) >= .020
        enters = ((speed > .012) & (speed > 2. * local_velocity[:, 0, 2:].abs()) & moves_in
                  & near_rim & ((body[:, 0, 2:] - height).abs() < .012) & side_allowed)
        was_side = self.side[:, i:i + 1]
        entering = enters & ~was_side
        self.feed_axis[:, i] = torch.where(entering, direction, self.feed_axis[:, i])
        approach = (velocity * self.feed_axis[:, i]).sum(dim=-1, keepdim=True)
        inward = dq.index_select(1, robot.rail_indices)
        opens = inward.mean(dim=-1, keepdim=True) < -.025
        side = (was_side | enters) & ~opens
        self.side[:, i:i + 1] = side
        self.side_feed[:, i] = torch.where(side, (self.side_feed[:, i] + approach * DT).clamp_min(0.), 0.)
        # The available normal force comes from the real simulated motor channel.
        # A closing PD effort is separated from external rail reactions by the port.
        normal = motor.index_select(1, robot.rail_indices).clamp_min(0.)
        return ParcelAction(pads, velocities, normal, inward, self.side_feed[:, i].expand_as(normal),
                            side.expand_as(normal).to(dq.dtype), self.empty_action.external_force,
                            self.empty_action.direct_press_N)

    def _apply_action(self):
        kinematics = [port.read() for port in self.ports]
        q = [robot.data.joint_pos.torch for robot in self.robots]
        dq = [robot.data.joint_vel.torch for robot in self.robots]
        motor = [robot.actuators.applied_effort.torch for robot in self.robots]
        cmds, quats, rates = [], [], []
        for i, (robot, sample) in enumerate(zip(self.robots, kinematics, strict=True)):
            robot.actuators.target_command.set_position_index(value=self.q_des[i], full_data=True)
            cmds.append(self._parcel_action(i, robot, sample, dq[i], motor[i]))
            quat = sample.wrist_quat_xyzw
            quats.append(torch.cat((quat[:, 3:], quat[:, :3]), dim=-1))
            rates.append(sample.wrist_omega)
        if self.arm_count == 1:
            cmds.append(self.empty_action)
            quats.append(self.identity)
            rates.append(self.empty_action.external_force)

        def articulated_mobility(r, direction):
            blocks = [port.mobility(sample, r[:, 2 * arm:2 * arm + 2], direction[:, 2 * arm:2 * arm + 2])
                      for arm, (port, sample) in enumerate(zip(self.ports, kinematics, strict=True))]
            zeros = torch.zeros_like(blocks[0])
            second = zeros if self.arm_count == 1 else blocks[1]
            return torch.cat((torch.cat((blocks[0], zeros), dim=-1), torch.cat((zeros, second), dim=-1)), dim=-2)

        self.result = self.parcel_sim.step(
            self.parcel_state, DualArmAction(*cmds, self.enabled, self.side),
            jaw_orientation=torch.stack(quats, dim=1), jaw_angular_velocity=torch.stack(rates, dim=1),
            pad_reaction_out=self.reaction, neck_port_mobility=articulated_mobility)
        self.parcel_state = self.result.state
        self._track_hold()
        for i, (port, sample) in enumerate(zip(self.ports, kinematics, strict=True)):
            port.write(sample, self.result.wrist_wrench[:, i], self.reaction[:, i])
        self._observe_ports(kinematics, q, dq, motor)
        if self.markers is not None and self._sim_step_counter % self.cfg.decimation == 0:
            self._draw_parcel()

    def _observe_ports(self, kinematics, q, dq, motor):
        def dofs_to_motor(values):
            return torch.stack([self.motor_channels(value, robot)
                                for value, robot in zip(values, self.robots, strict=True)], dim=1)

        opening = torch.stack([(s.pads[:, 1:] - s.pads[:, :1]).norm(dim=-1) for s in kinematics], dim=1)
        obs = self.current_filter.step(dofs_to_motor(motor), dofs_to_motor(q), dofs_to_motor(dq),
                                       dofs_to_motor(self.q_des), opening)
        self.sensed = torch.cat(tuple(obs), dim=-1)
        for i, (projector, sample) in enumerate(zip(self.projectors, kinematics, strict=True)):
            center = sample.pads.mean(dim=1)
            up = (sample.pad_velocity[..., 2].mean(dim=1) > .004) & self.have_history[:, i]
            travel = (.90 * self.height[:, i:i + 1]).clamp(.04, .10)
            rgb = projector.step(self.parcel_state.payload, sample.wrist_camera, center,
                                 up.unsqueeze(-1), lift_travel_m=travel)
            self.height[:, i:i + 1] = torch.where(rgb.wrist_vis_valid, rgb.wrist_profile[:, :1],
                                                   self.height[:, i:i + 1])
            self.visual[:, i] = torch.cat(rgb, dim=-1)
            self.have_history[:, i] = True

    def _track_hold(self):
        both = (self.parcel_state.contacts_A[..., ContactState.CAPTURE] == 1.).all(dim=1)
        airborne = (self.result.table_clearance[:, 0] > .001) & (self.result.support_N[:, 0] == 0.)
        quiet = self.parcel_sim.payload.com_velocity(self.parcel_state.payload).norm(dim=1) < .020
        intact = ~self.parcel_state.torn_flag.flatten(1).any(dim=1)
        self.ever_captured |= both
        self.ever_airborne |= airborne & both
        self.hold = torch.where(airborne & both & quiet & intact, self.hold + 1, 0)
        torch.maximum(self.longest_hold, self.hold, out=self.longest_hold)

    def _get_observations(self):
        q = torch.stack([self.motor_channels(robot.data.joint_pos.torch, robot) for robot in self.robots], dim=1)
        dq = torch.stack([self.motor_channels(robot.data.joint_vel.torch, robot) for robot in self.robots], dim=1)
        actor = torch.cat((q, dq, self.sensed, self.actions.unflatten(1, (self.arm_count, JOINT_CHANNELS)),
                           self.visual), dim=-1).flatten(1)
        contacts = torch.cat((self.parcel_state.contacts_A.flatten(1), self.parcel_state.contacts_B.flatten(1)), dim=1)
        return {"policy": actor, "critic": torch.cat((actor, self.parcel_sim.get_privileged_state(self.parcel_state),
                                                     contacts), dim=1)}

    def _get_rewards(self):
        return (self.hold >= 1500).to(torch.float32)

    def _get_dones(self):
        torn = self.parcel_state.torn_flag.flatten(1).any(dim=1)
        return torn, self.episode_length_buf >= self.max_episode_length

    def _make_markers(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(prim_path="/Visuals/Parcel", markers={
            "block": sim_utils.CuboidCfg(size=(1., 1., 1.)),
            "envelope": sim_utils.CuboidCfg(
                size=(1., 1., 1.), visual_material=sim_utils.PreviewSurfaceCfg(opacity=.2)),
        })
        return VisualizationMarkers(cfg)

    def _draw_parcel(self):
        from isaaclab.utils.math import quat_from_matrix

        model, parcel = self.parcel_sim.payload, self.parcel_state.payload
        vertices = self.projectors[0].vertices(parcel)
        lower, upper = vertices.amin(dim=1), vertices.amax(dim=1)
        center = torch.cat((parcel.pos, (.5 * (lower + upper)).unsqueeze(1)), dim=1)
        sizes = torch.cat((model.params.bag_half_extents[:, None, :2].expand(-1, 4, -1),
                           2. * model.half_thickness(parcel.d_surf).unsqueeze(-1)), dim=-1)
        sizes = torch.cat((sizes, (upper - lower).unsqueeze(1)), dim=1)
        quat = quat_from_matrix(model.frame(parcel))
        identity = quat.new_tensor([0., 0., 0., 1.]).expand(self.num_envs, 1, -1)
        self.markers.visualize((center + self.scene.env_origins[:, None]).flatten(0, 1),
                               torch.cat((quat[:, None].expand(-1, 4, -1), identity), dim=1).flatten(0, 1),
                               sizes.flatten(0, 1), [0, 0, 0, 0, 1] * self.num_envs)


class YAMPickEnv(YAMParcelBaseEnv):
    """One ABC YAM works with the parcel positioned on the table."""


class YAMHandoverPlaceEnv(YAMParcelBaseEnv):
    """Two ABC YAMs share the same deformable parcel and reserve budget."""

    arm_count = 2
