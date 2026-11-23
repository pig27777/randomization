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
from .yam_scene import abc_scene_cfg, load_scene, load_workcell


JOINT_CHANNELS = 7
OBS_PER_ARM = 57
PLACE_OBS = 4


def yam_parcel_cfg(num_envs, *, dual_arm=False, seed=17, use_newton_actuators=False):
    # The left arm remains physically in the same workcell during Policy 1.
    scene = abc_scene_cfg(num_envs, dual_arm=True)
    scene.arm_a.class_type = "src.isaac_yam_parcel.yam_load_port:RailLoadedYam"
    scene.arm_b.class_type = scene.arm_a.class_type
    arms = 2 if dual_arm else 1
    obs = arms * OBS_PER_ARM + (PLACE_OBS if dual_arm else 0)
    return DirectRLEnvCfg(
        sim=sim_utils.SimulationCfg(device="cuda:0", dt=DT, render_interval=20, physics=PhysxCfg(),
                                    use_newton_actuators=use_newton_actuators),
        scene=scene, decimation=20, seed=seed, episode_length_s=18.,
        action_space=arms * JOINT_CHANNELS, observation_space=obs, state_space=obs + 54 + 24,
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
        self.workcell = load_workcell()
        self.scene_robots = (self.scene["arm_a"], self.scene["arm_b"])
        self.robots = self.scene_robots[:self.arm_count]
        origin = self.scene.env_origins
        self.ports = tuple(YamLoadPort(robot, scene, origin) for robot in self.robots)
        self.randomizer = DomainRandomizer(self.device, cfg.seed)
        self.curriculum_alpha = curriculum_alpha
        self.fixed_episode = episode is not None
        self.episode = episode if self.fixed_episode else self.randomizer.sample(
            self.num_envs, curriculum_alpha=curriculum_alpha)
        self.parcel_sim = self.episode.make_sim()
        pick = self.workcell["right_pick"]
        self.pick_center = origin.new_tensor(pick["center_m"])
        self.pick_half = origin.new_tensor(pick["sampling_half_xy_m"])
        place = self.workcell["left_place"]
        self.place_center = origin.new_tensor(place["center_m"][:2])
        self.place_half = origin.new_tensor(place["half_xy_m"])
        handover = self.workcell["center_handover"]
        self.handover_center = origin.new_tensor(handover["center_m"])
        self.handover_half = origin.new_tensor(handover["half_extents_m"])
        self.parcel_state = self._parcel_table_state()
        self.q_des = [robot.data.default_joint_pos.torch.clone() for robot in self.scene_robots]
        self.gravity_feedforward = [torch.zeros_like(target) for target in self.q_des]
        self.step_limit = origin.new_tensor([.04] * 6 + [.008])
        self.joint_limits = [robot.data.soft_joint_pos_limits.torch for robot in self.robots]
        self.enabled = torch.arange(2, device=self.device).expand(self.num_envs, -1) < self.arm_count
        self.reaction = origin.new_zeros((self.num_envs, 2, 2, 3))
        self.fold_reaction = torch.zeros_like(self.reaction)
        self.fold_work = origin.new_zeros((self.num_envs, 2, 1))
        self.empty_action = ParcelAction(self.reaction[:, 1].clone(), self.reaction[:, 1].clone(),
                                         *(origin.new_zeros((self.num_envs, 2)) for _ in range(4)),
                                         torch.zeros_like(origin), origin.new_zeros((self.num_envs, 1)))
        self.identity = origin.new_tensor([1., 0., 0., 0.]).expand(self.num_envs, -1)
        self.side = torch.zeros_like(self.enabled)
        self.entered_above_lip = torch.zeros_like(self.enabled)
        self.feed_axis = torch.zeros_like(origin[:, None].expand(-1, 2, -1))
        self.side_feed = origin.new_zeros((self.num_envs, 2, 1))
        self.net_side_feed = torch.zeros_like(self.side_feed)
        self.top_press = torch.zeros_like(self.side_feed)
        self.film_load = torch.zeros_like(self.side_feed)
        self.pad_normal = origin.new_zeros((self.num_envs, 2, 2))
        self.have_history = torch.zeros_like(self.enabled)
        self.hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.longest_hold = torch.zeros_like(self.hold)
        self.ever_captured = torch.zeros_like(self.hold, dtype=torch.bool)
        self.ever_airborne = torch.zeros_like(self.ever_captured)
        self.handover_seen = torch.zeros_like(self.ever_captured)
        self.left_carrying = torch.zeros_like(self.ever_captured)
        self.placed_supported = torch.zeros_like(self.ever_captured)
        self.place_streak = torch.zeros_like(self.hold)
        gripper = self.workcell["gripper"]
        self.rail_close = float(gripper["close_calibration_m"])
        self.film_gap = float(gripper["film_gap_m"])
        self.side_fold_gap = float(gripper["side_fold_gap_m"])
        self.side_fold_damping = float(gripper["side_fold_damping_ns_m"])
        self.rail_overtravel = float(gripper["max_preload_n"]) / self.scene_robots[0].cfg.actuators["rails"].stiffness
        # Align the simulation's closing stop to the CAD zero-gap position.
        # This one-time property write keeps the real source USD and 2 ms loop untouched.
        for robot in self.scene_robots:
            rails = robot.data.joint_pos_limits.torch.index_select(1, robot.rail_indices)
            rails[..., 1] = self.rail_close
            robot.write_joint_position_limit_to_sim_index(limits=rails, joint_ids=robot.rail_indices)
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

    def _parcel_table_state(self):
        state = self.episode.table_state(self.parcel_sim)
        shift = self.pick_center.expand(self.num_envs, -1).clone()
        if not self.fixed_episode:
            rank = torch.rand((self.num_envs, 2), device=self.device, generator=self.randomizer.rng)
            shift[:, :2] += (2. * rank - 1.) * self.pick_half
        return state._replace(payload=state.payload._replace(pos=state.payload.pos + shift[:, None]),
                              side_origin=state.side_origin + shift[:, None])

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        mask = torch.zeros_like(self.ever_captured)
        mask[env_ids] = True
        for i, robot in enumerate(self.scene_robots):
            q = robot.data.default_joint_pos.torch[env_ids].clone()
            if not self.fixed_episode and i < self.arm_count:
                q[:, :6] += .01 * torch.rand(q[:, :6].shape, device=self.device, generator=self.randomizer.rng) - .005
            self.q_des[i][env_ids] = q
            robot.rail_load[env_ids] = 0.
            robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        if not self.fixed_episode:
            self.episode = self.randomizer.resample(self.episode, mask, curriculum_alpha=self.curriculum_alpha)
            self.parcel_sim = self.episode.make_sim()
        fresh = self._parcel_table_state()

        def reset_tensor(new, old):
            return torch.where(mask.reshape(-1, *([1] * (old.ndim - 1))), new, old)

        self.parcel_state = tree_map(reset_tensor, fresh, self.parcel_state)
        for projector in self.projectors:
            projector.payload = self.parcel_sim.payload
            projector.reset(mask, params=self.episode.rgb)
        self.current_filter.reset(mask, self._current_params())
        for buffer in (self.hold, self.longest_hold, self.ever_captured, self.ever_airborne,
                       self.handover_seen, self.left_carrying, self.placed_supported, self.place_streak,
                       self.side, self.entered_above_lip, self.feed_axis, self.side_feed, self.net_side_feed, self.top_press,
                       self.film_load, self.pad_normal, self.fold_work,
                       self.have_history, self.sensed, self.visual):
            buffer[env_ids] = 0
        self.height[env_ids] = .045
        self.actions[env_ids] = 0.

    @staticmethod
    def motor_channels(dof_values, robot):
        rails = dof_values.index_select(1, robot.rail_indices)
        return torch.cat((dof_values[:, :6], rails.mean(dim=-1, keepdim=True)), dim=-1)

    def _pre_physics_step(self, actions):
        self.actions.copy_(actions.clamp(-1., 1.))
        for i, (robot, limits) in enumerate(zip(self.robots, self.joint_limits, strict=True)):
            target = self.q_des[i]
            q = self.motor_channels(robot.data.joint_pos.torch, robot)
            command = q + self.actions[:, i * JOINT_CHANNELS:(i + 1) * JOINT_CHANNELS] * self.step_limit
            target[:, :6] = command[:, :6].clamp(limits[:, :6, 0], limits[:, :6, 1])
            rail = command[:, 6:7]
            min_rail = limits[:, robot.rail_ids, 0].amax(dim=-1, keepdim=True)
            # The last part of a closing action requests motor preload against
            # the calibrated physical stop through the original rail PD drive.
            preload = ((rail - self.rail_close) / self.step_limit[-1]).clamp(0., 1.)
            physical_target = torch.maximum(rail, min_rail).clamp_max(self.rail_close)
            target[:, robot.rail_ids] = physical_target + preload * self.rail_overtravel

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
        lip_offset = body[:, 0, 2:] - height
        near_rim = (rim_margin >= -.015) & (rim_margin < .016)
        moves_in = (body[:, 0, :2] * horizontal).sum(dim=-1, keepdim=True) < 0.
        side_allowed = self.episode.clearance_m.amax(dim=1, keepdim=True) >= .020
        enters = ((speed > .012) & (speed > 2. * local_velocity[:, 0, 2:].abs()) & moves_in
                  & near_rim & (lip_offset.abs() < .012) & side_allowed)
        was_side = self.side[:, i:i + 1]
        entering = enters & ~was_side
        self.feed_axis[:, i] = torch.where(entering, direction, self.feed_axis[:, i])
        approach = (velocity * self.feed_axis[:, i]).sum(dim=-1, keepdim=True)
        inward = dq.index_select(1, robot.rail_indices)
        opens = inward.mean(dim=-1, keepdim=True) < -.025
        opening = opens | (self.q_des[i][:, robot.rail_ids].mean(dim=-1, keepdim=True)
                           < robot.data.joint_pos.torch[:, robot.rail_ids].mean(dim=-1, keepdim=True) - .0001)
        side = (was_side | enters) & ~opens
        self.side[:, i:i + 1] = side
        above_lip = self.entered_above_lip[:, i:i + 1]
        above_lip.copy_(side & (above_lip | (entering & (lip_offset >= 0.))))
        self.side_feed[:, i] = torch.where(side, (self.side_feed[:, i] + approach * DT).clamp_min(0.), 0.)
        history = self.parcel_state.contacts_A if i == 0 else self.parcel_state.contacts_B
        gathered = (history[..., ContactState.SLACK] > 0.).all(dim=-1, keepdim=True)
        captured = (history[..., ContactState.CAPTURE] > .5).any(dim=-1, keepdim=True)
        inside = rim_margin > 0.
        at_lip = (lip_offset.abs() < .012) & inside
        side_film = side & (self.net_side_feed[:, i] > 0.) & (at_lip | (captured & gathered))
        local_pads = (pads - parcel.pos.mean(dim=1, keepdim=True)) @ frame
        weights = payload.weights_at(local_pads[..., :2])
        indentation = (weights @ parcel.d_surf.unsqueeze(-1)).squeeze(-1)
        soft = payload.params.max_indent_ratio > 0.
        pads_inside = (local_pads[..., :2].abs() <= footprint[:, None]).all(dim=-1)
        pressing_film = (soft & (self.top_press[:, i] > 0.) & pads_inside.all(dim=-1, keepdim=True)
                         & (indentation > 1e-6).any(dim=-1, keepdim=True))
        top_film = ~side & soft & gathered & (pressing_film | captured)
        pad_gap = (pads[:, 1:] - pads[:, :1]).norm(dim=-1)
        close_force = motor.index_select(1, robot.rail_indices).clamp_min(0.)
        rail_position = robot.data.joint_pos.torch.index_select(1, robot.rail_indices)
        against_limit = rail_position >= self.rail_close - .00008
        loaded = torch.minimum(close_force, (-robot.rail_load).clamp_min(0.))
        approaching_film = ~opening & (side_film | (~side & pressing_film))
        normal = torch.where(~against_limit & approaching_film, close_force, loaded)
        opposed = close_force.amin(dim=-1, keepdim=True)
        compact = pad_gap <= self.film_gap
        skin = (weights @ payload.half_thickness(parcel.d_surf).unsqueeze(-1)).squeeze(-1)
        free_fold = (local_pads[..., 2] > skin).all(dim=-1, keepdim=True)
        if self.arm_count == 2 and self.result is not None:
            # A tilted parcel can enter between widely spaced receiving pads
            # before both pads clear the lip. Confirm the later geometry only
            # if the other arm is still supporting the airborne parcel.
            other = self.parcel_state.contacts_B if i == 0 else self.parcel_state.contacts_A
            supported_by_other = (other[..., ContactState.CAPTURE] > .5).all(dim=-1, keepdim=True)
            airborne = (self.result.table_clearance > .001) & (self.result.support_N == 0.)
            above_lip.logical_or_(side_film & free_fold & supported_by_other & airborne & ~opening)
        side_enabled = ~opening & side_film & above_lip
        damping = self._compress_side_fold(i, robot, port_state, pad_gap, side_enabled & free_fold)
        static = torch.where(compact & (side_enabled | (~opening & top_film)), opposed, 0.)
        self.film_load[:, i] = torch.maximum(damping, static)
        normal = torch.maximum(normal, self.film_load[:, i])
        self.pad_normal[:, i] = normal
        # A held rail cannot continue gathering through solver velocity jitter
        # at the hard stop. Keep its actual velocity in the wrench port.
        gathering_velocity = torch.where(against_limit, 0., inward)
        return ParcelAction(pads, velocities, normal, gathering_velocity, self.side_feed[:, i].expand_as(normal),
                            side.expand_as(normal).to(dq.dtype), self.empty_action.external_force,
                            self.empty_action.direct_press_N)

    def _compress_side_fold(self, i, robot, port_state, gap, enabled):
        """Dissipate pad approach in the loose fold before the compact stop."""
        pads, speed = port_state.pads, port_state.pad_velocity
        axis = (pads[:, 1] - pads[:, 0]) / gap.clamp_min(1e-8)
        closure = (-(speed[:, 1] - speed[:, 0]) * axis).sum(dim=-1, keepdim=True).clamp_min(0.)
        width = self.side_fold_gap - self.film_gap
        enter = ((self.side_fold_gap - gap) / (.25 * width)).clamp(0., 1.)
        compact = ((gap - self.film_gap) / (.25 * width)).clamp(0., 1.)
        fade = enter.square() * (3. - 2. * enter) * compact.square() * (3. - 2. * compact)
        damping = self.side_fold_damping * fade * enabled
        jac = ((port_state.pad_jacobian[:, 1] - port_state.pad_jacobian[:, 0]) * axis.unsqueeze(-1)).sum(dim=1)
        mobility = torch.linalg.solve_ex(robot.data.mass_matrix.torch, jac.unsqueeze(-1)).result.squeeze(-1)
        inverse_mass = (jac * mobility).sum(dim=-1, keepdim=True).clamp_min(0.)
        force = damping * closure / (1. + DT * damping * inverse_mass)
        self.fold_work[:, i] = -DT * force * (closure - .5 * DT * inverse_mass * force)
        # Equal, opposite, collinear pad reactions have zero wrench at the wrist.
        reaction = force * axis
        self.fold_reaction[:, i, 0] = -reaction
        self.fold_reaction[:, i, 1] = reaction
        return force

    def _apply_action(self):
        kinematics = [port.read() for port in self.ports]
        q = [robot.data.joint_pos.torch for robot in self.robots]
        dq = [robot.data.joint_vel.torch for robot in self.robots]
        motor = [robot.actuators.applied_effort.torch for robot in self.robots]
        cmds, quats, rates = [], [], []
        for i, robot in enumerate(self.scene_robots):
            robot.actuators.target_command.set_position_index(value=self.q_des[i], full_data=True)
            # MIT-style joint feedforward supplies the URDF arm's own weight.
            # The two rails retain their independent position/effort channels.
            gravity = self.gravity_feedforward[i]
            gravity[:, :6].copy_(robot.data.gravity_compensation_forces.torch[:, :6])
            robot.actuators.target_command.set_effort_index(value=gravity, full_data=True)
        for i, (robot, sample) in enumerate(zip(self.robots, kinematics, strict=True)):
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
        self.net_side_feed.copy_(self.result.net_feed.amin(dim=-1, keepdim=True))
        self.top_press.copy_(self.result.press_N.unsqueeze(-1))
        self._track_hold()
        self.reaction.add_(self.fold_reaction)
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
                                       dofs_to_motor(self.q_des[:self.arm_count]), opening)
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
        if self.arm_count == 2:
            contact_a = self.parcel_state.contacts_A[..., ContactState.CAPTURE] == 1.
            contact_b = self.parcel_state.contacts_B[..., ContactState.CAPTURE] == 1.
            parcel = self.parcel_state.payload
            center = self.parcel_sim.payload.com(parcel)
            at_handover = ((center - self.handover_center).abs() <= self.handover_half).all(dim=1)
            self.handover_seen |= at_handover & airborne & contact_a.all(dim=1) & contact_b.all(dim=1)
            corners = self.projectors[0].vertices(parcel)[..., :2]
            in_target = ((corners - self.place_center).abs() <= self.place_half).all(dim=(1, 2))
            right_open = ~contact_a.any(dim=1)
            released = right_open & ~contact_b.any(dim=1)
            supported = (self.result.support_N[:, 0] > 0.) & (self.result.table_clearance[:, 0] <= .001)
            self.left_carrying |= self.handover_seen & right_open & contact_b.all(dim=1)
            self.placed_supported = released & (self.placed_supported | (self.left_carrying & supported & in_target))
            self.left_carrying &= ~released
            settled = self.placed_supported & intact & quiet & in_target & supported
            self.place_streak = torch.where(settled, self.place_streak + 1, 0)

    def _get_observations(self):
        q = torch.stack([self.motor_channels(robot.data.joint_pos.torch, robot) for robot in self.robots], dim=1)
        dq = torch.stack([self.motor_channels(robot.data.joint_vel.torch, robot) for robot in self.robots], dim=1)
        actor = torch.cat((q, dq, self.sensed, self.actions.unflatten(1, (self.arm_count, JOINT_CHANNELS)),
                           self.visual), dim=-1).flatten(1)
        if self.arm_count == 2:
            location = torch.cat((self.place_center, self.place_half)).expand(self.num_envs, -1)
            actor = torch.cat((actor, location), dim=1)
        contacts = torch.cat((self.parcel_state.contacts_A.flatten(1), self.parcel_state.contacts_B.flatten(1)), dim=1)
        return {"policy": actor, "critic": torch.cat((actor, self.parcel_sim.get_privileged_state(self.parcel_state),
                                                     contacts), dim=1)}

    def _get_rewards(self):
        if self.arm_count == 2:
            return (self.place_streak >= 250).to(torch.float32)
        return (self.hold >= 1500).to(torch.float32)

    def _get_dones(self):
        torn = self.parcel_state.torn_flag.flatten(1).any(dim=1)
        return torn, self.episode_length_buf >= self.max_episode_length

    def _make_markers(self):
        if self.num_envs == 1 and self.arm_count == 1:
            from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer
            from .visual_skin import NewtonParcelSkin
            from .yam_scene import YAM_USD

            for viewer in self.sim._visualizers:
                if isinstance(viewer, NewtonGLVisualizer):
                    return NewtonParcelSkin(YAM_USD, viewer, str(self.device))
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(prim_path="/Visuals/Parcel", markers={
            "block": sim_utils.CuboidCfg(size=(1., 1., 1.)),
            "envelope": sim_utils.CuboidCfg(
                size=(1., 1., 1.), visual_material=sim_utils.PreviewSurfaceCfg(opacity=.2)),
        })
        return VisualizationMarkers(cfg)

    def _draw_parcel(self):
        from .visual_skin import NewtonParcelSkin

        if isinstance(self.markers, NewtonParcelSkin):
            self.markers.draw(self)
            return
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
