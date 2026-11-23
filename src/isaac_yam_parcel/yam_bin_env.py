"""Single ABC right arm picking a low-order parcel and clearing a four-wall bin."""

from dataclasses import replace
import json
import math

import torch
from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg

from src.low_order_parcel.domain_randomizer import DomainRandomizer
from src.low_order_parcel.lumped_contact_proxy import ContactState

from .yam_parcel_base_env import OBS_PER_ARM, YAMPickEnv, yam_parcel_cfg
from .yam_scene import SCENE_FILE, load_scene


LAYOUT_FILE = SCENE_FILE.with_name("route_b_single_arm_bin.yaml")
BIN_FEATURES = 6
PRIV_BIN_FEATURES = 30


def load_bin_layout(opening_m=None):
    layout = json.loads(LAYOUT_FILE.read_text())
    bin_cfg = layout["bin"]
    if opening_m is not None:
        half = [value / 2. for value in opening_m]
        lo, hi = bin_cfg["inner_half_xy_range_m"]
        if not all(a <= x <= b for x, a, b in zip(half, lo, hi, strict=True)):
            raise ValueError("Bin openings must be inside the configured range.")
        bin_cfg["inner_half_xy_m"] = half
    scene = load_scene()
    if (scene["arm_bases_m"][1] != layout["arm"]["base_m"]
            or scene["table"]["center_m"][2] + scene["table"]["size_m"][2] / 2. != 0.):
        raise ValueError("The single-arm layout requires the ABC right base and the zero-height table.")
    return layout


def bin_wall_boxes(bin_cfg):
    """Centers and half extents in ABC coordinates, also used to audit parcel scrapes."""
    x, y, floor = bin_cfg["center_m"]
    hx, hy = bin_cfg["inner_half_xy_m"]
    h = bin_cfg["rim_height_m"] / 2.
    t = bin_cfg["wall_thickness_m"] / 2.
    return (
        ((x - hx - t, y, floor + h), (t, hy + 2. * t, h)),
        ((x + hx + t, y, floor + h), (t, hy + 2. * t, h)),
        ((x, y - hy - t, floor + h), (hx, t, h)),
        ((x, y + hy + t, floor + h), (hx, t, h)),
    )


def yam_bin_cfg(num_envs, layout, *, seed=1):
    cfg = yam_parcel_cfg(num_envs, park_left=False, seed=seed)
    cfg.scene.arm_a.actuators["proximal"].stiffness = 140.
    cfg.scene.arm_a.actuators["proximal"].damping = 6.6
    cfg.scene.arm_a.spawn.activate_contact_sensors = True
    cfg.scene.yam_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Yam_B/Geometry/.*", history_length=0, update_period=0.,
        max_contact_data_count_per_prim=4,
    )
    gripper = "{ENV_REGEX_NS}/Yam_B/Geometry/base/link1/link2/link3/link4/link5/gripper"
    for side, opposite in (("left", "right"), ("right", "left")):
        setattr(cfg.scene, f"tip_{side}_mating", ContactSensorCfg(
            prim_path=f"{gripper}/tip_{side}", filter_prim_paths_expr=[f"{gripper}/tip_{opposite}"],
            history_length=0, update_period=0., max_contact_data_count_per_prim=4,
        ))
    for name, (center, half) in zip(("West", "East", "Near", "Far"), bin_wall_boxes(layout["bin"]), strict=True):
        setattr(cfg.scene, f"bin_{name.lower()}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/BinWall{name}",
            spawn=sim_utils.CuboidCfg(size=tuple(2. * hi for hi in half),
                                      collision_props=sim_utils.UsdPhysicsCollisionCfg()),
            init_state=AssetBaseCfg.InitialStateCfg(pos=center),
        ))
    cfg.observation_space = OBS_PER_ARM + BIN_FEATURES
    cfg.state_space = cfg.observation_space + 54 + 24 + PRIV_BIN_FEATURES
    return cfg


class BinRandomizer(DomainRandomizer):
    def __init__(self, device, seed, layout):
        super().__init__(device, seed)
        self.layout = layout

    def sample(self, count, *, curriculum_alpha=1.):
        episode = super().sample(count, curriculum_alpha=curriculum_alpha)
        dr, pick, basket = (self.layout[key] for key in ("dr", "pick", "bin"))
        small, large = .5 * self.tensor(dr["parcel_size_m"])
        half = torch.lerp(episode.payload.bag_half_extents, torch.lerp(small, large, self._uniform((count, 3))),
                          curriculum_alpha)
        half = half.clamp(small, large)
        payload = episode.payload
        old_cavity = payload.cavity_slack(payload.bag_half_extents, payload.fill_ratio)
        new_cavity = payload.cavity_slack(half, payload.fill_ratio)
        slide = (payload.slide_slack_max / old_cavity.clamp_min(1e-8)).clamp(0., 1.) * new_cavity
        payload = replace(payload, bag_half_extents=half, slide_slack_max=slide)
        lo, hi = dr["free_neck_m"]
        reference = torch.lerp(episode.proxy.L0, lo + (hi - lo) * self._uniform((count, 1)), curriculum_alpha)
        proxy = replace(episode.proxy, L0=reference, S_bag=torch.minimum(episode.proxy.S_bag,
                                                                        payload.fill_slack_capacity()))
        # Large packages and large yaw are correlated by the gap to the near wall.
        # Find the yaw cap on the rising part of hx*sin(yaw) + hy*cos(yaw).
        hx, hy = half[:, :1], half[:, 1:2]
        available = self.near_wall_y - (pick["center_m"][1] - pick["half_xy_m"][1])
        amax = math.radians(pick["max_abs_yaw_deg"])
        peak_yaw = torch.atan2(hx, hy)
        yaw_limit = (torch.asin((available / torch.hypot(hx, hy)).clamp(-1., 1.)) - torch.atan2(hy, hx))
        yaw_limit = torch.where(available >= hx * peak_yaw.clamp_max(amax).sin()
                                + hy * peak_yaw.clamp_max(amax).cos(), amax, yaw_limit).clamp(0., amax)
        yaw = (2. * self._uniform((count, 1)) - 1.) * yaw_limit * (.25 + .75 * curriculum_alpha)
        return replace(episode, payload=payload, proxy=proxy, yaw_rad=yaw)

    @property
    def near_wall_y(self):
        basket = self.layout["bin"]
        return basket["center_m"][1] - basket["inner_half_xy_m"][1] - basket["wall_thickness_m"] - .001


class YAMSingleArmBinEnv(YAMPickEnv):
    """One 7-channel action, zero-privilege actor, and audited over-rim releases."""

    def __init__(self, cfg, layout, *, episode=None, curriculum_alpha=1.):
        self.layout = layout
        self.bin_cfg = layout["bin"]
        super().__init__(cfg, episode=episode, curriculum_alpha=curriculum_alpha)
        like = self.scene.env_origins
        self.bin_center = like.new_tensor(self.bin_cfg["center_m"])
        self.bin_half = like.new_tensor(self.bin_cfg["inner_half_xy_m"])
        self.rim = self.bin_cfg["rim_height_m"]
        self.bin_actor = torch.cat((self.bin_center, self.bin_half, like.new_tensor([self.rim])))
        walls = like.new_tensor(bin_wall_boxes(self.bin_cfg))
        self.wall_low = walls[:, 0] - walls[:, 1]
        self.wall_high = walls[:, 0] + walls[:, 1]
        self.nominal_step = self.step_limit
        self.step_limit = self.nominal_step.expand(self.num_envs, -1).clone()
        self.arm_pd_scale = like.new_ones((self.num_envs, 6))
        drives = self.robots[0].cfg.actuators
        self.damping_over_stiffness = like.new_tensor(
            [drives[name].damping / drives[name].stiffness for name in ("proximal", "distal")]).repeat_interleave(3)
        self.arm_ids = torch.arange(6, device=self.device)
        self.drive_command = torch.empty_like(self.q_des[0])
        self.peak_contact = like.new_zeros(self.num_envs)
        self.bin_streak = torch.zeros_like(self.hold)
        self.entered_bin = torch.zeros_like(self.ever_captured)
        self.inside_before = torch.zeros_like(self.entered_bin)
        self.opened_in_bin = torch.zeros_like(self.entered_bin)
        self.valid_release = torch.zeros_like(self.entered_bin)
        self.rim_scrape = torch.zeros_like(self.entered_bin)
        self.dropped_early = torch.zeros_like(self.entered_bin)
        self.moving_held = torch.zeros_like(self.entered_bin)
        self.tension = like.new_zeros((self.num_envs, 2))
        self.contacts = self.scene["yam_contacts"]
        self.mating_contacts = (self.scene["tip_left_mating"], self.scene["tip_right_mating"])
        self.tip_contact_ids = torch.tensor([self.contacts.body_names.index(name) for name in ("tip_left", "tip_right")],
                                            device=self.device)
        sample = self.ports[0].read()
        distance = ((sample.pads[:, 1] - sample.pads[:, 0]) * sample.rail_axis[:, 0]).sum(dim=-1)
        self.zero_position_aperture = (distance + self._rail_sum()).median()

    def _make_randomizer(self, seed):
        return BinRandomizer(self.device, seed, self.layout)

    def _parcel_table_state(self):
        state = self.episode.table_state(self.parcel_sim)
        pick = self.layout["pick"]
        like = self.scene.env_origins
        center = like.new_tensor(pick["center_m"])
        half = like.new_tensor(pick["half_xy_m"])
        hx, hy = self.episode.payload.bag_half_extents[:, :2].unbind(dim=1)
        angle = self.episode.yaw_rad[:, 0]
        reach_y = hx * angle.sin().abs() + hy * angle.cos().abs()
        clear_y = self.randomizer.near_wall_y - reach_y
        if self.fixed_episode:
            shift = center.expand_as(like).clone()
            shift[:, 1] = torch.minimum(shift[:, 1], clear_y)
        else:
            lower, upper = center[:2] - half, center[:2] + half
            upper_y = clear_y.clamp(min=lower[1], max=upper[1])
            upper = torch.stack((upper[0].expand_as(upper_y), upper_y), dim=1)
            xy = torch.lerp(lower, upper, self.randomizer._uniform((self.num_envs, 2)))
            shift = torch.cat((xy, center[2:].expand(self.num_envs, 1)), dim=1)
        return state._replace(payload=state.payload._replace(pos=state.payload.pos + shift[:, None]),
                              side_origin=state.side_origin + shift[:, None])

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dr = self.layout["dr"]
        pd_lo, pd_hi = dr["arm_pd_scale"]
        step_lo, step_hi = dr["step_scale"]
        friction_lo, friction_hi = dr["arm_static_friction_nm"]
        draw = self.randomizer._uniform((n, 3, 6))
        alpha = self.curriculum_alpha if not self.fixed_episode else 0.
        self.arm_pd_scale[env_ids] = 1. + alpha * (pd_lo + (pd_hi - pd_lo) * draw[:, 0] - 1.)
        self.step_limit[env_ids, :6] = self.nominal_step[:6] * (
            1. + alpha * (step_lo + (step_hi - step_lo) * draw[:, 1] - 1.))
        friction = alpha * (friction_lo + (friction_hi - friction_lo) * draw[:, 2])
        self.robots[0].write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction, joint_ids=self.arm_ids, env_ids=env_ids)
        for value in (self.peak_contact, self.bin_streak, self.entered_bin, self.inside_before,
                      self.opened_in_bin, self.valid_release, self.rim_scrape, self.dropped_early,
                      self.moving_held, self.tension):
            value[env_ids] = 0

    def _drive_target(self, i, robot):
        # Encode episode PD gain variation through the explicit motor's position
        # input. Its computed effort and hardware clipping still use the stock path.
        q = robot.data.joint_pos.torch[:, :6]
        dq = robot.data.joint_vel.torch[:, :6]
        self.drive_command.copy_(self.q_des[i])
        self.drive_command[:, :6] += (self.arm_pd_scale - 1.) * (
            self.q_des[i][:, :6] - q - self.damping_over_stiffness * dq)
        return self.drive_command

    def _observe_ports(self, kinematics, q, dq, motor):
        super()._observe_ports(kinematics, q, dq, motor)
        z_ee = kinematics[0].pads[..., 2].mean(dim=1)
        rail_speed = dq[0].index_select(1, self.robots[0].rail_indices).mean(dim=1)
        self._track_bin(z_ee, rail_speed)
        self.tension.copy_(self.result.tension[:, 0])
        self._record_link_contacts()

    def _track_bin(self, z_ee, rail_speed):
        state = self.parcel_state
        payload = self.parcel_sim.payload
        vertices = self.projectors[0].vertices(state.payload)
        blocks = vertices.unflatten(1, (4, 8))
        low, high = blocks.amin(dim=2), blocks.amax(dim=2)
        overlaps = ((high[:, :, None] > self.wall_low) & (low[:, :, None] < self.wall_high)).all(dim=-1)
        self.rim_scrape |= overlaps.any(dim=(1, 2))

        center = payload.com(state.payload)
        inside = ((center[:, :2] - self.bin_center[:2]).abs() <= self.bin_half).all(dim=1)
        within = ((vertices[..., :2] - self.bin_center[:2]).abs() <= self.bin_half).all(dim=(1, 2))
        bottom = low[..., 2].amin(dim=1)
        both = (state.contacts_A[..., ContactState.CAPTURE] == 1.).all(dim=1)
        no_grip = (state.contacts_A[..., ContactState.CAPTURE] == 0.).all(dim=1)
        supported = self.result.support_N[:, 0] > 0.
        prior_carry = self.moving_held.clone()
        self.moving_held |= both & ~supported & (self.result.table_clearance[:, 0] > .001)
        crossing = inside & ~self.inside_before & both & self.moving_held
        above = bottom >= self.rim + self.bin_cfg["clearance_over_rim_m"]
        self.rim_scrape |= crossing & ~above
        self.entered_bin |= crossing & above
        self.inside_before.copy_(inside)
        releasing = self.entered_bin & within & (z_ee <= self.rim + self.bin_cfg["release_z_above_rim_m"])
        self.opened_in_bin &= releasing & (rail_speed <= .002)
        self.opened_in_bin |= releasing & (rail_speed < -.002) & prior_carry
        self.valid_release |= releasing & self.opened_in_bin & no_grip
        self.dropped_early |= prior_carry & no_grip & ~self.valid_release
        quiet = state.payload.vel.norm(dim=-1).amax(dim=1) < self.bin_cfg["settled_speed_m_s"]
        settled = self.valid_release & no_grip & supported & within & quiet & (self.result.table_clearance[:, 0] <= .001)
        self.bin_streak.copy_(torch.where(settled, self.bin_streak + 1, 0))

    def _rail_sum(self):
        robot = self.robots[0]
        return robot.data.joint_pos.torch.index_select(1, robot.rail_indices).sum(dim=1)

    def _record_link_contacts(self):
        net = self.contacts.data.net_normal_forces_w.torch
        # Once calibrated from the open, collinear rails, the encoder sum keeps
        # sub-millimetre accuracy even for clones far from the world origin.
        gap = self.zero_position_aperture - self._rail_sum()
        mating = torch.cat([sensor.data.normal_force_matrix_w.torch.sum(dim=2)
                            for sensor in self.mating_contacts], dim=1)
        # PhysX reports speculative contact between the two mating sliders before the
        # CAD pad faces touch. Subtract only the measured force from that exact pair
        # at a positive, sub-stop gap. Table, basket and other link forces remain.
        clear_faces = (gap > 0.) & (gap <= self.film_gap)
        tip = net.index_select(1, self.tip_contact_ids) - torch.where(clear_faces[:, None, None], mating, 0.)
        force = net.norm(dim=-1).index_copy(1, self.tip_contact_ids, tip.norm(dim=-1)).amax(dim=1)
        torch.maximum(self.peak_contact, force, out=self.peak_contact)

    def _success(self):
        ticks = round(self.bin_cfg["settled_duration_s"] / self.cfg.sim.dt)
        return (self.bin_streak >= ticks) & ~self._failure()

    def _failure(self):
        torn = self.parcel_state.torn_flag.flatten(1).any(dim=1)
        return torn | self.rim_scrape | (self.peak_contact > .02) | self.dropped_early

    def _get_observations(self):
        obs = super()._get_observations()
        policy = torch.cat((obs["policy"], self.bin_actor.expand(self.num_envs, -1)), dim=1)
        state = self.parcel_state
        physics = torch.cat((state.payload.pos.flatten(1), state.payload.vel.flatten(1),
                             self.tension, state.contact_reference[:, 0], self.pad_normal[:, 0]), dim=1)
        critic = torch.cat((policy, obs["critic"][:, OBS_PER_ARM:], physics), dim=1)
        return {"policy": policy, "critic": critic}

    def _get_dones(self):
        # Read the final completed PhysX substep before returning the terminal batch.
        self._record_link_contacts()
        return self._success() | self._failure(), self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        return 5. * self._success().float() - 5. * self._failure().float()
