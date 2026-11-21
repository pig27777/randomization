"""Shared pad-space execution and hardware observations for Stage 4."""

import math
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils._pytree import tree_map

from ..dual_arm_parcel_sim import DualArmAction
from ..hardware_filters import YAMCurrentEstimatorFilter
from ..lumped_contact_proxy import ContactState
from ..macro_parcel_sim import ParcelAction
from ..pad_torsion import rotation_quaternion
from ..rl.asymmetric_ppo import Observation
from ..unified_payload import DT
from ..virtual_rgb import CameraPose, VirtualRGBProjector


class Command(NamedTuple):
    center: Tensor      # [B,2,3], world-frame gripper FK target
    half_span: Tensor   # [B,2]
    grip: Tensor        # [B,2], equivalent jaw effort in N
    feed: Tensor        # [B,2], planned advance past the estimated edge
    side: Tensor        # [B,2], position-controlled side-feed route
    angle: Tensor       # [B,2,3], commanded rotation vector
    stiffness: Tensor   # [B,2,3], passive wrist spring, N/m


class Transition(NamedTuple):
    obs: Observation
    reward: Tensor
    done: Tensor
    info: dict[str, Tensor]


def smooth(t):
    return .5 - .5 * (math.pi * t.clamp(0., 1.)).cos()


def scaled(action, low, high):
    return low + .5 * (high - low) * (action + 1.)


def has_grasp(contacts):
    return (contacts[..., ContactState.CAPTURE] > .5).all(dim=1)


def select(mask, new, old):
    return torch.where(mask.reshape(-1, *([1] * (new.ndim - 1))), new, old)


def columns(*items):
    return torch.cat(tuple(x.flatten(1) if x.ndim > 1 else x[:, None] for x in items), dim=1)


def phase_encoding(phase, count):
    return (phase[:, None] == torch.arange(count, device=phase.device)).float()


class ParcelEnv:
    frames = 28

    def __init__(self, episode, *, seed=347):
        self.episode = episode
        self.mass = episode.payload.total_mass
        self.batch = len(self.mass)
        self.device = self.mass.device
        self.seed = seed
        self.up = self.vector([0., 0., 1.])
        self.along = self.vector([0., 1., 0.])
        self.eye_x = self.vector([1., 0., 0.])
        self.wrist_frame = self.vector([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
        self.arm_mask = torch.ones((self.batch, 2), dtype=torch.bool, device=self.device)

    def vector(self, values):
        return self.mass.new_tensor(values)

    def zeros(self, *shape):
        return self.mass.new_zeros((self.batch, *shape))

    def flags(self, value=False):
        return torch.full((self.batch,), value, dtype=torch.bool, device=self.device)

    def _setup(self, state=None, sim=None):
        self.sim = self.episode.make_sim(
            mount_xy=self.vector([[[-.024, 0.], [.024, 0.]]] * 2).expand(self.batch, -1, -1, -1)
        ) if sim is None else sim
        self.state = self.episode.table_state(self.sim) if state is None else state
        self.yam = YAMCurrentEstimatorFilter(
            self.episode.yam, generator=torch.Generator(device=self.device).manual_seed(self.seed))
        diagonal = math.sqrt(.5)
        top = CameraPose(self.vector([0., -.60, .65]).expand(self.batch, -1),
                         self.vector([[1., 0., 0.], [0., -diagonal, diagonal],
                                      [0., -diagonal, -diagonal]]).expand(self.batch, -1, -1))
        # Calibrated workcell boundaries include the two longitudinal obstacles.
        hy = self.episode.payload.bag_half_extents[:, 1]
        clearance = self.episode.clearance_m
        bounds = columns(self.zeros() - .60, self.zeros() + .60, -hy - clearance[:, 0], hy + clearance[:, 1])
        self.rgb_sensor = VirtualRGBProjector(
            self.sim.payload, top, self.episode.rgb, table_bounds=bounds,
            generator=torch.Generator(device=self.device).manual_seed(self.seed + 1))
        self.offset = self.zeros(2, 3)
        self.offset_velocity = self.zeros(2, 3)
        self.load = self.zeros(2, 3)
        self.tare = self.zeros(2, 3)
        self.phase = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        self.attempts = torch.zeros_like(self.phase)
        self.done = self.flags()
        self.success = self.flags()
        self.dropped = self.flags()
        self.torn = self.flags()
        self.best_progress = self.zeros()
        self.history = self.zeros(self.frames, self.seq_size)
        self.valid = torch.zeros_like(self.history[..., 0], dtype=torch.bool)
        self.cursor = torch.zeros_like(self.phase)
        self.enabled = self.arm_mask

    def _idle(self, center, *, grip=None, half_span=None):
        return Command(center, self.zeros(2) + .04 if half_span is None else half_span,
                       self.zeros(2) if grip is None else grip, self.zeros(2),
                       self.zeros(2).bool(), self.zeros(2, 3),
                       self.vector([90., 90., 650.]).expand_as(center))

    def _tick(self, requested, active, *, lift=False):
        previous = self.command
        cmd = tree_map(lambda new, old: select(active, new, old), requested, previous)
        # Passive wrist load and deflection are part of the mechanical actuator.
        # Software primitives only read the resulting encoder and delayed sensors.
        k = cmd.stiffness
        damping = self.vector([12., 12., 28.])
        speed = (.25 * self.offset_velocity + DT * (self.load - k * self.offset)) / (
            .25 + DT * damping + DT**2 * k)
        offset = (self.offset + DT * speed.clamp(-.24, .24)).clamp(-.055, .055)
        delta = (offset - self.offset) / DT
        self.offset = select(active, offset, self.offset)
        self.offset_velocity = select(active, delta, self.offset_velocity)
        actual = cmd.center + self.offset
        velocity = (cmd.center - previous.center) / DT + self.offset_velocity
        jaw_q = rotation_quaternion(cmd.angle)
        angle_rate = (cmd.angle - previous.angle) / DT
        pads = self._pad_positions(actual, cmd.half_span, jaw_q)
        jaw_speed = ((previous.half_span - cmd.half_span) / DT).clamp_min(0.)
        pad_velocity = (pads - self.pads) / DT
        self.pads = select(active, pads, self.pads)
        arms = tuple(ParcelAction(pads[:, arm], pad_velocity[:, arm],
                                  cmd.grip[:, arm:arm+1].expand(-1, 2), jaw_speed[:, arm:arm+1].expand(-1, 2),
                                  cmd.feed[:, arm:arm+1].expand(-1, 2), self.ones_pads,
                                  self.zero_xyz, self.zero_column) for arm in range(2))
        out = self.sim.step(self.state, DualArmAction(*arms, self.enabled & active[:, None], cmd.side),
                            jaw_orientation=jaw_q, jaw_angular_velocity=angle_rate)
        self.state = tree_map(lambda new, old: select(active, new, old), out.state, self.state)
        self.load = select(active, out.wrist_wrench[..., :3], self.load)
        self.command, self.actual, self.result = cmd, actual, out
        self.sensed = self.yam.step(self.load, velocity, -self.up.expand_as(velocity),
                                    cmd.grip[..., None], 2. * cmd.half_span[..., None], actual, cmd.center)
        wrist = CameraPose(actual[:, 1] + self.vector([0., .015, .015]),
                           self.wrist_frame.expand(self.batch, -1, -1))
        self.rgb = self.rgb_sensor.step(self.state.payload, wrist, actual[:, 0],
                                        active & lift)
        self.torn |= active & self.state.torn_flag.flatten(1).any(dim=1)

    def _pad_positions(self, center, half_span, quaternion):
        # The first column of the rotation matrix gives the pad separation axis.
        w, x, y, z = quaternion.unbind(dim=-1)
        axis = torch.stack((1. - 2. * (y*y + z*z), 2. * (x*y + w*z), 2. * (x*z - w*y)), dim=-1)
        half = half_span[..., None] * axis
        return center[:, :, None] + self.vector([-1., 1.])[None, None, :, None] * half[:, :, None]

    def _start_command(self, command):
        self.command = command
        self.actual = command.center + self.offset
        self.zero_xyz = self.zeros(3)
        self.zero_column = self.zeros(1)
        self.ones_pads = self.zeros(2) + 1.
        self.pads = self._pad_positions(self.actual, command.half_span, rotation_quaternion(command.angle))

    def _clear_history(self, mask):
        self.history = select(mask, torch.zeros_like(self.history), self.history)
        self.valid = select(mask, torch.zeros_like(self.valid), self.valid)
        self.cursor = torch.where(mask, 0, self.cursor)

    def _record(self, features, mask):
        # Append to the valid prefix; roll the oldest frame out only on overflow.
        full = self.cursor >= self.frames
        index = self.cursor.clamp_max(self.frames - 1)
        rows = torch.arange(self.batch, device=self.device)
        history = select(mask & full, self.history.roll(-1, dims=1), self.history)
        self.history = history.index_put((rows, index), select(mask, features, history[rows, index]))
        self.cursor = self.cursor + mask.long()
        self.valid = torch.arange(self.frames, device=self.device)[None] < self.cursor[:, None]

    def _stable(self, arm):
        contacts = self.state.contacts_A if arm == 0 else self.state.contacts_B
        return (has_grasp(contacts) & (self.result.table_clearance[:, 0] > .001)
                & (self.result.support_N[:, 0] == 0.) & ~self.torn
                & (self.sim.payload.com_velocity(self.state.payload).norm(dim=-1) < .020))

    def _outer(self):
        p, x = self.episode.proxy, self.episode.payload
        radius = torch.broadcast_to(torch.as_tensor(self.episode.torsion.radius_m, device=self.device,
                                                     dtype=self.mass.dtype),
                                    self.mass.shape)
        stiffness = (p.k / 1000.).clamp(0., 3.)
        return columns(p.S_bag, stiffness, p.c_d, p.mu_eff, p.alpha, p.F_lock, p.F_release,
                       self.sim.F_half, self.sim.T_tear, radius, x.fill_ratio)

    def _hardware(self, handover):
        y, v = self.episode.yam, self.episode.rgb
        if handover:
            return columns(y.sensor_gain, y.coulomb_N, y.cutoff_hz, y.delay_steps,
                           v.delay_steps[:, 1], v.wrist_min_range_m)
        return columns(y.sensor_gain[:, 0], y.coulomb_N[:, 0], y.pose_bias_N[:, 0], y.cutoff_hz[:, 0],
                       y.delay_steps[:, 0], v.delay_steps[:, 0], v.drop_probability[:, 0], v.mask_jitter[:, 0])

    def _obs(self, static, priv, forced=None):
        forced = torch.full_like(self.phase, -1) if forced is None else forced
        return Observation(static, self.history, self.valid, priv,
                           torch.where(self.done, -1, self.phase), forced)

    def _transition(self, reward, **details):
        return Transition(self.observation(), reward, self.done.clone(),
                          dict(success=self.success.clone(), dropped=self.dropped.clone(),
                               torn=self.torn.clone(), attempts=self.attempts.clone(), **details))
