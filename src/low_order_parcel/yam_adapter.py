"""Couple the frozen contact proxy to the shared articulated YAM trial."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from experience.yam_platform.protocol import MuJoCoSim, TrialConfig, attach_robot, measure_contacts, tips, vec
from experience.yam_platform.rigid_control import NoPinch
from . import lumped_contact_proxy, macro_parcel_sim
from .lumped_contact_proxy import ContactState
from .macro_parcel_sim import MacroParcelSim, ParcelAction, rotate


ENVELOPES = Path(__file__).resolve().parents[2] / "configs/proxy_material_envelopes.yaml"


@dataclass
class Config(TrialConfig):
    dt: float = .002
    pinch_coeff: float = 0.
    pickup_only: bool = True
    pickup_release_seconds: float = .8
    lift_height: float = .15
    lift_speed_m_s: float = .04
    lift_hold_seconds: float = 3.
    simulation_seconds: float = 15.
    fold_width_m: float = .024
    fold_normal_stiffness_N_m: float = 1600.
    fold_free_length_m: float = .008
    fold_mount_spacing_m: float = .020
    grasp_face: str = "top"
    slack_budget_m: float | None = None
    payload_size_m: tuple[float, float, float] = (.15, .20, .032)
    payload_mass_kg: float | None = None
    payload_offset_xy_m: tuple[float, float] = (0., 0.)
    bag_inner_size_m: tuple[float, float] | None = None


class ProxyYamSim(MuJoCoSim):
    def __init__(self, built):
        super().__init__(built)
        self.parcel = built.parcel
        self.state = self.parcel.initial_state()
        self.state.com[0, :2] += torch.from_numpy(built.center)
        self.qpos_adr = int(self.m.jnt_qposadr[self.m.joint("parcel_pose").id])
        self.qvel_adr = int(self.m.jnt_dofadr[self.m.joint("parcel_pose").id])
        self._write_parcel_pose()
        self.last_step = None
        self.normal = np.zeros(2)
        self.feed = np.zeros(2)
        self.reactions = np.zeros((2, 3))
        self.grip = np.zeros(2)
        self.pad_position = np.zeros((2, 3))
        self.peak_tension = np.zeros(2)
        self.max_clearance = 0.
        self.bilateral_hover_s = 0.
        self.longest_hover_s = 0.
        self.capture_time = [None, None]
        self.release_time = [None, None]
        self.failure = None

    def bind_robot(self, robot):
        self.robot = robot
        self.finger_bodies = [self.m.body(f"{side}_finger").id for side in ("left", "right")]
        self.pad_position, _ = self._inner_tips()
        # At the end of a 100 mm approach the distal pad edge reaches the
        # exposed end of the fold. Its position then follows the parcel frame.
        entry = self.pad_position.mean(0) + .10 * robot.approach
        self.fold_entry = entry - self.com()

    def _write_parcel_pose(self):
        adr = self.qpos_adr
        self.d.qpos[adr:adr + 3] = self.com()
        self.d.qpos[adr + 3:adr + 7] = self.state.orientation[0].numpy()
        # The proxy owns parcel integration. The uncollidable free joint is
        # only a pose carrier for the platform's standard trajectory recorder.
        self.d.qvel[self.qvel_adr:self.qvel_adr + 6] = 0.

    def _inner_tips(self):
        pad = tips(self.robot)
        axis = pad[1] - pad[0]
        axis /= np.linalg.norm(axis)
        return pad + np.array([.004, -.004])[:, None] * axis, axis

    def com(self):
        return self.state.com[0].numpy().copy()

    def step(self):
        m, d = self.m, self.d
        robot, parcel, state = self.robot, self.parcel, self.state
        pad, axis = self._inner_tips()
        velocity = (pad - self.pad_position) / parcel.dt
        self.pad_position = pad
        sides = np.array([-axis, axis])
        inward = -((velocity - velocity.mean(0)) * sides).sum(1)
        entry = self.com() + rotate(state.orientation, torch.from_numpy(self.fold_entry[None]))[0].numpy()
        feed = np.maximum(0., (pad - entry) @ robot.approach)
        captured = state.contacts[0, :, ContactState.CAPTURE].numpy() > .5
        inserted = (feed > 0.) & (feed <= .04)
        gap = robot.gap()
        # Effective normal compliance of the exposed PE fold. The servo
        # supplies compression by closing its actual rails against this width.
        normal_each = self.built.cfg.fold_normal_stiffness_N_m * max(0., (self.built.cfg.fold_width_m - gap) / 2.)
        normal = normal_each * (inserted | captured)
        # Small unilateral entry reaction gives the shared side-approach servo
        # a surface to detect before the jaws close around the fold.
        front = (np.minimum(60. * feed, .09) * inserted * ~captured)[:, None] * robot.approach
        fold_axis = rotate(state.orientation, state.com.new_tensor([[1., 0., 0.]]))[0].numpy()
        align = abs(float(axis @ fold_axis))
        tensor = state.com.new_tensor
        direct_pad_load = normal[:, None] * sides - front
        action = ParcelAction(tensor(pad[None]), tensor(velocity[None]), tensor(normal[None]),
                              tensor(inward[None]), tensor(feed[None]), tensor([[align, align]]),
                              tensor(-direct_pad_load.sum(0)[None]), tensor([[0.]]))
        result = parcel.step(state, action)
        if not all(torch.isfinite(channel).all() for channel in (*result.state, result.wrist_wrench, result.tension)):
            self.failure = "nonfinite parcel proxy state"
            raise RuntimeError(self.failure)
        mounts = (state.com[:, None] + rotate(state.orientation[:, None], parcel.mounts))[0].numpy()
        line = pad - mounts
        line /= np.maximum(np.linalg.norm(line, axis=1, keepdims=True), 1e-12)
        reaction = direct_pad_load - result.tension[0].numpy()[:, None] * line
        d.qfrc_applied[:] = 0.
        for force, point, body in zip(reaction, pad, self.finger_bodies, strict=True):
            mujoco.mj_applyFT(m, d, force, np.zeros(3), point, body, d.qfrc_applied)
        mujoco.mj_step(m, d)
        self.state = result.state
        self.last_step = result
        self.normal = normal
        self.feed = feed
        self.reactions = reaction
        self.grip = np.maximum(0., (reaction * sides).sum(axis=1))
        self.track_events(captured)
        self._write_parcel_pose()
        mujoco.mj_kinematics(m, d)

    def track_events(self, was_captured):
        result = self.last_step
        captured = result.state.contacts[0, :, ContactState.CAPTURE].numpy() > .5
        for side in range(2):
            if captured[side] and not was_captured[side] and self.capture_time[side] is None:
                self.capture_time[side] = float(self.d.time)
            if was_captured[side] and not captured[side] and self.release_time[side] is None:
                self.release_time[side] = float(self.d.time)
        clearance = float(result.table_clearance[0, 0])
        self.max_clearance = max(self.max_clearance, clearance)
        self.peak_tension = np.maximum(self.peak_tension, result.tension[0].numpy())
        stable = (captured.all() and clearance > .001 and float(result.support_N[0, 0]) == 0.
                  and float(torch.linalg.vector_norm(self.state.velocity[0])) < .020)
        self.bilateral_hover_s = self.bilateral_hover_s + self.parcel.dt if stable else 0.
        self.longest_hover_s = max(self.longest_hover_s, self.bilateral_hover_s)


def scene(cfg):
    if cfg.grasp_face not in ("top", "side"):
        raise ValueError("grasp_face must be top or side")
    if cfg.slack_budget_m is not None and (not np.isfinite(cfg.slack_budget_m) or cfg.slack_budget_m < 0.):
        raise ValueError("slack budget must be finite and nonnegative")
    if not all(np.isfinite(value) and value > 0 for value in
               (cfg.fold_width_m, cfg.fold_normal_stiffness_N_m, cfg.fold_free_length_m,
                cfg.fold_mount_spacing_m)):
        raise ValueError("fold geometry and normal stiffness must be finite and positive")
    envelopes = ENVELOPES.read_bytes()
    nominal = json.loads(envelopes)["materials"]["pe"]["nominal"]
    nominal["proxy"]["L0"] = [cfg.fold_free_length_m] * 2
    if cfg.payload_mass_kg is not None:
        nominal["mass_kg"] = cfg.payload_mass_kg
    size = np.asarray(cfg.payload_size_m, dtype=float)
    shift = np.asarray(cfg.payload_offset_xy_m, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0.).any():
        raise ValueError("payload size must contain three finite positive lengths")
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("payload offset must contain two finite lengths")
    bag_center = np.array([-.25, .04])
    center = bag_center + shift
    fill = None
    outline = ""
    slack_m = cfg.slack_budget_m
    if cfg.bag_inner_size_m is not None:
        inner = np.asarray(cfg.bag_inner_size_m, dtype=float)
        if inner.shape != (2,) or not np.isfinite(inner).all() or (inner <= 0.).any():
            raise ValueError("bag interior must contain two finite positive lengths")
        available = (inner - size[:2] - size[2]) / 2.
        if (available < np.abs(shift) - 1e-12).any():
            raise ValueError("payload and planar offset must fit inside the seams")
        length, width = inner
        optimum = length * width / (length + width + np.sqrt(length**2 - length * width + width**2))
        fill = float(np.prod(size) / (optimum * np.prod(inner - optimum)))
        section = max(0., float(available[1] + shift[1]))
        slack_m = min(section, nominal["proxy"]["S_bag"] if slack_m is None else slack_m)
        corners = [np.r_[bag_center + .5 * inner * sign, .001] for sign in
                   ((-1, -1), (-1, 1), (1, 1), (1, -1))]
        outline = "".join(f'<geom type="capsule" fromto="{vec([a, b])}" size=".0011" '
                          'contype="0" conaffinity="0" rgba=".24 .69 .72 1"/>'
                          for a, b in zip(corners, corners[1:] + corners[:1], strict=True))
    offset = (.003, 0., 0.)
    half = size / 2.
    parcel = MacroParcelSim.from_nominal(nominal, cfg.dt, slack_m=slack_m, com_offset=offset, half_extents=half)
    if cfg.fold_mount_spacing_m > 2. * half[0]:
        raise ValueError("the two fold mounts must fit on the package top")
    parcel.mounts[0, :, 0] = parcel.mounts.new_tensor([-.5, .5]) * cfg.fold_mount_spacing_m - offset[0]
    mass = nominal["mass_kg"]
    if cfg.grasp_face == "side":
        parcel.mounts[0, :, 1] = -half[1]
        parcel.mounts[0, :, 2] = half[2] - .010
        pitch = np.deg2rad(25.)
        direction = np.array([0., np.cos(pitch), -np.sin(pitch)])
        # End insertion just outside the front wall at the upper part of the
        # side film. The grasp site is behind the eventual distal pad edge.
        contact_point = np.array([center[0], center[1] - half[1] - .012, size[2] - .005]) - .0255 * direction
        lift_retract = .020
    else:
        pitch = np.deg2rad(35.)
        direction = np.array([0., np.cos(pitch), -np.sin(pitch)])
        # Leave room to insert before reaching the line between the top mounts.
        contact_point = [center[0], center[1] - .020, size[2] + .023]
        lift_retract = 0.
    xml = f'''
    <mujoco model="YAM Stage 2 PE {cfg.grasp_face} proxy">
      <option timestep="{cfg.dt}" integrator="implicitfast" gravity="0 0 -9.81"/>
      <worldbody>
        <light pos="-.4 -.4 .8" dir=".3 .4 -1" directional="true"/>
        <geom name="table" type="box" pos="0 0 -.025" size="1 .22 .025" rgba=".24 .28 .32 1"/>
        {outline}
        <body name="parcel" pos="{vec([*center, half[2]])}" gravcomp="1">
          <freejoint name="parcel_pose"/>
          <geom name="package" type="box" pos="{vec(-np.array(offset))}" size="{vec(half)}"
                mass="{mass}" contype="0" conaffinity="0" rgba=".72 .56 .33 1"/>
          <geom name="seal" type="box" pos="-.003 0 {half[2]}" size=".020 {half[1]} .00035"
                mass="0" contype="0" conaffinity="0" rgba=".94 .78 .52 1"/>
          <site name="proxy_left" pos="{vec(parcel.mounts[0, 0])}" size=".0025" rgba=".13 .78 1 1"/>
          <site name="proxy_right" pos="{vec(parcel.mounts[0, 1])}" size=".0025" rgba="1 .49 .17 1"/>
        </body>
      </worldbody>
      <contact/><equality/>
    </mujoco>'''
    built = SimpleNamespace(xml=xml, desc=SimpleNamespace(), cfg=cfg, parcel=parcel, center=center,
                            envelope_sha256=sha256(envelopes).hexdigest(), fill_fraction=fill)
    approach = dict(direction=direction, closing=[-1., 0., 0.],
                    contact_point=contact_point, close_advance_m=.1155, lift_retract_m=lift_retract)
    robot = attach_robot(cfg, built, ProxyYamSim, center, size[2], size, mass, approach)
    robot.sim.bind_robot(robot)
    return robot


def measure(robot):
    sim = robot.sim
    table_contact = measure_contacts(robot, target_geoms={sim.table_geom})
    support = 0. if sim.last_step is None else float(sim.last_step.support_N[0, 0])
    return dict(grip_each_N=sim.grip.tolist(), contact_magnitude_each_N=np.linalg.norm(sim.reactions, axis=1).tolist(),
                force_on_fingers_N=sim.reactions.tolist(), support_N=support, top_contact_each=[False, False],
                penetration_mm=table_contact["penetration_mm"], nonfinger_mm=table_contact["nonfinger_mm"],
                table_contact_each_N=table_contact["contact_magnitude_each_N"])


def attachment(robot, cfg):
    return NoPinch()


def check_state(sim):
    if sim.last_step is None:
        return sim.failure, {}
    result = sim.last_step
    return sim.failure, dict(proxy_pad_m=sim.pad_position.tolist(), proxy_normal_N=sim.normal.tolist(),
                             proxy_feed_m=sim.feed.tolist(), proxy_tension_N=result.tension[0].tolist(),
                             proxy_capture=sim.state.contacts[0, :, ContactState.CAPTURE].tolist(),
                             proxy_slack_m=sim.state.contacts[0, :, ContactState.SLACK].tolist(),
                             proxy_slip_m=sim.state.contacts[0, :, ContactState.SLIP_CUM].tolist(),
                             proxy_clearance_m=float(result.table_clearance[0, 0]),
                             proxy_longest_hover_s=sim.longest_hover_s,
                             proxy_max_clearance_m=sim.max_clearance, proxy_peak_tension_N=sim.peak_tension.tolist(),
                             proxy_capture_s=sim.capture_time.copy(), proxy_release_s=sim.release_time.copy())


def summarize(trace):
    final = next((sample for sample in reversed(trace) if "proxy_peak_tension_N" in sample), {})
    keys = ("proxy_longest_hover_s", "proxy_max_clearance_m", "proxy_peak_tension_N",
            "proxy_capture_s", "proxy_release_s")
    return {key: final[key] for key in keys if key in final}


def dimensions(sim):
    return dict(nv=sim.m.nv, size_mm=(sim.robot.size * 1000.).tolist(), mass_kg=sim.robot.mass,
                proxy_envelope_sha256=sim.built.envelope_sha256, fill_fraction=sim.built.fill_fraction,
                slack_budget_mm=float(sim.parcel.proxy.params.S_bag[0, 0]) * 1000.)


def source_files():
    return dict(proxy=Path(lumped_contact_proxy.__file__), macro=Path(macro_parcel_sim.__file__))
