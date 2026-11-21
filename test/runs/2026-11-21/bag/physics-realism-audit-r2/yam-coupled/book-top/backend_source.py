"""Run Stage 2.5 audit cases through the shared articulated YAM protocol."""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experience.yam_platform import protocol
from src.low_order_parcel import yam_adapter
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction
from src.low_order_parcel.unified_payload import UnifiedPayloadTemplate
from test_unified_payload import model, pe_proxy


@dataclass
class Config(yam_adapter.Config):
    preset: str = "rigid_box"
    fill: float = .75
    indent: float = 0.
    friction: float = .40
    mass: float = .24
    entry: float = .012
    insertion: float = .015
    yaw: float = 270.
    grasp_opening_m: float = .045
    lift_hold_seconds: float = .45
    simulation_seconds: float = 21.
    pickup_release_seconds: float = .8
    close_gap_hold_s: float = .20


class StepForces:
    """Observe the exact boundary forces used by the frozen macro step."""

    def __init__(self, sim):
        self.sim = sim
        self.proxy_step = sim.proxy.step
        self.surface_step = sim.payload.surface_contact
        self.side_step = sim._side_contact
        self.payload_step = sim.payload.step
        sim.proxy.step = self.contact_step
        sim.payload.surface_contact = self.surface_contact
        sim._side_contact = self.side_contact
        sim.payload.step = self.advance

    def contact_step(self, state, inputs):
        self.inputs = inputs
        self.contact = self.proxy_step(state, inputs)
        return self.contact

    def surface_contact(self, *args):
        self.surface = self.surface_step(*args)
        return self.surface

    def side_contact(self, *args):
        result = self.side_step(*args)
        self.side = result[2]
        return result

    def advance(self, state, force, depth):
        self.block_force = force
        return self.payload_step(state, force, depth)

    def reaction(self, old, action, result):
        payload = self.sim.payload
        tip = action.pinch_position
        local = torch.bmm(tip - old.payload.pos.mean(dim=1, keepdim=True), payload.frame(old.payload))
        weights = payload.weights_at(local[:, :, :2])
        normal = payload.frame(old.payload)[:, :, 2]
        direct = (.5 * action.direct_press_N -
                  (self.surface.pad_force_N * normal[:, 2:3]).clamp_min(0.)).clamp_min(0.)
        jaw = tip[:, 1] - tip[:, 0]
        axis = jaw / jaw.norm(dim=1, keepdim=True).clamp_min(1e-8)
        crush = self.inputs.normal_force - action.normal_force
        jaw_force = crush[:, :, None] * torch.stack((axis, -axis), dim=1)
        top_force = self.surface.pad_force_N.unsqueeze(-1) * normal[:, None] + direct.unsqueeze(-1) * self.sim.up[:, None]
        reaction = -self.contact.force - jaw_force - self.side + top_force
        arm = tip - tip.mean(dim=1, keepdim=True)
        wrench = torch.cat((reaction.sum(dim=1), torch.linalg.cross(arm, reaction).sum(dim=1)), dim=1)
        torch.testing.assert_close(wrench, result.wrist_wrench, atol=1e-9, rtol=1e-9)
        return reaction, jaw_force, top_force, weights


class AuditYamSim(protocol.MuJoCoSim):
    def __init__(self, built):
        super().__init__(built)
        self.parcel = built.parcel
        self.state = self.parcel.initial_state()
        shift = self.state.com.new_tensor([*built.center, 0.])
        self.state = self.state._replace(com=self.state.com + shift, side_origin=self.state.side_origin + shift)
        self.state.payload.pos.add_(shift)
        self.last_step = None
        self.reactions = np.zeros((2, 3))
        self.grip = np.zeros(2)
        self.normal = np.zeros(2)
        self.feed = np.zeros(2)
        self.max_clearance = self.bilateral_hover_s = self.longest_hover_s = 0.
        self.peak_tension = np.zeros(2)
        self.capture_time = [None, None]
        self.release_time = [None, None]
        self.failure = None
        self.boundary = StepForces(self.parcel)
        self.samples = []

    def bind(self, robot):
        self.robot = robot
        self.pad_position, _ = yam_adapter.ProxyYamSim._inner_tips(self)
        self.fingers = [self.m.body(f"{side}_finger").id for side in ("left", "right")]

    def com(self):
        return self.state.com[0].numpy().copy()

    @torch.inference_mode()
    def step(self):
        dt = self.parcel.dt
        pad, axis = yam_adapter.ProxyYamSim._inner_tips(self)
        velocity = (pad - self.pad_position) / dt
        self.pad_position = pad
        outward = np.stack((-axis, axis))
        inward = -((velocity - velocity.mean(0)) * outward).sum(1)
        old = self.state
        tensor = old.com.new_tensor
        tip = tensor(pad[None])
        payload = self.parcel.payload
        local = torch.bmm(tip - old.payload.pos.mean(dim=1, keepdim=True), payload.frame(old.payload))[0]
        inside = (local[:, :2].abs() <= self.parcel.half_extents[0, :2]).all(dim=1)
        weights = payload.weights_at(local[None, :, :2])
        thickness = torch.bmm(weights, payload.half_thickness(old.payload.d_surf).unsqueeze(-1))[0, :, 0]
        captured = old.contacts[0, :, ContactState.CAPTURE] > .5
        compression = self.built.cfg.fold_normal_stiffness_N_m * max(
            0., (self.built.cfg.fold_width_m - self.robot.gap()) / 2.)
        feed = np.zeros(2)
        alignment = np.zeros(2)
        if self.built.cfg.grasp_face == "side":
            # Gross feed is measured from the original wrapper wall. The frozen
            # model subtracts parcel translation after its side contact latches.
            feed = (pad[:, 1] - self.built.entry_y).clip(min=0.)
            alignment[:] = abs(axis[0])
            touch = (tensor(feed) > 0.) & (local[:, 0].abs() <= self.parcel.half_extents[0, 0])
        else:
            touch = inside & (local[:, 2] <= thickness + .004) & (local[:, 2] > -thickness)
        self.normal = compression * (touch | captured).numpy()
        action = ParcelAction(tip, tensor(velocity[None]), tensor(self.normal[None]), tensor(inward[None]),
                              tensor(feed[None]), tensor(alignment[None]), torch.zeros_like(old.com),
                              torch.zeros_like(old.indentation_power))
        result = self.parcel.step(old, action)
        reaction, _, _, _ = self.boundary.reaction(old, action, result)
        self.reactions = reaction[0].numpy() + self.normal[:, None] * outward
        self.d.qfrc_applied[:] = 0.
        for force, point, body in zip(self.reactions, pad, self.fingers, strict=True):
            mujoco.mj_applyFT(self.m, self.d, force, np.zeros(3), point, body, self.d.qfrc_applied)
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_kinematics(self.m, self.d)
        self.last_step, self.state = result, result.state
        self.feed = self.state.contacts[0, :, ContactState.FEED_DEPTH].numpy()
        self.grip = np.maximum(0., (self.reactions * outward).sum(axis=1))
        yam_adapter.ProxyYamSim.track_events(self, captured.numpy())
        self.samples.append((float(self.d.time), action, old, result))
        if not all(torch.isfinite(x).all() for x in self.state.payload):
            self.failure = "nonfinite Stage 2.5 state"
            raise RuntimeError(self.failure)


def scene(cfg):
    params = model(cfg.preset, dtype=torch.float64, bag_half_extents=torch.tensor([[.085, .105, .05]], dtype=torch.float64)).params
    params = replace(params, **{field: params.total_mass.new_tensor([[value]]) for field, value in
                               (("fill_ratio", cfg.fill), ("max_indent_ratio", cfg.indent),
                                ("table_friction", cfg.friction), ("total_mass", cfg.mass))})
    parcel = MacroParcelSim(pe_proxy(1), payload=UnifiedPayloadTemplate(params))
    center = np.array([-.25, .04])
    size = 2. * params.bag_half_extents[0].numpy()
    xml = """<mujoco model="YAM Stage 2.5 physical audit">
    <option timestep=".002" integrator="implicitfast"/>
    <worldbody><geom name="table" type="box" pos="0 0 -.025" size="1 .24 .025"/></worldbody>
    <contact/><equality/></mujoco>"""
    wall = center[1] - size[1] / 2.
    built = SimpleNamespace(xml=xml, desc=SimpleNamespace(), cfg=cfg, parcel=parcel, center=center, entry_y=wall)
    approach = None
    if cfg.grasp_face == "side":
        angle = np.deg2rad(25.)
        direction = np.array([0., np.cos(angle), -np.sin(angle)])
        point = np.array([center[0], wall, size[2] + cfg.entry])
        approach = dict(direction=direction, closing=[-1., 0., 0.], contact_point=point,
                        close_advance_m=.10 + cfg.insertion / direction[1])
    robot = protocol.attach_robot(cfg, built, AuditYamSim, center, size[2], size, cfg.mass, approach)
    robot.sim.bind(robot)
    return robot


def measure(robot):
    measured = yam_adapter.measure(robot)
    measured["direct_support_N"] = sum(measured["table_contact_each_N"])
    return measured


def check_state(sim):
    failure, health = yam_adapter.check_state(sim)
    if sim.last_step is not None:
        health["indent_mm"] = 1000. * float(sim.state.payload.d_surf.max())
        health["tether_energy_J"] = sim.last_step.elastic_energy[0].tolist()
    return failure, health


def dimensions(sim):
    return dict(nv=sim.m.nv, blocks=4, size_mm=(1000. * sim.robot.size).tolist(), mass_kg=sim.robot.mass,
                proxy_slack_budget_mm=1000. * float(sim.parcel.base_slack[0, 0]))


attachment = yam_adapter.attachment
summarize = yam_adapter.summarize


def source_files():
    from src.low_order_parcel import unified_payload
    return {**yam_adapter.source_files(), "payload": Path(unified_payload.__file__)}


def cases():
    top = Config(press_target_N=12., lift_height=.12, press_start_offset_m=.07)
    side = Config(grasp_face="side", lift_height=.12)
    return {
        "book-top": replace(top, press_only=True, press_target_N=5.),
        "clothes-top": replace(top, preset="folded_clothes", indent=.35),
        "pillow-top": replace(top, preset="fluffy_pillow", indent=.70),
        "heavy-side": replace(side, mass=1.5, insertion=.023),
        "light-side": replace(side, mass=.08, insertion=.008),
        "low-side-push": replace(side, mass=.065, friction=.06, entry=-.014),
        "raised-side-entry": replace(side, mass=.065, friction=.06, entry=.014),
    }


def run_case(name, output):
    robot = None

    def record(current, sample):
        nonlocal robot
        robot = current

    result = protocol.run(sys.modules[__name__], cases()[name], output, on_sample=record)
    if robot is not None:
        samples = robot.sim.samples
        time, actions, _, results = zip(*samples, strict=True)
        data = {f"action_{field}": torch.cat([getattr(a, field) for a in actions]).numpy()
                for field in ParcelAction._fields}
        np.savez_compressed(output / "boundary.npz", time_s=time, **data,
                            capture=torch.cat([r.state.contacts[:, :, ContactState.CAPTURE] for r in results]).numpy(),
                            clearance_m=torch.cat([r.table_clearance for r in results]).numpy(),
                            indentation_m=torch.cat([r.state.payload.d_surf for r in results]).numpy())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="*", choices=tuple(cases()))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    for case in args.cases or cases():
        run_case(case, args.output / case)
