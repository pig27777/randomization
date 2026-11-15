"""Existing two-sheet parcel through the shared full-YAM contact protocol."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experience")]
from bag_sim.physical_parcel import Config as ParcelConfig, Parcel  # noqa: E402
from yam_platform.protocol import MuJoCoSim, TrialConfig, attach_robot, measure_contacts  # noqa: E402


@dataclass
class Config(TrialConfig):
    dt: float = 0.000025
    pinch_coeff: float = 0.0
    contact_time: float = 0.001
    mode: str = "rigid"
    bag_size: tuple = (0.35, 0.25)
    payload_size: tuple = (0.12, 0.08, 0.03)
    payload_mass: float = 0.3
    joint_stiffness: float = 0.08
    simulation_seconds: float = 30.0


class ContactOnly:
    active = None
    load = 0.0
    events = ()
    reference_gap = None

    def try_bind(self, tips):
        return False

    def check_load(self, *args):
        return False

    def release(self, *args):
        pass

    def arm(self, *args):
        pass

    def force(self, gap):
        return 0.0


class ParcelSim(MuJoCoSim):
    def __init__(self, built):
        super().__init__(built)
        self.parcel = built.parcel
        self.parcel.initialize(self.m, self.d)
        self.film_dofs = self.parcel.node_dofs(self.m)
        self.payload_bodies = np.array([b for b in range(self.m.nbody)
            if self.m.body(b).name.startswith(("payload_", "plate_"))])

    def step(self):
        mujoco.mj_step1(self.m, self.d)
        self.parcel.apply_internal(self.m, self.d, self.film_dofs)
        mujoco.mj_step2(self.m, self.d)

    def com(self):
        return np.average(self.d.xipos[self.payload_bodies], axis=0,
                          weights=self.m.body_mass[self.payload_bodies])


def scene(cfg):
    p = Parcel(ParcelConfig(mode=cfg.mode, bag_size=tuple(cfg.bag_size),
        payload_size=tuple(cfg.payload_size), payload_mass=cfg.payload_mass,
        joint_stiffness=cfg.joint_stiffness, center=(-0.25, 0.04),
        timestep=cfg.dt, contact_time=cfg.contact_time))
    root = ET.fromstring(p.xml)
    root.find("worldbody").remove(root.find(".//body[@name='contact_probe']"))
    current = p.envelope["current"]
    top = float(current[:, 2].max() + p.cfg.radius)
    built = SimpleNamespace(xml=ET.tostring(root, encoding="unicode"),
        desc=SimpleNamespace(), cfg=cfg, parcel=p, compile_model=p.compile_model)
    return attach_robot(cfg, built, ParcelSim, np.array(p.cfg.center), top,
                        np.array([*cfg.bag_size, cfg.payload_size[2]]),
                        cfg.payload_mass + p.envelope["masses"].sum())


def measure(robot):
    flex = mujoco.mj_name2id(robot.sim.m, mujoco.mjtObj.mjOBJ_FLEX, "bag")
    return measure_contacts(robot, target_flexes={flex})


def attachment(robot, cfg):
    return ContactOnly()


def gathering_spring(robot, cfg):
    return ContactOnly()


def check_state(sim):
    finite = np.isfinite(sim.d.qpos).all() and np.isfinite(sim.d.qvel).all()
    warning = int(sim.d.warning.number.sum())
    reason = "nonfinite state" if not finite else "native warning" if warning else None
    return reason, dict(finite=bool(finite), warnings=warning)


def dimensions(sim):
    return dict(nv=sim.m.nv, film_vertices=sim.m.nflexvert,
                film_triangles=sim.m.nflexelem, artificial_attachments=0)


def refresh(sim):
    sim.parcel.forward(sim.m, sim.d, sim.film_dofs)


def summarize(trace):
    return dict(load_transfer="native finger-film-payload contacts", attachments=0)


def source_files():
    return {name: ROOT / path for name, path in dict(
        parcel_model="bag_sim/physical_parcel.py", parcel_material="bag_sim/bag_model.py",
        parcel_wrap="bag_sim/wrap.py").items()}


def save_model(sim, output):
    mujoco.mj_saveModel(sim.m, str(output / "model.mjb"), None)
