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
from bag_sim.physical_parcel import Config as ParcelConfig, Parcel, vec  # noqa: E402
from yam_platform.protocol import MuJoCoSim, TrialConfig, attach_robot, measure_contacts  # noqa: E402
from physical_parcel_checks import film_box_geometry, film_payload_geometry  # noqa: E402


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
    grasp_face: str = "top"
    support_height: float = 0.0
    side_start_distance: float = 0.1
    side_height_offset: float = 0.0

    def __post_init__(self):
        if self.grasp_face not in ("top", "side"):
            raise ValueError("grasp_face must be top or side")
        if not np.isfinite(self.support_height) or self.support_height < 0:
            raise ValueError("support_height must be finite and nonnegative")
        if self.grasp_face == "side" and not 0 <= self.tilt < 90:
            raise ValueError("side grasp tilt must be a downward pitch in [0, 90)")
        if not np.isfinite(self.side_start_distance) or self.side_start_distance <= 0:
            raise ValueError("side_start_distance must be finite and positive")
        if not np.isfinite(self.side_height_offset):
            raise ValueError("side_height_offset must be finite")


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
        height = built.cfg.support_height
        self.d.qpos[self.film_dofs[:, 2]] += height
        payload_qpos = self.m.joint("payload_free").qposadr[0]
        self.d.qpos[payload_qpos + 2] += height
        self.floor_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.parcel.forward(self.m, self.d, self.film_dofs)

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
    world = root.find("worldbody")
    world.remove(root.find(".//body[@name='contact_probe']"))
    if cfg.support_height:
        table = world.find("geom[@name='table']")
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        half = np.array([.035, .025, cfg.support_height / 2])
        ET.SubElement(root.find("asset"), "mesh", name="raised_support", vertex=vec(corners * half))
        table.set("type", "mesh")
        table.set("mesh", "raised_support")
        table.attrib.pop("size")
        table.set("pos", f"{p.cfg.center[0]} {p.cfg.center[1]} {cfg.support_height / 2}")
        ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .05",
                      contype="1", conaffinity="2", rgba=".5 .5 .5 1")
    current = p.envelope["current"]
    top = float(current[:, 2].max() + p.cfg.radius + cfg.support_height)
    approach = None
    if cfg.grasp_face == "side":
        pitch = np.deg2rad(cfg.tilt)
        yaw = np.deg2rad(cfg.yaw)
        inward = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        up = np.array([0., 0., 1.])
        center = p.envelope["payload_origin"] + [0, 0, cfg.support_height]
        projected = np.abs(inward[:2])
        distances = np.divide(np.asarray(cfg.payload_size[:2]) / 2, projected,
                              out=np.full(2, np.inf), where=projected > 1e-9)
        direction = np.cos(pitch) * inward - np.sin(pitch) * up
        contact = center - inward * distances.min() + up * cfg.side_height_offset
        # The shared entry backs off 100 mm from its anchor. Shift that anchor
        # to start at the configured physical clearance from the chosen face.
        approach = dict(direction=direction,
                        closing=np.sin(pitch) * inward + np.cos(pitch) * up,
                        contact_point=contact + (.1 - cfg.side_start_distance) * direction)
    built = SimpleNamespace(xml=ET.tostring(root, encoding="unicode"),
        desc=SimpleNamespace(), cfg=cfg, parcel=p, compile_model=p.compile_model)
    return attach_robot(cfg, built, ParcelSim, np.array(p.cfg.center), top,
                        np.array([*cfg.bag_size, cfg.payload_size[2]]),
                        cfg.payload_mass + p.envelope["masses"].sum(), approach=approach)


def measure(robot):
    flex = mujoco.mj_name2id(robot.sim.m, mujoco.mjtObj.mjOBJ_FLEX, "bag")
    result = measure_contacts(robot, target_flexes={flex})
    if robot.sim.floor_geom < 0:
        return result
    floor_support = 0.0
    buffer = np.zeros(6)
    for index, contact in enumerate(robot.sim.d.contact):
        if robot.sim.floor_geom not in contact.geom or flex not in contact.flex:
            continue
        if contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(robot.sim.m, robot.sim.d, index, buffer)
        world = contact.frame.reshape(3, 3).T @ buffer[:3]
        floor_support += (1 if contact.geom[0] == robot.sim.floor_geom else -1) * world[2]
    result.update(post_support_N=result["support_N"], floor_support_N=float(floor_support))
    result["support_N"] += floor_support
    return result


def attachment(robot, cfg):
    return ContactOnly()


def gathering_spring(robot, cfg):
    return ContactOnly()


def check_state(sim):
    finite = np.isfinite(sim.d.qpos).all() and np.isfinite(sim.d.qvel).all()
    warning = int(sim.d.warning.number.sum())
    reason = "nonfinite state" if not finite else "native warning" if warning else None
    if reason:
        return reason, dict(finite=bool(finite), warnings=warning)
    geometry = film_payload_geometry(sim.m, sim.d, sim.parcel.envelope["faces"])
    if geometry["geometric_depth_upper_m"] > .001 or geometry["outside_corners"]:
        reason = "film/payload geometry exceeds 1 mm or exposes a payload corner"
    health = dict(finite=bool(finite), warnings=warning, payload_geometry=geometry)
    if sim.built.cfg.support_height:
        support = film_box_geometry(sim.m, sim.d, sim.parcel.envelope["faces"], sim.table_geom)
        health["support_geometry"] = support
        if support["geometric_depth_upper_m"] > .001:
            reason = "film/support geometry exceeds 1 mm"
    return reason, health


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
        parcel_wrap="bag_sim/wrap.py", parcel_geometry_check="test/bag_model/physical_parcel_checks.py").items()}


def save_model(sim, output):
    mujoco.mj_saveModel(sim.m, str(output / "model.mjb"), None)
