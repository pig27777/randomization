"""Existing two-sheet parcel through the shared full-YAM contact protocol."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experience")]
from bag_sim.physical_parcel import Config as ParcelConfig, Parcel, material_parameters, parcel_profile, vec  # noqa: E402
from bag_sim.bag_model import cushion_contact  # noqa: E402
from yam_platform.protocol import MuJoCoSim, TrialConfig, attach_robot, measure_contacts  # noqa: E402
from physical_parcel_checks import film_box_geometry, film_payload_geometry  # noqa: E402


@dataclass
class Config(TrialConfig):
    dt: float = 0.000025
    pinch_coeff: float = 0.0
    contact_time: float = 0.001
    mode: str = "rigid"
    geometry: str = "geometry/kraft_coarse_35.npz"
    bag_size: tuple = (0.35, 0.25)
    payload_size: tuple = (0.12, 0.08, 0.03)
    payload_mass: float = 0.3
    joint_stiffness: float = 0.08
    simulation_seconds: float = 30.0
    grasp_face: str = "top"
    support_height: float = 0.0
    side_start_distance: float = 0.1
    side_height_offset: float = 0.0
    initial_state: str | None = None
    bend_damping_ratio: float = 0.1
    rayleigh: float = 1e-4
    material: dict | None = None
    bag_profile: str | None = None
    friction: float = .45
    compression_k1: float = 800.
    compression_k2: float = 20000.
    compression_damping: float = 1.
    ccd_tolerance_m: float | None = None
    payload_offset_xy: tuple = (0.0, 0.0)

    def __post_init__(self):
        if self.grasp_face not in ("top", "side"):
            raise ValueError("grasp_face must be top or side")
        if self.mode == "empty" and self.grasp_face == "side":
            raise ValueError("the empty bag uses the top approach")
        if not np.isfinite(self.support_height) or self.support_height < 0:
            raise ValueError("support_height must be finite and nonnegative")
        if self.grasp_face == "side" and not 0 <= self.tilt < 90:
            raise ValueError("side grasp tilt must be a downward pitch in [0, 90)")
        if not np.isfinite(self.side_start_distance) or self.side_start_distance <= 0:
            raise ValueError("side_start_distance must be finite and positive")
        if not np.isfinite(self.side_height_offset):
            raise ValueError("side_height_offset must be finite")
        if self.ccd_tolerance_m is not None and (
                not np.isfinite(self.ccd_tolerance_m) or self.ccd_tolerance_m <= 0.):
            raise ValueError("ccd_tolerance_m must be positive and finite")


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
        self.film_dofs = self.parcel.node_dofs(self.m)
        self.payload_bodies = np.array([b for b in range(self.m.nbody)
            if self.m.body(b).name.startswith(("payload_", "plate_"))], dtype=int)
        self.floor_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        initialize_state(self.parcel, self.m, self.d, built.cfg)

    def step(self):
        mujoco.mj_step1(self.m, self.d)
        self.parcel.apply_internal(self.m, self.d, self.film_dofs)
        mujoco.mj_step2(self.m, self.d)

    def com(self):
        if not len(self.payload_bodies):
            return self.parcel.film_com(self.d)
        return np.average(self.d.xipos[self.payload_bodies], axis=0,
                          weights=self.m.body_mass[self.payload_bodies])


def compile_yam_model(parcel, built, xml):
    """Use complete convex film contact at the shared YAM's box-shaped pads."""
    root = ET.fromstring(xml)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for geom in root.find(".//body[@name='arm']").iter("geom"):
        if geom.get("class") != "visual":
            geom.set("solref", f"{built.cfg.contact_time} 1")
            geom.set("solimp", ".999 .999 .001")
    for side in ("left", "right"):
        core = root.find(f".//geom[@name='{side}_pad']")
        if core.get("type") == "box":
            half = np.fromstring(core.attrib.pop("size"), sep=" ")
            name = f"{side}_pad_convex"
            ET.SubElement(root.find("asset"), "mesh", name=name, vertex=vec(corners * half))
            core.set("type", "mesh")
            core.set("mesh", name)
    built.xml = ET.tostring(root, encoding="unicode")
    return parcel.compile_model(built.xml)


def parcel_world(cfg):
    profile = parcel_profile(cfg.bag_profile) if cfg.bag_profile is not None else None
    material = profile["shell"].copy() if profile is not None else {}
    if cfg.material is not None:
        material.update(material_parameters(**cfg.material))
    p = Parcel(ParcelConfig(mode=cfg.mode, geometry=cfg.geometry, bag_size=tuple(cfg.bag_size),
        payload_size=tuple(cfg.payload_size), payload_mass=0. if cfg.mode == "empty" else cfg.payload_mass,
        payload_offset_xy=tuple(cfg.payload_offset_xy),
        joint_stiffness=cfg.joint_stiffness, center=(-0.25, 0.04),
        timestep=cfg.dt, contact_time=cfg.contact_time,
        bend_damping_ratio=cfg.bend_damping_ratio, rayleigh=cfg.rayleigh,
        friction=cfg.friction, compression_k1=cfg.compression_k1,
        compression_k2=cfg.compression_k2, compression_damping=cfg.compression_damping,
        **material))
    root = ET.fromstring(p.xml)
    if cfg.ccd_tolerance_m is not None:
        root.find("option").set("ccd_tolerance", str(cfg.ccd_tolerance_m))
    if profile is not None:
        asset = root.find("asset")
        ET.SubElement(asset, "texture", name="bag_texture", type="2d", file=str(profile["texture"]))
        ET.SubElement(asset, "material", name="bag_material", texture="bag_texture", rgba="1 1 1 1",
                      specular=".05", shininess=".1")
        film = root.find(".//flexcomp[@name='bag']")
        film.set("material", "bag_material")
        film.set("rgba", "1 1 1 .65" if cfg.bag_profile == "pe" else "1 1 1 1")
        reference = p.envelope["reference"][:, :2]
        film.set("texcoord", vec((reference-reference.min(0)) / np.ptp(reference, axis=0)))
        if profile["cushion"] is not None:
            solref, solimp = cushion_contact(profile["cushion"], p.cfg.areal_mass)
            contact = film.find("contact")
            contact.set("priority", "2")
            contact.set("solref", vec(solref))
            contact.set("solimp", vec(solimp))
        p.metadata["bag_profile"] = dict(name=cfg.bag_profile, source_sha256=profile["source_sha256"],
                                         cushion=profile["cushion"] is not None)
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
    return p, ET.tostring(root, encoding="unicode")


def prepared_signature(parcel, cfg):
    model = asdict(parcel.cfg)
    # Prepared archives from before positioning support use the centered payload.
    if not any(parcel.cfg.payload_offset_xy):
        model.pop("payload_offset_xy")
    settings = dict(model=model, support_height=cfg.support_height)
    if cfg.ccd_tolerance_m is not None:
        settings["ccd_tolerance_m"] = cfg.ccd_tolerance_m
    return json.dumps(settings, sort_keys=True)


def initialize_state(parcel, model, data, cfg):
    parcel.initialize(model, data)
    dofs = parcel.node_dofs(model)
    data.qpos[dofs[:, 2]] += cfg.support_height
    if cfg.mode != "empty":
        data.qpos[model.joint("payload_free").qposadr[0] + 2] += cfg.support_height
    if cfg.initial_state:
        with np.load(cfg.initial_state) as state:
            qpos, qvel = state["qpos"], state["qvel"]
            if state["config_json"].item() != prepared_signature(parcel, cfg):
                raise ValueError("prepared state uses a different parcel or support configuration")
            if qpos.shape != (parcel.m.nq,) or qvel.shape != (parcel.m.nv,):
                raise ValueError("prepared state has incompatible dimensions")
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError("prepared state contains nonfinite values")
            # The shared assembler appends robot joints after the parcel joints.
            data.qpos[:len(qpos)] = qpos
            data.qvel[:len(qvel)] = qvel
    parcel.forward(model, data, dofs)


def save_initial_state(parcel, data, cfg, path):
    np.savez_compressed(path, qpos=data.qpos, qvel=data.qvel, config_json=np.asarray(prepared_signature(parcel, cfg)))


def scene(cfg, *, sim_factory=ParcelSim):
    p, xml = parcel_world(cfg)
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
    built = SimpleNamespace(xml=xml,
        desc=SimpleNamespace(), cfg=cfg, parcel=p,
        compile_model=lambda xml: compile_yam_model(p, built, xml))
    height = np.ptp(current[:, 2]) + 2 * p.cfg.radius if cfg.mode == "empty" else cfg.payload_size[2]
    robot = attach_robot(cfg, built, sim_factory, np.array(p.cfg.center), top,
                        np.array([*cfg.bag_size, height]),
                        p.cfg.payload_mass + p.envelope["masses"].sum(), approach=approach)
    initial = measure(robot)
    if initial["penetration_mm"] > 1:
        raise RuntimeError(f"initial film contact penetration exceeds 1 mm: {initial['penetration_mm']:.6g} mm")
    return robot


def measure(robot):
    flex = mujoco.mj_name2id(robot.sim.m, mujoco.mjtObj.mjOBJ_FLEX, "bag")
    result = measure_contacts(robot, target_flexes={flex},
                              monitor_support=robot.sim.built.cfg.press_target_N is not None)
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
                film_triangles=sim.m.nflexelem, payload_bodies=len(sim.payload_bodies),
                artificial_attachments=0)


def refresh(sim):
    sim.parcel.forward(sim.m, sim.d, sim.film_dofs)


def summarize(trace):
    has_payload = any(row.get("payload_geometry", {}).get("boxes") for row in trace)
    return dict(load_transfer="native finger-film-payload contacts" if has_payload else "native finger-film contacts",
                attachments=0)


def source_files():
    return {name: ROOT / path for name, path in dict(
        parcel_model="bag_sim/physical_parcel.py", parcel_material="bag_sim/bag_model.py",
        parcel_wrap="bag_sim/wrap.py", parcel_geometry_check="test/bag_model/physical_parcel_checks.py").items()}


def save_model(sim, output):
    mujoco.mj_saveModel(sim.m, str(output / "model.mjb"), None)
