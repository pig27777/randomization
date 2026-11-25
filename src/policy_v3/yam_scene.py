"""Policy V3 scene composition using the shared ABC workcell and PE material."""

from dataclasses import asdict
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from yam_platform.abc_workcell import add_abc_workcell

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs"
SCENE = json.loads((CONFIG / "route_b_yam_scene.json").read_text())
WORKCELL = json.loads((CONFIG / "route_b_workcell.yaml").read_text())
BIN = json.loads((CONFIG / "route_b_single_arm_bin.yaml").read_text())["bin"]
PICK = np.array([-.03, -.225, 0.])
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def values(array):
    return " ".join(map(str, array))


def box(parent, name, pos, half, **attrs):
    return ET.SubElement(parent, "geom", name=name, pos=values(pos), size=values(half), type="box", **attrs)


def add_workcell(root):
    park = WORKCELL["arms"]["arm_b"]
    add_abc_workcell(root, SCENE, parked_arm=(park["abc_index"], park["ready_joints_rad"]))
    world = root.find("worldbody")
    center = np.array(BIN["center_m"])
    hx, hy = BIN["inner_half_xy_m"]
    t = BIN["wall_thickness_m"]
    z = BIN["rim_height_m"] / 2
    box(world, "bin_floor", center + [0, 0, .002], [hx, hy, .002], rgba=".16 .35 .46 1")
    for k, (x, y) in enumerate(((hx, 0), (-hx, 0), (0, hy), (0, -hy))):
        half = [t / 2, hy + t, z] if x else [hx, t / 2, z]
        box(world, f"bin_wall_{k}", center + [x, y, z], half, rgba=".20 .44 .56 1")


class RigidScene:
    payload_body = "payload"
    payload_geom = "payload_geom"
    config = {}
    compile_model = staticmethod(mujoco.MjModel.from_xml_string)
    initialize = None
    is_flex = False
    half = np.array([.026, .024, .0325])

    @staticmethod
    def add_scene(root):
        add_workcell(root)
        obj = ET.SubElement(root.find("worldbody"), "body", name="payload", pos=values(PICK + [0, 0, .0325]))
        ET.SubElement(obj, "freejoint", name="payload_free")
        box(obj, "payload_geom", [0, 0, 0], RigidScene.half, rgba=".80 .44 .19 1",
            mass=".12", friction="1 .005 .0005")

    def episode(self, model, initial, rng, scripted):
        state = initial.copy()
        addr = model.joint("payload_free").qposadr[0]
        xy = np.zeros(2) if scripted else rng.uniform([-.023, -.017], [.023, .017])
        yaw = 0. if scripted else rng.uniform(-.6, .6)
        state[addr:addr + 2] += xy
        state[addr + 3:addr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        return state, dict(payload_xy_offset_m=xy.tolist(), payload_yaw_rad=yaw, payload_mass_kg=.12)


class FlexScene:
    payload_body = "payload_0"
    payload_geom = "payload_geom_0"
    config = {"physics_dt_s": .0002}
    is_flex = True
    half = np.array([.088, .068, .024]) / 2

    def __init__(self):
        from bag_sim.physical_parcel import Config, Parcel, parcel_profile

        self.parcel_type = Parcel
        self.parcel_config = Config(geometry="geometry/kraft_edge_sealed_35.npz", bag_size=(.23, .17),
                                    payload_mass=.20, payload_size=tuple(2 * self.half),
                                    timestep=.0002, center=tuple(PICK[:2]), friction=.35,
                                    **parcel_profile("pe")["shell"])
        self.parcel = Parcel(self.parcel_config)
        self.compile_model = self.parcel.compile_model
        self.initialize = self.parcel.initialize

    def add_scene(self, root):
        add_workcell(root)
        parcel = ET.fromstring(self.parcel.xml)
        root.remove(root.find("option"))
        root.insert(1, parcel.find("option"))
        root.find("asset").extend(parcel.find("asset"))
        # Scope PE defaults to the PE geometry rather than changing ABC robot/fixture defaults.
        defaults = parcel.find("default")
        defaults.set("class", "policy_v3_pe")
        root.find("default").append(defaults)
        world = root.find("worldbody")
        # PE uses the qualified stiff planar contact. Keep the ABC table box
        # for rigid workcell collision and prevent a second film–table pair.
        table = world.find("geom[@name='table']")
        table.set("contype", "16")
        table.set("conaffinity", "9")
        support = parcel.find("worldbody/geom[@name='table']")
        support.attrib.update(name="table_flex_support", contype="0", conaffinity="2",
                              rgba="0 0 0 0")
        support.set("class", "policy_v3_pe")
        world.append(support)
        for node in parcel.find("worldbody"):
            if node.tag == "flexcomp":
                world.append(node)
            elif node.tag == "body" and node.get("name", "").startswith("payload_"):
                node.set("childclass", "policy_v3_pe")
                world.append(node)
        root.find(".//default[@class='collision']/geom").set("conaffinity", "9")
        for side in ("left", "right"):
            pad = root.find(f".//geom[@name='{side}_pad']")
            name = f"{side}_pad_convex"
            half = np.fromstring(pad.attrib.pop("size"), sep=" ")
            ET.SubElement(root.find("asset"), "mesh", name=name, vertex=values((half * CORNERS).ravel()))
            pad.set("type", "mesh")
            pad.set("mesh", name)

    def episode(self, model, initial, rng, scripted):
        from dataclasses import replace

        offset = float(0. if scripted else rng.uniform(-.018, .018))
        config = replace(self.parcel_config, payload_offset_xy=(offset, 0.))
        envelope = self.parcel_type.load_envelope(config)
        state = initial.copy()
        offsets = envelope["current"] - envelope["reference"]
        for index, delta in enumerate(offsets):
            body = model.body(f"bag_{index}").id
            joint = model.body_jntadr[body]
            state[model.jnt_qposadr[joint:joint + 3]] = delta
        addr = model.joint("payload_free").qposadr[0]
        state[addr:addr + 3] = envelope["payload_origin"]
        return state, asdict(config)

    def couple(self, sim, guard):
        import warp as wp
        from bag_sim.physical_parcel_gpu import bending

        p, d = self.parcel, sim.data
        dofs = p.node_dofs(sim.model)
        with wp.ScopedDevice(sim.device):
            args = [d.flexvert_xpos, d.qvel, wp.array(dofs[:, 0], dtype=int), wp.array(p.hinges, dtype=wp.vec4i),
                    wp.array(p.rest_angles, dtype=float), wp.array(p.hinge_stiffness, dtype=float),
                    wp.array(p.hinge_damping, dtype=float), d.qfrc_passive]

        def apply(model, data):
            wp.launch(bending, (sim.worlds, len(p.hinges)), args, device=sim.device)
            guard(model, data)

        sim.before_solve = apply
