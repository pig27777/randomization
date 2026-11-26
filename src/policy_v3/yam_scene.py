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
    # Device contact filter applies this only to the two pad–film pairs.
    pad_film_contact = (.004, .95, .999, .0003)
    half = np.array([.088, .068, .024]) / 2

    def __init__(self, solver_profile="nominal", *, timestep=.0002, iterations=None, ls_iterations=None,
                 tolerance=None, solver=None):
        from bag_sim.physical_parcel import Config, Parcel, parcel_profile

        if solver_profile not in ("nominal", "a2"):
            raise ValueError(f"Unknown Flex solver profile {solver_profile}")
        self.solver_profile = solver_profile
        self.config = {"physics_dt_s": timestep}
        self.solver_options = {}
        for key, value in (("iterations", iterations), ("ls_iterations", ls_iterations)):
            if value is not None:
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                self.solver_options[key] = str(value)
        if tolerance is not None:
            if not np.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("Solver tolerance must be finite and positive")
            self.solver_options["tolerance"] = str(tolerance)
        if solver is not None:
            if solver not in ("Newton", "CG"):
                raise ValueError(f"Unsupported Flex solver {solver}")
            self.solver_options["solver"] = solver
        self.parcel_type = Parcel
        self.parcel_config = Config(geometry="geometry/kraft_edge_sealed_35.npz", bag_size=(.23, .17),
                                    payload_mass=.20, payload_size=tuple(2 * self.half),
                                    timestep=timestep, center=tuple(PICK[:2]), friction=.35,
                                    **parcel_profile("pe")["shell"])
        self.parcel = Parcel(self.parcel_config)
        self.compile_model = self.parcel.compile_model
        self.initialize = self.parcel.initialize

    def add_scene(self, root):
        add_workcell(root)
        parcel = ET.fromstring(self.parcel.xml)
        root.remove(root.find("option"))
        root.insert(1, parcel.find("option"))
        if self.solver_profile == "a2":
            # Previously measured Flex A2 preset. Its margin enters the native contact solve.
            root.find("option").attrib.update(iterations="360", ls_iterations="75", impratio="8")
            parcel.find(".//flexcomp/contact").set("margin", ".0008")
        root.find("option").attrib.update(self.solver_options)
        root.find("asset").extend(parcel.find("asset"))
        # Scope PE defaults to the PE geometry rather than changing ABC robot/fixture defaults.
        defaults = parcel.find("default")
        defaults.set("class", "policy_v3_pe")
        root.find("default").append(defaults)
        world = root.find("worldbody")
        # The PE uses its planar support. The ABC box supplies rigid contacts.
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
        # Arm contact keeps the workcell and the film through their affinity,
        # but the core's bit 8 is not a collision partner of the arm.
        root.find(".//default[@class='collision']/geom").set("conaffinity", "1")
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
        state = self.place_envelope(model, initial, envelope["current"] - envelope["reference"],
                                    envelope["payload_origin"])
        return state, asdict(config)

    @staticmethod
    def place_envelope(model, initial, offsets, payload_origin, yaw=0.):
        state = initial.copy()
        for index, delta in enumerate(offsets):
            body = model.body(f"bag_{index}").id
            joint = model.body_jntadr[body]
            state[model.jnt_qposadr[joint:joint + 3]] = delta
        addr = model.joint("payload_free").qposadr[0]
        state[addr:addr + 3] = payload_origin
        state[addr + 3:addr + 7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        return state

    def randomized_episode(self, model, initial, rng):
        from bag_sim.legacy.contents import normalize_contents
        from bag_sim.wrap import triangle_box_intersections, winding_number, wrap_generic

        from .environment import MotionStop

        # A rigid core is rewrapped at its sampled yaw before any dynamics.
        offset = rng.uniform([-.018, -.010], [.018, .010])
        yaw = rng.uniform(-np.deg2rad(10.), np.deg2rad(10.))
        mass, friction = rng.uniform([.150, .28], [.400, .48])
        envelope = self.parcel.envelope
        shift = np.array([*self.parcel_config.center, 0.])
        reference = envelope["reference"] - shift
        faces = envelope["faces"]
        item = normalize_contents([dict(shape="box", size_m=list(self.parcel_config.payload_size),
                                        yaw_deg=float(np.rad2deg(yaw)), mass_kg=float(mass),
                                        friction=float(friction))])[0]
        current, preparation = wrap_generic(reference, reference.copy(), faces, envelope["panels"],
                                             [0], [3], item, self.parcel_config.radius,
                                             box_face_clearance=True, content_offset_xy=offset)
        center, rotation = item["position"], item["rotation"]
        corners = CORNERS * self.half @ rotation.T + center
        if (preparation["maximum_edge_strain"] > .05
                or triangle_box_intersections(current, faces, self.half, center, rotation).any()
                or np.any(np.abs(winding_number(current, faces, corners)) < .5)):
            raise MotionStop("initial_enclosure_failed", physics_valid=False)
        state = self.place_envelope(model, initial, current - reference, center + shift, yaw)
        return state, dict(payload_xy_offset_m=offset.tolist(), payload_yaw_rad=float(yaw),
                           payload_mass_kg=float(mass), payload_friction=float(friction))

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
