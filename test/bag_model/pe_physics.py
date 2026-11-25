"""Run the robot-free PE film physics benchmark with the existing Parcel model."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bag_sim.bag_model import _topology, dihedrals
from bag_sim.physical_parcel import Config, Parcel, load_envelope, parcel_profile, vec

PAD = np.array([.0144, .0164])
GEOMETRY = "geometry/kraft_edge_sealed_35.npz"
BASE = Config(geometry=GEOMETRY, mode="empty", friction=.35, **parcel_profile("pe")["shell"])
FIXTURES = ("contact_probe", "platen", "left_pad", "right_pad", "obstacle")
PAIRS = ("film_self", "film_table", "film_payload", "film_contact_probe", "film_platen",
         "film_left_pad", "film_right_pad", "payload_table", "payload_fixture")
MAX_STRETCH = 1.05
MAX_PENETRATION = .001


def save_json(path, contents):
    path.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")


def coupon(cfg, layers=1):
    """80 × 40 mm film, 10 mm grid, with the same constitutive settings as the bag."""
    x, y = np.meshgrid(np.linspace(-.04, .04, 9), np.linspace(-.02, .02, 5), indexing="ij")
    sheet = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    a = (np.arange(8)[:, None] * 5 + np.arange(4)).ravel()
    sheet_faces = np.concatenate((np.column_stack((a, a + 5, a + 6)),
                                  np.column_stack((a, a + 6, a + 1))))
    points = np.concatenate([sheet + (0, 0, i * .0015) for i in range(layers)])
    faces = np.concatenate([sheet_faces + i * len(sheet) for i in range(layers)])
    edges, _, _, _, _, areas, _ = _topology(points, faces)
    masses = np.zeros(len(points))
    np.add.at(masses, faces.ravel(), np.repeat(areas * cfg.areal_mass / 3, 3))
    return dict(reference=points, current=points.copy(), faces=faces, edges=edges,
                panels=np.repeat(np.arange(layers), len(sheet_faces)), masses=masses,
                payload_origin=None, metadata=dict(vertices=len(points), triangles=len(faces),
                reference_area_m2=float(areas.sum()), film_mass_kg=float(masses.sum())))


def parcel_config(spec):
    return replace(BASE, mode="rigid" if spec.mass else "empty", timestep=spec.dt,
                   iterations=spec.iterations, gravity=spec.gravity, payload_mass=spec.mass,
                   payload_size=spec.size, payload_offset_xy=spec.offset,
                   self_collision=spec.self_collision)


def prepared_envelope(spec, cfg):
    if spec.coupon:
        film = coupon(cfg, spec.layers)
        if spec.camber:
            x = film["reference"][:, 0]
            film["current"][:, 2] += spec.camber * np.sin(np.pi * (x + .04) / .08)
        if spec.fold_angles:
            angle = np.deg2rad(spec.fold_angles)
            nodes = .01 * np.vstack((np.zeros(2), np.cumsum(np.column_stack((np.cos(angle), np.sin(angle))), axis=0)))
            nodes[:, 0] -= nodes[:, 0].mean()
            film["current"][:, [0, 2]] = np.repeat(nodes, 5, axis=0)
    elif spec.yaw:
        # Loose initial wrap around the bounding box of the rotated core.
        # Only the specified, smaller rotated solid is inserted in the scene.
        c, s = abs(np.cos(spec.yaw)), abs(np.sin(spec.yaw))
        x, y, z = spec.size
        film = load_envelope(replace(cfg, payload_size=(c * x + s * y, s * x + c * y, z)))
        film["metadata"]["bounding_box_wrap_yaw_rad"] = spec.yaw
    else:
        film = load_envelope(cfg)
    film["current"][:, 2] += spec.height
    if film["payload_origin"] is not None:
        film["payload_origin"][2] += spec.height
    return film


class FlexBagBackend(Parcel):
    @classmethod
    def reset(cls, config, seed=0):
        return cls(config, seed)

    def __init__(self, config, seed):
        self.spec = config
        self.seed = seed
        super().__init__(parcel_config(config))
        self.film_qpos = self.m.jnt_qposadr[self.m.dof_jntid[self.film_dofs]]
        self.fixture_ids = {name: self.m.body(name).id for name in FIXTURES}
        self.fixture_joints = {name: self.m.body_jntadr[index] for name, index in self.fixture_ids.items()}
        self.clamp_ids = [i for i in range(self.m.neq) if self.m.eq(i).name.startswith("clamp_")]
        self.last_action = {}

    def load_envelope(self, cfg):
        return prepared_envelope(self.spec, cfg)

    def compile_model(self, xml):
        tree = ET.fromstring(xml)
        world = tree.find("worldbody")
        world.find("flexcomp/contact").set("solref", f"{self.spec.self_contact_time} 1")
        if not self.spec.table:
            world.remove(world.find("geom[@name='table']"))
        else:
            table = world.find("geom[@name='table']")
            if not self.spec.plane_table:
                table.set("type", "box")
                table.set("pos", "0 0 -.01")
                table.set("size", ".35 .35 .01")
            table.set("conaffinity", "10")
        world.remove(world.find("body[@name='contact_probe']"))
        if self.spec.yaw:
            world.find("body[@name='payload_0']").set("quat", vec(yaw_quaternion(self.spec.yaw)))
        sizes = (("sphere", ".01"), ("box", ".20 .15 .005"),
                 ("box", vec([.003, PAD[0] / 2, PAD[1] / 2])),
                 ("box", vec([.003, PAD[0] / 2, PAD[1] / 2])), ("box", ".025 .025 .015"))
        for index, (name, (shape, size)) in enumerate(zip(FIXTURES, sizes)):
            p = self.spec.initial_tools.get(name, (0, 0, 1 + index))
            body = ET.SubElement(world, "body", name=name, pos=vec(p), gravcomp="1")
            ET.SubElement(body, "freejoint", name=name)
            ET.SubElement(body, "inertial", pos="0 0 0", mass="10", diaginertia="1 1 1")
            ET.SubElement(body, "geom", name=name, type=shape, size=size, contype="4",
                          conaffinity="2", friction=f"{self.spec.tool_mu} .005 .0001",
                          rgba=".20 .55 .80 1")
        eq = tree.find("equality")
        for label, ids in self.spec.clamped_nodes(self.envelope).items():
            for vertex in ids:
                ET.SubElement(eq, "connect", name=f"clamp_{label}_{vertex}", body1=f"bag_{vertex}",
                              anchor="0 0 0", solref=".00015 1", solimp=".9999 .9999 .0001")
        self.xml = ET.tostring(tree, encoding="unicode")
        return super().compile_model(self.xml)

    def initialize(self, model, data):
        super().initialize(model, data)
        # XML anchors were defined at the undeformed reference. Hold prescribed
        # material boundary points at their initialized world positions.
        for index in range(model.neq):
            vertex = int(model.eq(index).name.rsplit("_", 1)[1])
            model.eq_data[index, 3:6] = self.envelope["current"][vertex]
        mujoco.mj_forward(model, data)

    def step(self, action):
        self.last_action = action
        for name, (position, velocity, quaternion) in action["tools"].items():
            joint = self.fixture_joints[name]
            q, v = self.m.jnt_qposadr[joint], self.m.jnt_dofadr[joint]
            self.d.qpos[q:q + 3] = position
            self.d.qpos[q + 3:q + 7] = quaternion
            self.d.qvel[v:v + 6] = [*velocity, 0., 0., 0.]
        for index in self.clamp_ids:
            _, label, vertex = self.m.eq(index).name.split("_")
            self.m.eq_data[index, 3:6] = self.envelope["current"][int(vertex)] + action["clamps"].get(label, 0)
        self.d.qfrc_applied[:] = 0
        ids = self.spec.loaded_nodes(self.envelope)
        if len(ids):
            self.d.qfrc_applied[self.film_dofs[ids]] = action["load"] / len(ids)
        super().step()

    def get_bag_state(self):
        # Position kinematics are evaluated before integration in mj_step2.
        # Nodal free slide displacements give the synchronized post-step state.
        return self.envelope["reference"] + self.d.qpos[self.film_qpos], self.d.qvel[self.film_dofs].copy()

    def get_payload_state(self):
        if not self.spec.mass:
            return np.empty(0), np.empty(0)
        joint = self.m.joint("payload_free").id
        q, v = self.m.jnt_qposadr[joint], self.m.jnt_dofadr[joint]
        return self.d.qpos[q:q + 7].copy(), self.d.qvel[v:v + 6].copy()

    def get_fixture_state(self):
        return np.array([self.d.qpos[self.m.jnt_qposadr[j]:self.m.jnt_qposadr[j] + 7]
                         for j in self.fixture_joints.values()])

    def probe_normal_reaction(self):
        geom = self.m.geom("contact_probe").id
        force = np.zeros(6)
        reaction = 0.
        for i, contact in enumerate(self.d.contact):
            if contact.efc_address < 0 or (geom != contact.geom[0] and geom != contact.geom[1]):
                continue
            mujoco.mj_contactForce(self.m, self.d, i, force)
            reaction += (1 if geom == contact.geom[1] else -1) * (contact.frame[2::3] @ force[:3])
        return reaction

    def get_contact_wrench(self):
        forces = np.zeros((len(FIXTURES) + 2, 6))
        counts = np.zeros(len(PAIRS), dtype=int)
        normal_force = np.zeros(len(PAIRS))
        depth = np.zeros(len(PAIRS))
        contact_force = np.zeros(6)
        for i, c in enumerate(self.d.contact):
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, contact_force)
            if contact_force[0] <= 0.:
                continue
            names = ["film" if g < 0 else self.m.geom(g).name for g in c.geom]
            pair = pair_name(names)
            if pair is not None:
                j = PAIRS.index(pair)
                counts[j] += 1
                normal_force[j] += contact_force[0]
                depth[j] = max(depth[j], -c.dist)
            matrix = c.frame.reshape(3, 3).T
            world_force = matrix @ contact_force[:3]
            world_torque = matrix @ contact_force[3:]
            for side, name in enumerate(names):
                if name == "film":
                    continue
                key = "payload" if name.startswith("payload") else name
                if key not in (*FIXTURES, "table", "payload"):
                    continue
                row = (*FIXTURES, "table", "payload").index(key)
                geom = c.geom[side]
                torque = world_torque + np.cross(c.pos - self.d.xpos[self.m.geom_bodyid[geom]], world_force)
                forces[row] += (2 * side - 1) * np.r_[world_force, torque]
        clamped = self.d.qfrc_constraint[self.film_dofs[self.spec.fixed_nodes(self.envelope)]]
        return dict(wrench=forces, pairs=counts, pair_normal=normal_force, pair_depth=depth,
                    clamp_force=clamped.sum(axis=0))

    def get_physics_diagnostics(self, positions):
        angles, _ = dihedrals(positions, self.hinges, gradient=False)
        delta = np.arctan2(np.sin(angles - self.rest_angles), np.cos(angles - self.rest_angles))
        return dict(bending_energy=float(np.sum(.5 * self.hinge_stiffness * delta**2)),
                    solver_potential=float(self.d.energy[0]),
                    warnings=sum(int(w.number) for w in self.d.warning),
                    contact_overflow=int(self.d.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number
                                         + self.d.warning[mujoco.mjtWarning.mjWARN_CNSTRFULL].number))


def yaw_quaternion(yaw):
    return (np.cos(yaw / 2), 0., 0., np.sin(yaw / 2))


def pair_name(names):
    if "film" in names:
        a, b = names
        other = b if a == "film" else a
        tag = ("film_self" if other == "film" else
               "film_payload" if other.startswith("payload") else f"film_{other}")
        return tag if tag in PAIRS else None
    if any(name.startswith("payload") for name in names):
        return "payload_table" if "table" in names else "payload_fixture"
    return None
