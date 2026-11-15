"""Physical contact parcel using the archived, connected two-sheet kraft mesh.

Reference material coordinates determine native flex elasticity. Wrapped points
are only the initial state. Payload, table and tools exchange load through flex
contacts. No payload-film or gripper-film attachments are created.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from .bag_model import _topology, dihedrals
from .wrap import wrap_generic

HERE = Path(__file__).resolve().parent
# The existing wrapping routine uses these legacy geometry utilities.
sys.path.insert(0, str(HERE / "legacy"))
from contents import normalize_contents  # noqa: E402


def vec(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def material_parameters(young_pa, thickness_m, density_kg_m3, poisson=0.3):
    """Correlated SI parameters for an isotropic elastic sheet."""
    values = np.asarray([young_pa, thickness_m, density_kg_m3, poisson], dtype=float)
    if not np.isfinite(values).all() or min(values[:3]) <= 0 or not -1 < poisson < 0.5:
        raise ValueError("material needs positive E, thickness, density and -1 < poisson < 0.5")
    return dict(membrane=young_pa * thickness_m,
                bending=young_pa * thickness_m**3 / (12 * (1 - poisson**2)),
                areal_mass=density_kg_m3 * thickness_m,
                material_thickness=thickness_m, poisson=poisson)


@dataclass(frozen=True)
class Config:
    geometry: str = "geometry/kraft_coarse_35.npz"
    bag_size: tuple = (0.35, 0.25)  # length x, width y [m]
    payload_size: tuple = (0.12, 0.08, 0.03)
    payload_mass: float = 0.3  # kg
    payload_com: tuple = (0.0, 0.0, 0.0)  # rigid mode only, local m
    payload_inertia: tuple | None = None  # rigid mode, kg m^2
    mode: str = "rigid"
    areal_mass: float = 0.12  # kg/m^2, engineering setting
    membrane: float = 1000.0  # E*t [N/m]
    poisson: float = 0.3
    bending: float = 2.0604395604395605e-6  # E*t^3/[12*(1-nu^2)] [N m]
    precreased_ratio: float = 0.01
    bend_damping_ratio: float = 0.1
    material_thickness: float = 0.00015  # constitutive thickness [m]
    rayleigh: float = 1e-4  # s
    radius: float = 0.0005  # numerical contact skin [m]
    friction: float = 0.45
    joint_stiffness: float = 0.08  # N m/rad
    joint_damping: float = 0.003  # N m s/rad
    compression_k1: float = 800.0  # N/m per segment top plate
    compression_k2: float = 20000.0  # N/m^2
    compression_damping: float = 1.0  # N s/m
    timestep: float = 0.000025  # resolves explicit hinge modes of the archived seams
    contact_time: float = 0.001
    iterations: int = 300  # also bounds the flex effective-metric solve
    center: tuple = (0.0, 0.0)
    gravity: tuple = (0.0, 0.0, -9.81)
    self_collision: bool = True
    probe_shape: str = "sphere"

    def __post_init__(self):
        if self.mode not in ("rigid", "bendable", "compressible"):
            raise ValueError("mode must be rigid, bendable or compressible")
        for name, length in (("bag_size", 2), ("payload_size", 3), ("payload_com", 3),
                             ("center", 2), ("gravity", 3)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f"{name} needs {length} finite values")
        positive = ("payload_mass", "areal_mass", "membrane", "bending", "radius", "material_thickness",
                    "timestep", "contact_time")
        nonnegative = ("rayleigh", "friction", "joint_stiffness", "joint_damping",
                       "compression_k1", "compression_k2", "compression_damping", "precreased_ratio", "bend_damping_ratio")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"invalid {name}")
        if min(self.bag_size) <= 0 or min(self.payload_size) <= 0:
            raise ValueError("dimensions must be positive")
        if self.payload_inertia is not None:
            inertia = np.asarray(self.payload_inertia, dtype=float)
            if inertia.shape != (3,) or not np.isfinite(inertia).all() or min(inertia) <= 0:
                raise ValueError("payload_inertia needs three positive principal inertias")
            if 2 * max(inertia) > sum(inertia):
                raise ValueError("principal inertias violate the triangle inequality")
        if self.mode != "rigid" and (self.payload_inertia is not None or any(self.payload_com)):
            raise ValueError("explicit COM and principal inertias apply to rigid mode")
        if self.probe_shape not in ("box", "sphere", "capsule"):
            raise ValueError("unsupported probe shape")
        if not math.isfinite(self.poisson) or not -1 < self.poisson < 0.5:
            raise ValueError("poisson must be between -1 and 0.5")
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")


def load_envelope(cfg):
    """Reuse topology and scale reference material, then wrap an independent state."""
    path = (HERE / cfg.geometry).resolve()
    with np.load(path) as source:
        vertices = source["vertices"].copy()
        faces = source["faces"].copy()
        panels = source["panel_ids"].copy()
    reference = np.column_stack((-vertices[:, 1], vertices[:, 0], vertices[:, 2]))
    reference[:, 0] *= cfg.bag_size[0] / 0.35
    reference[:, 1] *= cfg.bag_size[1] / 0.25
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(f"{path.name} is not closed; seam repair is required")
    # An oriented closed surface traverses every edge once in each direction.
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    if len(np.unique(directed, axis=0)) != len(directed):
        raise ValueError("inconsistent triangle orientation")
    triangles = reference[faces]
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area <= 0):
        raise ValueError("degenerate reference triangles")
    masses = np.zeros(len(reference))
    np.add.at(masses, faces.ravel(), np.repeat(area * cfg.areal_mass / 3, 3))
    item = normalize_contents([dict(shape="box", size_m=list(cfg.payload_size),
                                    mass_kg=cfg.payload_mass)])[0]
    current, preparation = wrap_generic(reference, reference.copy(), faces, panels,
                                         [0], [3], item, cfg.radius)
    if preparation["maximum_edge_strain"] > 0.05:
        raise ValueError(f"initial film strain exceeds 5%: {preparation}")
    shift = np.array([*cfg.center, 0.0])
    reference += shift
    current += shift
    item["position"] += shift
    lengths = np.linalg.norm(reference[unique[:, 1]] - reference[unique[:, 0]], axis=1)
    return dict(reference=reference, current=current, faces=faces, panels=panels,
                edges=unique, masses=masses, payload_origin=item["position"],
                metadata=dict(source=str(path.relative_to(HERE.parent)),
                              source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              vertices=len(reference), triangles=len(faces), edges=len(unique),
                              reference_area_m2=float(area.sum()), film_mass_kg=float(masses.sum()),
                              min_reference_edge_m=float(lengths.min()),
                              max_reference_edge_m=float(lengths.max()), preparation=preparation))


def box_inertia(mass, size):
    size = np.asarray(size)
    return mass * (np.sum(size * size) - size * size) / 12


def payload_xml(cfg, origin):
    count = 1 if cfg.mode == "rigid" else 3
    length, width, height = cfg.payload_size
    segment = np.array([length / count, width, height])
    mass = cfg.payload_mass / count
    output = []
    for index in range(count):
        pos = np.array(origin) + [-length / 2 + segment[0] / 2, 0, 0] if index == 0 else [segment[0], 0, 0]
        output.append(f'<body name="payload_{index}" pos="{vec(pos)}">')
        if index == 0:
            output.append('<freejoint name="payload_free"/>')
        else:
            output.append(f'<joint name="bend_{index}" type="hinge" axis="0 1 0" '
                          f'pos="{-segment[0]/2} 0 0" range="{vec([-math.pi/4, math.pi/4])}" '
                          f'stiffness="{cfg.joint_stiffness}" damping="{cfg.joint_damping}"/>')
        compressible = cfg.mode == "compressible"
        core_mass = mass * (0.9 if compressible else 1.0)
        core_size = segment * [1, 1, 0.3] if compressible else segment
        core_z = -0.35 * height if compressible else 0.0
        inertia = cfg.payload_inertia if cfg.payload_inertia is not None else box_inertia(core_mass, core_size)
        com = cfg.payload_com if count == 1 else (0, 0, core_z)
        output.append(f'<inertial pos="{vec(com)}" mass="{core_mass}" diaginertia="{vec(inertia)}"/>')
        output.append(f'<geom name="payload_geom_{index}" type="box" pos="0 0 {core_z}" '
                      f'size="{vec(core_size/2)}" contype="8" conaffinity="2" rgba=".85 .50 .20 1"/>')
        if compressible:
            plate_size = np.array([segment[0], width, min(0.002, height * 0.1)])
            output.append(f'<body name="plate_{index}" pos="0 0 {(height-plate_size[2])/2}">'
                          f'<joint name="compression_{index}" type="slide" axis="0 0 -1" '
                          f'range="0 {height*0.5-plate_size[2]}"/>'
                          f'<inertial pos="0 0 0" mass="{mass*.1}" '
                          f'diaginertia="{vec(box_inertia(mass*.1,plate_size))}"/>'
                          f'<geom name="plate_geom_{index}" type="box" size="{vec(plate_size/2)}" '
                          'contype="8" conaffinity="2" rgba=".90 .65 .25 1"/></body>')
    output.extend('</body>' for _ in range(count))
    return "\n".join(output)


def compression_force(delta, velocity, cfg):
    delta = np.maximum(delta, 0.0)
    return np.maximum(0.0, cfg.compression_k1 * delta + cfg.compression_k2 * delta**2
                      + cfg.compression_damping * velocity)


class Parcel:
    """CPU model and geometry-aware compiler also used by the YAM adapter."""

    def __init__(self, cfg=Config()):
        if not mujoco.__version__.startswith("4."):
            raise RuntimeError("Physical parcel requires the MuJoCo 4 environment in envs/physical-parcel-requirements.txt")
        self.cfg = cfg
        self.envelope = load_envelope(cfg)
        film = self.envelope
        _, _, self.hinges, weights, adjacent, _, _ = _topology(film["reference"], film["faces"])
        self.rest_angles, gradients = dihedrals(film["reference"], self.hinges)
        creased = film["panels"][adjacent[:, 0]] != film["panels"][adjacent[:, 1]]
        self.hinge_stiffness = cfg.bending * weights * np.where(creased, cfg.precreased_ratio, 1.0)
        inverse_inertia = np.sum(gradients**2 / film["masses"][self.hinges, None], axis=(1, 2))
        self.hinge_damping = 2 * cfg.bend_damping_ratio * np.sqrt(self.hinge_stiffness / inverse_inertia)
        omega_bound = np.sqrt(np.max(self.hinge_stiffness * inverse_inertia))
        if cfg.timestep * omega_bound > 0.2:
            raise ValueError(f"timestep does not resolve explicit film bending; use <= {0.2/omega_bound:.6g} s")
        thickness = cfg.material_thickness
        probe_size = {"sphere": ".01", "box": ".01 .01 .01", "capsule": ".006 .01"}[cfg.probe_shape]
        self.xml = f'''<mujoco model="physical_parcel">
          <compiler angle="radian"/>
          <option timestep="{cfg.timestep}" integrator="discrete" solver="Newton"
            iterations="{cfg.iterations}" ls_iterations="50" tolerance="1e-8" cone="elliptic"
            gravity="{vec(cfg.gravity)}"><flag energy="enable"/></option>
          <default><joint damping="0"/><geom condim="3" friction="{cfg.friction} .005 .0001"
            priority="1" solref="{cfg.contact_time} 1" solimp=".999 .999 .001"/></default>
          <worldbody><light pos="0 0 2"/>
            <geom name="table" type="plane" size="1 1 .05" contype="1" conaffinity="2"/>
            <flexcomp name="bag" type="direct" dim="2" point="{vec(film['reference'])}"
              element="{' '.join(map(str,film['faces'].ravel()))}" mass="{film['masses'].sum()}"
              radius="{cfg.radius}" rgba=".72 .56 .36 .8">
              <elasticity young="{cfg.membrane/thickness}" poisson="{cfg.poisson}" thickness="{thickness}"
                damping="{cfg.rayleigh}" elastic2d="stretch"/>
              <contact contype="2" conaffinity="15" priority="0" condim="3"
                friction="{cfg.friction} .005 .0001" solref="{cfg.contact_time} 1"
                solimp=".999 .999 .001" selfcollide="{'auto' if cfg.self_collision else 'none'}"/>
            </flexcomp>
            {payload_xml(cfg,film['payload_origin'])}
            <body name="contact_probe" mocap="true" pos="0 0 1">
              <geom name="contact_probe" type="{cfg.probe_shape}" size="{probe_size}"
                contype="4" conaffinity="2" rgba=".2 .6 .8 1"/>
            </body>
          </worldbody><contact/><equality/>
        </mujoco>'''
        self.m = self.compile_model(self.xml)
        self.d = mujoco.MjData(self.m)
        self.initialize(self.m, self.d)
        self.film_dofs = self.node_dofs(self.m)
        self.payload_bodies = np.array([b for b in range(self.m.nbody)
                                        if self.m.body(b).name.startswith(("payload_", "plate_"))])
        self.compression_joints = np.array([j for j in range(self.m.njnt)
                                            if self.m.joint(j).name.startswith("compression_")], dtype=int)
        self.compression_names = [self.m.joint(j).name for j in self.compression_joints]
        self.metadata = dict(config=asdict(cfg), geometry=film["metadata"],
                             nq=self.m.nq, nv=self.m.nv, neq=self.m.neq,
                             bending="reference dihedral energy, existing bag_model gradient",
                             compression_axis="segment-local -z")

    def compile_model(self, xml):
        model = mujoco.MjModel.from_xml_string(xml)
        for index, mass in enumerate(self.envelope["masses"]):
            body = model.body(f"bag_{index}").id
            model.body_mass[body] = mass
            # Flex point bodies carry translation only; rotational inertia is unused.
        mujoco.mj_setConst(model, mujoco.MjData(model))
        return model

    def node_dofs(self, model):
        return np.array([model.body_dofadr[model.body(f"bag_{i}").id] + np.arange(3)
                         for i in range(len(self.envelope["reference"]))])

    def initialize(self, model, data):
        for index, delta in enumerate(self.envelope["current"] - self.envelope["reference"]):
            body = model.body(f"bag_{index}").id
            joint = model.body_jntadr[body]
            addresses = model.jnt_qposadr[joint:joint + 3]
            data.qpos[addresses] = delta
        mujoco.mj_forward(model, data)

    def apply_internal(self, model=None, data=None, node_dofs=None):
        model = self.m if model is None else model
        data = self.d if data is None else data
        node_dofs = self.film_dofs if node_dofs is None else node_dofs
        points = data.flexvert_xpos[:len(node_dofs)]
        angles, gradients = dihedrals(points, self.hinges)
        delta = np.arctan2(np.sin(angles - self.rest_angles), np.cos(angles - self.rest_angles))
        rates = np.sum(gradients * data.qvel[node_dofs][self.hinges], axis=(1, 2))
        moment = self.hinge_stiffness * delta + self.hinge_damping * rates
        forces = np.zeros_like(points)
        np.add.at(forces, self.hinges.ravel(), (-moment[:, None, None] * gradients).reshape(-1, 3))
        data.qfrc_passive[node_dofs] += forces
        for name in self.compression_names:
            joint = model.joint(name).id
            q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            data.qfrc_passive[v] -= compression_force(data.qpos[q], data.qvel[v], self.cfg)

    def step(self):
        mujoco.mj_step1(self.m, self.d)
        self.apply_internal()
        mujoco.mj_step2(self.m, self.d)

    def bending_energy(self):
        angles, _ = dihedrals(self.d.flexvert_xpos, self.hinges, gradient=False)
        delta = np.arctan2(np.sin(angles - self.rest_angles), np.cos(angles - self.rest_angles))
        return float(np.sum(0.5 * self.hinge_stiffness * delta**2))

    def forward(self, model=None, data=None, node_dofs=None):
        """Refresh reactions with the same custom passive forces as step()."""
        model = self.m if model is None else model
        data = self.d if data is None else data
        mujoco.mj_step1(model, data)
        self.apply_internal(model, data, node_dofs)
        mujoco.mj_fwdActuation(model, data)
        mujoco.mj_fwdAcceleration(model, data)
        mujoco.mj_fwdConstraint(model, data)

    def com(self):
        return np.average(self.d.xipos[self.payload_bodies], axis=0,
                          weights=self.m.body_mass[self.payload_bodies])

    def contacts(self):
        """Signed world forces on payload, table and tool from native contacts."""
        result = dict(table_support_N=0.0, probe_force_N=np.zeros(3),
                      payload_contact_N=np.zeros(3), payload_table_contacts=0,
                      film_payload_contacts=0, film_probe_contacts=0,
                      film_self_contacts=0, penetration_m=0.0, ncon=self.d.ncon)
        force = np.zeros(6)
        payload = set(self.payload_bodies)
        for index, contact in enumerate(self.d.contact):
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.m, self.d, index, force)
            world = contact.frame.reshape(3, 3).T @ force[:3]
            result["penetration_m"] = max(result["penetration_m"], -float(contact.dist))
            names = [self.m.geom(g).name if g >= 0 else "film" for g in contact.geom]
            has_film = any(f >= 0 for f in contact.flex)
            for side, geom in enumerate(contact.geom):
                if geom < 0:
                    continue
                reaction = world if side == 1 else -world
                if names[side] == "table":
                    result["table_support_N"] -= reaction[2]
                if names[side] == "contact_probe":
                    result["probe_force_N"] += reaction
                    result["film_probe_contacts"] += int(has_film)
                if self.m.geom_bodyid[geom] in payload:
                    result["payload_contact_N"] += reaction
                    result["film_payload_contacts"] += int(has_film)
                    result["payload_table_contacts"] += int("table" in names)
            result["film_self_contacts"] += int(all(f >= 0 for f in contact.flex))
        return result

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "model.xml").write_text(self.xml)
        (output / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        np.savez_compressed(output / "geometry.npz", **{k: self.envelope[k] for k in
                            ("reference", "current", "faces", "panels", "masses", "edges")})
