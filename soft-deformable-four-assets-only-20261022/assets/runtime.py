"""Free-node paper shells for native MuJoCo 3.14; SI units throughout.

The reference mesh, lumped masses and initial placement belong to each asset.
Native flex edge equalities provide an authored low-stretch approximation (not
a fitted tensile modulus). All nodes retain three world-aligned slide DOFs.
There are no pins, pose constraints, pressure or rigid internal supports.

Bending uses a discrete dihedral energy with geometric edge weights, a permanent
elastoplastic rotation, and a transient Maxwell branch in parallel with the
equilibrium branch (a short-time standard-linear-solid approximation). Neither
the plastic yield angle nor the relaxation time is a measured paper constant.
The analytical gradient and return-map structure were adapted from the existing
workspace/asset/plastic-bag-filled-asset/{shell,rheology,model}.py; that source
package is left intact. Angles are continuously unwrapped at accepted steps.

MuJoCo may evaluate passive forces repeatedly; callbacks never mutate material
history. Only a successfully completed mj_step advances plastic/viscous state.
Opening the generated XML alone omits this Python constitutive law.

Required geometry.npz: vertices (N,3), faces (F,3), uv (N,2), masses_kg (N,).
Optional initial_vertices (N,3) separates an authored placement from reference.
Optional panel_ids (F,) select opposing collision groups and precreased hinges;
they do not introduce rigid panels. They are required when self_contact is true.
Self contact acts only between the configured front/back panel groups. Same-side
folds and other side/gusset panels are outside that approximation. Contact-only
patches share the original vertex bodies and add no mass or mechanical support.
Optional hinge_stiffness_scale (H,) follows lexicographically sorted interior
edges, as returned in runtime.hinges[:, :2]. Default is one on every edge.
Optional hinge_yield_scale (H,) similarly scales the grid-dependent yield angle.

load_paper(..., initial_vertices=Nx3, contents=(...)) can prepare a filled scene
without modifying its unloaded reference mesh. Contents are independent rigid
mesh bodies with free joints and their own mass; native mesh collision uses a
convex hull, so concavities in the visible mesh are not collision cavities.
Each content dictionary supplies name, mesh_path, position_m, mass_kg and rgba;
quaternion_wxyz, scale and texture_path are optional. Mesh and texture paths
may be absolute or relative to the paper asset directory. Initial coordinates
are preparation, not evidence of a simulated insertion or a calibrated fit.

material.json accepts the DEFAULTS keys below and arbitrary provenance metadata.
The physical D is mapped to hinge K = D * 3*l^2/(A_left+A_right). This is an
isotropic discrete-shell approximation; mesh convergence and real calibration
remain separate from a finite rollout. External xfrc_applied/qfrc_applied loads
are left intact, including the real2sim viewer's force-based dragging.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import weakref
import warnings
import xml.etree.ElementTree as ET

import numpy as np


DEFAULTS = {
    "bending_rigidity_nm": 0.00135,
    "plastic_yield_angle_deg": 25.0,
    "plastic_hardening_ratio": 0.08,
    "plasticity": True,
    "precreased_stiffness_ratio": 1.0,
    "precreased_panel_ids": [],
    "precreased_panel_boundaries": True,
    "memory_ratio": 0.6,
    "recovery_tau_s": 2.0,
    "bending_damping_nms": 3e-7,
    "timestep_s": 0.00005,
    "frame_dt_s": 0.01,
    "collision_radius_m": 0.00035,
    "collision_patch_faces": 32,
    "contact_timeconst_s": 0.0002,
    "friction": 0.45,
    "velocity_damping_per_s": 3.0,
    "self_contact": True,
    "self_contact_front_panel_ids": [0],
    "self_contact_back_panel_ids": [1, 2],
    "gravity": True,
    "edge_solref_s": 0.001,
    "edge_solimp": [0.99, 0.999, 0.0001],
    "iterations": 200,
    "solver": "CG",
    "tolerance": 1e-8,
}

_RUNTIMES = {}
_PREVIOUS_PASSIVE = None


def _numbers(values):
    return " ".join(f"{float(value):.14g}" for value in np.asarray(values).ravel())


def _wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def _parameters(source):
    params = {**DEFAULTS, **{key: source[key] for key in DEFAULTS if key in source}}
    bool_keys = {"plasticity", "self_contact", "gravity", "precreased_panel_boundaries"}
    panel_list_keys = {"precreased_panel_ids", "self_contact_front_panel_ids", "self_contact_back_panel_ids"}
    if params['solver'] not in {'CG','Newton'}:
        raise ValueError('solver must be CG or Newton')
    for key in bool_keys:
        if not isinstance(params[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in DEFAULTS.keys() - bool_keys - {"edge_solimp", "solver"} - panel_list_keys:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if value < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("bending_rigidity_nm", "plastic_yield_angle_deg", "recovery_tau_s", "timestep_s",
                "frame_dt_s", "collision_radius_m", "contact_timeconst_s", "edge_solref_s", "iterations", "tolerance"):
        if params[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= params["plastic_hardening_ratio"] < 1:
        raise ValueError("plastic_hardening_ratio must be in [0,1)")
    if not 0 < params["precreased_stiffness_ratio"] <= 1:
        raise ValueError("precreased_stiffness_ratio must be in (0,1]")
    for key in panel_list_keys:
        if not isinstance(params[key], list) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in params[key]):
            raise ValueError(f"{key} must be a list of integer panel identifiers")
    if set(params["self_contact_front_panel_ids"]) & set(params["self_contact_back_panel_ids"]):
        raise ValueError("Self-contact front and back panel identifier groups must not overlap")
    if int(params["iterations"]) != params["iterations"]:
        raise ValueError("iterations must be integer")
    if int(params["collision_patch_faces"]) != params["collision_patch_faces"]:
        raise ValueError("collision_patch_faces must be a nonnegative integer (zero uses one patch per panel)")
    ratio = params["frame_dt_s"] / params["timestep_s"]
    if abs(ratio - round(ratio)) > 1e-7 or ratio < 1:
        raise ValueError("frame_dt_s must be an integer multiple of timestep_s")
    impedance = np.asarray(params["edge_solimp"], dtype=float)
    if (impedance.shape != (3,) or not np.isfinite(impedance).all()
            or not 0 < impedance[0] <= impedance[1] < 1 or impedance[2] <= 0):
        raise ValueError("edge_solimp must be [0 < dmin <= dmax < 1, positive width]")
    return params


def _contents(source, asset_dir):
    """Normalize local rigid-mesh content descriptions without moving sources."""
    result = []
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise ValueError(f"contents[{index}] must be a dictionary")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"contents[{index}].name must be a nonempty string")
        normalized = {"name": name, "xml_name": f"content_{index}_" + re.sub(r"[^A-Za-z0-9_]+", "_", name)}
        for key in ("mesh_path", "texture_path"):
            value = item.get(key)
            if value is None and key == "texture_path":
                normalized[key] = None
                continue
            if not isinstance(value, (str, Path)):
                raise ValueError(f"contents[{index}].{key} must identify a local file")
            path = Path(value).expanduser()
            path = (asset_dir / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            normalized[key] = str(path)
        for key, size, default in (("position_m", 3, None), ("quaternion_wxyz", 4, [1, 0, 0, 0]),
                                   ("scale", 3, [1, 1, 1]), ("rgba", 4, None)):
            value = np.asarray(item.get(key, default), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"contents[{index}].{key} must have {size} finite values")
            normalized[key] = value
        if np.any(normalized["scale"] <= 0):
            raise ValueError(f"contents[{index}].scale must be positive")
        if np.any((normalized["rgba"] < 0) | (normalized["rgba"] > 1)):
            raise ValueError(f"contents[{index}].rgba must be in [0,1]")
        norm = np.linalg.norm(normalized["quaternion_wxyz"])
        if norm < 1e-12:
            raise ValueError(f"contents[{index}].quaternion_wxyz must be nonzero")
        normalized["quaternion_wxyz"] = normalized["quaternion_wxyz"] / norm
        mass = item.get("mass_kg")
        if isinstance(mass, bool) or not isinstance(mass, (float, int)) or not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"contents[{index}].mass_kg must be positive and finite")
        normalized["mass_kg"] = float(mass)
        result.append(normalized)
    return result


def dihedrals(vertices, hinges, *, gradient=True):
    """Signed angle and analytical derivative on [edge_a, edge_b, opp_c, opp_d]."""
    if not len(hinges):
        return np.zeros(0), np.zeros((0, 4, 3)) if gradient else None
    a, b, c, d = (vertices[hinges[:, index]] for index in range(4))
    edge = b - a
    length2 = np.einsum("ij,ij->i", edge, edge)[:, None]
    length = np.sqrt(np.maximum(length2, 1e-24))
    normal1 = np.cross(edge, c - a)
    normal2 = np.cross(d - a, edge)
    norm1 = np.einsum("ij,ij->i", normal1, normal1)[:, None]
    norm2 = np.einsum("ij,ij->i", normal2, normal2)[:, None]
    valid = (length[:, 0] > 1e-10) & (norm1[:, 0] > 1e-20) & (norm2[:, 0] > 1e-20)
    norm1 = np.maximum(norm1, 1e-24)
    norm2 = np.maximum(norm2, 1e-24)
    length2 = np.maximum(length2, 1e-24)
    unit1, unit2 = normal1 / np.sqrt(norm1), normal2 / np.sqrt(norm2)
    angle = np.arctan2(np.einsum("ij,ij->i", edge / length, np.cross(unit1, unit2)),
                       np.einsum("ij,ij->i", unit1, unit2))
    if not gradient:
        return angle, None
    gc, gd = -normal1 * length / norm1, -normal2 * length / norm2
    ga = (gc * (-np.sum((b - c) * edge, axis=1)[:, None] / length2)
          + gd * (-np.sum((b - d) * edge, axis=1)[:, None] / length2))
    gb = (gc * (-np.sum((c - a) * edge, axis=1)[:, None] / length2)
          + gd * (-np.sum((d - a) * edge, axis=1)[:, None] / length2))
    result = np.stack((ga, gb, gc, gd), axis=1)
    result[~valid] = 0
    return angle, result


def _topology(vertices, faces):
    adjacent = {}
    areas = 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                        vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    if np.any(areas <= 1e-12):
        raise ValueError("Reference mesh has degenerate or excessively small triangles")
    if len({tuple(sorted(face)) for face in faces}) != len(faces):
        raise ValueError("Reference mesh has duplicate triangles")
    for face_id, face in enumerate(faces):
        for index in range(3):
            a, b = sorted((int(face[index]), int(face[(index + 1) % 3])))
            adjacent.setdefault((a, b), []).append((int(face[(index + 2) % 3]), face_id))
    if any(len(value) > 2 for value in adjacent.values()):
        raise ValueError("Reference mesh has non-manifold edges")
    ordered = sorted(adjacent)
    edges = np.asarray(ordered, dtype=np.int32)
    interior = [edge for edge in ordered if len(adjacent[edge]) == 2]
    hinges = np.asarray([[*edge, adjacent[edge][0][0], adjacent[edge][1][0]]
                         for edge in interior], dtype=np.int32).reshape((-1, 4))
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    weight = np.asarray([3 * np.sum((vertices[edge[1]] - vertices[edge[0]]) ** 2)
                         / sum(areas[face] for _, face in adjacent[edge]) for edge in interior])
    report = {"vertices": len(vertices), "faces": len(faces), "edges": len(edges),
              "hinges": len(hinges), "boundary_edges": sum(len(v) == 1 for v in adjacent.values()),
              "area_m2": float(areas.sum()), "nonmanifold_edges": 0,
              "world_fixed_nodes": 0, "rigid_internal_supports": 0}
    face_ids = np.asarray([[face for _, face in adjacent[edge]] for edge in interior], dtype=np.int32).reshape((-1, 2))
    return edges, lengths, hinges, weight, face_ids, report


def _passive_dispatch(model, data):
    if _PREVIOUS_PASSIVE is not None:
        _PREVIOUS_PASSIVE(model, data)
    reference = _RUNTIMES.get(model._address)
    runtime = reference() if reference is not None else None
    if runtime is not None and not runtime.closed:
        runtime._passive(model, data)


class PaperRuntime:
    """Factory contract: model, data, step(), close(); history is instance-owned."""

    def __init__(self, asset_dir, *, support=True, initial_vertices=None, contents=()):
        import mujoco

        if not mujoco.__version__.startswith("3.14."):
            raise RuntimeError("Paper shell runtime targets MuJoCo >=3.14,<3.15")
        self.asset_dir = Path(asset_dir).resolve()
        self.material_source = json.loads((self.asset_dir / "material.json").read_text())
        self.params = _parameters(self.material_source)
        self.contents = _contents(contents, self.asset_dir)
        self.support = bool(support)
        self.closed = False
        with np.load(self.asset_dir / "geometry.npz", allow_pickle=False) as geometry:
            self.rest = np.array(geometry["vertices"], dtype=float)
            self.faces = np.array(geometry["faces"], dtype=np.int32)
            self.uv = np.array(geometry["uv"], dtype=float)
            self.masses_kg = np.array(geometry["masses_kg"], dtype=float)
            self.panel_ids = np.array(geometry["panel_ids"], dtype=np.int32) if "panel_ids" in geometry else None
            initial = np.array(geometry["initial_vertices"], dtype=float) if "initial_vertices" in geometry else None
            scale = np.array(geometry["hinge_stiffness_scale"], dtype=float) if "hinge_stiffness_scale" in geometry else None
            yield_scale = np.array(geometry["hinge_yield_scale"], dtype=float) if "hinge_yield_scale" in geometry else None
        if initial_vertices is not None:
            initial = np.array(initial_vertices, dtype=float, copy=True)
        count = len(self.rest)
        if (self.rest.shape != (count, 3) or count < 3 or not np.isfinite(self.rest).all()
                or self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces)
                or np.any(self.faces < 0) or np.any(self.faces >= count)
                or self.uv.shape != (count, 2) or not np.isfinite(self.uv).all()
                or self.masses_kg.shape != (count,) or not np.isfinite(self.masses_kg).all()
                or np.any(self.masses_kg <= 0)):
            raise ValueError("Invalid geometry vertices/faces/uv/lumped masses")
        if self.panel_ids is not None and self.panel_ids.shape != (len(self.faces),):
            raise ValueError("panel_ids must have one entry per face")
        if self.params["self_contact"] and self.panel_ids is None:
            raise ValueError("Self contact requires panel_ids and configured opposing panel groups")
        self.edges, self.rest_lengths, self.hinges, weights, hinge_faces, self.topology = _topology(self.rest, self.faces)
        self.stiffness = self.params["bending_rigidity_nm"] * weights
        self.hinge_damping = np.full(len(self.hinges), self.params["bending_damping_nms"])
        self.precreased = np.zeros(len(self.hinges), dtype=bool)
        if self.panel_ids is not None:
            panel_pairs = self.panel_ids[hinge_faces]
            self.precreased |= np.any(np.isin(panel_pairs, self.params["precreased_panel_ids"]), axis=1)
            if self.params["precreased_panel_boundaries"]:
                self.precreased |= panel_pairs[:, 0] != panel_pairs[:, 1]
        crease_scale = np.where(self.precreased, self.params["precreased_stiffness_ratio"], 1.0)
        self.stiffness *= crease_scale
        self.hinge_damping *= crease_scale
        self.topology["precreased_hinges"] = int(np.count_nonzero(self.precreased))
        self.yield_angles = np.full(len(self.hinges), np.deg2rad(self.params["plastic_yield_angle_deg"]))
        for name, values, target in (("hinge_stiffness_scale", scale, self.stiffness),
                                     ("hinge_yield_scale", yield_scale, self.yield_angles)):
            if values is not None:
                if values.shape != target.shape or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"{name} must have one positive finite value per interior edge")
                target *= values
        self.rest_angles, _ = dihedrals(self.rest, self.hinges, gradient=False)
        self.last_angles = self.rest_angles.copy()
        self.total_angles = np.zeros(len(self.hinges))
        self.plastic = np.zeros(len(self.hinges))
        self.accumulated_plastic = np.zeros(len(self.hinges))
        self.memory = np.zeros(len(self.hinges))
        self.plastic_dissipated_j = 0.0
        self.memory_dissipated_j = 0.0
        self.callback_calls = 0
        self.accepted_steps = 0
        self.xml = self._xml()
        self.compiler_warnings = []
        previous_warning = mujoco.get_mju_user_warning()

        def compile_warning(message):
            expected = (message.startswith("flex 'paper_contact_")
                        and "is not rigid and has no equality constraints or passive forces" in message)
            self.compiler_warnings.append({"message": message, "expected_contact_patch_notice": expected})
            if not expected:
                if previous_warning is not None:
                    previous_warning(message)
                else:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)

        # MuJoCo cannot know that a collision-only flex shares all of its bodies
        # with the mechanically constrained main flex. Retain this exact known
        # compiler notice as evidence, without printing one copy per patch.
        try:
            mujoco.set_mju_user_warning(compile_warning)
            self.model = mujoco.MjModel.from_xml_string(self.xml)
        finally:
            mujoco.set_mju_user_warning(previous_warning)
        self.data = mujoco.MjData(self.model)
        self.flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "paper")
        address = int(self.model.flex_vertadr[self.flex_id])
        self.vert_slice = slice(address, address + count)
        self.body_ids = self.model.flex_vertbodyid[self.vert_slice].copy()
        self.content_body_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, item["xml_name"])
                                            for item in self.contents], dtype=np.int32)
        self.content_geom_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, item["xml_name"] + "_geom")
                                            for item in self.contents], dtype=np.int32)
        joint_ids = self.model.body_jntadr[self.body_ids, None] + np.arange(3)
        if (not np.all(self.model.body_jntnum[self.body_ids] == 3)
                or not np.all(self.model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_SLIDE)
                or not np.allclose(self.model.jnt_axis[joint_ids], np.broadcast_to(np.eye(3), (count, 3, 3)))
                or not np.all(self.model.body_parentid[self.body_ids] == 0)):
            raise RuntimeError("Expected independent world-aligned three-translation vertex bodies")
        self.dofs = self.model.jnt_dofadr[joint_ids]
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.model.dof_damping[self.dofs] = self.masses_kg[:, None] * self.params["velocity_damping_per_s"]
        if initial is None:
            initial = self.rest.copy()
            if self.support:
                initial[:, 2] += max(0.0, self.params["collision_radius_m"] + 0.00015 - initial[:, 2].min())
        if initial.shape != self.rest.shape or not np.isfinite(initial).all():
            raise ValueError("initial_vertices must match reference vertex shape and be finite")
        self.initial = initial.copy()
        self.data.qpos[self.qpos_ids] = self.initial - self.rest
        # The initial placement does not redefine the paper's unloaded reference.
        # Its initial deformation is included on the first force evaluation.
        global _PREVIOUS_PASSIVE
        if not _RUNTIMES:
            current = mujoco.get_mjcb_passive()
            _PREVIOUS_PASSIVE = current if current is not _passive_dispatch else None
            mujoco.set_mjcb_passive(_passive_dispatch)
        elif mujoco.get_mjcb_passive() is not _passive_dispatch:
            raise RuntimeError("Another owner replaced the process passive callback; close runtimes before loading")
        _RUNTIMES[self.model._address] = weakref.ref(self)
        mujoco.mj_forward(self.model, self.data)
        self.initial_warning_counts = self.data.warning.number.copy()

    @property
    def positions(self):
        return self.rest + self.data.qpos[self.qpos_ids]

    def _xml(self):
        p = self.params
        root = ET.Element("mujoco", model="paper_shell")
        ET.SubElement(root, "compiler", angle="radian")
        opt = ET.SubElement(root, "option", timestep=str(p["timestep_s"]), integrator="discrete", solver=p['solver'],
                            jacobian="sparse", iterations=str(int(p["iterations"])), tolerance=str(p["tolerance"]),
                            gravity="0 0 -9.81" if p["gravity"] else "0 0 0")
        ET.SubElement(opt, "flag", autoreset="disable")
        ET.SubElement(root, "size", memory="128M")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="1200", offheight="900")
        ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".45 .45 .45", specular=".1 .1 .1")
        ET.SubElement(visual, "rgba", haze=".92 .94 .96 1")
        asset = ET.SubElement(root, "asset")
        texture = self.asset_dir / "texture.png"
        if texture.is_file():
            ET.SubElement(asset, "texture", name="paper_texture", type="2d", file=str(texture))
            ET.SubElement(asset, "material", name="paper_material", texture="paper_texture", rgba=".85 .85 .85 1", emission=".18", specular=".05", shininess=".1")
        else:
            ET.SubElement(asset, "material", name="paper_material", rgba=".68 .46 .23 1", specular=".05", shininess=".1")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", name="key", pos="-.3 -.4 1.2", dir=".2 .3 -1", diffuse=".65 .65 .65", castshadow="false")
        ET.SubElement(world, "light", name="fill", pos=".4 .4 .8", dir="-.3 -.3 -1", diffuse=".25 .25 .25", castshadow="false")
        ET.SubElement(world, "camera", name="overview", pos="0 -.57 .65", xyaxes="1 0 0 0 .75 .66", fovy="42")
        ET.SubElement(world, "camera", name="top", pos="0 0 .9", xyaxes="1 0 0 0 1 0", fovy="42")
        if self.support:
            ET.SubElement(world, "geom", name="table", type="plane", size=".65 .65 .03", rgba=".22 .27 .32 1",
                          contype="2", conaffinity="1",
                          friction=_numbers([p["friction"], .005, .0001]), solref=".002 1", solimp=".99 .999 .0001")
        body_names = []
        for index, (point, mass) in enumerate(zip(self.rest, self.masses_kg)):
            name = f"paper_v{index}"
            body_names.append(name)
            body = ET.SubElement(world, "body", name=name, pos=_numbers(point))
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass), diaginertia="1e-10 1e-10 1e-10")
            for label, axis in zip("xyz", np.eye(3)):
                ET.SubElement(body, "joint", name=f"{name}_{label}", type="slide", axis=_numbers(axis), limited="false")
        for item in self.contents:
            name = item["xml_name"]
            ET.SubElement(asset, "mesh", name=name + "_mesh", file=item["mesh_path"], scale=_numbers(item["scale"]), inertia="convex")
            material_attributes = {"name": name + "_material", "rgba": _numbers(item["rgba"]), "specular": ".1", "shininess": ".1"}
            if item["texture_path"] is not None:
                ET.SubElement(asset, "texture", name=name + "_texture", type="2d", file=item["texture_path"])
                material_attributes["texture"] = name + "_texture"
            ET.SubElement(asset, "material", **material_attributes)
            body = ET.SubElement(world, "body", name=name, pos=_numbers(item["position_m"]), quat=_numbers(item["quaternion_wxyz"]))
            ET.SubElement(body, "freejoint", name=name + "_free")
            ET.SubElement(body, "geom", name=name + "_geom", type="mesh", mesh=name + "_mesh", material=name + "_material",
                          mass=str(item["mass_kg"]), contype="2", conaffinity="3", condim="3",
                          friction=_numbers([p["friction"], .005, .0001]), solref=".002 1", solimp=".99 .999 .0001")
        deformable = ET.SubElement(root, "deformable")
        flex = ET.SubElement(deformable, "flex", name="paper", dim="2", body=" ".join(body_names),
                             element=" ".join(str(int(value)) for value in self.faces.ravel()),
                             texcoord=_numbers(self.uv), radius=str(p["collision_radius_m"]),
                             material="paper_material", flatskin="true")
        # The complete flex supplies rendering and the only edge equality.
        # Its single self-contact pair can exhaust MuJoCo's contact budget when
        # the broad opposing sheets meet. Instead, split collision-only patches
        # within each authored panel and permit only front/back cross contacts.
        # The selected masks avoid same-side patch seam contacts and duplicate
        # self contact, while every panel retains rigid surface/contents contact.
        ET.SubElement(flex, "contact", selfcollide="none", contype="0", conaffinity="0", internal="false")
        patch_size = int(p["collision_patch_faces"]) or len(self.faces)
        panel_ids = self.panel_ids if self.panel_ids is not None else np.zeros(len(self.faces), dtype=int)
        front = set(p["self_contact_front_panel_ids"])
        back = set(p["self_contact_back_panel_ids"])
        patch_id = 0
        for panel in np.unique(panel_ids):
            panel_faces = self.faces[panel_ids == panel]
            for start in range(0, len(panel_faces), patch_size):
                triangles = panel_faces[start:start + patch_size]
                used, remapped = np.unique(triangles, return_inverse=True)
                patch = ET.SubElement(deformable, "flex", name=f"paper_contact_{patch_id}", dim="2",
                                      body=" ".join(body_names[index] for index in used),
                                      element=" ".join(str(int(value)) for value in remapped.ravel()),
                                      radius=str(p["collision_radius_m"]), group="5", rgba="0 0 0 0")
                contype, conaffinity = 1, 2
                if p["self_contact"] and panel in front:
                    contype |= 8
                    conaffinity |= 16
                elif p["self_contact"] and panel in back:
                    contype |= 16
                    conaffinity |= 8
                ET.SubElement(patch, "contact", selfcollide="none", internal="false",
                              contype=str(contype), conaffinity=str(conaffinity), condim="3",
                              friction=_numbers([p["friction"], .005, .0001]),
                              solref=_numbers([p["contact_timeconst_s"], 1]), solimp=".99 .999 .0001", margin="0")
                patch_id += 1
        equality = ET.SubElement(root, "equality")
        ET.SubElement(equality, "flex", name="paper_low_stretch", flex="paper",
                      solref=_numbers([p["edge_solref_s"], 1]), solimp=_numbers(p["edge_solimp"]))
        return ET.tostring(root, encoding="unicode")

    def _trial_plastic(self, delta):
        if not self.params["plasticity"]:
            return self.plastic, self.accumulated_plastic, np.zeros_like(delta)
        hardening = self.params["plastic_hardening_ratio"] / (1 - self.params["plastic_hardening_ratio"])
        trial = delta - self.plastic
        increment = np.maximum(np.abs(trial) - self.yield_angles - hardening * self.accumulated_plastic, 0) / (1 + hardening)
        return self.plastic + increment * np.sign(trial), self.accumulated_plastic + increment, increment

    def _passive(self, model, data):
        self.callback_calls += 1
        q = data.flexvert_xpos[self.vert_slice]
        angle, gradient = dihedrals(q, self.hinges)
        delta = self.total_angles + _wrap(angle - self.last_angles)
        plastic, _, _ = self._trial_plastic(delta)
        moment = self.stiffness * ((delta - plastic) + self.params["memory_ratio"] * (delta - self.memory))
        rate = np.einsum("hij,hij->h", gradient, data.qvel[self.dofs][self.hinges])
        moment += self.hinge_damping * rate
        force = np.zeros_like(q)
        contribution = -moment[:, None, None] * gradient
        for index in range(4):
            np.add.at(force, self.hinges[:, index], contribution[:, index])
        data.qfrc_passive[self.dofs] += force

    def _advance_history(self):
        angle, _ = dihedrals(self.positions, self.hinges, gradient=False)
        self.total_angles += _wrap(angle - self.last_angles)
        self.last_angles = angle
        self.plastic, self.accumulated_plastic, increment = self._trial_plastic(self.total_angles)
        self.plastic_dissipated_j += float(np.sum(self.stiffness * self.yield_angles * increment))
        old = self.total_angles - self.memory
        fraction = -np.expm1(-self.params["timestep_s"] / self.params["recovery_tau_s"])
        self.memory += fraction * old
        self.memory_dissipated_j += float(np.sum(.5 * self.stiffness * self.params["memory_ratio"]
                                               * (old ** 2 - (self.total_angles - self.memory) ** 2)))
        self.accepted_steps += 1

    def step_substeps(self, count=1):
        import mujoco

        if self.closed:
            raise RuntimeError("Paper runtime is closed")
        if isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError("count must be a nonnegative integer")
        for _ in range(int(count)):
            previous_time = float(self.data.time)
            warnings = self.data.warning.number.copy()
            mujoco.mj_step(self.model, self.data)
            if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                    or not np.isfinite(self.data.qacc).all()
                    or abs(self.data.time - previous_time - self.params["timestep_s"]) > 1e-9
                    or np.any(self.data.warning.number > warnings)):
                raise RuntimeError("MuJoCo paper step failed or issued a warning; material history was not advanced")
            self._advance_history()

    def step(self):
        self.step_substeps(round(self.params["frame_dt_s"] / self.params["timestep_s"]))

    def state_report(self):
        positions = self.positions
        lengths = np.linalg.norm(positions[self.edges[:, 1]] - positions[self.edges[:, 0]], axis=1)
        return {"time_s": float(self.data.time), "accepted_steps": self.accepted_steps,
                "callback_calls": self.callback_calls, "topology": self.topology,
                "mass_kg": float(self.masses_kg.sum()), "finite": bool(np.isfinite(positions).all()),
                "max_abs_edge_strain": float(np.max(np.abs(lengths / self.rest_lengths - 1))),
                "min_vertex_z_m": float(positions[:, 2].min()), "max_vertex_z_m": float(positions[:, 2].max()),
                "max_speed_m_s": float(np.max(np.linalg.norm(self.data.qvel[self.dofs], axis=1))),
                "max_permanent_crease_deg": float(np.rad2deg(np.max(np.abs(self.plastic), initial=0))),
                "plastic_dissipated_j": self.plastic_dissipated_j,
                "memory_dissipated_j": self.memory_dissipated_j,
                "warnings": self.data.warning.number.tolist(), "contacts": int(self.data.ncon),
                "expected_contact_patch_compiler_notices": sum(item["expected_contact_patch_notice"] for item in self.compiler_warnings),
                "unexpected_compiler_warnings": [item["message"] for item in self.compiler_warnings if not item["expected_contact_patch_notice"]],
                "content_count": len(self.contents),
                "contents": [{"name": item["name"], "mesh_path": item["mesh_path"], "mass_kg": item["mass_kg"],
                              "body_id": int(body_id), "geom_id": int(geom_id),
                              "position_m": self.data.xpos[body_id].tolist(),
                              "quaternion_wxyz": self.data.xquat[body_id].tolist(),
                              "collision_model": "native_mesh_convex_hull"}
                             for item, body_id, geom_id in zip(self.contents, self.content_body_ids, self.content_geom_ids)],
                "physical_calibration": False}

    def close(self):
        import mujoco

        if self.closed:
            return
        self.closed = True
        _RUNTIMES.pop(self.model._address, None)
        if not _RUNTIMES and mujoco.get_mjcb_passive() is _passive_dispatch:
            mujoco.set_mjcb_passive(_PREVIOUS_PASSIVE)


def load_paper(asset_dir, *, support=True, initial_vertices=None, contents=()):
    return PaperRuntime(asset_dir, support=support, initial_vertices=initial_vertices, contents=contents)
