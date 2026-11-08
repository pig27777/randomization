"""One physical model for every bag; materials differ only by profile parameters.

Each bag is a triangulated midsurface whose nodes carry three world-aligned slide
DOFs in native MuJoCo 3.14. Every profile selects values for the same five parts:

  mass        lumped node masses, optionally rescaled to a total or areal density
  membrane    inextensible edges | tension-only edges | elastic StVK membrane
  bending     discrete dihedral energy, D weighted by 3*l^2/(A_left+A_right), with
              region/anisotropy scales, optional bilinear knee, permanent plastic
              rotation, short-time Maxwell memory and crease softening history
  thickness   collision radius plus an optional through-thickness cushion
  contact     friction, self-contact panel groups and solver settings

Any part can be switched off by its parameters, so the PE film, the paper bags and
the bubble mailer are the same equations with different numbers.

The MuJoCo cushion uses soft contact. Its force scales with the effective mass at
the contact, so per patch the direct solref stiffness is divided by the patch areal
density. That removes the dependence on areal density, but not on the mesh or the
colliding geom: a box pressing a sheet came out 10x (10 mm mesh) to 36x (25-30 mm)
stiffer than the law, a plane 9x to 15x (check_unified.py cushion). The Newton
backend (newton_bag.py) is the one to use for cushioned bags. solimp gives the stiffening
from the soft modulus to cushion_lock_ratio times that modulus over cushion_range_m.
The law acts on each face of the midsurface, so a wall pinched between two surfaces
compresses twice the per-face depth. With the cushion on, the bag's contact priority
makes its friction and cushion govern every pair it touches. It has viscous damping
only: no hysteresis, air leakage or bubble rupture.

MuJoCo may evaluate passive forces repeatedly; callbacks never mutate material
history. Only a successfully completed mj_step advances plastic, viscous and crease
state. Opening the generated XML alone omits this Python constitutive law.

Geometry npz: vertices (N,3), faces (F,3), uv (N,2); optional masses_kg (N,),
panel_ids (F,) for self-contact groups and precreased boundaries, region_ids (F,)
for material regions, initial_vertices (N,3), hinge_stiffness_scale and
hinge_yield_scale (H,) on lexicographically sorted interior edges.
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

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    # mass
    "mass_kg": None,
    "areal_density_gsm": None,
    # membrane
    "membrane_model": "inextensible",
    "edge_solref_s": 0.001,
    "edge_solimp": [0.99, 0.999, 0.0001],
    "young_pa": 2e7,
    "poisson": 0.3,
    "membrane_thickness_m": 4e-4,
    "membrane_damping": 0.004,
    # bending
    "bending_rigidity_nm": 0.00135,
    "region_bending_scale": {},
    "anisotropy_axis": [0.0, 1.0, 0.0],
    "anisotropy_ratio": 1.0,
    "bending_knee_deg": None,
    "bending_post_ratio": 1.0,
    "plasticity": True,
    "plastic_yield_angle_deg": 25.0,
    "plastic_hardening_ratio": 0.08,
    "precreased_stiffness_ratio": 1.0,
    "precreased_panel_ids": [],
    "precreased_panel_boundaries": True,
    "memory_ratio": 0.6,
    "recovery_tau_s": 2.0,
    "bending_damping_nms": 3e-7,
    "crease_softening": False,
    "crease_onset_deg": 25.0,
    "crease_width_deg": 35.0,
    "crease_max_softening": 0.65,
    # thickness and contact
    "collision_radius_m": 0.00035,
    "collision_patch_faces": 32,
    "contact_timeconst_s": 0.0002,
    "cushion": False,
    "cushion_range_m": 0.0025,
    "cushion_soft_modulus_pa_per_m": 5e6,
    "cushion_lock_ratio": 10.0,
    "cushion_damping_ratio": 1.0,
    "friction": 0.45,
    "table_friction": 0.8,
    "self_contact": True,
    "self_contact_front_panel_ids": [0],
    "self_contact_back_panel_ids": [1, 2],
    # world and solver
    "velocity_damping_per_s": 3.0,
    "gravity": True,
    "timestep_s": 0.00005,
    "frame_dt_s": 0.01,
    "iterations": 200,
    "solver": "CG",
    "tolerance": 1e-8,
    # Newton (VBD) backend only; MuJoCo ignores these. VBD's per-vertex solve sees the membrane's
    # out-of-plane Hessian, so with few iterations bending converges only when the substep is
    # below the membrane wave step h*sqrt(rho/mu): a single hinge kept 0.7/6/48 % of its stiffness
    # at 1/10/100 iterations and 1 ms. The substep is newton_substep_courant times that step,
    # capped by newton_substep_s.
    "newton_substep_s": 0.001,
    "newton_substep_courant": 0.7,
    # Float32 positions: below about 5e-5 s a substep's displacement is rounded away and VBD,
    # which derives velocity from positions, freezes (a 10 mm strip at 2.4e-5 s never moved).
    # When the wave step demands less, the membrane is softened until it fits; state_report
    # gives the factor.
    "newton_min_substep_s": 5e-5,
    "newton_iterations": 10,
    "newton_contact_modulus_pa_per_m": 2e8,
    # Wall-to-wall interaction distance as a multiple of collision_radius_m (2 = walls of
    # thickness 2r). VBD self contact is a log barrier below half that distance, so a starting
    # pair closer than that is launched (a wrapped PE had an edge pair 0.018 mm apart). The
    # distance therefore grows from zero to its value over newton_self_contact_ramp_s, pushing
    # such pairs apart through the linear penalty first.
    "newton_self_contact_scale": 2.0,
    "newton_self_contact_ramp_s": 0.1,
    # Newton-only switch, so the MuJoCo self_contact setting and its regression stay untouched.
    "newton_self_contact": True,
    # Self-contact stiffness as a modulus (Pa/m); None = the body contact modulus. VBD uses one
    # particle stiffness for self contact and averages it with each rigid shape's, so shapes get
    # 2*body - self and body contact keeps its stiffness. A PE wrapped round a book blew apart at
    # 2e8, slid 33 mm at 2e7 and settled at 2e6 (28 mg nodes).
    "newton_self_contact_modulus_pa_per_m": None,
    # Rigid shapes also collide with bag edges and faces, not only vertices, so sharp corners
    # cannot slip between nodes (13 mm apart on the bubble mailer).
    "newton_full_surface_contact": True,
}

MEMBRANES = {"inextensible", "tension_only", "elastic"}
BOOL_KEYS = {"plasticity", "self_contact", "gravity", "precreased_panel_boundaries", "crease_softening", "cushion",
             "newton_full_surface_contact", "newton_self_contact"}
LIST_KEYS = {"precreased_panel_ids", "self_contact_front_panel_ids", "self_contact_back_panel_ids"}
OPTIONAL_KEYS = {"mass_kg", "areal_density_gsm", "bending_knee_deg", "newton_self_contact_modulus_pa_per_m"}
SPECIAL_KEYS = {"membrane_model", "solver", "edge_solimp", "region_bending_scale", "anisotropy_axis"}
POSITIVE_KEYS = {"bending_rigidity_nm", "plastic_yield_angle_deg", "recovery_tau_s", "timestep_s", "frame_dt_s",
                 "collision_radius_m", "contact_timeconst_s", "edge_solref_s", "iterations", "tolerance",
                 "young_pa", "membrane_thickness_m", "anisotropy_ratio", "bending_post_ratio",
                 "crease_onset_deg", "crease_width_deg", "cushion_range_m", "cushion_soft_modulus_pa_per_m",
                 "cushion_damping_ratio", "newton_substep_s", "newton_substep_courant", "newton_min_substep_s", "newton_iterations", "newton_contact_modulus_pa_per_m", "newton_self_contact_scale"}
CUSHION_DMAX = 0.99

_RUNTIMES = {}
_PREVIOUS_PASSIVE = None


def _numbers(values):
    return " ".join(f"{float(value):.14g}" for value in np.asarray(values).ravel())


def _wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def parameters(source):
    unknown = set(source) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **source}
    if p["membrane_model"] not in MEMBRANES:
        raise ValueError(f"membrane_model must be one of {sorted(MEMBRANES)}")
    if p["solver"] not in {"CG", "Newton"}:
        raise ValueError("solver must be CG or Newton")
    for key in BOOL_KEYS:
        if not isinstance(p[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in LIST_KEYS:
        if not isinstance(p[key], list) or any(isinstance(v, bool) or not isinstance(v, int) for v in p[key]):
            raise ValueError(f"{key} must be a list of integer identifiers")
    numeric = set(DEFAULTS) - BOOL_KEYS - LIST_KEYS - SPECIAL_KEYS
    for key in numeric:
        value = p[key]
        if value is None and key in OPTIONAL_KEYS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
        if key in POSITIVE_KEYS and value <= 0:
            raise ValueError(f"{key} must be positive")
    if p["mass_kg"] is not None and p["areal_density_gsm"] is not None:
        raise ValueError("Set mass_kg or areal_density_gsm, not both")
    if not 0 <= p["plastic_hardening_ratio"] < 1:
        raise ValueError("plastic_hardening_ratio must be in [0,1)")
    if not 0 < p["precreased_stiffness_ratio"] <= 1:
        raise ValueError("precreased_stiffness_ratio must be in (0,1]")
    if not 0 <= p["crease_max_softening"] < 1:
        raise ValueError("crease_max_softening must be in [0,1)")
    if not 0 <= p["poisson"] < 0.5:
        raise ValueError("poisson must be in [0,0.5)")
    if p["cushion_lock_ratio"] < 1:
        raise ValueError("cushion_lock_ratio must be at least 1")
    if p["cushion"] and p["cushion_range_m"] >= p["collision_radius_m"]:
        raise ValueError("cushion_range_m is per face and must be smaller than collision_radius_m")
    if set(p["self_contact_front_panel_ids"]) & set(p["self_contact_back_panel_ids"]):
        raise ValueError("Self-contact front and back panel groups must not overlap")
    if any(int(p[k]) != p[k] for k in ("iterations", "collision_patch_faces", "newton_iterations")):
        raise ValueError("iterations, collision_patch_faces and newton_iterations must be integers")
    ratio = p["frame_dt_s"] / p["timestep_s"]
    if abs(ratio - round(ratio)) > 1e-7 or ratio < 1:
        raise ValueError("frame_dt_s must be an integer multiple of timestep_s")
    impedance = np.asarray(p["edge_solimp"], dtype=float)
    if impedance.shape != (3,) or not np.isfinite(impedance).all() or not 0 < impedance[0] <= impedance[1] < 1 or impedance[2] <= 0:
        raise ValueError("edge_solimp must be [0 < dmin <= dmax < 1, positive width]")
    axis = np.asarray(p["anisotropy_axis"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("anisotropy_axis must be a nonzero 3-vector")
    if not isinstance(p["region_bending_scale"], dict) or any(
            not re.fullmatch(r"\d+", str(k)) or isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
            for k, v in p["region_bending_scale"].items()):
        raise ValueError("region_bending_scale maps region id strings to positive scales")
    return p


def cushion_contact(params, areal_density):
    """solref/solimp whose contact pressure follows the profile's cushion law.

    Steady soft contact gives force ~ m_eff * d/(1-d) * k * depth with m_eff ~ areal
    density times contact area, so dividing k by the areal density yields pressure.
    """
    g = lambda d: d / (1 - d)
    g_min = g(CUSHION_DMAX) / params["cushion_lock_ratio"]
    d_min = g_min / (1 + g_min)
    k = params["cushion_soft_modulus_pa_per_m"] / (areal_density * g_min)
    b = 2 * params["cushion_damping_ratio"] * math.sqrt(k / g_min)
    return [-k, -b], [d_min, CUSHION_DMAX, params["cushion_range_m"], 0.5, 1.0]


def cushion_pressure(params, depth):
    """Pressure in Pa predicted by the cushion law at a compression depth in m."""
    g = lambda d: d / (1 - d)
    g_min = g(CUSHION_DMAX) / params["cushion_lock_ratio"]
    d_min = g_min / (1 + g_min)
    x = np.clip(np.asarray(depth, float) / params["cushion_range_m"], 0, 1)
    d = d_min + (CUSHION_DMAX - d_min) * x
    return params["cushion_soft_modulus_pa_per_m"] / g_min * g(d) * np.asarray(depth, float)


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
    return edges, lengths, hinges, weight, face_ids, areas, report


def _contents(source, base_dir):
    """Normalize independent rigid-mesh contents without moving their sources."""
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
            path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            normalized[key] = str(path)
        for key, size, default in (("position_m", 3, None), ("quaternion_wxyz", 4, [1, 0, 0, 0]),
                                   ("scale", 3, [1, 1, 1]), ("rgba", 4, [.6, .3, .15, 1])):
            value = np.asarray(item.get(key, default), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"contents[{index}].{key} must have {size} finite values")
            normalized[key] = value
        norm = np.linalg.norm(normalized["quaternion_wxyz"])
        if np.any(normalized["scale"] <= 0) or norm < 1e-12:
            raise ValueError(f"contents[{index}] needs positive scale and a nonzero quaternion")
        normalized["quaternion_wxyz"] /= norm
        mass = item.get("mass_kg")
        if isinstance(mass, bool) or not isinstance(mass, (float, int)) or not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"contents[{index}].mass_kg must be positive and finite")
        normalized["mass_kg"] = float(mass)
        normalized["friction"] = float(item.get("friction", 0.6))
        result.append(normalized)
    return result


def load_material(geometry, p, initial_vertices=None):
    """Geometry plus every per-node and per-hinge material array; shared by both backends."""
    with np.load(geometry, allow_pickle=False) as g:
        rest = np.array(g["vertices"], dtype=float)
        faces = np.array(g["faces"], dtype=np.int32)
        uv = np.array(g["uv"], dtype=float)
        masses = np.array(g["masses_kg"], dtype=float) if "masses_kg" in g else None
        panel_ids = np.array(g["panel_ids"], dtype=np.int32) if "panel_ids" in g else np.zeros(len(faces), np.int32)
        region_ids = np.array(g["region_ids"], dtype=np.int32) if "region_ids" in g else np.zeros(len(faces), np.int32)
        initial = np.array(g["initial_vertices"], dtype=float) if "initial_vertices" in g else None
        scale = np.array(g["hinge_stiffness_scale"], dtype=float) if "hinge_stiffness_scale" in g else None
        yield_scale = np.array(g["hinge_yield_scale"], dtype=float) if "hinge_yield_scale" in g else None
    if initial_vertices is not None:
        initial = np.array(initial_vertices, dtype=float, copy=True)
    count = len(rest)
    if (rest.shape != (count, 3) or count < 3 or not np.isfinite(rest).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or np.any(faces < 0) or np.any(faces >= count)
            or uv.shape != (count, 2) or not np.isfinite(uv).all()
            or panel_ids.shape != (len(faces),) or region_ids.shape != (len(faces),)):
        raise ValueError("Invalid geometry vertices/faces/uv/panel_ids/region_ids")
    edges, rest_lengths, hinges, weights, hinge_faces, areas, topology = _topology(rest, faces)

    # Mass: stored lumped masses, optionally rescaled; or recomputed from areal density.
    if p["areal_density_gsm"] is not None or masses is None:
        density = (p["areal_density_gsm"] if p["areal_density_gsm"] is not None else 100.) * 1e-3
        masses = np.zeros(count)
        np.add.at(masses, faces.ravel(), np.repeat(areas * density / 3, 3))
    if p["mass_kg"] is not None:
        masses = masses * (p["mass_kg"] / masses.sum())
    if masses.shape != (count,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Lumped masses must be positive and finite")
    masses_kg = masses

    # Bending stiffness per hinge: D * geometric weight * region * anisotropy * crease scale.
    stiffness = p["bending_rigidity_nm"] * weights
    region_scale = np.array([float(p["region_bending_scale"].get(str(r), 1.0)) for r in range(region_ids.max() + 1)])
    stiffness *= np.min(region_scale[region_ids[hinge_faces]], axis=1)
    axis = np.asarray(p["anisotropy_axis"], float)
    axis /= np.linalg.norm(axis)
    direction = rest[hinges[:, 1]] - rest[hinges[:, 0]]
    along = (direction @ axis) ** 2 / np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-24)
    # Hinge edges normal to the axis resist curvature along it.
    stiffness *= 1 + (p["anisotropy_ratio"] - 1) * (1 - along)
    hinge_damping = np.full(len(hinges), p["bending_damping_nms"])
    panel_pairs = panel_ids[hinge_faces]
    precreased = np.any(np.isin(panel_pairs, p["precreased_panel_ids"]), axis=1)
    if p["precreased_panel_boundaries"]:
        precreased |= panel_pairs[:, 0] != panel_pairs[:, 1]
    crease_scale = np.where(precreased, p["precreased_stiffness_ratio"], 1.0)
    stiffness *= crease_scale
    hinge_damping *= crease_scale
    topology["precreased_hinges"] = int(np.count_nonzero(precreased))
    yield_angles = np.full(len(hinges), np.deg2rad(p["plastic_yield_angle_deg"]))
    for label, values, target in (("hinge_stiffness_scale", scale, stiffness),
                                  ("hinge_yield_scale", yield_scale, yield_angles)):
        if values is not None:
            if values.shape != target.shape or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{label} must have one positive finite value per interior edge")
            target *= values
    base_stiffness = stiffness.copy()
    knee = None if p["bending_knee_deg"] is None else np.deg2rad(p["bending_knee_deg"])
    return dict(rest=rest, faces=faces, uv=uv, panel_ids=panel_ids, region_ids=region_ids, initial=initial,
                edges=edges, rest_lengths=rest_lengths, hinges=hinges, hinge_faces=hinge_faces,
                face_areas=areas, topology=topology, masses_kg=masses_kg, stiffness=stiffness,
                base_stiffness=base_stiffness, hinge_damping=hinge_damping, precreased=precreased,
                yield_angles=yield_angles, knee=knee)


def _passive_dispatch(model, data):
    if _PREVIOUS_PASSIVE is not None:
        _PREVIOUS_PASSIVE(model, data)
    reference = _RUNTIMES.get(model._address)
    runtime = reference() if reference is not None else None
    if runtime is not None and not runtime.closed:
        runtime._passive(model, data)


class BagModel:
    """Factory contract: model, data, step(), close(); history is instance-owned."""

    def __init__(self, geometry, params, *, texture=None, name="bag", support=True,
                 initial_vertices=None, contents=(), base_dir=None):
        import mujoco

        if not mujoco.__version__.startswith("3.14."):
            raise RuntimeError("Unified bag model targets MuJoCo >=3.14,<3.15")
        self.name = name
        self.params = parameters(params)
        self.texture = None if texture is None else str(Path(texture).resolve())
        self.contents = _contents(contents, Path(base_dir or HERE))
        self.support = bool(support)
        self.closed = False
        p = self.params
        material = load_material(geometry, p, initial_vertices)
        initial = material.pop("initial")
        self.__dict__.update(material)
        count = len(self.rest)

        self.rest_angles, _ = dihedrals(self.rest, self.hinges, gradient=False)
        self.last_angles = self.rest_angles.copy()
        self.total_angles = np.zeros(len(self.hinges))
        self.plastic = np.zeros(len(self.hinges))
        self.accumulated_plastic = np.zeros(len(self.hinges))
        self.memory = np.zeros(len(self.hinges))
        self.crease_peak = np.zeros(len(self.hinges))
        self.crease_damage = np.zeros(len(self.hinges))
        self.plastic_dissipated_j = 0.0
        self.memory_dissipated_j = 0.0
        self.callback_calls = 0
        self.accepted_steps = 0

        self.xml = self._xml()
        self.compiler_warnings = []
        previous_warning = mujoco.get_mju_user_warning()

        def compile_warning(message):
            expected = ((message.startswith("flex 'bag_contact_") or message.startswith("flex 'bag'"))
                        and "is not rigid and has no equality constraints or passive forces" in message)
            self.compiler_warnings.append({"message": message, "expected_contact_patch_notice": expected})
            if not expected:
                if previous_warning is not None:
                    previous_warning(message)
                else:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)

        try:
            mujoco.set_mju_user_warning(compile_warning)
            self.model = mujoco.MjModel.from_xml_string(self.xml)
        finally:
            mujoco.set_mju_user_warning(previous_warning)
        self.data = mujoco.MjData(self.model)
        flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "bag")
        address = int(self.model.flex_vertadr[flex_id])
        self.vert_slice = slice(address, address + count)
        self.body_ids = self.model.flex_vertbodyid[self.vert_slice].copy()
        self.content_body_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, item["xml_name"])
                                            for item in self.contents], dtype=np.int32)
        joint_ids = self.model.body_jntadr[self.body_ids, None] + np.arange(3)
        if (not np.all(self.model.body_jntnum[self.body_ids] == 3)
                or not np.all(self.model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_SLIDE)
                or not np.allclose(self.model.jnt_axis[joint_ids], np.broadcast_to(np.eye(3), (count, 3, 3)))
                or not np.all(self.model.body_parentid[self.body_ids] == 0)):
            raise RuntimeError("Expected independent world-aligned three-translation vertex bodies")
        self.dofs = self.model.jnt_dofadr[joint_ids]
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.model.dof_damping[self.dofs] = self.masses_kg[:, None] * p["velocity_damping_per_s"]
        if initial is None:
            initial = self.rest.copy()
            if self.support:
                initial[:, 2] += max(0.0, p["collision_radius_m"] + 0.00015 - initial[:, 2].min())
        if initial.shape != self.rest.shape or not np.isfinite(initial).all():
            raise ValueError("initial_vertices must match reference vertex shape and be finite")
        self.initial = initial.copy()
        self.data.qpos[self.qpos_ids] = self.initial - self.rest
        global _PREVIOUS_PASSIVE
        if not _RUNTIMES:
            current = mujoco.get_mjcb_passive()
            _PREVIOUS_PASSIVE = current if current is not _passive_dispatch else None
            mujoco.set_mjcb_passive(_passive_dispatch)
        elif mujoco.get_mjcb_passive() is not _passive_dispatch:
            raise RuntimeError("Another owner replaced the process passive callback; close it before loading")
        _RUNTIMES[self.model._address] = weakref.ref(self)
        mujoco.mj_forward(self.model, self.data)

    @property
    def positions(self):
        return self.rest + self.data.qpos[self.qpos_ids]

    def _contact_attributes(self, patch_faces):
        p = self.params
        attributes = dict(friction=_numbers([p["friction"], .005, .0001]), margin="0")
        if p["cushion"]:
            used = np.unique(self.faces[patch_faces])
            density = self.masses_kg[used].sum() / max(self.face_areas[patch_faces].sum(), 1e-12)
            solref, solimp = cushion_contact(p, density)
            # Priority makes the bag's cushion law and friction govern every pair it touches.
            attributes.update(solref=_numbers(solref), solimp=_numbers(solimp), priority="2")
        else:
            attributes.update(solref=_numbers([p["contact_timeconst_s"], 1]), solimp=".99 .999 .0001")
        return attributes

    def _xml(self):
        p = self.params
        root = ET.Element("mujoco", model=f"unified_{self.name}")
        ET.SubElement(root, "compiler", angle="radian")
        opt = ET.SubElement(root, "option", timestep=str(p["timestep_s"]), integrator="discrete", solver=p["solver"],
                            jacobian="sparse", iterations=str(int(p["iterations"])), tolerance=str(p["tolerance"]),
                            gravity="0 0 -9.81" if p["gravity"] else "0 0 0")
        ET.SubElement(opt, "flag", autoreset="disable")
        ET.SubElement(root, "size", memory="256M")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="1200", offheight="900")
        ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".45 .45 .45", specular=".1 .1 .1")
        asset = ET.SubElement(root, "asset")
        if self.texture is not None and Path(self.texture).is_file():
            ET.SubElement(asset, "texture", name="bag_texture", type="2d", file=self.texture)
            ET.SubElement(asset, "material", name="bag_material", texture="bag_texture", rgba=".85 .85 .85 1",
                          emission=".18", specular=".05", shininess=".1")
        else:
            ET.SubElement(asset, "material", name="bag_material", rgba=".68 .46 .23 1", specular=".05", shininess=".1")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", name="key", pos="-.3 -.4 1.2", dir=".2 .3 -1", diffuse=".65 .65 .65", castshadow="false")
        ET.SubElement(world, "light", name="fill", pos=".4 .4 .8", dir="-.3 -.3 -1", diffuse=".25 .25 .25", castshadow="false")
        ET.SubElement(world, "camera", name="overview", pos="0 -.57 .65", xyaxes="1 0 0 0 .75 .66", fovy="42")
        if self.support:
            ET.SubElement(world, "geom", name="table", type="plane", size=".65 .65 .03", rgba=".22 .27 .32 1",
                          contype="2", conaffinity="1",
                          friction=_numbers([p["table_friction"], .005, .0001]), solref=".002 1", solimp=".99 .999 .0001")
        body_names = []
        for index, (point, mass) in enumerate(zip(self.rest, self.masses_kg)):
            body_name = f"bag_v{index}"
            body_names.append(body_name)
            body = ET.SubElement(world, "body", name=body_name, pos=_numbers(point))
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass), diaginertia="1e-10 1e-10 1e-10")
            for label, axis in zip("xyz", np.eye(3)):
                ET.SubElement(body, "joint", name=f"{body_name}_{label}", type="slide", axis=_numbers(axis), limited="false")
            if p["membrane_model"] == "tension_only":
                ET.SubElement(body, "site", name=f"{body_name}_s", size=".0003", group="5")
        for item in self.contents:
            xml_name = item["xml_name"]
            ET.SubElement(asset, "mesh", name=xml_name + "_mesh", file=item["mesh_path"], scale=_numbers(item["scale"]), inertia="convex")
            material = {"name": xml_name + "_material", "rgba": _numbers(item["rgba"]), "specular": ".1", "shininess": ".1"}
            if item["texture_path"] is not None:
                ET.SubElement(asset, "texture", name=xml_name + "_texture", type="2d", file=item["texture_path"])
                material["texture"] = xml_name + "_texture"
            ET.SubElement(asset, "material", **material)
            body = ET.SubElement(world, "body", name=xml_name, pos=_numbers(item["position_m"]), quat=_numbers(item["quaternion_wxyz"]))
            ET.SubElement(body, "freejoint", name=xml_name + "_free")
            ET.SubElement(body, "geom", name=xml_name + "_geom", type="mesh", mesh=xml_name + "_mesh", material=xml_name + "_material",
                          mass=str(item["mass_kg"]), contype="2", conaffinity="3", condim="3",
                          friction=_numbers([item["friction"], .005, .0001]), solref=".002 1", solimp=".99 .999 .0001")
        deformable = ET.SubElement(root, "deformable")
        flex = ET.SubElement(deformable, "flex", name="bag", dim="2", body=" ".join(body_names),
                             element=" ".join(str(int(v)) for v in self.faces.ravel()),
                             texcoord=_numbers(self.uv), radius=str(p["collision_radius_m"]),
                             material="bag_material", flatskin="true")
        ET.SubElement(flex, "contact", selfcollide="none", contype="0", conaffinity="0", internal="false")
        if p["membrane_model"] == "elastic":
            ET.SubElement(flex, "elasticity", young=str(p["young_pa"]), poisson=str(p["poisson"]),
                          thickness=str(p["membrane_thickness_m"]), damping=str(p["membrane_damping"]), elastic2d="stretch")
        # Collision-only patches share the vertex bodies; only front/back panel groups self-collide.
        patch_size = int(p["collision_patch_faces"]) or len(self.faces)
        front, back = set(p["self_contact_front_panel_ids"]), set(p["self_contact_back_panel_ids"])
        patch_id = 0
        for panel in np.unique(self.panel_ids):
            panel_face_ids = np.flatnonzero(self.panel_ids == panel)
            for start in range(0, len(panel_face_ids), patch_size):
                face_ids = panel_face_ids[start:start + patch_size]
                used, remapped = np.unique(self.faces[face_ids], return_inverse=True)
                patch = ET.SubElement(deformable, "flex", name=f"bag_contact_{patch_id}", dim="2",
                                      body=" ".join(body_names[i] for i in used),
                                      element=" ".join(str(int(v)) for v in remapped.ravel()),
                                      radius=str(p["collision_radius_m"]), group="5", rgba="0 0 0 0")
                contype, conaffinity = 1, 2
                if p["self_contact"] and panel in front:
                    contype |= 8
                    conaffinity |= 16
                elif p["self_contact"] and panel in back:
                    contype |= 16
                    conaffinity |= 8
                ET.SubElement(patch, "contact", selfcollide="none", internal="false", contype=str(contype),
                              conaffinity=str(conaffinity), condim="3", **self._contact_attributes(face_ids))
                patch_id += 1
        if p["membrane_model"] == "inextensible":
            equality = ET.SubElement(root, "equality")
            ET.SubElement(equality, "flex", name="bag_low_stretch", flex="bag",
                          solref=_numbers([p["edge_solref_s"], 1]), solimp=_numbers(p["edge_solimp"]))
        elif p["membrane_model"] == "tension_only":
            # One-sided length limits: edges resist stretching but shorten freely, which
            # stands in for wrinkles below the mesh resolution.
            tendon = ET.SubElement(root, "tendon")
            for (a, b), length in zip(self.edges, self.rest_lengths):
                spatial = ET.SubElement(tendon, "spatial", limited="true", range=f"0 {length:.12g}", width=".0001",
                                        solreflimit=_numbers([p["edge_solref_s"], 1]), solimplimit=_numbers(p["edge_solimp"]))
                ET.SubElement(spatial, "site", site=f"bag_v{a}_s")
                ET.SubElement(spatial, "site", site=f"bag_v{b}_s")
        return ET.tostring(root, encoding="unicode")

    # ---------- bending law ----------

    def _trial_plastic(self, delta):
        if not self.params["plasticity"]:
            return self.plastic, self.accumulated_plastic, np.zeros_like(delta)
        hardening = self.params["plastic_hardening_ratio"] / (1 - self.params["plastic_hardening_ratio"])
        trial = delta - self.plastic
        increment = np.maximum(np.abs(trial) - self.yield_angles - hardening * self.accumulated_plastic, 0) / (1 + hardening)
        return self.plastic + increment * np.sign(trial), self.accumulated_plastic + increment, increment

    def _equilibrium(self, elastic):
        """Linear, or bilinear above the knee angle."""
        if self.knee is None:
            return elastic
        a = np.abs(elastic)
        return np.sign(elastic) * np.where(a <= self.knee, a, self.knee + self.params["bending_post_ratio"] * (a - self.knee))

    def _energy_density(self, elastic, delta):
        p = self.params
        if self.knee is None:
            equilibrium = .5 * elastic ** 2
        else:
            a = np.abs(elastic)
            u, v = np.minimum(a, self.knee), np.maximum(a - self.knee, 0)
            equilibrium = .5 * u * u + self.knee * v + .5 * p["bending_post_ratio"] * v * v
        return equilibrium + .5 * p["memory_ratio"] * (delta - self.memory) ** 2

    def _passive(self, model, data):
        self.callback_calls += 1
        q = data.flexvert_xpos[self.vert_slice]
        angle, gradient = dihedrals(q, self.hinges)
        delta = self.total_angles + _wrap(angle - self.last_angles)
        plastic, _, _ = self._trial_plastic(delta)
        moment = self.stiffness * (self._equilibrium(delta - plastic) + self.params["memory_ratio"] * (delta - self.memory))
        rate = np.einsum("hij,hij->h", gradient, data.qvel[self.dofs][self.hinges])
        moment += self.hinge_damping * rate
        force = np.zeros_like(q)
        contribution = -moment[:, None, None] * gradient
        for index in range(4):
            np.add.at(force, self.hinges[:, index], contribution[:, index])
        data.qfrc_passive[self.dofs] += force

    def _advance_history(self):
        p = self.params
        angle, _ = dihedrals(self.positions, self.hinges, gradient=False)
        self.total_angles += _wrap(angle - self.last_angles)
        self.last_angles = angle
        self.plastic, self.accumulated_plastic, increment = self._trial_plastic(self.total_angles)
        self.plastic_dissipated_j += float(np.sum(self.stiffness * self.yield_angles * increment))
        old = self.total_angles - self.memory
        self.memory += -np.expm1(-p["timestep_s"] / p["recovery_tau_s"]) * old
        self.memory_dissipated_j += float(np.sum(.5 * self.stiffness * p["memory_ratio"]
                                               * (old ** 2 - (self.total_angles - self.memory) ** 2)))
        if p["crease_softening"]:
            # Irreversible bounded softening driven by the peak bend; the rest angle stays fixed.
            self.crease_peak = np.maximum(self.crease_peak, np.abs(self.total_angles))
            excess = np.maximum(self.crease_peak - np.deg2rad(p["crease_onset_deg"]), 0)
            self.crease_damage = p["crease_max_softening"] * -np.expm1(-excess / np.deg2rad(p["crease_width_deg"]))
            self.stiffness = self.base_stiffness * (1 - self.crease_damage)
        self.accepted_steps += 1

    # ---------- stepping and reporting ----------

    def step_substeps(self, count=1):
        import mujoco

        if self.closed:
            raise RuntimeError("Bag model is closed")
        if isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError("count must be a nonnegative integer")
        for _ in range(int(count)):
            previous_time = float(self.data.time)
            before = self.data.warning.number.copy()
            mujoco.mj_step(self.model, self.data)
            if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                    or not np.isfinite(self.data.qacc).all()
                    or abs(self.data.time - previous_time - self.params["timestep_s"]) > 1e-9
                    or np.any(self.data.warning.number > before)):
                raise RuntimeError("MuJoCo bag step failed or issued a warning; material history was not advanced")
            self._advance_history()

    def step(self):
        self.step_substeps(round(self.params["frame_dt_s"] / self.params["timestep_s"]))

    def state_report(self):
        positions = self.positions
        lengths = np.linalg.norm(positions[self.edges[:, 1]] - positions[self.edges[:, 0]], axis=1)
        strain = lengths / self.rest_lengths - 1
        elastic = self.total_angles - self.plastic
        return {"profile": self.name, "time_s": float(self.data.time), "accepted_steps": self.accepted_steps,
                "topology": self.topology, "mass_kg": float(self.masses_kg.sum()),
                "membrane_model": self.params["membrane_model"], "cushion": self.params["cushion"],
                "finite": bool(np.isfinite(positions).all()),
                "max_edge_stretch": float(strain.max()), "max_edge_compression": float(-strain.min()),
                "min_vertex_z_m": float(positions[:, 2].min()), "max_vertex_z_m": float(positions[:, 2].max()),
                "max_speed_m_s": float(np.max(np.linalg.norm(self.data.qvel[self.dofs], axis=1))),
                "max_permanent_crease_deg": float(np.rad2deg(np.max(np.abs(self.plastic), initial=0))),
                "max_crease_softening": float(self.crease_damage.max(initial=0)),
                "bending_energy_j": float(np.sum(self.stiffness * self._energy_density(elastic, self.total_angles))),
                "plastic_dissipated_j": self.plastic_dissipated_j, "memory_dissipated_j": self.memory_dissipated_j,
                "warnings": self.data.warning.number.tolist(), "contacts": int(self.data.ncon),
                "solver_iterations": int(self.data.solver_niter[0]),
                "unexpected_compiler_warnings": [w["message"] for w in self.compiler_warnings if not w["expected_contact_patch_notice"]],
                "physical_calibration": False}

    def close(self):
        import mujoco

        if self.closed:
            return
        self.closed = True
        _RUNTIMES.pop(self.model._address, None)
        if not _RUNTIMES and mujoco.get_mjcb_passive() is _passive_dispatch:
            mujoco.set_mjcb_passive(_PREVIOUS_PASSIVE)


def load_profile(name, *, params=None, **kwargs):
    """Load profiles/<name>.json; params override profile values for this instance only."""
    path = HERE / "profiles" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown profile {name!r}; available: {sorted(f.stem for f in (HERE / 'profiles').glob('*.json'))}")
    profile = json.loads(path.read_text())
    merged = {**profile["params"], **(params or {})}
    # Profile paths are relative to this directory.
    geometry = (HERE / profile["geometry"]).resolve()
    texture = (HERE / profile["texture"]).resolve() if profile.get("texture") else None
    return BagModel(geometry, merged, texture=texture, name=name, base_dir=HERE, **kwargs)
