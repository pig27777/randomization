"""Native MuJoCo 3D-flex flat compression gate; no added material forces.

Run from the repository root. Each invocation requires a new output directory.
Defaults inherit the earlier contact-transfer slab and load protocol, with YAM
fingertips from bag_sim. REPORT.md records the restored first-round plan.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bag_sim.mj_loose import fingertip_xml


@dataclass
class Config:
    case: str = "plate"
    cells: tuple = (12, 10, 4)
    size: tuple = (.30, .25, .05)
    mass: float = .3
    young: float = 3600.
    poisson: float = .2
    damping: float = .02
    dt: float = .001
    seconds: float = 8.
    ramp_s: float = .2
    radius: float = .001
    solref: tuple = (.005, 1.)
    solimp: tuple = (.95, .99, .001)
    iterations: int = 100
    solver: str = "Newton"
    purpose: str = "preflight"


def values(items):
    return " ".join(format(float(x), ".17g") for x in items)


def rest_tets(vertices, tets):
    p = vertices[tets]
    dm = np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=2)
    return np.linalg.inv(dm), np.abs(np.linalg.det(dm)) / 6


def stvk(vertices, tets, inverse_rest, volumes, young, poisson):
    """Independent float64 StVK energy/forces, used only for measurement."""
    p = vertices[tets]
    ds = np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=2)
    deformation = ds @ inverse_rest
    strain = (deformation.transpose(0, 2, 1) @ deformation - np.eye(3)) / 2
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    trace = np.trace(strain, axis1=1, axis2=2)
    energy = np.sum(volumes * (mu * np.sum(strain**2, axis=(1, 2)) + lam * trace**2 / 2))
    stress = deformation @ (2 * mu * strain + lam * trace[:, None, None] * np.eye(3))
    corners = -volumes[:, None, None] * (stress @ inverse_rest.transpose(0, 2, 1)).transpose(0, 2, 1)
    forces = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(forces, tets[:, j + 1], corners[:, j])
    np.add.at(forces, tets[:, 0], -corners.sum(axis=1))
    return float(energy), forces, np.linalg.det(deformation)


def build(cfg):
    count = np.array(cfg.cells) + 1
    spacing = np.array(cfg.size) / np.array(cfg.cells)
    # Same physical position on both grids, between nodes on both grids.
    xy = np.array([.025 / 3, .025 / 3]) if cfg.case == "finger_off" else np.zeros(2)
    load = 19.62 if cfg.case == "plate" else 2.
    if cfg.case == "plate":
        body = ET.fromstring('<body name="tool"><inertial mass="2" pos="0 0 0" diaginertia="1 1 1"/>'
                             '<geom name="tool_core" type="box" size=".12 .10 .01" contype="4" conaffinity="0"/></body>')
        z0 = cfg.size[2] + cfg.radius + .005 + .01
        half_thickness = .01
        axis = "0 0 1"
    else:
        body = ET.fromstring(fingertip_xml("tool", 1, (0, 0, 0)))
        body.set("quat", values((2**-.5, 2**-.5, 0, 0)))
        z0 = cfg.size[2] + cfg.radius + .005 + .004
        half_thickness = .004
        axis = "0 1 0"
    for joint in list(body.findall("joint")):
        body.remove(joint)
    ET.SubElement(body, "joint", name="tool_z", type="slide", axis=axis)
    body.set("pos", values((*xy, z0)))
    body.find("inertial").set("mass", str(load / 9.81))
    tool_xml = ET.tostring(body, encoding="unicode")
    fixed = [i for i in range(int(np.prod(count))) if i % count[2] == 0]
    xml = f'''<mujoco model="flex3d_T1_{cfg.case}">
      <option integrator="discrete" timestep="{cfg.dt}" solver="{cfg.solver}" iterations="{cfg.iterations}"
        tolerance="1e-8" ls_iterations="50" gravity="0 0 0"/>
      <default><geom condim="1" solref="{values(cfg.solref)}" solimp="{values(cfg.solimp)}" friction="0 0 0"/></default>
      <worldbody>{tool_xml}
        <flexcomp name="contents" type="grid" dim="3" dof="full" count="{values(count)}"
          spacing="{values(spacing)}" pos="0 0 {cfg.size[2] / 2}" mass="{cfg.mass}" radius="{cfg.radius}">
          <pin id="{' '.join(map(str, fixed))}"/>
          <elasticity young="{cfg.young}" poisson="{cfg.poisson}" damping="{cfg.damping}"/>
          <contact selfcollide="none" contype="2" conaffinity="4" condim="1" friction="0 0 0"
            solref="{values(cfg.solref)}" solimp="{values(cfg.solimp)}"/>
        </flexcomp>
      </worldbody></mujoco>'''
    # count is an integer-valued XML attribute.
    xml = xml.replace(f'count="{values(count)}"', f'count="{" ".join(map(str, count))}"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml, dict(load=load, xy=xy.tolist(), initial_tool_z=z0,
                                  tool_half_thickness=half_thickness, fixed=fixed)


def contact_force(model, data, tool):
    force = np.zeros(3)
    buffer = np.zeros(6)
    distances = []
    for i, c in enumerate(data.contact):
        if c.efc_address < 0:
            continue
        if c.geom[0] >= 0 and model.geom_bodyid[c.geom[0]] == tool:
            sign = -1
        elif c.geom[1] >= 0 and model.geom_bodyid[c.geom[1]] == tool:
            sign = 1
        else:
            continue
        mujoco.mj_contactForce(model, data, i, buffer)
        force += sign * (c.frame.reshape(3, 3).T @ buffer[:3])
        distances.append(c.dist)
    return force, max([0.] + [-x for x in distances])


def run(cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model, data, xml, metadata = build(cfg)
    (output / "model.xml").write_text(xml)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    tets = model.flex_elem.reshape(-1, 4).copy()
    reference = data.flexvert_xpos.copy()
    inv, volumes = rest_tets(reference, tets)
    bodies = model.flex_vertbodyid.copy()
    active = bodies > 0
    dofs = model.jnt_dofadr[model.body_jntadr[bodies[active]]]
    velocity_ids = dofs[:, None] + np.arange(3)
    tool = model.body("tool").id
    tool_q = model.jnt_qposadr[model.joint("tool_z").id]
    tool_dof = model.jnt_dofadr[model.joint("tool_z").id]
    trace = []
    tail_positions = []
    peak_speed = 0.
    min_det = 1.
    penetration = 0.
    failure = None
    sample_every = max(1, round(.01 / cfg.dt))
    steps = round(cfg.seconds / cfg.dt)
    for step in range(steps):
        fraction = np.clip((data.time + cfg.dt) / cfg.ramp_s, 0, 1)
        fraction = fraction**2 * (3 - 2 * fraction)
        data.qfrc_applied[tool_dof] = -metadata["load"] * fraction
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or abs(data.time - (step + 1) * cfg.dt) > cfg.dt / 2:
            failure = "nonfinite_or_automatic_reset"
            break
        speed = max(float(np.linalg.norm(data.qvel[velocity_ids], axis=1).max()), abs(float(data.qvel[tool_dof])))
        if data.time >= cfg.seconds - .5:
            peak_speed = max(peak_speed, speed)
        if step % sample_every != 0 and step != steps - 1:
            continue
        mujoco.mj_forward(model, data)
        if data.time >= cfg.seconds - .5:
            tail_positions.append(data.flexvert_xpos.copy())
        energy, forces, determinants = stvk(data.flexvert_xpos, tets, inv, volumes, cfg.young, cfg.poisson)
        contact, overlap = contact_force(model, data, tool)
        min_det = min(min_det, float(determinants.min()))
        penetration = max(penetration, overlap)
        trace.append(dict(time=float(data.time), support=float(-forces[metadata["fixed"], 2].sum()),
                          contact_z=float(contact[2]), tool_z=float(data.qpos[tool_q] + metadata["initial_tool_z"]),
                          max_speed_mm_s=speed * 1000, penetration_mm=overlap * 1000,
                          min_det=float(determinants.min()), elastic_energy=energy, contacts=int(data.ncon)))
        if min_det <= 0:
            failure = "tetrahedron_inversion"
            break
    tail = [x for x in trace if x["time"] >= cfg.seconds - .5]
    complete = failure is None and len(tail) >= 2 and data.time >= cfg.seconds - cfg.dt / 2
    metrics = None
    checks = {}
    if complete:
        support = float(np.mean([x["support"] for x in tail]))
        contact_z = float(np.mean([x["contact_z"] for x in tail]))
        tool_z = np.array([x["tool_z"] for x in tail])
        node_drift = float(np.linalg.norm(np.ptp(tail_positions, axis=0), axis=1).max() * 1000)
        metrics = dict(support_error_pct=abs(support / metadata["load"] - 1) * 100,
                       support_n=support, contact_z_n=contact_z,
                       drift_mm=max(node_drift, float(np.ptp(tool_z) * 1000)),
                       tool_drift_mm=float(np.ptp(tool_z) * 1000), node_drift_mm=node_drift,
                       max_speed_mm_s=peak_speed * 1000,
                       penetration_beyond_radius_mm=penetration * 1000,
                       indentation_mm=float((cfg.size[2] + cfg.radius + metadata["tool_half_thickness"] - tool_z.mean()) * 1000),
                       min_det=min_det)
        checks = dict(support=metrics["support_error_pct"] <= 5,
                      drift=metrics["drift_mm"] <= .05, speed=metrics["max_speed_mm_s"] <= 1,
                      penetration=metrics["penetration_beyond_radius_mm"] <= 1)
    np.savez_compressed(output / "final_state.npz", reference=reference, vertices=data.flexvert_xpos,
                        tets=tets, qpos=data.qpos, qvel=data.qvel)
    (output / "trace.json").write_text(json.dumps(trace, indent=2))
    result = dict(config=asdict(cfg), hostname=platform.node(), python=sys.executable, mujoco=mujoco.__version__,
                  metadata=metadata, vertices=model.nflexvert, tetrahedra=model.nflexelem, nv=model.nv,
                  model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete=complete, failure=failure, simulated_seconds=float(data.time),
                  metrics=metrics, checks=checks, local_checks_pass=complete and all(checks.values()),
                  mesh_check="separate coarse/fine comparison required", wall_s=time.perf_counter() - start,
                  timing_is_throughput=False, geometry_sampling_s=sample_every * cfg.dt)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("plate", "finger", "finger_off"), default="plate")
    parser.add_argument("--cells", type=int, nargs=3, default=(12, 10, 4))
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--radius", type=float, default=.001)
    parser.add_argument("--solref", type=float, nargs=2, default=(.005, 1.))
    parser.add_argument("--solimp", type=float, nargs=3, default=(.95, .99, .001))
    parser.add_argument("--solver", choices=("Newton", "CG"), default="Newton")
    parser.add_argument("--purpose", choices=("preflight", "formal", "diagnostic"), default="preflight")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = run(Config(**args), output)
    return 0 if result["local_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
