"""Attach the existing PE Parcel flex model to the Yam_B GPU controlled actor."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import warp as wp

from bag_sim.physical_parcel import Config, Parcel, parcel_profile, vec
from bag_sim.physical_parcel_gpu import bending
from bag_sim.wrap import triangle_box_intersections, winding_number
from yam_platform.robot import asset_fingerprint
from yam_platform.yam_model import initial_state, save_yam_xml
from yam_platform.yam_warp import SimYAMInterface
from smoke import Trial

ROOT = Path(__file__).resolve().parents[2]
CENTER = (-.03, -.225)
DT = .0002
SIZE = (.088, .068, .024)


class PEAdapter:
    def __init__(self, offset):
        self.parcel = Parcel(Config(geometry="geometry/kraft_edge_sealed_35.npz",
                                    payload_mass=.20, payload_size=SIZE, payload_offset_xy=(offset, 0.),
                                    timestep=DT, center=CENTER, friction=.35, **parcel_profile("pe")["shell"]))

    def add_scene(self, root):
        parcel = ET.fromstring(self.parcel.xml)
        # Keep the material setup and solver from the original PE model.
        root.remove(root.find("option"))
        root.insert(1, parcel.find("option"))
        root.find("asset").extend(parcel.find("asset"))
        parent_defaults = root.find("default")
        parent_defaults.extend(parcel.find("default"))
        world = root.find("worldbody")
        for node in parcel.find("worldbody"):
            if node.get("name") != "contact_probe" and node.tag != "light":
                world.append(node)

        # Keep rigid–rigid collision candidates available so the test can
        # measure any direct load on the hidden core independently of film.
        root.find(".//default[@class='collision']/geom").set("conaffinity", "9")
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for side in ("left", "right"):
            pad = root.find(f".//geom[@name='{side}_pad']")
            mesh = f"{side}_pad_convex"
            half = np.fromstring(pad.attrib.pop("size"), sep=" ")
            ET.SubElement(root.find("asset"), "mesh", name=mesh, vertex=vec(half * corners))
            pad.set("type", "mesh")
            pad.set("mesh", mesh)

    def couple(self, sim):
        p, d = self.parcel, sim.data
        dofs = p.node_dofs(sim.model)
        with wp.ScopedDevice(sim.device):
            args = [d.flexvert_xpos, d.qvel, wp.array(dofs[:, 0], dtype=int),
                    wp.array(p.hinges, dtype=wp.vec4i), wp.array(p.rest_angles, dtype=float),
                    wp.array(p.hinge_stiffness, dtype=float), wp.array(p.hinge_damping, dtype=float),
                    d.qfrc_passive]

        launch_dim = (sim.worlds, len(p.hinges))
        device = sim.device

        def apply(model, data):
            wp.launch(bending, launch_dim, args, device=device)

        sim.before_solve = apply


def geometry(data, faces):
    points = data.flexvert_xpos
    payload = data.body("payload_0")
    rotation = payload.xmat.reshape(3, 3)
    half = np.asarray(SIZE) / 2
    inset = half - .002
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    probes = np.vstack((payload.xpos, corners * inset @ rotation.T + payload.xpos))
    inside = np.abs(winding_number(points, faces, probes)) >= .5
    solid_crossing = np.any(triangle_box_intersections(points, faces, inset, payload.xpos, rotation))
    return bool(inside.all() and not solid_crossing), max(0., float(-points[:, 2].min()))


def run_batch(output, scan_z):
    places = ("left", "center", "right")
    adapters = [PEAdapter(x) for x in (-.055, 0., .055)]
    nominal = adapters[1]
    sim = SimYAMInterface(len(places), config={"physics_dt_s": DT}, add_scene=nominal.add_scene,
                          compile_model=nominal.parcel.compile_model, initialize=nominal.parcel.initialize, try_capacity=72,
                          capacities={"nconmax": 1024, "njmax": 4096, "njmax_nnz": 32768})
    nominal.couple(sim)
    model = sim.model
    t = Trial("bag", sim=sim)
    qadr = model.joint("payload_free").qposadr[0]
    above = np.array([CENTER[0] - .038, CENTER[1], .061])
    parked = t.ik_path(above)
    start_rows = []
    for adapter in adapters:
        d = initial_state(model, sim.config)
        d.qpos[t.pos[:6]] = parked
        d.qpos[t.pos[6:]] = .006
        d.qpos[qadr:qadr + 3] = adapter.parcel.envelope["payload_origin"]
        adapter.parcel.initialize(model, d)
        start_rows.append(d.qpos.copy())
    sim.reset(initial_qpos=np.array(start_rows, dtype=np.float32))

    arm_body = model.body("arm").id
    robot = set()
    for i in range(model.nbody):
        if i == arm_body or model.body_parentid[i] in robot:
            robot.add(i)
    traces = [[] for _ in places]
    faces = nominal.parcel.envelope["faces"]
    force = np.zeros(6)
    command = np.zeros((len(places), 7), dtype=np.float32)

    def advance(position, gap, phase):
        command[:] = np.r_[t.ik_path(position), gap]
        sim.send_joint_command(command)
        sim.step()
        obs = sim.obs.numpy()
        overflow = sim.data.overflow.numpy()
        for w, trace in enumerate(traces):
            s = sim.snapshot(w)
            load = bypass = depth = film_payload = 0.
            for i, c in enumerate(s.contact):
                if c.efc_address < 0:
                    continue
                geom = [x for x in c.geom if x >= 0]
                robot_contact = any(model.geom_bodyid[x] in robot for x in geom)
                payload_contact = any(model.geom(x).name.startswith("payload_geom") for x in geom)
                film_contact = any(x >= 0 for x in c.flex)
                mujoco.mj_contactForce(model, s, i, force)
                if film_contact and robot_contact:
                    load += force[0]
                    depth = max(depth, -c.dist)
                if payload_contact and robot_contact:
                    bypass += force[0]
                if payload_contact and film_contact:
                    film_payload += force[0]
            enclosed, table_depth = geometry(s, faces)
            trace.append(dict(phase=phase, time=s.time, tcp=s.site_xpos[t.site].copy(), qpos=s.qpos.copy(),
                              payload=s.qpos[qadr:qadr + 7].copy(), obs=obs[w],
                              tau_external=sim.contact_torque(snapshot=s), robot_film_N=load,
                              robot_core_N=bypass, film_core_N=film_payload, robot_film_depth_m=depth,
                              table_depth_m=table_depth, enclosed=enclosed, overflow=int(overflow[w])))

    press = above.copy()
    press[2] = scan_z
    end = press + [.076, 0., 0.]
    for name, start, stop, count, g0, g1 in (
        ("prepare", above, above, 8, .012, .012),
        ("press", above, press, 14, .012, .012),
        ("scan", press, end, 19, .012, .012),
        ("retract", end, end + [0, 0, .022], 5, .012, .012),
    ):
        print(f"{name} {count} steps", flush=True)
        if name == "press":
            sim.history.start_try(sim.all_worlds)
        for a in np.linspace(0, 1, count + 1)[1:]:
            a = a * a * (3. - 2. * a)
            advance(start + a * (stop - start), g0 + a * (g1 - g0), name)
    sim.history.stop_try(sim.all_worlds)

    save_yam_xml(sim.xml, output / "model.xml")
    (output / "config.json").write_text(json.dumps(
        {"scan_z_m": scan_z, "cases": {place: asdict(adapter.parcel.cfg)
                                      for place, adapter in zip(places, adapters, strict=True)}}, indent=2) + "\n")
    archive = {name: value.numpy() for name, value in sim.history.read(full_try=True).items()}
    np.savez_compressed(output / "try_history.npz", **archive)
    results, signals = {}, {}
    for place, trace in zip(places, traces, strict=True):
        raw = {k: np.array([row[k] for row in trace]) for k in trace[0]}
        np.savez_compressed(output / f"{place}.npz", **raw)
        active = np.isin(raw["phase"], ("press", "scan"))
        time = raw["time"][active]
        indices = np.flatnonzero(raw["robot_film_N"][active] >= .08)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        sustained = max((time[r[-1]] - time[r[0]] for r in runs if len(r)), default=0.)
        finite = bool(np.isfinite(raw["obs"]).all() and np.isfinite(raw["qpos"]).all())
        results[place] = dict(peak_robot_film_N=float(raw["robot_film_N"].max()),
                              peak_film_core_N=float(raw["film_core_N"].max()),
                              peak_direct_core_N=float(raw["robot_core_N"].max()),
                              peak_robot_film_depth_m=float(raw["robot_film_depth_m"].max()),
                              peak_table_depth_m=float(raw["table_depth_m"].max()),
                              sustained_contact_s=float(sustained), enclosed=bool(raw["enclosed"].all()),
                              overflow=int(np.bitwise_or.reduce(raw["overflow"])), finite=finite)
        signals[place] = raw["obs"][active, 12:18]
    return results, signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scan-z-m", type=float, default=.039)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT / "test/yam/smoke.py", ROOT / "bag_sim/physical_parcel.py",
             ROOT / "bag_sim/physical_parcel_gpu.py", ROOT / "bag_sim/profiles/pe.json",
             ROOT / "bag_sim/geometry/kraft_edge_sealed_35.npz",
             *sorted((ROOT / "experience/yam_platform").glob("yam_*.*"))]
    (output / "source.json").write_text(json.dumps({
        "host": socket.gethostname(), "mujoco": mujoco.__version__, "warp": wp.__version__,
        "assets": asset_fingerprint(),
        "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }, indent=2) + "\n")
    results, signals = run_batch(output, args.scan_z_m)
    pairs = {"left_center": ("left", "center"), "left_right": ("left", "right"), "center_right": ("center", "right")}
    delta = {k: float(np.max(np.linalg.norm(signals[a] - signals[b], axis=1))) for k, (a, b) in pairs.items()}
    passed = min(delta.values()) >= .03 and all(
        x["sustained_contact_s"] >= .02 and x["enclosed"] and x["peak_robot_film_depth_m"] <= .002
        and x["peak_table_depth_m"] <= .002 and x["peak_direct_core_N"] <= .001 and
        x["finite"] and not x["overflow"] for x in results.values())
    result = dict(pass_gate=passed, dt_s=DT, scan_z_m=args.scan_z_m, cases=results, peak_effort_delta_Nm=delta)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
