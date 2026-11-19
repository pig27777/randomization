"""E1: shared fingertip normal contact on the unchanged physical parcel."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experience")]
from bag_sim.physical_parcel import Config, Parcel, vec
from yam_platform.geometry import fingertip_xml
from yam_platform.protocol import measure_contacts
from physical_parcel_checks import film_payload_geometry
from types import SimpleNamespace


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))


def build(geometry, enabled=True, core="mesh"):
    p = Parcel(Config(geometry=geometry))
    root = ET.fromstring(p.xml)
    root.find("worldbody").remove(root.find(".//body[@name='contact_probe']"))
    tip = ET.fromstring(fingertip_xml("e1_tip", 1, (0, 0, 1)))
    tip.set("mocap", "true")
    for child in list(tip):
        if child.tag != "geom":
            tip.remove(child)
    for geom in tip:
        geom.set("friction", ".8 .005 .0001")
        geom.set("priority", "1")
        geom.set("solref", f"{p.cfg.contact_time} 1")
        geom.set("solimp", ".999 .999 .001")
        geom.set("contype", "4" if enabled else "0")
        geom.set("conaffinity", "3" if enabled else "0")
    if core == "mesh":
        g = tip[0]
        half = np.fromstring(g.attrib.pop("size"), sep=" ")
        corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        ET.SubElement(root.find("asset"), "mesh", name="e1_core_mesh", vertex=vec(corners * half))
        g.set("type", "mesh")
        g.set("mesh", "e1_core_mesh")
    root.find("worldbody").append(tip)
    p.xml = ET.tostring(root, encoding="unicode")
    p.m = p.compile_model(p.xml)
    p.d = mujoco.MjData(p.m)
    p.film_dofs = p.node_dofs(p.m)
    p.initialize(p.m, p.d)
    return p


def tool_contacts(p, group):
    force = np.zeros(3)
    records = []
    for index, c in enumerate(p.d.contact):
        slots = [s for s in (0,1) if int(c.geom[s]) in group]
        if not slots:
            continue
        local = np.zeros(6)
        if c.efc_address >= 0:
            mujoco.mj_contactForce(p.m, p.d, index, local)
        world = c.frame.reshape(3,3).T @ local[:3]
        reaction = (1 if slots[0] == 1 else -1) * world
        force += reaction
        records.append(dict(index=index, geom=c.geom.copy(), flex=c.flex.copy(),
                            elem=c.elem.copy(), vert=c.vert.copy(), dim=int(c.dim),
                            exclude=int(c.exclude), efc_address=int(c.efc_address),
                            dist_m=float(c.dist), pos_m=c.pos.copy(), frame=c.frame.copy(),
                            friction=c.friction.copy(), force_local_N=local,
                            force_on_tip_N=reaction))
    return force, records


def film_constraint_reaction(parcel, records):
    """Recover the tip reaction from the native constraint Jacobian on film DOFs."""
    m, d = parcel.m, parcel.d
    selected = np.zeros(d.nefc)
    for contact in records:
        start = contact["efc_address"]
        if start >= 0:
            stop = start + contact["dim"]  # Elliptic cone in the parcel model.
            selected[start:stop] = d.efc_force[start:stop]
    generalized = np.zeros(m.nv)
    mujoco.mj_mulJacTVec(m, d, generalized, selected)
    return -generalized[parcel.film_dofs].sum(axis=0)


def run(output, geometry, enabled=True, core="mesh", smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    p = build(geometry, enabled, core)
    m,d = p.m,p.d
    tipbody = m.body("e1_tip").id
    mocap = int(m.body_mocapid[tipbody])
    group = set(np.flatnonzero(m.geom_bodyid == tipbody).tolist())
    core_id = m.geom("e1_tip_core").id
    # An unused far-away geom defines the projection axis only; the shared
    # measurement still collects force from exactly this real fingertip group.
    robot = SimpleNamespace(sim=SimpleNamespace(m=m,d=d,table_geom=m.geom("table").id),
        all_finger_sets=[group,set()], pad_geom=[core_id,m.geom("payload_geom_0").id],
        nonfinger_mask=np.zeros(m.ngeom,dtype=bool))
    dt = p.cfg.timestep
    settle = .03 if not smoke else .00025
    for _ in range(round(settle/dt)):
        p.step()
    p.forward()
    triangles = d.flexvert_xpos[p.envelope["faces"]]
    centers = triangles.mean(1)
    normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    candidates = np.flatnonzero((normals[:,2] > .8) & (abs(centers[:,0]) < .05) & (abs(centers[:,1]) < .03))
    if not len(candidates):
        raise RuntimeError("No real top triangle over the payload")
    selected = candidates[np.argmin(np.linalg.norm(centers[candidates,:2],axis=1))]
    target = centers[selected].copy()
    reference = d.flexvert_xpos.copy()
    start_z = target[2] + .020 + p.cfg.radius + .002
    press_z = start_z - .003
    phases = [("hover", .005), ("descend", .15), ("hold", .03), ("retract", .15), ("separated", .01)]
    if smoke:
        phases = [("hover", .00025)]
    config = dict(parcel=asdict(p.cfg), enabled=enabled, core=core, timestep_s=dt,
                  phases=phases, target_triangle=int(selected), target_m=target,
                  start_z_m=start_z, press_z_m=press_z, tip_geom_ids=sorted(group),
                  artificial_attachments=int(m.neq), nsensor=int(m.nsensor),
                  sample_time="contact/poses evaluated at t_eval; integration ends at t_eval+dt")
    write_json(output/"config.json", config)
    (output/"model.xml").write_text(p.xml)
    np.savez_compressed(output/"initial_state.npz", qpos=d.qpos, qvel=d.qvel,
                        reference=reference, faces=p.envelope["faces"])
    rows,states,times = [],[],[]
    step_id = 0
    peaks = dict(force_z_N=0., tangential_N=0., candidates=0, active=0,
                 tool_penetration_m=0., all_penetration_m=0., constraint_mapping_error_N=0., shared_mapping_error_N=0.)
    stage_stats = {}
    with (output/"contacts.jsonl").open("w") as contacts_file:
        for phase, duration in phases:
            stats = dict(steps=0, candidate_steps=0, active_steps=0, peak_force_N=0., peak_force_z_N=0.)
            nsteps = round(duration/dt)
            for step in range(nsteps):
                fraction = (step+1)/nsteps
                z = start_z if phase in ("hover","separated") else press_z if phase == "hold" else (
                    start_z + (press_z-start_z)*fraction if phase == "descend" else press_z+(start_z-press_z)*fraction)
                d.mocap_pos[mocap] = [target[0], target[1], z]
                sampled = step_id % max(1,round(.001/dt)) == 0 or step == nsteps-1
                if sampled:
                    state = d.qpos.copy()
                t_eval = float(d.time)
                p.step()
                force,records = tool_contacts(p,group)
                active = sum(r["efc_address"] >= 0 for r in records)
                stats["steps"] += 1
                stats["candidate_steps"] += int(bool(records))
                stats["active_steps"] += int(active > 0)
                stats["peak_force_N"] = max(stats["peak_force_N"],float(np.linalg.norm(force)))
                stats["peak_force_z_N"] = max(stats["peak_force_z_N"],float(force[2]))
                pen = max([0.] + [-r["dist_m"] for r in records])
                peaks["force_z_N"] = max(peaks["force_z_N"],float(force[2]))
                peaks["tangential_N"] = max(peaks["tangential_N"],float(np.linalg.norm(force[:2])))
                peaks["candidates"] = max(peaks["candidates"],len(records))
                peaks["active"] = max(peaks["active"],active)
                peaks["tool_penetration_m"] = max(peaks["tool_penetration_m"],pen)
                peaks["all_penetration_m"] = max(peaks["all_penetration_m"], max(0.,-float(d.contact.dist.min())) if d.ncon else 0.)
                contacts_file.write(json.dumps(dict(t_eval=t_eval, phase=phase, candidate_count=len(records),
                    active_count=active, force_on_tip_N=force, contacts=records),
                    default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+"\n")
                if sampled:
                    # MuJoCo 4.0.1's cfrc_ext omits contacts involving a flex.
                    constraint_force = film_constraint_reaction(p, records)
                    shared = measure_contacts(robot,target_flexes={0})
                    shared_force = np.asarray(shared["force_on_fingers_N"])[0]
                    peaks["constraint_mapping_error_N"] = max(peaks["constraint_mapping_error_N"],float(np.linalg.norm(force-constraint_force)))
                    peaks["shared_mapping_error_N"] = max(peaks["shared_mapping_error_N"],float(np.linalg.norm(force-shared_force)))
                    niter = int(np.max(d.solver_niter))
                    solver = d.solver
                    gradient = float(solver[max(0,niter-1)].gradient) if niter else 0.
                    rows.append(dict(t_eval=t_eval,phase=phase,tip_pos_m=d.xpos[tipbody].copy(),
                        commanded_tip_pos_m=d.mocap_pos[mocap].copy(), velocity_z_m_s=(0. if phase in ("hover","hold","separated") else -.02 if phase=="descend" else .02),
                        candidate_count=len(records),active_count=active,force_on_tip_N=force,
                        constraint_force_N=constraint_force,shared_force_N=shared_force,
                        shared_grip_N=shared["grip_each_N"],tool_penetration_m=pen,
                        max_node_displacement_m=float(np.linalg.norm(d.flexvert_xpos-reference,axis=1).max()),
                        solver_niter=niter,solver_last_gradient=gradient,warnings=d.warning.number.copy()))
                    states.append(state); times.append(t_eval)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning.number.sum():
                    raise RuntimeError(f"Unhealthy simulation at {d.time}")
                step_id += 1
            stage_stats[phase] = stats
    write_json(output/"trace.json",rows)
    np.savez_compressed(output/"trajectory.npz",qpos=states,time=times,final_qpos=d.qpos,final_qvel=d.qvel)
    checks = dict(finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
        no_warning=bool(d.warning.number.sum()==0),no_attachment=m.neq==0,
        hover_clear=stage_stats["hover"]["candidate_steps"]==0,
        constraint_mapping=peaks["constraint_mapping_error_N"]<1e-6,
        shared_mapping=peaks["shared_mapping_error_N"]<1e-6,
        penetration_budget=peaks["all_penetration_m"]<=.001)
    if not smoke:
        checks["separated_clear"] = stage_stats["separated"]["candidate_steps"]==0
        checks["positive_contact" if enabled else "disabled_no_contact"] = (
            stage_stats["hold"]["active_steps"]>0 and stage_stats["hold"]["peak_force_z_N"]>.001
            if enabled else peaks["candidates"]==0 and peaks["force_z_N"]<1e-9)
    p.forward()
    result = dict(passed=all(checks.values()),checks=checks,peaks=peaks,stages=stage_stats,
                  machine=platform.node(),mujoco=mujoco.__version__,smoke_only=smoke,
                  wall_s=time.perf_counter()-started,final_payload_geometry=film_payload_geometry(m,d,p.envelope["faces"]))
    write_json(output/"summary.json",result)
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--geometry",default="geometry/kraft_edge_sealed_35.npz")
    parser.add_argument("--disabled",action="store_true")
    parser.add_argument("--core",choices=("box","mesh"),default="mesh")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    result=run(args.output,args.geometry,not args.disabled,args.core,args.smoke)
    sources=[Path(__file__),ROOT/"bag_sim/physical_parcel.py",ROOT/"bag_sim/bag_model.py",
             ROOT/"bag_sim/wrap.py",ROOT/"experience/yam_platform/geometry.py",ROOT/"experience/yam_platform/protocol.py"]
    (args.output/"source").mkdir()
    for source in sources:
        (args.output/"source"/source.name).write_bytes(source.read_bytes())
    write_json(args.output/"sources.json",{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
