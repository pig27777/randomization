"""Reproducible physical diagnostics for the archived two-sheet parcel."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bag_sim.physical_parcel import Config, Parcel, compression_force  # noqa: E402
from bag_sim.bag_model import dihedrals  # noqa: E402


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def simulate(parcel, seconds, drive=None):
    trace = []
    stride = max(1, round(0.01 / parcel.cfg.timestep))
    peak = 0.0
    for step in range(round(seconds / parcel.cfg.timestep)):
        if drive is not None:
            drive(parcel, step * parcel.cfg.timestep)
        parcel.step()
        if not np.isfinite(parcel.d.qpos).all() or not np.isfinite(parcel.d.qvel).all():
            raise FloatingPointError(f"nonfinite state at {parcel.d.time}")
        if step % stride == 0:
            parcel.forward()
            forces = parcel.contacts()
            peak = max(peak, forces["penetration_m"])
            trace.append(dict(time_s=float(parcel.d.time), com=parcel.com().copy(),
                              energy_native=parcel.d.energy.copy(), bending_energy_J=parcel.bending_energy(),
                              warning=parcel.d.warning.number.copy(), **forces))
    return trace, peak


def audit(cfg, output):
    p = Parcel(cfg)
    p.save(output)
    rest = mujoco.MjData(p.m)
    p.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    p.m.opt.gravity[:] = 0
    mujoco.mj_forward(p.m, rest)
    p.apply_internal(p.m, rest)
    residual = float(np.max(abs(rest.qfrc_passive)))
    e = p.envelope
    result = dict(passed=bool(residual < 1e-7 and np.all(e["masses"] > 0) and p.m.neq == 0),
                  reference_max_internal_force_N=residual, metadata=p.metadata,
                  initial_penetration_m=p.contacts()["penetration_m"],
                  mass_error_kg=float(e["masses"].sum() - e["metadata"]["reference_area_m2"] * cfg.areal_mass))
    return result


def support(cfg, output):
    p = Parcel(cfg)
    p.save(output)
    start = time.perf_counter()
    trace, peak = simulate(p, 0.3)
    late = [row for row in trace if row["time_s"] >= 0.2]
    weight = (cfg.payload_mass + p.envelope["masses"].sum()) * 9.81
    mean = float(np.mean([row["table_support_N"] for row in late]))
    penetration = max(row["penetration_m"] for row in late)
    bypass = sum(row["payload_table_contacts"] for row in trace)
    warnings = int(p.d.warning.number.sum())
    error = abs(mean - weight) / weight
    np.savez_compressed(output / "final_state.npz", qpos=p.d.qpos, qvel=p.d.qvel)
    (output / "trace.json").write_text(json.dumps(trace, default=serial, indent=2))
    return dict(passed=bool(error <= 0.05 and penetration <= .001 and bypass == 0 and warnings == 0),
                support_N=mean, expected_weight_N=weight, relative_balance_error=error,
                late_peak_penetration_m=penetration, startup_peak_penetration_m=peak,
                payload_table_contacts=bypass, warnings=warnings, wall_s=time.perf_counter()-start)


def contacts(cfg, output):
    output.mkdir(parents=True)
    baseline = Parcel(replace(cfg, gravity=(0, 0, 0)))
    baseline.m.geom_contype[baseline.m.geom("table").id] = 0
    baseline.m.geom_conaffinity[baseline.m.geom("table").id] = 0
    simulate(baseline, .02)
    control = baseline.d.flexvert_xpos.copy()
    rows = []
    for shape in ("box", "capsule", "sphere"):
        for axis, sign, label in ((2,1,"TOP"),(2,-1,"BOTTOM"),(0,-1,"LEFT"),
                                  (0,1,"RIGHT"),(1,-1,"FRONT"),(1,1,"BACK")):
            p = Parcel(replace(cfg, gravity=(0,0,0), probe_shape=shape))
            table = p.m.geom("table").id
            p.m.geom_contype[table] = p.m.geom_conaffinity[table] = 0
            vertices = p.d.flexvert_xpos.copy()
            tri = vertices[p.envelope["faces"]]
            centers = tri.mean(axis=1)
            normals = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals /= np.linalg.norm(normals,axis=1)[:,None]
            candidates = np.flatnonzero(normals[:,axis]*sign > .1)
            if not len(candidates):
                rows.append(dict(shape=shape,face=label,passed=False,error='no outward triangle along this axis'))
                continue
            index = candidates[np.argmax(centers[candidates,axis]*sign)]
            normal = normals[index]
            center = centers[index]
            # Align a probe face/end with the triangle, then approach along the
            # requested world axis. The endpoint targets the actual triangle.
            extent = .016 if shape == "capsule" else .01
            direction=np.eye(3)[axis]*sign
            incidence=float(normal@direction)
            initial = center + normal*(extent+cfg.radius-.002)+direction*(.004/incidence)
            mocap = p.m.body("contact_probe").mocapid[0]
            tangent=np.cross(normal,np.array([0.,0.,1.]))
            if np.linalg.norm(tangent)<1e-8:
                tangent=np.array([1.,0.,0.])
            tangent/=np.linalg.norm(tangent)
            rotation=np.column_stack((tangent,np.cross(normal,tangent),normal))
            mujoco.mju_mat2Quat(p.d.mocap_quat[mocap],rotation.ravel())
            peak_force = 0.0
            count = 0
            max_pen = 0.0
            for step in range(round(.02/cfg.timestep)):
                fraction = step/max(round(.02/cfg.timestep)-1,1)
                p.d.mocap_pos[mocap] = initial-direction*(.004/incidence)*fraction
                p.step()
                force = p.contacts()
                peak_force=max(peak_force,float(np.linalg.norm(force["probe_force_N"])))
                count=max(count,force["film_probe_contacts"])
                max_pen=max(max_pen,force["penetration_m"])
            mujoco.mj_forward(p.m,p.d)
            remote=np.linalg.norm(vertices-center,axis=1)>.04
            transmission=float(np.max(np.linalg.norm(p.d.flexvert_xpos[remote]-control[remote],axis=1)))
            rows.append(dict(shape=shape,face=label,passed=bool(count>0 and peak_force>.001),
                             peak_force_N=peak_force,contacts=count,peak_penetration_m=max_pen,
                             remote_displacement_against_control_m=transmission,
                             warnings=p.d.warning.number.tolist()))
    return dict(passed=all(r["passed"] for r in rows),cases=rows)


def pull(cfg, output):
    p=Parcel(cfg)
    p.save(output)
    simulate(p,.2)
    vertices=p.d.flexvert_xpos.copy()
    upper=np.unique(p.envelope["faces"][p.envelope["panels"]==0])
    # The force is applied to actual film nodes, with no constraint to payload.
    selected=upper[np.argsort(np.linalg.norm(vertices[upper,:2]-np.array(cfg.center),axis=1))[:4]]
    bodies=p.m.flex_vertbodyid[selected]
    z0=float(p.com()[2])
    def load(parcel,t):
        parcel.d.xfrc_applied[:]=0
        parcel.d.xfrc_applied[bodies,2]=3.8*min(t/.1,1)/len(bodies)
    trace,peak=simulate(p,.3,load)
    lift=float(p.com()[2]-z0)
    final_support=float(np.mean([row["table_support_N"] for row in trace[-5:]]))
    (output/'trace.json').write_text(json.dumps(trace,default=serial,indent=2))
    return dict(passed=bool(lift>.002 and final_support<.5 and peak<=.001),
                diagnostic=True,loading="external ramped 3.8N on four real film nodes",
                captured_by_friction=False,vertices=selected.tolist(),payload_lift_m=lift,
                late_table_support_N=final_support,peak_penetration_m=peak)


def compliance(cfg, output):
    output.mkdir(parents=True)
    rows=[]
    for k in (.8,.08,.008):
        p=Parcel(replace(cfg,mode='bendable',joint_stiffness=k))
        # Free payload relaxation diagnoses the two actual relative joints.
        p.m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT
        p.m.opt.gravity[:]=0
        j=p.m.joint('bend_1').id
        q=p.m.jnt_qposadr[j]
        p.d.qpos[q]=.15
        mujoco.mj_forward(p.m,p.d)
        samples=[]
        for step in range(round(.4/cfg.timestep)):
            p.step()
            if step%max(1,round(.002/cfg.timestep))==0:
                samples.append((float(p.d.time),float(p.d.qpos[q]),float(p.d.qvel[p.m.jnt_dofadr[j]])))
        data=np.asarray(samples)
        crossings=np.flatnonzero((data[:-1,1]>0)&(data[1:,1]<=0))
        frequency=float(1/np.mean(np.diff(data[crossings,0]))) if len(crossings)>1 else None
        rows.append(dict(stiffness_Nm_rad=k,mass_kg=float(p.m.body_mass[p.payload_bodies].sum()),
                         frequency_Hz=frequency,angle_range_rad=[float(data[:,1].min()),float(data[:,1].max())],
                         late_amplitude_rad=float(np.max(abs(data[-20:,1])))))
        np.savez_compressed(output/f'bend_{k}.npz',samples=data)
    p=Parcel(replace(cfg,mode='compressible'))
    cap=p.m.joint('compression_0').id
    q,v=p.m.jnt_qposadr[cap],p.m.jnt_dofadr[cap]
    p.m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT
    p.m.opt.gravity[:]=0
    p.d.qpos[q]=.004
    mujoco.mj_step1(p.m,p.d)
    p.apply_internal()
    expected=float(compression_force(.004,0,p.cfg))
    actual=float(-p.d.qfrc_passive[v])
    trace,_=simulate(p,.1)
    force_error=abs(actual-expected)
    return dict(passed=bool(force_error<1e-9 and all(abs(r['mass_kg']-cfg.payload_mass)<1e-10 for r in rows)),
                bending=rows,compression=dict(delta_m=.004,expected_N=expected,actual_N=actual,
                                              force_error_N=force_error,final_delta_m=float(p.d.qpos[q])),
                limitations=['joint relaxation isolates payload from envelope contacts',
                             'natural frequency unresolved when fewer than two crossings are recorded'])


def momentum(parcel):
    m,d=parcel.m,parcel.d
    linear=np.zeros(3);angular=np.zeros(3)
    jacp=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv))
    for body in range(1,m.nbody):
        if m.body_mass[body]<=0:continue
        mujoco.mj_jacBodyCom(m,d,jacp,jacr,body)
        velocity=jacp@d.qvel;omega=jacr@d.qvel
        rotation=d.ximat[body].reshape(3,3)
        linear+=m.body_mass[body]*velocity
        angular+=np.cross(d.xipos[body],m.body_mass[body]*velocity)+rotation@(m.body_inertia[body]*(rotation.T@omega))
    return linear,angular


def conservation(cfg,output):
    p=Parcel(replace(cfg,gravity=(0,0,0)))
    p.save(output)
    p.m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT
    p.d.qvel[p.film_dofs]=np.array([.03,.02,.01])
    p.d.qvel[p.m.jnt_dofadr[p.m.joint('payload_free').id]:][:3]=[.03,.02,.01]
    mujoco.mj_forward(p.m,p.d)
    before=momentum(p)
    e0=float(p.d.energy.sum()+p.bending_energy())
    trace,_=simulate(p,.04)
    mujoco.mj_forward(p.m,p.d)
    after=momentum(p)
    e1=float(p.d.energy.sum()+p.bending_energy())
    linear=float(np.linalg.norm(after[0]-before[0])/max(np.linalg.norm(before[0]),1e-8))
    angular=float(np.linalg.norm(after[1]-before[1])/max(np.linalg.norm(before[1]),1e-8))
    return dict(passed=bool(linear<=.01 and angular<=.01 and e1<=e0*1.01),
                initial_linear_momentum=before[0],final_linear_momentum=after[0],
                relative_linear_drift=linear,relative_angular_drift=angular,
                initial_energy_J=e0,final_energy_J=e1,warnings=p.d.warning.number)


CASES=dict(audit=audit,support=support,contacts=contacts,pull=pull,compliance=compliance,conservation=conservation)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--case',choices=[*CASES,'all'],default='all')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    cfg=Config(**json.loads(args.config.read_text())) if args.config else Config()
    args.output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),ROOT/'bag_sim/physical_parcel.py',ROOT/'bag_sim/bag_model.py',ROOT/'bag_sim/wrap.py']
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (args.output/'sources.json').write_text(json.dumps(sources,indent=2))
    (args.output/'source').mkdir()
    for path in source_paths:
        (args.output/'source'/path.name).write_bytes(path.read_bytes())
    selected=CASES if args.case=='all' else {args.case:CASES[args.case]}
    results={}
    for name,function in selected.items():
        start=time.perf_counter()
        result=function(cfg,args.output/name)
        result['wall_s']=time.perf_counter()-start
        results[name]=result
        (args.output/f'{name}.json').write_text(json.dumps(result,default=serial,indent=2))
        print(json.dumps(dict(case=name,**result),default=serial),flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(machine=platform.node(),results=results),default=serial,indent=2))
    if not all(r['passed'] for r in results.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
