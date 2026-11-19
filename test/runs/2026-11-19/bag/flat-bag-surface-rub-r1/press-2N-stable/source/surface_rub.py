"""Normal-loaded surface rubbing on the unchanged, prepared physical parcel."""
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
sys.path[:0] = [str(ROOT), str(ROOT / 'experience')]
import physical_parcel_yam as backend
from physical_parcel_checks import film_box_geometry, film_payload_geometry
from rub_contact_e1 import film_constraint_reaction, tool_contacts, write_json
from bag_sim.bag_model import _topology, dihedrals
from bag_sim.physical_parcel import vec
from yam_platform.geometry import fingertip_xml

PHASES = [('hover', .02), ('press', 1.0), ('loaded', .05), ('rub', .40), ('unload', .60), ('relax', .10)]


def surface_height(points, faces, xy):
    triangles = points[faces]
    a, u, v = triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    determinant = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    delta = xy-a[:, :2]
    divisor = np.where(abs(determinant)>1e-20, determinant, 1.)
    b = (delta[:, 0]*v[:, 1]-delta[:, 1]*v[:, 0])/divisor
    c = (u[:, 0]*delta[:, 1]-u[:, 1]*delta[:, 0])/divisor
    inside = (b>=0) & (c>=0) & (b+c<=1) & (abs(determinant)>1e-20)
    if not inside.any():
        raise ValueError('No real top triangle beneath the prescribed pad')
    return float(np.max((a[:, 2]+b*u[:, 2]+c*v[:, 2])[inside]))


def build(cfg, friction):
    parcel, xml = backend.parcel_world(cfg)
    base = parcel.compile_model(xml)
    data = mujoco.MjData(base)
    parcel.m = base
    backend.initialize_state(parcel, base, data, cfg)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    center = np.asarray(parcel.cfg.center)
    top = parcel.envelope['faces'][parcel.envelope['panels']==0]
    positions = np.array([[* (center+[-.03, 0]), 0.], [* (center+[.03, 0]), 0.]])
    for index in range(2):
        positions[index, 2] = surface_height(data.flexvert_xpos, top, positions[index, :2])+.020+parcel.cfg.radius+.0005
    root = ET.fromstring(xml)
    for name, position in zip(('rub_left', 'rub_right'), positions):
        pad = ET.fromstring(fingertip_xml(name, 0, position))
        for geom in pad.findall('geom'):
            geom.set('conaffinity', '2')  # Material only; exclude direct support contact.
            geom.set('priority', '1')
            geom.set('friction', f'{friction} 0 0')
            geom.set('solref', f'{parcel.cfg.contact_time} 1')
            geom.set('solimp', '.999 .999 .001')
            if geom.get('type') == 'box':
                half = np.fromstring(geom.attrib.pop('size'), sep=' ')
                corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                ET.SubElement(root.find('asset'), 'mesh', name=name+'_mesh', vertex=vec(corners*half))
                geom.set('type', 'mesh'); geom.set('mesh', name+'_mesh')
        root.find('worldbody').append(pad)
    parcel.xml = ET.tostring(root, encoding='unicode')
    parcel.m = parcel.compile_model(parcel.xml)
    parcel.d = mujoco.MjData(parcel.m)
    parcel.d.qpos[:len(initial_qpos)] = initial_qpos
    parcel.d.qvel[:len(initial_qvel)] = initial_qvel
    parcel.film_dofs = parcel.node_dofs(parcel.m)
    parcel.forward()
    bodies = np.array([parcel.m.body(name).id for name in ('rub_left','rub_right')])
    joints = parcel.m.body_jntadr[bodies, None]+np.arange(3)
    qids, vids = parcel.m.jnt_qposadr[joints], parcel.m.jnt_dofadr[joints]
    groups = [set(np.flatnonzero(parcel.m.geom_bodyid==body).tolist()) for body in bodies]
    return parcel, positions, qids, vids, bodies, groups


def measurement_setup(p):
    reference = p.envelope['reference']
    center = p.cfg.center
    top = np.unique(p.envelope['faces'][p.envelope['panels']==0])
    patch = top[(abs(reference[top, 0]-center[0])<=.071) & (abs(reference[top, 1]-center[1])<=.036)]
    rows = []
    for y in np.unique(np.round(reference[patch, 1], 12)):
        ids = patch[np.isclose(reference[patch, 1], y)]
        if len(ids)>=3:
            rows.append(ids[np.argsort(reference[ids, 0])])
    if not rows:
        raise ValueError('The unchanged mesh has no measurable material row here')
    _, _, _, _, adjacent, _, _ = _topology(reference, p.envelope['faces'])
    midpoint = reference[p.hinges].mean(1)
    hinges = np.flatnonzero((p.envelope['panels'][adjacent]==0).all(1)
        & (abs(midpoint[:,0]-center[0])<=.071) & (abs(midpoint[:,1]-center[1])<=.036))
    faces = p.envelope['faces'][(p.envelope['panels']==0)
        & np.isin(p.envelope['faces'], patch).all(1)]
    rest = reference[faces, :2]
    inverse = np.linalg.inv(np.stack((rest[:,1]-rest[:,0], rest[:,2]-rest[:,0]), axis=2))
    return rows, patch, hinges, faces, inverse


def metrics(p, setup, initial, angles0, com0):
    rows, patch, hinges, faces, inverse = setup
    points = p.d.flexvert_xpos
    heights, lengths = [], []
    for ids in rows:
        fraction = (p.envelope['reference'][ids,0]-p.envelope['reference'][ids[0],0])/np.ptp(p.envelope['reference'][ids,0])
        line = points[ids[0]]+fraction[:,None]*(points[ids[-1]]-points[ids[0]])
        heights.append(float(np.max(points[ids,2]-line[:,2])))
        lengths.append(float(np.linalg.norm(np.diff(points[ids],axis=0),axis=1).sum()-np.linalg.norm(points[ids[-1]]-points[ids[0]])))
    angles, _ = dihedrals(points, p.hinges, gradient=False)
    angle_delta = np.arctan2(np.sin(angles[hinges]-angles0[hinges]),np.cos(angles[hinges]-angles0[hinges]))
    triangles = points[faces]
    deformation = np.stack((triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]),axis=2) @ inverse
    strain = np.sqrt(np.maximum(np.linalg.eigvalsh(np.swapaxes(deformation,1,2) @ deformation),0))-1
    relative = points[patch]-initial[patch]-(p.com()-com0)
    return dict(row_crest_m=max(heights),row_excess_length_m=max(lengths),
                max_dihedral_change_deg=float(np.max(abs(angle_delta),initial=0)*180/np.pi),
                min_principal_material_strain=float(strain.min()),
                local_material_dx_range_m=[float(relative[:,0].min()),float(relative[:,0].max())],
                payload_com_displacement_m=(p.com()-com0).copy())


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg = backend.Config(**json.loads(args.config.read_text()))
    p, origins, qids, vids, bodies, groups = build(cfg,args.friction)
    m,d = p.m,p.d
    if args.dt:
        m.opt.timestep = args.dt
    dt = float(m.opt.timestep)
    if m.neq:
        raise RuntimeError('Surface diagnostic must not introduce material constraints')
    setup = measurement_setup(p)
    initial = d.flexvert_xpos.copy(); com0=p.com().copy()
    angles0,_ = dihedrals(initial,p.hinges,gradient=False)
    initial_metrics=metrics(p,setup,initial,angles0,com0)
    phases=[(name,args.press_seconds if name=='press' else duration) for name,duration in PHASES]
    write_json(args.output/'config.json',dict(parcel=asdict(p.cfg),source_config=str(args.config),
        effective_dt_s=dt,friction=args.friction,distance_per_pad_m=args.distance_mm/1000,
        target_force_each_N=args.press_force,normal_gain_m_per_N_s=args.normal_gain,
        phases=phases,pad_origins_m=origins,
        initial_metrics=initial_metrics,rows=[ids.tolist() for ids in setup[0]],
        numerical_contact_radius_m=p.cfg.radius,artificial_attachments=m.neq,
        prescribed_boundary='shared translation-stage fingertips; actual qvel supplied to contact solver',
        machine=platform.node(),mujoco=mujoco.__version__))
    (args.output/'model.xml').write_text(p.xml)
    mujoco.mj_saveModel(m,str(args.output/'model.mjb'),None)
    forces = np.zeros((2,3));z=origins[:,2].copy();previous=origins.copy()
    rows=[];states=[];velocities=[];times=[];nodes=[]
    max_penetration=0.;mapping_error=0.;failure=None;pressure_gate=False
    phases=phases[:1] if args.smoke else phases
    stride=max(1,round(.002/dt));step_id=0
    with (args.output/'contacts.jsonl').open('w') as log:
        for phase,duration in phases:
            retract_start=z.copy()
            steps=2 if args.smoke else round(duration/dt)
            for step in range(steps):
                fraction=(step+1)/steps
                positions=origins.copy()
                if phase in ('press','loaded','rub'):
                    target=args.press_force*min(1.,(step+1)*dt/.1) if phase=='press' else args.press_force
                    z += np.clip(args.normal_gain*(forces[:,2]-target),-.02,.02)*dt
                elif phase=='unload':
                    z = retract_start + fraction*(origins[:,2]+.001-retract_start)
                positions[:,2]=z
                distance=args.distance_mm/1000
                if phase=='rub':
                    displacement=distance*(3*fraction**2-2*fraction**3)
                    positions[:,0]+=np.array([1.,-1.])*displacement
                elif phase in ('unload','relax'):
                    positions[:,0]+=np.array([1.,-1.])*distance
                d.qpos[qids]=positions-origins
                d.qvel[vids]=(positions-previous)/dt
                previous=positions.copy()
                sampled=step_id%stride==0 or step==steps-1
                if sampled:
                    state=d.qpos.copy();velocity=d.qvel.copy()
                evaluated=float(d.time)
                p.step()
                contacts=[];active=[]
                for pad,group in enumerate(groups):
                    _,all_records=tool_contacts(p,group)
                    records=[r for r in all_records if 0 in r['flex']]
                    forces[pad]=np.sum([r['force_on_tip_N'] for r in records],axis=0) if records else 0.
                    contacts.append(records);active.append(sum(r['efc_address']>=0 for r in records))
                penetration=max(0.,-float(d.contact.dist.min())) if d.ncon else 0.
                max_penetration=max(max_penetration,penetration)
                if sampled:
                    for pad,records in enumerate(contacts):
                        independent=film_constraint_reaction(p,records)
                        mapping_error=max(mapping_error,float(np.linalg.norm(independent-forces[pad])))
                    geometry=film_payload_geometry(m,d,p.envelope['faces'])
                    support=film_box_geometry(m,d,p.envelope['faces'],m.geom('table').id)
                    measure=metrics(p,setup,initial,angles0,com0)
                    row=dict(t_eval=evaluated,phase=phase,force_on_pad_N=forces.copy(),
                        pad_pos_m=d.xpos[bodies].copy(),pad_velocity_m_s=velocity[vids],
                        active_each=active,friction_force_magnitude_each_N=[sum(np.linalg.norm(r['force_local_N'][1:3]) for r in c) for c in contacts],
                        penetration_m=penetration,payload_geometry=geometry,support_geometry=support,
                        solver_niter=int(d.solver_niter.max()),warnings=int(d.warning.number.sum()),**measure)
                    rows.append(row);states.append(state);velocities.append(velocity);times.append(evaluated);nodes.append(d.flexvert_xpos.copy())
                    log.write(json.dumps(dict(t_eval=evaluated,phase=phase,contacts=contacts),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
                    if max(geometry['geometric_depth_upper_m'],support['geometric_depth_upper_m'])>.001:
                        failure='independent geometric penetration exceeds 1 mm'
                if d.warning.number.sum() or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    failure='warning or nonfinite state'
                if penetration>.001:
                    failure='native contact penetration exceeds 1 mm'
                if np.max(origins[:,2]-z)>.020:
                    failure='normal controller exceeded 20 mm travel without establishing load'
                step_id+=1
                if failure:break
            print(json.dumps(dict(phase=phase,time_s=float(d.time),force_z_each_N=forces[:,2].tolist(),failure=failure)),flush=True)
            if phase=='loaded':
                recent=[r for r in rows if r['phase']=='loaded' and r['t_eval']>=d.time-.03]
                mean=np.mean([r['force_on_pad_N'][:,2] for r in recent],axis=0)
                pressure_gate=bool(np.all(abs(mean-args.press_force)<=.2*args.press_force)
                    and all(min(r['active_each'])>0 for r in recent) and mapping_error<1e-6)
                if not pressure_gate:failure=f'normal-load gate failed: mean each {mean.tolist()} N'
            if failure:break
    write_json(args.output/'trace.json',rows)
    np.savez_compressed(args.output/'trajectory.npz',qpos=states,qvel=velocities,time=times,film=nodes,
        initial_film=initial,final_qpos=d.qpos,final_qvel=d.qvel,reference=p.envelope['reference'],faces=p.envelope['faces'])
    healthy=bool(not d.warning.number.sum() and np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        and max_penetration<=.001 and all(max(r['payload_geometry']['geometric_depth_upper_m'],
        r['support_geometry']['geometric_depth_upper_m'])<=.001 for r in rows))
    summary=dict(pressure_gate=pressure_gate,entry_only=args.smoke,failure=failure,
        numerically_valid=healthy,protocol_completed=failure is None,peak_penetration_m=max_penetration,
        final_contact_force_each_N=np.linalg.norm(forces,axis=1),
        reaction_mapping_error_N=mapping_error,initial_metrics=initial_metrics,
        peak_row_crest_m=max(r['row_crest_m'] for r in rows),
        peak_row_excess_length_m=max(r['row_excess_length_m'] for r in rows),
        peak_dihedral_change_deg=max(r['max_dihedral_change_deg'] for r in rows),
        simulated_s=float(d.time),wall_s=time.perf_counter()-started,completed_phases=list(dict.fromkeys(r['phase'] for r in rows)))
    write_json(args.output/'summary.json',summary)
    sources=[Path(__file__),Path(backend.__file__),ROOT/'bag_sim/physical_parcel.py',ROOT/'bag_sim/wrap.py',ROOT/'bag_sim/bag_model.py',ROOT/'experience/yam_platform/geometry.py',ROOT/'test/bag_model/rub_contact_e1.py']
    (args.output/'source').mkdir()
    for source in sources:(args.output/'source'/source.name).write_bytes(source.read_bytes())
    write_json(args.output/'sources.json',{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
    print(json.dumps(summary,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),flush=True)
    return failure is None


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distance-mm',type=float,default=10.)
    parser.add_argument('--friction',type=float,default=.8)
    parser.add_argument('--press-force',type=float,default=.5)
    parser.add_argument('--press-seconds',type=float,default=1.)
    parser.add_argument('--normal-gain',type=float,default=.05)
    parser.add_argument('--dt',type=float)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not 0<=args.distance_mm<=15 or not 0<=args.friction<=1 or not .01<=args.press_force<=2:
        parser.error('distance 0–15 mm, friction 0–1, force .01–2 N required')
    if args.dt is not None and not 0<args.dt<=.0001:
        parser.error('dt must be positive and no larger than .1 ms')
    if not .1<=args.press_seconds<=3 or not .001<=args.normal_gain<=.05:
        parser.error('press duration .1–3 s and gain .001–.05 m/(N s) required')
    if not run(args):raise SystemExit(1)


if __name__=='__main__':
    main()
