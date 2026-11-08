"""Fallback contact gate using the same tet mesh and float64 NH force law.

MuJoCo handles rigid/flex contact; tetrahedral forces are applied explicitly.
The membrane uses MuJoCo's native stretch-only elastic surface (StVK), so its
contact-transfer gate is comparable but its displacement is not a NH reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

from metrics import pressure_weights, surface_height, tet_response

HERE = Path(__file__).resolve().parent


def run(case, dt=.0002, seconds=2., tag='mj', cells_source='guided_base_plate.npz', drag=2.):
    start = time.perf_counter()
    with np.load(HERE / 'results' / cells_source) as g:
        rest, tets, top_faces, fixed = [g[k].copy() for k in ('initial', 'tets', 'top_faces', 'fixed_nodes')]
    n = len(rest)
    faces = []
    unique = {}
    for tet in tets:
        for ids in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
            face = tuple(tet[list(ids)]); key = tuple(sorted(face))
            unique[key] = face if key not in unique else None
    faces = np.array([f for f in unique.values() if f is not None], int)
    pts = rest[tets]
    inv = np.linalg.inv(np.stack([pts[:, i]-pts[:, 0] for i in (1, 2, 3)],axis=2))
    mats = np.tile([1500., 1000., 1.], (len(tets), 1))
    volumes = 1/(6*np.linalg.det(inv))
    masses = np.zeros(n)
    np.add.at(masses,tets.ravel(),np.repeat(volumes / volumes.sum() * .3 / 4, 4))
    membrane = case == 'membrane'
    direct = case.startswith('direct_')
    kind = case.removeprefix('direct_')
    half = (.12,.10,.01) if kind in ('plate','membrane') else (.01,.01,.01)
    xy = (.0125,.0125) if kind=='finger_off' else (0.,0.)
    load = 19.62 if kind in ('plate','membrane') else 2.
    weights = pressure_weights(rest,top_faces,half,xy)
    cloth_faces = np.empty((0,3),int)
    if membrane:
        cloth_q=np.array([(x,y,.059) for x in np.linspace(-.12,.12,13) for y in np.linspace(-.10,.10,11)])
        cloth_faces=np.array([(a*11+b+i,a*11+b+j,a*11+b+k) for a in range(12) for b in range(10)
                              for i,j,k in [(0,11,1),(1,11,12)]],int)+n
        rest=np.r_[rest,cloth_q]
        # Lumped cloth mass from triangle area; preserves total mass 12 g.
        cm=np.zeros(len(rest)-n)
        tri=rest[cloth_faces]
        areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        np.add.at(cm,(cloth_faces-n).ravel(),np.repeat(.012*areas/areas.sum()/3,3))
        masses=np.r_[masses,cm]
    nodes=[]; fixed_set=set(fixed)
    for i,p in enumerate(rest):
        joints='' if i in fixed_set else ''.join(f'<joint name="n{i}_{k}" type="slide" axis="{int(k==0)} {int(k==1)} {int(k==2)}" damping="{drag*masses[i]}"/>' for k in range(3))
        nodes.append(f'<body name="n{i}" pos="{p[0]} {p[1]} {p[2]}">{joints}<geom type="sphere" size=".001" mass="{masses[i]}" contype="0" conaffinity="0"/></body>')
    tool=''
    if not direct and not membrane:
        tool=(f'<body name="tool" pos="{xy[0]} {xy[1]} {.055+half[2]}"><joint name="tool_z" type="slide" axis="0 0 1"/>'
              f'<geom type="box" size="{half[0]} {half[1]} {half[2]}" mass="{load/9.81}" contype="1" conaffinity="2" '
              'friction="0 0 0" condim="1" solref=".001 1" solimp=".99 .999 .0001"/></body>')
    def flex(name,fs,ctype,affinity,elastic=''):
        used,remap=np.unique(fs,return_inverse=True)
        return (f'<flex name="{name}" dim="2" radius=".004" body="'+ ' '.join(f'n{i}' for i in used)+'" vertex="'+
            ' '.join(['0']*(3*len(used)))+'" element="'+' '.join(map(str,remap.ravel()))+ '"><contact selfcollide="none" '
            f'contype="{ctype}" conaffinity="{affinity}" condim="1" friction="0 0 0" solref=".001 1" solimp=".99 .999 .0001"/>'
            '<edge damping="1e-12"/>'+elastic+'</flex>')
    skin=flex('slab_skin',faces,2,1)
    if membrane:
        skin+=flex('membrane',cloth_faces,1,2,'<elasticity young="11666666.6667" poisson="0.1666666667" thickness=".001" elastic2d="stretch" damping=".001"/>')
    integrator='discrete' if membrane else 'Euler'
    xml=(f'<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="{integrator}" solver="CG" iterations="100" tolerance="1e-9" jacobian="sparse"/>'
         '<size memory="500M"/><worldbody>'+''.join(nodes)+tool+'</worldbody><deformable>'+skin+'</deformable></mujoco>')
    model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
    free=np.array([i for i in range(len(rest)) if i not in fixed_set])
    qids=np.array([[model.jnt_qposadr[model.joint(f'n{i}_{k}').id] for k in range(3)] for i in free])
    dids=np.array([[model.jnt_dofadr[model.joint(f'n{i}_{k}').id] for k in range(3)] for i in free])
    ext=np.zeros_like(rest)
    if direct: ext[:n,2]=-load*weights
    if membrane: ext[:,2]=-load*pressure_weights(rest,cloth_faces,half,xy)
    tool_dof=None if (direct or membrane) else model.jnt_dofadr[model.joint('tool_z').id]
    e0=tet_response(rest,tets,inv,mats)[0]
    rows=[]; frames=[]; failure=None
    every=round(.05/dt)
    previous_warnings=data.warning.number.copy()
    for step in range(round(seconds/dt)):
        q=rest.copy(); q[free]+=data.qpos[qids]
        energy,ef,j=tet_response(q,tets,inv,mats)
        f=min(1.,(step+1)*dt/.2); f=f*f*(3-2*f)
        data.qfrc_applied[:]=0
        data.qfrc_applied[dids]=(ef+f*ext)[free]
        if tool_dof is not None: data.qfrc_applied[tool_dof]=-load*f
        mujoco.mj_step(model,data)
        if np.any(data.warning.number>previous_warnings):
            failure=f'MuJoCo warning at {data.time:.6f} s'; break
        if (step+1)%every: continue
        mujoco.mj_forward(model,data)
        q=rest.copy(); q[free]+=data.qpos[qids]
        energy,ef,j=tet_response(q,tets,inv,mats)
        cf=np.zeros_like(q); cf[free]=data.qfrc_constraint[dids]
        residual=ef+cf+f*ext
        active=free[free<n]
        speed=np.linalg.norm(data.qvel[dids],axis=1)
        row=dict(t=round(float(data.time),6),support_N=-float(ef[fixed,2].sum()),
            sink_mm=float(weights@(.05-q[:n,2])*1e3),
            slab_contact_N=-float(cf[:n,2].sum()), active_residual_L1_N=float(np.linalg.norm(residual[active],axis=1).sum()),
            max_speed_mm_s=float(speed.max()*1e3),min_J=float(j.min()),inverted_tets=int(np.sum(j<=0)),energy_J=energy-e0,
            contacts=int(data.ncon))
        if tool_dof is not None:
            z=float(data.xpos[model.body('tool').id,2])-half[2]
            samples=np.array([(x,y) for x in np.linspace(xy[0]-.99*half[0],xy[0]+.99*half[0],7)
                                   for y in np.linspace(xy[1]-.99*half[1],xy[1]+.99*half[1],7)])
            row.update(tool_reaction_N=float(data.qfrc_constraint[tool_dof]),tool_z_mm=(z+half[2])*1e3,
                       tool_speed_mm_s=abs(float(data.qvel[tool_dof]))*1e3,
                       min_interface_clearance_mm=float(np.nanmin(z-surface_height(q,top_faces,samples))*1e3))
        elif membrane:
            clearance=q[n:,2]-surface_height(q,top_faces,q[n:,:2])
            if not np.isfinite(clearance).any():
                failure=f'membrane left measurable top surface at {data.time:.6f} s';break
            row['min_interface_clearance_mm']=float(np.nanmin(clearance)*1e3)
        rows.append(row); frames.append(q.astype(np.float32))
        if row['inverted_tets']:
            failure=f"{row['inverted_tets']} inverted tets at {row['t']} s";break
    late=[r for r in rows if r['t']>(rows[-1]['t'] if rows else seconds)-.5]
    summary={k:float(np.mean([r[k] for r in late])) for k in late[0]} if late else {}
    summary['support_error_pct']=abs(summary.get('support_N',0)/load-1)*100
    summary['material_residual_pct']=summary.get('active_residual_L1_N',float('inf'))/load*100
    summary['sink_drift_last_half_second_mm']=late[-1]['sink_mm']-late[0]['sink_mm'] if late else None
    summary['min_J_all_samples']=min((r['min_J'] for r in rows),default=0.)
    summary['inverted_tets_all_samples']=max((r['inverted_tets'] for r in rows),default=0)
    summary['minimum_interface_clearance_all_samples_mm']=min((r.get('min_interface_clearance_mm',float('inf')) for r in rows),default=float('inf'))
    if not np.isfinite(summary['minimum_interface_clearance_all_samples_mm']): summary['minimum_interface_clearance_all_samples_mm']=None
    checks=dict(finished=failure is None and bool(late) and rows[-1]['t']>=seconds-.001,
        support_balance=summary['support_error_pct']<=5,material_equilibrium=summary['material_residual_pct']<=5,
        no_inversion=bool(rows) and summary['inverted_tets_all_samples']==0,
        settled=bool(late) and abs(summary['sink_drift_last_half_second_mm'])<=.05 and summary['max_speed_mm_s']<=1.)
    if tool_dof is not None:
        summary['tool_force_error_pct']=abs(summary['tool_reaction_N']/load-1)*100
        checks['tool_force_balance']=summary['tool_force_error_pct']<=5
    if not direct: checks['no_geometric_penetration']=summary['minimum_interface_clearance_all_samples_mm']>=-.05
    result=dict(config=dict(case=case,dt=dt,seconds=seconds,tag=tag,cells_source=cells_source,particle_drag_per_s=drag,membrane_native_stvk=membrane),
        environment=dict(mujoco=mujoco.__version__,executable=sys.executable),load_N=load,actual_dt_s=dt,
        failure=failure,summary=summary,checks=checks,passed=all(checks.values()),trace=rows,wall_s=round(time.perf_counter()-start,3))
    path=HERE/'results'/f'{tag}_{case}'
    path.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(path.with_suffix('.npz'),frames=np.array(frames),top_faces=top_faces,tets=tets,fixed_nodes=fixed,initial=rest,cloth_faces=cloth_faces)
    print(json.dumps(dict(name=path.stem,summary=summary,checks=checks,wall_s=result['wall_s'])),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',nargs='+',default=['plate','finger','finger_off','direct_plate','membrane'])
    ap.add_argument('--dt',type=float,default=.0002)
    ap.add_argument('--seconds',type=float,default=2.)
    ap.add_argument('--tag',default='mj')
    ap.add_argument('--drag',type=float,default=2.)
    a=ap.parse_args()
    for case in a.cases: run(case,a.dt,a.seconds,a.tag,drag=a.drag)
