"""Read-only Phase 0 audit of the saved failed YAM run; no integration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"experience")]
from yam_platform.protocol import tips
from types import SimpleNamespace
from rub_contact_e1 import write_json


def audit(trial,output):
    output.mkdir(parents=True,exist_ok=False)
    result=json.loads((trial/"result.json").read_text())
    trace=json.loads((trial/"trace.json").read_text())
    data=np.load(trial/"trajectory.npz")
    m=mujoco.MjModel.from_binary_path(str(trial/"model.mjb"))
    d=mujoco.MjData(m)
    pad_ids=[m.geom(f"{s}_pad").id for s in ("left","right")]
    robot=SimpleNamespace(sim=SimpleNamespace(m=m,d=d),pad_geom=pad_ids)
    geometry=[]
    initial_axis=None
    pad_bodies=[m.body(f"{side}_pad").id for side in ("left","right")]
    groups=[{g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])} for body in pad_bodies]
    for index,qpos in enumerate(data["qpos"]):
        d.qpos[:]=qpos
        mujoco.mj_kinematics(m,d)
        mujoco.mj_flex(m,d)
        points=d.flexvert_xpos.copy()
        edges=tips(robot)
        rotations=d.xmat[pad_bodies].reshape(2,3,3)
        if initial_axis is None:
            direction=d.geom_xpos[pad_ids[1]]-d.geom_xpos[pad_ids[0]]
            initial_axis=1. if direction @ rotations[0,:,1]>0 else -1.
        closing=initial_axis*rotations[0,:,1]
        intervals=[]
        for group in groups:
            bounds=[]
            for gid in group:
                rotation=d.geom_xmat[gid].reshape(3,3)
                mid=float(d.geom_xpos[gid]@closing)
                if m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=m.geom_dataid[gid]
                    adr,num=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                    points_world=m.mesh_vert[adr:adr+num]@rotation.T+d.geom_xpos[gid]
                    projection=points_world@closing
                    bounds.append([float(projection.min()),float(projection.max())])
                elif m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_CAPSULE:
                    reach=m.geom_size[gid,0]+m.geom_size[gid,1]*abs(rotation[:,2]@closing)
                    bounds.append([mid-reach,mid+reach])
            intervals.append([min(x[0] for x in bounds),max(x[1] for x in bounds)])
        facing_gap=intervals[1][0]-intervals[0][1]
        # Both cores are exact rectangular boxes stored as eight-vertex meshes.
        # SAT checks all face normals and cross-edge axes, independently of masks.
        core_rot=d.geom_xmat[pad_ids].reshape(2,3,3)
        delta=d.geom_xpos[pad_ids[1]]-d.geom_xpos[pad_ids[0]]
        axes=[*core_rot[0].T,*core_rot[1].T]
        axes += [np.cross(a,b) for a in core_rot[0].T for b in core_rot[1].T]
        overlap=True
        for axis in axes:
            if np.linalg.norm(axis)<1e-9:
                continue
            r0=np.abs(core_rot[0].T@axis)@m.geom_size[pad_ids[0]]
            r1=np.abs(core_rot[1].T@axis)@m.geom_size[pad_ids[1]]
            if abs(delta@axis)>r0+r1:
                overlap=False
                break
        faces=m.flex_elem.reshape(-1,3)
        triangles=points[faces]
        # Nearest point on each triangle, considering plane projection and all
        # edges: this is a geometric gap at the distal-tip centers, not a contact.
        gaps=[]
        for tip in edges:
            a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
            u,v=b-a,c-a
            n=np.cross(u,v)
            nn=np.einsum('ij,ij->i',n,n)
            signed=np.einsum('ij,ij->i',tip-a,n)/np.maximum(nn,1e-30)
            projection=tip-signed[:,None]*n
            dotuu=np.einsum('ij,ij->i',u,u); dotuv=np.einsum('ij,ij->i',u,v)
            dotvv=np.einsum('ij,ij->i',v,v)
            w=projection-a
            dotwu=np.einsum('ij,ij->i',w,u); dotwv=np.einsum('ij,ij->i',w,v)
            det=np.maximum(dotuu*dotvv-dotuv**2,1e-30)
            s=(dotvv*dotwu-dotuv*dotwv)/det; t=(dotuu*dotwv-dotuv*dotwu)/det
            inside=(s>=0)&(t>=0)&(s+t<=1)
            best=np.min(np.where(inside,abs(signed)*np.sqrt(nn),np.inf))
            for first,last in ((a,b),(b,c),(c,a)):
                e=last-first
                ratio=np.clip(np.einsum('ij,ij->i',tip-first,e)/np.maximum(np.einsum('ij,ij->i',e,e),1e-30),0,1)
                best=min(best,float(np.linalg.norm(tip-first-ratio[:,None]*e,axis=1).min()))
            gaps.append(float(best-m.flex_radius[0]))
        geometry.append(dict(time_s=float(data["time"][index]),phase=trace[index]["phase"],
            tip_edges_m=edges.copy(),facing_surface_gap_m=float(facing_gap),core_obb_overlap=bool(overlap),pad_pos_m=d.geom_xpos[pad_ids].copy(),pad_rotation=d.geom_xmat[pad_ids].copy(),
            tip_center_surface_gap_m=gaps,actual_gap_m=float(sum(d.qpos[m.joint(f'{s}_finger').qposadr[0]] for s in ('left','right')))))
    pos=np.array([row["tip_edges_m"] for row in geometry])
    velocity=np.gradient(pos,data["time"],axis=0)
    for row,vel in zip(geometry,velocity):
        row["tip_velocity_world_m_s"]=vel
    phase_summary=[]
    for phase in result["phases"]:
        rows=[(i,row) for i,row in enumerate(trace) if row["phase"]==phase["phase"]]
        ids=[i for i,_ in rows]
        forces=np.array([row["force_on_fingers_N"] for _,row in rows])
        grip=np.array([row["grip_each_N"] for _,row in rows])
        phase_summary.append(dict(**phase,samples=len(rows),
            peak_force_magnitude_each_N=np.linalg.norm(forces,axis=2).max(0),
            peak_grip_each_N=grip.max(0),both_grip_above_50mN_samples=int(np.all(grip>.05,axis=1).sum()),
            fingertip_travel_each_m=np.sum(np.linalg.norm(np.diff(pos[ids],axis=0),axis=2),axis=0),
            min_facing_surface_gap_m=min(geometry[i]["facing_surface_gap_m"] for i in ids),
            overlapping_core_samples=sum(geometry[i]["core_obb_overlap"] for i in ids),
            actual_gap_range_m=[min(geometry[i]["actual_gap_m"] for i in ids),max(geometry[i]["actual_gap_m"] for i in ids)],
            command_gap_range_m=[min(row["command_gap"] for _,row in rows),max(row["command_gap"] for _,row in rows)]))
    named={"table":m.geom("table").id,"payload":m.geom("payload_geom_0").id,
           "left_pad":pad_ids[0],"right_pad":pad_ids[1]}
    masks={name:dict(geom=gid,contype=int(m.geom_contype[gid]),conaffinity=int(m.geom_conaffinity[gid]),
                    type=int(m.geom_type[gid]),body=int(m.geom_bodyid[gid]),friction=m.geom_friction[gid],
                    priority=int(m.geom_priority[gid])) for name,gid in named.items()}
    masks["film"]=dict(contype=int(m.flex_contype[0]),conaffinity=int(m.flex_conaffinity[0]),
                        friction=m.flex_friction[0],priority=int(m.flex_priority[0]),selfcollide=int(m.flex_selfcollide[0]))
    pairs=[]
    for i,(name,a) in enumerate(masks.items()):
        for other,b in list(masks.items())[i+1:]:
            pairs.append(dict(a=name,b=other,mask_allows=bool((a['contype']&b['conaffinity'])or(b['contype']&a['conaffinity']))))
    sources={}
    for key,relative in (("yam_protocol","experience/yam_platform/protocol.py"),
                         ("yam_robot","experience/yam_platform/robot.py"),
                         ("yam_contacts","experience/yam_platform/contacts.py"),
                         ("parcel_model","bag_sim/physical_parcel.py")):
        digest=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        sources[relative]=dict(live_sha256=digest,recorded_sha256=result['source_sha256'][key],
                               same=digest==result['source_sha256'][key])
    write_json(output/"geometry.json",geometry)
    summary=dict(source_trial=str(trial.relative_to(ROOT)),integration_run=False,mujoco=mujoco.__version__,
        nsensor=int(m.nsensor),nflexvert=int(m.nflexvert),nflexelem=int(m.nflexelem),
        source_comparison=sources,phases=phase_summary,masks=masks,mask_pairs=pairs,
        first_failure=result["grasp_failure"],legacy_content_geom_set_empty=True,
        limitations=["saved trajectory lacks per-sample qvel/ctrl and raw contacts; forces above are original recorded values",
                     "geometric gaps measure distal edge center points, not the minimum over complete fingertip surfaces",
                     "finite-difference velocities are at 20 ms resolution; no claimed solver-force replay from qpos alone"])
    write_json(output/"summary.json",summary)
    print(json.dumps(summary,default=lambda x:x.tolist()),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--trial",type=Path,default=ROOT/"test/runs/2026-11-16/bag/physical-parcel-top-parameters-r2/medium/recording/trial")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.trial,args.output)
