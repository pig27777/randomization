"""R2/R3 GPU behavior check with the frozen CPU Closer called independently per world.

Only the small per-world controller/pose arrays cross the device boundary. These
runs are behavior diagnostics, not throughput measurements.
"""
import argparse
import json
import math
from collections import deque

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from flex_batch import BatchData, reactions
from flex_geometry import penetration_vertices, penetration_corners
from round3_flex import OUT, CACHE, CONFIGS, append, content_kappa, check_deadline, arm_deadline
from loose_protocols import make, rotate_package, GAP
from mj_adapter import PAD_T
from model import PAD
from protocols import Closer, move_profile


@wp.kernel
def set_pads(d:BatchData,q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),
             p0:wp.array(dtype=float),qa:wp.array(dtype=int),va:wp.array(dtype=int)):
    w,j=wp.tid()
    d.qpos[w,qa[j]]=q[w,j]-p0[j]
    d.qvel[w,va[j]]=v[w,j]


@wp.kernel
def read_positions(d:BatchData,body:wp.array(dtype=int),positions:wp.array2d(dtype=float)):
    w=wp.tid()
    for k in range(3):
        for j in range(3): positions[w,3*k+j]=d.xipos[w,body[k]][j]


def prepare(test,bag):
    k=content_kappa('box_loose')
    if test=='r3':
        sim=make(k,bag,[('pA',1,(0,-1,.5)),('pB',1,(0,1,.5))],dt=.0008,cache_dir=CACHE)
        cxy,size,h=sim.footprint()
        starts=np.array([[cxy[0],cxy[1]-size[1]/2-GAP-PAD_T,h/2],
                         [cxy[0],cxy[1]+size[1]/2+GAP+PAD_T,h/2]])
        return sim,starts,np.array([[0,1,0],[0,-1,0]]),.45*size[1],.05
    def built(pads,xmin=-.6):
        s=make(k,bag,pads,table_xmin=xmin,dt=.0008,cache_dir=CACHE)
        cxy,_,_=s.footprint()
        rotate_package(s,np.r_[cxy,0.],math.pi)
        return s
    sim=built([('pA',2,(0,0,.5)),('pB',2,(0,0,-.5))])
    cxy,size,_=sim.footprint()
    edge=cxy[0]-size[0]/2;x0=edge+PAD/2
    sim=built([('pA',2,(x0,cxy[1],.5)),('pB',2,(x0,cxy[1],-.5))],edge+.05)
    pts=sim.bag_points();near=pts[np.abs(pts[:,0]-x0)<PAD/2+.01]
    top=float(near[:,2].max())+.002
    starts=np.array([[x0,cxy[1],top+GAP+PAD_T],[x0,cxy[1],-GAP-PAD_T]])
    return sim,starts,np.array([[0,0,-1],[0,0,1]]),.01,max(.1,size[0]+.05)


def run(test,nworld,config,ncon,njmax):
    check_deadline()
    sim,starts,direction,travel,height=prepare(test,CONFIGS[config])
    for p,x in zip(sim.pad_names,starts): sim.set_pad(p,x,np.zeros(3))
    mujoco.mj_forward(sim.m,sim.d)
    wm=mjw.put_model(sim.m)
    wd=mjw.put_data(sim.m,sim.d,nworld=nworld,nconmax=ncon,naconmax=ncon*nworld,njmax=njmax)
    bd=BatchData()
    bd.qpos,bd.qvel,bd.xipos=wd.qpos,wd.qvel,wd.xipos
    bd.nacon,bd.geom,bd.frame=wd.nacon,wd.contact.geom,wd.contact.frame
    bd.address,bd.worldid,bd.force=wd.contact.efc_address,wd.contact.worldid,wd.efc.force
    bd.njmax=wd.njmax
    qa=np.r_[sim.pad_q['pA'],sim.pad_q['pB']].astype(np.int32)
    va=np.r_[sim.pad_v['pA'],sim.pad_v['pB']].astype(np.int32)
    p0=wp.array(np.r_[sim.pad_p0['pA'],sim.pad_p0['pB']].astype(np.float32),dtype=float)
    wqa,wva=wp.array(qa,dtype=int),wp.array(va,dtype=int)
    tq=wp.array(np.tile(starts.ravel(),(nworld,1)).astype(np.float32),dtype=float)
    tv=wp.zeros((nworld,6),dtype=float)
    lookup=np.full(sim.m.ngeom,-1,dtype=np.int32)
    for j,p in enumerate(sim.pad_names): lookup[list(sim.pad_geoms[p])]=j
    lookup=wp.array(lookup,dtype=int)
    raw=wp.zeros((nworld,3),dtype=float)
    pos=wp.zeros((nworld,9),dtype=float)
    obj=sim.seg_bodies[0]
    bodies=wp.array([obj,*[sim.m.body(p).id for p in sim.pad_names]],dtype=int)
    faces=wp.array(sim.meta['faces'].astype(np.int32),dtype=wp.vec3i)
    state=wp.zeros((nworld,23),dtype=float)
    finished=wp.zeros(nworld,dtype=int)
    peak=wp.zeros(nworld,dtype=float);escaped=wp.zeros(nworld,dtype=int)
    half=wp.vec3(*sim.desc.segments[0].half)
    axis=2 if test=='r2' else 1
    def physics():
        mjw.step(wm,wd)
        raw.zero_()
        wp.launch(reactions,wd.naconmax,inputs=[bd,lookup,sim.table_geom,raw,axis])
        wp.launch(read_positions,nworld,inputs=[bd,bodies,pos])
    physics()
    with wp.ScopedCapture() as cap: physics()
    # Compilation does not consume diagnostic trajectory steps.
    wd.qpos.assign(np.tile(sim.d.qpos,(nworld,1)).astype(np.float32))
    wd.qvel.assign(np.tile(sim.d.qvel,(nworld,1)).astype(np.float32))
    wd.qacc_warmstart.assign(np.tile(sim.d.qacc_warmstart,(nworld,1)).astype(np.float32))
    wd.time.fill_(0.)
    mjw.forward(wm,wd)
    wp.launch(read_positions,nworld,inputs=[bd,bodies,pos]);wp.synchronize()
    starts_dict=dict(zip(sim.pad_names,starts))
    directions=dict(zip(sim.pad_names,direction))
    ctrls=[Closer(starts_dict,directions,.025,travel,20.) for _ in range(nworld)]
    closed=np.full(nworld,np.nan);lift=np.full(nworld,np.nan)
    distance0=np.full(nworld,np.nan);initial_z=np.full(nworld,np.nan)
    previous=np.tile(starts,(nworld,1,1))
    base=np.tile(starts,(nworld,1,1))
    done=np.zeros(nworld,dtype=np.int32)
    samples=[deque() for _ in range(nworld)]
    raw_host=np.zeros((nworld,3));positions=pos.numpy()
    dt=.0008
    for step in range(round(40/dt)):
        if step%1200==0: check_deadline()
        t=step*dt
        targets=previous.copy()
        for w in np.flatnonzero(done==0):
            force={p:float(raw_host[w,j]) for j,p in enumerate(sim.pad_names)}
            if t<.3: target=starts_dict
            elif np.isnan(closed[w]) or t<closed[w]+.2:
                target=ctrls[w](dt,force,t)
                if (ctrls[w].frozen or t-.3>8.) and np.isnan(closed[w]):
                    closed[w]=t;base[w]=np.array([target[p] for p in sim.pad_names])
            else:
                if np.isnan(lift[w]):
                    lift[w]=t
                    grip=.5*(positions[w,3:6]+positions[w,6:9])
                    distance0[w]=np.linalg.norm(positions[w,:3]-grip)
                tau=t-lift[w]
                if test=='r2':
                    hold_start=height/.05;duration=hold_start+2.
                    offset=np.array([0.,0.,min(.05*tau,height)])
                else:
                    hold_start=2.3;duration=3.3
                    offset=np.array([move_profile(min(max(tau-1.5,0.),.8)),0.,min(.05*tau,.05)])
                if tau>=duration:
                    done[w]=1;continue
                held=ctrls[w].hold(dt,force,dict(zip(sim.pad_names,base[w])))
                target={p:held[p]+offset for p in sim.pad_names}
            targets[w]=[target[p] for p in sim.pad_names]
        if done.all(): break
        velocities=(targets-previous)/dt
        tq.assign(previous.reshape(nworld,6).astype(np.float32))
        tv.assign(velocities.reshape(nworld,6).astype(np.float32))
        wp.launch(set_pads,(nworld,6),inputs=[bd,tq,tv,p0,wqa,wva])
        wp.capture_launch(cap.graph)
        previous=targets
        raw_host=raw.numpy();positions=pos.numpy()
        if (step+1)%12==0:
            finished.assign(done)
            wp.launch(penetration_vertices,(nworld,sim.m.nflexvert),inputs=[wd.flexvert_xpos,wd.xpos,wd.xmat,obj,half,state,finished,peak])
            wp.launch(penetration_corners,(nworld,9),inputs=[wd.flexvert_xpos,faces,wd.xpos,wd.xmat,obj,half,state,finished,peak,escaped])
            for w in np.flatnonzero(done==0):
                cz=float(positions[w,2])
                if np.isnan(initial_z[w]): initial_z[w]=cz
                if not np.isnan(lift[w]):
                    hold_start=height/.05 if test=='r2' else 2.3
                    if t+dt-lift[w]>=hold_start:
                        grip=.5*(positions[w,3:6]+positions[w,6:9])
                        samples[w].append((t+dt,cz,float(np.linalg.norm(positions[w,:3]-grip))))
                        while samples[w] and samples[w][0][0]<t+dt-.5: samples[w].popleft()
            if np.any(wd.overflow.numpy()!=0): break
    wp.synchronize()
    overflow=wd.overflow.numpy()
    finite=np.isfinite(wd.qpos.numpy()).all(axis=1)
    metric=[]
    for w,rows in enumerate(samples):
        if rows:
            mean=np.array(rows).mean(0)
            metric.append((mean[2]-distance0[w])*1e3 if test=='r2' else 50.-(mean[1]-initial_z[w])*1e3)
        else: metric.append(float('nan'))
    values=np.asarray(metric)
    baseline=76. if test=='r2' else 26.
    passed=(np.isfinite(values)&(np.abs(values-baseline)<=.3*baseline)&finite&
            (overflow==0)&(escaped.numpy()==0)&(peak.numpy()<=.001)&(done!=0))
    return dict(test=test,nworld=nworld,config=config,metric='slide_mm' if test=='r2' else 'sink_mm',
                metric_percentiles_mm=np.nanpercentile(values,[0,50,100]).tolist(),
                worlds_within_original_standard=int(passed.sum()),completed=int(done.sum()),
                overflow_worlds=int(np.count_nonzero(overflow)),finite_worlds=int(finite.sum()),
                escaped_worlds=int(np.count_nonzero(escaped.numpy())),max_penetration_mm=float(peak.numpy().max()*1e3),
                ncon_per_world=ncon,njmax=njmax,throughput_measurement=False)


def main():
    arm_deadline()
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('test',choices=['r2','r3']);p.add_argument('--nworld',type=int,required=True)
    a=p.parse_args()
    cpu=[]
    for name in ['base_behavior','base_speed','base_speed_cached','final_behavior']:
        path=OUT/(name+'.jsonl')
        if path.exists(): cpu.extend(json.loads(l) for l in path.read_text().splitlines())
    ncon=math.ceil(1.5*max(r['max_ncon'] for r in cpu))
    njmax=math.ceil(1.5*max(r['max_nefc'] for r in cpu))
    result=run(a.test,a.nworld,'base',ncon,njmax)
    append(OUT/f'gpu_final_{a.test}.jsonl',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
