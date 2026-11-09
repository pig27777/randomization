"""F6b closed-loop side-grasp batch (hard box only), matching loose_protocols.evaluate.

GPU timestep/controller are captured together. Contacts and constraints are sized from CPU maxima.
Every overflow invalidates the batch. Finished worlds are counted once, without a reset/settle charge.
"""
import argparse
import gc
import json
import math
import subprocess
import threading
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from pathlib import Path
import sys

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

from loose_protocols import make, rotate_package, _rot, _pad_reach, GAP
from mj_adapter import PAD_T
from round3_flex import OUT, CACHE, CONFIGS, content_kappa, f3_inputs, check_deadline, arm_deadline, append
from flex_geometry import penetration_vertices, penetration_corners


@wp.struct
class BatchData:
    qpos: wp.array2d(dtype=float)
    qvel: wp.array2d(dtype=float)
    xipos: wp.array2d(dtype=wp.vec3)
    nacon: wp.array(dtype=int)
    geom: wp.array(dtype=wp.vec2i)
    frame: wp.array(dtype=wp.mat33)
    address: wp.array2d(dtype=int)
    worldid: wp.array(dtype=int)
    force: wp.array2d(dtype=float)
    njmax: int


@wp.kernel
def overflow_count(current:wp.array(dtype=int),bits:wp.array(dtype=int),events:wp.array(dtype=int)):
    w=wp.tid()
    if current[w]!=0:
        bits[w]=bits[w]|current[w]
        events[w]=events[w]+1


@wp.kernel
def reactions(d: BatchData, pad: wp.array(dtype=int), table: int, raw: wp.array2d(dtype=float), axis: int):
    c = wp.tid()
    if c < d.nacon[0]:
        a = d.address[c, 0]
        w = d.worldid[c]
        if a >= 0 and a < d.njmax:
            f = d.force[w, a]
            g = d.geom[c]
            normal = d.frame[c]
            for j in range(2):
                if g[j] >= 0:
                    p = pad[g[j]]
                    if p >= 0:
                        wp.atomic_add(raw,w,p,f*wp.abs(normal[0,axis]))
                    if g[j] == table:
                        wp.atomic_add(raw,w,2,f*wp.abs(normal[0,2]))


@wp.kernel
def observe(d: BatchData, raw: wp.array2d(dtype=float), state: wp.array2d(dtype=float),
            phase: wp.array(dtype=int), outcome: wp.array(dtype=int), obj: int,
            bag_bodies: wp.array(dtype=int), masses: wp.array(dtype=float), mass: float, dt: float):
    w = wp.tid()
    if outcome[w] == 0:
        # state: 0 time, 1 phase start, 2/3 approach close, 4/5 touched, 6 squeeze,
        # 7/8 filtered forces, 9 measured force, 10 lost time, 11 steps,
        # 12/13/14 rest COM, 15/16/17 arm, 18 place height, 19 peak measured force.
        f = float(20.)
        for p in range(2):
            old = state[w,7+p]
            state[w,7+p] = old+dt/(.005+dt)*(wp.min(raw[w,p],wp.max(2.*f,old+20.))-old)
        measured = state[w,9]+dt/(.020+dt)*(.5*(raw[w,0]+raw[w,1])-state[w,9])
        state[w,9] = measured
        state[w,19] = wp.max(state[w,19],measured)
        state[w,0] = state[w,0]+dt
        step = int(state[w,11])+1
        state[w,11] = float(step)
        if step%12 == 0 and (phase[w] == 4 or phase[w] == 5):
            lost = float(0.)
            if wp.min(state[w,7],state[w,8]) < .05:
                lost = state[w,10]+.010
            state[w,10] = lost
            if lost >= .05:
                outcome[w] = 2 if phase[w] == 4 else 3
        if state[w,0] > 30.:
            outcome[w] = 4


@wp.func
def pkg_com(d: BatchData, w: int, obj: int, bag: wp.array(dtype=int), masses: wp.array(dtype=float), mass: float):
    value = d.xipos[w,obj]*masses[obj]
    for i in range(bag.shape[0]):
        b = bag[i]
        value = value+d.xipos[w,b]*masses[b]
    return value/mass


@wp.kernel
def controller(d: BatchData, state: wp.array2d(dtype=float), phase: wp.array(dtype=int),
               outcome: wp.array(dtype=int), previous: wp.array2d(dtype=float),
               start: wp.array(dtype=float), p0: wp.array(dtype=float),
               qa: wp.array(dtype=int), va: wp.array(dtype=int), raw: wp.array2d(dtype=float),
               obj: int, bag: wp.array(dtype=int), masses: wp.array(dtype=float), mass: float,
               travel: float, carry: wp.vec3, carry_t: float, dt: float):
    w = wp.tid()
    if outcome[w] != 0:
        if state[w,22] < 0.:
            state[w,22] = d.xipos[w,obj][2]*1000.
        for j in range(6):
            d.qvel[w,va[j]] = 0.
        return
    t = state[w,0]
    tau = t-state[w,1]
    p = phase[w]
    next_p = p
    com = pkg_com(d,w,obj,bag,masses,mass)
    arm = wp.vec3(state[w,15],state[w,16],state[w,17])
    if p == 0:
        if tau >= .3:
            for j in range(3): state[w,12+j] = com[j]
            state[w,21] = d.xipos[w,obj][2]
            next_p = 1
    elif p == 1:
        for j in range(2):
            if state[w,4+j] < 0.:
                if state[w,7+j] > .05: state[w,4+j] = state[w,2+j]
                else: state[w,2+j] = state[w,2+j]+.025*dt
        if state[w,4] >= 0. and state[w,5] >= 0.: next_p = 2
        elif tau > 5.: outcome[w] = 2
    elif p == 2:
        if .5*(state[w,7]+state[w,8]) >= 19. or state[w,6] >= travel or tau > 8.: next_p = 3
    elif p == 3:
        if tau >= .2: next_p = 4
    elif p == 4:
        arm = wp.vec3(0.,0.,wp.min(.1*tau,.1))
        if tau >= 1.2:
            state[w,20] = (d.xipos[w,obj][2]-state[w,21])*1000.
            if com[2]-state[w,14] < .05: outcome[w] = 2
            else: next_p = 5
    elif p == 5:
        fraction = float(1.)
        if tau < carry_t: fraction = tau/carry_t-wp.sin(2.*wp.pi*tau/carry_t)/(2.*wp.pi)
        arm = wp.vec3(0.,0.,.1)+fraction*carry
        if tau >= carry_t+.3:
            if com[2]-state[w,14] < .03: outcome[w] = 3
            else: next_p = 6
    elif p == 6:
        if int(state[w,11])%12 == 0 and raw[w,2] > .3*mass*9.81: state[w,18] = arm[2]
        if state[w,18] < 0.:
            arm = wp.vec3(0.,0.,wp.max(.1-.2*tau,.005))+carry
            if arm[2] <= .005: state[w,18] = .005
        if state[w,18] >= 0.: next_p = 7
    elif p == 7:
        arm[2] = state[w,18]
        opened = wp.min(2.*.025*tau,.03)
        for j in range(2): state[w,2+j] = state[w,4+j]+state[w,6]-opened
        if tau >= 1.1:
            offset = com-wp.vec3(state[w,12],state[w,13],state[w,14])-carry
            direction = carry/.5
            x = offset[0]*direction[0]+offset[1]*direction[1]
            y = -offset[0]*direction[1]+offset[1]*direction[0]
            outcome[w] = 1 if wp.abs(x) <= .1 and wp.abs(y) <= .1 else 3
    if next_p != p:
        phase[w] = next_p
        state[w,1] = t
    p = next_p
    if p >= 2 and p <= 6:
        err = 20.-.5*(state[w,7]+state[w,8])
        squeeze = wp.clamp(state[w,6]+wp.clamp(.0025*err,-.025,.025)*dt,-.005,travel)
        state[w,6] = squeeze
        for j in range(2): state[w,2+j] = state[w,4+j]+squeeze
    for j in range(3): state[w,15+j] = arm[j]
    for j in range(6):
        side = j//3
        axis = j%3
        target = start[j]+arm[axis]
        if axis == 1: target = target+(1.-2.*float(side))*state[w,2+side]
        d.qpos[w,qa[j]] = previous[w,j]-p0[j]
        d.qvel[w,va[j]] = (target-previous[w,j])/dt
        previous[w,j] = target


def prepare(bag, params):
    k = content_kappa('box_loose')
    pads = [('pA',1,(0,-1,.5)),('pB',1,(0,1,.5))]
    sim = make(k,bag,pads,big_table=True,dt=.0008,cache_dir=CACHE)
    cxy,size,h = sim.footprint()
    u,v,psi,F,lv,accel=params
    d_obj=np.array([math.cos(psi),math.sin(psi)])
    e_obj=np.array([d_obj[1],-d_obj[0]])
    g=np.array([u,v])*size
    ra=_pad_reach(g,-d_obj,e_obj,size/2)
    rb=_pad_reach(g,d_obj,e_obj,size/2)
    yaw=math.pi/2-math.atan2(d_obj[1],d_obj[0])
    rotate_package(sim,np.r_[cxy,0.],yaw)
    gw=cxy+_rot(yaw)@g
    starts=np.array([[gw[0],gw[1]-ra-GAP-PAD_T,h/2],[gw[0],gw[1]+rb+GAP+PAD_T,h/2]])
    for p,x in zip(sim.pad_names,starts): sim.set_pad(p,x,np.zeros(3))
    mujoco.mj_forward(sim.m,sim.d)
    return sim,starts,min(ra,rb)


class MemoryPeak:
    def __init__(self): self.peak=0; self.stop=threading.Event()
    def __enter__(self):
        def poll():
            while not self.stop.is_set():
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                self.peak=max(self.peak,int(value.strip()))
                self.stop.wait(.2)
        self.thread=threading.Thread(target=poll,daemon=True); self.thread.start(); return self
    def __exit__(self,*args): self.stop.set();self.thread.join()


def run(nworld, config, case, params, seed, njmax, ncon):
    check_deadline()
    sim,starts,travel=prepare(CONFIGS[config],params)
    with MemoryPeak() as mem:
        wm=mjw.put_model(sim.m)
        wd=mjw.put_data(sim.m,sim.d,nworld=nworld,nconmax=ncon,naconmax=ncon*nworld,njmax=njmax)
        bd=BatchData()
        bd.qpos,bd.qvel,bd.xipos=wd.qpos,wd.qvel,wd.xipos
        bd.nacon,bd.geom,bd.frame=wd.nacon,wd.contact.geom,wd.contact.frame
        bd.address,bd.worldid,bd.force=wd.contact.efc_address,wd.contact.worldid,wd.efc.force
        bd.njmax=wd.njmax
        state=np.zeros((nworld,23),dtype=np.float32);state[:,4:6]=-1;state[:,18]=-1
        state[:,20]=np.nan;state[:,22]=-1
        s=wp.array(state,dtype=float);phase=wp.zeros(nworld,dtype=int);outcome=wp.zeros(nworld,dtype=int)
        raw=wp.zeros((nworld,3),dtype=float)
        prev=wp.array(np.tile(starts.ravel(),(nworld,1)).astype(np.float32),dtype=float)
        start=wp.array(starts.ravel().astype(np.float32),dtype=float)
        p0=wp.array(np.concatenate([sim.pad_p0[p] for p in sim.pad_names]).astype(np.float32),dtype=float)
        qa=wp.array(np.concatenate([sim.pad_q[p] for p in sim.pad_names]).astype(np.int32),dtype=int)
        va=wp.array(np.concatenate([sim.pad_v[p] for p in sim.pad_names]).astype(np.int32),dtype=int)
        lookup=np.full(sim.m.ngeom,-1,dtype=np.int32)
        for j,p in enumerate(sim.pad_names): lookup[list(sim.pad_geoms[p])]=j
        lookup=wp.array(lookup,dtype=int)
        bodies=wp.array(sim.bag_bodies.astype(np.int32),dtype=int)
        masses=wp.array(sim.m.body_mass.astype(np.float32),dtype=float)
        angle=float(np.random.default_rng(seed).uniform(0,2*math.pi))
        carry=wp.vec3(.5*math.cos(angle),.5*math.sin(angle),0.)
        carry_t=math.sqrt(2*math.pi*.5/2.)
        obj=sim.seg_bodies[0]
        faces=wp.array(sim.meta['faces'].astype(np.int32),dtype=wp.vec3i)
        peak_pen=wp.zeros(nworld,dtype=float)
        escaped=wp.zeros(nworld,dtype=int)
        overflow_bits=wp.zeros(nworld,dtype=int)
        overflow_events=wp.zeros(nworld,dtype=int)
        half=wp.vec3(*sim.desc.segments[0].half)
        def step():
            wp.launch(controller,nworld,inputs=[bd,s,phase,outcome,prev,start,p0,qa,va,raw,obj,bodies,masses,sim.pkg_mass,travel,carry,carry_t,.0008])
            wd.overflow.zero_()
            mjw.step(wm,wd)
            wp.launch(overflow_count,nworld,inputs=[wd.overflow,overflow_bits,overflow_events])
            raw.zero_()
            wp.launch(reactions,wd.naconmax,inputs=[bd,lookup,sim.table_geom,raw,1])
            wp.launch(observe,nworld,inputs=[bd,raw,s,phase,outcome,obj,bodies,masses,sim.pkg_mass,.0008])
            wp.launch(penetration_vertices,(nworld,sim.m.nflexvert),inputs=[wd.flexvert_xpos,wd.xpos,wd.xmat,obj,half,s,outcome,peak_pen])
            wp.launch(penetration_corners,(nworld,9),inputs=[wd.flexvert_xpos,faces,wd.xpos,wd.xmat,obj,half,s,outcome,peak_pen,escaped])
        step(); wp.synchronize()
        # Two initial steps compile/capture; the remaining complete protocol is timed.
        with wp.ScopedCapture() as cap: step()
        wp.synchronize()
        t0=time.perf_counter()
        executed=2
        while executed < round(30/.0008)+24:
            for _ in range(24): wp.capture_launch(cap.graph)
            wp.synchronize();executed+=24
            codes=outcome.numpy()
            if np.all(codes != 0) or np.any(overflow_bits.numpy() != 0): break
        wall=time.perf_counter()-t0
        codes=outcome.numpy();states=s.numpy()
        overflow=overflow_bits.numpy()
        q=wd.qpos.numpy()
        positions=wd.xipos.numpy()[:,obj]
        lift_heights=states[np.isfinite(states[:,20]),20]
        invalid=int(np.count_nonzero(overflow))
        finite=np.isfinite(q).all(axis=1)
        codes[~finite]=4
        codes[escaped.numpy()!=0]=4
        names={0:'unfinished',1:'success',2:'grasp_fail',3:'drop',4:'sim_fail'}
        categories={name:int((codes==code).sum()) for code,name in names.items()}
        valid=invalid==0 and bool(finite.all()) and bool(np.all(codes!=0))
        majority=max(categories,key=categories.get)
        # Exact CPU containment/penetration for the representative of each observed outcome.
        checks=[]
        for code in np.unique(codes):
            i=int(np.flatnonzero(codes==code)[0])
            mjw.get_data_into(sim.d,sim.m,wd,i)
            checks.append(dict(world=i,outcome=names[int(code)],outside=sim.outside_fraction(),penetration_mm=sim.penetration()*1e3))
        result=dict(config=config,case=case,params=params,seed=seed,nworld=nworld,njmax=njmax,ncon_per_world=ncon,
                    wall_s=wall,completed=int((codes!=0).sum()),valid=valid,evals_per_s=int((codes!=0).sum())/wall if valid else None,
                    env_steps_per_s=nworld*(executed-2)/wall if valid else None,overflow_worlds=invalid,
                    overflow_bits=np.unique(overflow).tolist(),finite_worlds=int(finite.sum()),outcomes=categories,
                    majority=majority,content_z_mm=np.percentile(positions[:,2]*1e3,[0,50,100]).tolist(),
                    completion_sim_s=np.percentile(states[:,0],[0,50,100]).tolist(),end_checks=checks)
        result['content_lift_mm'] = np.percentile(lift_heights,[0,50,100]).tolist() if len(lift_heights) else None
        end_z=states[states[:,22]>=0,22]
        result['completion_content_z_mm'] = np.percentile(end_z,[0,50,100]).tolist() if len(end_z) else None
        result['max_penetration_mm'] = float(peak_pen.numpy().max()*1e3)
        result['escaped_worlds'] = int(np.count_nonzero(escaped.numpy()))
        result['overflow_world_steps'] = int(overflow_events.numpy().sum())
    result['peak_total_gpu_mib']=mem.peak
    return result


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--nworld',type=int,required=True)
    ap.add_argument('--config',choices=CONFIGS,default='base')
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    arm_deadline()
    cpu=[json.loads(l) for l in (OUT/'base_behavior.jsonl').read_text().splitlines()]
    for filename in ['base_speed.jsonl','base_speed_cached.jsonl','base_r2_retry.jsonl','final_behavior.jsonl']:
        path=OUT/filename
        if path.exists(): cpu.extend(json.loads(l) for l in path.read_text().splitlines())
    njmax=math.ceil(1.5*max(r['max_nefc'] for r in cpu))
    ncon=math.ceil(1.5*max(r['max_ncon'] for r in cpu))
    cases=list(f3_inputs())
    if a.smoke: cases=cases[:1]
    path=OUT/f'gpu_{a.config}_{a.nworld}.jsonl'
    completed=set()
    if path.exists():
        for line in path.read_text().splitlines():
            old=json.loads(line)
            if 'seed' in old and 'error' not in old:
                completed.add((old['case'],old['seed']))
    for name,params,seed in cases:
        if (name,seed) in completed: continue
        try:
            result=run(a.nworld,a.config,name,params,seed,njmax,ncon)
        except (RuntimeError, MemoryError) as error:
            result=dict(config=a.config,case=name,nworld=a.nworld,njmax=njmax,ncon_per_world=ncon,
                        error=str(error),valid=False,evals_per_s=None,env_steps_per_s=None)
            append(path,result)
            print(json.dumps(result),flush=True)
            raise
        append(path,result)
        print(json.dumps(result),flush=True)
        gc.collect()


if __name__=='__main__': main()
