"""B4 hard-content batches: native MJWarp physics, vectorized host top-grasp control.

The frozen v1.3 soft-content GPU backend is not available. This entry point refuses
soft contents; host control and transfers are included in the measured wall time.
"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import datetime

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from round3_flex import OUT, append, content_kappa
from round3_effect import RESULTS
from effect_model import EffectBag
from loose_protocols import make
from top_grasp import TopGrasp, task_scene
from flex_batch import BatchData, MemoryPeak, overflow_count
from mj_loose import CONTENT_RELEASE_SHA256


@wp.struct
class Contacts:
    count: wp.array(dtype=int)
    geom: wp.array(dtype=wp.vec2i)
    frame: wp.array(dtype=wp.mat33)
    address: wp.array2d(dtype=int)
    world: wp.array(dtype=int)
    dim: wp.array(dtype=int)
    distance: wp.array(dtype=float)
    force: wp.array2d(dtype=float)


@wp.kernel
def read_contacts(c:Contacts,pad:wp.array(dtype=int),package:wp.array(dtype=int),table:int,
                  bag_geom:int,raw:wp.array3d(dtype=float),facts:wp.array2d(dtype=float)):
    i=wp.tid()
    if i<c.count[0]:
        w=c.world[i];a=c.address[i,0]
        if a>=0:
            g=c.geom[i]
            if g[0]>=0 and g[1]>=0:
                local=wp.vec3(c.force[w,a],0.,0.)
                if c.dim[i]>1: local[1]=c.force[w,a+1]
                if c.dim[i]>2: local[2]=c.force[w,a+2]
                on2=wp.transpose(c.frame[i])*local
                for side in range(2):
                    p=pad[g[side]]
                    if p>=0 and package[g[1-side]]!=0:
                        sign=-1. if side==0 else 1.
                        for axis in range(3): wp.atomic_add(raw,w,p,axis,sign*on2[axis])
                        if g[1-side]==bag_geom and c.force[w,a]>.05:
                            wp.atomic_max(facts,w,p,1.)
                        for axis in range(3):
                            wp.atomic_add(facts,w,4+3*p+axis,c.force[w,a]*wp.abs(c.frame[i][0,axis]))
                    if g[side]==table and package[g[1-side]]!=0:
                        sign=1. if side==0 else -1.
                        wp.atomic_add(facts,w,2,sign*on2[2])
                        if c.distance[i]<=0.: wp.atomic_max(facts,w,3,1.)


@wp.kernel
def set_gripper(d:BatchData,q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),
                qa:wp.array(dtype=int),va:wp.array(dtype=int),p0:wp.array(dtype=float),
                mocap:wp.array2d(dtype=wp.vec3),target:wp.array(dtype=wp.vec3),
                active:wp.array2d(dtype=bool),enabled:wp.array(dtype=int),
                spring:wp.array(dtype=float),bodies:wp.array(dtype=int),
                applied:wp.array2d(dtype=wp.spatial_vector)):
    w=wp.tid()
    for j in range(6):
        d.qpos[w,qa[j]]=q[w,j]-p0[j]
        d.qvel[w,va[j]]=v[w,j]
    mocap[w,0]=target[w]
    if active.shape[1]>0: active[w,0]=enabled[w]!=0
    applied[w,bodies[0]]=wp.spatial_vector(0.,-spring[w],0.,0.,0.,0.)
    applied[w,bodies[1]]=wp.spatial_vector(0.,spring[w],0.,0.,0.,0.)


def check_deadline():
    m=json.loads((OUT/'run_manifest.json').read_text())
    end=datetime.fromisoformat(m['effect_deadline'])
    if datetime.now(end.tzinfo)>=end: raise TimeoutError('B1-B4 deadline reached')


def prepare(with_bag,u):
    k=content_kappa('box_loose')
    assert k.rigid
    if with_bag:
        park=[('pA',1,(0,-1,.5)),('pB',1,(0,1,.5))]
        sim=make(k,replace(EffectBag(size=(.35,.25)),pinch=True),(),big_table=True,
                 dt=.0004,fingertips=park,cache_dir=RESULTS/'cache')
        cxy,size,h=sim.footprint()
        xy=cxy+np.array([u,0.])*size
        starts=np.array([[xy[0],xy[1]-.032,h+.025],[xy[0],xy[1]+.032,h+.025]])
        bag_geom=sim.m.geom('bag_envelope').id
        eligible=bool(abs(u*size[0])>k.size[0]/2)
        if not eligible: sim.m.geom_conaffinity[bag_geom]&=~4
        mass=sim.pkg_mass
    else:
        sim,start,h=task_scene(k,TopGrasp(u=u,w=.060,d=.012,F=20.),0.)
        starts=np.array(list(start.values()))
        bag_geom=-1;eligible=False;mass=sim.mass
        # Cache the same short static preparation used by the bag scene.
        key=hashlib.sha256((sim.built.xml+CONTENT_RELEASE_SHA256).encode()).hexdigest()
        cache=RESULTS/'cache'/f'bare_{key}.npz'
        state_type=mujoco.mjtState.mjSTATE_INTEGRATION
        if cache.exists():
            with np.load(cache) as saved:
                mujoco.mj_setState(sim.m,sim.d,saved['state'],state_type)
        else:
            for _ in range(250): sim.step()
            sim.d.qvel[:]=0.;sim.d.time=0.;mujoco.mj_forward(sim.m,sim.d)
            cache.parent.mkdir(parents=True,exist_ok=True)
            state=np.empty(mujoco.mj_stateSize(sim.m,state_type))
            mujoco.mj_getState(sim.m,sim.d,state,state_type)
            np.savez(cache,state=state)
    for p,x in zip(sim.pad_names,starts): sim.set_pad(p,x,np.zeros(3))
    mujoco.mj_forward(sim.m,sim.d)
    return sim,starts,h,bag_geom,eligible,mass


class TopControl:
    """NumPy form of the existing hard-content v2.2 executor; all worlds finish once."""
    def __init__(self,n,starts,h,eligible,mass,seed):
        self.n=n;self.starts=starts;self.h=h;self.eligible=eligible;self.mass=mass
        self.phase=np.zeros(n,int);self.pt=np.zeros(n);self.code=np.zeros(n,int)
        self.arm=np.zeros((n,3));self.squeeze=np.zeros(n);self.release_squeeze=np.zeros(n)
        self.ctrl=np.zeros(n);self.measured=np.zeros(n);self.current=np.zeros(n)
        self.press=np.zeros(n);self.depth=np.full(n,.012);self.lift=np.zeros(n)
        self.touch=np.zeros(n);self.spring_gap=np.full(n,np.nan)
        self.enabled=np.zeros(n,int);self.broken=np.zeros(n,bool);self.load=np.zeros(n)
        self.binds=np.zeros(n,int);self.overloads=np.zeros(n,int);self.lost=np.zeros(n)
        self.rest=np.zeros((n,3));self.completion=np.full(n,np.nan)
        angle=float(np.random.default_rng(seed).uniform(0,2*math.pi))
        self.carry=.5*np.array([math.cos(angle),math.sin(angle),0.])
        self.angle=angle;self.carry_t=math.sqrt(2*math.pi*.5/2.)
        self.previous=np.tile(starts,(n,1,1));self.t=0.;self.step=0

    def candidate(self,positions,facts):
        tips=positions[:,1:3,2]-.020
        return self.eligible & (~self.broken) & (self.enabled==0) & \
               np.all(self.touch[:,None]-tips>=.003-1e-12,axis=1) & np.all(facts[:,:2]>0.,axis=1)

    def control(self,positions,raw,facts):
        old_phase=self.phase.copy();tau=self.t-self.pt;alive=self.code==0
        candidates=self.candidate(positions,facts)
        new_bind=np.zeros(self.n,bool)
        for p in range(8):
            mask=alive & (old_phase==p)
            if p==0:
                go=mask & (tau>=.3);self.rest[go]=positions[go,0];self.phase[go]=1
            elif p==1:
                hit=mask & np.any(np.linalg.norm(raw,axis=2)>.05,axis=1)
                self.touch[hit]=positions[hit,1:3,2].mean(1)-.020
                self.depth[hit]=np.minimum(.012,.6*np.maximum(self.touch[hit],0.))
                self.press[hit]=self.arm[hit,2];self.phase[hit]=2
                miss=mask & ~hit
                self.code[miss & (self.arm[:,2]<-self.h-.010)]=2
                self.arm[miss,2]=-.010*tau[miss]
            elif p==2:
                stop=mask & ((np.maximum(raw[:,:,2].sum(1),0.)>=20.) | (tau>=self.depth/.010))
                gap=positions[:,2,1]-positions[:,1,1]-.004
                self.spring_gap[stop & candidates]=gap[stop & candidates]
                self.phase[stop]=3
                move=mask & ~stop;self.arm[move,2]=self.press[move]-np.minimum(.010*tau[move],self.depth[move])
            elif p==3:
                stop=mask & ((self.ctrl>=19.) | (self.squeeze>=.030) | (tau>=8.))
                bind=stop & candidates & (self.current>0.)
                self.enabled[bind]=1;self.binds[bind]+=1;new_bind[bind]=True
                self.lift[bind]=self.arm[bind,2];self.phase[bind]=4
                self.code[stop & ~bind]=2
            elif p==4:
                self.arm[mask,2]=self.lift[mask]+np.minimum(.1*tau[mask],.250)
                finish=mask & (tau>=2.7)
                self.code[finish & (facts[:,3]>0.)]=2
                self.phase[finish & (facts[:,3]==0.)]=5
            elif p==5:
                f=np.minimum(tau/self.carry_t,1.)
                f=f-np.sin(2*math.pi*f)/(2*math.pi)
                self.arm[mask]=f[mask,None]*self.carry
                self.arm[mask,2]=self.lift[mask]+.250
                finish=mask & (tau>=self.carry_t+.3)
                drop=finish & (positions[:,0,2]-self.rest[:,2]<.030)
                self.code[drop]=3;self.phase[finish & ~drop]=6
            elif p==6:
                self.arm[mask]=self.carry
                self.arm[mask,2]=np.maximum(self.lift[mask]+.250-.2*tau[mask],self.lift[mask]+.005)
                opened=mask & ((facts[:,2]>=.3*self.mass*9.81) | (self.arm[:,2]<=self.lift+.005))
                self.enabled[opened]=0;self.spring_gap[opened]=np.nan
                self.release_squeeze[opened]=self.squeeze[opened];self.phase[opened]=7
            else:
                self.squeeze[mask]=self.release_squeeze[mask]-np.minimum(.05*tau[mask],.030)
                finish=mask & (tau>=1.1)
                offset=positions[:,0,:2]-self.rest[:,:2]-self.carry[:2]
                x=offset[:,0]*math.cos(self.angle)+offset[:,1]*math.sin(self.angle)
                y=-offset[:,0]*math.sin(self.angle)+offset[:,1]*math.cos(self.angle)
                ok=(np.abs(x)<=.100)&(np.abs(y)<=.100)
                self.code[finish & ok]=1;self.code[finish & ~ok]=3
        self.pt[self.phase!=old_phase]=self.t
        closing=(self.code==0)&(self.phase>=3)&(self.phase<=6)
        self.squeeze[closing]=np.clip(self.squeeze[closing]+np.clip(.0025*(20.-self.ctrl[closing]),-.025,.025)*.0004,0.,.030)
        targets=self.starts[None,:,:]+self.arm[:,None,:]
        targets[:,0,1]+=self.squeeze;targets[:,1,1]-=self.squeeze
        targets[self.code!=0]=self.previous[self.code!=0]
        velocity=(targets-self.previous)/.0004
        point=targets.mean(1)-[0,0,.020]
        previous=self.previous.copy();self.previous=targets
        return previous,velocity,point,new_bind

    def observe(self,positions,raw,facts,load):
        gap=positions[:,2,1]-positions[:,1,1]-.004
        spring=np.where(np.isfinite(self.spring_gap),40000.*np.maximum(self.spring_gap-.010-gap,0.),0.)
        real=(np.maximum(-raw[:,0,1],0.)+np.maximum(raw[:,1,1],0.))/2
        self.current=real+spring
        self.ctrl+=.0004/.0054*(self.current-self.ctrl)
        self.measured+=.0004/.0204*(self.current-self.measured)
        self.load+=.0004/.0054*(np.linalg.norm(load,axis=1)-self.load)
        broke=(self.enabled!=0)&(self.load>self.current)
        self.enabled[broke]=0;self.broken[broke]=True;self.spring_gap[broke]=np.nan;self.overloads[broke]+=1
        alive=self.code==0
        self.code[alive & (self.measured>50.)]=5
        self.t+=.0004;self.step+=1
        if self.step%25==0:
            held=(self.phase==4)|(self.phase==5)
            contact=np.maximum(np.where(self.enabled!=0,self.current,0.),facts[:,4:].max(axis=1))
            self.lost=np.where(held & (contact<.05),self.lost+.010,0.)
            lost=alive & held & (self.lost>.050)
            self.code[lost]=np.where(self.phase[lost]==5,3,2)
        self.code[(self.code==0)&(self.t>35.)]=2
        end=(self.code!=0)&np.isnan(self.completion);self.completion[end]=self.t


def run(nworld,with_bag,u,seed):
    check_deadline()
    sim,starts,h,bag_geom,eligible,mass=prepare(with_bag,u)
    ncon,njmax=128,512
    wm=mjw.put_model(sim.m)
    if with_bag: wm.eq_data=wp.array(np.tile(wm.eq_data.numpy(),(nworld,1,1)),dtype=wm.eq_data.dtype)
    wd=mjw.put_data(sim.m,sim.d,nworld=nworld,nconmax=ncon,naconmax=ncon*nworld,njmax=njmax)
    bd=BatchData();bd.qpos,bd.qvel=wd.qpos,wd.qvel
    c=Contacts();c.count=wd.nacon;c.geom=wd.contact.geom;c.frame=wd.contact.frame
    c.address=wd.contact.efc_address;c.world=wd.contact.worldid;c.dim=wd.contact.dim
    c.distance=wd.contact.dist;c.force=wd.efc.force
    lookup=np.full(sim.m.ngeom,-1,np.int32);package=np.zeros(sim.m.ngeom,np.int32)
    package[list(sim.obj_geoms)]=1
    if with_bag: package[bag_geom]=1
    for i,p in enumerate(sim.pad_names): lookup[list(sim.pad_geoms[p])]=i
    lookup=wp.array(lookup,dtype=int);package=wp.array(package,dtype=int)
    qa=wp.array(np.r_[sim.pad_q['pA'],sim.pad_q['pB']].astype(np.int32),dtype=int)
    va=wp.array(np.r_[sim.pad_v['pA'],sim.pad_v['pB']].astype(np.int32),dtype=int)
    p0=wp.array(np.r_[sim.pad_p0['pA'],sim.pad_p0['pB']].astype(np.float32),dtype=float)
    q=wp.zeros((nworld,6),dtype=float);v=wp.zeros((nworld,6),dtype=float)
    point=wp.zeros(nworld,dtype=wp.vec3);enabled=wp.zeros(nworld,dtype=int)
    spring=wp.zeros(nworld,dtype=float)
    pad_bodies=wp.array([sim.m.body(p).id for p in sim.pad_names],dtype=int)
    raw=wp.zeros((nworld,2,3),dtype=float);facts=wp.zeros((nworld,10),dtype=float)
    bits=wp.zeros(nworld,dtype=int);events=wp.zeros(nworld,dtype=int)
    def physics():
        mjw.step(wm,wd)
        wp.launch(overflow_count,nworld,inputs=[wd.overflow,bits,events])
        raw.zero_();facts.zero_()
        wp.launch(read_contacts,wd.naconmax,inputs=[c,lookup,package,sim.table_geom,bag_geom,raw,facts])
    q.assign(np.tile(starts.ravel(),(nworld,1)).astype(np.float32))
    wp.launch(set_gripper,nworld,inputs=[bd,q,v,qa,va,p0,wd.mocap_pos,point,wd.eq_active,enabled,spring,pad_bodies,wd.xfrc_applied])
    physics()
    with wp.ScopedCapture() as capture: physics()
    # Warmup is excluded; restore the cached state before the complete timed action.
    wd.qpos.assign(np.tile(sim.d.qpos,(nworld,1)).astype(np.float32))
    wd.qvel.assign(np.tile(sim.d.qvel,(nworld,1)).astype(np.float32))
    wd.qacc_warmstart.assign(np.tile(sim.d.qacc_warmstart,(nworld,1)).astype(np.float32))
    wd.time.fill_(0.);bits.zero_();events.zero_();wd.overflow.zero_()
    mjw.forward(wm,wd);wp.synchronize()
    control=TopControl(nworld,starts,h,eligible,mass,seed)
    bodies=[sim.seg_bodies[0],sim.m.body('pA').id,sim.m.body('pB').id]
    positions=wd.xpos.numpy()[:,bodies]
    rh=np.zeros((nworld,2,3));fh=np.zeros((nworld,10))
    anchor=wm.eq_data.numpy() if with_bag else None
    load=np.zeros((nworld,3));finite=True;escaped=np.zeros(nworld,bool)
    with MemoryPeak() as memory:
        t0=time.perf_counter()
        while not np.all(control.code!=0):
            if control.step%1000==0: check_deadline()
            old,velocity,target,bind=control.control(positions,rh,fh)
            if bind.any():
                rotation=wd.xmat.numpy()[:,sim.bag_bodies[0]].reshape(nworld,3,3)
                bag_pos=wd.xpos.numpy()[:,sim.bag_bodies[0]]
                anchor[bind,0,:3]=0.
                anchor[bind,0,3:6]=np.einsum('nji,nj->ni',rotation[bind],target[bind]-bag_pos[bind])
                wm.eq_data.assign(anchor)
                control.load[bind]=0.
            q.assign(old.reshape(nworld,6).astype(np.float32));v.assign(velocity.reshape(nworld,6).astype(np.float32))
            point.assign(target.astype(np.float32));enabled.assign(control.enabled.astype(np.int32))
            gap=positions[:,2,1]-positions[:,1,1]-.004
            sf=np.where(np.isfinite(control.spring_gap),40000.*np.maximum(control.spring_gap-.010-gap,0.),0.)
            spring.assign(sf.astype(np.float32))
            wp.launch(set_gripper,nworld,inputs=[bd,q,v,qa,va,p0,wd.mocap_pos,point,wd.eq_active,enabled,spring,pad_bodies,wd.xfrc_applied])
            wd.overflow.zero_();wp.capture_launch(capture.graph)
            positions=wd.xpos.numpy()[:,bodies];rh=raw.numpy();fh=facts.numpy()
            if with_bag: load=wd.efc.force.numpy()[:,:3]
            control.observe(positions,rh,fh,load)
            if control.step%25==0:
                finite=bool(np.isfinite(wd.qpos.numpy()).all())
                if with_bag:
                    rotation=wd.xmat.numpy()[:,sim.bag_bodies[0]].reshape(nworld,3,3)
                    bag_pos=wd.xpos.numpy()[:,sim.bag_bodies[0]]
                    local=np.einsum('nji,nj->ni',rotation,positions[:,0]-bag_pos)
                    outside=np.any(np.abs(local[:,:2])>np.asarray(sim.bag.size)/2,axis=1)
                    escaped|=outside & (control.code==0)
                    control.code[outside & (control.code==0)]=4
                if not finite or np.any(bits.numpy()!=0): break
        wp.synchronize();wall=time.perf_counter()-t0
    completed=int(np.count_nonzero(control.code))
    overflow=int(np.count_nonzero(bits.numpy()))
    valid=finite and overflow==0 and completed==nworld
    labels={0:'unfinished',1:'success',2:'grasp_fail',3:'drop',4:'sim_fail',5:'over_squeeze'}
    return dict(test='B4',content='box_loose',bag=with_bag,u=u,seed=seed,nworld=nworld,
                controller='vectorized NumPy host; transfers included',wall_s=wall,completed=completed,
                evals_per_s=completed/wall if valid else None,valid=valid,overflow_worlds=overflow,
                overflow_world_steps=int(events.numpy().sum()),finite=finite,
                escaped_worlds=int(escaped.sum()),
                outcomes={name:int((control.code==i).sum()) for i,name in labels.items()},
                binds=int(control.binds.sum()),overloads=int(control.overloads.sum()),
                peak_total_gpu_mib=memory.peak,sim_s_max=control.t,
                completion_sim_s=np.nanpercentile(control.completion,[0,50,100]).tolist(),
                ncon_per_world=ncon,njmax=njmax,cache_preparation_timed=False)


def batch(nworld,smoke=False):
    RESULTS.mkdir(parents=True,exist_ok=True)
    path=RESULTS/(f'b4_smoke_{nworld}.jsonl' if smoke else f'b4_{nworld}.jsonl')
    done=set()
    if path.exists():
        done={(r['bag'],r['u'],r['seed']) for r in map(json.loads,path.read_text().splitlines()) if r.get('valid')}
    grid=[(b,u,s) for b in [False,True] for u in [0.,.4] for s in (range(1) if smoke else range(3))]
    for b,u,seed in grid:
        if (b,u,seed) in done: continue
        try:
            result=run(nworld,b,u,seed)
        except Exception as error:
            append(path,dict(test='B4',content='box_loose',bag=b,u=u,seed=seed,nworld=nworld,
                             valid=False,evals_per_s=None,error=str(error)))
            raise
        append(path,result);print(json.dumps(result),flush=True)
        if not result['valid']: raise RuntimeError('invalid B4 batch; inspect before retrying')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--nworld',type=int,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    batch(a.nworld,a.smoke)


if __name__=='__main__': main()
