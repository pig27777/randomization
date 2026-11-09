"""Observe the unchanged CPU/GPU R3 controller, including native contact-force decoding.

Exploration only. No protocol, contact material, solver or frozen source is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

CONTENT = Path(__file__).resolve().parents[2] / "contents" / "segment"
sys.path.insert(0, str(CONTENT))
import time

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw

import mj_adapter as A
import model
import protocols as P
from mjwarp_backend import DiagnosticSim

ROOT=CONTENT


def hashes():
    names=('mj_adapter.py','model.py','protocols.py','metrics.py',
           'mjwarp_backend.py','warp_effects.py','warp_mechanics.py')
    sources = {n: ROOT / n for n in names}
    script = Path(__file__).resolve()
    sources[str(script.relative_to(script.parents[2]))] = script
    return {n: hashlib.sha256(path.read_bytes()).hexdigest() for n, path in sources.items()}


def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=1,allow_nan=False)


class SmokeEnd(Exception):
    pass


class Observer:
    def __init__(self,backend,maximum):
        self.backend=backend;self.maximum=maximum;self.rows=[];self.events=[];self.controller=None
        self.sim=None;self.last=-1.;self.native_ids=None

    def native_contacts(self,sim):
        """Independent official GPU decoder, including pyramidal edge-force synthesis."""
        d=sim.batch.data;m=sim.batch.model
        if self.native_ids is None:
            self.native_ids=wp.array(np.arange(d.naconmax,dtype=np.int32),device='cuda:0')
            self.native_force=wp.zeros(d.naconmax,dtype=wp.spatial_vector,device='cuda:0')
        mjw.contact_force(m,d,self.native_ids,False,self.native_force)
        count=int(d.nacon.numpy()[0])
        if count>d.naconmax:raise RuntimeError('native diagnostic contact buffer overflow')
        force=self.native_force.numpy()[:count]
        geom=d.contact.geom.numpy()[:count];address=d.contact.efc_address.numpy()[:count]
        dim=d.contact.dim.numpy()[:count];position=d.contact.pos.numpy()[:count]
        dist=d.contact.dist.numpy()[:count];frame=d.contact.frame.numpy()[:count]
        world=d.contact.worldid.numpy()[:count];efc=d.efc.force.numpy()[0]
        cone=int(m.opt.cone)
        rows=[]
        for i in np.flatnonzero(world==0):
            ndim=max(1,2*(int(dim[i])-1)) if cone==0 else int(dim[i])
            adr=address[i,:ndim];valid=adr>=0
            edges=np.where(valid,efc[np.where(valid,adr,0)],0.)
            synthesized=float(edges.sum() if cone==0 and dim[i]>1 else edges[0])
            rows.append(dict(index=int(i),geom=geom[i].tolist(),dim=int(dim[i]),
                addresses=adr.tolist(),row_forces=edges.tolist(),normal_synthesized_N=synthesized,
                native_force_local_N=force[i].tolist(),pos_m=position[i].tolist(),dist_m=float(dist[i]),
                frame=frame[i].reshape(-1).tolist()))
        return rows,cone

    def capture(self,sim,t):
        self.sim=sim
        dense=.345<=t<=.405
        if self.rows and not dense and t-self.last<.010-1e-10:return
        self.last=t;m,d=sim.m,sim.d
        contacts=[];normal={p:0. for p in sim.pad_names};direct={p:0. for p in sim.pad_names}
        f=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i];mujoco.mj_contactForce(m,d,i,f)
            nrows=max(1,2*(int(c.dim)-1)) if int(m.opt.cone)==0 else int(c.dim)
            a=int(c.efc_address)
            rows=d.efc_force[a:a+nrows] if a>=0 else np.zeros(nrows)
            direct_value=float(d.efc_force[a]) if a>=0 else 0.
            for p,geoms in sim.pad_geoms.items():
                if int(c.geom[0]) in geoms or int(c.geom[1]) in geoms:
                    normal[p]+=float(f[0]*abs(c.frame[1]));direct[p]+=float(direct_value*abs(c.frame[1]))
            contacts.append(dict(index=i,geom=c.geom.tolist(),names=[m.geom(int(g)).name for g in c.geom],
                dim=int(c.dim),efc_address=a,row_forces=rows.tolist(),direct_first_row_N=direct_value,
                normal_synthesized_N=float(rows.sum() if int(m.opt.cone)==0 and c.dim>1 else rows[0]),
                contact_force_local_N=f.tolist(),pos_m=c.pos.tolist(),dist_m=float(c.dist),frame=c.frame.tolist()))
        ids=sorted(sim.obj_geoms)
        half=m.geom_size[ids,:3]
        span=np.einsum('nij,nj->ni',np.abs(d.geom_xmat[ids].reshape(-1,3,3)),half)
        width=float(np.max(d.geom_xpos[ids,1]+span[:,1])-np.min(d.geom_xpos[ids,1]-span[:,1]))
        pads=np.array([sim.pad_pos(p) for p in sim.pad_names]);gap=float(pads[1,1]-pads[0,1]-2*A.PAD_T)
        ctrl=self.controller
        row=dict(t=t,pad_positions_m=pads.tolist(),pad_qpos=d.qpos[sim._qadr].tolist() if self.backend=='gpu' else
            d.qpos[np.concatenate([sim.pad_q[p] for p in sim.pad_names])].tolist(),
            pad_gap_y_m=gap,object_width_y_m=width,object_compression_m=model.ANCHORS['hard'].size[1]-width,
            pad_indentation_m=width-gap,legacy_normal_y_N={p:sim.pad_normal_force(p,1) for p in sim.pad_names},
            independent_normal_y_N=normal,direct_first_rows_y_N=direct,com_m=sim.com().tolist(),
            cone_cpu_model=int(m.opt.cone),cone_cpu_name=mujoco.mjtCone(int(m.opt.cone)).name,
            ncon=int(d.ncon),nefc=int(d.nefc),solver_niter=int(d.solver_niter[0]),contacts=contacts,
            closer=dict(frozen=ctrl.frozen,t_touch=ctrl.t_touch,squeeze_m=ctrl.s_sq,
                        hold_squeeze_m=ctrl.hold_sq,filtered_force_N=dict(ctrl.f)))
        if self.backend=='gpu':row['native_contacts'],row['cone_gpu_model']=self.native_contacts(sim)
        self.rows.append(row)

    def run(self):
        observer=self
        old_sim,old_runner,old_closer=P.Sim,P.Runner,P.Closer
        class Runner(old_runner):
            def step(self,*a,**kw):
                rec=super().step(*a,**kw)
                observer.capture(self.sim,self.t)
                if observer.maximum is not None and self.t>=observer.maximum:raise SmokeEnd()
                return rec
        class Closer(old_closer):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw);observer.controller=self
            def __call__(self,dt,forces,t):
                frozen=self.frozen
                target=super().__call__(dt,forces,t)
                if self.frozen and not frozen:observer.events.append(dict(event='close_stop',t=t,forces_N=dict(forces),
                    filtered_force_N=dict(self.f),squeeze_m=self.s_sq,targets_m={p:x.tolist() for p,x in target.items()}))
                return target
        P.Sim=DiagnosticSim if self.backend=='gpu' else A.Sim
        P.Runner,P.Closer=Runner,Closer
        start=time.perf_counter();result=None
        try:result=P.r3(model.ANCHORS['hard'])
        except SmokeEnd:result=dict(incomplete=True,reason='explicit short exploration limit')
        finally:P.Sim,P.Runner,P.Closer=old_sim,old_runner,old_closer
        return dict(backend=self.backend,wall_s=time.perf_counter()-start,result=result,events=self.events,rows=self.rows)


def summarize(runs):
    out={}
    for name,r in runs.items():
        result=r['result'];ev=result.get('events',{})
        pre=[x for x in r['rows'] if ev.get('pre',np.inf)<=x['t']<ev.get('move',-np.inf)]
        out[name]=dict(result=result,close_stop=r['events'],samples=len(r['rows']),
            max_decoder_difference_N=max((abs(x['normal_synthesized_N']-x['contact_force_local_N'][0])
                for row in r['rows'] for x in row['contacts']),default=0.))
        if pre:
            out[name]['pre_means']=dict(pad_gap_y_m=float(np.mean([x['pad_gap_y_m'] for x in pre])),
                pad_indentation_m=float(np.mean([x['pad_indentation_m'] for x in pre])),
                object_compression_m=float(np.mean([x['object_compression_m'] for x in pre])),
                independent_force_N=float(np.mean([sum(x['independent_normal_y_N'].values())/2 for x in pre])),
                legacy_force_N=float(np.mean([sum(x['legacy_normal_y_N'].values())/2 for x in pre])),
                contact_count=float(np.mean([x['ncon'] for x in pre])))
        if name=='gpu':
            out[name]['max_native_export_force_difference_N']=max((
                abs(x['normal_synthesized_N']-x['native_force_local_N'][0])
                for row in r['rows'] for x in row['native_contacts']),default=0.)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--backend',choices=['cpu','gpu','both'],default='both')
    ap.add_argument('--max-time',type=float)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    before=hashes();runs={}
    for backend in ('cpu','gpu') if args.backend=='both' else (args.backend,):
        runs[backend]=Observer(backend,args.max_time).run()
        write(args.output/f'{backend}.json',runs[backend])
        print(json.dumps(dict(backend=backend,wall_s=runs[backend]['wall_s'],result=runs[backend]['result'])),flush=True)
    after=hashes()
    write(args.output/'summary.json',dict(exploration=True,formal=False,arguments=dict(backend=args.backend,max_time=args.max_time),
        versions=dict(mujoco=mujoco.__version__,warp=wp.__version__,mjwarp=mjw.__version__),
        source_sha256=before,source_unchanged=before==after,summary=summarize(runs)))
    if before!=after:raise RuntimeError('source changed during diagnostic')


if __name__=='__main__':main()
