"""Round 3 B1-B4 effect-model checks; import only the verified v1.3 content archive."""
import argparse
import json
import math
import time
from dataclasses import asdict
from multiprocessing import Pool

import mujoco
import numpy as np

from pathlib import Path
import sys

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

from round3_flex import OUT, append, content_kappa
from loose_protocols import make, r2, r3
from effect_model import EffectBag
from film_grasp import evaluate_film

RESULTS=OUT/'effect'


def bag_for(k):
    return EffectBag(size=(.35,.25)) if k.rigid else EffectBag(fill=.75)


def settle(name):
    k=content_kappa(name)
    sim=make(k,bag_for(k),(),dt=.0004)
    violations=outside=0.
    failure=None
    for i in range(round(.3/sim.m.opt.timestep)):
        sim.step()
        if i%25==0:
            violations=max(violations,sim.rope_violation())
            outside=max(outside,sim.outside_fraction()[0])
            failure=sim.failure() or failure
    return dict(test='B1_cpu',content=name,bag=asdict(sim.bag),finite=bool(np.isfinite(sim.d.qpos).all()),
                failure=failure,centres_outside_max=outside,corners_outside_end=sim.outside_fraction()[1],
                rope_limit_excess_mm=violations*1e3,nv=sim.m.nv,ntendon=sim.m.ntendon,
                max_ncon=sim.max_ncon,max_nefc=sim.max_nefc)


def one(job):
    test,name,u,seed=job
    k=content_kappa(name);bag=bag_for(k)
    if test=='B1_cpu': return settle(name)
    if test in ('R2','R3'):
        result=(r2 if test=='R2' else r3)(name,bag,dt=.0004,**({'end':1} if test=='R2' else {}))
    elif test=='side':
        size=bag.resolved(k.size).size
        if min(size)>.085:
            return dict(test='B3_side',content=name,seed=seed,outcome='action_invalid',
                        reason='bag closing dimension exceeds protocol v2.2 85 mm side-grasp range',bag_size=size)
        raise RuntimeError('this B3 grid expected oversized bags; add the valid side executor before changing the grid')
    else:
        params=[u,0.,math.pi/2,.060,.012,20. if k.rigid else 10.,.1,2.]
        result=evaluate_film(k,bag,params,bag.bag_pinch_coeff,seed)
    # The carrier has no membrane; the common executor's zero is not a measured film penetration.
    result['penetration_mm']=None
    result['membrane_penetration_applicable']=False
    return dict(result,test=test,content=name,u=u,seed=seed)


def gpu_settle():
    import warp as wp
    import mujoco_warp as mjw
    from flex_batch import overflow_count
    sim=make(content_kappa('box_loose'),EffectBag(size=(.35,.25)),(),dt=.0004)
    wm=mjw.put_model(sim.m)
    wd=mjw.put_data(sim.m,sim.d,nworld=1,nconmax=max(64,sim.max_ncon*2),njmax=max(256,sim.max_nefc*2))
    bits=wp.zeros(1,dtype=int);events=wp.zeros(1,dtype=int)
    def step():
        wd.overflow.zero_()
        mjw.step(wm,wd)
        wp.launch(overflow_count,1,inputs=[wd.overflow,bits,events])
    step()
    with wp.ScopedCapture() as cap: step()
    finite=True;failure=None
    centres=corners=violation=0.
    executed=2
    while executed<750:
        count=min(25,750-executed)
        for _ in range(count): wp.capture_launch(cap.graph)
        executed+=count
        mjw.get_data_into(sim.d,sim.m,wd,0)
        finite=finite and bool(np.isfinite(sim.d.qpos).all())
        failure=sim.failure() or failure
        a,b=sim.outside_fraction()
        centres=max(centres,a);corners=max(corners,b)
        violation=max(violation,sim.rope_violation())
    wp.synchronize()
    return dict(test='B1_gpu',content='box_loose',finite=finite,failure=failure,
                centres_outside_max=centres,corners_outside_max=corners,
                rope_limit_excess_mm=violation*1e3,overflow_bits=bits.numpy().tolist(),
                overflow_world_steps=int(events.numpy().sum()),sample_interval_s=.01,
                sim_s=.3,throughput_measurement=False)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('mode',choices=['b1_cpu','b1_gpu','b2','b3','b4'])
    ap.add_argument('--procs',type=int,default=8)
    ap.add_argument('--top-middle',action='store_true',help='rerun B3 content-target cases after the envelope-contact fix')
    ap.add_argument('--nworld',type=int,choices=[256,1024])
    a=ap.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    if a.top_middle and a.mode!='b3': ap.error('--top-middle is a B3 correction only')
    if a.mode=='b4':
        if a.nworld is None: ap.error('B4 requires --nworld 256 or 1024')
        from effect_batch import batch
        batch(a.nworld)
        return
    path=RESULTS/(('b3_content_contact_fix' if a.top_middle else a.mode)+'.jsonl')
    if path.exists(): raise FileExistsError(path)
    if a.mode=='b1_gpu':
        append(path,gpu_settle())
        for content in ['bendable','compressible']:
            append(path,dict(test='B1_gpu',content=content,status='blocked',
                            reason='verified v1.3 release has no accepted GPU content backend; Python bushing forces cannot be omitted'))
        return
    names=['box_loose','bendable','compressible']
    if a.mode=='b1_cpu': jobs=[('B1_cpu',c,0.,0) for c in names]
    elif a.mode=='b2':
        jobs=[('R2','box_loose',0.,0),('R3','box_loose',0.,0)]
        jobs+=[('B2_empty_top','box_loose',.4,s) for s in range(3)]
    else:
        jobs=[('B3_top',c,u,s) for c in names for u in [0.,.4] for s in range(3)]
        jobs+=[('side',c,0.,s) for c in names for s in range(3)]
        if a.top_middle: jobs=[j for j in jobs if j[0]=='B3_top' and j[2]==0.]
    with Pool(a.procs) as pool:
        for result in pool.imap_unordered(one,jobs):
            append(path,result)
            print(json.dumps({k:result.get(k) for k in ['test','content','u','seed','outcome','failure','fail']}),flush=True)


if __name__=='__main__': main()
