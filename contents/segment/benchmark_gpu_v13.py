"""G5 reference throughput/profiling; run each measurement under the shared flock.

Device activity samples are separate from additive host-wall categories: downloads
include synchronization waits, so device execution time must not be added again.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parent


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=1,allow_nan=False)


def prepare(n,category,seed):
    from model import ANCHORS
    from sampling import sample_kappa
    rng=np.random.default_rng(seed);specs=[]
    for i in range(n):
        k=sample_kappa(rng,'mixed') if category=='random' else dataclasses.replace(ANCHORS[category])
        if category=='random':
            g=dict(u=float(rng.uniform(-.5,.5)),v=float(rng.uniform(-.5,.5)),psi=float(rng.uniform(0,np.pi)),
                w=float(rng.uniform(.020,.095)),d=float(rng.uniform(0,.040)),F=float(rng.uniform(2,50)),
                v_lift=float(rng.uniform(.05,.3)),a_max=float(rng.uniform(.5,4)))
        else:g=dict(w=.04,d=.015,F=10.)
        specs.append(dict(kappa=k.to_dict(),grasp=g,seed=seed+i,label=dict(category=k.category,index=i)))
    return dict(category=category,seed=seed,n=n,specs=specs)


class Memory:
    """Process sample peak, plus Warp's allocator high-water mark at completion."""
    def __init__(self):self.stop=threading.Event();self.values=[];self.errors=[]
    def sample(self):
        while not self.stop.is_set():
            r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],
                text=True,capture_output=True)
            if r.returncode:self.errors.append(r.stderr.strip());return
            for line in r.stdout.splitlines():
                pid,used=(s.strip() for s in line.split(','))
                if pid==str(os.getpid()) and used.isdigit():self.values.append(int(used))
            self.stop.wait(.2)
    def __enter__(self):
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*unused):self.stop.set();self.thread.join()
    def result(self):
        return dict(process_sample_peak_MiB=max(self.values) if self.values else None,
            sample_interval_s=.2,samples=len(self.values),errors=self.errors,
            caveat='Process peak is sampled, not an exact lifetime peak; Warp pool high-water excludes other allocators.')


@wp.kernel
def count_iterations(niter:wp.array[int],nefc:wp.array[int],nacon:wp.array[int],
                     histogram:wp.array[int],overflow:wp.array[int]):
    i=wp.tid();v=niter[i]
    if v<0 or v>=histogram.shape[0]:wp.atomic_add(overflow,0,1)
    else:wp.atomic_add(histogram,v,1)
    wp.atomic_max(overflow,1,nefc[i])
    if i==0:wp.atomic_max(overflow,2,nacon[0])


class Profile:
    def __init__(self,activity):
        self.activity=activity;self.depth=0;self.methods_s=0.;self.launch_s=0.;self.download_s=0.
        self.launch_calls=0;self.download_calls=0;self.step_calls=0;self.first_step_s=None
        self.device_samples=0;self.device_s=defaultdict(float);self.method_s=defaultdict(float)

    def attach(self,batch):
        hist=wp.zeros(101,dtype=int,device='cuda:0');overflow=wp.zeros(3,dtype=int,device='cuda:0')
        for name in ('step','promote','failures','overflowed','candidates','bind','release'):
            original=getattr(batch,name)
            def observed(*args,_name=name,_original=original,**kwargs):
                outer=self.depth==0;self.depth+=1;start=time.perf_counter()
                try:
                    if _name=='step' and self.activity and self.step_calls and self.step_calls%1000==0:
                        timer=wp.ScopedTimer('G5 device activity',cuda_filter=wp.TIMING_KERNEL,print=False)
                        with timer:value=_original(*args,**kwargs)
                        self.device_samples+=1
                        for result in timer.timing_results:self.device_s[result.name]+=result.elapsed/1000
                    else:value=_original(*args,**kwargs)
                    if _name=='step':
                        wp.launch(count_iterations,dim=batch.nworld,inputs=[batch.data.solver_niter,
                            batch.data.nefc,batch.data.nacon,hist,overflow])
                        self.step_calls+=1
                        if self.first_step_s is None:self.first_step_s=time.perf_counter()-start
                    return value
                finally:
                    self.depth-=1
                    if outer:
                        elapsed=time.perf_counter()-start;self.methods_s+=elapsed;self.method_s[_name]+=elapsed
            setattr(batch,name,observed)
        return hist,overflow

    def __enter__(self):
        if not self.activity:return self
        self.original_launch=wp.launch;self.original_numpy=wp.array.numpy
        def launch(*args,**kwargs):
            if not self.depth:return self.original_launch(*args,**kwargs)
            start=time.perf_counter()
            try:return self.original_launch(*args,**kwargs)
            finally:self.launch_s+=time.perf_counter()-start;self.launch_calls+=1
        def numpy(array,*args,**kwargs):
            if not self.depth:return self.original_numpy(array,*args,**kwargs)
            start=time.perf_counter()
            try:return self.original_numpy(array,*args,**kwargs)
            finally:self.download_s+=time.perf_counter()-start;self.download_calls+=1
        wp.launch=launch;wp.array.numpy=numpy;return self

    def __exit__(self,*unused):
        if self.activity:wp.launch=self.original_launch;wp.array.numpy=self.original_numpy

    def result(self,wall):
        result=dict(run_wall_s=wall,host_numpy_state_machine_and_orchestration_s=max(wall-self.methods_s,0.),
            host_launch_api_s=self.launch_s,host_readback_and_gpu_wait_s=self.download_s,
            host_stepper_other_s=self.methods_s-self.launch_s-self.download_s,
            method_wall_s=dict(self.method_s),launch_calls=self.launch_calls,readback_calls=self.download_calls,
            physics_step_calls=self.step_calls,first_step_wall_s=self.first_step_s,
            sampled_device_steps=self.device_samples,sampled_device_kernel_s=sum(self.device_s.values()),
            device_kernel_samples_s=dict(sorted(self.device_s.items(),key=lambda x:-x[1])[:40]),
            notes=['Four host-wall categories form an additive partition; readback includes pending GPU execution.',
                'Device kernel activity times overlap host-wall categories and must not be added to them.',
                'Launch API includes Python/driver cost and possible backpressure, not solely hardware launch latency.',
                'Numpy state-machine wall is outside Batch methods; CUDA activity profiling is separately identified.',
                'Iteration histogram observes all actual world-steps, including completed worlds still stepped until batch end.'])
        result['host_breakdown_instrumented']=self.activity
        if not self.activity:
            for k in ('host_launch_api_s','host_readback_and_gpu_wait_s','host_stepper_other_s'):result[k]=None
        return result


def cpu_one(spec):
    import validate_warp_v13 as V
    return V.cpu_evaluate(spec)


def run_gpu(specs,activity):
    import mjwarp_v13 as G
    outputs=[None]*len(specs);profiles=[];initialization=0.;pool_peak=0
    for key,indices in G.buckets(specs).items():
        start=time.perf_counter();batch=G.Batch([specs[i] for i in indices])
        bucket_init=time.perf_counter()-start;initialization+=bucket_init
        profile=Profile(activity);hist,overflow=profile.attach(batch)
        start=time.perf_counter()
        with profile:result=G.run(batch)
        wp.synchronize();elapsed=time.perf_counter()-start
        observed_limits=overflow.numpy()
        if observed_limits[0]:raise RuntimeError('solver_niter outside histogram capacity')
        counts=hist.numpy();observed=np.flatnonzero(counts)
        iterations=dict(histogram=counts.tolist(),minimum=int(observed.min()),maximum=int(observed.max()),
            mean=float(np.dot(np.arange(len(counts)),counts)/counts.sum()),world_steps=int(counts.sum()),
            at_iteration_cap_fraction=float(counts[int(batch.model.opt.iterations)]/counts.sum()))
        pool_peak=max(pool_peak,int(wp.get_mempool_used_mem_high('cuda:0')))
        profiles.append(dict(bucket=str(key),n=len(indices),initialization_excluded_s=bucket_init,
            peak_nefc=int(observed_limits[1]),peak_nacon=int(observed_limits[2]),
            capacity_nefc=batch.njmax,capacity_nacon=batch.data.naconmax,
            timing=profile.result(elapsed),solver_niter=iterations))
        for i,r in zip(indices,result):outputs[i]=r
    return outputs,dict(buckets=profiles,initialization_s=initialization,warp_pool_high_water_bytes=pool_peak,
        cuda_activity_profile=activity,iteration_observer='one passive histogram kernel per physical batch step')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','gpu','cpu'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--input',type=Path)
    ap.add_argument('--n',type=int,default=64);ap.add_argument('--seed',type=int,default=20261204)
    ap.add_argument('--category',choices=['hard','bendable','compressible','random'],default='bendable')
    ap.add_argument('--profile',action='store_true')
    a=ap.parse_args()
    if a.n<1:ap.error('--n must be positive')
    if a.output.exists():ap.error('refusing to overwrite output')
    if a.action=='prepare':save(a.output,prepare(a.n,a.category,a.seed));return
    if a.input is None:ap.error('measurement requires --input')
    from model import Kappa
    document=json.loads(a.input.read_text());specs=[dict(s,kappa=Kappa.from_dict(s['kappa'])) for s in document['specs']]
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    start=time.perf_counter();memory=None
    if a.action=='gpu':
        with Memory() as memory:result,details=run_gpu(specs,a.profile)
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(28) as pool:result=pool.map(cpu_one,specs)
        details=dict(workers=28)
    wall=time.perf_counter()-start;counts=Counter(r['outcome'] for r in result)
    valid=len(result)-counts['sim_fail'];overflows=sum(r.get('overflow_checks',0) for r in result)
    payload=dict(backend=a.action,n=len(specs),category=document['category'],wall_s=wall,
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),outcomes=dict(counts),
        completed_valid_grasps_per_s=valid/wall,returned_attempts_per_s=len(result)/wall,
        successful_grasps_per_s=counts['success']/wall,sim_fail=counts['sim_fail'],overflow_checks=overflows,
        profile=details,memory=memory.result() if memory else None,results=result,
        source_sha256=before,cached_rest=False,early_termination=True,
        initialization_in_wall=True,formal_queue_required=True)
    used=[]
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.parent==ROOT:
                if p.name not in before or hashlib.sha256(p.read_bytes()).hexdigest()!=before[p.name]:
                    raise RuntimeError(f'source changed during benchmark: {p.name}')
                used.append(p.name)
    payload['checked_imported_sources']=sorted(set(used));payload['sources_unchanged']=True
    if a.action=='gpu':
        payload['observed_capacity_overflow']=any(b['peak_nefc']>b['capacity_nefc'] or b['peak_nacon']>b['capacity_nacon']
            for b in details['buckets'])
    save(a.output,payload)
    print(json.dumps({k:payload[k] for k in ('backend','n','wall_s','outcomes','completed_valid_grasps_per_s',
        'successful_grasps_per_s','overflow_checks')},indent=2),flush=True)
    if overflows or payload.get('observed_capacity_overflow'):raise RuntimeError('GPU contact/constraint overflow; measurement invalid')


if __name__=='__main__':main()
