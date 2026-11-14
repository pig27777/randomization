"""Reproducible material fixtures, stability and performance measurements.

Run from repository root with the mjwarp environment and the GPU lock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import mujoco
import numpy as np
import warp as wp
from contents.parcel import Batch


def snapshot(b):
    d = b.data
    p,s = b.params.numpy(),b.states.numpy()
    return dict(qpos=d.qpos.numpy(),qvel=d.qvel.numpy(),xpos=d.xpos.numpy(),
                params=p,state=s,normal=b.normal.numpy(),forces=d.xfrc_applied.numpy())


def run(b, seconds, bag_bind=False):
    samples=[]
    # Phase changes occur between graph replays, never inside a device step.
    graphs={}
    for close,bind in ((0,0),(1,int(bag_bind))):
        graphs[close,bind]=b.graph(close=close,bind=bind)
    b.reset()
    stride=max(1,round(.02/b.dt))
    for step in range(round(seconds/b.dt)):
        close=int(step*b.dt>=.8)
        bind=int(bag_bind and step*b.dt>=.8)
        key=(close,bind)
        wp.capture_launch(graphs[key])
        if step%stride==0:
            state=snapshot(b)
            state['time']=step*b.dt
            samples.append(state)
    wp.synchronize()
    return samples


def fixtures(out, seconds=2.0, n=8, dt=.0005):
    results=[]
    for prototype in range(6):
        for action,level,label in ((0,0,'static'),(1,0,'compression'),(2,1,'bag'),(2,2,'engagement'),(2,1,'eccentric')):
            b=Batch(n,rigid=prototype<2,prototype=prototype,action=action,level=level,
                    dt=dt,eccentric=.04 if label=='eccentric' else 0)
            samples=run(b,seconds,bag_bind=level==1)
            final=samples[-1]
            finite=all(np.isfinite(x['qpos']).all() and np.isfinite(x['qvel']).all() for x in samples)
            ids=b.ids.numpy()
            core=ids[:1 if b.rigid else 3,0]
            caps=ids[:1 if b.rigid else 3,9]
            velocity=float(np.max(np.abs(final['qvel'])))
            initial=samples[0]['xpos'][:,core,2].mean(axis=1)
            rise=float((final['xpos'][:,core,2].mean(axis=1)-initial).mean())
            depth=0.0 if b.rigid else float(max(np.max(-x['qpos'][:,caps]) for x in samples))
            row=dict(prototype=prototype,test=label,finite=bool(finite),max_velocity=velocity,
                     rise=rise,compression=depth,modes=final['state']['mode'].tolist(),
                     reasons=final['state']['reason'].tolist(),
                     max_tension=float(max(np.max(x['state']['tension']) for x in samples)))
            results.append(row)
            np.savez_compressed(out/f'p{prototype}-{label}.npz',
                time=[x['time'] for x in samples],qpos=[x['qpos'] for x in samples],
                xpos=[x['xpos'] for x in samples],normal=[x['normal'] for x in samples],
                state=[x['state'] for x in samples],params=final['params'])
            print(json.dumps(row),flush=True)
            del b
    return results


def benchmark(out, sizes, steps):
    results=[]
    for rigid in (True,False):
        for n in sizes:
            for level in (0,1,2):
                b=Batch(n,rigid=rigid,level=level)
                graph=b.graph()
                for _ in range(20):
                    wp.capture_launch(graph)
                wp.synchronize()
                begin=time.perf_counter()
                for _ in range(steps):
                    wp.capture_launch(graph)
                wp.synchronize()
                elapsed=time.perf_counter()-begin
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
                row=dict(rigid=rigid,n=n,level=level,steps=steps,seconds=elapsed,
                         physics_steps_s=steps/elapsed,environment_steps_s=steps*n/elapsed,
                         device_memory_used_mib=memory,warp_pool_used_bytes=wp.get_mempool_used_mem_current(),
                         warp_pool_peak_bytes=wp.get_mempool_used_mem_high(),cuda_graph=True,workload="unengaged resting",finite=bool(np.isfinite(b.data.qpos.numpy()).all()))
                results.append(row)
                print(json.dumps(row),flush=True)
                del b,graph
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=['fixtures','benchmark','stress'],default='fixtures')
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--dt',type=float,default=.0005)
    parser.add_argument('--sizes',default='1,64,512,2048,4096')
    parser.add_argument('--steps',type=int,default=200)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    meta=dict(machine=platform.node(),mujoco=mujoco.__version__,warp=wp.__version__,dt=args.dt,
              gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('contents/parcel').glob('*') if p.is_file()})
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    if args.mode=='fixtures':
        result=fixtures(args.output,args.seconds,args.n,args.dt)
    elif args.mode=='benchmark':
        result=benchmark(args.output,[int(n) for n in args.sizes.split(',')],args.steps)
    else:
        result=[]
        for rigid in (True,False):
            b=Batch(args.n,rigid=rigid,action=2,dt=args.dt)
            samples=run(b,args.seconds)
            q=np.array([x['qpos'] for x in samples])
            failed=~np.isfinite(q).all(axis=(0,2))
            row=dict(rigid=rigid,n=args.n,seconds=args.seconds,failed=int(failed.sum()),
                     failure_rate=float(failed.mean()),max_abs_qpos=float(np.max(np.abs(q))),
                     modes=samples[-1]['state']['mode'].tolist())
            result.append(row)
            print(json.dumps(row),flush=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
