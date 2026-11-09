"""Behavior matrix or locked CPU throughput with the same grasp protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import platform
import time
from angles import execute

def worker(job):
    backend,options,output=job
    result=execute(backend,options,output)
    return dict(output=str(output),outcome=result['outcome'],sim_time_s=result.get('sim_time_s',0.),
                wall_s=result.get('wall_s'),config=result['config'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('behavior','speed'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',default='native_flex')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--trials',type=int,default=28)
    parser.add_argument('--locked',action='store_true',help='Launch speed via flock /tmp/claude-gpu.lock')
    args=parser.parse_args()
    if args.mode=='speed' and not args.locked:parser.error('speed requires --locked and external flock')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.mode=='behavior':
        conditions=[dict(category=c,yaw=y,tilt=t) for c in ('bendable','compressible')
                    for y,t in ((0,0),(45,0),(90,0),(135,0),(0,-30),(0,30))]
        conditions += [dict(category=c,pinch_coeff=0.) for c in ('bendable','compressible')]
    else:conditions=[dict(category='compressible') for _ in range(args.trials)]
    (args.output/'jobs.json').write_text(json.dumps(conditions,indent=2))
    jobs=[(args.backend,opts,args.output/f'trial_{i:03d}') for i,opts in enumerate(conditions)]
    started=time.perf_counter()
    if args.workers==1:results=[worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(worker,jobs))
    wall=time.perf_counter()-started
    counts=Counter(r['outcome'] for r in results)
    complete=[r for r in results if r['outcome'] not in ('setup_fail','execution_error','compute_limit','incomplete','smoke_only')]
    summary=dict(hostname=platform.node(),backend=args.backend,mode=args.mode,workers=args.workers,
                 trials=len(results),outcomes=dict(counts),wall_s=wall,results=results,
                 timing_is_throughput=args.mode=='speed',lock='/tmp/claude-gpu.lock' if args.locked else None)
    if args.mode=='speed':
        summary.update(completed_evaluations_per_s=len(complete)/wall,successful_grasps_per_s=counts['success']/wall,
                       aggregate_simulated_seconds_per_wall_s=sum(r['sim_time_s'] for r in results)/wall)
        if args.workers==1:summary['single_environment_realtime_ratio']=sum(r['sim_time_s'] for r in results)/wall
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
