"""Distinguish loading transients from persistent motion before L5 grasp acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import warp as wp

from package_scene import PackageScene
from package_metrics import geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scale',type=float,required=True)
    parser.add_argument('--mass-scale',type=float,default=1.)
    parser.add_argument('--duration',type=float,default=8.)
    parser.add_argument('--dt',type=float)
    args=parser.parse_args()
    if args.output.exists():parser.error('refusing to overwrite output')
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.glob('package_*.py')}
    start=time.perf_counter();scene=PackageScene(scale=args.scale,mass_scale=args.mass_scale,dt=args.dt)
    graph=None;trace=[];failure=None;dt=.005
    for i in range(round(args.duration/dt)):
        scene.frame(graph)
        if i==19:
            snap={k:getattr(scene.s0,k).numpy() for k in ('body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd')}
            failures=scene.failure_metrics.numpy().copy()
            scene.frame();wp.synchronize()
            for state in (scene.s0,scene.s1):
                for k,v in snap.items():getattr(state,k).assign(v)
            scene.solver.reset(scene.s0,flags=0)
            scene.failure_metrics.assign(failures)
            with wp.ScopedCapture() as capture:
                for _ in range(scene.substeps):scene.step()
            graph=capture.graph
            for state in (scene.s0,scene.s1):
                for k,v in snap.items():getattr(state,k).assign(v)
            scene.solver.reset(scene.s0,flags=0)
            scene.failure_metrics.assign(failures)
        if (i+1)%20:continue
        state=scene.state_report();entry=dict(t=(i+1)*dt,**state)
        entry['simulation_failure_metrics']=scene.failure_metrics.numpy().tolist()
        if scene.failure_reason():
            failure=scene.failure_reason();trace.append(entry);break
        if not state['finite']:
            failure='non-finite state';trace.append(entry);break
        body=scene.s0.body_q.numpy();cloth=scene.s0.particle_q.numpy()
        entry.update(geometry(scene,body,cloth,penetration=True))
        entry['cloth_kinetic_J']=float(np.sum(scene.masses*np.sum(scene.s0.particle_qd.numpy()**2,axis=1))/2)
        entry['content_linear_speed_m_s']=float(np.linalg.norm(scene.s0.body_qd.numpy()[scene.box,:3]))
        trace.append(entry)
        if entry['centres_outside']>0:failure='content centre outside bag';break
        if (i+1)%200==0:print(json.dumps(entry),flush=True)
    wp.synchronize()
    result=dict(arguments=vars(args)|{'output':str(args.output)},wall_with_initialization_s=time.perf_counter()-start,
        failure=failure,trace=trace,source_sha256=source_sha256,dt=scene.dt,
        note='rest diagnostic, not a grasp or L6 throughput measurement')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    if failure:raise SystemExit(3)


if __name__=='__main__':main()
