"""Resident GPU timing of frozen segmented contents on the shared YAM scene.

Reuse the validated material/compaction stepper, with no host controller in timing.
Run in .venv (MuJoCo/MJWarp 3.17), under flock /tmp/claude-gpu.lock.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "test/flex3d"), str(ROOT)]
import segmented
import mj_adapter as A
sys.path.append(str(ROOT / "contents/segment"))  # GPU operators absent from the frozen CPU archive.


def working_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "contents/segment" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = working_module("segmented_benchmark_backend", "mjwarp_backend.py")
K = working_module("segmented_benchmark_metrics", "warp_v13.py")


@wp.kernel
def health(q: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
           failure: wp.array2d(dtype=wp.float64), raw_overflow: wp.array(dtype=int),
           packed_overflow: wp.array(dtype=int), nefc: wp.array(dtype=int),
           nacon: wp.array(dtype=int), iterations: wp.array(dtype=int),
           njmax: int, naconmax: int, cap: int, output: wp.array2d(dtype=wp.float64)):
    w = wp.tid()
    flags = int(0)
    for i in range(q.shape[1]):
        if not wp.isfinite(q[w, i]): flags = flags | 1
    for i in range(v.shape[1]):
        if not wp.isfinite(v[w, i]): flags = flags | 1
    if failure[w,3] == wp.float64(0.): flags = flags | 1
    if failure[w,0] > wp.float64(.5) or failure[w,1] > wp.float64(1.2) or failure[w,2] > wp.float64(1.2):
        flags = flags | 4
    if (raw_overflow[w] & ~1024) != 0 or packed_overflow[0] != 0 or nefc[w] > njmax or nacon[0] > naconmax:
        flags = flags | 2
    if (raw_overflow[w] & 1024) != 0: flags = flags | 16
    if iterations[w] >= cap: flags = flags | 8
    output[w,0] = wp.float64(int(output[w,0]) | flags)
    output[w,1] += wp.float64(1.)
    output[w,2] = wp.max(output[w,2], failure[w,0])
    output[w,3] = wp.max(output[w,3], failure[w,1])
    output[w,4] = wp.max(output[w,4], failure[w,2])
    output[w,5] = wp.max(output[w,5], wp.float64(iterations[w]))
    output[w,6] = wp.float64(int(output[w,6]) | raw_overflow[w])


def run_group(sim, nworld, seconds, repeats, folder, diagnostic_linesearch=False):
    with wp.ScopedDevice("cuda:0"):
        batch = B.BatchStepper(sim, nworld, nconmax=1024, njmax=4096)
        if diagnostic_linesearch:
            batch.model.opt.warn_overflow = 0  # Keep raw bits; suppress repeated GPU printf only.
        d, m = batch.data, batch.model
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = wp.empty((nworld, mujoco.mj_stateSize(sim.m, spec)), dtype=float)
        mjw.get_state(m, d, state, spec)
        initial = state.numpy().copy()
        initial_eq = batch.model.eq_data.numpy().copy()
        metrics = wp.zeros((nworld,7), dtype=wp.float64)
        failure = wp.array(np.tile([0.,0.,0.,1.], (nworld,1)), dtype=wp.float64)
        raw = getattr(d, "overflow", wp.zeros(nworld, dtype=int))
        limits = []
        if batch._material:
            b = sim.bushings
            f = lambda value, dtype: wp.array(value, dtype=dtype)
            limits = [f(b.a, int), f(b.b, int), f(b.axis, int),
                      f(np.broadcast_to(b.ra, (nworld,*b.ra.shape)).copy(), wp.vec3d),
                      f(np.broadcast_to(b.compression_limit, (nworld,b.n)).copy(), wp.float64),
                      f(np.broadcast_to(np.where(np.isfinite(b.limit),b.limit,1e300), (nworld,*b.limit.shape)).copy(), wp.vec3d)]
        def advance():
            batch.step()
            if batch._material:
                wp.launch(K.failure_metrics, nworld, inputs=[*limits,batch.xpos,batch.xmat,batch.xquat,d.qpos,d.qvel,failure])
            wp.launch(health, nworld, inputs=[d.qpos,d.qvel,failure,raw,batch.overflow,d.nefc,d.nacon,
                d.solver_niter,d.njmax,d.naconmax,sim.m.opt.iterations,metrics])
        before = d.time.numpy().copy()
        with wp.ScopedCapture() as capture: advance()
        np.testing.assert_array_equal(before,d.time.numpy())
        for _ in range(20): wp.capture_launch(capture.graph)
        wp.synchronize_device()
        rows=[]
        steps=round(seconds/sim.m.opt.timestep)
        start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for repeat in range(repeats):
            state.assign(initial)
            mjw.set_state(m,d,state,spec)
            batch.model.eq_data.assign(initial_eq)
            for name in ("external","forces","filtered","raw","overflow"):
                if hasattr(batch,name): getattr(batch,name).zero_()
            raw.zero_()
            metrics.zero_()
            mjw.forward(m,d)
            gpu_s,wall_s,count=0.,0.,0
            while count<steps:
                block=min(100,steps-count)
                wp.synchronize_device()
                wall_start=time.perf_counter()
                wp.record_event(start)
                for _ in range(block): wp.capture_launch(capture.graph)
                wp.record_event(end)
                wp.synchronize_device()
                wall_s+=time.perf_counter()-wall_start
                gpu_s+=wp.get_event_elapsed_time(start,end)/1000
                count+=block
                values=metrics.numpy()
                flags=values[:,0].astype(np.int64)
                if np.any(flags & ~16) or (not diagnostic_linesearch and np.any(flags)): break
            valid=count==steps and not np.any(values[:,0])
            complete=count==steps and not np.any(values[:,0].astype(np.int64) & ~16)
            rows.append(dict(repeat=repeat,steps=count,gpu_seconds=gpu_s,wall_seconds=wall_s,
                             valid=bool(valid),diagnostic_complete=bool(complete),health=values.tolist()))
            if not valid:
                mjw.get_state(m,d,state,spec)
                np.savez_compressed(folder/f"failure-{nworld}w.npz",integration_state=state.numpy(),integration_state_spec=spec)
                if not (diagnostic_linesearch and complete): break
        result=dict(valid=all(x['valid'] for x in rows) and len(rows)==repeats,nworld=nworld,
                    timestep=sim.m.opt.timestep,steps=steps,rows=rows,nv=sim.m.nv,segments=len(sim.desc.segments),
                    size=list(sim.desc.kappa.size),mass_kg=sim.mass,solver=str(sim.m.opt.solver),
                    integrator=str(sim.m.opt.integrator),raw_overflow_available=hasattr(d,'overflow'),
                    diagnostic_linesearch=diagnostic_linesearch,
                    health_columns=['failure_bits','steps','max_strain','compression_limit_ratio','bend_limit_ratio','peak_iterations','raw_overflow'],
                    kappa=sim.desc.kappa.to_dict())
        result['diagnostic_complete']=len(rows)==repeats and all(x['diagnostic_complete'] for x in rows)
        if result['valid'] or (diagnostic_linesearch and result['diagnostic_complete']):
            elapsed=statistics.median(x['gpu_seconds'] for x in rows)
            timing=dict(median_gpu_steps_per_second=steps/elapsed,gpu_wall_seconds_per_simulated_second=elapsed/(steps*sim.m.opt.timestep),
                        aggregate_world_steps_per_second=nworld*steps/elapsed,
                        aggregate_simulated_seconds_per_gpu_second=nworld*steps*sim.m.opt.timestep/elapsed)
            if result['valid']: result.update(timing)
            else: result['diagnostic_timing']=timing
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,nargs='+',default=[1,32])
    parser.add_argument('--categories',nargs='+',default=['hard','bendable','compressible'])
    parser.add_argument('--seconds',type=float,default=1.)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--diagnostic-linesearch',action='store_true')
    args=parser.parse_args()
    if mujoco.__version__!='3.17.0': raise RuntimeError('Run the frozen comparison in .venv / MuJoCo3.17')
    args.output.mkdir(parents=True,exist_ok=False)
    files={Path(__file__),Path(segmented.__file__),Path(A.__file__),Path(B.__file__),Path(K.__file__),
           Path(B.E.__file__),Path(sys.modules['warp_mechanics'].__file__),ROOT/'test/flex3d/grasp_platform.py'}
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for i,p in enumerate(sorted(files)): (args.output/f'source_{i}_{p.name}').write_bytes(p.read_bytes())
    report=dict(hostname=platform.node(),mujoco=mujoco.__version__,mujoco_warp=importlib.metadata.version('mujoco-warp'),
                warp=wp.__version__,sources=sources,groups=[],control='initialized YAM controls held; no CPU controller in timing')
    wp.init()
    if not wp.get_cuda_devices(): raise RuntimeError('CUDA required')
    for category in args.categories:
        robot=segmented.scene(segmented.Config(category=category,size=(.24,.16,.05),mass=.29,pinch_coeff=0.))
        folder=args.output/category;folder.mkdir()
        for nworld in args.worlds:
            try: result=run_group(robot.sim,nworld,args.seconds,args.repeats,folder,args.diagnostic_linesearch)
            except Exception as error: result=dict(valid=False,nworld=nworld,error=str(error),traceback=traceback.format_exc())
            result['category']=category
            (folder/f'{nworld}w.json').write_text(json.dumps(result,indent=2))
            report['groups'].append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('rows','kappa','health_columns','traceback')}),flush=True)
            (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    report['sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in sources.items())
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
