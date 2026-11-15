"""Independent-parameter GPU step benchmarks for the physical parcel."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bag_sim.physical_parcel import Config  # noqa: E402
from bag_sim.physical_parcel_gpu import Batch  # noqa: E402


def memory_mib():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    return int(result.stdout.splitlines()[0])


def benchmark(worlds,cfg,steps,repeats,output,initial_qpos=None):
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    batch=Batch(worlds,cfg)
    if initial_qpos is not None:
        batch.reset(initial_qpos=initial_qpos)
    wp.synchronize()
    memory=[memory_mib()]
    graph=batch.graph(steps=min(10,steps))
    wp.synchronize()
    setup=time.perf_counter()-start
    counts=[]
    timings=[]
    diagnostics=[]
    block=min(10,steps)
    if steps%block:
        raise ValueError('steps must be divisible by the captured block length')
    for _ in range(repeats):
        batch.reset()
        wp.synchronize()
        start=time.perf_counter()
        for _ in range(steps//block):
            wp.capture_launch(graph)
        wp.synchronize()
        timings.append(time.perf_counter()-start)
        diagnostics.append(batch.diagnostics())
        counts.append(batch.contact_counts.numpy().tolist())
        memory.append(memory_mib())
    last=diagnostics[-1]
    masses=batch.m.body_mass.numpy()[:,batch.parcel.payload_bodies].sum(axis=1)
    parameters=batch.parameters.numpy()
    payload_geom=int(batch.payload_geoms.numpy()[0])
    actual_mu=batch.m.geom_friction.numpy()[:,payload_geom,0]
    actual_table_mu=batch.m.geom_friction.numpy()[:,batch.parcel.m.geom('table').id,0]
    actual_parameters=np.column_stack((masses,actual_mu,actual_table_mu))
    consistent=bool(np.allclose(masses,cfg.payload_mass*parameters[:,0],rtol=1e-6)
                    and np.allclose(actual_mu,parameters[:,1])
                    and np.allclose(actual_table_mu,parameters[:,2]))
    # Independent draws may produce equal float32 masses. Verify the complete
    # applied parameter tuples and correspondence to each world's sample.
    independent=bool(consistent and (worlds==1 or len(np.unique(actual_parameters,axis=0))==worlds))
    finite=all(not any(row['nonfinite']) for row in diagnostics)
    overflow=all(not any(row['overflow_bits']) for row in diagnostics)
    penetration=max(max(row['peak_penetration_m']) for row in diagnostics)
    converged=all(max(row['peak_solver_iterations'])<cfg.iterations for row in diagnostics)
    valid=bool(finite and overflow and converged and penetration<=.001 and independent)
    median=float(np.median(timings))
    result=dict(worlds=worlds,mode=cfg.mode,steps_per_repeat=steps,repeats=repeats,
                setup_compile_capture_s=setup,wall_seconds=timings,
                mean_step_ms=float(np.mean(timings)/steps*1000),
                median_environment_steps_per_second=worlds*steps/median if valid else None,
                diagnostic_environment_steps_per_second=worlds*steps/median,
                valid=valid,finite=finite,zero_overflow=overflow,solver_under_cap=converged,
                peak_contact_penetration_m=penetration,independent_actual_mass=independent,
                actual_parameter_tuples_unique=independent,per_world_updates_consistent=consistent,
                actual_mass_range_kg=[float(masses.min()),float(masses.max())],
                last_contacts_range=[min(last['last_contacts_per_world']),max(last['last_contacts_per_world'])],
                peak_solver_iterations=max(max(row['peak_solver_iterations']) for row in diagnostics),
                sampled_device_memory_peak_MiB=max(memory),memory_includes_desktop=True,
                simulated_seconds_per_repeat=steps*cfg.timestep,
                diagnostics=diagnostics,config=asdict(cfg))
    (output/'result.json').write_text(json.dumps(result,indent=2))
    batch.parcel.save(output/'cpu_model')
    return {k:v for k,v in result.items() if k not in ('diagnostics','config')}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--worlds',type=int,nargs='+',default=[1,32,128,512])
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--mode',choices=['rigid','bendable','compressible'],default='rigid')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-state',type=Path,help='Matching saved CPU qpos for loaded-contact timing')
    args=parser.parse_args()
    if args.steps<1 or args.repeats<1:
        parser.error('steps and repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'bag_sim/physical_parcel.py',ROOT/'bag_sim/physical_parcel_gpu.py',ROOT/'bag_sim/bag_model.py']
    manifest=dict(machine=platform.node(),versions={name:metadata.version(name) for name in
                  ('mujoco','mujoco-warp','warp-lang')},
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    initial_qpos=None
    if args.initial_state:
        with np.load(args.initial_state) as source:
            initial_qpos=source['qpos'].copy()
        manifest['initial_state']=str(args.initial_state)
        manifest['initial_state_sha256']=hashlib.sha256(args.initial_state.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'source').mkdir()
    for path in sources:
        (args.output/'source'/path.name).write_bytes(path.read_bytes())
    results=[]
    for worlds in args.worlds:
        result=benchmark(worlds,Config(mode=args.mode),args.steps,args.repeats,args.output/f'worlds-{worlds}',initial_qpos)
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)
    if not all(row['valid'] for row in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
