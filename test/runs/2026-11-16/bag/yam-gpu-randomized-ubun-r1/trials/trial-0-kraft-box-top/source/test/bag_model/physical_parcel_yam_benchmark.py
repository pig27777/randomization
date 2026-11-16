"""Full-YAM parcel trials and loaded-state resident GPU benchmarks.

Run each command under flock /tmp/claude-gpu.lock. Capacity workers are separate
processes so a failed CUDA allocation cannot contaminate later measurements.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experience'),str(ROOT/'test/bag_model')]
import physical_parcel_yam as backend
from physical_parcel_checks import serial,film_payload_geometry
from physical_parcel_randomization import sample_config,sample_surface,RANGES,ARCHETYPES
from yam_platform.protocol import run
from yam_platform.provenance import source_files
from yam_platform.robot import asset_fingerprint


def write(path,value):
    path.write_text(json.dumps(value,default=serial,indent=2))


def manifest(output):
    files={**source_files(),**backend.source_files(),
           'benchmark':Path(__file__),'randomization':ROOT/'test/bag_model/physical_parcel_randomization.py',
           'gpu_yam':ROOT/'test/bag_model/physical_parcel_yam_gpu.py',
           'gpu_model':ROOT/'bag_sim/physical_parcel_gpu.py',
           'adapter':ROOT/'test/bag_model/physical_parcel_yam.py',
           'checks':ROOT/'test/bag_model/physical_parcel_checks.py'}
    (output/'source').mkdir(exist_ok=True)
    hashes={}
    for path in files.values():
        path=Path(path)
        name=str(path.relative_to(ROOT))
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    write(output/'manifest.json',dict(machine=platform.node(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
          versions={p:metadata.version(p) for p in ['mujoco','mujoco-warp','warp-lang','numpy']},
          sources=hashes,robot_assets=asset_fingerprint()))


def prepare(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    manifest(output);write(output/'random_ranges.json',RANGES)
    rng=np.random.default_rng(args.seed)
    groups=[];rejects=[]
    for profile in ['kraft','pe','bubble']:
        for shape,mode in ARCHETYPES:
            index=len(groups);label=f'{profile}-{mode}-{shape}'
            for retry in range(8):
                cfg=sample_config(rng,profile,shape,mode)
                start=time.perf_counter()
                try:
                    parcel,_=backend.parcel_world(cfg)
                except ValueError as exc:
                    rejects.append(dict(group=label,retry=retry,reason=str(exc),config=asdict(cfg)))
                    continue
                break
            else: raise RuntimeError(f'initialization exhausted: {label}')
            group=dict(id=index,label=label,config=asdict(cfg),parcel_config=asdict(parcel.cfg),
                       compile_initialization_s=time.perf_counter()-start,geometry=parcel.envelope['metadata'],
                       payload_mass_kg=float(parcel.m.body_mass[parcel.payload_bodies].sum()),
                       payload_com=parcel.com(),surface_samples=[])
            for region in ['top','edge','corner','side']:
                for retry in range(4):
                    sample,actual=sample_surface(parcel,cfg,rng,region)
                    trial=replace(cfg,**sample)
                    begin=time.perf_counter()
                    entry=dict(region=region,retry=retry,sample=actual,config=asdict(trial))
                    try:
                        robot=backend.scene(trial)
                    except RuntimeError as exc:
                        if not any(x in str(exc) for x in ['initial pose unreachable','initial film contact']):raise
                        entry.update(accepted=False,reason=str(exc))
                    else:
                        initial=backend.measure(robot)
                        entry.update(accepted=True,ik=robot.ik_errors[-1],initial=initial)
                    entry['scene_ik_s']=time.perf_counter()-begin
                    group['surface_samples'].append(entry)
                    if entry['accepted']:break
            groups.append(group)
            write(output/'groups.json',groups);write(output/'initialization_rejections.json',rejects)
            print(json.dumps(dict(group=label,accepted=sum(s['accepted'] for s in group['surface_samples']),samples=len(group['surface_samples']))),flush=True)
    summary=dict(groups=len(groups),attempted=sum(len(g['surface_samples']) for g in groups),
                 accepted=sum(s['accepted'] for g in groups for s in g['surface_samples']),seed=args.seed)
    write(output/'sampling_summary.json',summary)


def trial(args):
    groups=json.loads((args.groups/'groups.json').read_text())
    group=groups[args.group]
    accepted=[s for s in group['surface_samples'] if s['accepted'] and (not args.region or s['region']==args.region)]
    if not accepted:raise ValueError('no accepted sample for requested group/region')
    # Prefer the requested surface; an explicit index allows reproduction.
    sample=accepted[args.sample%len(accepted)]
    cfg=backend.Config(**sample['config'])
    if args.smoke_seconds:cfg=replace(cfg,smoke_seconds=args.smoke_seconds)
    module=backend
    if args.gpu:
        import physical_parcel_yam_gpu as module
    args.output.mkdir(parents=True,exist_ok=False)
    manifest(args.output)
    write(args.output/'selection.json',dict(group=args.group,label=group['label'],sample=sample,gpu=args.gpu))
    arrays=[];rows=[];last_print=0.;started=time.perf_counter();last_phase=None
    timings=dict(cpu_command_s=0.,cpu_ik_s=0.,cpu_geometry_check_s=0.,cpu_feedback_s=0.,physical_step_s=0.)
    final_sim=None
    for name,key in [('measure','cpu_feedback_s'),('check_state','cpu_geometry_check_s')]:
        original=getattr(module,name)
        def timed_backend(*a,_original=original,_key=key,**k):
            start=time.perf_counter()
            try:return _original(*a,**k)
            finally:timings[_key]+=time.perf_counter()-start
        setattr(module,name,timed_backend)
    instrumented=False
    def observe(robot,row):
        nonlocal last_print,last_phase,instrumented,final_sim
        sim=robot.sim
        final_sim=sim
        if not instrumented:
            # Instrument existing methods without changing their behavior.
            def timed(owner,name,key):
                original=getattr(owner,name)
                def wrapped(*a,**k):
                    start=time.perf_counter()
                    try:return original(*a,**k)
                    finally:timings[key]+=time.perf_counter()-start
                setattr(owner,name,wrapped)
            timed(robot,'command','cpu_command_s');timed(robot,'solve','cpu_ik_s');timed(sim,'step','physical_step_s')
            instrumented=True
        arrays.append((sim.d.qpos.copy(),sim.d.qvel.copy(),sim.d.ctrl.copy()))
        rows.append(dict(t=row['t'],phase=row['phase'],grip_N=row['grip_N'],support_N=row['support_N'],
                         com=row['com'],tip_mid=row['tip_mid'],gap=row['gap'],
                         geometry=row['payload_geometry'],penetration_mm=row['penetration_mm']))
        now=time.perf_counter()
        if row['phase']!=last_phase or now-last_print>20:
            write(args.output/'progress.json',dict(t=row['t'],phase=row['phase'],wall_s=now-started))
            print(json.dumps(dict(t=row['t'],phase=row['phase'],wall_s=now-started)),flush=True)
            last_phase=row['phase'];last_print=now
        sim.benchmark_timings=timings
    result=run(module,cfg,args.output/'trial',on_sample=observe)
    qpos,qvel,ctrl=map(np.array,zip(*arrays))
    np.savez_compressed(args.output/'states.npz',qpos=qpos,qvel=qvel,ctrl=ctrl)
    write(args.output/'states.json',rows)
    result['benchmark_timings']=timings
    if hasattr(final_sim,'gpu_timing'):
        result['gpu_timing']=final_sim.gpu_timing
        result['gpu_health']=final_sim.gpu.diagnostics()
    result['attempts_per_second']=1/result['wall_s']
    result['execution_engine']='GPU physics + CPU shared controller' if args.gpu else 'CPU shared full protocol'
    # Select actual beginning/middle/end of each phase, retain load evidence.
    stages={}
    for phase in ['approach','close','lift','lift_hold']:
        ids=[i for i,r in enumerate(rows) if r['phase']==phase]
        if ids:
            i=ids[len(ids)//2]
            stages[phase]=dict(index=i,**rows[i],loaded=bool(rows[i]['grip_N']>.05))
        else:stages[phase]=dict(missing=True)
    result['stages']=stages
    write(args.output/'summary.json',result)
    write(args.output/'stages.json',stages)


def memory():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().splitlines()[0])


def benchmark(args):
    import warp as wp
    from bag_sim.physical_parcel_gpu import Batch
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.trial/'summary.json').read_text())
    cfg=backend.Config(**saved['config'])
    robot=backend.scene(cfg);sim=robot.sim
    stages=saved['stages']
    if args.phase not in stages or stages[args.phase].get('missing'):
        write(output/'result.json',dict(status='missing_phase',phase=args.phase));return
    index=stages[args.phase]['index']
    with np.load(args.trial/'states.npz') as states:
        qpos,qvel,ctrl=[states[k][index].copy() for k in ['qpos','qvel','ctrl']]
    sim.d.qpos[:]=qpos;sim.d.qvel[:]=qvel;sim.d.ctrl[:]=ctrl
    backend.refresh(sim)
    before=film_payload_geometry(sim.m,sim.d,sim.parcel.envelope['faces'])
    begin=time.perf_counter()
    batch=Batch(args.worlds,sim.parcel.cfg,seed=args.seed,randomize=True,
                parcel=sim.parcel,model=sim.m,data=sim.d,
                contact_capacity=args.contacts,constraint_capacity=args.constraints)
    wp.synchronize();uploaded=time.perf_counter()-begin
    write(output/'loaded.json',dict(worlds=args.worlds,uploaded_s=uploaded,device_memory_MiB=memory(),
          mempool_allocated_peak_MiB=wp.get_mempool_used_mem_high('cuda:0')/2**20))
    block=min(10,args.steps)
    if args.steps%block:raise ValueError('steps must be divisible by captured block')
    begin=time.perf_counter();graph=batch.graph(block);wp.synchronize();capture=time.perf_counter()-begin
    for _ in range(3):wp.capture_launch(graph)
    wp.synchronize()
    samples=[]
    for repeat in range(3):
        begin=time.perf_counter();batch.reset(initial_qpos=qpos,initial_qvel=qvel,initial_ctrl=ctrl)
        wp.synchronize();reset=time.perf_counter()-begin
        begin=time.perf_counter()
        for _ in range(args.steps//block):wp.capture_launch(graph)
        wp.synchronize();wall=time.perf_counter()-begin
        diag=batch.diagnostics()
        # All environments get the complete independent tuple, verify uploaded values.
        masses=batch.m.body_mass.numpy()[:,sim.parcel.payload_bodies].sum(axis=1)
        frictions=batch.m.geom_friction.numpy()[:,batch.payload_geoms.numpy(),0]
        pars=batch.parameters.numpy()
        consistent=bool(np.allclose(masses,cfg.payload_mass*pars[:,0],rtol=1e-6)
                        and np.allclose(frictions,pars[:,1,None]))
        np.savez_compressed(output/f'parameters-{repeat}.npz',parameters=pars,payload_mass=masses,
                            payload_friction=frictions,qpos_initial=qpos,qvel_initial=qvel,ctrl_initial=ctrl)
        # Validate final geometry in every environment outside the timed interval.
        positions=batch.d.qpos.numpy();velocities=batch.d.qvel.numpy()
        bad_geo=0;worst=0.;exposed=0
        check_begin=time.perf_counter()
        for world in range(args.worlds):
            sim.d.qpos[:]=positions[world];sim.d.qvel[:]=velocities[world]
            mujoco.mj_kinematics(sim.m,sim.d);mujoco.mj_comPos(sim.m,sim.d)
            mujoco.mj_flex(sim.m,sim.d)
            geo=film_payload_geometry(sim.m,sim.d,sim.parcel.envelope['faces'])
            worst=max(worst,geo['geometric_depth_upper_m']);exposed+=int(geo['outside_corners']>0)
            bad_geo+=int(geo['geometric_depth_upper_m']>.001 or geo['outside_corners']>0)
        finite=not any(diag['nonfinite']);zero_overflow=not any(diag['overflow_bits'])
        penetration=max(diag['peak_penetration_m'])
        converged=max(diag['peak_solver_iterations'])<sim.m.opt.iterations
        numeric_valid=finite and zero_overflow
        physical_valid=numeric_valid and consistent and bad_geo==0 and penetration<=.001 and converged
        row=dict(repeat=repeat,wall_s=wall,reset_s=reset,environment_steps_per_s=args.worlds*args.steps/wall,
                 batch_step_ms=wall/args.steps*1000,per_environment_realtime=cfg.dt*args.steps/wall,
                 simulated_seconds_per_environment=args.steps*cfg.dt,numerically_valid=numeric_valid,
                 physically_valid=physical_valid,parameters_applied=consistent,device_memory_MiB=memory(),
                 mempool_allocated_peak_MiB=wp.get_mempool_used_mem_high('cuda:0')/2**20,
                 peak_contact_penetration_m=penetration,peak_geometry_depth_m=worst,exposed_environments=exposed,
                 invalid_geometry_environments=bad_geo,geometry_check_s=time.perf_counter()-check_begin,
                 solver_under_cap=converged,peak_solver_iterations=max(diag['peak_solver_iterations']),
                 contact_count_range=[min(diag['last_contacts_per_world']),max(diag['last_contacts_per_world'])],
                 total_contacts=diag['native_nacon'],overflow_worlds=sum(x!=0 for x in diag['overflow_bits']),
                 nonfinite_worlds=sum(diag['nonfinite']))
        samples.append(row);write(output/'samples.json',samples)
    write(output/'result.json',dict(status='completed',worlds=args.worlds,phase=args.phase,
          source_stage=stages[args.phase],model=asdict(sim.parcel.cfg),profile=cfg.bag_profile,
          complete_robot=dict(nq=sim.m.nq,nv=sim.m.nv,nu=sim.m.nu,nbody=sim.m.nbody,ngeom=sim.m.ngeom,
                              actuator_forcerange=sim.m.actuator_forcerange),
          upload_reset_s=uploaded,warmup_capture_compile_s=capture,initial_geometry=before,samples=samples,
          median_environment_steps_per_s=float(np.median([s['environment_steps_per_s'] for s in samples])),
          numerically_valid=all(s['numerically_valid'] for s in samples),
          physically_valid=all(s['physically_valid'] for s in samples),
          control='actual recorded actuator commands held for short phase-local timing window; full shared control timed separately'))
    print(json.dumps(dict(worlds=args.worlds,phase=args.phase,result=str(output/'result.json'))),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['prepare','trial','benchmark']:
        p=sub.add_parser(command);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261116)
        if command=='trial':
            p.add_argument('--groups',type=Path,required=True);p.add_argument('--group',type=int,required=True)
            p.add_argument('--region');p.add_argument('--sample',type=int,default=0);p.add_argument('--gpu',action='store_true')
            p.add_argument('--smoke-seconds',type=float,default=0.)
        if command=='benchmark':
            p.add_argument('--trial',type=Path,required=True);p.add_argument('--phase',choices=['approach','close','lift','lift_hold'],required=True)
            p.add_argument('--worlds',type=int,required=True);p.add_argument('--steps',type=int,default=100)
            p.add_argument('--contacts',type=int,default=512);p.add_argument('--constraints',type=int,default=2048)
    args=parser.parse_args()
    {'prepare':prepare,'trial':trial,'benchmark':benchmark}[args.command](args)


if __name__=='__main__':main()
