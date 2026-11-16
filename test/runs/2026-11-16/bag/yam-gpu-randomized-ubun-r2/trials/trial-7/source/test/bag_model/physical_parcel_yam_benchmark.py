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
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experience'),str(ROOT/'test/bag_model')]
import physical_parcel_yam as backend
from physical_parcel_checks import serial,film_payload_geometry,film_box_geometry
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
           'checks':ROOT/'test/bag_model/physical_parcel_checks.py',
           'orchestration':ROOT/'test/bag_model/run_physical_parcel_gpu_suite.py',
           'mixed':ROOT/'test/bag_model/physical_parcel_mixed_gpu.py',
           'preparation':ROOT/'test/bag_model/prepare_physical_parcel.py'}
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
    previous=json.loads((args.configs_from/'groups.json').read_text()) if args.configs_from else None
    from prepare_physical_parcel import prepare as prepare_state
    for profile in ['kraft','pe','bubble']:
        for shape,mode in ARCHETYPES:
            index=len(groups);label=f'{profile}-{mode}-{shape}'
            for retry in range(8):
                cfg=backend.Config(**previous[index]['config']) if previous else sample_config(rng,profile,shape,mode)
                cfg=replace(cfg,initial_state=None,complete_motion=False,
                            check_initial_robot_contacts=True,
                            support_size=tuple(np.asarray(cfg.bag_size)+args.support_margin) if args.support_margin>0 else cfg.support_size)
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
            points=None
            if args.settle_seconds:
                destination=output/'prepared'/f'group-{index}'
                try:
                    parcel,preparation=prepare_state(cfg,destination,args.settle_seconds)
                except RuntimeError as exc:
                    group['preparation']=dict(passed=False,reason=str(exc),path=str(destination))
                    groups.append(group);write(output/'groups.json',groups)
                    print(json.dumps(dict(group=label,preparation_failed=str(exc))),flush=True)
                    continue
                cfg=replace(cfg,initial_state=str((destination/'initial_state.npz').resolve()))
                group['config']=asdict(cfg);group['preparation']=preparation
                points=parcel.d.flexvert_xpos.copy()
                np.savez_compressed(destination/'sampling_surface.npz',points=points,faces=parcel.envelope['faces'])
            for region in ['top','edge','corner','side']:
                for retry in range(4):
                    try:
                        sample,actual=sample_surface(parcel,cfg,rng,region,points=points)
                    except ValueError as exc:
                        group['surface_samples'].append(dict(region=region,retry=retry,accepted=False,reason=str(exc)))
                        break
                    trial=replace(cfg,**sample)
                    begin=time.perf_counter()
                    entry=dict(region=region,retry=retry,sample=actual,config=asdict(trial))
                    try:
                        robot=backend.scene(trial)
                    except RuntimeError as exc:
                        if not any(x in str(exc) for x in ['initial pose unreachable','initial film contact','initial robot collision']):raise
                        entry.update(accepted=False,reason=str(exc))
                    else:
                        initial=backend.measure(robot)
                        entry.update(accepted=True,ik=robot.ik_errors[-1],initial=initial,
                                     initial_all_contacts=getattr(robot,'initial_collision_report',None))
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
    if args.gpu:cfg=replace(cfg,wall_limit=3600.)
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
        contacts=[]
        finger_geoms=set.union(*robot.all_finger_sets)
        for contact in sim.d.contact:
            if contact.efc_address<0 or not any(int(g) in finger_geoms for g in contact.geom):continue
            for slot in (0,1):
                if contact.flex[slot]>=0:
                    contacts.append(dict(position_m=contact.pos.copy(),film_triangle=int(contact.elem[slot]),
                                         film_vertex=int(contact.vert[slot]),distance_m=float(contact.dist)))
        rows.append(dict(t=row['t'],phase=row['phase'],grip_N=row['grip_N'],support_N=row['support_N'],
                         finger_film_contacts=contacts,
                         com=row['com'],tip_mid=row['tip_mid'],gap=row['gap'],
                         geometry=row['payload_geometry'],penetration_mm=row['penetration_mm']))
        now=time.perf_counter()
        if row['phase']!=last_phase or now-last_print>20:
            write(args.output/'progress.json',dict(t=row['t'],phase=row['phase'],wall_s=now-started,
                grip_N=row['grip_N'],support_N=row['support_N'],com=row['com'],tracking_mm=row['tracking_mm'],
                penetration_mm=row['penetration_mm'],
                gpu_timing=getattr(sim,'gpu_timing',None),
                gpu_peak_solver_iterations=int(sim.gpu.peak_iterations.numpy()[0]) if hasattr(sim,'gpu') else None))
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
            i=max(ids,key=lambda i:rows[i]['grip_N']) if phase in ('close','lift','lift_hold') else ids[len(ids)//2]
            bearing=rows[i]['support_N']<.05 and rows[i]['com'][2]-result['initial_com'][2]>=.05
            stages[phase]=dict(index=i,**rows[i],loaded=bool(rows[i]['grip_N']>.05),load_bearing=bool(bearing))
        else:stages[phase]=dict(missing=True)
    result['stages']=stages
    write(args.output/'summary.json',result)
    write(args.output/'stages.json',stages)


def sample_timing(args):
    args.output.mkdir(parents=True,exist_ok=False)
    from physical_parcel_randomization import RANGES
    rng=np.random.default_rng(args.seed)
    sample_config(rng,'pe','box','rigid')
    timings=[]
    for _ in range(3):
        start=time.perf_counter()
        for profile in ['kraft','pe','bubble']:
            for shape,mode in ARCHETYPES:sample_config(rng,profile,shape,mode)
        timings.append(time.perf_counter()-start)
    write(args.output/'result.json',dict(seed=args.seed,configs_per_repeat=18,repeats=3,
          wall_s=timings,median_sampling_us_per_config=float(np.median(timings)/18*1e6),
          scope='CPU parameter sampling only; geometry wrap/compilation and GPU reset are separate',ranges=RANGES))


def memory():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().splitlines()[0])


def source_contact_audit(directory):
    """Geometry-only replay of recorded states; no new simulated trajectory."""
    model_path=directory/'trial/model.mjb';state_path=directory/'states.npz'
    signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [model_path,state_path]}
    destination=directory/'contact_audit.json'
    if destination.exists():
        saved=json.loads(destination.read_text())
        if saved['sources']==signature:return saved
    begin=time.perf_counter();model=mujoco.MjModel.from_binary_path(str(model_path));data=mujoco.MjData(model)
    bodies={model.body('arm').id}
    for body in range(model.nbody):
        if model.body_parentid[body] in bodies:bodies.add(body)
    robot=np.isin(model.geom_bodyid,list(bodies))
    support={model.geom('table').id}
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    if floor>=0:support.add(floor)
    maximum=dict(robot_self_m=0.,robot_support_m=0.,all_native_m=0.)
    over=0
    with np.load(state_path) as source:positions=source['qpos']
    for qpos in positions:
        data.qpos[:]=qpos
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_flex(model,data)
        mujoco.mj_collision(model,data)
        invalid=False
        for contact in data.contact:
            depth=max(0.,-float(contact.dist));maximum['all_native_m']=max(maximum['all_native_m'],depth)
            geoms=[int(g) for g in contact.geom if g>=0]
            robots=[g for g in geoms if robot[g]]
            if len(robots)==2:maximum['robot_self_m']=max(maximum['robot_self_m'],depth)
            if robots and any(g in support for g in geoms):maximum['robot_support_m']=max(maximum['robot_support_m'],depth)
            invalid |= depth>.001
        over+=int(invalid)
    result=dict(sources=signature,method='CPU collision geometry recomputed at recorded qpos, no time integration',
                samples=len(positions),samples_over_1mm=over,maximum=maximum,wall_s=time.perf_counter()-begin)
    write(destination,result);return result


def material_evidence(batch,sim,output):
    checks={};arrays={}
    for name in ['flex_stiffness','flex_damping','flex_friction','flex_radius']:
        actual=getattr(batch.m,name).numpy()
        expected=np.asarray(getattr(sim.m,name))
        checks[name]=bool(np.allclose(actual.ravel(),expected.ravel(),rtol=2e-6,atol=1e-12))
        arrays[name]=actual
    for name,device,expected in [('hinge_stiffness',batch.k,sim.parcel.hinge_stiffness),
                                 ('hinge_damping',batch.c,sim.parcel.hinge_damping),
                                 ('hinge_reference_angles',batch.rest,sim.parcel.rest_angles)]:
        actual=device.numpy();checks[name]=bool(np.allclose(actual,expected,rtol=2e-6,atol=1e-12));arrays[name]=actual
    positions=batch.m.body_pos.numpy()
    checks['per_world_reference_body_positions']=bool(np.allclose(positions,sim.m.body_pos[None],rtol=2e-6,atol=1e-8))
    arrays['reference_body_positions']=positions[0]
    nodes=np.array([sim.m.body(f'bag_{i}').id for i in range(sim.m.nflexvert)])
    masses=batch.m.body_mass.numpy()[:,nodes]
    checks['film_lumped_masses']=bool(np.allclose(masses,sim.m.body_mass[None,nodes],rtol=2e-6,atol=1e-12))
    arrays['film_lumped_masses']=masses[0]
    np.savez_compressed(output,**arrays)
    return checks


def friction_evidence(batch,sim,output):
    """Read actual active-contact coefficients, including native priority masking."""
    count=min(int(batch.d.nacon.numpy()[0]),batch.d.naconmax)
    world=batch.d.contact.worldid.numpy()[:count]
    geom=batch.d.contact.geom.numpy()[:count]
    flex=batch.d.contact.flex.numpy()[:count]
    mu=batch.d.contact.friction.numpy()[:count,0]
    addresses=batch.d.contact.efc_address.numpy()[:count,0]
    active=(addresses>=0)&(addresses<batch.d.njmax)
    normal=np.zeros(count)
    forces=batch.d.efc.force.numpy()
    normal[active]=forces[world[active],addresses[active]]-batch.d.contact.adhesion.numpy()[:count][active]
    film=(flex>=0).any(axis=1)&active
    payload=np.isin(geom,batch.payload_geoms.numpy()).any(axis=1)
    table=(geom==sim.table_geom).any(axis=1)
    masks=dict(payload=film&payload,table=film&table,self_contact=film&(geom<0).all(axis=1))
    ranges={};result={}
    for name,mask in masks.items():
        counts=np.bincount(world[mask],minlength=batch.nworld)
        minimum=np.full(batch.nworld,np.inf);maximum=np.full(batch.nworld,-np.inf)
        np.minimum.at(minimum,world[mask],mu[mask]);np.maximum.at(maximum,world[mask],mu[mask])
        minimum[counts==0]=np.nan;maximum[counts==0]=np.nan
        ranges[name+'_counts']=counts;ranges[name+'_min']=minimum;ranges[name+'_max']=maximum
        result[name]=dict(active_contacts=int(mask.sum()),actual_mu_range=[float(mu[mask].min()),float(mu[mask].max())] if mask.any() else None)
    finger_normal=np.zeros((batch.nworld,2))
    for side,name in enumerate(('left','right')):
        ids=set(sim.pad_geoms[name])|{g for g in range(sim.m.ngeom)
                  if sim.m.geom(g).name==name+'_pad' or sim.m.geom(g).name.startswith(name+'_round_')}
        mask=film&np.isin(geom,list(ids)).any(axis=1)
        finger_normal[:,side]=np.bincount(world[mask],weights=np.maximum(normal[mask],0),minlength=batch.nworld)
    ranges['finger_normal_N']=finger_normal
    result['worlds_with_any_finger_normal_load']=int(np.sum((finger_normal>.05).any(axis=1)))
    result['worlds_with_both_finger_normal_load']=int(np.sum((finger_normal>.05).all(axis=1)))
    result['normal_load_threshold_N']=.05
    np.savez_compressed(output,**ranges)
    film_priority=int(sim.m.flex_priority[0])
    result['native_priority']=dict(film=film_priority,payload=sorted(set(int(sim.m.geom_priority[g]) for g in batch.payload_geoms.numpy())),table=int(sim.m.geom_priority[sim.table_geom]))
    result['bag_mu']=sim.parcel.cfg.friction
    result['interpretation']='Higher contact priority selects friction; equal priority uses max. Model-field readback alone is not proof that each material coefficient affects every contact pair.'
    return result


def benchmark(args):
    import warp as wp
    from bag_sim.physical_parcel_gpu import Batch
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.trial/'summary.json').read_text())
    cfg=backend.Config(**saved['config'])
    cpu_begin=time.perf_counter();robot=backend.scene(cfg);sim=robot.sim
    cpu_scene_s=time.perf_counter()-cpu_begin
    stages=saved['stages']
    if args.phase not in stages or stages[args.phase].get('missing'):
        write(output/'result.json',dict(status='missing_phase',phase=args.phase));return
    index=stages[args.phase]['index']
    with np.load(args.trial/'states.npz') as states:
        qpos,qvel,ctrl=[states[k][index].copy() for k in ['qpos','qvel','ctrl']]
    sim.d.qpos[:]=qpos;sim.d.qvel[:]=qvel;sim.d.ctrl[:]=ctrl
    backend.refresh(sim)
    before=film_payload_geometry(sim.m,sim.d,sim.parcel.envelope['faces'])
    audit=source_contact_audit(args.trial)
    recorded_gpu_peak=max(saved.get('gpu_health',{}).get('peak_penetration_m',[0.]))
    recorded_solver_peak=max(saved.get('gpu_health',{}).get('peak_solver_iterations',[0]))
    source_valid=(saved['outcome']!='sim_fail' and saved['peaks']['penetration_mm']<=1
                  and not any(saved['warnings'].values()) and before['geometric_depth_upper_m']<=.001
                  and before['outside_corners']==0 and audit['maximum']['all_native_m']<=.001
                  and recorded_gpu_peak<=.001 and recorded_solver_peak<sim.parcel.cfg.iterations)
    (output/'model_config.json').write_text(json.dumps(asdict(sim.parcel.cfg),default=serial,indent=2))
    manifest(output)
    contact_capacity=2**int(np.ceil(np.log2(max(args.contacts,2*int(sim.d.ncon)))))
    constraint_capacity=2**int(np.ceil(np.log2(max(args.constraints,2*int(sim.d.nefc)))))
    begin=time.perf_counter()
    batch=Batch(args.worlds,sim.parcel.cfg,seed=args.seed,randomize=True,
                parcel=sim.parcel,model=sim.m,data=sim.d,
                contact_capacity=contact_capacity,constraint_capacity=constraint_capacity)
    wp.synchronize();uploaded=time.perf_counter()-begin
    write(output/'loaded.json',dict(worlds=args.worlds,uploaded_s=uploaded,device_memory_MiB=memory(),
          mempool_allocated_peak_MiB=wp.get_mempool_used_mem_high('cuda:0')/2**20))
    materials=material_evidence(batch,sim,output/'actual-material.npz')
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
        friction=friction_evidence(batch,sim,output/f'effective-friction-{repeat}.npz')
        # All environments get the complete independent tuple, verify uploaded values.
        masses=batch.m.body_mass.numpy()[:,sim.parcel.payload_bodies].sum(axis=1)
        frictions=batch.m.geom_friction.numpy()[:,batch.payload_geoms.numpy(),0]
        pars=batch.parameters.numpy()
        inertias=batch.m.body_inertia.numpy()[:,sim.parcel.payload_bodies]
        centers=batch.m.body_ipos.numpy()[:,sim.parcel.payload_bodies]
        quaternions=batch.m.body_iquat.numpy()[:,sim.parcel.payload_bodies]
        table_friction=batch.m.geom_friction.numpy()[:,sim.table_geom,0]
        joints=batch.bend_joints.numpy();dofs=batch.bend_dofs.numpy()
        joint_k=batch.m.jnt_stiffness.numpy()[:,joints]
        joint_d=batch.m.dof_damping.numpy()[:,dofs]
        applied_checks=dict(mass=bool(np.allclose(masses,cfg.payload_mass*pars[:,0],rtol=1e-6)),
             inertia=bool(np.allclose(inertias,batch.base_inertia.numpy()[None,sim.parcel.payload_bodies]*pars[:,0,None,None],rtol=1e-6,atol=1e-12)),
             com=bool(np.allclose(centers,sim.m.body_ipos[None,sim.parcel.payload_bodies])),
             payload_friction=bool(np.allclose(frictions,pars[:,1,None])),
             table_friction=bool(np.allclose(table_friction,pars[:,2])),
             joint_stiffness=bool(np.allclose(joint_k,batch.base_k.numpy()[None,joints]*pars[:,3,None])),
             joint_damping=bool(np.allclose(joint_d,batch.base_damping.numpy()[None,dofs]*pars[:,4,None])))
        applied_checks.update(materials)
        consistent=all(applied_checks.values())
        np.savez_compressed(output/f'parameters-{repeat}.npz',parameters=pars,payload_mass=masses,
                            payload_friction=frictions,table_friction=table_friction,
                            body_inertia=inertias,body_com=centers,body_inertia_quat=quaternions,
                            joint_stiffness=joint_k,joint_damping=joint_d,
                            qpos_initial=qpos,qvel_initial=qvel,ctrl_initial=ctrl)
        # Validate final geometry in every environment outside the timed interval.
        positions=batch.d.qpos.numpy();velocities=batch.d.qvel.numpy()
        bad_geo=0;worst=0.;exposed=0;support_worst=0.;geometry_per_world=[]
        check_begin=time.perf_counter()
        for world in range(args.worlds):
            sim.d.qpos[:]=positions[world];sim.d.qvel[:]=velocities[world]
            mujoco.mj_kinematics(sim.m,sim.d);mujoco.mj_comPos(sim.m,sim.d)
            mujoco.mj_flex(sim.m,sim.d)
            geo=film_payload_geometry(sim.m,sim.d,sim.parcel.envelope['faces'])
            support_geo=film_box_geometry(sim.m,sim.d,sim.parcel.envelope['faces'],sim.table_geom)
            support_depth=support_geo['geometric_depth_upper_m']
            support_worst=max(support_worst,support_depth)
            worst=max(worst,geo['geometric_depth_upper_m']);exposed+=int(geo['outside_corners']>0)
            bad_geo+=int(max(geo['geometric_depth_upper_m'],support_depth)>.001 or geo['outside_corners']>0)
            geometry_per_world.append((geo['geometric_depth_upper_m'],geo['outside_corners'],support_depth))
        finite=not any(diag['nonfinite']);zero_overflow=not any(diag['overflow_bits'])
        penetration=max(diag['peak_penetration_m'])
        converged=max(diag['peak_solver_iterations'])<sim.m.opt.iterations
        numeric_valid=finite and zero_overflow and converged
        local_valid=numeric_valid and consistent and bad_geo==0 and penetration<=.001 and converged
        physical_valid=local_valid and source_valid
        row=dict(repeat=repeat,wall_s=wall,reset_s=reset,environment_steps_per_s=args.worlds*args.steps/wall,
                 batch_step_ms=wall/args.steps*1000,per_environment_realtime=cfg.dt*args.steps/wall,
                 simulated_seconds_per_environment=args.steps*cfg.dt,
                 actual_gpu_time_range_s=[float(batch.d.time.numpy().min()),float(batch.d.time.numpy().max())],
                 finite_zero_overflow=finite and zero_overflow,numerically_valid=numeric_valid,
                 physically_valid=physical_valid,local_contact_geometry_valid=local_valid,source_trial_geometry_valid=source_valid,parameters_applied=consistent,applied_checks=applied_checks,device_memory_MiB=memory(),
                 mempool_allocated_peak_MiB=wp.get_mempool_used_mem_high('cuda:0')/2**20,
                 peak_contact_penetration_m=penetration,peak_geometry_depth_m=worst,peak_support_geometry_depth_m=support_worst,exposed_environments=exposed,
                 invalid_geometry_environments=bad_geo,geometry_check_s=time.perf_counter()-check_begin,
                 solver_under_cap=converged,peak_solver_iterations=max(diag['peak_solver_iterations']),
                 contact_count_range=[min(diag['last_contacts_per_world']),max(diag['last_contacts_per_world'])],
                 total_contacts=diag['native_nacon'],overflow_worlds=sum(x!=0 for x in diag['overflow_bits']),
                 nonfinite_worlds=sum(diag['nonfinite']))
        row.update(effective_friction=friction,
                   finger_load_fraction_final=friction['worlds_with_any_finger_normal_load']/args.worlds,
                   loaded_window_eligible=bool(stages[args.phase]['loaded'] and friction['worlds_with_any_finger_normal_load']==args.worlds),
                   seed=args.seed,episode=batch.episode-1,unique_parameter_tuples=len(np.unique(pars,axis=0)))
        np.savez_compressed(output/f'health-{repeat}.npz',overflow_bits=diag['overflow_bits'],
                            nonfinite=diag['nonfinite'],solver_iterations=diag['peak_solver_iterations'],
                            contact_counts=diag['last_contacts_per_world'],contact_penetration=diag['peak_penetration_m'],
                            geometry_depth_and_exposed=np.array(geometry_per_world))
        samples.append(row);write(output/'samples.json',samples)
    write(output/'result.json',dict(status='completed',worlds=args.worlds,phase=args.phase,
          source_stage=stages[args.phase],source_trial_geometry_valid=source_valid,source_contact_audit=audit,material_checks=materials,
          buffers={name:int(getattr(batch.d,name)) for name in ['naconmax','naccdmax','njmax','njmax_nnz'] if hasattr(batch.d,name)},
          requested_contact_capacity=args.contacts,requested_constraint_capacity=args.constraints,
          allocated_contact_capacity=contact_capacity,allocated_constraint_capacity=constraint_capacity,model=asdict(sim.parcel.cfg),profile=cfg.bag_profile,
          complete_robot=dict(nq=sim.m.nq,nv=sim.m.nv,nu=sim.m.nu,nbody=sim.m.nbody,ngeom=sim.m.ngeom,
                              actuator_forcerange=sim.m.actuator_forcerange),
          cpu_scene_compile_ik_s=cpu_scene_s,upload_reset_s=uploaded,setup_components=batch.setup_timing,warmup_capture_compile_s=capture,initial_geometry=before,samples=samples,
          median_environment_steps_per_s=float(np.median([s['environment_steps_per_s'] for s in samples])),
          numerically_valid=all(s['numerically_valid'] for s in samples),
          physically_valid=all(s['physically_valid'] for s in samples),
          control='actual recorded actuator commands held for short phase-local timing window; full shared control timed separately'))
    print(json.dumps(dict(worlds=args.worlds,phase=args.phase,result=str(output/'result.json'))),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['prepare','trial','benchmark','sample-timing']:
        p=sub.add_parser(command);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261116)
        if command=='prepare':
            p.add_argument('--configs-from',type=Path)
            p.add_argument('--settle-seconds',type=float,default=.5)
            p.add_argument('--support-margin',type=float,default=.04)
        if command=='trial':
            p.add_argument('--groups',type=Path,required=True);p.add_argument('--group',type=int,required=True)
            p.add_argument('--region');p.add_argument('--sample',type=int,default=0);p.add_argument('--gpu',action='store_true')
            p.add_argument('--smoke-seconds',type=float,default=0.)
        if command=='benchmark':
            p.add_argument('--trial',type=Path,required=True);p.add_argument('--phase',choices=['approach','close','lift','lift_hold'],required=True)
            p.add_argument('--worlds',type=int,required=True);p.add_argument('--steps',type=int,default=100)
            p.add_argument('--contacts',type=int,default=512);p.add_argument('--constraints',type=int,default=2048)
    args=parser.parse_args()
    {'prepare':prepare,'trial':trial,'benchmark':benchmark,'sample-timing':sample_timing}[args.command](args)


if __name__=='__main__':main()
