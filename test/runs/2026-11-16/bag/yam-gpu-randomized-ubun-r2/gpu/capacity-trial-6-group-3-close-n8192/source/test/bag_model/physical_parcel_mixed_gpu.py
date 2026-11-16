"""Simultaneously resident heterogeneous full-YAM compile groups."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import warp as wp

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experience'),str(ROOT/'test/bag_model')]
import physical_parcel_yam as backend
from bag_sim.physical_parcel_gpu import Batch
from physical_parcel_checks import film_payload_geometry,film_box_geometry
from physical_parcel_yam_benchmark import write,manifest,memory,friction_evidence,material_evidence,source_contact_audit,refresh_saved_stages


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds-per-group',type=int,required=True);p.add_argument('--steps',type=int,default=100)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);manifest(args.output)
    entries=json.loads((args.run/'trial_summary.json').read_text());groups=[]
    for entry in entries:
        directory=ROOT/entry['path'];summary=refresh_saved_stages(directory)
        cfg=backend.Config(**summary['config']);cpu_begin=time.perf_counter()
        robot=backend.scene(cfg);sim=robot.sim;cpu_scene_s=time.perf_counter()-cpu_begin
        stages={k:v for k,v in summary['stages'].items() if not v.get('missing')}
        if stages:
            phase,stage=max(stages.items(),key=lambda kv:(int(kv[1].get('load_bearing',False)),int(kv[1]['loaded']),max(kv[1]['contact_magnitude_each_N'])))
            index=stage['index']
        else:
            phase='early_stopped_settle';stage=dict(loaded=False,load_bearing=False);index=-1
        with np.load(directory/'states.npz') as data:
            qpos,qvel,ctrl=[data[k][index].copy() for k in ['qpos','qvel','ctrl']]
        sim.d.qpos[:]=qpos;sim.d.qvel[:]=qvel;sim.d.ctrl[:]=ctrl;backend.refresh(sim)
        contact_capacity=2**int(np.ceil(np.log2(max(512,2*int(sim.d.ncon)))))
        constraint_capacity=2**int(np.ceil(np.log2(max(2048,2*int(sim.d.nefc)))))
        begin=time.perf_counter()
        batch=Batch(args.worlds_per_group,sim.parcel.cfg,parcel=sim.parcel,model=sim.m,data=sim.d,
                    seed=20261116+entry['group'],randomize=True,contact_capacity=contact_capacity,constraint_capacity=constraint_capacity)
        wp.synchronize();upload=time.perf_counter()-begin
        sim.material_checks=material_evidence(batch,sim,args.output/f'group-{len(groups)}-actual-material.npz')
        audit=source_contact_audit(directory)
        sim.source_valid=(summary['outcome']!='sim_fail' and summary['peaks']['penetration_mm']<=1
                          and not any(summary['warnings'].values()) and audit['maximum']['all_native_m']<=.001
                          and max(summary.get('gpu_health',{}).get('peak_penetration_m',[0.]))<=.001
                          and max(summary.get('gpu_health',{}).get('peak_solver_iterations',[0]))<sim.parcel.cfg.iterations)
        groups.append((sim,batch,qpos,qvel,ctrl))
        record=dict(group=entry['group'],seed=batch.seed,trial=entry['path'],profile=cfg.bag_profile,
                    model=asdict(sim.parcel.cfg),phase=phase,source_stage=stage,state_index=index,
                    cpu_scene_compile_ik_s=cpu_scene_s,source_trial_geometry_valid=sim.source_valid,material_checks=sim.material_checks,
                    upload_s=upload,contact_capacity=contact_capacity,constraint_capacity=constraint_capacity,
                    setup_components=batch.setup_timing,nq=sim.m.nq,nu=sim.m.nu)
        write(args.output/f'group-{len(groups)-1}.json',record)
        write(args.output/'loaded.json',dict(loaded_groups=len(groups),worlds=args.worlds_per_group*len(groups),
              worlds_per_group=args.worlds_per_group,device_memory_MiB=memory()))
        batch.step()
    block=10
    if args.steps%block:raise ValueError('steps must be divisible by ten')
    begin=time.perf_counter()
    with wp.ScopedCapture() as capture:
        for _ in range(block):
            for _,batch,*_ in groups:batch.step()
    graph=capture.graph;wp.synchronize();capture_s=time.perf_counter()-begin
    for _ in range(3):wp.capture_launch(graph)
    wp.synchronize()
    results=[]
    for repeat in range(3):
        begin=time.perf_counter()
        for _,batch,q,v,c in groups:batch.reset(initial_qpos=q,initial_qvel=v,initial_ctrl=c)
        wp.synchronize();reset_s=time.perf_counter()-begin
        begin=time.perf_counter()
        for _ in range(args.steps//block):wp.capture_launch(graph)
        wp.synchronize();wall=time.perf_counter()-begin
        health=[]
        for index,(sim,batch,*_) in enumerate(groups):
            diag=batch.diagnostics();parameters=batch.parameters.numpy()
            friction=friction_evidence(batch,sim,args.output/f'group-{index}-effective-friction-{repeat}.npz')
            masses=batch.m.body_mass.numpy()[:,sim.parcel.payload_bodies]
            inertia=batch.m.body_inertia.numpy()[:,sim.parcel.payload_bodies]
            joints=batch.bend_joints.numpy();dofs=batch.bend_dofs.numpy()
            checks=dict(sim.material_checks,
                mass=bool(np.allclose(masses.sum(1),sim.parcel.cfg.payload_mass*parameters[:,0],rtol=1e-6)),
                inertia=bool(np.allclose(inertia,batch.base_inertia.numpy()[None,sim.parcel.payload_bodies]*parameters[:,0,None,None],rtol=1e-6)),
                com=bool(np.allclose(batch.m.body_ipos.numpy()[:,sim.parcel.payload_bodies],sim.m.body_ipos[None,sim.parcel.payload_bodies])),
                payload_friction=bool(np.allclose(batch.m.geom_friction.numpy()[:,batch.payload_geoms.numpy(),0],parameters[:,1,None])),
                table_friction=bool(np.allclose(batch.m.geom_friction.numpy()[:,sim.table_geom,0],parameters[:,2])),
                joint_stiffness=bool(np.allclose(batch.m.jnt_stiffness.numpy()[:,joints],batch.base_k.numpy()[None,joints]*parameters[:,3,None])),
                joint_damping=bool(np.allclose(batch.m.dof_damping.numpy()[:,dofs],batch.base_damping.numpy()[None,dofs]*parameters[:,4,None])))
            consistent=all(checks.values())
            np.savez_compressed(args.output/f'group-{index}-parameters-{repeat}.npz',parameters=parameters,
                mass=masses,inertia=inertia,com=batch.m.body_ipos.numpy()[:,sim.parcel.payload_bodies],
                inertia_quat=batch.m.body_iquat.numpy()[:,sim.parcel.payload_bodies],
                geom_friction=batch.m.geom_friction.numpy(),joint_stiffness=batch.m.jnt_stiffness.numpy(),
                joint_damping=batch.m.dof_damping.numpy())
            positions=batch.d.qpos.numpy();geometry=[]
            for world in range(args.worlds_per_group):
                sim.d.qpos[:]=positions[world]
                mujoco.mj_kinematics(sim.m,sim.d);mujoco.mj_comPos(sim.m,sim.d);mujoco.mj_flex(sim.m,sim.d)
                geo=film_payload_geometry(sim.m,sim.d,sim.parcel.envelope['faces'])
                support=film_box_geometry(sim.m,sim.d,sim.parcel.envelope['faces'],sim.table_geom)
                geometry.append((max(geo['geometric_depth_upper_m'],support['geometric_depth_upper_m']),geo['outside_corners']))
            finite_zero_overflow=not any(diag['overflow_bits']) and not any(diag['nonfinite'])
            numeric=finite_zero_overflow and max(diag['peak_solver_iterations'])<sim.m.opt.iterations
            physical=sim.source_valid and numeric and consistent and max(diag['peak_penetration_m'])<=.001 and max(diag['peak_solver_iterations'])<sim.m.opt.iterations and all(g[0]<=.001 and not g[1] for g in geometry)
            health.append(dict(group=index,finite_zero_overflow=finite_zero_overflow,numerically_valid=numeric,physically_valid=physical,
                               parameters_applied=consistent,applied_checks=checks,effective_friction=friction,peak_penetration_m=max(diag['peak_penetration_m']),
                               contacts=diag['native_nacon'],peak_iterations=max(diag['peak_solver_iterations']),
                               overflow_worlds=sum(x!=0 for x in diag['overflow_bits']),nonfinite_worlds=sum(diag['nonfinite'])))
            write(args.output/f'group-{index}-health-{repeat}.json',dict(seed=batch.seed,episode=batch.episode-1,diagnostics=diag,geometry=geometry))
        total=len(groups)*args.worlds_per_group
        results.append(dict(repeat=repeat,total_worlds=total,groups=len(groups),wall_s=wall,reset_s=reset_s,
            environment_steps_per_s=total*args.steps/wall,batch_step_ms=wall/args.steps*1000,
            per_environment_realtime=args.steps*groups[0][0].built.cfg.dt/wall,
            simulated_seconds_per_environment=args.steps*groups[0][0].built.cfg.dt,
            finite_zero_overflow=all(g['finite_zero_overflow'] for g in health),numerically_valid=all(g['numerically_valid'] for g in health),
            physically_valid=all(g['physically_valid'] for g in health),health=health,
            device_memory_MiB=memory(),mempool_peak_MiB=wp.get_mempool_used_mem_high('cuda:0')/2**20))
        write(args.output/'samples.json',results)
    write(args.output/'result.json',dict(status='completed',groups=len(groups),worlds=total,
           worlds_per_group=args.worlds_per_group,capture_compile_s=capture_s,samples=results,
           median_environment_steps_per_s=float(np.median([r['environment_steps_per_s'] for r in results])),
           numerically_valid=all(r['numerically_valid'] for r in results),
           physically_valid=all(r['physically_valid'] for r in results)))


if __name__=='__main__':main()
