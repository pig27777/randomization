"""Small CPU diagnostics before spending the next parcel-heap trial budget."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

import parcel_heap as heap
from probe_feedback import digest,save

RUN = heap.RUN/'continuation'


def correct_curvature(sim,run):
    return heap.apply_curvature(sim,run)


def dofs_for_bodies(model,bodies):
    return np.concatenate([np.arange(model.body_dofadr[b],model.body_dofadr[b]+model.body_dofnum[b])
                           for b in bodies])


def force_probe(run):
    rows=[]
    for case in heap.CASES:
        robot=heap.scene(case,'top')
        sim=robot.sim; m,d=sim.m,sim.d
        dofs=dofs_for_bodies(m,sim.bag_bodies)
        before=d.qfrc_spring[dofs].reshape(-1,3).copy()
        changes=correct_curvature(sim,run)
        after=d.qfrc_spring[dofs].reshape(-1,3).copy()
        rows.append(dict(case=case,before_max_N=float(np.linalg.norm(before,axis=1).max()),
                         after_max_N=float(np.linalg.norm(after,axis=1).max()),
                         before_l2_N=float(np.linalg.norm(before)),after_l2_N=float(np.linalg.norm(after)),
                         metadata=changes))
    return rows


def profile(run,steps):
    records=[]
    mujoco.set_mjcb_time(time.perf_counter)
    for case in ('book','clothes','empty_envelope'):
        for solver in ('Newton','CG'):
            robot=heap.scene(case,'top')
            sim=robot.sim;m,d=sim.m,sim.d
            changes=correct_curvature(sim,run)
            m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON if solver=='Newton' else mujoco.mjtSolver.mjSOL_CG
            m.opt.iterations=100 if solver=='Newton' else 400
            max_iterations=0; max_penetration=0.
            start=time.perf_counter()
            for step in range(steps):
                robot.command(robot.initial_target,.095)
                sim.step()
                max_iterations=max(max_iterations,int(d.solver_niter.max()))
                max_penetration=max(max_penetration,max((-c.dist*1000 for c in d.contact),default=0.))
            wall=time.perf_counter()-start
            timers={name:float(d.timer[value].duration) for name,value in mujoco.mjtTimer.__members__.items()
                    if name!='mjNTIMER'}
            row=dict(case=case,solver=solver,steps=steps,wall_s=wall,max_solver_iterations=max_iterations,
                     budget=int(m.opt.iterations),max_all_contact_penetration_mm=max_penetration,
                     warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(d.warning) if w.number},
                     timers=timers,shell=sim.shell_points().tolist(),content=d.xipos[sim.content_bodies].tolist())
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('shell','content','timers')}),flush=True)
    mujoco.set_mjcb_time(None)
    comparisons=[]
    for baseline,candidate in zip(records[::2],records[1::2]):
        delta=float(np.linalg.norm(np.asarray(baseline['shell'])-candidate['shell'],axis=1).max()*1000)
        content=np.asarray(baseline['content'])-candidate['content']
        content_delta=float(np.linalg.norm(content,axis=1).max()*1000) if len(content) else 0.
        comparisons.append(dict(case=baseline['case'],shell_delta_mm=delta,content_delta_mm=content_delta,
                                speed_ratio=baseline['wall_s']/candidate['wall_s'],
                                passed=delta<=.1 and content_delta<=.1 and not candidate['warnings'] and
                                       candidate['max_solver_iterations']<candidate['budget']))
    return dict(rows=records,comparisons=comparisons)


def workspace():
    settings=replace(heap.HeapSettings(),stack_lift=.15,adaptive_close=True)
    rows=[]
    for case in ('book','empty_envelope'):
        for layout in ('top','gap'):
            for action in ('center','end','corner','side'):
                row=dict(case=case,layout=layout,action=action)
                try:
                    robot=heap.scene(case,layout,action,settings=settings)
                except RuntimeError as error:
                    if not str(error).startswith('YAM initial pose unreachable:'):raise
                    rows.append(dict(**row,passed=False,reason=str(error)));continue
                sim=robot.sim;m=sim.m; probe=mujoco.MjData(m)
                poses=[robot.initial_target,robot.grasp_target,robot.grasp_target+[0.,0.,.15],
                       robot.grasp_target+[-.1,0.,.15]]
                samples=[]
                for segment,(start,end) in enumerate(zip(poses,poses[1:])):
                    for a in np.linspace(0.,1.,11):
                        target=start+(end-start)*a
                        error=robot.solve(target,multistart=True)
                        probe.qpos[:]=sim.d.qpos
                        probe.qpos[robot.qadr]=robot.qtarget
                        mujoco.mj_fwdPosition(m,probe)
                        peak=0.;witness=None
                        for c in probe.contact:
                            if c.dist>=-1e-6:continue
                            g=c.geom; flex=c.flex
                            for j in range(2):
                                if g[j]<0 or not robot.robot_mask[g[j]]:continue
                                other=g[1-j]
                                obstacle=(other in sim.neighbor_geoms or other==sim.table_geom or
                                          (other>=0 and robot.robot_mask[other]) or
                                          (robot.nonfinger_mask[g[j]] and (other in sim.obj_geoms or flex[1-j]>=0)))
                                if obstacle and -c.dist*1000>peak:
                                    peak=-float(c.dist)*1000
                                    witness=[m.geom(int(x)).name if x>=0 else 'flex' for x in g]
                        samples.append(dict(segment=segment,fraction=float(a),target=target.tolist(),
                                            **error,collision_mm=peak,witness=witness))
                passed=all(s['position_mm']<=1 and s['rotation_deg']<=1 and s['collision_mm']==0 for s in samples)
                rows.append(dict(**row,passed=passed,settings=asdict(settings),samples=samples))
                print(json.dumps({**row,'passed':passed,'ik_max_mm':max(s['position_mm'] for s in samples),
                                  'collision_max_mm':max(s['collision_mm'] for s in samples)}),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('force','profile','workspace'))
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    heap.install_backend(args.run)
    path=args.run/(args.stage+(f'_{args.steps}' if args.stage=='profile' else '')+'.json')
    if path.exists(): raise FileExistsError(path)
    snapshot=args.run/'diagnostic_source'/path.stem;snapshot.mkdir(parents=True)
    for file in (__file__,heap.__file__):
        (snapshot/Path(file).name).write_bytes(Path(file).read_bytes())
    metadata=dict(source_sha256=digest(__file__),driver_sha256=digest(heap.__file__),
                  helper_sha256=digest(args.run/'source_helpers/flex_curvature.py'),host=platform.node(),
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    result=(force_probe(args.run) if args.stage=='force' else
            profile(args.run,args.steps) if args.stage=='profile' else workspace())
    save(path,dict(**metadata,result=result))
    if args.stage=='force':print(json.dumps([{k:v for k,v in r.items() if k!='metadata'} for r in result]))


if __name__=='__main__':main()
