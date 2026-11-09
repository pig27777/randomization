"""Small CPU diagnostics before spending the next parcel-heap trial budget."""
import argparse
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('force','profile'))
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
    result=force_probe(args.run) if args.stage=='force' else profile(args.run,args.steps)
    save(path,dict(**metadata,result=result))
    if args.stage=='force':print(json.dumps([{k:v for k,v in r.items() if k!='metadata'} for r in result]))


if __name__=='__main__':main()
