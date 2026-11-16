"""Serial, resumable orchestration of the full-YAM randomized GPU experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
DRIVER=ROOT/'test/bag_model/physical_parcel_yam_benchmark.py'
SCHEDULE=[(0,'top'),(9,'edge'),(17,'edge'),(1,'corner'),(10,'top'),(14,'top'),(3,'side'),(11,'top'),(16,'top')]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--skip-trials',action='store_true')
    p.add_argument('--mixed-only',action='store_true')
    p.add_argument('--gpu-trial-indices',type=int,nargs='*',default=[])
    p.add_argument('--max-worlds',type=int,default=65536)
    args=p.parse_args();run=args.run.resolve()
    if args.mixed_only:args.skip_trials=True
    env=dict(os.environ,PYTHONPATH=f'{ROOT}/test/bag_model:{ROOT}/experience:{ROOT}',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def execute(command,log,timeout=3600):
        command=[str(x) for x in command]
        with (run/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=command))+'\n')
        with log.open('w') as stream:
            try:
                completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
                return completed.returncode
            except subprocess.TimeoutExpired:
                stream.write('\nWORKER TIME LIMIT\n');return 124
    trials=[]
    for index,(group,region) in enumerate(SCHEDULE):
        directory=run/'trials'/f'trial-{index}'
        if index==0:directory=run/'trials/trial-0-kraft-box-top'
        if not (directory/'summary.json').exists() and not args.skip_trials:
            if directory.exists():
                # An interrupted partial worker is retained as evidence; resume in a new directory.
                count=1
                while directory.with_name(directory.name+f'-resume-{count}').exists():count+=1
                directory=directory.with_name(directory.name+f'-resume-{count}')
            command=[sys.executable,'-u','-B',DRIVER,'trial','--groups',run/'preparation',
                     '--group',group,'--region',region,'--output',directory]
            if index in args.gpu_trial_indices:command+=['--gpu']
            print(json.dumps(dict(event='trial_start',index=index,group=group,region=region,gpu=index in args.gpu_trial_indices)),flush=True)
            code=execute(command,run/f'trial-{index}-worker.log')
            if code:print(json.dumps(dict(event='trial_error',index=index,returncode=code)),flush=True)
        if (directory/'summary.json').exists():
            result=json.loads((directory/'summary.json').read_text())
            trials.append(dict(index=index,group=group,region=region,path=str(directory.relative_to(ROOT)),
                               outcome=result['outcome'],phase=result['phase'],wall_s=result['wall_s'],sim_s=result['sim_time_s'],
                               engine=result['execution_engine']))
            print(json.dumps(dict(event='trial_done',**trials[-1])),flush=True)
        (run/'trial_summary.json').write_text(json.dumps(trials,indent=2))
    if not trials:raise RuntimeError('no completed trial records')
    if not args.mixed_only:
        measurements=[]
        def benchmark(trial,phase,worlds,label):
            output=run/'gpu'/label
            if (output/'result.json').exists():
                result=json.loads((output/'result.json').read_text())
                code=0
            else:
                if output.exists():
                    output=output.with_name(output.name+'-retry')
                command=[sys.executable,'-u','-B',DRIVER,'benchmark','--trial',ROOT/trial['path'],
                         '--phase',phase,'--worlds',worlds,'--steps',100,'--output',output]
                print(json.dumps(dict(event='gpu_start',group=trial['group'],phase=phase,worlds=worlds)),flush=True)
                code=execute(command,run/f'{label}.log',timeout=1800)
                result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else dict(status='worker_failure')
            row=dict(group=trial['group'],trial=trial['path'],phase=phase,worlds=worlds,returncode=code,
                     path=str(output.relative_to(ROOT)),**{k:result.get(k) for k in ['status','numerically_valid','physically_valid','median_environment_steps_per_s']})
            measurements.append(row)
            (run/'gpu_summary.json').write_text(json.dumps(measurements,indent=2))
            print(json.dumps(dict(event='gpu_done',**row)),flush=True)
            return code,result,output
        # Small loaded-state check for each actual model and phase before scaling.
        for trial in trials:
            result=json.loads((ROOT/trial['path']/'summary.json').read_text())
            for phase,stage in result['stages'].items():
                if stage.get('missing'):continue
                benchmark(trial,phase,8,f'group-{trial["group"]}-{phase}-n8')
        # Choose the actual strongest load-bearing state; retain evidence if none lifted.
        candidates=[]
        for trial in trials:
            result=json.loads((ROOT/trial['path']/'summary.json').read_text())
            for phase,stage in result['stages'].items():
                if stage.get('missing'):continue
                bearing=stage['support_N']<.05 and stage['com'][2]-result['initial_com'][2]>=.05
                score=(int(bearing),stage['grip_N'])
                candidates.append((score,trial,phase))
        _,chosen,phase=max(candidates,key=lambda x:x[0])
        lower=0;upper=None
        for worlds in [1,8,32,128,512,1024,2048,4096,8192,16384,32768,65536]:
            if worlds>args.max_worlds:break
            code,result,output=benchmark(chosen,phase,worlds,f'capacity-{phase}-n{worlds}')
            if code:
                log=(run/f'capacity-{phase}-n{worlds}.log').read_text()
                resource=any(word in log.lower() for word in ['out of memory','out_of_memory','failed to allocate','allocation failed','cuda error 2']) or code in (-9,137)
                if not resource:
                    (run/'capacity_error.json').write_text(json.dumps(dict(worlds=worlds,reason='worker error is not a memory boundary',log=str(run/f'capacity-{phase}-n{worlds}.log')),indent=2))
                    break
                upper=worlds;break
            lower=worlds
        if upper is not None:
            for _ in range(2):
                worlds=(lower+upper)//2
                code,result,output=benchmark(chosen,phase,worlds,f'capacity-refine-{phase}-n{worlds}')
                if code:upper=worlds
                else:lower=worlds
        (run/'capacity_summary.json').write_text(json.dumps(dict(selected_trial=chosen,phase=phase,max_executed_worlds=lower,
               first_failed_worlds=upper,finite_refinements=2 if upper else 0,
               memory_boundary_found=upper is not None,requested_cap=args.max_worlds,
               note='Max executed is not necessarily numerically or physically valid. Loading-only status is recorded separately.'),indent=2))

    mixed=[]
    low=0;high=None
    def mixed_worker(per_group,label):
        output=run/'mixed'/label
        if (output/'result.json').exists():
            result=json.loads((output/'result.json').read_text());code=0
        else:
            command=[sys.executable,'-u','-B',ROOT/'test/bag_model/physical_parcel_mixed_gpu.py',
                     '--run',run,'--worlds-per-group',per_group,'--output',output]
            print(json.dumps(dict(event='mixed_start',worlds_per_group=per_group)),flush=True)
            code=execute(command,run/f'{label}.log',timeout=1800)
            result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
        row=dict(worlds_per_group=per_group,returncode=code,path=str(output.relative_to(ROOT)),
                 **{k:result.get(k) for k in ['worlds','groups','median_environment_steps_per_s','numerically_valid','physically_valid']})
        mixed.append(row);(run/'mixed_summary.json').write_text(json.dumps(mixed,indent=2))
        print(json.dumps(dict(event='mixed_done',**row)),flush=True)
        return code
    for per_group in [1,8,32,128,512,1024,2048,4096,8192]:
        code=mixed_worker(per_group,f'mixed-per-group-{per_group}')
        if code:
            log=(run/f'mixed-per-group-{per_group}.log').read_text().lower()
            if any(word in log for word in ['out of memory','out_of_memory','failed to allocate','allocation failed','cuda error 2']):high=per_group
            break
        low=per_group
    if high:
        for index in range(2):
            count=(low+high)//2
            code=mixed_worker(count,f'mixed-refine-per-group-{count}')
            if code:high=count
            else:low=count


if __name__=='__main__':main()
