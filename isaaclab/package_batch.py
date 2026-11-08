"""L5 immutable case plan, per-attempt GPU queueing, and auditable partial summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'isaaclab'


def fingerprints():
    paths=sorted(SOURCE.glob('package_*.py'))+[SOURCE/'pinch_effect.py',SOURCE/'segment_adapter.py']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(directory,assets):
    inputs=SOURCE/'results/loose_package/reference_inputs'
    manifest=json.loads((assets/'manifest.json').read_text())
    cases=[]
    for variant in ('baseline','mass5','mass10','coarse150','no_self_contact'):
        for category in ('hard','compressible'):
            for fill in ('loose','medium','tight'):
                geometry=next(c for c in manifest['cases'] if c['category']==category and c['fill']==fill)
                for grasp in ('side','top'):
                    for seed in (20261105,20261106,20261107):
                        name=f'{variant}-{category}-{fill}-{grasp}-{seed}'
                        cases.append(dict(name=name,variant=variant,category=category,fill=fill,
                            grasp=grasp,seed=seed,scale=geometry['scale']))
    directory.mkdir(parents=True,exist_ok=False)
    plan=dict(cases=cases,source_sha256=fingerprints(),input_root=str(inputs),
        input_sha256=json.loads((inputs/'manifest.json').read_text()),assets=str(assets),
        area_definition=manifest['area_definition'],protocol='v2.2',pinch_coeff=2.)
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')


def run(directory,max_cases):
    plan=json.loads((directory/'plan.json').read_text());records=[]
    env=dict(os.environ,RANDOM_L5_INPUT_ROOT=plan['input_root'])
    for name,sha in plan['input_sha256'].items():
        if hashlib.sha256((Path(plan['input_root'])/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'frozen input changed: {name}')
    for case in plan['cases'][:max_cases]:
        if fingerprints()!=plan['source_sha256']:raise RuntimeError('source changed after plan creation')
        output=directory/(case['name']+'.json');log=output.with_suffix('.log')
        if output.exists():
            result=json.loads(output.read_text())['result'];records.append(dict(case=case,result=result));continue
        cache=directory/f"cache-{case['variant']}-{case['category']}-{case['fill']}.npz"
        command=['flock','/tmp/claude-gpu.lock',sys.executable,'-B',str(SOURCE/'package_grasp.py'),
            '--category',case['category'],'--grasp',case['grasp'],'--seed',str(case['seed']),
            '--scale',str(case['scale']),'--cache',str(cache),'--output',str(output)]
        if case['variant'] in ('mass5','mass10'):command+=['--mass-scale',case['variant'][4:]]
        if case['variant']=='coarse150':command+=['--coarse',str(Path(plan['assets'])/'kraft_150.npz')]
        if case['variant']=='no_self_contact':command+=['--no-self-contact']
        start=time.perf_counter()
        with log.open('x') as stream:completed=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result=json.loads(output.read_text())['result'] if output.exists() else None
        records.append(dict(case=case,exit_code=completed.returncode,result=result,
                            queue_and_process_wall_s=time.perf_counter()-start))
        summary=dict(planned=len(plan['cases']),requested=min(max_cases,len(plan['cases'])),
            executed=len(records),pending=len(plan['cases'])-len(records),records=records,
            stop_reason=None if completed.returncode==0 else 'simulation or implementation failure; inspect log')
        temporary=directory/'summary.tmp'
        temporary.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        temporary.replace(directory/'summary.json')
        print(json.dumps(dict(case=case['name'],exit_code=completed.returncode,
                              outcome=None if result is None else result['outcome'])),flush=True)
        if completed.returncode!=0:return completed.returncode
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--max-cases',type=int,default=180)
    args=parser.parse_args();directory=args.directory.resolve()
    if args.max_cases<1:parser.error('--max-cases must be positive')
    if args.prepare:
        if args.assets is None:parser.error('--prepare requires --assets')
        prepare(directory,args.assets.resolve())
    else:raise SystemExit(run(directory,args.max_cases))
