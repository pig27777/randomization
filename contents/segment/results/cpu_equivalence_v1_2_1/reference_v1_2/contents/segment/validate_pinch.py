"""Ordered E1--E6 acceptance of the top-pinch effect; use a new run-id under flock."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'results/pinch_effect_v1_2'


def save(path, data):
    from response_distribution import save as write
    write(path, data)


def metadata(out, runtime, args):
    import mujoco
    import mj_adapter as A
    paths = sorted(runtime.glob('*.py')) + sorted((runtime / 'tests').glob('*.py'))
    data = dict(arguments=vars(args), mujoco=mujoco.__version__, solver=A.SOLVER, timestep=A.BUSHING_DT,
                task_protocol_sha256=hashlib.sha256((ROOT.parents[1]/'test/policy/README.md').read_bytes()).hexdigest(),
                source_sha256={str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(out / 'metadata.json', data)


def response(category, test):
    import model
    import mj_adapter as A
    import protocols as P
    k = model.ANCHORS[category]
    desc = model.build(k, omega_max=A.OMEGA_MAX)
    digest, frames, xml_hash = hashlib.sha256(), 0, None
    original = P.Runner.step
    def observed(runner, *args, **kwargs):
        nonlocal frames, xml_hash
        value = original(runner, *args, **kwargs)
        if value:
            digest.update(runner.sim.d.qpos.tobytes())
            digest.update(runner.sim.d.qvel.tobytes())
            frames += 1
            if xml_hash is None:
                xml_hash = hashlib.sha256(runner.sim.built.xml.encode()).hexdigest()
        return value
    P.Runner.step = observed
    try:
        result = (P.r1(k, test[-1], desc=desc) if test.startswith('r1') else
                  P.r2(k, 'edge', desc=desc) if test == 'r2' else P.r3(k, desc=desc))
    finally:
        P.Runner.step = original
    return dict(result=result, trajectory_sha256=digest.hexdigest(), frames=frames, xml_sha256=xml_hash)


def worker(job):
    category = job.get('category')
    out = Path(job['out']); out.mkdir(); os.chdir(out)
    start = time.perf_counter()
    row = {k: v for k, v in job.items() if k != 'out'}
    try:
        import model
        if job['kind'] == 'response':
            row.update(response(category, job['test']))
        else:
            from top_grasp import evaluate, measure_r4
            k = model.Kappa.from_dict(job['kappa']) if 'kappa' in job else model.ANCHORS[category]
            row['kappa'] = k.to_dict()
            if 'coefficient_scale' in job:
                k = replace(k, pinch_coeff=k.pinch_coeff * job['coefficient_scale'])
                row['kappa'] = k.to_dict()
            if job['kind'] == 'r4':
                row['result'] = measure_r4(k)
            else:
                if job['kind'] == 'random':
                    import mj_adapter as A
                    import protocols as P
                    from response_distribution import observe_protocols
                    observe_protocols(P)
                    row['rest'] = P.rest(k, desc=model.build(k, omega_max=A.OMEGA_MAX))
                    if row['rest']['fail']:
                        row['rest']['failure_state'] = P.measurement_runner.failure_state
                row['result'] = evaluate(k, job['grasp'], job['seed'])
            row['confirmed_failure'] = bool(row.get('rest', {}).get('fail') or row['result']['outcome'] == 'sim_fail')
        row['execution_error'] = False
    except Exception:
        row.update(execution_error=True, traceback=traceback.format_exc())
    row['total_wall_s'] = time.perf_counter() - start
    save(out / 'result.json', row)
    return row


def jobs_run(jobs, workers):
    rows = []
    if workers == 1:
        iterator = map(worker, jobs)
        for row in iterator:
            rows.append(row)
            print(f"{row.get('category')} {row.get('test',row['kind'])}: error={row['execution_error']} "
                  f"outcome={row.get('result',{}).get('outcome')} {row['total_wall_s']:.2f}s", flush=True)
    else:
        with Pool(workers) as pool:
            for row in pool.imap_unordered(worker, jobs):
                rows.append(row)
                print(f"{row.get('category')} {row.get('test',row['kind'])}: error={row['execution_error']} "
                      f"outcome={row.get('result',{}).get('outcome')} {row['total_wall_s']:.2f}s", flush=True)
    return rows


def compare(out):
    old = json.loads((BASE / 'e2_reference/summary.json').read_text())['rows']
    new = json.loads((BASE / 'e2_candidate/summary.json').read_text())['rows']
    old = {(r['category'], r['test']): r for r in old}
    checks = []
    for b in new:
        key = b['category'], b['test']; a = old[key]
        physical = lambda r: {k:v for k,v in r['result'].items() if k not in ('rtf','wall_s')}
        checks.append(dict(category=key[0], test=key[1],
            passed=not a['execution_error'] and not b['execution_error'] and physical(a)==physical(b)
                   and a['trajectory_sha256']==b['trajectory_sha256'] and a['xml_sha256']==b['xml_sha256']
                   and a['frames']==b['frames']))
    passed = len(checks)==12 and all(r['passed'] for r in checks)
    save(out / 'summary.json', dict(passed=passed, rows=checks))
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('units','response','compare','r4','scan','random','speed'))
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--count', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261119)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--categories', nargs='+', default=['hard','bendable','compressible'])
    ap.add_argument('--tests', nargs='+', default=['r1y','r1z','r2','r3'])
    args = ap.parse_args()
    out = (BASE / args.run_id).resolve(); out.mkdir()
    runtime = (BASE / 'reference_v1/contents/segment').resolve() if args.reference else ROOT
    sys.path.insert(0, str(runtime))
    metadata(out, runtime, args)
    if args.action == 'units':
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        save(out/'summary.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
        return 0 if result.wasSuccessful() else 2
    if args.action == 'compare':
        return 0 if compare(out) else 2
    import model
    jobs = []
    if args.action == 'response':
        jobs = [dict(kind='response',category=c,test=t) for c in args.categories for t in args.tests]
    elif args.action == 'r4':
        jobs = [dict(kind='r4',category=c,repeat=i) for c in args.categories for i in range(args.count)]
    elif args.action == 'scan':
        for c in args.categories:
            forces = (5.,10.,15.) if c=='compressible' else (5.,15.,25.)
            scales = (1.,) if c=='hard' else (.25,.5,1.)
            seeds = (20261116,) if c=='hard' else (20261116,20261117,20261118)
            for w,d,f,scale,seed in itertools.product((.04,.06,.08),(.005,.01,.02,.03),forces,scales,seeds):
                jobs.append(dict(kind='scan',category=c,seed=seed,coefficient_scale=scale,
                                 grasp=dict(w=w,d=d,F=f)))
    elif args.action == 'random':
        from sampling import sample_kappa
        rng = np.random.default_rng(args.seed)
        for i in range(args.count):
            k = sample_kappa(rng,'mixed')
            jobs.append(dict(kind='random',category=k.category,kappa=k.to_dict(),seed=args.seed+i,
                grasp=dict(u=float(rng.uniform(-.5,.5)),v=float(rng.uniform(-.5,.5)),psi=float(rng.uniform(0,math.pi)),
                           w=float(rng.uniform(.020,.095)),d=float(rng.uniform(0,.040)),F=float(rng.uniform(2,50)),
                           v_lift=float(rng.uniform(.05,.3)),a_max=float(rng.uniform(.5,4)))))
    else:
        jobs = [dict(kind='speed',category=c,seed=i,grasp=dict(w=.04,d=.015,F=10.))
                for c in args.categories for i in range(args.count)]
        args.workers = 1
    for i, job in enumerate(jobs):job['index']=i;job['out']=str(out/f'case_{i:04d}')
    rows = jobs_run(jobs,args.workers)
    summary = dict(rows=rows,total=len(rows),execution_errors=sum(r['execution_error'] for r in rows))
    passed = not summary['execution_errors']
    if args.action == 'r4':
        for r in rows:
            if r['execution_error']:continue
            target=r['kappa']['pinch_coeff'];value=r['result']['measured_pinch_coeff']
            r['relative_error']=abs(value-target)/target if target else abs(value)
            r['passed']=r['relative_error']<=.10 and (r['result']['outcome']=='pull_off' if target else r['result']['outcome']=='grasp_fail')
        passed = passed and all(r.get('passed',False) for r in rows)
    elif args.action == 'scan':
        groups={}
        for c in args.categories:
            groups[c]={}
            for scale in ((1.,) if c=='hard' else (.25,.5,1.)):
                part=[r for r in rows if r['category']==c and r['coefficient_scale']==scale]
                groups[c][str(scale)]=dict(outcomes=dict(Counter(r.get('result',{}).get('outcome','error') for r in part)),
                    successes=[dict(seed=r['seed'],**r['grasp']) for r in part if r.get('result',{}).get('outcome')=='success'])
            def successes(scale):return {json.dumps(p,sort_keys=True) for p in groups[c][str(scale)]['successes']}
            if c=='hard':passed=passed and all(r.get('result',{}).get('outcome')=='grasp_fail' for r in rows if r['category']==c)
            else:
                monotone=successes(.25)<=successes(.5)<=successes(1.)
                slip=any(any(e['event']=='overload' for e in r.get('result',{}).get('pinch_events',[])) for r in rows if r['category']==c)
                groups[c]['monotone']=monotone;groups[c]['has_capacity_slip']=slip
                passed=passed and bool(successes(1.)) and monotone and slip
        summary['groups']=groups
        summary['static_following']={}
        for c in args.categories:
            values=[r['result']['static_attachment_error_max_mm'] for r in rows
                    if r['category']==c and not r['execution_error']
                    and r['result']['static_attachment_error_max_mm'] is not None]
            summary['static_following'][c]=dict(measured_cases=len(values),
                max_error_mm=max(values,default=None),passed=all(v<=5 for v in values))
            passed=passed and summary['static_following'][c]['passed']
    elif args.action == 'random':
        summary['confirmed_failures']=sum(r.get('confirmed_failure',False) for r in rows)
        passed=passed and summary['confirmed_failures']/args.count<=.01
    elif args.action == 'speed':
        baseline=dict(hard=.694657,bendable=6.791214,compressible=7.880028)
        summary['median_wall_s']={c:float(np.median([r['result']['wall_s'] for r in rows if r['category']==c and not r['execution_error']])) for c in args.categories}
        summary['ratios_to_side']={c:v/baseline[c] for c,v in summary['median_wall_s'].items()}
        passed=passed and all(v<=1.5 for v in summary['ratios_to_side'].values())
    summary['passed']=passed
    save(out/'summary.json',summary)
    return 0 if passed else 2


if __name__=='__main__':
    import math
    raise SystemExit(main())
