"""Reproducible F6/F7 jobs. Formal timing commands must be run under the shared GPU flock."""
import argparse
import hashlib
import json
import math
import os
import time
import itertools
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from loose_protocols import KRAFT_SIZE, content_kappa, evaluate, r2, r3
from mj_loose import LooseBag

OUT = Path(__file__).resolve().parent / 'outputs/round3_flex'
CACHE = OUT / 'cache'
BASE = LooseBag(kind='flex', size=KRAFT_SIZE, edge=.035, radius=.005)
CONFIGS = {'base': BASE, 'precision': replace(BASE, tolerance=1e-6, iterations=30, ls_iterations=10),
           'partition': replace(BASE, bottom_edge=.050)}


def check_deadline():
    deadline = datetime.fromisoformat(json.loads((OUT / 'run_manifest.json').read_text())['deadline'])
    if datetime.now(deadline.tzinfo) >= deadline:
        raise TimeoutError('F6 + F7 48-hour deadline reached')


def arm_deadline():
    """Stop even an in-flight case at the user's hard F6/F7 deadline."""
    deadline=datetime.fromisoformat(json.loads((OUT/'run_manifest.json').read_text())['deadline'])
    remaining=(deadline-datetime.now(deadline.tzinfo)).total_seconds()
    if remaining<=0: raise TimeoutError('F6 + F7 48-hour deadline reached')
    def stop(signum,frame): raise TimeoutError('F6 + F7 48-hour deadline reached')
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,remaining)


def f3_inputs():
    # Exact inputs of round 2 F3, including the six-decimal orientation.
    for name, u, v, psi in [('centre', 0., 0., 1.570796), ('empty_end', .42, 0., 1.570796),
                            ('corner', .42, .42, .785398)]:
        for seed in range(3):
            yield name, [u, v, psi, 20., .1, 2.], seed


def spec(bag, cache=False):
    return dict(kappa=content_kappa('box_loose').to_dict(), cls='hard', bag=asdict(bag), dt=.0008,
                cache_dir=str(CACHE) if cache else None)


def one(job):
    arm_deadline()
    check_deadline()
    config, test, params, seed, cached = job
    bag = CONFIGS[config]
    start = time.perf_counter()
    if test == 'f7':
        from film_pinch import evaluate_film
        params, coefficient = params
        result = evaluate_film(content_kappa('box_loose'), bag, params, coefficient, seed,
                               cache_dir=CACHE)
    elif test in ('r2', 'r3'):
        result = (r2 if test == 'r2' else r3)('box_loose', bag, dt=.0008, **({'end': 1} if test == 'r2' else {}))
    else:
        digest = hashlib.sha256()
        def observe(s):
            digest.update(s.d.qpos.tobytes())
            digest.update(s.d.qvel.tobytes())
            digest.update(s.d.qacc_warmstart.tobytes())
        result = evaluate(spec(bag, cached), params, seed, observer=observe if test == 'cache_check' else None)
        if test == 'cache_check':
            result['trajectory_sha256'] = digest.hexdigest()
    return dict(result, config=config, test=test, params=params, seed=seed, cached=cached,
                total_wall_s=time.perf_counter()-start)


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['behavior', 'cache_check', 'speed', 'f7', 'f7_smoke', 'timing_suite', 'soft_reference'])
    p.add_argument('--config', choices=CONFIGS, default='base')
    p.add_argument('--procs', type=int, default=28)
    p.add_argument('--cached', action='store_true')
    p.add_argument('--name', default=None)
    p.add_argument('--pause-parent', type=int, default=None, help='temporarily pause this run\'s F7 workers during timing')
    a = p.parse_args()
    arm_deadline()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'soft_reference':
        path=OUT/'soft_reference.jsonl'
        if path.exists(): raise FileExistsError(path)
        result=r3('bendable',replace(BASE,size=None,fill=.75),dt=.0004)
        result['throughput_measurement']=False
        append(path,result)
        print(json.dumps(result),flush=True)
        return
    if a.mode == 'timing_suite':
        workers = []
        if a.pause_parent:
            for proc in Path('/proc').iterdir():
                if proc.name.isdigit():
                    try:
                        status = (proc/'status').read_text()
                    except FileNotFoundError:
                        continue
                    if f'PPid:\t{a.pause_parent}\n' in status:
                        workers.append(int(proc.name))
            for pid in workers: os.kill(pid, signal.SIGSTOP)
        try:
            for cfg, cached in [('base',False),('base',True),('precision',False),('partition',False)]:
                command=['flock','/tmp/claude-gpu.lock',sys.executable,'-B',__file__,'speed','--config',cfg]
                if cached: command.append('--cached')
                subprocess.run(command,check=True)
        finally:
            for pid in workers:
                try: os.kill(pid,signal.SIGCONT)
                except ProcessLookupError: pass
        return
    if a.mode.startswith('f7'):
        levels = itertools.product([0.,.4],[.040,.060,.090],[.006,.012],[10.,20.],[.5,1.,2.],range(3))
        jobs = [(a.config,'f7',([u,0.,math.pi/2,w,d,F,.1,2.],coeff),seed,True)
                for u,w,d,F,coeff,seed in levels]
        if a.mode == 'f7_smoke':
            jobs = [(a.config,'f7',([u,0.,math.pi/2,.060,.012,10.,.1,2.],c),0,True)
                    for u,c in itertools.product([0.,.4],[.1,1.,2.])]
        else:
            jobs += [(a.config,'f7',([.4,0.,math.pi/2,.060,.012,10.,.1,2.],.1),seed,True) for seed in range(3)]
    elif a.mode == 'speed':
        rng = np.random.default_rng(0)
        ranges = [(-.5,.5),(-.5,.5),(0,math.pi),(2,50),(.05,.3),(.5,4)]
        jobs = [(a.config, 'evaluate', [float(rng.uniform(lo, hi)) for lo, hi in ranges],
                 int(rng.integers(1 << 30)), a.cached) for _ in range(28)]
    else:
        jobs = [(a.config, a.mode if a.mode == 'cache_check' else 'evaluate', params, seed, cache)
                for _, params, seed in f3_inputs()
                for cache in ([False, True] if a.mode == 'cache_check' else [a.cached])]
        if a.mode == 'behavior':
            jobs += [(a.config, t, None, 0, False) for t in ('r2', 'r3')]
    name = a.name or (a.config + '_' + a.mode + ('_cached' if a.cached else ''))
    path = OUT / (name + '.jsonl')
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    wall = time.perf_counter()
    rows = []
    with Pool(a.procs) as pool:
        for result in pool.imap_unordered(one, jobs, chunksize=1):
            append(path, result)
            rows.append(result)
            print(json.dumps({k: result.get(k) for k in ('config','test','seed','cached','outcome','fail','penetration_mm','slide_mm','sink_mm')}), flush=True)
    elapsed = time.perf_counter() - wall
    summary = dict(mode=a.mode, config=a.config, cached=a.cached, procs=a.procs, n=len(rows),
                   wall_s=elapsed, evals_per_s=len(rows)/elapsed if a.mode == 'speed' else None,
                   max_ncon=max(r['max_ncon'] for r in rows), max_nefc=max(r['max_nefc'] for r in rows),
                   outcomes={v:sum(r.get('outcome') == v for r in rows) for v in {r.get('outcome') for r in rows}},
                   finished_at=datetime.now().astimezone().isoformat())
    append(OUT / 'summaries.jsonl', dict(summary, name=name))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
