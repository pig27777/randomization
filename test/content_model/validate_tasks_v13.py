"""G3/G4 continuation with retained raw results, overflow checks and cumulative stability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))

import validate_warp_v13 as V


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=1,allow_nan=False,
        default=lambda x:x.item() if hasattr(x,'item') else str(x))


def failure_distribution(rows):
    """Retain failures and the corresponding full input distribution; no exclusion by kappa."""
    records=[]
    for r in rows:
        k=r['kappa']
        records.append(dict(label=r['label'],category=k.get('category'),segments=k['segments'],size=k['size'],
            squeeze_y=k['squeeze_stiffness_y'],cpu_sim_fail=r['cpu_outcome']=='sim_fail',
            gpu_sim_fail=r['gpu_outcome']=='sim_fail',cpu_reason=r['cpu_reason'],gpu_reason=r['gpu_reason'],
            cpu_failure_state=r.get('cpu_failure_state'),gpu_failure_state=r['gpu_failure_state']))
    summary={}
    for category in sorted({r['category'] for r in records}):
        group=[r for r in records if r['category']==category]
        summary[category]=dict(total=len(group),cpu_failures=sum(r['cpu_sim_fail'] for r in group),
            gpu_failures=sum(r['gpu_sim_fail'] for r in group))
    return dict(by_category=summary,inputs=records,failures=[r for r in records if r['cpu_sim_fail'] or r['gpu_sim_fail']],
        caveat='Small failure counts cannot establish concentration; retain all input values for comparison.')


def e4_specs():
    """Keep all 30x3 requested inputs; mark absent CPU seed references for an actual run."""
    from model import Kappa
    jobs=[j for j in json.loads(V.S2_MANIFEST.read_text())['jobs'] if j['test']=='e4']
    references={}
    for r in json.loads((V.V13/'r2_p4_e4/summary.json').read_text())['rows']:
        g=r['grasp'];references[(r['category'],g['w'],g['d'],g['F'],r['seed'],r['kappa']['pinch_coeff'])]=r['result']
    specs=[]
    for j in jobs:
        for seed in V.E4_SEEDS:
            g=j['grasp'];key=(j['category'],g['w'],g['d'],g['F'],seed,j['kappa']['pinch_coeff'])
            specs.append(dict(kappa=Kappa.from_dict(j['kappa']),grasp=dict(g),seed=seed,
                label=dict(set='E4',category=j['category'],original_index=j['original_index'],seed=seed),
                cpu=references.get(key)))
    return specs


def g4(out,workers,root):
    from multiprocessing import get_context
    specs=V.same_bucket_specs(64)
    # compare_tasks also reports hard-object force; all specs need an explicit category label.
    for s in specs:s['label']['category']=s['kappa'].category
    with get_context('spawn').Pool(min(28,os.cpu_count())) as pool:
        cpu=pool.map(V.cpu_evaluate,specs)
    for s,c in zip(specs,cpu):s['cpu']=c
    gpu=V.gpu_run(specs)
    summary=V.compare_tasks(specs,gpu)
    original=json.loads((V.BASE/'g2_formal/summary.json').read_text())['rows']
    repeats=[json.loads((root/f'r1y_{i}/summary.json').read_text())['row'] for i in range(2,6)]
    tasks=json.loads((root/'g3/summary.json').read_text())
    counts=dict(g2=dict(n=len(original)+len(repeats),failures=sum(bool(r['gpu_fail']) for r in original+repeats)),
        g3=dict(n=sum(len(tasks[n]['rows']) for n in ('E4','E5')),
                failures=sum(tasks[n]['gpu_sim_fail'] for n in ('E4','E5'))),
        hetero=dict(n=len(specs),failures=summary['gpu_sim_fail']))
    total=sum(c['n'] for c in counts.values());bad=sum(c['failures'] for c in counts.values())
    summary.update(counts=counts,cumulative_attempts=total,cumulative_gpu_failures=bad,
        cumulative_failure_rate=bad/total,hetero_failure_rate=summary['gpu_sim_fail']/len(specs),
        bucket=specs[0]['label']['bucket'],batched_fields=gpu[0]['batched_fields'],passed=bad/total<=.01)
    save(out/'summary.json',summary)
    save(out/'failure_distribution.json',failure_distribution(summary['rows']))
    print(json.dumps({k:summary[k] for k in ('passed','counts','cumulative_failure_rate','hetero_failure_rate','agreement')},indent=2),flush=True)
    return summary['passed']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['describe','g3','g4'])
    ap.add_argument('--directory',type=Path)
    ap.add_argument('--resume-root',type=Path,default=V.BASE/'resume_20261108')
    ap.add_argument('--workers',type=int,default=5)
    a=ap.parse_args()
    if a.action=='describe':
        import mjwarp_v13 as G
        print(json.dumps({name:dict(n=len(specs),missing_cpu=sum(s.get('cpu') is None for s in specs),
            buckets=[len(x) for x in G.buckets(specs).values()])
            for name,specs in dict(E4=e4_specs(),E5=V.e5_specs()).items()},indent=2))
        return 0
    if a.directory is None:ap.error('--directory required for formal runs')
    out=a.directory;out.mkdir(parents=True,exist_ok=False)
    V.metadata(out,a)
    before={str(p.relative_to(V.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in V.ROOT.glob('*.py')}
    original_compare=V.compare_tasks
    def checked_compare(specs,gpu):
        name=specs[0]['label']['set']
        save(out/f'{name}_raw_gpu.json',gpu)
        if any(r['overflow_checks'] for r in gpu):raise RuntimeError(f'{name}: GPU buffer overflow; raw results retained')
        result=original_compare(specs,gpu)
        for row,spec in zip(result['rows'],specs):row['cpu_failure_state']=spec['cpu'].get('failure_state')
        return result
    V.compare_tasks=checked_compare
    original_e4=V.e4_specs
    try:
        if a.action=='g3':
            from multiprocessing import get_context
            specs=e4_specs();missing=[s for s in specs if s['cpu'] is None]
            with get_context('spawn').Pool(min(28,os.cpu_count())) as pool:
                cpu=pool.map(V.cpu_evaluate,missing)
            for s,c in zip(missing,cpu):s['cpu']=c
            save(out/'supplemental_cpu_references.json',[dict(label=s['label'],kappa=s['kappa'].to_dict(),
                grasp=s['grasp'],seed=s['seed'],cpu=s['cpu']) for s in missing])
            V.e4_specs=lambda:specs
        ok=V.g3(out,a.workers) if a.action=='g3' else g4(out,a.workers,a.resume_root)
        if a.action=='g3':
            data=json.loads((out/'summary.json').read_text())
            save(out/'failure_distribution.json',failure_distribution(data['E4']['rows']+data['E5']['rows']))
    finally:V.compare_tasks=original_compare;V.e4_specs=original_e4
    used=[]
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(V.ROOT):
                n=str(p.relative_to(V.ROOT));used.append(n)
                if n not in before or hashlib.sha256(p.read_bytes()).hexdigest()!=before[n]:
                    raise RuntimeError(f'source changed during formal task check: {n}')
    save(out/'source_audit.json',dict(unchanged=True,imported_sources=sorted(set(used)),source_sha256=before))
    return 0 if ok else 2


if __name__=='__main__':raise SystemExit(main())
