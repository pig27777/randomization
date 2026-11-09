"""G3/G4 continuation with retained raw results, overflow checks and cumulative stability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import socket
from datetime import datetime, timezone

import numpy as np


CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))

import validate_warp_v13 as V
from source_manifest import source_paths,source_name


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


def a0_smoke(out):
    specs=[V.e5_specs()[i] for i in (7,9)]
    single=[]
    for s in specs:
        single+=V.gpu_run([s])
        print('A0 singleton',s['label'],single[-1]['outcome'],flush=True)
    paired=V.gpu_run(specs)
    save(out/'single_gpu.json',single);save(out/'paired_gpu.json',paired)
    agreement=[a['outcome']==b['outcome'] for a,b in zip(single,paired)]
    passed=all(agreement) and not any(r['outcome']=='sim_fail' or r['overflow_checks'] for r in single+paired)
    result=dict(passed=passed,paired_agreement=agreement,
        labels=[s['label'] for s in specs],cpu_outcomes=[s['cpu']['outcome'] for s in specs],
        single_outcomes=[r['outcome'] for r in single],paired_outcomes=[r['outcome'] for r in paired],
        shared_model_fields=paired[0]['shared_model_fields'])
    save(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return passed


def e4_audit(out,specs,source):
    """Review retained formal E4 data without running any CPU/GPU physics."""
    from frozen_release import verify
    gpu=json.loads(source.read_text())
    if len(gpu)!=len(specs):raise ValueError('retained E4 result count does not match')
    result=V.compare_tasks(specs,gpu)
    force=result['hard_force']
    passed=result['agreement']>=.95 and force['relative']<=.10 and force['hard_agreement']>=.95
    passed=passed and result['gpu_sim_fail']/len(specs)<=.01
    by_category={}
    for category in ('hard','bendable','compressible'):
        rows=[r for r in result['rows'] if r['label']['category']==category]
        by_category[category]=dict(total=len(rows),agree=sum(r['agree'] for r in rows))
    original_meta=source.parent/'metadata.json'
    previous=json.loads(original_meta.read_text())['source_sha256']
    checked={name:hashlib.sha256((V.ROOT/name).read_bytes()).hexdigest()==previous[name]
        for name in ('top_grasp.py','pinch.py','model.py','bushing.py','warp_v13.py','warp_effects.py','mjwarp_backend.py')}
    tag=json.loads((V.V13/'release_v1_3/v1.3.json').read_text())
    save(out/'frozen_integrity.json',verify(tag,V.ROOT))
    result.update(passed=bool(passed),by_category=by_category,
        source=dict(path=str(source.resolve()),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            metadata_sha256=hashlib.sha256(original_meta.read_bytes()).hexdigest(),
            key_sources_equal_to_original_start=checked,
            limitation='Original G3 ended in E5 construction error, without final source audit or hostname.'),
        interpretation='Offline audit of retained formal E4; not a new GPU repeat. E5/G4/G5 not run.')
    save(out/'summary.json',result)
    save(out/'counterexamples.json',[r for r in result['rows'] if not r['agree']])
    print(json.dumps({k:result[k] for k in ('passed','agreement','by_category','hard_force','gpu_sim_fail')},indent=2),flush=True)
    return passed


def g3(out,workers,specs,e4_reuse):
    from multiprocessing import get_context
    import mjwarp_v13 as G
    sets=dict(E4=specs,E5=V.e5_specs());summary={}
    with get_context('spawn').Pool(workers) as pool:
        for name,inputs in sets.items():
            groups=list(G.buckets(inputs).values())
            if name=='E4' and e4_reuse is not None:
                gpu=json.loads(e4_reuse.read_text())
                if len(gpu)!=len(inputs):raise ValueError('reused E4 result count does not match')
                save(out/'E4_reused_source.json',dict(path=str(e4_reuse),
                    sha256=hashlib.sha256(e4_reuse.read_bytes()).hexdigest(),reason='A0 changes only heterogeneous shared-field handling'))
            else:
                gpu=[None]*len(inputs)
                jobs=[[inputs[i] for i in indices] for indices in groups]
                for j,(indices,part) in enumerate(zip(groups,pool.imap(V.gpu_run,jobs))):
                    save(out/f'{name}_bucket_{j:02d}.json',dict(indices=indices,results=part))
                    for i,r in zip(indices,part):gpu[i]=r
                    print(name,'bucket',j+1,'/',len(groups),'n',len(part),flush=True)
            summary[name]=V.compare_tasks(inputs,gpu)
            summary[name]['buckets']=[len(x) for x in groups]
            print(name,'agreement',summary[name]['agreement'],'hard force',summary[name]['hard_force'],flush=True)
    ok=True
    for name in sets:
        r=summary[name];ok&=r['agreement']>=.95
        if r['hard_force']:ok&=r['hard_force']['relative']<=.10 and r['hard_force']['hard_agreement']>=.95
    failed=sum(summary[n]['gpu_sim_fail'] for n in sets);count=sum(len(x) for x in sets.values())
    summary.update(task_layer_passed=bool(ok),stability_passed=failed/count<=.01,passed=bool(ok and failed/count<=.01))
    save(out/'summary.json',summary)
    return summary['passed']


def g4(out,workers,root,g3_summary):
    from multiprocessing import get_context
    tasks=json.loads(g3_summary.read_text())
    if not tasks['passed']:raise ValueError('G4 requires a passed G3 summary')
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
    ap.add_argument('action',choices=['describe','a0-smoke','e4-audit','g3','g4'])
    ap.add_argument('--directory',type=Path)
    ap.add_argument('--resume-root',type=Path,default=V.BASE/'resume_20261108')
    ap.add_argument('--workers',type=int,default=5)
    ap.add_argument('--e4-reuse',type=Path)
    ap.add_argument('--cpu-reference-reuse',type=Path)
    ap.add_argument('--g3-summary',type=Path)
    a=ap.parse_args()
    if a.action=='describe':
        import mjwarp_v13 as G
        print(json.dumps({name:dict(n=len(specs),missing_cpu=sum(s.get('cpu') is None for s in specs),
            buckets=[len(x) for x in G.buckets(specs).values()])
            for name,specs in dict(E4=e4_specs(),E5=V.e5_specs()).items()},indent=2))
        return 0
    if a.directory is None:ap.error('--directory required for formal runs')
    if a.action=='e4-audit' and a.e4_reuse is None:ap.error('e4-audit requires --e4-reuse')
    out=a.directory;out.mkdir(parents=True,exist_ok=False)
    V.metadata(out,a)
    save(out/'run_context.json',dict(hostname=socket.gethostname(),started_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,g3_summary=str(a.g3_summary) if a.g3_summary else None))
    before={source_name(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()}
    original_compare=V.compare_tasks
    def checked_compare(specs,gpu):
        name=specs[0]['label']['set']
        save(out/f'{name}_raw_gpu.json',gpu)
        if any(r['overflow_checks'] for r in gpu):raise RuntimeError(f'{name}: GPU buffer overflow; raw results retained')
        result=original_compare(specs,gpu)
        for row,spec in zip(result['rows'],specs):
            row['cpu_failure_state']=spec['cpu'].get('failure_state')
            row['cpu_events']=spec['cpu']['pinch_events']
        return result
    V.compare_tasks=checked_compare
    try:
        if a.action in ('g3','e4-audit'):
            from multiprocessing import get_context
            specs=e4_specs()
            if a.cpu_reference_reuse:
                cached=json.loads(a.cpu_reference_reuse.read_text())
                save(out/'cpu_reference_source.json',dict(path=str(a.cpu_reference_reuse.resolve()),
                    sha256=hashlib.sha256(a.cpu_reference_reuse.read_bytes()).hexdigest(),count=len(cached)))
                key=lambda k,g,seed:json.dumps([k,g,seed],sort_keys=True)
                refs={key(r['kappa'],r['grasp'],r['seed']):r['cpu'] for r in cached}
                for s in specs:
                    if s['cpu'] is None:s['cpu']=refs.get(key(s['kappa'].to_dict(),s['grasp'],s['seed']))
            missing=[s for s in specs if s['cpu'] is None]
            if missing:
                if a.action=='e4-audit':raise ValueError('offline audit needs all CPU references supplied')
                with get_context('spawn').Pool(min(28,os.cpu_count())) as pool:
                    cpu=pool.map(V.cpu_evaluate,missing)
                for s,c in zip(missing,cpu):s['cpu']=c
            save(out/'supplemental_cpu_references.json',[dict(label=s['label'],kappa=s['kappa'].to_dict(),
                grasp=s['grasp'],seed=s['seed'],cpu=s['cpu']) for s in missing])
        if a.action=='g3':ok=g3(out,a.workers,specs,a.e4_reuse)
        elif a.action=='g4':ok=g4(out,a.workers,a.resume_root,a.g3_summary or a.resume_root/'g3/summary.json')
        elif a.action=='e4-audit':ok=e4_audit(out,specs,a.e4_reuse)
        else:ok=a0_smoke(out)
        if a.action=='g3':
            data=json.loads((out/'summary.json').read_text())
            save(out/'failure_distribution.json',failure_distribution(data['E4']['rows']+data['E5']['rows']))
    finally:V.compare_tasks=original_compare
    used=[]
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(V.ROOT) or p.is_relative_to(Path(__file__).parent):
                n=source_name(p);used.append(n)
                if n not in before or hashlib.sha256(p.read_bytes()).hexdigest()!=before[n]:
                    raise RuntimeError(f'source changed during formal task check: {n}')
    save(out/'source_audit.json',dict(unchanged=True,imported_sources=sorted(set(used)),source_sha256=before))
    return 0 if ok else 2


if __name__=='__main__':raise SystemExit(main())
