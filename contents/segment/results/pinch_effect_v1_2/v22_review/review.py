"""Recheck reviewed E4 rules and record focused native-contact diagnostics."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEGMENT = BASE.parents[2]
sys.path.insert(0, str(SEGMENT))


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def held(r):
    q = r['result']
    t = next((p['t'] for p in q['phases'] if p['phase']=='carry'), None)
    return bool(t is not None and q['lift_table'] and not q['lift_table']['contact']
                and any(e['event']=='bind' and e['t']<=t for e in q['pinch_events'])
                and not any(e['event']!='bind' and e['t']<=t for e in q['pinch_events']))


def audit_e4():
    source = BASE.parent / 'v22/e4_formal'
    s = json.loads((source/'summary.json').read_text())
    rows = sorted(s['rows'],key=lambda r:r['index'])
    with (BASE.parent/'v22/analysis/e4_feasibility.csv').open() as f:
        original_csv = {int(r['index']):r for r in csv.DictReader(f)}
    evidence = []
    for r in rows:
        assert json.loads((source/f"case_{r['index']:04d}/result.json").read_text())==r
        assert original_csv[r['index']]['outcome']==r['result']['outcome']
        carry = next((p['t'] for p in r['result']['phases'] if p['phase']=='carry'), None)
        evidence.append(dict(index=r['index'],category=r['category'],scale=r['coefficient_scale'],
            seed=r['seed'],**r['grasp'],bound_through_lift_hold=held(r),carry_start=carry,
            outcome=r['result']['outcome'],events=json.dumps(r['result']['pinch_events']),
            evidence=f"../v22/e4_formal/case_{r['index']:04d}/result.json"))
    with (BASE/'e4_lift_hold_events.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
    categories = {}
    violations = []
    key=lambda r:(r['grasp']['w'],r['grasp']['d'],r['grasp']['F'],r['seed'])
    for c in ('bendable','compressible'):
        by_scale={scale:{key(r):r for r in rows if r['category']==c and r['coefficient_scale']==scale}
                  for scale in (.25,.5,1.)}
        holdsets={scale:{k for k,r in part.items() if held(r)} for scale,part in by_scale.items()}
        counts={scale:sum(r['result']['outcome']=='success' for r in part.values()) for scale,part in by_scale.items()}
        bad=[]
        for k in by_scale[.25]:
            if any(by_scale[lo][k]['result']['outcome']=='success' and by_scale[hi][k]['result']['outcome']!='success'
                   for lo,hi in ((.25,.5),(.5,1.))):
                bad.append(k)
                violations.append(dict(category=c,w=k[0],d=k[1],F=k[2],seed=k[3]))
        categories[c]=dict(success_counts=counts,success_count_nondecreasing=counts[.25]<=counts[.5]<=counts[1.],
            bound_through_hold_counts={s:len(v) for s,v in holdsets.items()},
            hold_sets_nested=holdsets[.25]<=holdsets[.5]<=holdsets[1.],
            complete_task_counterexample_actions=len(bad),paired_actions=len(by_scale[.25]))
    total_pairs=sum(c['paired_actions'] for c in categories.values())
    failed=[r for r in rows if r['category']=='bendable' and r['coefficient_scale']==1 and r['grasp']['w']==.06
            and r['result']['outcome']=='grasp_fail']
    data=dict(basis='v2.2 E4 review, 2026-11-06',categories=categories,
        complete_task_counterexamples=len(violations),paired_actions=total_pairs,
        counterexample_rate=len(violations)/total_pairs,hard_rejections=sum(r['category']=='hard' and r['result']['outcome']=='grasp_fail' for r in rows),
        simulation_failures=sum(r['confirmed_failure'] for r in rows),
        w60_default_grasp_failures=len(failed),w60_default_grasp_failures_all_bound=all(any(e['event']=='bind' for e in r['result']['pinch_events']) for r in failed),
        original_summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
        original_feasibility_rows_checked=len(original_csv),counterexample_actions=violations)
    data['passed']=all(c['success_count_nondecreasing'] and c['hold_sets_nested'] for c in categories.values()) and data['counterexample_rate']<=.05 and data['hard_rejections']==36 and data['simulation_failures']==0
    save(BASE/'e4_review.json',data)
    print(json.dumps(data,ensure_ascii=False,indent=2))
    return 0 if data['passed'] else 2


def diagnose(name,w,d):
    import numpy as np
    from model import ANCHORS
    from top_grasp import evaluate, HEIGHT
    out=BASE/name;out.mkdir()
    seen=Counter(); snapshots=[]
    def observe(sim,binding,phase,t,force):
        seen[phase]+=1
        if phase not in ('close','lift') or seen[phase]>12:
            return
        tips=np.array([sim.pad_pos(p)-[0,0,HEIGHT/2] for p in sim.pad_names])
        contacts=binding.top_contacts()
        gap=sim.pad_pos('pB')[1]-sim.pad_pos('pA')[1]-.004
        snapshots.append(dict(t=t,phase=phase,current_force_N=force,gap_m=float(gap),
            depth_mm=((binding.original_top-tips[:,2])*1000).tolist(),
            top_contacts=[sorted(v) for v in contacts],
            top_contact_cells=[[sim.desc.segments[i].idx for i in sorted(v)] for v in contacts],
            candidate=binding.candidate(tips),active=binding.active is not None,
            pad_forces={p:f.tolist() for p,f in sim.contact_forces()[0].items()}))
    q=evaluate(ANCHORS['bendable'],dict(w=w,d=d,F=5.),20261116,trace=True,observer=observe)
    save(out/'result.json',dict(grasp=dict(w=w,d=d,F=5.),seed=20261116,result=q,snapshots=snapshots))
    print(name,q['outcome'],q['reason'],q['wall_s'],q['pinch_events'])
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('audit-e4','diagnose'))
    ap.add_argument('--name');ap.add_argument('--w',type=float);ap.add_argument('--d',type=float)
    a=ap.parse_args()
    sys.exit(audit_e4() if a.action=='audit-e4' else diagnose(a.name,a.w,a.d))
