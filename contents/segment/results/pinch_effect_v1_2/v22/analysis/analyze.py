"""Summarize recorded v2.2 acceptance data; performs no simulation."""
import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
PROJECT = BASE.parents[4]
SEGMENT = PROJECT / 'contents/segment'


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def csv_save(name, rows):
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary = read(BASE / 'e4_formal/summary.json')
rows = sorted(summary['rows'], key=lambda r: r['index'])
assert len(rows) == 684 and summary['execution_errors'] == 0
feasibility, holds, region = [], [], defaultdict(list)
for r in rows:
    q, g = r['result'], r['grasp']
    evidence = f"e4_formal/case_{r['index']:04d}/result.json"
    assert read(BASE / evidence) == r
    overloads = [e for e in q['pinch_events'] if e['event'] == 'overload']
    feasibility.append(dict(index=r['index'], category=r['category'], scale=r['coefficient_scale'],
        pinch_coeff=r['kappa']['pinch_coeff'], w_mm=g['w']*1000, d_mm=g['d']*1000,
        F_N=g['F'], seed=r['seed'], outcome=q['outcome'], reason=q['reason'],
        actual_depth_mm=q['actual_depth_m']*1000 if q['actual_depth_m'] is not None else None,
        max_grip_force_N=q['max_grip_force_N'], max_real_grip_force_N=q['max_real_grip_force_N'],
        max_spring_force_N=q['max_spring_force_N'], overload=bool(overloads),
        overload_t=overloads[0]['t'] if overloads else None,
        static_hold_sample_count=q['static_hold_sample_count'],
        static_attachment_error_max_mm=q['static_attachment_error_max_mm'],
        suspension_extension_mean_mm=q['suspension_extension_mean_mm'],
        suspension_extension_max_mm=q['suspension_extension_max_mm'],
        suspension_over_half_thickness=q['suspension_over_half_thickness'], evidence=evidence))
    for h in q['hold_samples']:
        holds.append(dict(index=r['index'], category=r['category'], scale=r['coefficient_scale'],
                          outcome=q['outcome'], **h))
    if r['category'] != 'hard':
        region[(r['category'],r['coefficient_scale'],g['w'],g['d'],g['F'])].append(r)
csv_save('e4_feasibility.csv', feasibility)
csv_save('hold_diagnostics.csv', holds)
region_rows = []
for (c, scale, w, d, f), part in sorted(region.items()):
    good = [r['seed'] for r in part if r['result']['outcome']=='success']
    region_rows.append(dict(category=c, scale=scale, pinch_coeff=part[0]['kappa']['pinch_coeff'],
        w_mm=w*1000, d_mm=d*1000, F_N=f, successes=len(good), trials=len(part),
        success_seeds=';'.join(map(str,sorted(good)))))
csv_save('e4_success_region.csv', region_rows)

def action_key(r):
    return (r['grasp']['w'],r['grasp']['d'],r['grasp']['F'],r['seed'])

counterexamples = []
for c in ('bendable','compressible'):
    by_scale = {s:{action_key(r):r for r in rows if r['category']==c and r['coefficient_scale']==s}
                for s in (.25,.5,1.)}
    for lo,hi in ((.25,.5),(.5,1.),(.25,1.)):
        for key,a in sorted(by_scale[lo].items()):
            b = by_scale[hi][key]
            if a['result']['outcome']!='success' or b['result']['outcome']=='success':
                continue
            counterexamples.append(dict(category=c,w_mm=key[0]*1000,d_mm=key[1]*1000,F_N=key[2],
                seed=key[3],low_scale=lo,high_scale=hi,adjacent=(lo,hi)!=(.25,1.),
                low_index=a['index'],high_index=b['index'],low_outcome='success',
                high_outcome=b['result']['outcome'],high_reason=b['result']['reason'],
                low_events=json.dumps(a['result']['pinch_events']),
                high_events=json.dumps(b['result']['pinch_events']),
                low_evidence=f"e4_formal/case_{a['index']:04d}/result.json",
                high_evidence=f"e4_formal/case_{b['index']:04d}/result.json"))
csv_save('monotonicity_counterexamples.csv',counterexamples)
groups = {}
for c in ('hard','bendable','compressible'):
    groups[c] = {}
    for scale in sorted({r['coefficient_scale'] for r in rows if r['category']==c}):
        part = [r for r in rows if r['category']==c and r['coefficient_scale']==scale]
        q = [r['result'] for r in part]
        successful = [r for r in q if r['outcome']=='success']
        held = [r for r in q if r['suspension_extension_mean_mm'] is not None]
        max_or_none = lambda values: max(values, default=None)
        groups[c][str(scale)] = dict(total=len(part), outcomes=dict(Counter(r['outcome'] for r in q)),
            capacity_releases=sum(any(e['event']=='overload' for e in r['pinch_events']) for r in q),
            successful_capacity_releases=sum(any(e['event']=='overload' for e in r['pinch_events']) for r in successful),
            hold_cases=len(held), over_half_thickness_cases=sum(r['suspension_over_half_thickness'] for r in held),
            success_over_half_thickness_cases=sum(r['suspension_over_half_thickness'] for r in successful),
            max_suspension_extension_mm=max_or_none(r['suspension_extension_max_mm'] for r in held),
            max_static_attachment_error_mm=max_or_none(r['static_attachment_error_max_mm'] for r in q
                                                     if r['static_attachment_error_max_mm'] is not None))
analysis = dict(total=len(rows), execution_errors=summary['execution_errors'],
    simulation_failures=sum(r['confirmed_failure'] for r in rows), groups=groups,
    static_following=summary['static_following'],
    adjacent_counterexample_count=sum(r['adjacent'] for r in counterexamples),
    all_pair_counterexample_count=len(counterexamples), passed=summary['passed'])
save('e4_analysis.json',analysis)

candidate = read(BASE / 'candidate_source.json')
stages = {}
for stage in ('units_formal','r4_formal','e4_formal'):
    m = read(BASE / stage / 'metadata.json')
    stages[stage] = dict(source_matches_current=all(sha(SEGMENT/p)==h for p,h in m['source_sha256'].items()),
        protocol_matches_current=sha(PROJECT/'test/policy/README.md')==m['task_protocol_sha256'])
with tarfile.open(BASE/'candidate_sources.tar.gz') as tar:
    archive_sources = {p:hashlib.sha256(tar.extractfile(p).read()).hexdigest()==h
                       for p,h in candidate['source_sha256'].items()}
v1tag=read(SEGMENT/'results/freeze_v1_newton/release_v1/v1.json')
v21=read(BASE.parent/'v21/candidate_source.json')
audit = dict(stages=stages,
    candidate_archive_matches_fingerprint=sha(BASE/'candidate_sources.tar.gz')==candidate['archive_sha256'],
    candidate_archive_file_count=len(archive_sources),candidate_archive_all_sources_match=all(archive_sources.values()),
    candidate_current_all_sources_match=all(sha(PROJECT/p)==h for p,h in candidate['source_sha256'].items()),
    changed_python_since_v21=[p for p,h in candidate['source_sha256'].items() if p.endswith('.py') and h!=v21['source_sha256'].get(p)],
    v1_core_unchanged={p:sha(SEGMENT/p)==v1tag['source_sha256'][p] for p in ('mj_adapter.py','protocols.py','bushing.py','metrics.py')},
    v1_archive_sha256=sha(SEGMENT/'results/freeze_v1_newton/release_v1/v1.tar.gz'),
    table_sha256=sha(SEGMENT/'results/freeze_v1_newton/mapping_pass1/table.json'),
    e4_raw_records=len(list((BASE/'e4_formal').glob('case_*/result.json'))),
    e4_raw_summary_identical=True, e5_run=False,e6_run=False)
assert all(all(v.values()) for v in stages.values())
assert audit['candidate_archive_matches_fingerprint'] and audit['candidate_archive_all_sources_match']
assert audit['candidate_current_all_sources_match'] and all(audit['v1_core_unchanged'].values())
assert audit['v1_archive_sha256']==v1tag['archive_sha256']
assert audit['table_sha256']=='e65abc104e0e8e3dc9e3f11437d41fe22318a32188fd17232e4d6819407009d9'
assert not (BASE/'e5_formal').exists() and not (BASE/'e6_formal').exists()
save('audit.json',audit)
r4=read(BASE/'r4_formal/summary.json')
acceptance=dict(protocol='v2.2',status='not_frozen',tag=None,
    E1=read(BASE/'units_formal/summary.json'),
    E2=dict(passed=True,basis='Reviewed 12/12 bitwise v1 regression retained; ordinary physical core unchanged'),
    E3=dict(passed=r4['passed'],repeats_per_anchor=3,
            measured={r['category']:r['result']['measured_pinch_coeff'] for r in r4['rows']}),
    static_following=summary['static_following'],
    E4=dict(passed=False,total=684,simulation_failures=0,reason='Success sets not nested',
            adjacent_counterexamples=6,all_pair_counterexamples=8,
            default_successes={'bendable':56,'compressible':60},hard_rejections=36),
    E5=dict(status='not_run',reason='E4 failed',planned_new_seed=20261126),
    E6=dict(status='not_run',reason='E4 failed'),source='candidate_source.json',evidence='analysis/audit.json')
with (BASE/'acceptance.json').open('x') as f:
    json.dump(acceptance,f,indent=2)
print(json.dumps(analysis,ensure_ascii=False,indent=2))
print('audit passed; candidate not frozen')
