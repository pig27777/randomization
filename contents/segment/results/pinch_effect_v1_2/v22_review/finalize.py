"""Summarize recorded acceptance, audit frozen mechanics, and tag only if all gates pass."""
import csv
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parent
SEGMENT=BASE.parents[2]
PROJECT=SEGMENT.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    with path.open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def csv_save(path,rows,fields):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    e4=read(BASE/'e4_review.json')
    e5=read(BASE/'e5_formal/summary.json')
    e6=read(BASE/'e6_formal/summary.json') if (BASE/'e6_formal/summary.json').exists() else None
    candidate=read(BASE.parent/'v22/candidate_source.json')
    v1tag=read(SEGMENT/'results/freeze_v1_newton/release_v1/v1.json')
    units=read(BASE.parent/'v22/units_formal/summary.json')
    r4=read(BASE.parent/'v22/r4_formal/summary.json')
    out=BASE/'analysis';out.mkdir()
    random_rows=[];failures=[]
    for r in sorted(e5['rows'],key=lambda r:r['index']):
        evidence=f"e5_formal/case_{r['index']:04d}/result.json"
        assert read(BASE/evidence)==r
        q=r.get('result',{})
        entry=dict(index=r['index'],category=r['category'],seed=r['seed'],
            rest_fail=r.get('rest',{}).get('fail'),outcome=q.get('outcome'),reason=q.get('reason'),
            confirmed_failure=r.get('confirmed_failure'),execution_error=r['execution_error'],
            failure_phase=(q.get('failure_state') or {}).get('phase'),
            kappa=json.dumps(r['kappa']),grasp=json.dumps(r['grasp']),evidence=evidence)
        random_rows.append(entry)
        if r['execution_error'] or r.get('confirmed_failure'):
            failures.append(entry)
    csv_save(out/'e5_cases.csv',random_rows,list(random_rows[0]))
    csv_save(out/'e5_failures.csv',failures,list(random_rows[0]))
    categories={}
    for c in ('hard','bendable','compressible'):
        rows=[r for r in e5['rows'] if r['category']==c]
        categories[c]=dict(total=len(rows),rest_failures=sum(bool(r['rest']['fail']) for r in rows),
            simulation_failures=sum(r['confirmed_failure'] for r in rows),
            outcomes=dict(Counter(r['result']['outcome'] for r in rows)))
    save(out/'e5_analysis.json',dict(total=e5['total'],execution_errors=e5['execution_errors'],
        seed=20261126,confirmed_failures=e5['confirmed_failures'],categories=categories,passed=e5['passed']))
    if e6:
        speed=[]
        for r in sorted(e6['rows'],key=lambda r:r['index']):
            q=r['result']
            assert read(BASE/f"e6_formal/case_{r['index']:04d}/result.json")==r
            assert not r['execution_error'] and not r['confirmed_failure']
            speed.append(dict(index=r['index'],category=r['category'],seed=r['seed'],
                outcome=q['outcome'],wall_s=q['wall_s'],sim_s=q['sim_s'],
                completed_task=q['outcome']=='success',execution_error=r['execution_error'],
                evidence=f"e6_formal/case_{r['index']:04d}/result.json"))
        csv_save(out/'e6_times.csv',speed,list(speed[0]))
    stages={}
    for base,names in ((BASE.parent/'v22',('units_formal','r4_formal','e4_formal')),
                       (BASE,('e5_formal', 'e6_formal') if e6 else ('e5_formal',))):
        for name in names:
            m=read(base/name/'metadata.json')
            stages[f'{base.name}/{name}']=dict(
                source_matches=all(sha(SEGMENT/p)==h for p,h in m['source_sha256'].items()),
                protocol_matches=sha(PROJECT/'test/policy/README.md')==m['task_protocol_sha256'])
    archive=PROJECT/candidate['archive']
    with tarfile.open(archive) as tar:
        archive_matches=all(hashlib.sha256(tar.extractfile(p).read()).hexdigest()==h
                            for p,h in candidate['source_sha256'].items())
    audit=dict(stages=stages,candidate_archive_sha256=sha(archive),
        candidate_archive_matches=sha(archive)==candidate['archive_sha256'] and archive_matches,
        candidate_current_sources_match=all(sha(PROJECT/p)==h for p,h in candidate['source_sha256'].items()),
        v1_core_unchanged={p:sha(SEGMENT/p)==v1tag['source_sha256'][p]
                           for p in ('mj_adapter.py','protocols.py','bushing.py','metrics.py')},
        v1_archive_sha256=sha(SEGMENT/'results/freeze_v1_newton/release_v1/v1.tar.gz'),
        table_sha256=sha(SEGMENT/'results/freeze_v1_newton/mapping_pass1/table.json'),
        e5_raw_records=len(list((BASE/'e5_formal').glob('case_*/result.json'))),
        e6_raw_records=len(list((BASE/'e6_formal').glob('case_*/result.json'))) if e6 else 0)
    assert all(all(v.values()) for v in stages.values())
    assert audit['candidate_archive_matches'] and audit['candidate_current_sources_match']
    assert all(audit['v1_core_unchanged'].values())
    assert audit['v1_archive_sha256']==v1tag['archive_sha256']
    assert audit['table_sha256']==v1tag['table']['sha256']
    assert audit['e5_raw_records']==200
    if e6: assert audit['e6_raw_records']==30
    save(out/'audit.json',audit)
    following=read(BASE.parent/'v22/e4_formal/summary.json')['static_following']
    checks=dict(E1=units['passed'],E2=True,E3=r4['passed'],E4=e4['passed'],E5=e5['passed'],
                E6=bool(e6 and e6['passed']),static_following=all(v['passed'] for v in following.values()))
    passed=all(checks.values())
    acceptance=dict(protocol='v2.2',status='frozen' if passed else 'not_frozen',checks=checks,
        E1=dict(tests=units['tests'],passed=units['passed'],evidence='../v22/units_formal/summary.json'),
        E2=dict(passed=True,basis='Reviewed 12/12 bitwise regression retained; unchanged ordinary physical core'),
        E3=dict(passed=r4['passed'],repeats_per_anchor=3,evidence='../v22/r4_formal/summary.json'),
        E4=dict(passed=e4['passed'],basis='Authorized revised E4 criteria',evidence='e4_review.json',
                simulation_failures=0,total=684),
        E5=dict(passed=e5['passed'],seed=20261126,total=e5['total'],confirmed_failures=e5['confirmed_failures'],
                execution_errors=e5['execution_errors'],evidence='e5_formal/summary.json'),
        E6=dict(passed=e6['passed'],total=e6['total'],median_wall_s=e6['median_wall_s'],
                ratios_to_side=e6['ratios_to_side'],evidence='e6_formal/summary.json') if e6 else dict(status='not_run'),
        static_following=following,source='../v22/candidate_source.json',audit='analysis/audit.json')
    save(BASE/'acceptance.json',acceptance)
    print(json.dumps(acceptance,ensure_ascii=False,indent=2))
    if not passed:
        return
    release=BASE/'release_v1_2';release.mkdir()
    target=release/'v1.2.tar.gz'
    shutil.copyfile(archive,target)
    assert sha(target)==candidate['archive_sha256']
    tag=dict(tag='v1.2',frozen=True,frozen_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        format='local immutable source archive and SHA256 tag; project has no Git metadata',
        protocol='v2.2',base='v1 Newton, cuboid bulk content, no skin or shape change',
        archive=str(target.relative_to(SEGMENT)),archive_sha256=sha(target),
        source_sha256=candidate['source_sha256'],table=v1tag['table'],
        acceptance=str((BASE/'acceptance.json').relative_to(SEGMENT)),
        acceptance_sha256=sha(BASE/'acceptance.json'),checks=checks,
        evidence={'review.py':sha(BASE/'review.py'),'e4_review.json':sha(BASE/'e4_review.json'),
                  'e5_formal/summary.json':sha(BASE/'e5_formal/summary.json'),
                  'e6_formal/summary.json':sha(BASE/'e6_formal/summary.json')},
        sampling=dict(R3='sampling.sample_kappa(rng, category)',RJ='sampling.sample_rj(rng, category)',
                      pinch_coeff='R4 pull-off load / current total grip force, hard=0'),
        limits=['v1.1 shape paused; no surface layer, no MJWarp or Isaac Lab speed claim',
                'E2 retained previously reviewed 12/12 bitwise tests; no rerun this round',
                'w60 bendable native-contact/force-control path remains sensitive to segment spacing',
                'E4 revised criteria allow <=5% complete-task paired counterexamples',
                'E5 is a 200-object sample, not a guarantee for all kappa/actions; random task success 8/200',
                'E6 hard timing is early grasp_fail; soft execution and final task outcomes are reported separately',
                '40 N/mm spring and 10 mm gathering are fixed effect assumptions without real calibration',
                'Historical thin bendable case116 still failed an exploratory replay; not removed'])
    save(release/'v1.2.json',tag)
    print('frozen',target,tag['archive_sha256'])


if __name__=='__main__':
    main()
