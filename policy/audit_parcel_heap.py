"""Summarize completed trials without converting incomplete/invalid runs to failures."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import platform
import sys

import numpy as np
import mujoco

from parcel_heap import RUN,CASES,LAYOUTS,CENTER,SIDE_ROTATION,install_backend,scene,contact_safety,outside_count
from probe_feedback import digest,save


def audit_continuation(root):
    rows,checks,completion=[],[],[]
    for path in sorted(root.glob('**/*.jsonl')):
        for number,line in enumerate(path.read_text().splitlines(),1):
            row=json.loads(line)
            if row.get('exploratory'):
                continue
            rows.append(dict(row,archive=str(path.relative_to(root)),row=number))
    for path in sorted(root.glob('**/*_metadata.json')):
        meta=json.loads(path.read_text())
        if 'files' not in meta:
            continue
        stage=path.name.removesuffix('_metadata.json')
        result=path.parent/(stage+'.jsonl')
        count=len(result.read_text().splitlines()) if result.exists() else 0
        terminated=(path.parent/(stage+'_termination.json')).exists()
        completion.append(dict(archive=str(result.relative_to(root)),expected=meta['jobs'],completed=count,
                               terminated=terminated,accounted_for=count==meta['jobs'] or terminated))
        for name,expected in meta['files'].items():
            archived=path.parent/'driver_source'/stage/Path(name).name
            checks.append(dict(file=str(archived.relative_to(root)),ok=digest(archived)==expected))
        for name,key in [('settings.json','settings_sha256'),('source_helpers/flex_curvature.py','helper_sha256')]:
            if meta.get(key):
                archived=path.parent/name
                checks.append(dict(file=str(archived.relative_to(root)),ok=digest(archived)==meta[key]))
    for path in sorted(root.glob('**/source_manifest.json')):
        for name,expected in json.loads(path.read_text())['files'].items():
            archived=path.parent/'source'/name
            checks.append(dict(file=str(archived.relative_to(root)),ok=digest(archived)==expected))
    for snapshot in sorted(root.glob('**/diagnostic_source/*')):
        result=snapshot.parent.parent/(snapshot.name+'.json')
        if not result.exists():
            continue
        meta=json.loads(result.read_text())
        for name,key in [('diagnose_parcel_heap.py','source_sha256'),('parcel_heap.py','driver_sha256')]:
            checks.append(dict(file=str((snapshot/name).relative_to(root)),ok=digest(snapshot/name)==meta[key]))
    fields=['archive','case','layout','action','jitter','outcome','reason','timestep_s','sim_time_s','wall_s',
            'lift_mm','carry_mm','penetration_mm','solver_iterations_peak','max_neighbor_displacement_mm',
            'content_centers_outside','direct_content_support_contacts_peak','support_N']
    with (root/'trials.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            flat={k:row.get(k,'') for k in fields}
            if 'displacement_mm' in row:
                flat.update(lift_mm=row['displacement_mm'][2],
                            carry_mm=float(np.sign(row['settings']['carry_x'])*row['displacement_mm'][0]))
            flat.update(penetration_mm=row.get('peaks',{}).get('penetration_mm',''),
                        support_N=row.get('last_contact',{}).get('support_N',''))
            writer.writerow(flat)
    def select(folder,case,preflight=False):
        return next((r for r in rows if r['archive'].split('/')[0]==folder and r['case']==case
                     and r.get('preflight',False)==preflight and
                     (preflight or (r.get('action')=='center' and r.get('jitter')==0))),None)
    step_comparison={}
    a,b=select('overhead','book'),select('overhead_quarter','book')
    if a and b:
        # Compare absolute endpoint COM, not displacement from different settled states.
        ends=[]
        for row in (a,b):
            install_backend((root/row['archive']).parent)
            robot=scene(row['case'],row['layout'],row['action'],row['jitter'])
            robot.sim.d.qpos[:]=row['keyframes'][-1]['qpos']
            mujoco.mj_forward(robot.sim.m,robot.sim.d)
            ends.append(robot.sim.package_com())
        delta=float(np.linalg.norm(ends[0]-ends[1])*1000)
        valid={'success','grasp_fail','neighbor_disturbance'}
        completed=all(r['phase_times'][-1]['phase']=='hold' for r in (a,b))
        step_comparison=dict(cases=[a['archive'],b['archive']],final_keyframe_com_delta_mm=delta,
                             outcomes=[a['outcome'],b['outcome']],
                             both_completed=completed,
                             passed=completed and a['outcome']==b['outcome'] and a['outcome'] in valid and delta<=5.)
    matched={}
    a,b=select('overhead','book',True),select('overhead','clothes',True)
    if a and b:
        delta=np.linalg.norm(np.asarray(a['shell_after'])-b['shell_after'],axis=1)*1000
        matched=dict(initial_shell_equal=a['shell_sha256']==b['shell_sha256'],
                     settled_shell_vertex_rms_mm=float(np.sqrt(np.mean(delta**2))),
                     settled_shell_vertex_max_mm=float(delta.max()),
                     mass_difference_kg=abs(a['package_mass_kg']-b['package_mass_kg']))
    final_checks=[]
    for row in rows:
        if row.get('preflight') or row.get('outcome')!='success':
            continue
        install_backend((root/row['archive']).parent)
        robot=scene(row['case'],row['layout'],row['action'],row['jitter'])
        sim=robot.sim
        sim.d.qpos[:]=row['keyframes'][-1]['qpos']
        mujoco.mj_forward(sim.m,sim.d)
        safety=contact_safety(robot)
        escaped=outside_count(sim.d.xipos[sim.content_bodies],sim.shell_points(),sim.info['faces'])
        passed=(safety['penetration_mm']<=(.15 if sim.info['thin'] else 1.) and escaped==0 and
                safety['nonfinger_mm']==0 and safety['self_contact_mm']==0 and
                safety['direct_content_support_contacts']==0)
        final_checks.append(dict(archive=row['archive'],safety=safety,escaped=escaped,passed=passed,
                                 note='Final qpos geometry replay; zero velocity, not dynamic contact force validation.'))
    summary=dict(total_formal_rows=len(rows),
                 preflight_outcomes=dict(Counter(r['outcome'] for r in rows if r.get('preflight'))),
                 grasp_outcomes=dict(Counter(r['outcome'] for r in rows if not r.get('preflight'))),
                 matched_exteriors=matched,step_comparison=step_comparison,source_checks=checks,
                 successful_final_pose_checks=final_checks,
                 stage_completion=completion,
                 cpu_authorization=json.loads((root/'cpu_authorization.json').read_text()),
                 note='Development trajectory JSON files are excluded. No population success-rate inference.')
    save(root/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_checks','cpu_authorization')}))
    if not checks or not all(c['ok'] for c in checks):
        raise SystemExit('archived source hash mismatch')
    if not all(c['accounted_for'] for c in completion):
        raise SystemExit('some planned stages have unfinished rows')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--continuation',action='store_true')
    args=parser.parse_args()
    root = args.run
    if args.continuation:
        audit_continuation(root)
        return
    all_rows = []
    source_checks = []
    for run in (root,root/'contact_fix'):
        for path in sorted(run.glob('*.jsonl')):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for i,row in enumerate(rows):
                row = dict(row,archive=str(path.relative_to(root)),row=i+1)
                all_rows.append(row)
        for path in run.glob('*_metadata.json'):
            stage = path.name.removesuffix('_metadata.json')
            meta = json.loads(path.read_text())
            for name,expected in meta['files'].items():
                actual = digest(run/'driver_source'/stage/Path(name).name)
                source_checks.append(dict(archive=str(path.relative_to(root)),file=name,ok=actual==expected))
    fields = ['archive','row','case','layout','action','jitter','outcome','reason','sim_time_s','wall_s',
              'initial_overlap_mm','film_thickness_mm','penetration_mm','support_N',
              'content_centers_outside','direct_content_support_contacts_peak',
              'max_neighbor_displacement_mm','lift_mm','carry_mm']
    with (root/'trials.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in all_rows:
            flat = {key:row.get(key,'') for key in fields}
            flat.update(penetration_mm=row['peaks'].get('penetration_mm',''),
                        support_N=row['last_contact']['support_N'])
            if 'displacement_mm' in row:
                flat.update(lift_mm=row['displacement_mm'][2],carry_mm=-row['displacement_mm'][0])
            writer.writerow(flat)
    formal = [r for r in all_rows if r['archive']=='contact_fix/preflight.jsonl']
    comparisons = []
    for layout in LAYOUTS:
        baseline = next((r for r in formal if r['case']=='book' and r['layout']==layout),None)
        other = next((r for r in formal if r['case']=='clothes' and r['layout']==layout),None)
        if baseline and other:
            delta = np.linalg.norm(np.asarray(baseline['shell_after'])-other['shell_after'],axis=1)*1000
            comparisons.append(dict(layout=layout,initial_shell_equal=baseline['shell_sha256']==other['shell_sha256'],
                                    settled_vertex_rms_mm=float(np.sqrt(np.mean(delta**2))),
                                    settled_vertex_max_mm=float(delta.max())))
    summary = dict(total_rows=len(all_rows),formal_preflight_rows=len(formal),
                   matrix_complete={(c,l) for c in CASES for l in LAYOUTS}=={(r['case'],r['layout']) for r in formal},
                   formal_preflight_outcomes=dict(Counter(r['outcome'] for r in formal)),
                   grasp_outcomes=dict(Counter(r['outcome'] for r in all_rows if not r.get('preflight',True))),
                   source_checks=source_checks,matched_shell_comparisons=comparisons,
                   audit_environment=dict(python=sys.version,mujoco=mujoco.__version__,numpy=np.__version__,host=platform.node()))
    save(root/'summary.json',summary)
    install_backend(root/'contact_fix')
    ik_rows = []
    for case in ('book','empty_envelope'):
        robot = scene(case,'top')
        robot.rotation = SIDE_ROTATION
        for z in ((.0809 if 'envelope' in case else .1035),.25):
            target = np.array([CENTER[0],CENTER[1]-.11*.48-.07,z])
            error = robot.solve(target,multistart=True)
            ik_rows.append(dict(case=case,target=target.tolist(),rotation=SIDE_ROTATION.tolist(),error=error,
                                pass_1mm_1deg=error['position_mm']<=1 and error['rotation_deg']<=1))
    save(root/'side_ik_candidates.json',dict(note='Read-only corrected +y approach; low target and raised 250 mm reachability control. Not a grasp experiment.',rows=ik_rows))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_checks','audit_environment')},ensure_ascii=False))
    if not all(row['ok'] for row in source_checks):
        raise SystemExit('archived source hash mismatch')


if __name__=='__main__':
    main()
