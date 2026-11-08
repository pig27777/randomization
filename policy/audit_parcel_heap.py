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

from parcel_heap import RUN,CASES,LAYOUTS,CENTER,SIDE_ROTATION,install_backend,scene
from probe_feedback import digest,save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=RUN)
    root = parser.parse_args().run
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
