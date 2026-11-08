"""Offline audit and CSV export of the completed YAM experiment."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from probe_feedback import ARMS,DEFAULT_RUN,FORCES,ROOT,digest,load_backend,read_rows,save


def audit(run):
    load_backend(run)
    from yam_probe_robot import asset_fingerprint
    checks={}
    checks['asset_hashes_unchanged']=asset_fingerprint()==json.loads((run/'yam_asset_sha256.json').read_text())
    checks['policy_hash_unchanged']=digest(run/'policy.json')==(run/'policy.sha256').read_text().strip()
    all_rows={}
    for stage in ('dev','test'):
        rows=read_rows(run/f'{stage}.jsonl')
        all_rows[stage]=rows
        meta=json.loads((run/f'{stage}_metadata.json').read_text())
        checks[f'{stage}_result_hash']=meta['output_sha256']==digest(run/f'{stage}.jsonl')
        checks[f'{stage}_source_snapshots']=all(digest(run/'driver_source'/stage/Path(path).name)==sha
                                               for path,sha in meta['code_sha256'].items())
        objects=json.loads((run/f'{stage}_objects.json').read_text())
        if stage=='dev':
            expected={(o['id'],31000+o['group'],arm,F) for o in objects
                      for arm in ('direct','probe_blind') for F in FORCES}
            actual={(r['object_id'],r['seed'],r['arm'],r['requested_force_N']) for r in rows}
        else:
            expected={(o['id'],41000+2*o['group']+rep,arm) for o in objects for rep in (0,1) for arm in ARMS}
            actual={(r['object_id'],r['seed'],r['arm']) for r in rows}
        checks[f'{stage}_complete_unique_matrix']=expected==actual and len(rows)==len(expected)
        checks[f'{stage}_matched_objects']=True
        for group in sorted({o['group'] for o in objects}):
            kappas=[o['kappa'] for o in objects if o['group']==group]
            checks[f'{stage}_matched_objects'] &= len(kappas)==3 and all(
                all(k[name]==kappas[0][name] for name in ('size','mass','surface_friction','com_offset')) for k in kappas)
        checks[f'{stage}_equal_thresholds']=all(r['threshold_N']=={'hard':50.,'bendable':25.,'compressible':15.}[r['category']]
                                               for r in rows)
    dev_ids={r['object_id'] for r in all_rows['dev']}
    test_ids={r['object_id'] for r in all_rows['test']}
    checks['development_test_ids_disjoint']=not(dev_ids & test_ids)
    checks['development_test_seeds_disjoint']=not({r['seed'] for r in all_rows['dev']} & {r['seed'] for r in all_rows['test']})
    test_meta=json.loads((run/'test_metadata.json').read_text())
    checks['runtime_code_unchanged_since_test_start']=all(digest(ROOT/path)==sha for path,sha in test_meta['code_sha256'].items()
                                                         if path.startswith('policy/'))
    rows=all_rows['test']
    keyed={(r['object_id'],r['seed'],r['arm']):r for r in rows}
    checks['all_48_probe_prefixes_identical']=all(
        all(r[name]==keyed[r['object_id'],r['seed'],'probe_blind'][name]
            for name in ('probe_digest','probe_raw','feedback','decision_time_s'))
        for r in rows if r['arm']=='probe_feedback')
    robot={name:float(max(r['robot'][name] for r in rows)) for name in
           ('ik_max_position_mm','ik_max_rotation_deg','tracking_max_mm','nonfinger_max_penetration_mm','self_max_penetration_mm')}
    robot['episode_tracking_p95_median_mm']=float(np.median([r['robot']['tracking_p95_mm'] for r in rows]))
    robot['actuator_peak_Nm']=np.max([r['robot']['actuator_peak_Nm'] for r in rows],axis=0).tolist()
    robot['control_clip_steps']=sum(r['robot']['control_clip_steps'] for r in rows)
    checks['arm_force_limits_respected']=all(np.all(np.asarray(r['robot']['actuator_peak_Nm']) <= [28.000001]*3+[10.000001]*3) for r in rows)
    checks['no_nonfinger_or_self_penetration']=robot['nonfinger_max_penetration_mm']==0. and robot['self_max_penetration_mm']==0.
    extra={}
    for base in ('direct','probe_blind'):
        pairs=[(r,keyed[r['object_id'],r['seed'],base]) for r in rows if r['arm']=='probe_feedback']
        both=[(r,b) for r,b in pairs if r['outcome']==b['outcome']=='success']
        extra[f'feedback_minus_{base}']=dict(
            both_success_pairs=len(both),
            mean_sim_time_difference_s=float(np.mean([r['sim_s']-b['sim_s'] for r,b in both])) if both else None,
            mean_peak_force_difference_N=float(np.mean([r['max_grip_force_N']-b['max_grip_force_N'] for r,b in both])) if both else None,
            force_choice_changes=sum(r['selected_force_N']!=b['selected_force_N'] for r,b in pairs))
    fields=['object_id','group','category','seed','arm','outcome','selected_force_N','max_grip_force_N','sim_s','wall_s','reason']
    with (run/'test_trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['probe_compression_mm','probe_relaxed_force_N','tracking_p95_mm'],lineterminator='\n')
        writer.writeheader()
        for r in rows:
            writer.writerow({**{name:r[name] for name in fields},
                             'probe_compression_mm':None if r['feedback'] is None else r['feedback'][0],
                             'probe_relaxed_force_N':None if r['feedback'] is None else r['feedback'][1],
                             'tracking_p95_mm':r['robot']['tracking_p95_mm']})
    result=dict(checks=checks,all_passed=all(checks.values()),robot=robot,paired_success_diagnostics=extra,
                failure_reasons=dict(Counter(r['reason'] for r in rows if r['outcome']!='success')))
    save(run/'audit.json',result)
    print(json.dumps(result,indent=2))
    if not result['all_passed']:
        raise RuntimeError('experiment audit failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=DEFAULT_RUN)
    audit(ap.parse_args().run)
